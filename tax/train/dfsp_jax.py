from typing import Any, Dict, Tuple, Callable, List
from functools import partial
import os
import json
import jaxtyping as jt
from tqdm.auto import tqdm
import numpy as np
import jax
from jax import lax
from jax.tree_util import tree_flatten
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec
from jax.experimental.shard_map import shard_map
from flax import linen as nn
from flax.training import train_state
import wandb
import orbax.checkpoint as ocp
import optax
from torch.utils.data import DataLoader, Dataset

from tax.config import Config
from tax.evals.base import Evaluator
from .utils import Parameter, Optimizer, TrainState, BatchNormTrainState


@jax.named_scope("shard_params")
def shard_params(
    params: Parameter, axis_name: str, min_weight_size: int = 2**18
) -> Parameter:
    """Shard parameters across the given mesh axis.

    Args:
        params: The parameters to shard.
        axis_name: The axis to shard parameters across.
        min_weight_size: The minimum size of a parameter to shard.
            Parameters with fewer values will not be sharded.

    Returns:
        PyTree of same structure as params, but with leaves sharded over new axis if possible.
    """
    axis_idx = jax.lax.axis_index(axis_name)
    axis_size = jax.lax.psum(1, axis_name)

    def _split(x: Parameter) -> Parameter:
        if isinstance(x, nn.Partitioned):
            value, names = x.value, x.names
        else:
            value = x
            names = (None,) * value.ndim
        if axis_name in names:
            print(
                f"Parameter {value.shape} with names {names}"
                "already sharded on axis {axis_name}."
            )
            return x
        elif value.size <= min_weight_size:
            print(
                f"Parameter {value.shape} with names {names} too small to shard,"
                "size {value.size} < {min_weight_size}."
            )
            return x
        else:
            shape = value.shape
            idx = np.argsort(shape)[::-1]  # Shard along largest possible axis.
            for i in idx:
                if shape[i] % axis_size == 0 and names[i] is None:
                    split_size = shape[i] // axis_size
                    p_sharded = nn.Partitioned(
                        value=lax.dynamic_slice_in_dim(  # Shard to keep on present device.
                            value, axis_idx * split_size, split_size, axis=i
                        ),
                        names=names[:i] + (axis_name,) + names[i + 1 :],
                    )
                    return p_sharded
            print(
                f"Could not shard {value.shape} with names {names} on axis {axis_name}, "
                "no suitable axis found."
            )
            return x

    return jax.tree_util.tree_map(
        _split,
        params,
        is_leaf=lambda x: isinstance(
            x, nn.Partitioned
        ),  # Consider a nn.Partitioned object as a leaf.
    )


def gather_array_with_mean_grads(x: jt.Array, axis: int, axis_name: str):
    """Gathering with averaging gradients across replicas."""
    axis_size = jax.lax.psum(1, axis_name)

    # Define a custom gradient for the gather operation.
    @jax.custom_gradient
    def f(x):
        def grad_fn(g):
            # pmean_scatter
            return (
                jax.lax.psum_scatter(g, axis_name, scatter_dimension=axis, tiled=True)
                / axis_size
            )

        return jax.lax.all_gather(x, axis_name, axis=axis, tiled=True), grad_fn

    return f(x)


@jax.named_scope("gather_params")
def gather_params(params: Parameter, axis_name: str) -> Parameter:
    """Gather parameters from all replicas across the given axis.

    Args:
        params: The parameters to gather.
        axis_name: The axis to gather parameters across.

    Returns:
        PyTree of same structure as params, but with leaves gathered
            if they were a nn.Partitioned object.
    """

    def _gather(p: Parameter) -> Parameter:
        if isinstance(p, nn.Partitioned) and axis_name in p.names:
            param_shard = p.names
            shard_axis = param_shard.index(axis_name)
            value = gather_array_with_mean_grads(
                p.value, axis=shard_axis, axis_name=axis_name
            )
            # If there are any other axes that are sharded, we need to keep the partitioned structure.
            # Otherwise, we can return the value directly.
            param_shard = (
                param_shard[:shard_axis] + (None,) + param_shard[shard_axis + 1 :]
            )
            if any([name is not None for name in param_shard]):
                return nn.Partitioned(value, param_shard)
            else:
                return value
        else:
            return p

    return jax.tree_util.tree_map(
        _gather, params, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def shard_module_params(
    target: nn.Module | Callable, axis_name: str, min_weight_size: int = 2**18
) -> nn.Module | Callable:
    """Shard parameters of a module across replicas.

    Args:
        target: The module to shard.
        axis_name: The axis name to shard parameters across.
        min_weight_size: The minimum size of a parameter to shard.
            Parameters with fewer values will not be sharded.

    Returns:
        The module with sharded parameters.
    """
    return nn.map_variables(
        target,
        trans_in_fn=partial(gather_params, axis_name=axis_name),
        trans_out_fn=partial(
            shard_params, axis_name=axis_name, min_weight_size=min_weight_size
        ),
        mapped_collections="params",
        mutable=True,
    )


def sync_gradients(
    grads: Parameter,
    axis_names: List[str],
) -> Parameter:
    """Synchronize gradients across devices.

    Gradients for parameters that are replicated over a given axis are averaged across devices.
    Parameters that are partitioned over a given axis are considered to already have a mean of
    the gradients on each device, and hence do not need to be altered.

    Args:
        grads: The gradients to synchronize.
        axis_names: The axis names to synchronize gradients across.

    Returns:
        The gradients averaged over the specified axes if they are replicated.
    """

    def sync_grad(g: Parameter) -> Parameter:
        if isinstance(g, nn.Partitioned):
            # Tree leaves for flattening potentially nested axis (multiple names can exist for single array axis).
            replication_axis_names = [
                name
                for name in axis_names
                if name not in jax.tree_util.tree_leaves(g.names)
            ]
            if len(replication_axis_names) == 0:
                # Parameters partitioned over all axes.
                return g
            else:
                # Average over remaining replicated axes.
                return g.replace(
                    value=jax.lax.pmean(g.value, axis_name=replication_axis_names)
                )
        else:
            # Parameters are replicated over all axes.
            return jax.lax.pmean(g, axis_name=axis_names)

    return jax.tree_map(
        sync_grad, grads, is_leaf=lambda x: isinstance(x, nn.Partitioned)
    )


def fold_rng_over_axis(rng: jax.random.PRNGKey, axis_name: str) -> jax.random.PRNGKey:
    """Folds the random number generator over the given axis.

    This is useful for generating a different random number for each device
    across a certain axis (e.g. the model axis).

    Args:
        rng: The random number generator.
        axis_name: The axis name to fold the random number generator over.

    Returns:
        A new random number generator, different for each device index along the axis.
    """
    axis_index = jax.lax.axis_index(axis_name)
    return jax.random.fold_in(rng, axis_index)


# Eval function
def eval_step_dp(
    batchnorm: bool, state: TrainState, batch: jt.Array
) -> Tuple[float, Dict[str, jt.Array]]:
    """
    Evaluation step to jit
    Args:
        batchnorm: Wheather the network uses batch normalisation or not
        state: current training state
        batch: input tuple to feed to the model
    Returns:
        out: Tuple of loss and model output
    """
    params = state.params
    batch = jax.lax.stop_gradient(batch)
    if batchnorm:
        output, _ = state.apply_fn(
            {"params": params, "batch_stats": state.batch_stats},
            *batch,
            train=False,
            mutable=["batch_stats"],
        )
    else:
        output = state.apply_fn({"params": params}, *batch, train=False)

    # loss is reduced to 1 and must be replicated -> P(), while logits still one per batch -> P("B")
    loss = output.pop("loss")
    # Sum metrics across replicas. Alternatively, we could keep the metrics separate
    # and only synchronize them before logging. For simplicity, we sum them here.
    with jax.named_scope("sync_loss"):
        loss = jax.lax.pmean(loss, axis_name="B")
    return loss, output


def train_step_dp(
    batchnorm: bool, model_rng: jt.Array, state: TrainState, batch: Tuple[jt.Array]
) -> Tuple[TrainState, float]:
    """
    Execute on train step. Function to be jited
    Args:
        batchnorm: Wheather the network uses batch normalisation or not
        state: current training state
        batch: input tuple to feed to the model
    Returns:
        Updated trainer state and loss
    """
    axis_index = jax.lax.axis_index("B")
    model_rng = jax.random.fold_in(model_rng, axis_index)
    model_rng, dropout_key = jax.random.split(model_rng)

    def loss_fn(params: Parameter):
        if batchnorm:
            output, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                *batch,
                train=True,
                rngs={"dropout": dropout_key},
                mutable=["batch_stats"],
            )
        else:
            output = state.apply_fn(
                {"params": params},
                *batch,
                train=True,
                rngs={"dropout": dropout_key},
            )
            updates = None
        loss = output["loss"]
        return loss, (output, updates)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (_, updates)), grads = gradient_fn(state.params)

    # Update parameters. We need to sync the gradients across devices before updating.
    with jax.named_scope("sync_gradients"):
        # grads = jax.tree_map(lambda g: jax.lax.pmean(g, axis_name="B"), grads)
        grads = sync_gradients(grads, ("B",))

    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)

    # mean loss across replicas. Alternatively, we could keep the metrics separate
    # and only synchronize them before logging. For simplicity, we sum them here.
    with jax.named_scope("sync_loss"):
        loss = jax.lax.pmean(loss, axis_name="B")

    return state, loss


def get_scheduler(config: Config, total_steps: int) -> optax.Schedule:
    """
    Default function for creating a learning rate scheduler.
    Args:
        config: experiment configuration
        total_steps: total iterations
    Returns:
        lr_scheduler: optax learning rate scheduler
    """
    total_steps = total_steps // config.grad_accumulation_steps
    warmup_steps = 0
    if config.warmup > 0:
        warmup_steps = config.warmup
    else:
        # it is 0 for no warmup
        warmup_steps = int(config.warmup_pc * total_steps)

    if config.lr_decay_fn == "cosine":
        jax.debug.print("total = {x}, warmup={y}", x=total_steps, y=warmup_steps)
        lr_scheduler = optax.cosine_decay_schedule(
            config.lr, decay_steps=(total_steps - warmup_steps), alpha=0.0
        )
    elif config.lr_decay_fn == "linear":
        lr_scheduler = optax.linear_schedule(
            init_value=config.lr,
            end_value=config.lr_end_value,
            transition_steps=(total_steps - warmup_steps),
            # transition_begin=int((total_steps - warmup_steps) * 0.25),
        )
    else:
        lr_scheduler = optax.constant_schedule(config.lr)
    # whether to add warmup or not
    if warmup_steps > 0:
        warmup_fn = optax.linear_schedule(
            init_value=0.0, end_value=config.lr, transition_steps=warmup_steps
        )
        lr_scheduler = optax.join_schedules(
            schedules=[warmup_fn, lr_scheduler], boundaries=[warmup_steps]
        )
    return lr_scheduler


def prepare_optimizer(
    config: Config, total_steps: int
) -> Tuple[optax.GradientTransformation, optax.Schedule]:
    """
    Default creation of optimizer and learnign rate scheduler.
    Args:
        config: configuration for the trainer
        total_steps: number of steps used in training
    Returns:
        optimizer and learning rate scheduler
    """
    lr_scheduler = get_scheduler(config=config, total_steps=total_steps)
    optimizer = optax.inject_hyperparams(optax.adamw)(
        learning_rate=lr_scheduler, weight_decay=config.weight_decay
    )

    if config.grad_accumulation_steps > 1:
        optimizer = optax.MultiSteps(optimizer, config.grad_accumulation_steps)
    # chain with norm
    optimizer = optax.chain(optax.clip_by_global_norm(1.0), optimizer)
    return optimizer, lr_scheduler


def init_train_state(
    batchnorm: bool,
    model: nn.Module,
    optimizer: optax.GradientTransformation,
    init_rng: jax.random.PRNGKey,
    batch: Tuple[jt.Array],
) -> TrainState:
    """
    Default initialisation of the trainer state
    Args:
        batchnorm: whether to use batch normalisation or not
        model: Model to train
        optimizer: optimizer used during training
        init_rng: random key used for initialisation
        batch: data used by the model
    Returns:
        state: Training state
    """
    init_rng, arg_rng, dropout_rng = jax.random.split(init_rng, 3)
    variables = model.init(arg_rng, *batch, train=False)
    # Create a State
    if batchnorm:
        state = BatchNormTrainState.create(
            apply_fn=model.apply,
            params=variables["params"],
            batch_stats=variables["batch_stats"],
            tx=optimizer,
            key=dropout_rng,
        )
    else:
        state = TrainState.create(
            apply_fn=model.apply,
            tx=optimizer,
            params=variables["params"],
            key=dropout_rng,
        )
    return state


def best_loss(structured: Parameter) -> float:
    """
    Calculate the best metric for checkpointing among proposed values.
    We use loss, so minimum is desired
    Args:
        structured: any pytree with the loss value
    Returns:
        minimum loss
    """
    flat, _ = tree_flatten(structured)
    flat = [float(x) for x in flat]
    return min(flat)


class DFSDPTrainer:
    """Fully sharded distributed training"""

    def __init__(
        self,
        config: Config,
        out_dir: str,
        model: nn.Module,
        train_data: Dataset,
        data_collator: Callable = None,
        evaluator: Evaluator = None,
        wandb_run: Callable = None,
        rng: jt.Array = None,
        model_inputs_orded: Tuple = ("input_ids", "labels"),
        model_outputs_orded: Tuple = ("loss", "logits"),
        init_state_fn: Callable = init_train_state,
        prepare_optimizer_fn: Optimizer = prepare_optimizer,
    ) -> None:
        """Data parallel fully sharded trainer

        Args:
            config (Config): Configuration with trainer fields
            out_dir (str): Directory where to save checkpoints and wandb data
            model (nn.Module): Model class to train
            train_data (Dataset): Tokenized train data
            data_collator (Callable, optional): Collator for batching. Defaults to None.
            evaluator (Evaluator, optional): Task specifc Evaluator. See evals.base
                for required interface. Defaults to None.
            wandb_run (Callable, optional): Wandb run to log experiments. Defaults to None.
            rng (jt.Array, optional): Random key. Defaults to None.
            model_inputs_orded (Tuple, optional): Order to select inputs when batch as a
                dict is transformed to a tuple. Should match the order of the model input.
                    Defaults to ("input_ids", "labels").
            model_outputs_orded (Tuple, optional): Keys in the dictionary outputed by the model.
                Defaults to ("loss", "logits").
            init_state_fn (Callable, optional): Function to initialise the model.
                Defaults to init_train_state.
            prepare_optimizer_fn (Optimizer, optional): Function which creates the optimizer.
                Defaults to prepare_optimizer.
        """
        init_rng, rng = jax.random.split(rng, 2)
        self.config = config
        self._out_dir = out_dir
        self.model = model
        self.train_data = train_data
        self.data_collator = data_collator
        self.eval_steps = self.config.eval_steps
        self.max_checkpoints = self.config.max_checkpoints
        self._evaluator = evaluator
        self.model_inputs_orded = model_inputs_orded
        self.model_outputs_orded = model_outputs_orded

        self._eval_metrics = []

        if jax.process_index() == 0:
            os.makedirs(self._out_dir, exist_ok=True)

        options = ocp.CheckpointManagerOptions(
            max_to_keep=self.max_checkpoints,
            create=True,
            best_fn=best_loss,
            best_mode="min",
        )

        self.checkpoint_manager = ocp.CheckpointManager(
            os.path.join(self._out_dir, "checkpoints"),
            options=options,
            item_names=("state", "metadata"),
            item_handlers={
                "state": ocp.StandardCheckpointHandler(),
                "metadata": ocp.JsonCheckpointHandler(),
            },
        )
        self.train_dl = DataLoader(
            self.train_data,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle_train,
            collate_fn=self.data_collator,
            drop_last=True,  # needs batch size to be devidable by number of gpus
        )
        self.wandb_run = wandb_run

        self.total_steps = self.calc_total_steps()
        self._optimizer, self._lr_scheduler = prepare_optimizer_fn(
            self.config, self.total_steps
        )
        self.state_spec, mesh = self.prepare_on_device(init_rng)

        self._jit_init_fn = lambda bn, m, o, k, d: jax.jit(
            shard_map(
                partial(init_state_fn, bn, m, o),
                mesh=mesh,
                in_specs=(
                    PartitionSpec(),
                    PartitionSpec(
                        "B",
                    ),
                ),  # PRNG key and x
                out_specs=self.state_spec,
                check_rep=False,
            )
        )(k, d)

        self._train_step_fn = jax.jit(
            shard_map(
                partial(train_step_dp, self.config.batchnorm),
                mesh,
                in_specs=(
                    PartitionSpec(),
                    self.state_spec,
                    PartitionSpec(
                        "B",
                    ),
                ),
                out_specs=(self.state_spec, PartitionSpec()),
                check_rep=False,
            ),
            donate_argnums=(1, 2),
        )
        # batchnorm, rng, state, batch
        self._eval_step_fn = jax.jit(
            shard_map(
                partial(eval_step_dp, self.config.batchnorm),
                mesh,
                in_specs=(
                    self.state_spec,
                    PartitionSpec(
                        "B",
                    ),
                ),
                out_specs=(
                    PartitionSpec(),
                    PartitionSpec(
                        "B",
                    ),
                ),
                check_rep=False,
            ),
            donate_argnums=(1,),
        )
        self.mesh = mesh

    @property
    def train_step_fn(
        self,
    ) -> Callable[
        [bool, jax.random.PRNGKey, train_state.TrainState, Tuple[jt.Array]],
        Tuple[train_state.TrainState, jt.Array],
    ]:
        """Return Jitted train function"""
        return self._train_step_fn

    @property
    def eval_step_fn(
        self,
    ) -> Callable[[bool, train_state.TrainState, Tuple[jt.Array]], Dict[str, jt.Array]]:
        """Return Jitted eval function"""
        return self._eval_step_fn

    def safe_wandb_log(self, log_data: Dict[str, Any]):
        """
        Log to wandb if a run was provided to the trainer
        Args:
            log_data: Data to log
        """
        if self.wandb_run is not None:
            generations = log_data.pop("generations", None)
            self.wandb_run.log(log_data)
            if generations is not None:
                colums = ["Prompt", "Expected", "Generation"]
                gen_table = wandb.Table(columns=colums, data=generations)
                self.wandb_run.log({"gen_table": gen_table})

    def sample_data(self) -> Tuple[jt.Array]:
        """
        Get a sample of data from the dataset, acorrding to the
        columns that trtainer uses.
        Returns:
            A tupke of jax array which are passed in order to the model.
        """
        data = next(iter(self.train_dl))
        # TODO: Investigate why dictionary does not work for jit
        data = tuple([data[k] for k in self.model_inputs_orded])
        return data

    def calc_total_steps(self):
        """
        Calculate total steps for traiinng if epochs provided in config instead of number of train steps
        """
        epochs = self.config.epochs
        if self.config.train_steps is None:
            total_steps = epochs * (
                np.ceil(len(self.train_data) / self.config.batch_size)
            )
            total_steps = int(total_steps)
        else:
            total_steps = self.config.train_steps

        return total_steps

    def prepare_on_device(self, init_rng: jt.Array):
        """
        Initialise the model to get the state and shard the state across devices
        """
        init_rng, abstract_state_rng = jax.random.split(init_rng)
        num_devices = len(jax.devices())
        if jax.process_index() == 0:
            jax.debug.print("Number of devices to shard is: {}", num_devices)
        data = next(iter(self.train_dl))
        print(data["input_ids"].shape)
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, axis_names=("B",))

        # model place on device
        # Empty Partition spec or with None for each axis means replicate
        data = tuple([data[k] for k in self.model_inputs_orded])
        # flax sharding
        jit_init_fn = shard_map(
            partial(
                init_train_state, self.config.batchnorm, self.model, self._optimizer
            ),
            mesh=mesh,
            in_specs=(
                PartitionSpec(),
                PartitionSpec(
                    "B",
                ),
            ),  # PRNG key and x
            out_specs=PartitionSpec(None),
            check_rep=False,
        )
        state_fsdp_shapes = jax.eval_shape(jit_init_fn, abstract_state_rng, data)
        state_fsdp_specs = nn.get_partition_spec(state_fsdp_shapes)

        return state_fsdp_specs, mesh

    def trainer_eval(
        self,
        state: train_state.TrainState,
        val_rng: jax.random.PRNGKey,
        batch: Dict[str, jt.Array],
    ) -> Tuple[jt.Array, Dict[str, jt.Array]]:
        """Places data on correct device and calls the model on the batch

        Args:
            state (train_state.TrainState): model current trained state
            val_rng (jax.random.PRNGKey): random number for validation (not currently used)
            batch (Dict[str, jt.Array]): Input data

        Returns:
            Tuple[jt.Array, Dict[str, jt.Array]]: Targets and model output
        """
        # batch = self.place_on_device(batch, self.sharding_data)
        inputs = tuple([batch[k] for k in self.model_inputs_orded])
        inputs = jax.lax.stop_gradient(inputs)
        output = self.eval_step_fn(
            state,
            inputs,
        )
        loss, output = (
            output  # jax.experimental.multihost_utils.process_allgather(output)
        )
        output["loss"] = loss
        # throw away unwanted data to save memory for evaluator,
        # where by default data is gathered on one gpu
        output = {k: output[k] for k in self.model_outputs_orded}
        return batch["labels"], output

    def _train(
        self,
        train_rng: jax.random.PRNGKey,
        total_steps: int,
        start_it: int,
        state: train_state.TrainState,
    ):
        it = start_it
        if jax.process_index() == 0:
            param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
            jax.debug.print("Number of parameters: {x} M", x=param_count / 1000000)
            progress_bar = tqdm(range(total_steps), position=0, leave=True, initial=it)
        all_scores = []
        losses = []
        while True:
            train_loss = []
            for batch in self.train_dl:
                train_rng, model_rng = jax.random.split(train_rng)
                # batch = self.place_on_device(batch, self.sharding_data)
                inputs = tuple([batch[k] for k in self.model_inputs_orded])
                state, loss = self.train_step_fn(
                    model_rng,
                    state,
                    inputs,
                )
                train_loss.append(loss)

                if (it > 0 and it % self.eval_steps == 0) or (it >= total_steps):
                    eval_rng, train_rng = jax.random.split(train_rng)
                    eval_fn = partial(self.trainer_eval, state, eval_rng)
                    scores = self._evaluator.evaluate(
                        trainer_eval_fn=eval_fn, prefix="eval_", state=state
                    )

                    tr_loss = train_loss
                    scores["train_loss"] = np.mean(tr_loss)
                    scores["learning_rate"] = float(self._lr_scheduler(it))

                    scores["it"] = it
                    all_scores.append(scores)
                    if jax.process_index() == 0:
                        jax.debug.print("Train Loss {x}", x=scores["train_loss"])
                        jax.debug.print("Evaluation scores: {x}", x=scores)
                        self.safe_wandb_log(scores)

                    metadata = {"step": it}
                    self.checkpoint_manager.save(
                        it,
                        metrics={"eval_loss": str(scores["eval_loss"])},
                        args=ocp.args.Composite(
                            state=ocp.args.StandardSave(state),
                            metadata=ocp.args.JsonSave(metadata),
                        ),
                    )
                it += 1
                if jax.process_index() == 0:
                    progress_bar.update(1)
                if it >= total_steps:
                    break
            train_loss = jax.device_get(train_loss)
            losses.append(np.mean(train_loss))
            if it >= total_steps:
                break
        self.checkpoint_manager.wait_until_finished()
        return state

    @staticmethod
    def load_trainer_state(
        empty_state: train_state.TrainState,
        check_dir: str = None,
        step_number: int = None,
    ) -> train_state.TrainState:
        """Takes a dummy state and loads the pre-trained state from a checkpoint.
        Dummy state is necessary for the orbax checkpoint manager. -- Blame the library not me
        Args:
            empty_state: dummy state sharded aross devices
            check_dir: Directory where the checkpoint are. Make sure that the path contains a "checkpoint/" at the end
            step_number: the number of checkpojnt to load if more are provided.
        """
        options = ocp.CheckpointManagerOptions(create=False)
        mngr = ocp.CheckpointManager(
            check_dir,
            options=options,
            # item_handlers=ocp.StandardCheckpointHandler()
            item_handlers={
                "state": ocp.StandardCheckpointHandler(),
                "metadata": ocp.JsonCheckpointHandler(),
            },
        )

        if step_number is None:
            step_number = mngr.latest_step()

        restored = mngr.restore(
            step_number,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(empty_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        return restored.state, restored.metadata

    def get_distrib_state(self, train_rng):
        init_rng, train_rng = jax.random.split(train_rng)
        init_data = self.sample_data()

        state = self._jit_init_fn(
            self.config.batchnorm, self.model, self._optimizer, init_rng, init_data
        )
        return state

    def train(
        self, train_rng: jax.random.PRNGKey, state=None, checkpoint_path: str = None
    ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        init_rng, train_rng = jax.random.split(train_rng)

        start_it = 0
        if state is None:
            state = self.get_distrib_state(init_rng)
        shape_m = jax.tree.map(
            lambda x: x.shape,
            nn.meta.unbox(state.params),
        )
        print("Shape model is: ", json.dumps(shape_m))
        if checkpoint_path is not None:
            jax.debug.print("Loading state")
            # serialised saved state has sharding information
            state, metadata = self.load_trainer_state(
                empty_state=state, check_dir=checkpoint_path, step_number=None
            )
            start_it = metadata["step"]

        jax.debug.print("Trainer total steps: {x}", x=self.total_steps)
        try:
            state = self._train(
                train_rng, total_steps=self.total_steps, start_it=start_it, state=state
            )
        except:
            raise
        finally:
            self.checkpoint_manager.wait_until_finished()
        # list of dicts to dicts of list
        metrics = self._eval_metrics
        if len(self._eval_metrics) > 0:
            metrics = {
                key: [i[key] for i in self._eval_metrics]
                for key in self._eval_metrics[0]
            }
        return metrics, state
