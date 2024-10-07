"""
Implement trainer on a single host (works with multile devices).
Implementation relies on automatic parallelization during compilation using jit.
Advatages:
 - Simple code: like for a single device setting
 - No need for per deivice explicit-collectives programming (like all_gather).
 - works on multiple devices if they are on the same host
Disadvantages:
 - Does not work in multihost environment
 - Less control over parallelization => Can be less optimal for custom operations
"""

from typing import Any, Dict, Tuple, Callable, Iterable, Union
import jaxtyping as jt
import json
from functools import partial
import os
from tqdm.auto import tqdm
import numpy as np
import jax
from jax import numpy as jnp
from jax.tree_util import tree_flatten
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from flax.training import train_state
from flax import linen as nn
import wandb
from orbax import checkpoint as ocp
import optax
from optax.schedules import Schedule
from torch.utils.data import DataLoader
import wandb.apis
import wandb.sdk

from tax.lr_schedules import vaswani_lr_schedule
from tax.evals.base import Evaluator
from tax.config import Config
from .utils import Optimizer, TrainState, BatchNormTrainState

WandbRun = Union[wandb.apis.public.Run, None]


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
def eval_step(
    batchnorm: bool,
    state: train_state.TrainState,
    batch: Tuple[jt.Array],
) -> Dict[str, jt.Array]:
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
        output = state.apply_fn(
            {"params": params},
            *batch,
            train=False,
        )
    return output


def train_step(
    batchnorm: bool,
    model_rng: jax.random.PRNGKey,
    state: train_state.TrainState,
    batch: Tuple[jt.Array],
) -> Tuple[train_state.TrainState, jt.Array]:
    """One train step

    Args:
        batchnorm (bool): whether to use batch normalisation or not
        model_rng (jax.random.PRNGKey): random key
        state (train_state.TrainState): Flax Trainer state
        batch (jt.Array): Input data, in the order expected by the model.

    Returns:
        Tuple[train_state.TrainState, jt.Array]: new state and the loss
    """
    dropout_train_key = jax.random.fold_in(key=model_rng, data=state.step)

    def loss_fn(params):
        if batchnorm:
            output, updates = state.apply_fn(
                {"params": params, "batch_stats": state.batch_stats},
                *batch,
                train=True,
                rngs={"dropout": dropout_train_key},
                mutable=["batch_stats"],
            )
        else:
            output = state.apply_fn(
                {"params": params},
                *batch,
                train=True,
                rngs={"dropout": dropout_train_key},
            )
            updates = None
        loss = output["loss"]
        return loss, (output, updates)

    gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (logits, updates)), grads = gradient_fn(state.params)
    if batchnorm:
        state = state.apply_gradients(grads=grads, batch_stats=updates["batch_stats"])
    else:
        state = state.apply_gradients(grads=grads)
    return state, loss


def get_scheduler(config: Config, total_steps: int) -> Schedule:
    """Create a scheduler from config

    Args:
        config (Config): Project config
        total_steps (int): number of training steps

    Returns:
        Callable: S
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
    elif config.lr_decay_fn == "vaswani":
        lr_scheduler = vaswani_lr_schedule(
            config.lr, config.hidden_dim, warmup_steps=warmup_steps
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
) -> Tuple[optax.GradientTransformation, Schedule]:
    """Create optimizer and lr_scheduler
    Args:
        config (Config): Configuration for the project
        total_steps (Config): Number of training steps

    Returns:
        Tuple[optax.GradientTransformation, Schedule]: Optimizer
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
) -> train_state.TrainState:
    """Initialiases the model and creates the training state

    Args:
        batchnorm (bool): Whether the flax model uses batch norm or not
        model (nn.Module): flax module
        optimizer (optax.GradientTransformation): Optimizer
        init_rng (jax.random.PRNGKey): initialisation key
        batch (Tuple[jt.Array]): Input data, in the order expected by the model.

    Returns:
        train_state.TrainState: Flax trainig state
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


def best_loss(structured: Any) -> float:
    """Minimum loss

    Args:
        structured (Any): Pytree of losses for each checkpoint

    Returns:
        float: _description_
    """
    flat, tree = tree_flatten(structured)
    flat = [float(x) for x in flat]
    return min(flat)


class Trainer:
    """Simple trainer on a single Host"""

    def __init__(
        self,
        config: Config,
        out_dir: str,
        model: nn.Module,
        train_data: Iterable = None,
        train_dl: Iterable = None,
        data_collator: Callable = None,
        evaluator: Evaluator = None,
        test_evaluator: Evaluator = None,
        wandb_run: WandbRun = None,
        rng: jax.random.PRNGKey = None,
        model_inputs_orded: Tuple[str] = ("input_ids", "labels"),
        prepare_opt_fn: Optimizer = prepare_optimizer,
    ) -> None:
        init_rng, rng = jax.random.split(rng, 2)
        self.config = config
        self._out_dir = out_dir
        self._model = model
        self.train_data = train_data
        self.data_collator = data_collator
        self.eval_steps = self.config.eval_steps
        self.max_checkpoints = self.config.max_checkpoints
        self._evaluator = evaluator
        self._test_evaluator = test_evaluator
        self.model_inputs_orded = model_inputs_orded

        self._eval_metrics = []

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
        if train_dl is None:
            self.train_dl = DataLoader(
                self.train_data,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle_train,
                collate_fn=self.data_collator,
                drop_last=True,
            )
        else:
            self.train_dl = train_dl

        self.wandb_run = wandb_run

        if self.eval_steps == 0:
            self.eval_steps = int(
                np.ceil(len(self.train_data) / self.config.batch_size)
            )

        self.total_steps = self.calc_total_steps()
        self._optimizer, self._lr_scheduler = prepare_opt_fn(
            self.config, self.total_steps
        )

        init_data = self.sample_data()
        jax.debug.print("Batch has the form: {x}", x=init_data)
        self._mesh, self._state_sharding, self._data_sharding = self.get_shardings(
            init_rng, init_data
        )
        # distribute state on multiple devices using jit
        self._jit_init_fn = jax.jit(
            init_train_state,
            static_argnums=(0, 1, 2),  # 2,3,4
            in_shardings=(
                NamedSharding(self._mesh, None),
                self._data_sharding,
            ),  # PRNG key and x
            out_shardings=self._state_sharding,
        )
        # jit for efficiency and parallelization
        self._train_step_fn = jax.jit(
            train_step,
            static_argnums=(0,),
            in_shardings=(
                NamedSharding(self._mesh, None),
                self._state_sharding,
                self._data_sharding,
            ),
            out_shardings=(self._state_sharding, NamedSharding(self._mesh, None)),
        )
        self._eval_step_fn = jax.jit(
            eval_step,
            static_argnums=(0,),
            in_shardings=(
                self._state_sharding,
                self._data_sharding,
            ),
            out_shardings=NamedSharding(self._mesh, None),
        )

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

    def sample_data(self) -> Tuple[jt.Array]:
        """Return Data sample

        Returns:
            Tuple[jt.Array]: Data in the order expected by the model
        """
        data = next(iter(self.train_dl))
        # TODO: Investigate why dictionary does not work for jit
        data = tuple([data[k] for k in self.model_inputs_orded])
        return data

    @staticmethod
    def place_on_device(
        batch: Iterable[jt.Array], data_sharding: NamedSharding = None
    ) -> Iterable[jt.Array]:
        """Move data to the device dicated by the sharding

        Args:
            batch (Iterable[jt.Array]): Input data
            data_sharding (NamedSharding, optional): Description of how data should be placed on the device. Defaults to None.

        Returns:
            Iterable[jt.Array]: Input data
        """
        if data_sharding is None:
            return batch
        for k, v in batch.items():
            batch[k] = jax.device_put(v, data_sharding)
        return batch

    def safe_wandb_log(self, log_data: Dict[str, Any]) -> None:
        """Log data wiith the wandb_run if config allows for it.

        Args:
            log_data (Dict[str, Any]): Data to log
        """
        if self.wandb_run is not None:
            generations = log_data.pop("generations", None)
            self.wandb_run.log(log_data)
            if generations is not None:
                colums = ["Prompt", "Expected", "Generation"]
                gen_table = wandb.Table(columns=colums, data=generations)
                self.wandb_run.log({"gen_table": gen_table})

    def get_shardings(
        self, init_rng: jax.random.PRNGKey, data: Tuple[jt.Array]
    ) -> Tuple[Mesh, NamedSharding, NamedSharding]:
        """Create descriptions on how the model and data should be split across devices.
        For this trainer the mdoel is replicated and data is split across batch dimension

        Args:
            init_rng (jax.random.PRNGKey): initialisation key
            data (Tuple[jt.Array]): Data as expected by the model.

        Returns:
            Tuple[Mesh, NamedSharding, NamedSharding]: The devices on the host and description on how to place arrays on them.
        """
        init_rng, abstract_state_rng = jax.random.split(init_rng)
        # total global devices
        num_devices = len(jax.devices())
        devices = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(devices, axis_names=("B",))

        # get empty shapes for partitioning
        abstract_variables = jax.eval_shape(
            partial(
                init_train_state,
                self.config.batchnorm,
                self._model,
                self._optimizer,
            ),
            abstract_state_rng,
            data,
        )
        # model place on device
        # Empty PartitionSpec() or with None for each axis means replicate
        state_sharding = nn.get_sharding(abstract_variables, mesh)
        # split data across Batch dimension
        data_sharding = NamedSharding(mesh, PartitionSpec("B"))
        return mesh, state_sharding, data_sharding

    def calc_total_steps(self) -> int:
        """Convert epochs into number of training steps.

        Returns:
            int: Total number of train steps.
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

    @staticmethod
    def load_trainer_state(
        empty_state: train_state.TrainState,
        check_dir: str = None,
        step_number: int = None,
    ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        """Load state from the checkpoint

        Args:
            empty_state (train_state.TrainState): a dummy input shape with shape and device placement info.
            check_dir (str, optional): Directory with checkpoints. Defaults to None.
            step_number (int, optional): The specific step to load. Defaults to None.

        Returns:
            Tuple[train_state.TrainState, Dict[str, Any]]: Loaded train state and metadata, like iteration number when training stopped.
        """
        options = ocp.CheckpointManagerOptions(create=False)
        mngr = ocp.CheckpointManager(
            check_dir,
            options=options,
            item_handlers={
                "state": ocp.StandardCheckpointHandler(),
                "metadata": ocp.JsonCheckpointHandler(),
            },
        )

        if step_number is None:
            available_checkpoints = mngr.all_steps(read=True)
            print(
                "Loading last of the availabel checkpoints: ",
                str(available_checkpoints),
            )
            step_number = mngr.latest_step()

        restored = mngr.restore(
            step_number,
            args=ocp.args.Composite(
                state=ocp.args.StandardRestore(empty_state),
                metadata=ocp.args.JsonRestore(),
            ),
        )
        return restored.state, restored.metadata

    @staticmethod
    def create_zero_state(
        rng: jax.random.PRNGKey,
        data: Tuple[jt.Array],
        config: Config,
        model: nn.Module,
    ) -> train_state.TrainState:
        """Create a dummy state with zeros. Helper for loading pre-traiend checkpoints.
        For a proper initialization with device placement, use the code in __init__
        """
        init_rng, rng = jax.random.split(rng)
        # total steps not important if state not initialised from trainer config
        total_steps = 150000  # self.calc_total_steps()
        optimizer = prepare_optimizer(config=config, total_steps=total_steps)

        state_shapes = jax.eval_shape(
            partial(
                init_train_state,
                config.batchnorm,
                model,
                optimizer,
            ),
            init_rng,
            data,
        )

        empty_state = jax.tree_util.tree_map(jnp.zeros_like, state_shapes)
        return empty_state

    def train(
        self, train_rng: jax.random.PRNGKey, checkpoint_path: str = None
    ) -> Tuple[train_state.TrainState, Dict[str, Any]]:
        init_rng, train_rng = jax.random.split(train_rng)
        init_data = self.sample_data()

        start_it = 0
        state = self._jit_init_fn(
            self.config.batchnorm, self._model, self._optimizer, init_rng, init_data
        )

        if checkpoint_path is not None:
            jax.debug.print("Loading state")
            # TODO: There are some problems with loading frim zero shape and training in the optimizer update_fn.
            # Should work fine for inference
            # state = self.create_zero_state(
            #     init_rng, init_data, self.config, self._model
            # )
            # serialised saved state has sharding information
            state, metadata = self.load_trainer_state(
                empty_state=state, check_dir=checkpoint_path, step_number=None
            )
            start_it = metadata["step"]

        jax.debug.print("Trainer total steps: {x}", x=self.total_steps)
        shape_m = jax.tree.map(
            lambda x: x.shape,
            nn.meta.unbox(state.params),
        )
        print("Shape model is: ", json.dumps(shape_m))
        print(self._model)
        state = self._train(
            train_rng, total_steps=self.total_steps, start_it=start_it, state=state
        )

        # list of dicts to dicts of list
        metrics = self._eval_metrics
        if len(self._eval_metrics) > 0:
            metrics = {
                key: [i[key] for i in self._eval_metrics]
                for key in self._eval_metrics[0]
            }
        return metrics, state

    def trainer_eval(
        self,
        state: train_state.TrainState,
        batch: dict[str, jt.Array],
    ) -> Tuple[jt.Array, Dict[str, jt.Array]]:
        """
        Places data on correct device and calls the model on the batch
        """
        inputs = tuple([batch[k] for k in self.model_inputs_orded])
        inputs = jax.lax.stop_gradient(inputs)
        output = self.eval_step_fn(
            self.config.batchnorm,
            state,
            inputs,
        )
        labels = jax.device_get(batch["labels"])
        output = jax.device_get(output)
        return labels, output

    def _train(
        self,
        train_rng: jax.random.PRNGKey,
        total_steps: int,
        state: train_state.TrainState,
        start_it: int = 0,
    ) -> train_state.TrainState:

        param_count = sum(x.size for x in jax.tree_util.tree_leaves(state.params))
        jax.debug.print("Number of parameters: {x} M", x=param_count / 1000000)

        it = start_it
        progress_bar = tqdm(range(total_steps), position=it, leave=True)
        all_scores = []
        losses = []
        while True:
            train_loss = []
            for batch in self.train_dl:
                train_rng, model_rng = jax.random.split(train_rng)
                batch = {k: batch[k] for k in self.model_inputs_orded}
                batch = self.place_on_device(batch, self._data_sharding)
                inputs = tuple([batch[k] for k in self.model_inputs_orded])
                state, loss = self.train_step_fn(
                    self.config.batchnorm,
                    model_rng,
                    state,
                    inputs,
                )
                train_loss.append(loss)

                if (it > 0 and it % self.eval_steps == 0) or (it >= total_steps):
                    scores = {}
                    # compute scores on test
                    if self._evaluator is not None:
                        train_rng, eval_rng = jax.random.split(train_rng)
                        eval_fn = partial(self.trainer_eval, state)
                        scores = self._evaluator.evaluate(
                            trainer_eval_fn=eval_fn, prefix="eval_", state=state
                        )
                    # compute scores on test
                    if self._test_evaluator is not None:
                        train_rng, eval_rng = jax.random.split(train_rng)
                        eval_fn = partial(self.trainer_eval, state)
                        test_scores = self._test_evaluator.evaluate(
                            trainer_eval_fn=eval_fn, prefix="test_", state=state
                        )
                        scores.update(test_scores)

                    tr_loss = jax.device_get(train_loss)
                    scores["train_loss"] = np.mean(tr_loss)
                    scores["learning_rate"] = float(self._lr_scheduler(it))

                    # scores["#Toks"] = (
                    #     self.config.max_seq_len * self.config.batch_size * it
                    # )
                    scores["Epoch"] = it // len(self.train_dl)

                    jax.debug.print("Train Loss {x}", x=scores["train_loss"])
                    jax.debug.print("Evaluation scores: {x}", x=scores)
                    self.safe_wandb_log(scores)
                    all_scores.append(scores)

                    metadata = {"step": it}
                    metric_save = "eval_loss" if "eval_loss" in scores else "train_loss"
                    self.checkpoint_manager.save(
                        it,
                        metrics={"eval_loss": str(scores[metric_save])},
                        args=ocp.args.Composite(
                            state=ocp.args.StandardSave(state),
                            metadata=ocp.args.JsonSave(metadata),
                        ),
                    )
                    self.checkpoint_manager.wait_until_finished()

                it += 1
                progress_bar.update(1)
                if it >= total_steps:
                    break
            train_loss = jax.device_get(train_loss)
            losses.append(np.mean(train_loss))
            if it >= total_steps:
                break
        return state
