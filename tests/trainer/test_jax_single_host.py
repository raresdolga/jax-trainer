"""Integration end to end test for the trainer"""

from typing import Callable, List, Dict
import os
from dataclasses import dataclass
from flax import linen as nn
import jax
from jax import numpy as jnp
from torch.utils.data import Dataset

from tax.evals.base import Evaluator
from tax.train.jax_single_host import Trainer


class DummyModule(nn.Module):
    @nn.compact
    def __call__(self, x, labels=None, train=False):
        W = self.param("W", jax.nn.initializers.lecun_normal(), (10, 1))
        tmp = W.T @ x
        return {"loss": jnp.mean(tmp), "logits": tmp}


class DummyEval(Evaluator):
    """Evaluator with dummy output"""

    def __init__(self) -> None:
        self._config = None

    def compute_metrics(self, *args, **kwargs):
        """Dummy metrics compute

        Returns:
            _type_: _description_
        """
        return -1

    def evaluate(self, trainer_eval_fn: Callable, prefix: str = "eval_", **kwargs):
        """Dummy evaluate

        Args:
            trainer_eval_fn (_type_): _description_
            prefix (str, optional): _description_. Defaults to "eval_".

        Returns:
            Dict[str, Any]: Output metrics
        """
        return {prefix + "loss": 1.0}


class DummyDataset(Dataset):
    """Toy dataset

    Args:
        Dataset (_type_): _description_
    """

    def __init__(self, key) -> None:
        self.x = jax.random.normal(key, (100, 10, 1))

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, index):
        return {"input_ids": self.x[index], "labels": jnp.array([1])}


@dataclass
class Config:
    """Example config"""

    batchnorm: bool = False
    epochs: int = 5
    eval_steps: int = 4
    batch_size: int = 2
    max_checkpoints: int = 2
    shuffle_train: bool = False
    train_steps: int = None
    grad_accumulation_steps: int = 1
    warmup: int = 2
    lr_decay_fn: str = "cosine"
    lr: float = 1e-3
    weight_decay: float = 0.04
    max_seq_len: int = 100


def data_collator(batch: List[Dict[str, jax.Array]]):
    """Concatenate examles in a batch

    Args:
        batch List[Dict[str, jax.Array]]: _description_

    Returns:
        _type_: _description_
    """
    input_ids, labels = [], []
    for element in batch:
        input_ids.append(element["input_ids"])
        labels.append(element["labels"])
    return {"input_ids": jnp.array(input_ids), "labels": jnp.array(labels)}


def test_trainer():
    """
    Entry point  for testing
    """
    key = jax.random.PRNGKey(seed=0)
    init_key, train_key, data_key, key = jax.random.split(key, 4)
    train_data = DummyDataset(data_key)
    model = DummyModule()
    config = Config()
    evaluator = DummyEval()
    out_dir = os.path.dirname(os.path.abspath(__file__))
    trainer = Trainer(
        config=config,
        out_dir=out_dir,
        model=model,
        train_data=train_data,
        data_collator=data_collator,
        evaluator=evaluator,
        rng=init_key,
        model_inputs_orded=("input_ids", "labels"),
    )
    trainer.train(train_key)


if __name__ == "__main__":
    test_trainer()
