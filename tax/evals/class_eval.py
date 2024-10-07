from typing import Dict, Callable
import jax
from jax import numpy as jnp
from torch.utils.data import Dataset
from tax.config import ModelConfig
from .base import Evaluator


def acc_class(output: Dict[str, jax.Array], labels: jax.Array) -> Dict[str, jax.Array]:
    """Calculate accuracy

    Args:
        output (Dict[str, jax.Array]): output of the model.
            Must include logits and loss
        labels (jax.Array[BT]): target

    Returns:
        Dict[str, jax.Array]: Output Metrics
    """
    logits = output["logits"]
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    metrics = {
        "loss": output["loss"],
        "accuracy": accuracy,
    }
    return metrics


class ClassificEvaluator(Evaluator):
    """Evaluation classificator"""

    def __init__(
        self, val_data: Dataset, data_collator: Callable, config: ModelConfig
    ) -> None:
        super().__init__(val_data, data_collator, config)
        self._batchnorm = config.batchnorm

    def compute_metrics(
        self, output: Dict[str, jax.Array], labels: jax.Array
    ) -> Dict[str, jax.Array]:
        """_summary_

        Args:
            output (_type_): _description_
            labels (_type_): _description_

        Returns:
            _type_: _description_
        """
        return acc_class(output, labels)
