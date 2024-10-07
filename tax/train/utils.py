"utils for trainer"
from typing import Dict, Tuple, Callable, Any
import jaxtyping as jt
from flax import linen as nn
import optax
from flax.training import train_state

from tax.config import Config

Parameter = jt.Array | nn.Partitioned
Metrics = Dict[str, Tuple[jt.Array, ...]]
Optimizer = Callable[[Config, int], Tuple[optax.GradientTransformation, optax.Schedule]]


class TrainState(train_state.TrainState):
    """Train state which supports dropout key"""

    key: jt.Array


class BatchNormTrainState(train_state.TrainState):
    """Train state which supports dropout key and stores batch statistics for batchnorm"""

    key: jt.Array
    batch_stats: Any
