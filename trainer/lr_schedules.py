from typing import Callable
from jax import numpy as jnp
import chex


def vaswani_lr_schedule(
    lr_mul: chex.Scalar, d_model: chex.Scalar, warmup_steps: chex.Scalar
) -> Callable:
    """Adaptive scheduler used by Attention is all you need
    https://arxiv.org/pdf/1706.03762.pdf
    """

    def schedule(count):
        count += 1
        lr_scale = (d_model**-0.5) * jnp.minimum(
            count ** (-0.5), count * warmup_steps ** (-1.5)
        )
        return lr_scale * lr_mul

    return schedule
