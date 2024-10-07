from flax import linen as nn
from jax import numpy as jnp


def cross_entropy_loss(logits, target):
    """Calculate cross entripy for classification

    Args:
        logits (jax.Array[BTD]): Predicted logits
        target (jax.Array[BT]): Labels

    Returns:
        _type_: _description_
    """
    target = nn.one_hot(target, num_classes=logits.shape[-1])
    loss = jnp.einsum("BH,BH->B", target, nn.log_softmax(logits, axis=-1))
    loss = jnp.mean(loss, axis=-1)
    return -loss
