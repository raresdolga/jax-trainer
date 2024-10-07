from flax import linen as nn
from jax import numpy as jnp


def cross_entropy_loss(logits, target):
    """Calculate cross entripy for classification

    Args:
        logits (jax.Array[BD]): Predicted logits
        target (jax.Array[B]): Labels

    Returns:
        jax.Array[(1,)]: mean loss
    """
    target = nn.one_hot(target, num_classes=logits.shape[-1])
    loss = jnp.einsum("BH,BH->B", target, nn.log_softmax(logits, axis=-1))
    loss = jnp.mean(loss, axis=-1)
    return -loss


def cross_entropy_loss_lm(logits, target, ignore_index=-100):
    """
    Args:
        logits (jax.Array[BTD]): Predicted logits
        target (jax.Array[BT]): Labels
        ignore_index (int): must be a negative value
     Returns:
        jax.Array[(1,)]: mean loss
    """
    num_valid = (target != ignore_index).sum(axis=-1)
    # Indices outside the range [0, num_classes) will be encoded as zeros:
    target = nn.one_hot(target, num_classes=logits.shape[-1])
    loss = jnp.einsum("BLH,BLH->BL", target, nn.log_softmax(logits, axis=-1))
    loss = jnp.sum(loss, axis=-1) / num_valid  # mean reduction on sequene level
    loss = jnp.mean(loss, axis=-1)
    return -loss
