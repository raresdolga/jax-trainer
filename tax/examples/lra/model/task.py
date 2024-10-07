from typing import Dict, Any
from flax import linen as nn
from jax import numpy as jnp
import jax

from tax.config import LRAConfig
from tax.evals.losses import cross_entropy_loss
from .lra_enc import TextImageEncoder


class Classification(nn.Module):
    """Encoder with a classification head"""

    config: LRAConfig
    vocab_size: int
    pad_id: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.encoder = TextImageEncoder(
            vocab_size=self.vocab_size, config=self.config, dtype=self.dtype
        )

        self.head = nn.Dense(features=self.config.num_classes, dtype=jnp.float32)

    def __call__(
        self,
        input_ids: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
        **kwargs,
    ) -> Dict[str, jnp.array]:
        """
        Args:
            input_ids: jnp.array(BL) - input ids
            labels: jnp.array(B)
            train: bool - used for dropout
        Returns:
            out: Dict[str, jnp.array] - loss and logits
        """
        batch_size, seq_len = input_ids.shape[0], input_ids.shape[1]
        # mean pooling for non-pad tokens
        if self.vocab_size is None:  # no padding for images
            attention_mask = jnp.ones_like(
                input_ids[..., 0]
            )  # Depth is not important for checks of padding
            sequence_lengths = jnp.ones((batch_size,), dtype=jnp.int32) * seq_len
        else:
            attention_mask = (input_ids != self.pad_id).astype(jnp.int32)
            sequence_lengths = (
                jnp.asarray(jax.lax.eq(input_ids, self.pad_id), dtype=jnp.int32).argmax(
                    -1
                )
                - 1
            )
        X = self.encoder(input_ids, train=train, attention_mask=attention_mask)
        if self.config.pool == "mean":
            X = jnp.einsum("BSH,BS->BSH", X, attention_mask)
            pooled_x = X.sum(axis=1) / attention_mask.sum(axis=-1)[..., None]
        elif self.config.pool == "last":
            # last non-pad token
            pooled_x = X[jnp.arange(batch_size), sequence_lengths]
        else:
            raise IOError("pooling mode node recognized")
        logits = self.head(pooled_x)
        if not labels is None:
            loss = cross_entropy_loss(logits=logits, target=labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}


class Retreival(nn.Module):
    config: LRAConfig
    vocab_size: int
    pad_id: int
    dtype: jnp.dtype = jnp.float32

    def setup(self) -> None:
        self.encoder = TextImageEncoder(
            vocab_size=self.vocab_size,
            config=self.config,
            dtype=self.dtype,
        )
        self.dense_1 = nn.Dense(features=self.config.hidden_dim, name="mlp")
        self.head = nn.Dense(features=self.config.num_classes, name="logits")

    def __call__(
        self,
        input_ids: jnp.array,
        labels: jnp.array = None,
        train: bool = False,
        **kwargs,
    ) -> Dict[str, jnp.array]:
        """
        Args:
            input_ids: jnp.array(B2L) - input ids
            labels: jnp.array(B)
            train: bool - used for dropout
        Returns:
            out: Dict[str, jnp.array] - loss and logits
        """
        batch_size, _, seq_len = input_ids.shape
        input_ids = input_ids.reshape(2 * batch_size, seq_len)
        # mean pooling for non-pad tokens
        if self.vocab_size is None:  # no padding for images
            attention_mask = jnp.ones_like(
                input_ids[..., 0]
            )  # Depth is not important for checks of padding
            sequence_lengths = seq_len
        else:
            attention_mask = (input_ids != self.pad_id).astype(jnp.int32)
            sequence_lengths = (
                jnp.asarray(jax.lax.eq(input_ids, self.pad_id), dtype=jnp.int32).argmax(
                    -1
                )
                - 1
            )
        X = self.encoder(input_ids, train=train, attention_mask=attention_mask)
        if self.config.pool == "mean":
            X = jnp.einsum("BSH,BS->BSH", X, attention_mask)
            pooled_x = X.sum(axis=1) / attention_mask.sum(axis=-1)[..., None]
        elif self.config.pool == "last":
            # last non-pad token
            pooled_x = X[jnp.arange(2 * batch_size), sequence_lengths, :]
        elif self.config.pool == "CLS":
            pooled_x = X[jnp.arange(2 * batch_size), 0, :]
        else:
            raise IOError("pooling mode node recognized")

        pooled_x = pooled_x.reshape(batch_size, 2, -1)
        out0, out1 = pooled_x[:, 0, :], pooled_x[:, 1, :]
        encoded = jnp.concatenate([out0, out1, out0 - out1, out0 * out1], axis=-1)

        out = nn.gelu(self.dense_1(encoded))
        logits = self.head(out)
        if not labels is None:
            loss = cross_entropy_loss(logits=logits, target=labels)
            return {"loss": loss, "logits": logits}
        return {"logits": logits}
