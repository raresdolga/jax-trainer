"""
Jax implementation of lava. With bidirectional encoder and bidirectional decoder
"""

from typing import Dict
import jaxtyping as jt
from flax import linen as nn
from jax import numpy as jnp
import jax
from tax.config import LMConfig
from flax.linen.dtypes import promote_dtype
from .gemma import GemmaDecoder
from tax.evals.losses import cross_entropy_loss_lm


class Embedder(nn.Module):
    """Embedder module.

    Attributes:
      vocab_size: The size of the token vocabulary.
      embed_dim: The dimensionality of each token embedding.
      scale_by_sqrt_dim: Whether to scale the output of the block by
        `sqrt(elf.embed_dim)`.
      dtype: dtype used for computation.
      param_dtype: dtype used for initializing parameters.
    """

    vocab_size: int
    embed_dim: int
    scale_by_sqrt_dim: bool
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        # Parameters.
        self.input_embedding_table = self.param(
            "embedding",
            nn.initializers.variance_scaling(
                scale=1.0,
                mode="fan_in",
                distribution="normal",
                in_axis=1,
                out_axis=0,
            ),
            (self.vocab_size, self.embed_dim),
        )

    def encode(self, x):
        """Encodes an input sequence of tokens."""
        x = self.input_embedding_table[(x,)]
        [x] = promote_dtype(x, dtype=self.dtype)

        if self.scale_by_sqrt_dim:
            # Cast to bfloat16 to match training.
            x = x * jnp.sqrt(self.embed_dim).astype(jnp.bfloat16)
        return x

    def decode(self, x):
        """Decodes an input sequence of activations."""
        x, embedding_table = promote_dtype(
            x,
            self.input_embedding_table,
            dtype=self.dtype,
        )
        return x @ embedding_table.T


class Gemma(nn.Module):
    config: LMConfig
    pad_id: int
    dtype: jnp.dtype = jnp.float32
    sharded: bool = False
    final_logit_softcapping: float = 30.0

    def lm_head(self, x, embeds):
        # tied embeddings
        return x @ embeds.embedding.T

    @nn.compact
    def __call__(
        self,
        input_ids: jt.Array,
        labels: jt.Array = None,
        train: bool = False,
    ) -> Dict[str, jnp.array]:
        """
        Args:
            input_ids (jnp.array[BL]): inputs
            labels (jnp.array[BL]): targets
            train (bool): used for dropout
        Returns:
            out (Dict[str, jnp.array]) - loss and logits
        """

        text_embed = Embedder(
            vocab_size=self.config.vocab_size,
            embed_dim=self.config.hidden_dim,
            scale_by_sqrt_dim=False,
            dtype=self.dtype,
            name="model_embed",
        )

        decoder = GemmaDecoder(
            self.config,
            sharded=self.sharded,
            dtype=self.dtype,
            name="model",
        )

        input_embeds = text_embed.encode(input_ids)
        normalizer = jnp.array(self.config.hidden_dim**0.5, dtype=input_embeds.dtype)
        input_embeds = input_embeds * normalizer
        # for causal language modeeling - no mask is required
        logits = decoder(input_embeds, attention_mask=None, train=train)
        logits = text_embed.decode(logits)  # self.lm_head(logits, text_embed)
        if self.final_logit_softcapping is not None:
            logits = logits / self.final_logit_softcapping
            logits = jax.nn.tanh(logits)
            logits = logits * self.final_logit_softcapping

        labels = jnp.where(labels == self.pad_id, -100, labels)
        if labels is None:
            return {"logits": logits, "loss": None}

        loss = cross_entropy_loss_lm(
            logits=logits[:, :-1, :], target=labels[:, 1:], ignore_index=-100
        )
        return {"loss": loss, "logits": logits}
