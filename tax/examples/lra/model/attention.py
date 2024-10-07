import jaxtyping as jt
import math
import jax
from flax import linen as nn
from jax import numpy as jnp
from tax.config import Config


def attention(attn_dropout, q, k, v, mask):
    # manual implementation of attention
    att = (q @ jnp.swapaxes(k, -2, -1)) * (1.0 / math.sqrt(k.shape[-1]))

    # att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
    att = jnp.where(mask == 0, -9e15, att)
    att = jax.nn.softmax(att, axis=-1)

    att = attn_dropout(att)
    y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
    return y


class BidirectionalAttention(nn.Module):
    config: Config
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, src: jt.Array, train: bool, attention_mask: jt.Array, **kwargs
    ) -> jt.Array:
        # key, query, value projections for all heads, but in a batch
        Wq = nn.Dense(
            self.config.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )
        Wk = nn.Dense(
            self.config.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )
        Wv = nn.Dense(
            self.config.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
        )

        # output projection
        c_proj = nn.Dense(
            self.config.hidden_dim,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
                / math.sqrt(2 * self.config.nlayers)
            ),
        )

        # regularization
        attn_dropout = nn.Dropout(rate=self.config.dropout_att, deterministic=not train)
        resid_dropout = nn.Dropout(rate=self.config.dropout, deterministic=not train)

        B, S, C = (
            src.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # expand source masK BS -> BHTS
        attention_mask = attention_mask[:, None, None, :]
        attention_mask = jnp.broadcast_to(attention_mask, shape=(B, 1, S, S))

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v = Wq(src), Wk(src), Wv(src)

        q = q.reshape(B, S, self.config.nheads, C // self.config.nheads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        k = k.reshape(B, S, self.config.nheads, C // self.config.nheads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        v = v.reshape(B, S, self.config.nheads, C // self.config.nheads).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)

        y = attention(attn_dropout, q, k, v, mask=attention_mask)
        y = y.transpose(0, 2, 1, 3).reshape(
            B, S, C
        )  # re-assemble all head outputs side by side

        # output projection
        y = resid_dropout(c_proj(y))
        return y
