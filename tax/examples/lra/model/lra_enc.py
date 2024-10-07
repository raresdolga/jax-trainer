"""
Encoder only layer used in lra
"""

import functools
import math
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from .attention import BidirectionalAttention
from tax.config import ModelConfig


class PositionalEncoding(nn.Module):
    d_model: int
    max_len: int = 2048
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        """
        Inputs
            d_model - Hidden dimensionality of the input.
            max_len - Maximum length of a sequence to expect.
        """
        max_len = self.max_len
        d_model = self.d_model
        # Create matrix of [SeqLen, HiddenDim] representing the positional encoding for max_len inputs
        pe = jnp.zeros((max_len, d_model))
        position = jnp.arange(0, max_len, dtype=jnp.float32)[..., None]
        div_term = jnp.exp(
            jnp.arange(0, d_model, 2, dtype=jnp.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
        pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
        self.pe = pe[None, ...].astype(self.dtype)

    def __call__(
        self,
        embeds: jnp.array,
        do_inference: bool = False,
        time_pos: int = None,
    ) -> jnp.array:
        """
        Args:
            embeds: jnp.array(BTD) - embedding vectors
            do_inference: bool - special treatment for inference
            time_pos: int - position in the sentence for sequential inference
        """
        if do_inference:
            return self.pe[:, time_pos] + embeds
        return self.pe[:, : embeds.shape[1]] + embeds


class GatedMLP(nn.Module):
    hidden_dim: int
    intermediate_dim: int
    initializer_range: float = 0.02
    final_w_init_variance_scale: float = 1.0
    dtype: jnp.dtype = jnp.float32

    @property
    def out_kernel_init(self) -> nn.initializers.Initializer:
        """Initialization of the kernel for the last layer of the block."""
        return nn.initializers.variance_scaling(
            scale=self.final_w_init_variance_scale,
            mode="fan_in",
            distribution="normal",
        )

    @nn.compact
    def __call__(self, x):
        intermediate_size = self.intermediate_dim
        gate_proj = nn.Dense(
            features=intermediate_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.initializer_range),
            dtype=self.dtype,
        )
        up_proj = nn.Dense(
            features=intermediate_size,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.initializer_range),
            dtype=self.dtype,
        )
        down_proj = nn.Dense(
            features=self.hidden_dim,
            use_bias=False,
            kernel_init=jax.nn.initializers.normal(stddev=self.initializer_range),
            dtype=self.dtype,
        )
        act_fn = jax.nn.silu
        out = down_proj(act_fn(gate_proj(x)) * up_proj(x))
        return out


class SotaTransBlock(nn.Module):
    config: ModelConfig
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        x,
        train: bool = False,
        attention_mask=None,
    ):
        pre_norm = nn.RMSNorm(dtype=self.dtype)
        mix_layer = BidirectionalAttention(self.config, self.dtype)
        mlp_pre_norm = nn.RMSNorm(dtype=self.dtype)
        mlp = GatedMLP(
            hidden_dim=self.config.hidden_dim,
            intermediate_dim=4 * self.config.hidden_dim,
            initializer_range=self.config.initializer_range,
            dtype=self.dtype,
        )

        residual = x
        x = pre_norm(x)
        x = mix_layer(x, attention_mask=attention_mask, train=train)

        # jax.debug.print("Mix block nan: {x}", x=jnp.isnan(x).any())
        residual = x + residual
        x = mlp_pre_norm(residual)
        x = mlp(x)
        return x + residual


class ConvEmbed(nn.Module):
    """
    Pass input through a convolution network
    """

    dropout: float
    hidden_dim: int

    @nn.compact
    def __call__(self, X: jnp.array, train: bool = False) -> jnp.array:
        """
        Args:
            X: (batch_size, (W*H), 1)
            train: bool. Used for dropout
        """
        conv_dims = (1, 24, 48, 96, self.hidden_dim)  # 192
        conv_layers = [
            nn.Conv(
                features=conv_dims[i + 1], kernel_size=(3, 3), strides=1, padding="SAME"
            )
            for i in range(0, len(conv_dims) - 1)
        ]
        norm = nn.LayerNorm()
        drop = nn.Dropout(self.dropout, deterministic=not train)

        batch_sz, seq_len, inp_ch = X.shape
        W = int(math.sqrt(seq_len))
        H = W
        X = X.reshape(batch_sz, W, H, 1)
        for l in conv_layers:
            X = l(X)
        X = X.reshape(batch_sz, seq_len, -1)
        X = norm(drop(X))
        return X


class TextImageEncoder(nn.Module):
    """
    Deals with images and text.
    """

    config: ModelConfig
    vocab_size: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, X: jnp.array, train: bool = False, attention_mask=None
    ) -> jnp.array:
        """
        Args:
            X: jnp.array(BTD), B = Batch size, T = sequence length, D = embed dimension
            train: bool - used for dropout
        Returns:
            out: jnp.array(BTD) - transformed output sequence
        """
        embed = nn.Dense(features=self.config.hidden_dim)
        if self.vocab_size is not None:
            X = jax.nn.one_hot(X, self.vocab_size)
        pos_embeds = None
        if self.config.embed_type == "absolute":  # absolute
            pos_embeds = PositionalEncoding(
                d_model=self.config.hidden_dim,
                max_len=self.config.pos_embed_max_len,
                dtype=self.dtype,
            )
        elif self.config.embed_type == "nope":
            pos_embeds = None
        else:
            raise ValueError(f"Embed type {self.config.embed_type} not accepted")
        ln = nn.RMSNorm()
        block = SotaTransBlock
        enc_layers = [
            block(config=self.config, dtype=self.dtype)
            for _ in range(self.config.nlayers)
        ]
        if not pos_embeds is None:
            X = pos_embeds(embed(X))
        else:  # relative or nope
            X = embed(X)

        if not self.config.prenorm:
            X = ln(X)
        for l in enc_layers:
            X = l(X, train, attention_mask=attention_mask)
        return X
