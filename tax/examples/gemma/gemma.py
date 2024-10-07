"""
The new architexure which is more coplex then the vanilla tranformer. Includes convolutions and gating before the mixing block.
"""

from functools import partial
import jax
from jax import numpy as jnp
from flax import linen as nn

from tax.config import LMConfig


class FlaxGemmaRMSNorm(nn.Module):
    config: LMConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        self.epsilon = 1e-6  # self.config.rms_norm_eps
        self.weight = self.param(
            "scale", lambda _, shape: jnp.ones(shape), self.config.hidden_dim
        )

    def __call__(self, hidden_states):
        variance = jnp.asarray(hidden_states, dtype=jnp.float32)
        variance = jnp.power(variance, 2)
        variance = variance.mean(-1, keepdims=True)
        # use `jax.numpy.sqrt` as `jax.lax.rsqrt` does not match `torch.rsqrt`
        hidden_states = hidden_states / jnp.sqrt(variance + self.epsilon)
        out = (1 + self.weight.astype(self.dtype)) * hidden_states.astype(self.dtype)
        return out


class Gemma2RotaryEmbedding(nn.Module):
    dim: int
    max_position_embeddings: int = 2048
    base: int = 10000

    def setup(self):

        self.inv_freq = 1.0 / (
            self.base
            ** (
                jnp.arange(0, self.dim, 2, dtype=jnp.int32).astype(jnp.float32)
                / self.dim
            )
        )

    def __call__(self, x, position_ids, seq_len=None):
        # print(x.shape)
        if position_ids is None:
            position_ids = jnp.arange(0, x.shape[2])[None, ...]

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].astype(jnp.float32)
        inv_freq_expanded = jnp.broadcast_to(
            inv_freq_expanded, (position_ids.shape[0], inv_freq_expanded.shape[1], 1)
        )
        position_ids_expanded = position_ids[:, None, :].astype(jnp.float32)
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        freqs = (
            inv_freq_expanded.astype(jnp.float32)
            @ position_ids_expanded.astype(jnp.float32)
        ).transpose(0, 2, 1)
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        cos = jnp.cos(emb)
        sin = jnp.sin(emb)
        return cos.astype(dtype=x.dtype), sin.astype(dtype=x.dtype)


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return jnp.concatenate((-x2, x1), axis=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = jnp.expand_dims(cos, unsqueeze_dim)
    sin = jnp.expand_dims(sin, unsqueeze_dim)
    # print(cos.shape, q.shape)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def apply_rotation(rel_pos, k, q):
    """
    Implement rotation where rel_pos is already A^t.
    Uses fast implementation of the parse matrix
    Args:
        rel_pos: Union[jnp.array(B, H, T, D], Tuple] -> TBHD
            Half sin & second half cos
        k,q: jnp.array(B,H,T,D) # B, nh, T, hs
            input matrix
        neg: bool
            Denotes weather we need to calculate R^{-s}
    """
    q = rel_pos(q)
    k = rel_pos(k)
    return k, q


def repeat_kv(hidden_states: jax.Array, n_rep: int) -> jax.Array:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = jnp.broadcast_to(
        hidden_states[:, :, None, :, :],
        shape=(batch, num_key_value_heads, n_rep, slen, head_dim),
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class GemmaCausalAttOld(nn.Module):
    config: LMConfig
    dtype: jnp.dtype = jnp.float32
    attn_logit_softcapping: float = 50.0

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        print("Shape of input X: ", X.shape)
        nheads = self.config.nheads
        head_dim = self.config.head_dim

        rotary_emb = Gemma2RotaryEmbedding(
            dim=self.config.head_dim,
            max_position_embeddings=self.config.pos_embed_max_len,
            base=10000.0,
        )
        # key, query, value projections for all heads, but in a batch
        q_proj = nn.Dense(
            nheads * head_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="q_proj",
        )

        k_proj = nn.Dense(
            nheads * head_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="k_proj",
        )
        v_proj = nn.Dense(
            nheads * head_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="v_proj",
        )
        # output projection
        c_proj = nn.Dense(
            self.config.hidden_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="out_proj",
        )

        # regularization
        attn_dropout = nn.Dropout(rate=self.config.dropout_att, deterministic=not train)
        B, T, C = (
            X.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = jnp.tril(jnp.ones(shape=(T, T))).reshape(1, 1, T, T)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = jnp.split(c_attn(X), 3, axis=2)
        q, k, v = q_proj(X), k_proj(X), v_proj(X)

        k = k.reshape(B, T, nheads, -1).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        q = q.reshape(B, T, nheads, -1).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, nheads, -1).transpose(0, 2, 1, 3)  # (B, nh, T, hs)

        print("Shape of input Q,K,V: ", q.shape, k.shape, v.shape)
        # k, q = apply_rotation(rot_embeds, k, q)  # BHTD
        cos, sin = rotary_emb(v, position_ids=None)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        # k = repeat_kv(k, num_key_value_groups)
        # v = repeat_kv(v, num_key_value_groups)
        print("V2 Shape of input Q,K,V: ", q.shape, k.shape, v.shape)

        att = (q @ jnp.swapaxes(k, -2, -1)) * (
            256**-0.5
        )  # * (1.0 / math.sqrt(k.shape[-1]))

        print("Att shape: ", att.shape)
        if self.attn_logit_softcapping is not None:
            att = att / self.attn_logit_softcapping
            att = jax.nn.tanh(att)
            att = att * self.attn_logit_softcapping

        # att = att.masked_fill(bias[:,:,:T,:T] == 0, float('-inf'))
        att = jnp.where(bias[:, :, :T, :T] == 0, float("-inf"), att)
        att = jax.nn.softmax(att, axis=-1)

        att = attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(0, 2, 1, 3).reshape(
            B, T, -1
        )  # re-assemble all head outputs side by side

        # output projection
        y = c_proj(y)  # resid_dropout(c_proj(y))
        return y


class GemmaCausalAtt(nn.Module):
    config: LMConfig
    dtype: jnp.dtype = jnp.float32
    attn_logit_softcapping: float = 50.0

    @nn.compact
    def __call__(self, X: jnp.array, train: bool, **kwargs) -> jnp.array:
        nheads = self.config.nheads
        if self.config.num_key_value_heads:
            num_key_value_heads = self.config.num_key_value_heads
        else:
            num_key_value_heads = self.config.nheads
        if self.config.head_dim:
            head_dim = self.config.head_dim
        else:
            head_dim = self.config.hidden_dim // self.config.nheads

        num_key_value_groups = self.config.nheads // num_key_value_heads

        rotary_emb = Gemma2RotaryEmbedding(
            dim=head_dim,
            max_position_embeddings=self.config.pos_embed_max_len,
            base=10000.0,
        )
        # key, query, value projections for all heads, but in a batch
        q_proj = nn.Dense(
            nheads * head_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="q_proj",
        )

        k_proj = nn.Dense(
            num_key_value_heads * head_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="k_proj",
        )
        v_proj = nn.Dense(
            num_key_value_heads * head_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="v_proj",
        )
        # output projection
        c_proj = nn.Dense(
            self.config.hidden_dim,
            use_bias=self.config.attention_bias,
            dtype=self.dtype,
            kernel_init=jax.nn.initializers.normal(
                stddev=self.config.initializer_range
            ),
            name="out_proj",
        )

        # regularization
        attn_dropout = nn.Dropout(rate=self.config.dropout_att, deterministic=not train)
        B, T, C = (
            X.shape
        )  # batch size, sequence length, embedding dimensionality (n_embd)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        bias = jnp.tril(jnp.ones(shape=(T, T))).reshape(1, 1, T, T)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # q, k, v = jnp.split(c_attn(X), 3, axis=2)
        q, k, v = q_proj(X), k_proj(X), v_proj(X)

        k = k.reshape(B, T, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)
        q = q.reshape(B, T, nheads, -1).transpose(0, 2, 1, 3)  # (B, nh, T, hs)
        v = v.reshape(B, T, num_key_value_heads, -1).transpose(
            0, 2, 1, 3
        )  # (B, nh, T, hs)

        # k, q = apply_rotation(rot_embeds, k, q)  # BHTD
        cos, sin = rotary_emb(v, position_ids=None)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        k = repeat_kv(k, num_key_value_groups)
        v = repeat_kv(v, num_key_value_groups)

        att = (q @ jnp.swapaxes(k, -2, -1)) * (
            256**-0.5
        )  # * (1.0 / math.sqrt(k.shape[-1]))

        if self.attn_logit_softcapping is not None:
            att = att / self.attn_logit_softcapping
            att = jax.nn.tanh(att)
            att = att * self.attn_logit_softcapping

        att = jnp.where(bias[:, :, :T, :T] == 0, float("-inf"), att)
        att = jax.nn.softmax(att, axis=-1)

        att = attn_dropout(att)

        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        y = y.transpose(0, 2, 1, 3).reshape(
            B, T, -1
        )  # re-assemble all head outputs side by side

        # output projection
        y = c_proj(y)  # resid_dropout(c_proj(y))
        return y


class GemmaMLP(nn.Module):
    config: LMConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        embed_dim = self.config.hidden_dim
        inner_dim = self.config.intermediate_dim
        kernel_init = jax.nn.initializers.normal(self.config.initializer_range)
        self.activation_fn = partial(
            jax.nn.gelu, approximate=True
        )  # ACT2FN[self.config.hidden_act]
        self.gate_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.down_proj = nn.Dense(
            embed_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )
        self.up_proj = nn.Dense(
            inner_dim, use_bias=False, dtype=self.dtype, kernel_init=kernel_init
        )

    def __call__(self, hidden_states):
        up_proj_states = self.up_proj(hidden_states)
        gate_states = jax.nn.gelu(
            self.gate_proj(hidden_states), approximate=True
        )  # self.activation_fn(self.gate_proj(hidden_states))

        hidden_states = self.down_proj(up_proj_states * gate_states)
        return hidden_states


class GemmaDecoderLayer(nn.Module):
    config: LMConfig
    dtype: jnp.dtype = jnp.float32

    def setup(self):

        self.self_attn = GemmaCausalAtt(
            config=self.config, dtype=self.dtype
        )  # get_decoder_mixer(self.config, self.dtype)
        self.input_layernorm = FlaxGemmaRMSNorm(
            config=self.config, dtype=self.dtype, name="input_layernorm"
        )
        self.post_attention_layernorm = FlaxGemmaRMSNorm(
            config=self.config, dtype=self.dtype, name="post_attention_layernorm"
        )
        self.pre_feedforward_layernorm = FlaxGemmaRMSNorm(
            config=self.config, dtype=self.dtype, name="pre_feedforward_layernorm"
        )
        self.post_feedforward_layernorm = FlaxGemmaRMSNorm(
            config=self.config, dtype=self.dtype, name="post_feedforward_layernorm"
        )
        self.mlp = GemmaMLP(self.config, dtype=self.dtype, name="mlp")

    def __call__(
        self,
        hidden_states,
        attention_mask=None,
        position_ids=None,
        train: bool = False,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            train=train,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        # residual connection
        hidden_states = residual + hidden_states
        return hidden_states


class GemmaDecoder(nn.Module):
    config: LMConfig
    sharded: bool
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, X, attention_mask, train=False):
        final_norm = FlaxGemmaRMSNorm(config=self.config, dtype=self.dtype, name="norm")
        block_fn = partial(
            GemmaDecoderLayer,
            config=self.config,
            dtype=self.dtype,
            name="residual_block",
        )
        # faster to compile - but a few disadvantages like merging bloks and lossing some default optimisations
        block = block_fn()  # (name="residual_block")
        if self.sharded:
            X, _ = nn.scan(
                lambda module, carry, _: (
                    module(carry, attention_mask, train=train),
                    None,
                ),
                variable_axes={"params": 0, "intermediates": 0},
                split_rngs={"params": True, "dropout": True},
                length=self.config.nlayers,
            )(block, X, ())
        else:
            X, _ = nn.scan(
                lambda module, carry, _: (
                    module(carry, attention_mask, train=train),
                    None,
                ),
                variable_axes={"params": 0, "intermediates": 0},
                split_rngs={"params": True, "dropout": True},
                length=self.config.nlayers,
                metadata_params={
                    "partition_name": None
                },  # We do not need to partition over the layer axis.
            )(block, X, ())
        # for i in range(self.config.nlayers):
        #     X = block_fn(name=f"residual_block_{i}")(
        #         X, attention_mask=attention_mask, train=train
        #     )
        # get logits
        X = final_norm(X)
        return X
