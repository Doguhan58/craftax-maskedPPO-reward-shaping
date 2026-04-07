"""
Reference: https://github.com/Reytuag/transformerXL_PPO_JAX
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax.numpy import einsum


class Gating(nn.Module):
    d_input: int
    bg: float = 2.0

    @nn.compact
    def __call__(self, x, y):
        dtype = x.dtype

        x_rz = nn.Dense(2 * self.d_input, use_bias=False, dtype=dtype, name="x_rz")(x)
        y_rzh = nn.Dense(3 * self.d_input, use_bias=False, dtype=dtype, name="y_rzh")(y)
        x_r, x_z = jnp.split(x_rz, 2, axis=-1)
        y_r, y_z, y_h = jnp.split(y_rzh, 3, axis=-1)
        r = nn.sigmoid(y_r + x_r)
        z = nn.sigmoid(y_z + x_z - self.param('gating_bias', nn.initializers.constant(self.bg), (self.d_input,)))
        h = jnp.tanh(y_h + nn.Dense(self.d_input, use_bias=False, dtype=dtype, name="x_h")(r * x))
        g = (1 - z) * x + z * h

        return g


class PositionalEmbedding(nn.Module):
    dim_emb: int

    def setup(self):
        self.inv_freq = 1.0 / (10000 ** (jnp.arange(0.0, self.dim_emb, 2.0) / self.dim_emb))

    def __call__(self, pos_seq):
        sinusoid_inp = jnp.outer(pos_seq, self.inv_freq)
        pos_emb = jnp.concatenate([jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)], axis=-1)
        return pos_emb


def _rel_shift(x):
    zero_pad = jnp.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    x_padded = jnp.concatenate([zero_pad, x], axis=-1)
    x_padded = x_padded.reshape(x.shape[:-2] + (x.shape[-1] + 1, x.shape[-2]))
    x = x_padded[..., 1:, :].reshape(x.shape)
    return x


class RelMultiHeadAttention(nn.Module):
    num_heads: int
    qkv_features: int
    out_features: int

    @nn.compact
    def __call__(self, inputs_kv, inputs_q, pos_embed, mask=None):
        batch_size = inputs_q.shape[0]
        query_len = inputs_q.shape[1]
        key_len = inputs_kv.shape[1]
        head_dim = self.qkv_features // self.num_heads

        #project Q, K, V
        W_q = nn.Dense(self.qkv_features, dtype=jnp.bfloat16)(inputs_q)
        W_k = nn.Dense(self.qkv_features, dtype=jnp.bfloat16)(inputs_kv)
        W_v = nn.Dense(self.qkv_features, dtype=jnp.bfloat16)(inputs_kv)

        #positional embedding
        W_r = nn.Dense(self.qkv_features, use_bias=False, dtype=jnp.bfloat16)(pos_embed)

        #reshape to [Batch, Len, Heads, HeadDim]
        W_q = W_q.reshape(batch_size, query_len, self.num_heads, head_dim)
        W_k = W_k.reshape(batch_size, key_len, self.num_heads, head_dim)
        W_v = W_v.reshape(batch_size, key_len, self.num_heads, head_dim)
        W_r = W_r.reshape(key_len, self.num_heads, head_dim)

        scale = head_dim ** -0.5

        r_w_bias = self.param('r_w_bias', nn.initializers.zeros, (self.num_heads, head_dim))
        r_r_bias = self.param('r_r_bias', nn.initializers.zeros, (self.num_heads, head_dim))

        #term (c): content bias — r_w_bias · K^T → [B, H, 1, K]
        content_bias = einsum('hd,bkhd->bhk', r_w_bias, W_k)[:, :, None, :] * scale

        #terms (b)+(d): position attention with relative shift
        #(Q + r_r_bias) · R^T → [B, H, Q, K], then rel_shift
        position_bias = einsum('bqhd,khd->bhqk', W_q + r_r_bias, W_r) * scale
        position_bias = _rel_shift(position_bias)

        #combine all biases
        combined_bias = content_bias + position_bias
        if mask is not None:
            mask_bias = jnp.where(mask, 0.0, -1e9)
            combined_bias = combined_bias + mask_bias

        output = jax.nn.dot_product_attention(query=W_q, key=W_k, value=W_v, bias=combined_bias, scale=scale)

        #reshape and project output
        output = output.reshape(batch_size, query_len, self.qkv_features)
        output = nn.Dense(self.out_features, dtype=jnp.bfloat16)(output)

        return output


class TransformerLayer(nn.Module):
    num_heads: int
    out_features: int
    qkv_features: int
    mlp_dim: int = None
    gating: bool = True
    gating_bias: float = 2.0


    def setup(self):
        self.attention = RelMultiHeadAttention(num_heads=self.num_heads, qkv_features=self.qkv_features, out_features=self.out_features)
        self.ln1 = nn.LayerNorm(dtype=jnp.bfloat16)
        self.ln2 = nn.LayerNorm(dtype=jnp.bfloat16)

        intermediate_dim = self.mlp_dim if self.mlp_dim is not None else self.out_features
        self.dense1 = nn.Dense(intermediate_dim, dtype=jnp.bfloat16)
        self.dense2 = nn.Dense(self.out_features, dtype=jnp.bfloat16)

        if self.gating:
            self.gate1 = Gating(self.out_features, self.gating_bias)
            self.gate2 = Gating(self.out_features, self.gating_bias)


    def __call__(self, values_keys, queries, pos_embed, mask=None):
        values_keys_n = self.ln1(values_keys)
        queries_n = self.ln1(queries)

        attention = self.attention(inputs_kv=values_keys_n, inputs_q=queries_n, pos_embed=pos_embed, mask=mask)

        if self.gating:
            out_attention = self.gate1(queries, nn.relu(attention))
        else:
            out_attention = queries + attention

        out_attention_n = self.ln2(out_attention)
        out = self.dense1(out_attention_n)
        out = nn.gelu(out)
        out = self.dense2(out)

        if self.gating:
            out= self.gate2(out_attention, jax.nn.relu(out))
        else:
            out = out + out_attention

        return out


class GTrXL(nn.Module):
    embed_dim: int = 256
    num_heads: int = 4
    head_dim: int = 64
    num_layers: int = 3
    mlp_dim: int = 512
    dropout_rate: float = 0.0
    gru_bias: float = 2.0
    max_seq_len: int = 1024
    use_remat: bool = True


    def setup(self):
        #sinus pos emb
        self.pos_emb = PositionalEmbedding(self.embed_dim)

        #cache
        max_cache_len = self.max_seq_len + 512
        cached_pos_seq = jnp.arange(max_cache_len, 0, -1, dtype=jnp.float32)

        self._cached_pos_embed = self.pos_emb(cached_pos_seq)
        self._max_cache_len = max_cache_len

        layer_cls = TransformerLayer

        if self.use_remat:
            layer_cls = nn.remat(TransformerLayer)

        qkv_features = self.num_heads * self.head_dim

        self.tf_layers = [
            layer_cls(num_heads=self.num_heads, qkv_features=qkv_features, out_features=self.embed_dim, mlp_dim=self.mlp_dim,
                      gating=True, gating_bias=self.gru_bias) for _ in range(self.num_layers)]


    def _get_pos_embed(self, total_len: int):
        start_idx = self._max_cache_len - total_len

        return jax.lax.dynamic_slice(self._cached_pos_embed, (start_idx, 0), (total_len, self.embed_dim))


    def forward_eval(self, memories, obs_emb, mask=None):
        x = obs_emb.astype(jnp.bfloat16)

        query_len = 1
        mem_len = memories.shape[1] if memories is not None else 0
        total_len = mem_len + query_len

        pos_embed = self._get_pos_embed(total_len)

        out_memory = jnp.zeros((x.shape[0], self.num_layers, self.embed_dim), dtype=jnp.bfloat16)

        for i in range(self.num_layers):
            out_memory = out_memory.at[:, i].set(x.astype(out_memory.dtype))

            layer_mem = memories[:, :, i, :] if memories is not None else None

            if layer_mem is not None:
                memory = jnp.concatenate([layer_mem, x[:, None, :].astype(layer_mem.dtype)], axis=1)
            else:
                memory = x[:, None, :]

            x = self.tf_layers[i](values_keys=memory, queries=x[:, None, :], pos_embed=pos_embed, mask=mask)
            x = x.squeeze(1)

        return x, out_memory


    def forward_train(self, memories, obs_emb, mask=None):
        x = obs_emb.astype(jnp.bfloat16)

        query_len = x.shape[1]
        mem_len = memories.shape[1] if memories is not None else 0
        total_len = mem_len + query_len

        pos_embed = self._get_pos_embed(total_len)

        for i in range(self.num_layers):
            layer_mem = memories[:, :, i, :] if memories is not None else None

            if layer_mem is not None:
                memory = jnp.concatenate([layer_mem, x], axis=1)
            else:
                memory = x

            x = self.tf_layers[i](values_keys=memory, queries=x, pos_embed=pos_embed,  mask=mask)

        return x


    @staticmethod
    def init_memory(batch_size: int, mem_len: int, num_layers: int, embed_dim: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, mem_len, num_layers, embed_dim), dtype=jnp.bfloat16)


    @staticmethod
    def init_mask(batch_size: int, num_heads: int, mem_len: int) -> jnp.ndarray:
        return jnp.zeros((batch_size, num_heads, 1, mem_len + 1), dtype=jnp.bool_)


    @staticmethod
    def update_memory(old_memory: jnp.ndarray, new_memory: jnp.ndarray, mem_len: int) -> jnp.ndarray:
        combined = jnp.concatenate([old_memory, new_memory[:, None, :, :]], axis=1)
        return combined[:, -mem_len:, :, :]