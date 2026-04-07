"""
Reference: https://github.com/Reytuag/transformerXL_PPO_JAX
"""
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen.initializers import constant, orthogonal
from src.modules.model.transformer import GTrXL


class ActorCriticTransformer(nn.Module):
    action_dim: int
    embed_dim: int = 256
    hidden_dim: int = 512
    num_heads: int = 8
    head_dim: int = 32
    num_layers: int = 2
    mlp_dim: int = 512
    dropout_rate: float = 0.0
    gru_bias: float = 2.0
    use_remat: bool = True
    norm_input: bool = True
    add_last_action: bool = True
    dtype: jnp.dtype = jnp.bfloat16

    def setup(self):
        self.transformer = GTrXL(embed_dim=self.embed_dim, num_heads=self.num_heads,
                                 head_dim=self.head_dim, num_layers=self.num_layers,
                                 mlp_dim=self.mlp_dim, dropout_rate=self.dropout_rate,
                                 gru_bias=self.gru_bias, use_remat=self.use_remat)

        #encoder
        self.input_norm = nn.LayerNorm(dtype=jnp.float32) if self.norm_input else None
        self.obs_encoder = nn.Dense(self.embed_dim, dtype=self.dtype)

        #actor head: 2 hidden layers + output
        self.actor_hidden1 = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), dtype=self.dtype)
        self.actor_hidden2 = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), dtype=self.dtype)
        self.actor_out = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0), dtype=self.dtype)

        #critic head: 2 hidden layers + output
        self.critic_hidden1 = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), dtype=self.dtype)
        self.critic_hidden2 = nn.Dense(self.hidden_dim, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0), dtype=self.dtype)
        self.critic_out = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0), dtype=self.dtype)


    def _normalize_and_encode_obs(self, obs, last_action=None):
        x = obs.astype(jnp.float32)

        if self.norm_input:
            x = self.input_norm(x)

        x = x.astype(self.dtype)

        if self.add_last_action and last_action is not None:
            last_action_oh = jax.nn.one_hot(last_action, self.action_dim).astype(self.dtype)
            x = jnp.concatenate([x, last_action_oh], axis=-1)

        obs_emb = self.obs_encoder(x)
        return obs_emb


    def _actor_critic_heads(self, x):
        actor_x = nn.relu(self.actor_hidden1(x))
        actor_x = nn.relu(self.actor_hidden2(actor_x))
        logits = self.actor_out(actor_x)

        critic_x = nn.relu(self.critic_hidden1(x))
        critic_x = nn.relu(self.critic_hidden2(critic_x))
        value = self.critic_out(critic_x)

        return logits, jnp.squeeze(value, axis=-1)


    #usually only used for evaluation
    def __call__(self, memories, obs, mask, last_action=None, train: bool = False):
        obs_emb = self._normalize_and_encode_obs(obs, last_action)
        x, memory_out = self.transformer.forward_eval(memories, obs_emb, mask)
        logits, value = self._actor_critic_heads(x)
        return memory_out, logits, value


    def model_forward_eval(self, memories, obs, mask, last_action=None):
        obs_emb = self._normalize_and_encode_obs(obs, last_action)
        x, memory_out = self.transformer.forward_eval(memories, obs_emb, mask)
        logits, value = self._actor_critic_heads(x)
        return logits, value, memory_out


    def model_forward_train_with_memory(self, memories, obs, mask, last_action=None):
        obs_emb = self._normalize_and_encode_obs(obs, last_action)
        x = self.transformer.forward_train(memories, obs_emb, mask)
        logits, value = self._actor_critic_heads(x)
        return logits, value, x


    def initialize_memory(self, batch_size: int, window_mem: int):
        return GTrXL.init_memory(batch_size, window_mem, self.num_layers, self.embed_dim)


    def initialize_mask(self, batch_size: int, window_mem: int):
        return GTrXL.init_mask(batch_size, self.num_heads, window_mem)
