from typing import Tuple
import jax
import jax.numpy as jnp
from src.modules.algorithms.shared.types import Transition


def compute_gae(traj_batch: Transition, last_val: jnp.ndarray, gamma: float, gae_lambda: float) -> Tuple[jnp.ndarray, jnp.ndarray]:
    last_val_f32 = last_val.astype(jnp.float32)
    traj_value_f32 = traj_batch.value.astype(jnp.float32)
    traj_reward_f32 = traj_batch.reward.astype(jnp.float32)


    def _get_advantages(gae_and_next_value, transition):
        gae, next_value = gae_and_next_value
        done, value, reward = transition
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * gae_lambda * (1 - done) * gae
        return (gae, value), gae


    _, advantages = jax.lax.scan(_get_advantages,(jnp.zeros_like(last_val_f32), last_val_f32), (traj_batch.done, traj_value_f32, traj_reward_f32),
                                 reverse=True, unroll=16)

    targets = advantages + traj_value_f32
    return advantages, targets


def normalize_advantages(advantages: jnp.ndarray) -> jnp.ndarray:
    return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
