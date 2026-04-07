from typing import Tuple, Any, Dict

import jax
import jax.numpy as jnp

from src.modules.algorithms.shared.types import Transition


def prepare_batch_transformer(traj_batch: Transition, advantages: jnp.ndarray, targets: jnp.ndarray, start_memories: jnp.ndarray) -> Tuple:
    init_mem_batch = start_memories
    per_step_masks = traj_batch.memories_mask

    return init_mem_batch, per_step_masks, traj_batch.obs, traj_batch.action, traj_batch.done, traj_batch.last_done, traj_batch.log_prob, traj_batch.value, traj_batch.last_action, advantages, targets



def compute_metrics(traj_batch: Transition, train_state: Any, total_timesteps: int) -> Dict[str, jnp.ndarray]:

    def _compute_masked_mean(x):
        mask = traj_batch.info["returned_episode"]
        if hasattr(x, "ndim") and x.ndim == 0:
            return x.astype(jnp.float32)
        if hasattr(x, "ndim") and x.ndim == 1 and x.shape[0] == mask.shape[0]:
            return x.astype(jnp.float32).mean()
        if x.shape != mask.shape:
            return jnp.array(0.0)
        return (x * mask).sum() / (mask.sum() + 1e-8)

    metric = jax.tree.map(_compute_masked_mean, traj_batch.info)

    metric["env_step"] = train_state.timesteps
    metric["update_steps"] = train_state.n_updates
    metric["progress_pct"] = (train_state.timesteps / total_timesteps) * 100.0

    return metric
