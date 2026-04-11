from typing import Any, Dict

import jax
import jax.numpy as jnp

from src.modules.algorithms.shared.memory import MemoryManager
from src.modules.constants import CRAFTAX_MAX_RETURN


def run_eval_rollout(train_state: Any, env: Any, env_params: Any, rng: jnp.ndarray, num_eval_envs: int,
                     num_eval_steps: int, memory_manager: MemoryManager, use_action_masking: bool = False, action_dim: int = 42) -> Dict[str, jnp.ndarray]:

    rng, env_rng = jax.random.split(rng)
    obs, env_state = env.reset(env_rng, env_params)

    mem = memory_manager.init_memory(num_eval_envs)
    done = jnp.zeros((num_eval_envs,), dtype=jnp.bool_)
    last_action = jnp.zeros((num_eval_envs,), dtype=jnp.int32)
    action_mask = jnp.ones((num_eval_envs, action_dim), dtype=jnp.bool_)

    def _eval_step(carry, unused):
        obs, env_state, mem, done, last_action, action_mask, rng = carry

        rng, step_rng = jax.random.split(rng)

        mem = memory_manager.reset_on_done(mem, done)

        new_mem, primary_out, _ = memory_manager.forward_eval(train_state.params, mem, obs, last_action)

        if use_action_masking:
            primary_out = jnp.where(action_mask, primary_out, -1e9)

        action = jnp.argmax(primary_out, axis=-1)

        next_obs, next_env_state, reward, next_done, info = env.step(step_rng, env_state, action, env_params)

        next_action_mask = info.get("action_mask", jnp.ones((num_eval_envs, action_dim), dtype=jnp.bool_))

        next_last_action = jnp.where(next_done, jnp.zeros_like(action, dtype=jnp.int32), action.astype(jnp.int32))

        carry = (next_obs, next_env_state, new_mem, next_done, next_last_action, next_action_mask, rng)

        step_metrics = {
            "reward": reward,
            "done": next_done.astype(jnp.float32),
            "returned_episode": info.get("returned_episode", next_done).astype(jnp.float32),
            "returned_episode_returns": info.get("returned_episode_returns", jnp.zeros((num_eval_envs,))),
            "returned_episode_lengths": info.get("returned_episode_lengths", jnp.zeros((num_eval_envs,), dtype=jnp.int32)),
            "info": info,
        }

        return carry, step_metrics

    init_carry = (obs, env_state, mem, done, last_action, action_mask, rng)
    _, metrics = jax.lax.scan(_eval_step, init_carry, None, num_eval_steps)

    return compute_eval_metrics(metrics)


def compute_eval_metrics(metrics: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    returned_episode = metrics["returned_episode"]
    returned_episode_returns = metrics["returned_episode_returns"]
    returned_episode_lengths = metrics["returned_episode_lengths"]

    mask = returned_episode
    total_episodes = jnp.sum(mask)

    def _masked_mean(x, m):
        return jnp.sum(x * m) / (jnp.sum(m) + 1e-8)

    has_episodes = total_episodes > 0.5
    masked_returns = jnp.where(mask > 0.5, returned_episode_returns, jnp.array(0.0))

    result = {
        "eval/episode_return_mean": _masked_mean(returned_episode_returns, mask),
        "eval/episode_return_max": jnp.where(has_episodes, jnp.max(masked_returns), jnp.array(0.0)),
        "eval/episode_return_min": jnp.where(has_episodes,
            jnp.min(jnp.where(mask > 0.5, returned_episode_returns, jnp.array(1e9))),
            jnp.array(0.0)),
        "eval/episode_length_mean": _masked_mean(returned_episode_lengths.astype(jnp.float32), mask),
        "eval/total_episodes": total_episodes,
        "eval/total_reward": jnp.sum(metrics["reward"]),
    }

    result["eval/normalized_return_pct_mean"] = (result["eval/episode_return_mean"] / CRAFTAX_MAX_RETURN) * 100.0
    result["eval/normalized_return_pct_max"] = (result["eval/episode_return_max"] / CRAFTAX_MAX_RETURN) * 100.0
    result["eval/normalized_return_pct_min"] = (result["eval/episode_return_min"] / CRAFTAX_MAX_RETURN) * 100.0

    if "info" in metrics:
        def _compute_masked_mean(x):
            if hasattr(x, "ndim") and x.ndim == 0:
                return x.astype(jnp.float32)
            if x.shape != mask.shape:
                return jnp.array(0.0)
            return jnp.sum(x * mask) / (jnp.sum(mask) + 1e-8)

        info_metrics = jax.tree.map(_compute_masked_mean, metrics["info"])
        for k, v in info_metrics.items():
            result[f"eval/{k}"] = v

    return result


def create_eval_env(config, basic_env):
    from src.modules.environments.wrappers import LogWrapper, BatchEnvWrapper
    from src.modules.environments.action_mask_wrapper import ActionMaskWrapper

    if config.train.use_action_masking:
        basic_env = ActionMaskWrapper(basic_env)

    log_env = LogWrapper(basic_env)
    eval_env = BatchEnvWrapper(log_env, num_envs=config.validation.num_eval_envs)

    return eval_env
