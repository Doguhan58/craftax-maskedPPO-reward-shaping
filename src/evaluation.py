import os
import time
import glob as glob_mod
from dataclasses import dataclass
from typing import Optional, Any

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax.training import checkpoints

from src.arguments import Config
from src.modules.environments.craftax_env import CraftaxEnvironment
from src.modules.model.factory import create_network
from src.modules.algorithms.shared.memory import MemoryManager
from src.modules.algorithms.shared.validation import compute_eval_metrics


@dataclass
class EvalConfig:
    checkpoint_path: str = "checkpoints/ppo_gtrxl_"
    num_envs: int = 2048
    eval_steps: int = 100000
    deterministic: bool = True
    log_interval: int = 10
    wandb_project: str = "craftax-eval"
    wandb_entity: Optional[str] = None
    wandb_mode: str = "online"


class _EvalTrainState:
    def __init__(self, params):
        self.params = params


def _load_train_config(checkpoint_path: str) -> Config:
    checkpoint_path = os.path.abspath(checkpoint_path)
    metadata_path = os.path.join(checkpoint_path, "metadata.json")

    if os.path.exists(metadata_path):
        print(f"Loading config from metadata: {metadata_path}")
        metadata = Config.load_metadata(metadata_path)
        config = Config.from_checkpoint_metadata(metadata)
        print(f"Loaded model: PPO + GTrXL")

        return config
    else:
        print(f"No metadata found at {metadata_path}, using default config")

        return Config()


def _load_checkpoint(checkpoint_path: str, network: Any, env_wrapper: CraftaxEnvironment, config: Config) -> Any:
    checkpoint_path = os.path.abspath(checkpoint_path)

    dummy_obs = jnp.zeros((1, *env_wrapper.observation_shape))
    init_key = jax.random.PRNGKey(0)

    dummy_mem = jnp.zeros((1, config.model.memory_len, config.model.num_layers, config.model.embed_dim), dtype=jnp.bfloat16)
    dummy_mask = jnp.zeros((1, config.model.num_heads, 1, config.model.memory_len + 1), dtype=jnp.bool_)
    dummy_last_action = jnp.zeros((1,), dtype=jnp.int32)

    variables = network.init(init_key, dummy_mem, dummy_obs, dummy_mask, dummy_last_action)
    dummy_params = variables["params"]
    target = {"params": dummy_params}

    for prefix in ("checkpoint_final_", "checkpoint_"):
        try:
            checkpoint = checkpoints.restore_checkpoint(ckpt_dir=checkpoint_path, target=target, step=None, prefix=prefix)

            if checkpoint is not None and "params" in checkpoint:
                pattern = os.path.join(checkpoint_path, f"{prefix}*")
                if glob_mod.glob(pattern):
                    print(f"Loaded checkpoint from {checkpoint_path} (prefix={prefix!r})")
                    return checkpoint["params"]
        except Exception:
            continue

    raise RuntimeError(f"Could not load checkpoint from {checkpoint_path}. "f"No files matching checkpoint_final_* or checkpoint_* found.")


def main():
    eval_config = EvalConfig()
    checkpoint_path = os.path.abspath(eval_config.checkpoint_path)

    config = _load_train_config(checkpoint_path)

    #override env settings for evaluation
    config.env.num_envs = eval_config.num_envs
    config.env.use_optimistic_resets = False
    config.reward_shaping.enabled = False

    env_wrapper = CraftaxEnvironment(config)
    network = create_network(config, env_wrapper.action_dim)

    params = _load_checkpoint(checkpoint_path, network, env_wrapper, config)
    train_state = _EvalTrainState(params)

    memory_manager = MemoryManager(config, network)

    checkpoint_name = os.path.basename(checkpoint_path)
    if eval_config.wandb_mode != "disabled":
        wandb.init(entity=eval_config.wandb_entity,
                   project=eval_config.wandb_project,
                   name=f"eval_{checkpoint_name}",
                   group=checkpoint_name,
                   job_type="eval",
                   config={
                       "checkpoint_path": checkpoint_path,
                       "num_envs": eval_config.num_envs,
                       "eval_steps": eval_config.eval_steps,
                       "deterministic": eval_config.deterministic,
                       "algorithm": "ppo",
                       "backend": "transformer"
                   }, mode=eval_config.wandb_mode)

    rng = jax.random.PRNGKey(0)
    chunk_size = eval_config.log_interval
    num_chunks = eval_config.eval_steps // chunk_size
    actual_steps = num_chunks * chunk_size
    print(f"\nRunning evaluation: {actual_steps} steps x {eval_config.num_envs} envs")

    env = env_wrapper.env
    env_params = env_wrapper.env_params
    num_eval_envs = eval_config.num_envs
    log_wandb = eval_config.wandb_mode != "disabled"

    use_action_masking = config.train.use_action_masking
    action_dim = env_wrapper.action_dim
    deterministic = eval_config.deterministic


    def _eval_step(carry, unused):
        obs, env_state, mem, done, last_action, action_mask, rng = carry

        rng, step_rng = jax.random.split(rng)

        mem = memory_manager.reset_on_done(mem, done)

        new_mem, primary_out, _ = memory_manager.forward_eval(train_state.params, mem, obs, last_action)

        if use_action_masking:
            primary_out = jnp.where(action_mask, primary_out, -1e9)

        if deterministic:
            action = jnp.argmax(primary_out, axis=-1)
        else:
            rng, action_rng = jax.random.split(rng)
            pi = distrax.Categorical(logits=primary_out)
            action = pi.sample(seed=action_rng)

        next_obs, next_env_state, reward, next_done, info = env.step(step_rng, env_state, action, env_params)

        next_action_mask = info.get("action_mask", jnp.ones((num_eval_envs, action_dim), dtype=jnp.bool_))

        next_last_action = jnp.where( next_done, jnp.zeros_like(action, dtype=jnp.int32), action.astype(jnp.int32))

        carry = (next_obs, next_env_state, new_mem, next_done, next_last_action, next_action_mask, rng)

        step_metrics = {
            "reward": reward,
            "done": next_done.astype(jnp.float32),
            "returned_episode": info.get("returned_episode", next_done).astype(jnp.float32),
            "returned_episode_returns": info.get("returned_episode_returns", jnp.zeros((num_eval_envs,))),
            "returned_episode_lengths": info.get("returned_episode_lengths", jnp.zeros((num_eval_envs,), dtype=jnp.int32)),
            "info": info}

        return carry, step_metrics


    def _outer_step(carry, chunk_idx):
        eval_carry = carry

        eval_carry, chunk_metrics = jax.lax.scan(_eval_step, eval_carry, None, chunk_size)

        agg = compute_eval_metrics(chunk_metrics)
        agg["eval/step"] = (chunk_idx + 1) * chunk_size

        if log_wandb:

            def callback(metric, step):
                wandb.log({k: float(v) for k, v in metric.items()}, step=int(step))

            jax.debug.callback(callback, agg, agg["eval/step"])

        return eval_carry, agg


    def run_eval_with_logging(rng):
        rng, env_rng = jax.random.split(rng)
        obs, env_state = env.reset(env_rng, env_params)

        mem = memory_manager.init_memory(num_eval_envs)
        done = jnp.zeros((num_eval_envs,), dtype=jnp.bool_)
        last_action = jnp.zeros((num_eval_envs,), dtype=jnp.int32)
        action_mask = jnp.ones((num_eval_envs, action_dim), dtype=jnp.bool_)

        init_carry = (obs, env_state, mem, done, last_action, action_mask, rng)
        _, all_chunk_metrics = jax.lax.scan(_outer_step, init_carry, jnp.arange(num_chunks))

        return all_chunk_metrics

    t0 = time.time()
    all_chunk_metrics = jax.jit(run_eval_with_logging)(rng)
    all_chunk_metrics = jax.tree_util.tree_map(
        lambda x: np.asarray(x), all_chunk_metrics)
    dt = time.time() - t0

    metrics = {
        "eval/episode_return_mean": float(np.mean(all_chunk_metrics["eval/episode_return_mean"])),
        "eval/episode_return_max": float(np.max(all_chunk_metrics["eval/episode_return_max"])),
        "eval/episode_return_min": float(np.min(all_chunk_metrics["eval/episode_return_min"])),
        "eval/episode_length_mean": float(np.mean(all_chunk_metrics["eval/episode_length_mean"])),
        "eval/total_episodes": float(np.sum(all_chunk_metrics["eval/total_episodes"])),
        "eval/total_reward": float(np.sum(all_chunk_metrics["eval/total_reward"]))}

    sps = (actual_steps * eval_config.num_envs) / max(dt, 1e-8)
    metrics["eval/sps"] = sps
    metrics["eval/wall_time"] = dt

    if eval_config.wandb_mode != "disabled":
        wandb.finish()


    print(f"Checkpoint: {checkpoint_path}")
    print(f"Algorithm: PPO + GTrXL")
    print(f"Steps: {actual_steps}")
    print(f"Num envs: {eval_config.num_envs}")
    print(f"Wall time: {dt:.1f}s")
    print(f"SPS: {sps:,.0f}")
    print(f"Episodes: {metrics.get('eval/total_episodes', 0):.0f}")
    print(f"Mean return: {metrics.get('eval/episode_return_mean', 0):.3f}")
    print(f"Max return: {metrics.get('eval/episode_return_max', 0):.3f}")
    print(f"Min return: {metrics.get('eval/episode_return_min', 0):.3f}")
    print(f"Mean length: {metrics.get('eval/episode_length_mean', 0):.1f}")
    print(f"Total reward: {metrics.get('eval/total_reward', 0):.3f}")


if __name__ == "__main__":
    main()
