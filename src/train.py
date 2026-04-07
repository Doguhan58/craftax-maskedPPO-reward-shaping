import os
import time

from jax import config as jaxconfig

os.environ["JAX_COMPILATION_CACHE_DIR"] = "/tmp/jax_cache"
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_triton_gemm_any=true '
    '--xla_gpu_enable_latency_hiding_scheduler=true '
    '--xla_gpu_enable_highest_priority_async_stream=true '
    '--xla_gpu_enable_pipelined_all_reduce=true '
)

jaxconfig.update("jax_compilation_cache_dir", "/tmp/jax_cache")
jaxconfig.update("jax_persistent_cache_min_entry_size_bytes", -1)
jaxconfig.update("jax_persistent_cache_min_compile_time_secs", 0)


import jax
import wandb
from flax.training import checkpoints

from src.arguments import Config
from src.modules.environments.craftax_env import CraftaxEnvironment
from src.modules.model.factory import create_network
from src.modules.algorithms.factory import create_algorithm


def save_checkpoint(config: Config, train_state, step: int, is_final: bool = False):
    checkpoint_dir = config.checkpoint_dir
    run_name = config.get_run_name()
    save_dir = os.path.abspath(os.path.join(checkpoint_dir, run_name))
    os.makedirs(save_dir, exist_ok=True)

    metadata_path = os.path.join(save_dir, "metadata.json")
    config.save_metadata(metadata_path)

    checkpoint_data = {"params": train_state.params}

    if is_final:
        checkpoints.save_checkpoint(ckpt_dir=save_dir, target=checkpoint_data, step=step,
                                    prefix="checkpoint_final_", keep=1, overwrite=True)
        print(f"Saved final checkpoint at step {step} to {save_dir}")
    else:
        checkpoints.save_checkpoint(ckpt_dir=save_dir, target=checkpoint_data, step=step,
                                    prefix="checkpoint_", keep=config.checkpoint.max_to_keep, overwrite=True)
        print(f"Saved checkpoint at step {step} to {save_dir}")


def train_entry(config: Config):
    craftax = CraftaxEnvironment(config)
    network = create_network(config, craftax.action_dim)
    algorithm = create_algorithm(config, craftax.env, network)

    if config.train.log_wandb:
        wandb.init(entity=config.train.wandb_entity, project=config.train.wandb_project,
                   tags=["PPO", "GTrXL", config.env.name.upper()],
                   name=config.get_run_name(), config=config.get_wandb_config(), mode="online")

        wandb.define_metric("walltime_sps", step_metric="walltime")
        wandb.define_metric("walltime_progress", step_metric="walltime")


    train_fn = algorithm.make_train()
    rng = jax.random.PRNGKey(config.train.seed)

    print(f"Starting training: {config.get_run_name()}")
    print(f"Algorithm: PPO + GTrXL")
    print(f"Environment: {config.env.name}")
    print(f"Num envs: {config.env.num_envs}")
    print(f"Total timesteps: {config.train.total_timesteps:,}")
    print(f"Num updates: {config.num_updates:,}")
    print(f"Window mem: {config.model.memory_len}")
    print(f"Window grad: {config.model.window_grad}")

    train_jit = jax.jit(train_fn)

    t0 = time.time()
    outs = jax.block_until_ready(train_jit(rng))
    total_time = time.time() - t0

    avg_sps = config.train.total_timesteps / total_time

    print(f"Training completed in {total_time:.2f}s")
    print(f"Average SPS: {avg_sps:,.0f}")

    if config.checkpoint.save_final:
        runner_state = outs["runner_state"]
        train_state = runner_state[0]
        save_checkpoint(config, train_state, config.num_updates, is_final=True)

    if config.train.log_wandb:
        wandb.run.summary["total_time_s"] = total_time
        wandb.run.summary["avg_sps"] = avg_sps
        wandb.run.summary["total_timesteps"] = config.train.total_timesteps
        wandb.finish()

    return outs


def main():
    config = Config()
    train_entry(config)


if __name__ == "__main__":
    main()
