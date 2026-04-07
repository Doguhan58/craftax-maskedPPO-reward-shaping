from typing import Tuple, Any

import jax
import jax.numpy as jnp

def shuffle_batch(batch: Tuple, permutation: jnp.ndarray) -> Tuple:

    def _shuffle(x):
        if x.ndim >= 2:
            return jnp.take(x, permutation, axis=1)
        return x

    return jax.tree.map(_shuffle, batch)


def split_into_minibatches(batch: Tuple, num_minibatches: int) -> Tuple:

    def _make_minibatches(x):
        if x.ndim >= 2:
            new_shape = [x.shape[0], num_minibatches, -1] + list(x.shape[2:])
            reshaped = jnp.reshape(x, new_shape)
            return jnp.swapaxes(reshaped, 0, 1)
        return x

    return jax.tree.map(_make_minibatches, batch)


def build_segment_mask(segment_done: jnp.ndarray,
    memory_mask: jnp.ndarray, window_mem: int,
    window_grad: int, num_heads: int) -> jnp.ndarray:

    B = segment_done.shape[1]
    H = num_heads
    T = window_grad

    #part 1: Memory portion
    mem_mask_base = memory_mask[:, :, 0, :window_mem]  # [B, H, window_mem]
    mem_mask = jnp.broadcast_to(mem_mask_base[:, :, None, :], (B, H, T, window_mem))

    #part 2:  causal mask with episode boundary handling
    done_shifted = jnp.concatenate(
        [jnp.zeros((1, B), dtype=segment_done.dtype), segment_done[:-1]], axis=0)
    episode_id = jnp.cumsum(done_shifted.astype(jnp.int32), axis=0)  # [T, B]

    episode_id_q = episode_id[:, :, None]  # [T, B, 1]
    episode_id_k = episode_id[None, :, :]  # [1, T, B]
    episode_id_k = jnp.moveaxis(episode_id_k, 1, 2)  # [1, B, T]
    same_episode = (episode_id_q == episode_id_k)  # [T, B, T]

    causal = jnp.tril(jnp.ones((T, T), dtype=jnp.bool_))[:, None, :]  # [T, 1, T]
    steps_mask = same_episode & causal  # [T, B, T]

    #invalidate memory attention if episode reseted
    mem_valid = (episode_id[:, :, None] == 0)  # [T, B, 1]

    mem_mask = mem_mask & jnp.broadcast_to(jnp.transpose(mem_valid, (1, 0, 2))[:, None, :, :],  # [B, 1, T, 1]
                                            (B, H, T, window_mem))

    # Reshape steps_mask to [B, H, T, T]
    steps_mask = jnp.transpose(steps_mask, (1, 0, 2))[:, None, :, :]
    steps_mask = jnp.broadcast_to(steps_mask, (B, H, T, T))

    return jnp.concatenate([mem_mask, steps_mask], axis=-1)


def segment_rollout_for_window_grad(obs: jnp.ndarray, action: jnp.ndarray, done: jnp.ndarray, log_prob: jnp.ndarray, value: jnp.ndarray, last_action: jnp.ndarray,
                                    advantages: jnp.ndarray, targets: jnp.ndarray, per_step_masks: jnp.ndarray, num_steps: int, window_grad: int, action_mask: jnp.ndarray = None) -> Tuple:

    num_segments = num_steps // window_grad

    def _segment(x):
        if x is None:
            return None
        return jax.tree.map(lambda a: jnp.reshape(a, (num_segments, window_grad) + a.shape[1:]), x)

    return ( _segment(obs), _segment(action), _segment(done), _segment(log_prob), _segment(value), _segment(last_action),
             _segment(advantages), _segment(targets), _segment(per_step_masks), _segment(action_mask))



def accumulate_grads(accum_grads: Any, new_grads: Any) -> Any:
    return jax.tree.map(lambda a, g: a + g, accum_grads, new_grads)


def average_grads(accum_grads: Any, num_accumulation_steps: int) -> Any:
    return jax.tree.map(lambda g: g / num_accumulation_steps, accum_grads)


def split_minibatches_for_accumulation(minibatches: Tuple, num_accumulation_steps: int) -> Tuple:

    def _reshape_for_accum(x):
        if x.ndim >= 1 and x.shape[0] > 1:
            num_minibatches = x.shape[0]
            accum_batch_size = num_minibatches // num_accumulation_steps
            new_shape = (num_accumulation_steps, accum_batch_size) + x.shape[1:]
            return jnp.reshape(x, new_shape)
        return x

    return jax.tree.map(_reshape_for_accum, minibatches)


def update_avg_return_ema(avg_ret_ema: jnp.ndarray, current_avg: jnp.ndarray, decay: float = 0.99) -> jnp.ndarray:
    return (avg_ret_ema * decay + current_avg * (1 - decay)).astype(jnp.float32)
