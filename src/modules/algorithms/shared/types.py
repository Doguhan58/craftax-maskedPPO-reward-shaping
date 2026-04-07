from typing import Any, NamedTuple

import jax.numpy as jnp
from flax import struct
from flax.training.train_state import TrainState


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    last_done: jnp.ndarray
    last_action: jnp.ndarray
    info: Any

    #ppo
    value: jnp.ndarray = None
    log_prob: jnp.ndarray = None

    #transformer memory
    memories_mask: jnp.ndarray = None
    memories_indices: jnp.ndarray = None

    #action masking (none when disabled)
    action_mask: jnp.ndarray = None


class MemoryState(NamedTuple):
    data: Any
    mask: jnp.ndarray = None #[B, H, 1, window_mem + 1]
    mask_idx: jnp.ndarray = None #[B,]


@struct.dataclass
class CustomTrainState(TrainState):
    timesteps: int = 0
    n_updates: int = 0
