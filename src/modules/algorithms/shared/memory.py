from typing import Any, Tuple

import jax.numpy as jnp
from src.modules.algorithms.shared.types import MemoryState


class MemoryManager:

    def __init__(self, config, network):
        self.config = config
        self.network = network
        self.window_mem = config.model.memory_len
        self.num_layers = config.model.num_layers
        self.encoder_size = config.model.embed_dim
        self.num_heads = config.model.num_heads


    def init_memory(self, num_envs: int) -> MemoryState:
        data = jnp.zeros((num_envs, self.window_mem, self.num_layers, self.encoder_size), dtype=jnp.bfloat16)

        mask = jnp.zeros((num_envs, self.num_heads, 1, self.window_mem + 1), dtype=jnp.bool_)

        mask_idx = jnp.full((num_envs,), self.window_mem + 1, dtype=jnp.int32)

        return MemoryState(data=data, mask=mask, mask_idx=mask_idx)


    def reset_on_done(self, mem: MemoryState, done: jnp.ndarray) -> MemoryState:
        mask_idx = jnp.where(done, jnp.full_like(mem.mask_idx, self.window_mem), jnp.clip(mem.mask_idx - 1, 0, self.window_mem))

        indices = jnp.arange(self.window_mem + 1)[None, None, None, :]
        threshold = mask_idx[:, None, None, None]
        num_envs = done.shape[0]

        mask = jnp.broadcast_to((indices >= threshold).astype(jnp.bool_), (num_envs, self.num_heads, 1, self.window_mem + 1))

        #zero out memory data on done
        data = jnp.where(done[:, None, None, None],
                         jnp.zeros_like(mem.data), mem.data)

        return MemoryState(data=data, mask=mask, mask_idx=mask_idx)


    def forward_eval(self, params: Any, mem: MemoryState, obs: jnp.ndarray, last_action: jnp.ndarray) -> Tuple[MemoryState, Any, Any]:
        logits, value, memory_out = self.network.apply({"params": params}, mem.data, obs, mem.mask, last_action, method=self.network.model_forward_eval)

        new_data = jnp.roll(mem.data, -1, axis=1).at[:, -1, :, :].set(memory_out.astype(mem.data.dtype))

        new_mem = MemoryState(data=new_data, mask=mem.mask, mask_idx=mem.mask_idx)

        return new_mem, logits, value
