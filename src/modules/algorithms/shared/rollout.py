from typing import Tuple

import distrax
import jax.numpy as jnp


def sample_action(logits: jnp.ndarray, rng: jnp.ndarray, action_mask: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
    if action_mask is not None:
        logits = jnp.where(action_mask, logits, -1e9)

    pi = distrax.Categorical(logits=logits)
    action = pi.sample(seed=rng)
    log_prob = pi.log_prob(action)
    return action, log_prob
