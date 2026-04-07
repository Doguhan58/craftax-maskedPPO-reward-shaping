from typing import Tuple

import distrax
import jax.numpy as jnp
from src.modules.algorithms.ppo.gae import normalize_advantages


#clipped ppo policy loss
def compute_policy_loss(logits: jnp.ndarray, actions: jnp.ndarray, old_log_probs: jnp.ndarray, advantages: jnp.ndarray,
                        clip_eps: float, normalize_adv: bool, action_mask: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, distrax.Categorical]:

    if action_mask is not None:
        logits = jnp.where(action_mask, logits, -1e9)

    pi = distrax.Categorical(logits=logits)
    log_prob_pred = pi.log_prob(actions)

    ratio = jnp.exp(log_prob_pred - old_log_probs)

    gae = normalize_advantages(advantages) if normalize_adv else advantages

    loss_actor1 = ratio * gae
    loss_actor2 = jnp.clip(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * gae
    loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

    entropy = pi.entropy().mean()

    return loss_actor, entropy, pi


def compute_value_loss(value_pred: jnp.ndarray, value_old: jnp.ndarray, targets: jnp.ndarray, clip_eps: float) -> jnp.ndarray:

    value_pred_n = value_pred.astype(jnp.float32)
    value_n = value_old.astype(jnp.float32)
    targets_n = targets.astype(jnp.float32)

    value_pred_clipped = value_n + (value_pred_n - value_n).clip(-clip_eps, clip_eps)

    value_losses = jnp.square(value_pred_n - targets_n)
    value_losses_clipped = jnp.square(value_pred_clipped - targets_n)

    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    return value_loss


def compute_ppo_loss(logits: jnp.ndarray, value_pred: jnp.ndarray, actions: jnp.ndarray,
                     old_log_probs: jnp.ndarray, old_values: jnp.ndarray, advantages: jnp.ndarray, targets: jnp.ndarray,
                     clip_eps: float, vf_coef: float, ent_coef: float, normalize_adv: bool, action_mask: jnp.ndarray = None) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:

    actor_loss, entropy, pi = compute_policy_loss(logits, actions, old_log_probs, advantages, clip_eps, normalize_adv, action_mask=action_mask)

    value_loss = compute_value_loss(value_pred, old_values, targets, clip_eps)

    total_loss = actor_loss + vf_coef * value_loss - ent_coef * entropy

    return total_loss, value_loss, actor_loss, entropy
