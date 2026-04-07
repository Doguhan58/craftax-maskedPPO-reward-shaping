
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RewardShapingEnvState:
    env_state: Any
    max_stone: jnp.ndarray #(num_envs,) high-water mark
    prev_features: jnp.ndarray  #(num_envs, 11) cached features from previous step


#indices
_KILLS = 0
_IRON = 1
_DIAMOND = 2
_COAL = 3
_WOOD = 4
_STONE = 5
_ARMOR = 6
_LEVEL = 7
_STR = 8
_DEX = 9
_INT = 10


class RewardShapingWrapper:
    def __init__(self, env, num_envs: int, config):
        self._env = env
        self.num_envs = num_envs
        self.max_shaping = config.max_shaping_per_step

        self._weights = jnp.array([config.w_kill, config.w_iron, config.w_diamond, config.w_coal,
                                   config.w_wood, config.w_stone, config.w_armor, config.w_levelup,
                                   config.w_levelup, config.w_levelup])


    def __getattr__(self, name):
        return getattr(self._env, name)


    def _get_inner_state(self, state):
        inner = state
        while hasattr(inner, 'env_state'):
            inner = inner.env_state

        return inner


    def _extract_features_array(self, inner):
        level = inner.player_level.astype(jnp.int32)
        kills = inner.monsters_killed[jnp.arange(inner.monsters_killed.shape[0]), level]
        inv = inner.inventory

        return jnp.stack([kills, inv.iron, inv.diamond, inv.coal, inv.wood, inv.stone, jnp.sum(inv.armour, axis=-1), level, inner.player_strength, inner.player_dexterity, inner.player_intelligence], axis=-1).astype(jnp.float32)


    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        inner = self._get_inner_state(env_state)
        state = RewardShapingEnvState(env_state=env_state, max_stone=jnp.zeros(self.num_envs, dtype=jnp.int32), prev_features=self._extract_features_array(inner))

        return obs, state


    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        obs, new_inner_state, reward, done, info = self._env.step(rng, state.env_state, action, params)

        feat_after = self._extract_features_array(self._get_inner_state(new_inner_state))
        feat_before = state.prev_features

        #kill check: capped at 8, same floor, floor > 0
        kills_b = jnp.minimum(feat_before[:, _KILLS], 8)
        kills_a = jnp.minimum(feat_after[:, _KILLS], 8)
        kill_happened = ((kills_a > kills_b) & (feat_before[:, _LEVEL] > 0) & (feat_after[:, _LEVEL] == feat_before[:, _LEVEL]))

        conditions = jnp.stack([ kill_happened,
                                 feat_after[:, _IRON] > feat_before[:, _IRON],
                                 feat_after[:, _DIAMOND] > feat_before[:, _DIAMOND],
                                 feat_after[:, _COAL] > feat_before[:, _COAL],
                                 feat_after[:, _WOOD] > feat_before[:, _WOOD],
                                 feat_after[:, _STONE] > state.max_stone,
                                 feat_after[:, _ARMOR] > feat_before[:, _ARMOR],
                                 feat_after[:, _STR] > feat_before[:, _STR],
                                 feat_after[:, _DEX] > feat_before[:, _DEX],
                                 feat_after[:, _INT] > feat_before[:, _INT]],
                               axis=-1)


        bonus = (conditions.astype(jnp.float32) * self._weights).sum(axis=-1)
        bonus = jnp.clip(bonus, 0.0, self.max_shaping)

        #zero on episode end
        bonus = bonus * (1.0 - done.astype(jnp.float32))

        info["shaping_reward"] = bonus
        reward = reward + bonus

        new_max_stone = jnp.maximum(state.max_stone, feat_after[:, _STONE].astype(jnp.int32))
        new_max_stone = jnp.where(done, jnp.int32(0), new_max_stone)

        new_state = RewardShapingEnvState(env_state=new_inner_state, max_stone=new_max_stone, prev_features=feat_after)

        return obs, new_state, reward, done, info
