
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


@struct.dataclass
class RewardShapingEnvState:
    env_state: Any
    max_stone: jnp.ndarray #scalar int, high-water mark
    prev_features: jnp.ndarray #(14,) cached features from previous step
    floor_cleared: jnp.ndarray #(9,) floor clears this episode
    armour_enchanted_seen: jnp.ndarray #(4,) bool, per-slot once per ep flags
    bow_enchanted_seen: jnp.ndarray #scalar bool, once per ep


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
_SAPPHIRE = 11
_RUBY = 12
_FLOOR_CLEARED = 13


class RewardShapingWrapper:
    def __init__(self, env, config):
        self._env = env
        self.max_shaping = config.max_shaping_per_step

        self._weights = jnp.array([config.w_kill, config.w_iron, config.w_diamond, config.w_coal,
                                   config.w_wood, config.w_stone, config.w_armor, config.w_levelup,
                                   config.w_levelup, config.w_levelup, config.w_sapphire, config.w_ruby])

        self._w_depth = config.w_depth
        self._depth_gamma = config.depth_gamma
        self._kill_floor_scale = config.kill_floor_scale
        self._w_kill_base = config.w_kill
        self._w_floor_clear = config.w_floor_clear
        self._w_enchant_armor = config.w_enchant_armor
        self._w_enchant_bow = config.w_enchant_bow


    def __getattr__(self, name):
        return getattr(self._env, name)


    def _extract_features_array(self, inner):
        level = inner.player_level.astype(jnp.int32)
        kills = inner.monsters_killed[level]
        inv = inner.inventory
        floor_cleared = (kills >= 8).astype(jnp.float32)

        return jnp.stack([kills, inv.iron, inv.diamond, inv.coal, inv.wood, inv.stone, jnp.sum(inv.armour, axis=-1), level,
                          inner.player_strength, inner.player_dexterity, inner.player_intelligence,
                          inv.sapphire, inv.ruby, floor_cleared]).astype(jnp.float32)


    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        obs, env_state = self._env.reset(rng, params)
        inner = env_state.env_state
        state = RewardShapingEnvState(env_state=env_state, max_stone=jnp.int32(0),
                                      prev_features=self._extract_features_array(inner),
                                      floor_cleared=jnp.zeros((9,), dtype=jnp.bool_),
                                      armour_enchanted_seen=jnp.zeros((4,), dtype=jnp.bool_), bow_enchanted_seen=jnp.bool_(False))

        return obs, state


    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):
        obs, new_inner_state, reward, done, info = self._env.step(rng, state.env_state, action, params)

        feat_after = self._extract_features_array(new_inner_state.env_state)
        feat_before = state.prev_features

        #kill check: capped at 8, same floor, floor > 0
        kills_b = jnp.minimum(feat_before[_KILLS], 8)
        kills_a = jnp.minimum(feat_after[_KILLS], 8)
        level_before = feat_before[_LEVEL]
        level_after_f = feat_after[_LEVEL]
        same_floor = level_after_f == level_before
        kill_happened = (kills_a > kills_b) & (level_before > 0) & same_floor

        conditions = jnp.stack([ kill_happened,
                                 feat_after[_IRON] > feat_before[_IRON],
                                 feat_after[_DIAMOND] > feat_before[_DIAMOND],
                                 feat_after[_COAL] > feat_before[_COAL],
                                 feat_after[_WOOD] > feat_before[_WOOD],
                                 feat_after[_STONE] > state.max_stone,
                                 feat_after[_ARMOR] > feat_before[_ARMOR],
                                 feat_after[_STR] > feat_before[_STR],
                                 feat_after[_DEX] > feat_before[_DEX],
                                 feat_after[_INT] > feat_before[_INT],
                                 feat_after[_SAPPHIRE] > feat_before[_SAPPHIRE],
                                 feat_after[_RUBY] > feat_before[_RUBY]])


        bonus = (conditions.astype(jnp.float32) * self._weights).sum()

        #floor-scaled kill extra: w_kill * kill_floor_scale * level
        floor_kill_extra = kill_happened.astype(jnp.float32) * self._w_kill_base * self._kill_floor_scale * level_after_f
        bonus = bonus + floor_kill_extra

        #depth pbrs: F(s,s') = gamma * phi(s') - phi(s), phi(s) = w_depth * level
        phi_before = self._w_depth * level_before
        phi_after = self._w_depth * level_after_f
        depth_bonus = self._depth_gamma * phi_after - phi_before
        bonus = bonus + depth_bonus

        #floor-clear milestone
        level_after = level_after_f.astype(jnp.int32)
        just_cleared = ((feat_before[_FLOOR_CLEARED] < 0.5) & (feat_after[_FLOOR_CLEARED] >= 0.5) & same_floor & (level_after_f > 0))

        already_cleared_this_ep = state.floor_cleared[level_after]
        floor_clear_bonus = just_cleared.astype(jnp.float32) * (~already_cleared_this_ep).astype(jnp.float32) * self._w_floor_clear
        bonus = bonus + floor_clear_bonus

        #enchantment milestones (one-shot per slot / per bow, per ep)
        inner_after = new_inner_state.env_state
        armour_now_ench = inner_after.armour_enchantments > 0
        bow_now_ench = inner_after.bow_enchantment > 0

        new_armour_slots = armour_now_ench & (~state.armour_enchanted_seen)
        new_bow = bow_now_ench & (~state.bow_enchanted_seen)

        armour_ench_bonus = jnp.sum(new_armour_slots.astype(jnp.float32)) * self._w_enchant_armor
        bow_ench_bonus = new_bow.astype(jnp.float32) * self._w_enchant_bow
        bonus = bonus + armour_ench_bonus + bow_ench_bonus

        #symmetric clip as pbrs can be negative on ascend
        bonus = jnp.clip(bonus, -self.max_shaping, self.max_shaping)

        #zero on episode end
        done_f = done.astype(jnp.float32)
        bonus = bonus * (1.0 - done_f)

        info["shaping_reward"] = bonus
        info["depth_bonus"] = depth_bonus * (1.0 - done_f)
        reward = reward + bonus

        new_max_stone = jnp.maximum(state.max_stone, feat_after[_STONE].astype(jnp.int32))
        new_max_stone = jnp.where(done, jnp.int32(0), new_max_stone)

        new_floor_cleared = state.floor_cleared.at[level_after].set(already_cleared_this_ep | just_cleared)
        new_floor_cleared = jnp.where(done, jnp.zeros_like(new_floor_cleared), new_floor_cleared)

        new_armour_seen = state.armour_enchanted_seen | new_armour_slots
        new_armour_seen = jnp.where(done, jnp.zeros_like(new_armour_seen), new_armour_seen)

        new_bow_seen = state.bow_enchanted_seen | new_bow
        new_bow_seen = jnp.where(done, jnp.bool_(False), new_bow_seen)

        new_state = RewardShapingEnvState(env_state=new_inner_state, max_stone=new_max_stone, prev_features=feat_after,
                                          floor_cleared=new_floor_cleared,
                                          armour_enchanted_seen=new_armour_seen,
                                          bow_enchanted_seen=new_bow_seen)

        return obs, new_state, reward, done, info
