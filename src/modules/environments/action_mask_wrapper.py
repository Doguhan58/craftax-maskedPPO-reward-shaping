from functools import partial
import jax
from craftax.craftax.constants import BlockType, ItemType, DIRECTIONS, CLOSE_BLOCKS
from jax import numpy as jnp
from src.modules.environments.wrappers import GymnaxWrapper


class ActionMaskWrapper(GymnaxWrapper):
    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        info["action_mask"] = compute_action_mask(state)
        return obs, state, reward, done, info


def compute_action_mask(state):
    inv = state.inventory
    level_map = state.map[state.player_level]

    near_table, near_furnace = _check_nearby_blocks(state, level_map)

    max_health = 8 + state.player_strength
    max_energy = 7 + 2 * state.player_dexterity
    can_sleep_rest = (~state.is_sleeping) & (~state.is_resting)

    proj_count = state.player_projectiles.mask[state.player_level].sum()
    can_shoot = proj_count < 3

    facing_block = _get_facing_block(state, level_map)
    facing_fire_table = facing_block == BlockType.ENCHANTMENT_TABLE_FIRE.value
    facing_ice_table = facing_block == BlockType.ENCHANTMENT_TABLE_ICE.value
    facing_enchant = facing_fire_table | facing_ice_table
    has_matching_gem = jnp.where(facing_fire_table, inv.ruby >= 1, inv.sapphire >= 1)
    ench_base = (state.player_mana >= 9) & facing_enchant & has_matching_gem

    item_on_tile = state.item_map[state.player_level][state.player_position[0], state.player_position[1]]

    has_xp = state.player_xp >= 1

    near_table_furnace = near_table & near_furnace

    mask = jnp.array([
        #0:NOOP
        True,
        #1:LEFT
        True,
        #2:RIGHT
        True,
        #3:UP
        True,
        #4:DOWN
        True,
        #5:DO
        True,
        #6:SLEEP
        (state.player_energy < max_energy) & can_sleep_rest,
        #7:PLACE_STONE
        inv.stone > 0,
        #8:PLACE_TABLE
        inv.wood >= 2,
        #9:PLACE_FURNACE
        inv.stone > 0,
        #10:PLACE_PLANT
        inv.sapling > 0,
        #11:MAKE_WOOD_PICKAXE
        near_table & (inv.wood >= 1) & (inv.pickaxe < 1),
        #12:MAKE_STONE_PICKAXE
        near_table & (inv.wood >= 1) & (inv.stone >= 1) & (inv.pickaxe < 2),
        #13:MAKE_IRON_PICKAXE
        near_table_furnace & (inv.wood >= 1) & (inv.stone >= 1) & (inv.iron >= 1) & (inv.coal >= 1) & (inv.pickaxe < 3),
        #14:MAKE_WOOD_SWORD
        near_table & (inv.wood >= 1) & (inv.sword < 1),
        #15:MAKE_STONE_SWORD
        near_table & (inv.wood >= 1) & (inv.stone >= 1) & (inv.sword < 2),
        #16:MAKE_IRON_SWORD
        near_table_furnace & (inv.wood >= 1) & (inv.stone >= 1) & (inv.iron >= 1) & (inv.coal >= 1) & (inv.sword < 3),
        #17:REST
        (state.player_health < max_health) & can_sleep_rest,
        #18:DESCEND
        (item_on_tile == ItemType.LADDER_DOWN.value) & (state.player_level < 8),
        #19:ASCEND
        (item_on_tile == ItemType.LADDER_UP.value) & (state.player_level > 0),
        #20:MAKE_DIAMOND_PICKAXE
        near_table & (inv.wood >= 1) & (inv.diamond >= 3) & (inv.pickaxe < 4),
        #21:MAKE_DIAMOND_SWORD
        near_table & (inv.wood >= 1) & (inv.diamond >= 2) & (inv.sword < 4),
        #22:MAKE_IRON_ARMOUR
        near_table_furnace & (inv.iron >= 3) & (inv.coal >= 3) & (inv.armour < 1).any(),
        #23:MAKE_DIAMOND_ARMOUR
        near_table & (inv.diamond >= 3) & (inv.armour < 2).any(),
        #24:SHOOT_ARROW
        (inv.bow >= 1) & (inv.arrows >= 1) & can_shoot,
        #25:MAKE_ARROW
        near_table & (inv.wood >= 1) & (inv.stone >= 1) & (inv.arrows < 99),
        #26:CAST_FIREBALL
        state.learned_spells[0] & (state.player_mana >= 2) & can_shoot,
        #27:CAST_ICEBALL
        state.learned_spells[1] & (state.player_mana >= 2) & can_shoot,
        #28:PLACE_TORCH
        inv.torches > 0,
        #29-34:DRINK_POTION_RED through YELLOW
        inv.potions[0] > 0,
        inv.potions[1] > 0,
        inv.potions[2] > 0,
        inv.potions[3] > 0,
        inv.potions[4] > 0,
        inv.potions[5] > 0,
        #35:READ_BOOK
        (inv.books > 0) & (~state.learned_spells.all()),
        #36:ENCHANT_SWORD
        ench_base & (inv.sword > 0),
        #37:ENCHANT_ARMOUR
        ench_base & (inv.armour.sum() > 0),
        #38:MAKE_TORCH
        near_table & (inv.wood >= 1) & (inv.coal >= 1) & (inv.torches < 99),
        #39:LEVEL_UP_DEXTERITY
        has_xp & (state.player_dexterity < 5),
        #40:LEVEL_UP_STRENGTH
        has_xp & (state.player_strength < 5),
        #41:LEVEL_UP_INTELLIGENCE
        has_xp & (state.player_intelligence < 5),
        #42:ENCHANT_BOW
        ench_base & (inv.bow > 0)
    ], dtype=jnp.bool_)

    return mask


def _get_facing_block(state, level_map):
    direction = DIRECTIONS[state.player_direction]
    target = state.player_position + direction
    map_h, map_w = level_map.shape
    target_clamped = jnp.clip(target, jnp.array([0, 0]), jnp.array([map_h - 1, map_w - 1]))
    return level_map[target_clamped[0], target_clamped[1]]


def _check_nearby_blocks(state, level_map):
    positions = state.player_position + CLOSE_BLOCKS  #(8, 2)
    map_h, map_w = level_map.shape
    in_bounds = ((positions[:, 0] >= 0) & (positions[:, 0] < map_h) & (positions[:, 1] >= 0) & (positions[:, 1] < map_w))

    clamped = jnp.clip(positions, 0, jnp.array([map_h - 1, map_w - 1]))
    blocks = level_map[clamped[:, 0], clamped[:, 1]]

    near_table = jnp.any(in_bounds & (blocks == BlockType.CRAFTING_TABLE.value))
    near_furnace = jnp.any(in_bounds & (blocks == BlockType.FURNACE.value))
    return near_table, near_furnace
