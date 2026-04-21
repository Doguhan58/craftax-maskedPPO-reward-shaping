**Always available:**
* 0 - NOOP
* 1 - LEFT
* 2 - RIGHT
* 3 - UP
* 4 - DOWN
* 5 - DO

**Rest / Sleep:**
* 6 - SLEEP if energy < max energy & is not resting / sleeping
* 17 - REST if health < max health & is not resting / sleeping

**Placement:**
* 7 - PLACE_STONE if stone > 0
* 8 - PLACE_TABLE if wood > 2
* 9 - PLACE_FURANCE stone > 0
* 10 - PLACE_PLANT sapling > 0
* 28 - PLACE_TORCH torches > 0 & is dark

**Equipment:**
For all of the following, the agent must be near a crafting table
* 11 - MAKE_WOOD_PICKAXE if wood ≥ 1 & pickaxe level < 1
* 12 - MAKE_STONE_PICKAXE if wood ≥ 1 & stone ≥ 1 & pickaxe level < 2
* 13 - MAKE_IRON_PICKAXE if near furnace & has wood, stone iron, coal ≥ 1 & pickaxe level < 3
* 20 - MAKE_DIAMOND_PICKAXE if diamond ≥ 3 & has wood ≥ 1 & pickaxe level < 4
* 14 - MAKE_WOOD_SWORD if wood ≥ 1 & sword level < 1
* 15 - MAKE_STONE_SWORD if wood ≥ 1 & stone ≥ 1 & sword level < 2
* 16 - MAKE_IRON_SWORD if near furnace & has wood, stone iron, coal ≥ 1 & sword level < 3
* 21 - MAKE_DIAMOND_SWORD if diamond ≥ 3 & has wood ≥ 1 & sword level < 4

**Armour:**
For all of the following, the agent must be near a crafting table
* 22 - MAKE_IRON_ARMOUR if near furnace & has iron, coal ≥ 3 & any Armour < 1
* 23 - MAKE_DIAMOND_ARMOUR if diamond ≥ 3 & any Armour < 2

**Ladders:**
* 18 - DESCEND if on tile LADDER_DOWN & level < 8
* 19 - ASCEND if on tile LADDER_UP & level > 0

**Combat:**
* 24 - SHOOT_ARROW if bow ≥ 1 & arrows ≥ 1
* 25 - MAKE_ARROW if near crafting table & arrows < 99 & wood ≥ 1 & stone ≥ 1
* 26 - CAST_FIREBALL if spell learned & mana ≥ 2
* 27 - CAST_ICEBALL if spell learned & mana ≥ 2

**Consumables:**
* 29 - DRINK_POTION_RED if red potion > 0
* 30 - DRINK_POTION_GREEN if green potion > 0
* 31 - DRINK_POTION_BLUE if blue potion > 0
* 32 - DRINK_POTION_PINK if pink potion > 0
* 33 - DRINK_POTION_CYAN if cyan potion > 0
* 34 - DRINK_POTION_YELLOW if yellow potion > 0
* 35 - READ_BOOK if book > 0 & not all spells learned

**Enchantment:**
For all of the following, the agent must have ≥ 9 mana & has to face the enchanting table & own its matching gem
* 36 - ENCHANT_SWORD if sword is available
* 37 - ENCHANT_ARMOUR if armour is available
* 42 - ENCHANT_BOW if bow is available

**Misc:**
* 38 - MAKE_TORCH if near a table & wood ≥ 1 & coal ≥ 1 & torches < 99
* 39 - LEVEL_UP_DEXTERITY if xp ≥ 1 & dexterity < 5
* 40 - LEVEL_UP_STRENGTH if xp ≥ 1 & strength < 5
* 41 - LEVEL_UP_INTELLIGENCE if xp ≥ 1 & intelligence < 5