"""Agent interaction rules for Overcooked V3."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    NUM_PREP_CHAINS,
    PREP_PROCESSED_SHIFT,
    PREP_RAW_START,
    Agent,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import EVENT_NAMES, MAX_POTS, SHAPED_REWARDS


def sample_pot_cook_time(key: chex.PRNGKey, config: OvercookedV3Config) -> chex.Array:
    """Sample an inclusive ready-time duration or return the fixed duration."""
    if not config.pot_cook_time_range:
        return jnp.array(config.pot_cook_time, dtype=jnp.int32)
    min_cook_time, max_cook_time = config.pot_cook_time_range
    return jax.random.randint(
        key, (), min_cook_time, max_cook_time + 1, dtype=jnp.int32
    )


def process_interact(
    grid: chex.Array,
    agent: Agent,
    all_inventories: jnp.ndarray,
    recipe: int,
    pot_timers: chex.Array,
    pot_positions: chex.Array,
    pot_active_mask: chex.Array,
    config: OvercookedV3Config,
    pot_cook_time: Optional[chex.Array] = None,
    plate_stack: chex.Array = 0,
    dirty_pile: chex.Array = 0,
):
    """Process one agent's interact action."""
    if pot_cook_time is None:
        pot_cook_time = jnp.array(config.pot_cook_time, dtype=jnp.int32)

    inventory = agent.inventory
    fwd_pos, in_bounds = agent.pos.checked_move(agent.dir, config.width, config.height)
    shaped_reward = jnp.array(0.0, dtype=jnp.float32)
    interact_item, interact_ingredients, interact_extra = grid[fwd_pos.y, fwd_pos.x]
    plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

    is_plate_pile = in_bounds & (interact_item == StaticObject.PLATE_PILE)
    is_stocked_plate_pile = is_plate_pile
    if config.enable_dish_washing:
        is_stocked_plate_pile &= plate_stack > 0
    is_ingredient_pile = in_bounds & StaticObject.is_ingredient_pile(interact_item)
    is_pile = is_stocked_plate_pile | is_ingredient_pile
    is_pot = in_bounds & (interact_item == StaticObject.POT)
    is_goal = in_bounds & (interact_item == StaticObject.GOAL)
    is_counter = in_bounds & (
        (interact_item == StaticObject.WALL)
        | (interact_item == StaticObject.MOVING_WALL)
    )
    is_conveyor = in_bounds & (
        (interact_item == StaticObject.ITEM_CONVEYOR)
        | (interact_item == StaticObject.PLAYER_CONVEYOR)
    )
    cell_empty = interact_ingredients == 0
    inv_empty = inventory == 0
    inv_ingredient = DynamicObject.is_ingredient(inventory)
    inv_plate = inventory == DynamicObject.PLATE
    inv_dish = (inventory & DynamicObject.COOKED) != 0
    merged = interact_ingredients + inventory

    def _pot_timer(pot_idx):
        y, x = pot_positions[pot_idx]
        matches = (y == fwd_pos.y) & (x == fwd_pos.x) & pot_active_mask[pot_idx]
        return jax.lax.select(matches, pot_timers[pot_idx], 0)

    current_timer = jnp.max(jax.vmap(_pot_timer)(jnp.arange(MAX_POTS)))
    pot_cooked = is_pot & ((interact_ingredients & DynamicObject.COOKED) != 0)
    pot_burned = is_pot & ((interact_ingredients & DynamicObject.BURNED) != 0)
    pot_idle = is_pot & (current_timer == 0) & ~pot_cooked & ~pot_burned
    any_pot_cooking = jnp.any(pot_timers > config.pot_burn_time)

    dish_pickup = pot_cooked & inv_plate
    if config.shaped_rewards_enabled:
        shaped_reward += (
            dish_pickup
            * (merged == plated_recipe)
            * SHAPED_REWARDS["SOUP_IN_DISH"]
        )
    pickup = (
        (is_pile & inv_empty)
        | dish_pickup
        | (is_counter & ~cell_empty & inv_empty)
        | (is_conveyor & ~cell_empty & inv_empty)
    )

    pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3
    same_type = (
        DynamicObject.get_ingredient_type(interact_ingredients)
        == DynamicObject.get_ingredient_type(inventory)
    ) | (interact_ingredients == 0)
    pot_placement = pot_idle & inv_ingredient & ~pot_full & same_type
    selector = inventory | (inventory << 1)
    useful_pot_placement = (interact_ingredients & selector) < (recipe & selector)
    if config.shaped_rewards_enabled:
        shaped_reward += (
            pot_placement
            * useful_pot_placement
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )

    counter_drop = (is_counter | is_conveyor) & cell_empty & ~inv_empty
    drop = counter_drop | pot_placement

    prep_placement = jnp.array(False)
    prep_pickup = jnp.array(False)
    prep_action = jnp.array(False)
    chop_completes = jnp.array(False)
    chopping = jnp.array(False)
    blend_start = jnp.array(False)
    is_grill = jnp.array(False)
    processed = jnp.array(0, dtype=jnp.int32)
    if config.has_prep_stations:
        is_board = in_bounds & (interact_item == StaticObject.CUTTING_BOARD)
        is_grill = in_bounds & (interact_item == StaticObject.GRILL)
        is_blender = in_bounds & (interact_item == StaticObject.BLENDER)
        is_station = is_board | is_grill | is_blender
        raw = (
            is_board * DynamicObject.ingredient(PREP_RAW_START)
            + is_grill * DynamicObject.ingredient(PREP_RAW_START + 1)
            + is_blender * DynamicObject.ingredient(PREP_RAW_START + 2)
        )
        processed = raw << PREP_PROCESSED_SHIFT
        has_raw = is_station & ~cell_empty & (interact_ingredients == raw)
        has_processed = is_station & ~cell_empty & (interact_ingredients == processed)
        prep_placement = is_station & cell_empty & (inventory == raw)
        chopping = is_board & has_raw & inv_empty
        chop_completes = chopping & (interact_extra + 1 >= config.chop_stages)
        blend_start = is_blender & has_raw & inv_empty & (interact_extra == 0)
        prep_pickup = inv_empty & has_processed
        pickup |= prep_pickup
        drop |= prep_placement
        prep_action = chopping | blend_start

        processed_selector = processed | (processed << 1)
        safe_processed = jnp.maximum(processed, 1)
        recipe_needs_processed = (recipe & processed_selector) != 0
        station_type = (
            is_board * StaticObject.CUTTING_BOARD
            + is_grill * StaticObject.GRILL
            + is_blender * StaticObject.BLENDER
        )
        units_in_play = (
            jnp.sum(
                (grid[:, :, 0] == station_type)
                & (grid[:, :, 1] == raw)
                & is_station
            )
            + jnp.sum((grid[:, :, 1] & processed_selector) // safe_processed)
            + jnp.sum((all_inventories & processed_selector) // safe_processed)
        )
        needed = (recipe & processed_selector) // safe_processed
        if config.shaped_rewards_enabled:
            shaped_reward += (
                prep_placement
                * recipe_needs_processed
                * (units_in_play < needed)
                * SHAPED_REWARDS["PREP_PLACEMENT"]
            )
            shaped_reward += (
                prep_action
                * recipe_needs_processed
                * (units_in_play <= needed)
                * SHAPED_REWARDS["PREP_ACTION"]
            )
            shaped_reward += (
                prep_pickup
                * recipe_needs_processed
                * (units_in_play <= needed)
                * SHAPED_REWARDS["PREP_PICKUP"]
            )

    delivery = is_goal & inv_dish
    plate_return = jnp.array(False)
    dirty_pickup = jnp.array(False)
    wash = jnp.array(False)
    if config.enable_dish_washing:
        is_sink = in_bounds & (interact_item == StaticObject.SINK)
        is_dirty_pile = in_bounds & (interact_item == StaticObject.DIRTY_PLATE_PILE)
        plate_return = is_plate_pile & inv_plate
        dirty_pickup = is_dirty_pile & inv_empty & (dirty_pile > 0)
        wash = is_sink & DynamicObject.is_dirty_plate(inventory)

    no_effect = ~(pickup | drop | delivery | plate_return | dirty_pickup | wash)
    pile_item = (
        is_stocked_plate_pile * DynamicObject.PLATE
        + is_ingredient_pile * StaticObject.get_ingredient(interact_item)
    )
    ingredient_pickup = is_ingredient_pile & inv_empty
    selector = pile_item | (pile_item << 1)
    safe_item = jnp.maximum(pile_item, 1)
    grid_supply = jnp.sum((grid[:, :, 1] & selector) // safe_item)
    inv_supply = jnp.sum((all_inventories & selector) // safe_item)
    needed = (recipe & selector) // safe_item
    useful_ingredient_pickup = grid_supply + inv_supply < needed
    if config.has_prep_stations:
        pile_idx = DynamicObject.get_ingredient_type(pile_item)
        prep_raw = (pile_idx >= PREP_RAW_START) & (
            pile_idx < PREP_RAW_START + NUM_PREP_CHAINS
        )
        processed_item = pile_item << PREP_PROCESSED_SHIFT
        processed_selector = processed_item | (processed_item << 1)
        safe_processed = jnp.maximum(processed_item, 1)
        processed_supply = jnp.sum(
            (grid[:, :, 1] & processed_selector) // safe_processed
        ) + jnp.sum((all_inventories & processed_selector) // safe_processed)
        chain_needed = (recipe & processed_selector) // safe_processed
        useful_ingredient_pickup = jnp.where(
            prep_raw,
            grid_supply + inv_supply + processed_supply < chain_needed,
            useful_ingredient_pickup,
        )
    if config.shaped_rewards_enabled:
        shaped_reward += (
            ingredient_pickup
            * useful_ingredient_pickup
            * SHAPED_REWARDS["INGREDIENT_PICKUP"]
        )

    new_ingredients = drop * merged + no_effect * interact_ingredients
    if config.has_prep_stations:
        new_ingredients = jnp.where(chop_completes, processed, new_ingredients)
    auto_cook = pot_placement & (DynamicObject.ingredient_count(new_ingredients) == 3)

    def _update_timer(pot_idx):
        y, x = pot_positions[pot_idx]
        matches = (y == fwd_pos.y) & (x == fwd_pos.x) & pot_active_mask[pot_idx]
        timer = jax.lax.select(
            matches & auto_cook,
            pot_cook_time + config.pot_burn_time,
            pot_timers[pot_idx],
        )
        return jax.lax.select(matches & dish_pickup, 0, timer)

    new_timers = jax.vmap(_update_timer)(jnp.arange(MAX_POTS))
    new_extra = interact_extra
    if config.has_prep_stations:
        new_extra = jnp.where(
            prep_placement,
            jnp.where(is_grill, config.grill_cook_time + config.grill_burn_time, 0),
            new_extra,
        )
        new_extra = jnp.where(
            chopping, jnp.where(chop_completes, 0, interact_extra + 1), new_extra
        )
        new_extra = jnp.where(blend_start, config.blend_time, new_extra)
        new_extra = jnp.where(prep_pickup, 0, new_extra)

    new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(
        jnp.array([interact_item, new_ingredients, new_extra])
    )
    new_inventory = pickup * (pile_item + merged) + no_effect * inventory
    new_plate_stack = plate_stack
    new_dirty_pile = dirty_pile
    if config.enable_dish_washing:
        new_inventory = jnp.where(
            dirty_pickup, DynamicObject.PLATE | DynamicObject.DIRTY, new_inventory
        )
        new_inventory = jnp.where(wash, DynamicObject.PLATE, new_inventory)
        new_inventory = jnp.where(plate_return, 0, new_inventory)
        drew_plate = is_stocked_plate_pile & inv_empty
        new_plate_stack = (
            plate_stack - drew_plate.astype(jnp.int32) + plate_return.astype(jnp.int32)
        )
        new_dirty_pile = (
            dirty_pile + delivery.astype(jnp.int32) - dirty_pickup.astype(jnp.int32)
        )
        if config.shaped_rewards_enabled:
            shaped_reward += dirty_pickup * SHAPED_REWARDS["DIRTY_PLATE_PICKUP"]
            shaped_reward += wash * SHAPED_REWARDS["PLATE_WASH"]

    new_agent = agent.replace(inventory=new_inventory)
    correct_delivery = delivery & (inventory == plated_recipe)
    reward = correct_delivery * jnp.array(config.delivery_reward, dtype=jnp.float32)

    if config.shaped_rewards_enabled:
        plate_pickup = pickup & (new_inventory == DynamicObject.PLATE)
        plates_in_play = jnp.sum(
            (grid[:, :, 1] & DynamicObject.PLATE) != 0
        ) + jnp.sum((all_inventories & DynamicObject.PLATE) != 0)
        pot_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(grid[:, :, 1])
        useful_pots = jnp.sum(
            (grid[:, :, 0] == StaticObject.POT)
            & (pot_counts == 3)
            & ((grid[:, :, 1] & DynamicObject.BURNED) == 0)
        )
        useful_plate = plates_in_play < useful_pots
        shaped_reward += useful_plate * plate_pickup * SHAPED_REWARDS["PLATE_PICKUP"]
        shaped_reward += (
            any_pot_cooking
            * useful_plate
            * plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP_DURING_COOKING"]
        )

    events = jnp.array(
        (
            auto_cook & pot_placement,
            pot_placement,
            pickup,
            counter_drop,
            dish_pickup,
            0.0,
            correct_delivery,
            0.0,
            prep_placement,
            prep_action,
            prep_pickup,
            0.0,
            dirty_pickup,
            wash,
            plate_return,
        ),
        dtype=jnp.float32,
    )
    assert events.shape[0] == len(EVENT_NAMES)
    return (
        new_grid,
        new_agent,
        correct_delivery,
        reward,
        shaped_reward,
        events,
        new_timers,
        new_plate_stack,
        new_dirty_pile,
    )
