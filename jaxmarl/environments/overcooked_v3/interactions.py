"""Agent interaction rules for Overcooked V3."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    DynamicObject,
    StaticObject,
    Agent,
    PREP_RAW_START,
    NUM_PREP_CHAINS,
    PREP_PROCESSED_SHIFT,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS, SHAPED_REWARDS


def sample_pot_cook_time(
    key: chex.PRNGKey, config: OvercookedV3Config
) -> chex.Array:
    """Sample an inclusive ready-time duration or return the fixed duration."""
    if not config.pot_cook_time_range:
        return jnp.array(config.pot_cook_time, dtype=jnp.int32)

    min_cook_time, max_cook_time = config.pot_cook_time_range
    return jax.random.randint(
        key,
        (),
        min_cook_time,
        max_cook_time + 1,
        dtype=jnp.int32,
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
    """Process an interact action for an agent."""
    if pot_cook_time is None:
        pot_cook_time = jnp.array(config.pot_cook_time, dtype=jnp.int32)

    inventory = agent.inventory
    fwd_pos, fwd_pos_in_bounds = agent.pos.checked_move(
        agent.dir, config.width, config.height
    )

    shaped_reward = jnp.array(0.0, dtype=float)

    interact_cell = grid[fwd_pos.y, fwd_pos.x]
    interact_item = interact_cell[0]
    interact_ingredients = interact_cell[1]
    interact_extra = interact_cell[2]

    plated_recipe = recipe | DynamicObject.PLATE | DynamicObject.COOKED

    # What is the object?
    object_is_plate_pile = fwd_pos_in_bounds & (
        interact_item == StaticObject.PLATE_PILE
    )
    # With dish washing the plate pile is a finite stack, so it can only be
    # drawn from while it still holds a plate. Otherwise it is infinite.
    if config.enable_dish_washing:
        object_is_plate_pile_stocked = object_is_plate_pile & (plate_stack > 0)
    else:
        object_is_plate_pile_stocked = object_is_plate_pile
    object_is_ingredient_pile = (
        fwd_pos_in_bounds & StaticObject.is_ingredient_pile(interact_item)
    )
    object_is_pile = object_is_plate_pile_stocked | object_is_ingredient_pile

    object_is_pot = fwd_pos_in_bounds & (interact_item == StaticObject.POT)
    object_is_goal = fwd_pos_in_bounds & (interact_item == StaticObject.GOAL)
    object_is_wall = fwd_pos_in_bounds & (
        (interact_item == StaticObject.WALL)
        | (interact_item == StaticObject.MOVING_WALL)
    )
    object_is_conveyor = fwd_pos_in_bounds & (
        (interact_item == StaticObject.ITEM_CONVEYOR)
        | (interact_item == StaticObject.PLAYER_CONVEYOR)
    )
    object_has_no_ingredients = interact_ingredients == 0

    # What is in inventory?
    inventory_is_empty = inventory == 0
    inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
    inventory_is_plate = inventory == DynamicObject.PLATE
    inventory_is_dish = (inventory & DynamicObject.COOKED) != 0

    merged_ingredients = interact_ingredients + inventory

    # Pot timers live in State; the grid's extra channel is reserved for
    # conveyor and moving-wall directions.
    def _timer_for_pot(pot_idx):
        pot_y, pot_x = pot_positions[pot_idx]
        is_this_pot = (
            (pot_y == fwd_pos.y)
            & (pot_x == fwd_pos.x)
            & pot_active_mask[pot_idx]
        )
        return jax.lax.select(is_this_pot, pot_timers[pot_idx], 0)

    current_pot_timer = jnp.max(jax.vmap(_timer_for_pot)(jnp.arange(MAX_POTS)))
    pot_is_cooked = object_is_pot * (
        (interact_ingredients & DynamicObject.COOKED) != 0
    )
    pot_is_cooking = object_is_pot * (current_pot_timer > 0) * ~pot_is_cooked
    pot_is_burned = object_is_pot * (
        (interact_ingredients & DynamicObject.BURNED) != 0
    )
    pot_is_idle = (
        object_is_pot
        * (current_pot_timer == 0)
        * ~pot_is_cooked
        * ~pot_is_burned
    )
    any_pot_cooking = jnp.any(pot_timers > config.pot_burn_time)

    # Check if pot is ready.
    pot_is_ready = pot_is_cooked

    # Pickup success conditions
    successful_dish_pickup = pot_is_ready * inventory_is_plate
    is_dish_pickup_useful = merged_ingredients == plated_recipe
    if config.shaped_rewards_enabled:
        shaped_reward += (
            successful_dish_pickup
            * is_dish_pickup_useful
            * SHAPED_REWARDS["SOUP_IN_DISH"]
        )

    successful_pickup = (
        object_is_pile * inventory_is_empty
        + successful_dish_pickup
        + object_is_wall * ~object_has_no_ingredients * inventory_is_empty
        + object_is_conveyor * ~object_has_no_ingredients * inventory_is_empty
    )

    # Pot placement
    pot_full = DynamicObject.ingredient_count(interact_ingredients) == 3

    # Check same ingredient type for pot
    pot_ingredient_type = DynamicObject.get_ingredient_type(interact_ingredients)
    inventory_ingredient_type = DynamicObject.get_ingredient_type(inventory)
    same_ingredient_type = (pot_ingredient_type == inventory_ingredient_type) | (
        interact_ingredients == 0
    )

    successful_pot_placement = (
        pot_is_idle * inventory_is_ingredient * ~pot_full * same_ingredient_type
    )
    ingredient_selector = inventory | (inventory << 1)
    is_pot_placement_useful = (interact_ingredients & ingredient_selector) < (
        recipe & ingredient_selector
    )
    if config.shaped_rewards_enabled:
        shaped_reward += (
            successful_pot_placement
            * is_pot_placement_useful
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )

    # Drop on counter/conveyor
    successful_counter_drop = (
        (object_is_wall | object_is_conveyor)
        * object_has_no_ingredients
        * ~inventory_is_empty
    )
    successful_drop = successful_counter_drop | successful_pot_placement

    # Prep stations: cutting board (repeated interacts), grill (auto-cook
    # timer), blender (manual start + timer). Each station only accepts a
    # single unit of its matching raw ingredient; the processed result is
    # a separate ingredient type (raw idx + offset).
    successful_prep_placement = jnp.array(False)
    successful_prep_pickup = jnp.array(False)
    prep_action = jnp.array(False)
    chop_completes = jnp.array(False)
    successful_chop = jnp.array(False)
    successful_blend_start = jnp.array(False)
    object_is_grill = jnp.array(False)
    station_processed = jnp.array(0)
    if config.has_prep_stations:
        object_is_cutting_board = fwd_pos_in_bounds & (
            interact_item == StaticObject.CUTTING_BOARD
        )
        object_is_grill = fwd_pos_in_bounds & (
            interact_item == StaticObject.GRILL
        )
        object_is_blender = fwd_pos_in_bounds & (
            interact_item == StaticObject.BLENDER
        )
        object_is_prep_station = (
            object_is_cutting_board | object_is_grill | object_is_blender
        )

        # Raw/processed encodings for the station being faced
        station_raw = (
            object_is_cutting_board * DynamicObject.ingredient(PREP_RAW_START)
            + object_is_grill * DynamicObject.ingredient(PREP_RAW_START + 1)
            + object_is_blender * DynamicObject.ingredient(PREP_RAW_START + 2)
        )
        station_processed = station_raw << PREP_PROCESSED_SHIFT

        station_is_empty = interact_ingredients == 0
        station_has_raw = object_is_prep_station & ~station_is_empty & (
            interact_ingredients == station_raw
        )
        station_has_processed = object_is_prep_station & ~station_is_empty & (
            interact_ingredients == station_processed
        )

        successful_prep_placement = (
            object_is_prep_station & station_is_empty & (inventory == station_raw)
        )

        # Cutting board: each empty-handed interact advances chopping
        successful_chop = (
            object_is_cutting_board & station_has_raw & inventory_is_empty
        )
        chop_completes = successful_chop & (
            interact_extra + 1 >= config.chop_stages
        )

        # Blender: empty-handed interact starts the (idle) blender
        successful_blend_start = (
            object_is_blender
            & station_has_raw
            & inventory_is_empty
            & (interact_extra == 0)
        )

        # Pickup: stations only ever release the processed result, so
        # placing a raw ingredient is a commitment (as with the pot).
        # Letting the grill hand raw meat back made place -> pick up ->
        # place a two-step loop that farmed PREP_PLACEMENT indefinitely,
        # and grill maps collapsed into pickup spam with zero deliveries.
        successful_prep_pickup = inventory_is_empty & station_has_processed

        successful_pickup = successful_pickup + successful_prep_pickup
        successful_drop = successful_drop | successful_prep_placement
        prep_action = successful_chop | successful_blend_start

        # Shaped rewards, gated on the recipe actually needing this chain
        # and on how many units (raw on stations + processed anywhere) are
        # already in play, so cycles of place/retrieve can't be farmed.
        processed_selector = station_processed | (station_processed << 1)
        recipe_needs_processed = (recipe & processed_selector) != 0
        safe_processed = jnp.maximum(station_processed, 1)
        station_type_val = (
            object_is_cutting_board * StaticObject.CUTTING_BOARD
            + object_is_grill * StaticObject.GRILL
            + object_is_blender * StaticObject.BLENDER
        )
        raw_on_stations = jnp.sum(
            (grid[:, :, 0] == station_type_val)
            & (grid[:, :, 1] == station_raw)
            & object_is_prep_station
        )
        processed_in_grid = jnp.sum(
            (grid[:, :, 1] & processed_selector) // safe_processed
        )
        processed_in_inventories = jnp.sum(
            (all_inventories & processed_selector) // safe_processed
        )
        prep_units_in_play = (
            raw_on_stations + processed_in_grid + processed_in_inventories
        )
        processed_needed = (recipe & processed_selector) // safe_processed
        if config.shaped_rewards_enabled:
            shaped_reward += (
                successful_prep_placement
                * recipe_needs_processed
                * (prep_units_in_play < processed_needed)
                * SHAPED_REWARDS["PREP_PLACEMENT"]
            )
            shaped_reward += (
                prep_action
                * recipe_needs_processed
                * (prep_units_in_play <= processed_needed)
                * SHAPED_REWARDS["PREP_ACTION"]
            )
            shaped_reward += (
                successful_prep_pickup
                * station_has_processed
                * recipe_needs_processed
                * (prep_units_in_play <= processed_needed)
                * SHAPED_REWARDS["PREP_PICKUP"]
            )

    # Delivery
    successful_delivery = object_is_goal * inventory_is_dish

    # Dish washing. Plates are conserved: the stack, the dirty pile, every
    # inventory and every plate lying on the grid always sum to num_plates.
    # Delivering sends a plate to the dirty pile, and only the sink turns a
    # dirty plate back into a clean one.
    successful_plate_return = jnp.array(False)
    successful_dirty_pickup = jnp.array(False)
    successful_wash = jnp.array(False)
    if config.enable_dish_washing:
        object_is_sink = fwd_pos_in_bounds & (interact_item == StaticObject.SINK)
        object_is_dirty_pile = fwd_pos_in_bounds & (
            interact_item == StaticObject.DIRTY_PLATE_PILE
        )
        inventory_is_dirty_plate = DynamicObject.is_dirty_plate(inventory)

        # Put a clean plate back on the stack
        successful_plate_return = object_is_plate_pile & inventory_is_plate
        # Collect a dirty plate that needs washing
        successful_dirty_pickup = (
            object_is_dirty_pile & inventory_is_empty & (dirty_pile > 0)
        )
        # Wash the held dirty plate clean
        successful_wash = object_is_sink & inventory_is_dirty_plate

    no_effect = (
        ~successful_pickup
        * ~successful_drop
        * ~successful_delivery
        * ~successful_plate_return
        * ~successful_dirty_pickup
        * ~successful_wash
    )

    # Compute new ingredient layer
    pile_ingredient = (
        object_is_plate_pile_stocked * DynamicObject.PLATE
        + object_is_ingredient_pile * StaticObject.get_ingredient(interact_item)
    )

    # Ingredient pickup reward. Infinite ingredient piles are easy to farm
    # by repeatedly picking up and dropping ingredients, so only pay for a
    # pile pickup while the current recipe still needs that ingredient in
    # play. Ingredients already on counters, in pots, or in inventories count
    # toward the recipe demand.
    successful_ingredient_pickup = object_is_ingredient_pile * inventory_is_empty
    ingredient_selector_for_pile = pile_ingredient | (pile_ingredient << 1)
    safe_pile_ingredient = jnp.maximum(pile_ingredient, 1)
    ingredients_in_grid = jnp.sum(
        (grid[:, :, 1] & ingredient_selector_for_pile) // safe_pile_ingredient
    )
    ingredients_in_inventories = jnp.sum(
        (all_inventories & ingredient_selector_for_pile) // safe_pile_ingredient
    )
    ingredients_needed = (
        recipe & ingredient_selector_for_pile
    ) // safe_pile_ingredient
    is_ingredient_pickup_useful = (
        ingredients_in_grid + ingredients_in_inventories
    ) < ingredients_needed
    if config.has_prep_stations:
        # Raw prep ingredients never appear in recipes directly - demand
        # comes from their processed form. Count both forms as supply.
        pile_idx = DynamicObject.get_ingredient_type(pile_ingredient)
        is_prep_raw_pile = (pile_idx >= PREP_RAW_START) & (
            pile_idx < PREP_RAW_START + NUM_PREP_CHAINS
        )
        processed_pile = pile_ingredient << PREP_PROCESSED_SHIFT
        processed_pile_selector = processed_pile | (processed_pile << 1)
        safe_processed_pile = jnp.maximum(processed_pile, 1)
        processed_supply = jnp.sum(
            (grid[:, :, 1] & processed_pile_selector) // safe_processed_pile
        ) + jnp.sum(
            (all_inventories & processed_pile_selector) // safe_processed_pile
        )
        chain_needed = (recipe & processed_pile_selector) // safe_processed_pile
        chain_supply = (
            ingredients_in_grid + ingredients_in_inventories + processed_supply
        )
        is_ingredient_pickup_useful = jnp.where(
            is_prep_raw_pile,
            chain_supply < chain_needed,
            is_ingredient_pickup_useful,
        )
    if config.shaped_rewards_enabled:
        shaped_reward += (
            successful_ingredient_pickup
            * is_ingredient_pickup_useful
            * SHAPED_REWARDS["INGREDIENT_PICKUP"]
        )

    new_ingredients = (
        successful_drop * merged_ingredients + no_effect * interact_ingredients
    )
    if config.has_prep_stations:
        # Final chop converts the raw item into its processed form
        new_ingredients = jnp.where(
            chop_completes, station_processed, new_ingredients
        )

    # Start cooking only when the final ingredient is placed.
    pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3
    auto_cook = successful_pot_placement & pot_full_after_drop
    initial_pot_timer = pot_cook_time + config.pot_burn_time

    # Update pot timer
    # Find which pot this is
    def _update_pot_timer(pot_idx):
        pot_y, pot_x = pot_positions[pot_idx]
        is_this_pot = (
            (pot_y == fwd_pos.y) & (pot_x == fwd_pos.x) & pot_active_mask[pot_idx]
        )
        new_timer = jax.lax.select(
            is_this_pot & auto_cook, initial_pot_timer, pot_timers[pot_idx]
        )
        # Reset timer on successful dish pickup
        new_timer = jax.lax.select(
            is_this_pot & successful_dish_pickup, 0, new_timer
        )
        return new_timer

    new_pot_timers = jax.vmap(_update_pot_timer)(jnp.arange(MAX_POTS))

    new_extra = interact_extra  # Keep conveyor directions etc
    if config.has_prep_stations:
        # The extra channel doubles as chop progress / grill timer /
        # blender timer on prep station tiles.
        grill_total_time = config.grill_cook_time + config.grill_burn_time
        new_extra = jnp.where(
            successful_prep_placement,
            jnp.where(object_is_grill, grill_total_time, 0),
            new_extra,
        )
        new_extra = jnp.where(
            successful_chop,
            jnp.where(chop_completes, 0, interact_extra + 1),
            new_extra,
        )
        new_extra = jnp.where(successful_blend_start, config.blend_time, new_extra)
        new_extra = jnp.where(successful_prep_pickup, 0, new_extra)

    new_cell = jnp.array([interact_item, new_ingredients, new_extra])
    new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

    new_inventory = (
        successful_pickup * (pile_ingredient + merged_ingredients)
        + no_effect * inventory
    )

    new_plate_stack = plate_stack
    new_dirty_pile = dirty_pile
    if config.enable_dish_washing:
        dirty_plate = DynamicObject.PLATE | DynamicObject.DIRTY
        new_inventory = jnp.where(successful_dirty_pickup, dirty_plate, new_inventory)
        new_inventory = jnp.where(
            successful_wash, DynamicObject.PLATE, new_inventory
        )
        new_inventory = jnp.where(successful_plate_return, 0, new_inventory)

        # A plate leaves the stack when drawn and returns when stacked back;
        # a delivery converts the served plate into a dirty one.
        drew_plate_from_pile = object_is_plate_pile_stocked & inventory_is_empty
        new_plate_stack = (
            plate_stack
            - drew_plate_from_pile.astype(jnp.int32)
            + successful_plate_return.astype(jnp.int32)
        )
        new_dirty_pile = (
            dirty_pile
            + successful_delivery.astype(jnp.int32)
            - successful_dirty_pickup.astype(jnp.int32)
        )

        if config.shaped_rewards_enabled:
            shaped_reward += (
                successful_dirty_pickup * SHAPED_REWARDS["DIRTY_PLATE_PICKUP"]
            )
            shaped_reward += successful_wash * SHAPED_REWARDS["PLATE_WASH"]

    new_agent = agent.replace(inventory=new_inventory)

    # Reward calculation
    is_correct_recipe = inventory == plated_recipe

    reward = jnp.array(0.0, dtype=float)
    reward += (
        successful_delivery
        * jax.lax.select(is_correct_recipe, 1.0, 0.0)
        * config.delivery_reward
    )

    # Plate pickup reward
    if config.shaped_rewards_enabled:
        inventory_is_plate_now = new_inventory == DynamicObject.PLATE
        successful_plate_pickup = successful_pickup * inventory_is_plate_now
        # Count plates already committed to the task, whether held or
        # dropped on counters. The previous gate only counted inventories,
        # so pickup->drop->pickup from a plate pile could repeatedly earn
        # PLATE_PICKUP while a full pot existed.
        num_plates_in_grid = jnp.sum((grid[:, :, 1] & DynamicObject.PLATE) != 0)
        num_plates_in_inventory = jnp.sum(
            (all_inventories & DynamicObject.PLATE) != 0
        )
        num_plates_in_play = num_plates_in_grid + num_plates_in_inventory
        pot_ingredient_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(
            grid[:, :, 1]
        )
        full_unburned_pots = (
            (grid[:, :, 0] == StaticObject.POT)
            & (pot_ingredient_counts == 3)
            & ((grid[:, :, 1] & DynamicObject.BURNED) == 0)
        )
        num_useful_pots = jnp.sum(full_unburned_pots)
        is_plate_pickup_useful = num_plates_in_play < num_useful_pots
        shaped_reward += (
            is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )
        shaped_reward += (
            any_pot_cooking
            * is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP_DURING_COOKING"]
        )

    correct_delivery = successful_delivery & is_correct_recipe
    event_metrics = jnp.array(
        (
            auto_cook & successful_pot_placement,
            successful_pot_placement,
            successful_pickup,
            successful_counter_drop,
            successful_dish_pickup,
            0.0,  # Filled in after movement with progress toward delivery.
            correct_delivery,
            0.0,  # Filled in after pot timers update if a pot burns.
            successful_prep_placement,
            prep_action,
            successful_prep_pickup,
            0.0,  # Filled in after prep timers update if a grill burns.
            successful_dirty_pickup,
            successful_wash,
            successful_plate_return,
        ),
        dtype=jnp.float32,
    )

    return (
        new_grid,
        new_agent,
        correct_delivery,
        reward,
        shaped_reward,
        event_metrics,
        new_pot_timers,
        new_plate_stack,
        new_dirty_pile,
    )
