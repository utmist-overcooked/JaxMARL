"""Agent interaction rules for Overcooked V3."""

from typing import Optional

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import DynamicObject, StaticObject, Agent
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
    object_is_ingredient_pile = (
        fwd_pos_in_bounds & StaticObject.is_ingredient_pile(interact_item)
    )
    object_is_pile = object_is_plate_pile | object_is_ingredient_pile

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
    pot_is_idle = object_is_pot * (current_pot_timer == 0) * ~pot_is_cooked

    # Check if pot is ready (in burning window)
    # In V3: dish_ready when cooking_timer is between 1 and burn_time
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
    successful_drop = (
        object_is_wall | object_is_conveyor
    ) * object_has_no_ingredients * ~inventory_is_empty + successful_pot_placement

    # Delivery
    successful_delivery = object_is_goal * inventory_is_dish
    no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

    # Compute new ingredient layer
    pile_ingredient = (
        object_is_plate_pile * DynamicObject.PLATE
        + object_is_ingredient_pile * StaticObject.get_ingredient(interact_item)
    )

    new_ingredients = (
        successful_drop * merged_ingredients + no_effect * interact_ingredients
    )

    # Start cooking when pot becomes full
    pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3
    auto_cook = pot_is_idle & pot_full_after_drop
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

    new_cell = jnp.array([interact_item, new_ingredients, new_extra])
    new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

    new_inventory = (
        successful_pickup * (pile_ingredient + merged_ingredients)
        + no_effect * inventory
    )
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
        num_plates_in_inventory = jnp.sum(all_inventories == DynamicObject.PLATE)
        pot_ingredient_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(
            grid[:, :, 1]
        )
        full_unburned_pots = (
            (grid[:, :, 0] == StaticObject.POT)
            & (pot_ingredient_counts == 3)
            & ((grid[:, :, 1] & DynamicObject.BURNED) == 0)
        )
        num_useful_pots = jnp.sum(full_unburned_pots)
        is_plate_pickup_useful = num_plates_in_inventory < num_useful_pots
        shaped_reward += (
            is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )

    correct_delivery = successful_delivery & is_correct_recipe

    return (
        new_grid,
        new_agent,
        correct_delivery,
        reward,
        shaped_reward,
        new_pot_timers,
    )
