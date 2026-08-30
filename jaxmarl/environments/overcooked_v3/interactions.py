"""Agent interaction rules for Overcooked V3."""

from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Agent,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import (
    BURN_PENALTY,
    MAX_POTS,
    ORDER_EXPIRED_PENALTY,
    REWARD_COMPONENT_KEYS,
    SHAPED_REWARDS,
)


def zero_reward_breakdown() -> Dict[str, chex.Array]:
    return {key: jnp.array(0.0, dtype=jnp.float32) for key in REWARD_COMPONENT_KEYS}


def compute_burn_penalty(
    pre_timers: chex.Array,
    post_timers: chex.Array,
    pot_active_mask: chex.Array,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Penalize pots that just burned (timer hit 0 while actively cooking)."""
    just_burned = (pre_timers > 0) & (post_timers == 0) & pot_active_mask
    penalty = jnp.sum(just_burned).astype(jnp.float32) * BURN_PENALTY
    breakdown = zero_reward_breakdown()
    breakdown["BURN_PENALTY"] = penalty
    return penalty, breakdown


def compute_order_expired_penalty(
    expired_mask: chex.Array,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Penalize orders that expired unfulfilled."""
    penalty = jnp.sum(expired_mask).astype(jnp.float32) * ORDER_EXPIRED_PENALTY
    breakdown = zero_reward_breakdown()
    breakdown["ORDER_EXPIRED_PENALTY"] = penalty
    return penalty, breakdown


def merge_reward_breakdowns(*breakdowns: Dict[str, chex.Array]) -> Dict[str, chex.Array]:
    """Sum any number of REWARD_COMPONENT_KEYS breakdown dicts together.

    Values may be per-agent arrays or scalars per key; scalar operands
    broadcast naturally against per-agent arrays under `+`.
    """
    merged = zero_reward_breakdown()
    for breakdown in breakdowns:
        merged = {key: merged[key] + breakdown[key] for key in REWARD_COMPONENT_KEYS}
    return merged


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
    reward_breakdown = zero_reward_breakdown()

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
        reward_breakdown["SOUP_IN_DISH"] = (
            successful_dish_pickup
            * is_dish_pickup_useful
            * SHAPED_REWARDS["SOUP_IN_DISH"]
        )
        shaped_reward += reward_breakdown["SOUP_IN_DISH"]

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
        reward_breakdown["PLACEMENT_IN_POT"] = (
            successful_pot_placement
            * is_pot_placement_useful
            * SHAPED_REWARDS["PLACEMENT_IN_POT"]
        )
        shaped_reward += reward_breakdown["PLACEMENT_IN_POT"]

    # Drop on counter/conveyor
    successful_drop = (
        object_is_wall | object_is_conveyor
    ) * object_has_no_ingredients * ~inventory_is_empty + successful_pot_placement

    # Drop on counter/conveyor
    successful_drop = (
        (object_is_wall | object_is_conveyor) * object_has_no_ingredients * ~inventory_is_empty
        + successful_pot_placement
    )
    successful_counter_drop = (
        (object_is_wall | object_is_conveyor)
        & object_has_no_ingredients
        & ~inventory_is_empty
        & ~inventory_is_dish
    )
    successful_counter_pickup = (
        (object_is_wall | object_is_conveyor)
        & ~object_has_no_ingredients
        & inventory_is_empty
    )
    above_y = jnp.maximum(fwd_pos.y - 1, 0)
    below_y = jnp.minimum(fwd_pos.y + 1, config.height - 1)
    above_static = grid[above_y, fwd_pos.x, 0]
    below_static = grid[below_y, fwd_pos.x, 0]
    above_walkable = (above_static == StaticObject.EMPTY) | (
        above_static == StaticObject.PLAYER_CONVEYOR
    )
    below_walkable = (below_static == StaticObject.EMPTY) | (
        below_static == StaticObject.PLAYER_CONVEYOR
    )
    is_handoff_counter = (
        (object_is_wall | object_is_conveyor)
        & above_walkable
        & below_walkable
    )
    min_pot_y = jnp.min(jnp.where(pot_active_mask, pot_positions[:, 0], config.height))
    agent_side = agent.pos.y - fwd_pos.y
    pot_side = min_pot_y - fwd_pos.y
    drop_toward_pot_side = (agent_side * pot_side) < 0
    pickup_on_pot_side = (agent_side * pot_side) > 0
    pot_ingredient_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(
        grid[:, :, 1]
    )
    full_unburned_pots = (
        (grid[:, :, 0] == StaticObject.POT)
        & (pot_ingredient_counts == 3)
        & ((grid[:, :, 1] & DynamicObject.BURNED) == 0)
    )
    has_plate_target = jnp.any(full_unburned_pots)
    counter_item_is_ingredient = DynamicObject.is_ingredient(interact_ingredients)
    counter_item_is_plate = interact_ingredients == DynamicObject.PLATE
    useful_drop_inventory = (
        (inventory_is_ingredient & ((recipe & inventory) != 0))
        | (inventory_is_plate & has_plate_target)
    )
    useful_pickup_item = (
        (counter_item_is_ingredient & ((recipe & interact_ingredients) != 0))
        | (counter_item_is_plate & has_plate_target)
    )
    if config.shaped_rewards_enabled:
        reward_breakdown["HANDOFF_DROP"] = (
            is_handoff_counter
            & drop_toward_pot_side
            & successful_counter_drop
            & useful_drop_inventory
        ) * SHAPED_REWARDS["HANDOFF_DROP"]
        reward_breakdown["HANDOFF_PICKUP"] = (
            is_handoff_counter
            & pickup_on_pot_side
            & successful_counter_pickup
            & useful_pickup_item
        ) * SHAPED_REWARDS["HANDOFF_PICKUP"]
        shaped_reward += (
            reward_breakdown["HANDOFF_DROP"] + reward_breakdown["HANDOFF_PICKUP"]
        )

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

    # Start cooking when pot becomes full. The pot cooks whatever it is given
    # -- that stays true regardless of the recipe -- but the REWARD is gated on
    # the contents actually matching the current recipe. Without that gate an
    # agent collects POT_START_COOKING for filling a pot with the wrong
    # ingredients, which pays just as well as cooking the right thing and so
    # removes any pressure to act on a communicated/observed recipe.
    # (PLACEMENT_IN_POT is already gated this way via is_pot_placement_useful.)
    pot_full_after_drop = DynamicObject.ingredient_count(new_ingredients) == 3
    auto_cook = pot_is_idle & pot_full_after_drop
    cooks_current_recipe = new_ingredients == recipe
    if config.shaped_rewards_enabled:
        reward_breakdown["POT_START_COOKING"] = (
            auto_cook * cooks_current_recipe * SHAPED_REWARDS["POT_START_COOKING"]
        )
        shaped_reward += reward_breakdown["POT_START_COOKING"]
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
    reward_breakdown["DELIVERY"] = (
        successful_delivery
        * jax.lax.select(is_correct_recipe, 1.0, 0.0)
        * config.delivery_reward
    )
    reward += reward_breakdown["DELIVERY"]

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
        reward_breakdown["PLATE_PICKUP"] = (
            is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )
        shaped_reward += reward_breakdown["PLATE_PICKUP"]

    correct_delivery = successful_delivery & is_correct_recipe

    return (
        new_grid,
        new_agent,
        correct_delivery,
        reward,
        shaped_reward,
        new_pot_timers,
        reward_breakdown,
    )


def task_target_mask(
    grid: chex.Array,
    recipe: int,
    agent: Agent,
    config: OvercookedV3Config,
) -> chex.Array:
    """Return the current useful object targets for one agent's subtask."""
    static_objects = grid[:, :, 0]
    dynamic_objects = grid[:, :, 1]
    height, width = config.height, config.width
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")

    pot_mask = static_objects == StaticObject.POT
    plate_mask = static_objects == StaticObject.PLATE_PILE
    goal_mask = static_objects == StaticObject.GOAL
    counter_mask = (
        (static_objects == StaticObject.WALL)
        | (static_objects == StaticObject.MOVING_WALL)
        | (static_objects == StaticObject.ITEM_CONVEYOR)
        | (static_objects == StaticObject.PLAYER_CONVEYOR)
    )
    walkable_mask = (static_objects == StaticObject.EMPTY) | (
        static_objects == StaticObject.PLAYER_CONVEYOR
    )
    above_walkable = jnp.concatenate(
        [jnp.zeros((1, width), dtype=bool), walkable_mask[:-1, :]],
        axis=0,
    )
    below_walkable = jnp.concatenate(
        [walkable_mask[1:, :], jnp.zeros((1, width), dtype=bool)],
        axis=0,
    )
    handoff_counter_mask = counter_mask & above_walkable & below_walkable
    empty_handoff_mask = handoff_counter_mask & (
        dynamic_objects == DynamicObject.EMPTY
    )

    ingredient_pile_mask = StaticObject.is_ingredient_pile(static_objects)
    pile_idx = jnp.maximum(static_objects - StaticObject.INGREDIENT_PILE_BASE, 0)
    pile_ingredient = jnp.left_shift(DynamicObject.BASE_INGREDIENT, 2 * pile_idx)
    useful_ingredient_mask = ingredient_pile_mask & ((recipe & pile_ingredient) != 0)

    ingredient_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(
        dynamic_objects
    )
    pot_burned = (dynamic_objects & DynamicObject.BURNED) != 0
    pot_cooked = (dynamic_objects & DynamicObject.COOKED) != 0
    pot_needs_ingredient = pot_mask & (ingredient_counts < 3) & ~pot_cooked & ~pot_burned
    pot_full_uncooked = pot_mask & (ingredient_counts == 3) & ~pot_cooked & ~pot_burned
    pot_ready = pot_mask & pot_cooked & ~pot_burned

    has_ready_pot = jnp.any(pot_ready)
    has_busy_pot = jnp.any(pot_full_uncooked)
    plate_should_be_collected = has_ready_pot | has_busy_pot
    handoff_item_is_ingredient = DynamicObject.is_ingredient(dynamic_objects)
    handoff_item_is_plate = dynamic_objects == DynamicObject.PLATE
    useful_handoff_pickup_mask = handoff_counter_mask & (
        handoff_item_is_ingredient | handoff_item_is_plate
    )
    min_pot_y = jnp.min(jnp.where(pot_mask, yy, height))

    inv = agent.inventory
    inventory_is_empty = inv == DynamicObject.EMPTY
    inventory_is_ingredient = DynamicObject.is_ingredient(inv)
    inventory_is_plate = inv == DynamicObject.PLATE
    inventory_is_dish = (inv & DynamicObject.COOKED) != 0

    source_target = jnp.where(
        plate_should_be_collected, plate_mask, useful_ingredient_mask
    )
    agent_side = agent.pos.y - yy
    pot_side = min_pot_y - yy
    pickup_on_pot_side = (agent_side * pot_side) > 0
    pickup_handoff_target = useful_handoff_pickup_mask & pickup_on_pot_side
    wait_handoff_target = empty_handoff_mask & pickup_on_pot_side
    empty_target = jnp.where(
        jnp.any(pickup_handoff_target), pickup_handoff_target, source_target
    )
    empty_target = jnp.where(
        jnp.any(wait_handoff_target) & ~jnp.any(pickup_handoff_target),
        wait_handoff_target,
        empty_target,
    )

    plate_target = jnp.where(has_ready_pot, pot_ready, pot_full_uncooked)
    target = jnp.where(inventory_is_empty, empty_target, source_target)
    target = jnp.where(inventory_is_ingredient, pot_needs_ingredient, target)
    target = jnp.where(inventory_is_plate, plate_target, target)
    target = jnp.where(inventory_is_dish, goal_mask, target)
    drop_from_far_side = (agent_side * pot_side) < 0
    drop_handoff_target = empty_handoff_mask & drop_from_far_side
    target = jnp.where(
        inventory_is_ingredient & jnp.any(drop_handoff_target),
        drop_handoff_target,
        target,
    )
    target = jnp.where(
        inventory_is_plate & jnp.any(drop_handoff_target),
        drop_handoff_target,
        target,
    )
    return target


def dense_task_shaping(
    grid: chex.Array,
    recipe: int,
    old_agents: Agent,
    new_agents: Agent,
    actions: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Dense potential shaping toward each agent's current useful task target.

    Scores the policy's already-resolved movement against the Overcooked
    subtask implied by that agent's inventory; never chooses or overwrites
    an action. Returns the per-agent total plus a per-component breakdown
    (TASK_PROGRESS/TASK_FACING/INVALID_MOVE; all other REWARD_COMPONENT_KEYS
    are zero here since this function never touches interaction rewards).
    """
    height, width = config.height, config.width
    yy, xx = jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij")

    def _min_dist(pos: Position, target_mask: chex.Array):
        dist = jnp.abs(yy - pos.y) + jnp.abs(xx - pos.x)
        return jnp.min(jnp.where(target_mask, dist, 1_000_000))

    def _agent_reward(old_agent: Agent, new_agent: Agent, action):
        target_mask = task_target_mask(grid, recipe, old_agent, config)
        target_valid = jnp.any(target_mask)
        old_dist = _min_dist(old_agent.pos, target_mask)
        new_dist = _min_dist(new_agent.pos, target_mask)

        progress = jnp.clip(old_dist - new_dist, -1.0, 1.0)
        progress_reward = target_valid * progress * SHAPED_REWARDS["TASK_PROGRESS"]

        is_movement = action < Actions.stay
        same_position = (old_agent.pos.x == new_agent.pos.x) & (
            old_agent.pos.y == new_agent.pos.y
        )
        invalid_move = is_movement & same_position
        invalid_move_reward = invalid_move * SHAPED_REWARDS["INVALID_MOVE"]

        fwd_pos = new_agent.get_fwd_pos()
        fwd_x = jnp.clip(fwd_pos.x, 0, width - 1)
        fwd_y = jnp.clip(fwd_pos.y, 0, height - 1)
        facing_target = target_mask[fwd_y, fwd_x]
        # Gated on actually having moved, not merely on requesting a movement
        # action: a blocked move still turns the agent, so rewarding intent
        # alone lets an agent stand beside a target and farm TASK_FACING
        # forever without ever interacting.
        facing_reward = (
            is_movement
            & ~same_position
            & target_valid
            & facing_target
        ) * SHAPED_REWARDS["TASK_FACING"]

        reward_breakdown = zero_reward_breakdown()
        reward_breakdown["TASK_PROGRESS"] = progress_reward
        reward_breakdown["TASK_FACING"] = facing_reward
        reward_breakdown["INVALID_MOVE"] = invalid_move_reward
        total = progress_reward + invalid_move_reward + facing_reward
        return total, reward_breakdown

    return jax.vmap(_agent_reward)(old_agents, new_agents, actions)
