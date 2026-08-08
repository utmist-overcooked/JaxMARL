"""Agent interaction rules for Overcooked V3."""

from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Agent,
    ButtonAction,
    Direction,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.initialization import select_recipe_type
from jaxmarl.environments.overcooked_v3.settings import (
    BURN_PENALTY,
    EVENT_NAMES,
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_BUTTON_TARGETS,
    MAX_MOVING_WALLS,
    MAX_POTS,
    SHAPED_REWARDS,
)
from jaxmarl.environments.overcooked_v3.state import State
from jaxmarl.environments.overcooked_v3.systems.barriers import barriers_occupied
from jaxmarl.environments.overcooked_v3.systems.pots import update_pot_timers


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


def apply_agent_button_interactions(
    state: State,
    actions: chex.Array,
    config: OvercookedV3Config,
) -> State:
    """Apply button interactions that affect moving walls and barriers."""
    if not config.enable_buttons:
        return state

    barrier_occupied = barriers_occupied(
        state.agents.pos.y,
        state.agents.pos.x,
        state.barrier_positions,
        state.barrier_active_mask,
    )

    def _process_agent_button(carry, x):
        mw_dirs, mw_paused, mw_bounce, btn_toggled, bar_active, bar_timer = carry
        agent, action = x
        is_interact = action == Actions.interact
        fwd_pos = agent.get_fwd_pos()
        fwd_static = state.grid[fwd_pos.y, fwd_pos.x, 0]
        is_button = fwd_static == StaticObject.BUTTON

        def _scan_buttons(carry):
            (
                mw_dirs,
                mw_paused,
                mw_bounce,
                btn_toggled,
                bar_active,
                bar_timer,
            ) = carry

            def _check_button(carry, button_idx):
                (
                    mw_dirs,
                    mw_paused,
                    mw_bounce,
                    btn_toggled,
                    bar_active,
                    bar_timer,
                ) = carry
                btn_y = state.button_positions[button_idx, 0]
                btn_x = state.button_positions[button_idx, 1]
                is_active = state.button_active_mask[button_idx]
                is_this = (btn_y == fwd_pos.y) & (btn_x == fwd_pos.x) & is_active

                action_type = state.button_action_type[button_idx]

                new_toggled = jax.lax.select(
                    is_this, ~btn_toggled[button_idx], btn_toggled[button_idx]
                )
                btn_toggled = btn_toggled.at[button_idx].set(new_toggled)

                def _apply_target(carry, target_slot):
                    (
                        mw_dirs,
                        mw_paused,
                        mw_bounce,
                        bar_active,
                        bar_timer,
                    ) = carry
                    target_idx = state.button_target_idxs[button_idx, target_slot]
                    target_enabled = state.button_target_mask[button_idx, target_slot]
                    should_apply = is_this & target_enabled
                    mw_idx = jnp.clip(target_idx, 0, MAX_MOVING_WALLS - 1)
                    barrier_idx = jnp.clip(target_idx, 0, MAX_BARRIERS - 1)

                    mw_paused = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TOGGLE_PAUSE),
                        mw_paused.at[mw_idx].set(~mw_paused[mw_idx]),
                        mw_paused,
                    )

                    new_dir = Direction.opposite(mw_dirs[mw_idx])
                    mw_dirs = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TOGGLE_DIRECTION),
                        mw_dirs.at[mw_idx].set(new_dir),
                        mw_dirs,
                    )

                    mw_bounce = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TOGGLE_BOUNCE),
                        mw_bounce.at[mw_idx].set(~mw_bounce[mw_idx]),
                        mw_bounce,
                    )

                    mw_paused = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TRIGGER_MOVE),
                        mw_paused.at[mw_idx].set(False),
                        mw_paused,
                    )

                    toggled_active = ~bar_active[barrier_idx]
                    safe_active = jnp.where(
                        toggled_active & barrier_occupied[barrier_idx],
                        bar_active[barrier_idx],
                        toggled_active,
                    )
                    bar_active = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TOGGLE_BARRIER),
                        bar_active.at[barrier_idx].set(safe_active),
                        bar_active,
                    )

                    bar_active = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TIMED_BARRIER),
                        bar_active.at[barrier_idx].set(False),
                        bar_active,
                    )
                    bar_timer = jax.lax.select(
                        should_apply & (action_type == ButtonAction.TIMED_BARRIER),
                        bar_timer.at[barrier_idx].set(
                            state.barrier_duration[barrier_idx]
                        ),
                        bar_timer,
                    )

                    return (
                        mw_dirs,
                        mw_paused,
                        mw_bounce,
                        bar_active,
                        bar_timer,
                    ), None

                (
                    mw_dirs,
                    mw_paused,
                    mw_bounce,
                    bar_active,
                    bar_timer,
                ), _ = jax.lax.scan(
                    _apply_target,
                    (
                        mw_dirs,
                        mw_paused,
                        mw_bounce,
                        bar_active,
                        bar_timer,
                    ),
                    jnp.arange(MAX_BUTTON_TARGETS),
                )

                return (
                    mw_dirs,
                    mw_paused,
                    mw_bounce,
                    btn_toggled,
                    bar_active,
                    bar_timer,
                ), None

            (
                (
                    mw_dirs,
                    mw_paused,
                    mw_bounce,
                    btn_toggled,
                    bar_active,
                    bar_timer,
                ),
                _,
            ) = jax.lax.scan(
                _check_button,
                (
                    mw_dirs,
                    mw_paused,
                    mw_bounce,
                    btn_toggled,
                    bar_active,
                    bar_timer,
                ),
                jnp.arange(MAX_BUTTONS),
            )
            return (
                mw_dirs,
                mw_paused,
                mw_bounce,
                btn_toggled,
                bar_active,
                bar_timer,
            )

        should_process = is_interact & is_button
        new_carry = jax.lax.cond(
            should_process,
            _scan_buttons,
            lambda c: c,
            (mw_dirs, mw_paused, mw_bounce, btn_toggled, bar_active, bar_timer),
        )

        return new_carry, None

    (
        (
            new_mw_directions,
            new_mw_paused,
            new_mw_bounce,
            new_btn_toggled,
            new_barrier_active,
            new_barrier_timer,
        ),
        _,
    ) = jax.lax.scan(
        _process_agent_button,
        (
            state.moving_wall_directions,
            state.moving_wall_paused,
            state.moving_wall_bounce,
            state.button_toggled,
            state.barrier_active,
            state.barrier_timer,
        ),
        (state.agents, actions),
    )

    return state.replace(
        moving_wall_directions=new_mw_directions,
        moving_wall_paused=new_mw_paused,
        moving_wall_bounce=new_mw_bounce,
        button_toggled=new_btn_toggled,
        barrier_active=new_barrier_active,
        barrier_timer=new_barrier_timer,
    )


def apply_agent_interact_actions(
    key: chex.PRNGKey,
    state: State,
    moved_agents: Agent,
    actions: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[State, float, chex.Array, chex.Array]:
    """Apply interact actions, update carried items, and advance pot timers."""
    num_events = len(EVENT_NAMES)

    def _interact_wrapper(carry, x):
        agent, action = x
        is_interact = action == Actions.interact

        def _interact(carry, agent):
            (
                grid,
                correct_delivery,
                reward,
                pot_timers,
                pot_cook_durations,
                key,
                available_order_mask,
            ) = carry

            key, subkey = jax.random.split(key)
            pot_cook_time = sample_pot_cook_time(subkey, config)

            (
                new_grid,
                new_agent,
                new_correct_delivery,
                interact_reward,
                shaped_reward,
                event_metrics,
                new_pot_timers,
            ) = process_interact(
                grid,
                agent,
                moved_agents.inventory,
                state.recipe,
                pot_timers,
                state.pot_positions,
                state.pot_active_mask,
                config,
                pot_cook_time,
                state.order_types,
                available_order_mask,
            )

            plated_recipe_encodings = (
                config.order_recipe_encodings
                | DynamicObject.PLATE
                | DynamicObject.COOKED
            )
            delivered_recipe_type = jnp.where(
                new_correct_delivery,
                jnp.argmax(agent.inventory == plated_recipe_encodings),
                0,
            ).astype(jnp.int32)

            # Reserve the oldest matching slot immediately so a later agent in
            # this scan cannot fulfill the same order during the same timestep.
            if config.enable_order_queue:
                matching_slots = available_order_mask & (
                    state.order_types == delivered_recipe_type
                )
                matching_slot_idx = jnp.argmax(matching_slots)
                should_reserve_slot = new_correct_delivery & jnp.any(matching_slots)
                available_order_mask = jax.lax.select(
                    should_reserve_slot,
                    available_order_mask.at[matching_slot_idx].set(False),
                    available_order_mask,
                )

            pot_started = (pot_timers == 0) & (new_pot_timers > 0)
            new_pot_cook_durations = jnp.where(
                pot_started, pot_cook_time, pot_cook_durations
            )
            new_pot_cook_durations = jnp.where(
                new_pot_timers == 0, 0, new_pot_cook_durations
            )

            carry = (
                new_grid,
                correct_delivery | new_correct_delivery,
                reward + interact_reward,
                new_pot_timers,
                new_pot_cook_durations,
                key,
                available_order_mask,
            )
            return carry, (
                new_agent,
                shaped_reward,
                event_metrics,
                delivered_recipe_type,
            )

        return jax.lax.cond(
            is_interact,
            _interact,
            lambda c, a: (
                c,
                (
                    a,
                    0.0,
                    jnp.zeros((num_events,), dtype=jnp.float32),
                    jnp.array(0, dtype=jnp.int32),
                ),
            ),
            carry,
            agent,
        )

    carry = (
        state.grid,
        False,
        0.0,
        state.pot_cooking_timer,
        state.pot_cook_durations,
        key,
        state.order_active_mask,
    )
    xs = (moved_agents, actions)
    (
        (
            new_grid,
            new_correct_delivery,
            reward,
            new_pot_timers,
            new_pot_cook_durations,
            recipe_key,
            _available_order_mask,
        ),
        (
            new_agents,
            shaped_rewards,
            event_metrics,
            new_correct_delivery_types,
        ),
    ) = jax.lax.scan(_interact_wrapper, carry, xs)

    shaped_rewards, event_metrics = add_dish_to_goal_progress_shaping(
        state.agents, new_agents, shaped_rewards, event_metrics, config
    )

    new_grid, new_pot_timers, burn_count = update_pot_timers(
        new_grid, new_pot_timers, state.pot_positions, state.pot_active_mask, config
    )
    new_pot_cook_durations = jnp.where(
        new_pot_timers == 0, 0, new_pot_cook_durations
    )

    # Queue mode advances recipes when orders are generated. Without a queue,
    # each successful delivery advances the same fixed/random/alternating stream.
    advance_recipe = new_correct_delivery & (not config.enable_order_queue)

    def _select_next_recipe(key):
        """Select and encode the next queue-off recipe."""
        recipe_type, next_recipe_idx = select_recipe_type(
            key,
            state.next_recipe_idx,
            config,
        )
        return config.order_recipe_encodings[recipe_type], next_recipe_idx

    new_recipe, new_next_recipe_idx = jax.lax.cond(
        advance_recipe,
        _select_next_recipe,
        lambda _: (state.recipe, state.next_recipe_idx),
        recipe_key,
    )

    reward = reward + burn_count * BURN_PENALTY
    burn_events = (
        jnp.zeros((config.num_agents,), dtype=jnp.float32).at[0].set(burn_count)
    )
    event_metrics = event_metrics.at[:, EVENT_NAMES.index("pot_burn")].set(burn_events)

    return (
        state.replace(
            agents=new_agents,
            grid=new_grid,
            pot_cooking_timer=new_pot_timers,
            pot_cook_durations=new_pot_cook_durations,
            recipe=new_recipe,
            next_recipe_idx=new_next_recipe_idx,
            new_correct_delivery=new_correct_delivery,
            new_correct_delivery_types=new_correct_delivery_types,
        ),
        reward,
        shaped_rewards,
        event_metrics,
    )


def add_dish_to_goal_progress_shaping(
    original_agents: Agent,
    new_agents: Agent,
    shaped_rewards: chex.Array,
    event_metrics: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[chex.Array, chex.Array]:
    """Add signed distance-to-delivery shaping for agents carrying plated soup."""
    goal_positions = jnp.asarray(config.goal_positions, dtype=jnp.float32)

    if goal_positions.shape[0] == 0:
        return shaped_rewards, event_metrics

    def _nearest_goal_distance(pos):
        dx = goal_positions[:, 1] - pos.x.astype(jnp.float32)
        dy = goal_positions[:, 0] - pos.y.astype(jnp.float32)
        return jnp.min(jnp.sqrt(dx * dx + dy * dy))

    old_goal_distance = jax.vmap(_nearest_goal_distance)(original_agents.pos)
    new_goal_distance = jax.vmap(_nearest_goal_distance)(new_agents.pos)
    carrying_dish_before = (original_agents.inventory & DynamicObject.COOKED) != 0
    carrying_dish_after = (new_agents.inventory & DynamicObject.COOKED) != 0
    dish_to_goal_progress = (
        carrying_dish_before & carrying_dish_after & config.shaped_rewards_enabled
    ) * (old_goal_distance - new_goal_distance)

    shaped_rewards = shaped_rewards + (
        dish_to_goal_progress * SHAPED_REWARDS["DISH_TO_GOAL_PROGRESS"]
    )
    event_metrics = event_metrics.at[
        :, EVENT_NAMES.index("dish_to_goal_progress")
    ].set(dish_to_goal_progress)

    return shaped_rewards, event_metrics


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
    order_types: Optional[chex.Array] = None,
    order_active_mask: Optional[chex.Array] = None,
):
    """Process an interact action for an agent."""
    if pot_cook_time is None:
        pot_cook_time = jnp.array(config.pot_cook_time, dtype=jnp.int32)
    if order_types is None:
        order_types = jnp.zeros(config.max_orders, dtype=jnp.int32)
    if order_active_mask is None:
        order_active_mask = jnp.zeros(config.max_orders, dtype=jnp.bool_)

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

    # Delivery
    successful_delivery = object_is_goal * inventory_is_dish
    no_effect = ~successful_pickup * ~successful_drop * ~successful_delivery

    # Compute new ingredient layer
    pile_ingredient = (
        object_is_plate_pile * DynamicObject.PLATE
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
    if config.shaped_rewards_enabled:
        shaped_reward += (
            successful_ingredient_pickup
            * is_ingredient_pickup_useful
            * SHAPED_REWARDS["INGREDIENT_PICKUP"]
        )

    new_ingredients = (
        successful_drop * merged_ingredients + no_effect * interact_ingredients
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

    new_cell = jnp.array([interact_item, new_ingredients, new_extra])
    new_grid = grid.at[fwd_pos.y, fwd_pos.x].set(new_cell)

    new_inventory = (
        successful_pickup * (pile_ingredient + merged_ingredients)
        + no_effect * inventory
    )
    new_agent = agent.replace(inventory=new_inventory)

    # Queue deliveries may fulfill any active recipe. Queue-off environments
    # continue to compare against their single current recipe.
    if config.enable_order_queue:
        safe_order_types = jnp.clip(
            order_types,
            0,
            config.order_recipe_encodings.shape[0] - 1,
        )
        queued_plated_recipes = (
            config.order_recipe_encodings[safe_order_types]
            | DynamicObject.PLATE
            | DynamicObject.COOKED
        )
        is_correct_recipe = jnp.any(
            order_active_mask & (inventory == queued_plated_recipes)
        )
    else:
        is_correct_recipe = inventory == plated_recipe

    # Reward calculation

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
    )
