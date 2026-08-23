"""Agent interaction rules for Overcooked V3."""

from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Agent,
    ButtonAction,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
    NUM_PREP_CHAINS,
    PREP_PROCESSED_SHIFT,
    PREP_RAW_START,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import (
    BURN_PENALTY,
    EVENT_NAMES,
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_BUTTON_TARGETS,
    MAX_MOVING_WALLS,
    MAX_POTS,
    ORDER_EXPIRED_PENALTY,
    REWARD_COMPONENT_KEYS,
    SHAPED_REWARDS,
)
from jaxmarl.environments.overcooked_v3.state import State
from jaxmarl.environments.overcooked_v3.systems.barriers import barriers_occupied
from jaxmarl.environments.overcooked_v3.systems.pots import update_pot_timers
from jaxmarl.environments.overcooked_v3.systems.prep_stations import (
    update_prep_stations,
)


def zero_reward_breakdown() -> Dict[str, chex.Array]:
    """Return an all-zero REWARD_COMPONENT_KEYS breakdown dict."""
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
) -> Tuple[State, float, chex.Array, Dict[str, chex.Array], chex.Array]:
    """Apply interact actions, update carried items, and advance pot timers.

    Threads both diagnostic channels out of the scan: the REWARD_COMPONENT_KEYS
    reward_breakdown (per-agent shaped-reward itemization) and the
    (num_agents, len(EVENT_NAMES)) event_metrics counters.
    """
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
                plate_stack,
                dirty_pile,
                key,
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
                interact_breakdown,
                new_plate_stack,
                new_dirty_pile,
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
                plate_stack,
                dirty_pile,
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
                new_plate_stack,
                new_dirty_pile,
                key,
            )
            return carry, (new_agent, shaped_reward, interact_breakdown, event_metrics)

        return jax.lax.cond(
            is_interact,
            _interact,
            lambda c, a: (
                c,
                (
                    a,
                    0.0,
                    zero_reward_breakdown(),
                    jnp.zeros((num_events,), dtype=jnp.float32),
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
        state.plate_stack_count,
        state.dirty_pile_count,
        key,
    )
    xs = (moved_agents, actions)
    (
        (
            new_grid,
            new_correct_delivery,
            reward,
            new_pot_timers,
            new_pot_cook_durations,
            new_plate_stack,
            new_dirty_pile,
            _key,
        ),
        (new_agents, shaped_rewards, reward_breakdown, event_metrics),
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

    burn_penalty = burn_count * BURN_PENALTY
    reward = reward + burn_penalty
    reward_breakdown = merge_reward_breakdowns(
        reward_breakdown, {**zero_reward_breakdown(), "BURN_PENALTY": burn_penalty}
    )
    burn_events = (
        jnp.zeros((config.num_agents,), dtype=jnp.float32).at[0].set(burn_count)
    )
    event_metrics = event_metrics.at[:, EVENT_NAMES.index("pot_burn")].set(burn_events)

    # Advance prep station timers (grill cooking/burning, blender mixing)
    if config.has_prep_stations:
        new_grid, prep_burn_count = update_prep_stations(new_grid, config)
        prep_burn_penalty = prep_burn_count * BURN_PENALTY
        reward = reward + prep_burn_penalty
        reward_breakdown = merge_reward_breakdowns(
            reward_breakdown,
            {**zero_reward_breakdown(), "BURN_PENALTY": prep_burn_penalty},
        )
        prep_burn_events = (
            jnp.zeros((config.num_agents,), dtype=jnp.float32)
            .at[0]
            .set(prep_burn_count)
        )
        event_metrics = event_metrics.at[
            :, EVENT_NAMES.index("prep_burn")
        ].set(prep_burn_events)

    return (
        state.replace(
            agents=new_agents,
            grid=new_grid,
            pot_cooking_timer=new_pot_timers,
            pot_cook_durations=new_pot_cook_durations,
            new_correct_delivery=new_correct_delivery,
            plate_stack_count=new_plate_stack,
            dirty_pile_count=new_dirty_pile,
        ),
        reward,
        shaped_rewards,
        reward_breakdown,
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

    # Handoff shaping predicates. These are deliberately narrower than
    # successful_counter_drop above: a plated dish put back on a counter is not
    # a useful handoff, so dishes are excluded here but still count as a drop.
    handoff_drop = (
        (object_is_wall | object_is_conveyor)
        & object_has_no_ingredients
        & ~inventory_is_empty
        & ~inventory_is_dish
    )
    handoff_pickup = (
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
            & handoff_drop
            & useful_drop_inventory
        ) * SHAPED_REWARDS["HANDOFF_DROP"]
        reward_breakdown["HANDOFF_PICKUP"] = (
            is_handoff_counter
            & pickup_on_pot_side
            & handoff_pickup
            & useful_pickup_item
        ) * SHAPED_REWARDS["HANDOFF_PICKUP"]
        shaped_reward += (
            reward_breakdown["HANDOFF_DROP"] + reward_breakdown["HANDOFF_PICKUP"]
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
    if config.shaped_rewards_enabled:
        reward_breakdown["POT_START_COOKING"] = (
            auto_cook * SHAPED_REWARDS["POT_START_COOKING"]
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
        reward_breakdown["PLATE_PICKUP"] = (
            is_plate_pickup_useful
            * successful_plate_pickup
            * SHAPED_REWARDS["PLATE_PICKUP"]
        )
        shaped_reward += reward_breakdown["PLATE_PICKUP"]
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
        reward_breakdown,
        new_plate_stack,
        new_dirty_pile,
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
        facing_reward = (
            is_movement * target_valid * facing_target * SHAPED_REWARDS["TASK_FACING"]
        )

        reward_breakdown = zero_reward_breakdown()
        reward_breakdown["TASK_PROGRESS"] = progress_reward
        reward_breakdown["TASK_FACING"] = facing_reward
        reward_breakdown["INVALID_MOVE"] = invalid_move_reward
        total = progress_reward + invalid_move_reward + facing_reward
        return total, reward_breakdown

    return jax.vmap(_agent_reward)(old_agents, new_agents, actions)
