"""Functional reset pipeline for Overcooked V3."""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax

from jaxmarl.environments.overcooked_v3.common import (
    Agent,
    Direction,
    Position,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.initialization import (
    randomize_agent_positions,
    randomize_state,
    sample_recipe,
)
from jaxmarl.environments.overcooked_v3.observations import get_obs
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_POTS,
    MAX_PRESSURE_PLATES,
)
from jaxmarl.environments.overcooked_v3.state import State

def reset_overcooked_v3(
    key: chex.PRNGKey, config: OvercookedV3Config
) -> Tuple[Dict[str, chex.Array], State]:
    """Build the initial state and observations for a new episode."""
    layout = config.layout

    static_objects = layout.static_objects
    grid = jnp.stack(
        [
            static_objects,
            jnp.zeros_like(static_objects),
            jnp.zeros_like(static_objects),
        ],
        axis=-1,
        dtype=jnp.int32,
    )

    for _i, (y, x, direction) in enumerate(layout.item_conveyor_info):
        grid = grid.at[y, x, 2].set(direction)
    for _i, (y, x, direction) in enumerate(layout.player_conveyor_info):
        grid = grid.at[y, x, 2].set(direction)

    for y, x, direction, _bounce in layout.moving_wall_info:
        grid = grid.at[y, x, 2].set(direction)

    x_positions, y_positions = map(jnp.array, zip(*layout.agent_positions))
    agents = Agent(
        pos=Position(x=x_positions, y=y_positions),
        dir=jnp.full((config.num_agents,), Direction.UP),
        inventory=jnp.zeros((config.num_agents,), dtype=jnp.int32),
    )

    key, subkey = jax.random.split(key)
    recipe = sample_recipe(subkey, config)

    order_types = jnp.zeros(config.max_orders, dtype=jnp.int32)
    order_expirations = jnp.zeros(config.max_orders, dtype=jnp.int32)
    order_active_mask = jnp.zeros(config.max_orders, dtype=jnp.bool_)
    if config.enable_order_queue and config.order_queue_mode == "alternating":
        first_order_type = jnp.array(1, dtype=jnp.int32)
        order_types = order_types.at[0].set(first_order_type)
        order_expirations = order_expirations.at[0].set(config.order_expiration_time)
        order_active_mask = order_active_mask.at[0].set(True)
        recipe = config.order_recipe_encodings[first_order_type]

    state = State(
        agents=agents,
        grid=grid,
        pot_positions=jnp.array(config.pot_positions),
        pot_cooking_timer=jnp.zeros(MAX_POTS, dtype=jnp.int32),
        pot_cook_durations=jnp.zeros(MAX_POTS, dtype=jnp.int32),
        pot_active_mask=jnp.array(config.pot_active_mask),
        order_types=order_types,
        order_expirations=order_expirations,
        order_active_mask=order_active_mask,
        item_conveyor_positions=jnp.array(config.item_conveyor_positions),
        item_conveyor_directions=jnp.array(config.item_conveyor_directions),
        item_conveyor_active_mask=jnp.array(config.item_conveyor_active_mask),
        player_conveyor_positions=jnp.array(config.player_conveyor_positions),
        player_conveyor_directions=jnp.array(config.player_conveyor_directions),
        player_conveyor_active_mask=jnp.array(config.player_conveyor_active_mask),
        moving_wall_positions=jnp.array(config.moving_wall_positions),
        moving_wall_directions=jnp.array(config.moving_wall_directions),
        moving_wall_active_mask=jnp.array(config.moving_wall_active_mask),
        moving_wall_paused=jnp.array(config.moving_wall_initial_paused),
        moving_wall_bounce=jnp.array(config.moving_wall_bounce),
        button_positions=jnp.array(config.button_positions),
        button_target_idxs=jnp.array(config.button_target_idxs),
        button_target_mask=jnp.array(config.button_target_mask),
        button_action_type=jnp.array(config.button_action_type),
        button_active_mask=jnp.array(config.button_active_mask),
        button_toggled=jnp.zeros(MAX_BUTTONS, dtype=jnp.bool_),
        barrier_positions=jnp.array(config.barrier_positions),
        barrier_active=jnp.array(config.barrier_initial_active),
        barrier_active_mask=jnp.array(config.barrier_active_mask),
        barrier_timer=jnp.zeros(MAX_BARRIERS, dtype=jnp.int32),
        barrier_duration=jnp.array(config.barrier_duration_config),
        pressure_plate_positions=jnp.array(config.pressure_plate_positions),
        pressure_plate_linked_barrier=jnp.array(config.pressure_plate_linked_barrier),
        pressure_plate_action_type=jnp.array(config.pressure_plate_action_type),
        pressure_plate_active_mask=jnp.array(config.pressure_plate_active_mask),
        pressure_plate_toggled=jnp.zeros(MAX_PRESSURE_PLATES, dtype=jnp.bool_),
        time=jnp.array(0),
        terminal=False,
        recipe=recipe,
        new_correct_delivery=False,
        plate_stack_count=jnp.array(
            config.num_plates if config.enable_dish_washing else 0,
            dtype=jnp.int32,
        ),
        dirty_pile_count=jnp.array(0, dtype=jnp.int32),
    )

    key, key_randomize = jax.random.split(key)
    if config.random_reset:
        state = randomize_state(state, key_randomize, config)
    elif config.random_agent_positions:
        state = randomize_agent_positions(state, key_randomize, config)

    obs = get_obs(state, config)

    return lax.stop_gradient(obs), lax.stop_gradient(state)
