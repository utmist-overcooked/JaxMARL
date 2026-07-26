"""Conveyor belt systems for Overcooked V3."""

import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.agent_step import is_agent_walkable
from jaxmarl.environments.overcooked_v3.common import DIR_TO_VEC, StaticObject
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_ITEM_CONVEYORS,
    MAX_PLAYER_CONVEYORS,
)
from jaxmarl.environments.overcooked_v3.state import State
from jaxmarl.environments.overcooked_v3.utils import tree_select

def move_items_on_item_conveyors(state: State, config: OvercookedV3Config) -> State:
    """Move items on item conveyor belts."""
    if not config.enable_item_conveyors:
        return state

    grid = state.grid

    def _move_item_on_conveyor(grid, conveyor_idx):
        pos = state.item_conveyor_positions[conveyor_idx]
        direction = state.item_conveyor_directions[conveyor_idx]
        is_active = state.item_conveyor_active_mask[conveyor_idx]

        y, x = pos[0], pos[1]
        current_item = grid[y, x, 1]
        has_item = current_item != 0

        # Calculate destination
        dir_vec = DIR_TO_VEC[direction]

        raw_dest_x = x + dir_vec[0]
        raw_dest_y = y + dir_vec[1]
        dest_in_bounds = (
            (raw_dest_x >= 0)
            & (raw_dest_x < config.width)
            & (raw_dest_y >= 0)
            & (raw_dest_y < config.height)
        )
        dest_x = jnp.clip(raw_dest_x, 0, config.width - 1)
        dest_y = jnp.clip(raw_dest_y, 0, config.height - 1)

        # Check if destination can receive item
        dest_static = grid[dest_y, dest_x, 0]
        dest_item = grid[dest_y, dest_x, 1]
        dest_can_receive = (
            dest_in_bounds
            & (
                (dest_static == StaticObject.WALL)
                | (dest_static == StaticObject.ITEM_CONVEYOR)
                | (dest_static == StaticObject.PLAYER_CONVEYOR)
                | (dest_static == StaticObject.GOAL)
                | (dest_static == StaticObject.MOVING_WALL)
            )
            & (dest_item == 0)
        )

        should_move = is_active & has_item & dest_can_receive
        should_disappear = is_active & has_item & ~dest_in_bounds

        # Move item
        new_grid = jax.lax.select(
            should_disappear,
            grid.at[y, x, 1].set(0),
            jax.lax.select(
                should_move,
                grid.at[y, x, 1].set(0).at[dest_y, dest_x, 1].set(current_item),
                grid,
            )
        )

        return new_grid, None

    new_grid, _ = jax.lax.scan(
        _move_item_on_conveyor, grid, jnp.arange(MAX_ITEM_CONVEYORS)
    )

    return state.replace(grid=new_grid)

def push_players_on_player_conveyors(state: State, config: OvercookedV3Config) -> State:
    """Push agents on player conveyor belts."""
    if not config.enable_player_conveyors:
        return state

    agents = state.agents
    grid = state.grid

    def _check_agent_on_conveyor(agent_pos, conveyor_idx):
        pos = state.player_conveyor_positions[conveyor_idx]
        is_active = state.player_conveyor_active_mask[conveyor_idx]
        is_on = (agent_pos.x == pos[1]) & (agent_pos.y == pos[0]) & is_active
        return is_on, state.player_conveyor_directions[conveyor_idx]

    def _push_agent(agent):
        # Check all conveyors
        on_conveyor_checks = jax.vmap(
            lambda idx: _check_agent_on_conveyor(agent.pos, idx)
        )(jnp.arange(MAX_PLAYER_CONVEYORS))

        is_on_any, directions = on_conveyor_checks
        # Take first active conveyor's direction
        conveyor_idx = jnp.argmax(is_on_any)
        is_on = jnp.any(is_on_any)
        push_direction = directions[conveyor_idx]

        # Calculate new position
        new_pos = agent.pos.move_in_bounds(push_direction, config.width, config.height)

        # Check if destination is walkable
        dest_static = grid[new_pos.y, new_pos.x, 0]
        dest_walkable = is_agent_walkable(dest_static, new_pos, state)

        should_push = is_on & dest_walkable

        final_pos = tree_select(should_push, new_pos, agent.pos)
        return agent.replace(pos=final_pos)

    new_agents = jax.vmap(_push_agent)(agents)

    return state.replace(agents=new_agents)
