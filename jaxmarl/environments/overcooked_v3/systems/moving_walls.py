"""Moving wall systems for Overcooked V3."""

import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    ButtonAction,
    DIR_TO_VEC,
    Direction,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_BARRIERS,
    MAX_BUTTONS,
    MAX_BUTTON_TARGETS,
    MAX_MOVING_WALLS,
)
from jaxmarl.environments.overcooked_v3.state import State

def move_moving_walls(state: State, config: OvercookedV3Config) -> State:
    """Move moving walls one step in their direction, pushing agents if needed."""
    if not config.enable_moving_walls:
        return state

    grid = state.grid

    # Build agent position arrays for collision checking
    agent_xs = state.agents.pos.x  # [num_agents]
    agent_ys = state.agents.pos.y  # [num_agents]

    def _move_single_wall(carry, wall_idx):
        grid, positions, directions, paused, bounce, ag_xs, ag_ys = carry

        y = positions[wall_idx, 0]
        x = positions[wall_idx, 1]
        direction = directions[wall_idx]
        is_active = state.moving_wall_active_mask[wall_idx]
        is_paused = paused[wall_idx]
        should_process = is_active & ~is_paused

        # Direction vector
        dir_vec = DIR_TO_VEC[direction]
        dest_x = x + dir_vec[0]
        dest_y = y + dir_vec[1]

        # Bounds check
        in_bounds = (
            (dest_x >= 0)
            & (dest_x < config.width)
            & (dest_y >= 0)
            & (dest_y < config.height)
        )

        # Safe indices for array access
        safe_dest_x = jnp.clip(dest_x, 0, config.width - 1)
        safe_dest_y = jnp.clip(dest_y, 0, config.height - 1)

        # Check destination cell
        dest_static = grid[safe_dest_y, safe_dest_x, 0]
        dest_is_empty = dest_static == StaticObject.EMPTY

        # Check if an agent is at the destination
        agent_at_dest = (ag_xs == safe_dest_x) & (ag_ys == safe_dest_y)
        any_agent_at_dest = jnp.any(agent_at_dest)

        # If agent at dest, check if we can push them
        # Beyond position = dest + dir_vec
        beyond_x = safe_dest_x + dir_vec[0]
        beyond_y = safe_dest_y + dir_vec[1]
        beyond_in_bounds = (
            (beyond_x >= 0)
            & (beyond_x < config.width)
            & (beyond_y >= 0)
            & (beyond_y < config.height)
        )
        safe_beyond_x = jnp.clip(beyond_x, 0, config.width - 1)
        safe_beyond_y = jnp.clip(beyond_y, 0, config.height - 1)

        beyond_static = grid[safe_beyond_y, safe_beyond_x, 0]

        # Check if beyond position has an active barrier
        is_barrier_tile = beyond_static == StaticObject.BARRIER
        barrier_blocks = False
        for i in range(MAX_BARRIERS):
            at_barrier_pos = (
                (state.barrier_positions[i, 0] == safe_beyond_y)
                & (state.barrier_positions[i, 1] == safe_beyond_x)
                & state.barrier_active_mask[i]
            )
            barrier_blocks = barrier_blocks | (
                at_barrier_pos & state.barrier_active[i]
            )

        beyond_walkable = (
            (beyond_static == StaticObject.EMPTY)
            | (beyond_static == StaticObject.ITEM_CONVEYOR)
            | (beyond_static == StaticObject.PLAYER_CONVEYOR)
            | (beyond_static == StaticObject.PRESSURE_PLATE)
            | (is_barrier_tile & ~barrier_blocks)
        )
        # Also check no other agent at beyond position
        agent_at_beyond = (ag_xs == safe_beyond_x) & (ag_ys == safe_beyond_y)
        no_agent_at_beyond = ~jnp.any(agent_at_beyond)

        can_push_agent = (
            any_agent_at_dest
            & beyond_in_bounds
            & beyond_walkable
            & no_agent_at_beyond
        )

        # Wall can move if dest is empty (no agent) or if we can push the agent
        # Only allow pushing agents on empty tiles to avoid overwriting static objects
        can_move = (
            should_process
            & in_bounds
            & (
                (dest_is_empty & ~any_agent_at_dest)
                | (can_push_agent & (dest_static == StaticObject.EMPTY))
            )
        )

        # Handle bounce: if blocked and bounce enabled, reverse direction
        is_blocked = should_process & ~can_move
        should_bounce = is_blocked & bounce[wall_idx]
        new_direction = jax.lax.select(
            should_bounce,
            Direction.opposite(direction),
            direction,
        )

        # Get item carried by this wall
        old_item = grid[y, x, 1]

        # Clear old position (becomes EMPTY with no item)
        cleared_grid = jax.lax.select(
            can_move,
            grid.at[y, x].set(jnp.array([StaticObject.EMPTY, 0, 0])),
            grid,
        )

        # Set new position with MOVING_WALL + carried item + direction in extra channel
        new_cell = jnp.array([StaticObject.MOVING_WALL, old_item, new_direction])
        moved_grid = jax.lax.select(
            can_move,
            cleared_grid.at[safe_dest_y, safe_dest_x].set(new_cell),
            cleared_grid,
        )

        # Update direction in grid at current position if staying (bounce case)
        final_grid = jax.lax.select(
            ~can_move & should_bounce,
            moved_grid.at[y, x, 2].set(new_direction),
            moved_grid,
        )

        # Update position array
        new_y = jax.lax.select(can_move, safe_dest_y, y)
        new_x = jax.lax.select(can_move, safe_dest_x, x)
        new_positions = positions.at[wall_idx].set(jnp.array([new_y, new_x]))

        # Update direction array
        new_directions = directions.at[wall_idx].set(new_direction)

        # Push agent: update agent positions if we pushed
        # Find which agent was pushed (first match)
        push_agent_idx = jnp.argmax(agent_at_dest)
        new_ag_xs = jax.lax.select(
            can_move & can_push_agent,
            ag_xs.at[push_agent_idx].set(safe_beyond_x),
            ag_xs,
        )
        new_ag_ys = jax.lax.select(
            can_move & can_push_agent,
            ag_ys.at[push_agent_idx].set(safe_beyond_y),
            ag_ys,
        )

        return (
            final_grid,
            new_positions,
            new_directions,
            paused,
            bounce,
            new_ag_xs,
            new_ag_ys,
        ), None

    init_carry = (
        grid,
        state.moving_wall_positions,
        state.moving_wall_directions,
        state.moving_wall_paused,
        state.moving_wall_bounce,
        agent_xs,
        agent_ys,
    )

    (
        (
            new_grid,
            new_positions,
            new_directions,
            new_paused,
            _bounce,
            new_ag_xs,
            new_ag_ys,
        ),
        _,
    ) = jax.lax.scan(_move_single_wall, init_carry, jnp.arange(MAX_MOVING_WALLS))

    # Re-pause walls linked to TRIGGER_MOVE buttons
    def _reapply_trigger_pause(paused, button_idx):
        is_active = state.button_active_mask[button_idx]
        is_trigger = (
            state.button_action_type[button_idx] == ButtonAction.TRIGGER_MOVE
        )

        def _pause_target(paused, target_slot):
            target_idx = state.button_target_idxs[button_idx, target_slot]
            target_enabled = state.button_target_mask[button_idx, target_slot]
            mw_idx = jnp.clip(target_idx, 0, MAX_MOVING_WALLS - 1)
            new_paused = jax.lax.select(
                is_active & is_trigger & target_enabled,
                paused.at[mw_idx].set(True),
                paused,
            )
            return new_paused, None

        paused, _ = jax.lax.scan(
            _pause_target, paused, jnp.arange(MAX_BUTTON_TARGETS)
        )
        return paused, None

    new_paused, _ = jax.lax.scan(
        _reapply_trigger_pause, new_paused, jnp.arange(MAX_BUTTONS)
    )

    # Rebuild agents with updated positions
    new_agents = state.agents.replace(pos=Position(x=new_ag_xs, y=new_ag_ys))

    return state.replace(
        grid=new_grid,
        agents=new_agents,
        moving_wall_positions=new_positions,
        moving_wall_directions=new_directions,
        moving_wall_paused=new_paused,
    )

