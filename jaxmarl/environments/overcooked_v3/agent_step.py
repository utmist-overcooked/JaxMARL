"""Functional agent movement and action phase logic for Overcooked V3."""

from typing import Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    ACTION_TO_DIRECTION,
    Actions,
    Agent,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.interactions import (
    apply_agent_button_interactions,
    process_interact,
    sample_pot_cook_time,
)
from jaxmarl.environments.overcooked_v3.settings import (
    MAX_BARRIERS,
    MAX_PRESSURE_PLATES,
)
from jaxmarl.environments.overcooked_v3.state import State
from jaxmarl.environments.overcooked_v3.systems.barriers import (
    barriers_occupied as barriers_occupied,
)
from jaxmarl.environments.overcooked_v3.systems.pots import update_pot_timers
from jaxmarl.environments.overcooked_v3.utils import tree_select

def is_agent_walkable(static_object, pos, state: State) -> chex.Array:
    """Return whether an agent can stand on a static object at a position."""
    is_barrier_tile = static_object == StaticObject.BARRIER
    at_barrier_pos = (
        (state.barrier_positions[:, 0] == pos.y)
        & (state.barrier_positions[:, 1] == pos.x)
        & state.barrier_active_mask[:]
    )
    barrier_blocks = jnp.any(at_barrier_pos & state.barrier_active)

    return (
        (static_object == StaticObject.EMPTY)
        | (static_object == StaticObject.PLAYER_CONVEYOR)
        | (static_object == StaticObject.PRESSURE_PLATE)
        | (is_barrier_tile & ~barrier_blocks)
    )

def run_agent_action_phase(
    key: chex.PRNGKey,
    state: State,
    actions: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[State, float, chex.Array]:
    """Run movement, collision handling, interactions, and button effects."""
    barrier_walkable_by_pressure_plate = (
        find_barriers_opened_by_current_pressure_plate_occupants(state, config)
    )
    moved_agents = move_agents_to_requested_positions(
        state, actions, barrier_walkable_by_pressure_plate, config
    )
    moved_agents = resolve_agent_destination_collisions(
        state.agents, moved_agents, config
    )
    moved_agents = prevent_agents_from_swapping_positions(state.agents, moved_agents)

    state, reward, shaped_rewards = apply_agent_interact_actions(
        key, state, moved_agents, actions, config
    )
    state = apply_agent_button_interactions(state, actions, config)

    return state, reward, shaped_rewards

def find_barriers_opened_by_current_pressure_plate_occupants(
    state: State, config: OvercookedV3Config
) -> chex.Array:
    """Return barriers that are walkable because linked pressure plates are pressed."""
    barrier_walkable_by_pressure_plate = jnp.zeros(MAX_BARRIERS, dtype=jnp.bool_)

    if config.enable_pressure_plates:

        def _check_pressure_plate(barrier_walkable, plate_idx):
            plate_valid = state.pressure_plate_active_mask[plate_idx]
            linked_barrier_mask = state.pressure_plate_linked_barrier[plate_idx]

            def _agent_on_plate(agent_pos):
                return (
                    (agent_pos.y == state.pressure_plate_positions[plate_idx, 0])
                    & (agent_pos.x == state.pressure_plate_positions[plate_idx, 1])
                )

            agent_on_plate = jax.vmap(_agent_on_plate)(state.agents.pos)
            plate_pressed = plate_valid & jnp.any(agent_on_plate)
            updated_barrier_walkable = barrier_walkable | (
                linked_barrier_mask & plate_pressed
            )
            return updated_barrier_walkable, None

        barrier_walkable_by_pressure_plate, _ = jax.lax.scan(
            _check_pressure_plate,
            barrier_walkable_by_pressure_plate,
            jnp.arange(MAX_PRESSURE_PLATES),
        )

    return barrier_walkable_by_pressure_plate

def move_agents_to_requested_positions(
    state: State,
    actions: chex.Array,
    barrier_walkable_by_pressure_plate: chex.Array,
    config: OvercookedV3Config,
) -> Agent:
    """Apply each movement action before resolving agent-agent conflicts."""
    grid = state.grid

    def _move_wrapper(agent, action):
        direction = ACTION_TO_DIRECTION[action]

        def _move(agent, dir):
            pos = agent.pos
            new_pos = pos.move_in_bounds(dir, config.width, config.height)
            new_cell_static = grid[new_pos.y, new_pos.x, 0]

            is_barrier_tile = new_cell_static == StaticObject.BARRIER
            at_barrier_pos = (
                (state.barrier_positions[:, 0] == new_pos.y)
                & (state.barrier_positions[:, 1] == new_pos.x)
                & state.barrier_active_mask
            )
            barrier_blocks = jnp.any(
                at_barrier_pos
                & state.barrier_active
                & ~barrier_walkable_by_pressure_plate
            )

            is_walkable = (
                (new_cell_static == StaticObject.EMPTY)
                | (new_cell_static == StaticObject.PLAYER_CONVEYOR)
                | (new_cell_static == StaticObject.PRESSURE_PLATE)
                | (is_barrier_tile & ~barrier_blocks)
            )

            new_pos = tree_select(is_walkable, new_pos, pos)
            return agent.replace(pos=new_pos, dir=direction)

        return jax.lax.cond(
            direction != -1,
            _move,
            lambda a, _: a,
            agent,
            direction,
        )

    return jax.vmap(_move_wrapper)(state.agents, actions)

def resolve_agent_destination_collisions(
    original_agents: Agent,
    proposed_agents: Agent,
    config: OvercookedV3Config,
) -> Agent:
    """Rollback proposed moves until no two agents occupy the same destination."""

    def _positions_with_masked_agents_rolled_back(mask):
        return tree_select(mask, original_agents.pos, proposed_agents.pos)

    def _get_collisions(mask):
        positions = _positions_with_masked_agents_rolled_back(mask)

        collision_grid = jnp.zeros((config.height, config.width))
        collision_grid, _ = jax.lax.scan(
            lambda grid, pos: (grid.at[pos.y, pos.x].add(1), None),
            collision_grid,
            positions,
        )

        collision_mask = collision_grid > 1
        return jax.vmap(lambda p: collision_mask[p.y, p.x])(positions)

    initial_mask = jnp.zeros((config.num_agents,), dtype=bool)
    rollback_mask = jax.lax.while_loop(
        lambda mask: jnp.any(_get_collisions(mask)),
        lambda mask: mask | _get_collisions(mask),
        initial_mask,
    )

    return proposed_agents.replace(
        pos=_positions_with_masked_agents_rolled_back(rollback_mask)
    )

def prevent_agents_from_swapping_positions(
    original_agents: Agent, proposed_agents: Agent
) -> Agent:
    """Rollback agents that attempted to move through each other's positions."""
    swap_mask = find_agents_that_swapped_positions(
        original_agents.pos, proposed_agents.pos
    )
    resolved_positions = tree_select(swap_mask, original_agents.pos, proposed_agents.pos)
    return proposed_agents.replace(pos=resolved_positions)

def find_agents_that_swapped_positions(original_positions, new_positions) -> chex.Array:
    """Return a mask for agents whose moves form pairwise position swaps."""
    original_positions = original_positions.to_array()
    new_positions = new_positions.to_array()

    original_pos_expanded = jnp.expand_dims(original_positions, axis=0)
    new_pos_expanded = jnp.expand_dims(new_positions, axis=1)

    swap_mask = (original_pos_expanded == new_pos_expanded).all(axis=-1)
    swap_mask = jnp.fill_diagonal(swap_mask, False, inplace=False)

    swap_pairs = jnp.logical_and(swap_mask, swap_mask.T)
    return jnp.any(swap_pairs, axis=0)

def apply_agent_interact_actions(
    key: chex.PRNGKey,
    state: State,
    moved_agents: Agent,
    actions: chex.Array,
    config: OvercookedV3Config,
) -> Tuple[State, float, chex.Array]:
    """Apply interact actions, update carried items, and advance pot timers."""

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
            ) = carry

            key, subkey = jax.random.split(key)
            pot_cook_time = sample_pot_cook_time(subkey, config)

            (
                new_grid,
                new_agent,
                new_correct_delivery,
                interact_reward,
                shaped_reward,
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
            )
            return carry, (new_agent, shaped_reward)

        return jax.lax.cond(
            is_interact, _interact, lambda c, a: (c, (a, 0.0)), carry, agent
        )

    carry = (
        state.grid,
        False,
        0.0,
        state.pot_cooking_timer,
        state.pot_cook_durations,
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
            _key,
        ),
        (new_agents, shaped_rewards),
    ) = jax.lax.scan(_interact_wrapper, carry, xs)

    new_grid, new_pot_timers = update_pot_timers(
        new_grid, new_pot_timers, state.pot_positions, state.pot_active_mask, config
    )
    new_pot_cook_durations = jnp.where(
        new_pot_timers == 0, 0, new_pot_cook_durations
    )

    return (
        state.replace(
            agents=new_agents,
            grid=new_grid,
            pot_cooking_timer=new_pot_timers,
            pot_cook_durations=new_pot_cook_durations,
            new_correct_delivery=new_correct_delivery,
        ),
        reward,
        shaped_rewards,
    )
