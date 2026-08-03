"""Functional agent action phase orchestration for Overcooked V3."""

from typing import Tuple

import chex

from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.interactions import (
    apply_agent_button_interactions,
    apply_agent_interact_actions,
)
from jaxmarl.environments.overcooked_v3.movement import (
    find_agents_that_swapped_positions,
    find_barriers_opened_by_current_pressure_plate_occupants,
    is_agent_walkable,
    move_agents_to_requested_positions,
    prevent_agents_from_swapping_positions,
    resolve_agent_destination_collisions,
)
from jaxmarl.environments.overcooked_v3.state import State
from jaxmarl.environments.overcooked_v3.systems.barriers import barriers_occupied

__all__ = [
    "apply_agent_button_interactions",
    "apply_agent_interact_actions",
    "barriers_occupied",
    "find_agents_that_swapped_positions",
    "find_barriers_opened_by_current_pressure_plate_occupants",
    "is_agent_walkable",
    "move_agents_to_requested_positions",
    "prevent_agents_from_swapping_positions",
    "resolve_agent_destination_collisions",
    "run_agent_action_phase",
]

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
