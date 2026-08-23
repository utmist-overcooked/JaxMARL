"""Functional agent action phase orchestration for Overcooked V3."""

from typing import Dict, Tuple

import chex
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.interactions import (
    apply_agent_button_interactions,
    apply_agent_interact_actions,
    dense_task_shaping,
    merge_reward_breakdowns,
)
from jaxmarl.environments.overcooked_v3.movement import (
    find_agents_that_swapped_positions,
    find_barriers_opened_by_current_pressure_plate_occupants,
    is_agent_walkable,
    move_agents_to_requested_positions,
    prevent_agents_from_swapping_positions,
    resolve_agent_destination_collisions,
)
from jaxmarl.environments.overcooked_v3.settings import REWARD_COMPONENT_KEYS
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
) -> Tuple[State, float, chex.Array, Dict[str, chex.Array], chex.Array]:
    """Run movement, collision handling, interactions, and button effects.

    Returns the new state, the team reward, per-agent shaped rewards, the
    reward_breakdown, and event_metrics.

    reward_breakdown is a dict of REWARD_COMPONENT_KEYS -> a (num_agents,)
    array each, itemizing what shaped_rewards/reward summed -- useful for
    reward-hacking diagnostics, otherwise safe to ignore. event_metrics is the
    (num_agents, len(EVENT_NAMES)) counter array surfaced as `event/*` in the
    step info dict.
    """
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

    # Dense navigation shaping is computed from the pre/post movement agents, so
    # it has to run before interactions mutate the grid.
    dense_shaped_rewards = jnp.zeros((config.num_agents,), dtype=jnp.float32)
    dense_breakdown = {
        key: jnp.zeros((config.num_agents,), dtype=jnp.float32)
        for key in REWARD_COMPONENT_KEYS
    }
    if config.shaped_rewards_enabled and config.dense_task_shaping:
        dense_shaped_rewards, dense_breakdown = dense_task_shaping(
            state.grid, state.recipe, state.agents, moved_agents, actions, config
        )

    (
        state,
        reward,
        shaped_rewards,
        interact_breakdown,
        event_metrics,
    ) = apply_agent_interact_actions(key, state, moved_agents, actions, config)
    shaped_rewards = shaped_rewards + dense_shaped_rewards
    reward_breakdown = merge_reward_breakdowns(interact_breakdown, dense_breakdown)
    state = apply_agent_button_interactions(state, actions, config)

    return state, reward, shaped_rewards, reward_breakdown, event_metrics
