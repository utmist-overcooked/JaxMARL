"""Functional timestep pipeline for Overcooked V3."""

from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
from jax import lax

from jaxmarl.environments.overcooked_v3.agent_step import run_agent_action_phase
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.initialization import sample_recipe
from jaxmarl.environments.overcooked_v3.observations import get_obs
from jaxmarl.environments.overcooked_v3.state import State
from jaxmarl.environments.overcooked_v3.systems.barriers import (
    update_barrier_timers,
    update_pressure_plates,
)
from jaxmarl.environments.overcooked_v3.systems.conveyors import (
    move_items_on_item_conveyors,
    push_players_on_player_conveyors,
)
from jaxmarl.environments.overcooked_v3.systems.moving_walls import move_moving_walls
from jaxmarl.environments.overcooked_v3.systems.orders import process_order_queue
from jaxmarl.environments.overcooked_v3.interactions import merge_reward_breakdowns

def step_overcooked_v3(
    key: chex.PRNGKey,
    state: State,
    actions: Dict[str, chex.Array],
    config: OvercookedV3Config,
) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
    """Run one environment timestep as a sequence of explicit state transforms."""
    agent_actions = translate_action_dict_to_ordered_action_array(actions, config)
    # Split off the recipe key only when the feature is on, so runs with it
    # disabled keep exactly the RNG stream they had before.
    if config.resample_recipe_on_delivery:
        key, recipe_key = jax.random.split(key)
    else:
        recipe_key = None
    agent_key, order_key = partition_step_key(key, config)

    state, reward, shaped_rewards, reward_breakdown = run_agent_action_phase(
        agent_key, state, agent_actions, config
    )
    state = resample_recipe_after_correct_delivery(recipe_key, state, config)
    state = advance_dynamic_environment_systems(state, config)
    state, reward, reward_breakdown = advance_order_queue_and_add_queue_reward(
        order_key, state, reward, reward_breakdown, config
    )
    state = advance_time_and_update_terminal_flag(state, config)

    return build_step_env_return_values(
        state, reward, shaped_rewards, reward_breakdown, config
    )

def translate_action_dict_to_ordered_action_array(
    actions: Dict[str, chex.Array], config: OvercookedV3Config
) -> chex.Array:
    """Convert per-agent action dictionary entries into the ordered action array."""
    return config.action_set.take(
        indices=jnp.array([actions[f"agent_{i}"] for i in range(config.num_agents)])
    )


def partition_step_key(
    key: chex.PRNGKey, config: OvercookedV3Config
) -> Tuple[chex.PRNGKey, Optional[chex.PRNGKey]]:
    """Isolate agent/pot randomness from order-queue randomness."""
    if config.enable_order_queue:
        agent_key, order_key = jax.random.split(key)
        return agent_key, order_key

    return key, None

def resample_recipe_after_correct_delivery(
    key: Optional[chex.PRNGKey], state: State, config: OvercookedV3Config
) -> State:
    """Draw a new recipe once a correct delivery lands.

    Without this the recipe is drawn once at reset and fixed for the whole
    episode, so every soup in an episode is the same order and the episode
    carries a single recipe's worth of information. Re-drawing per delivery
    means a partner has to be told the current order repeatedly.

    Runs after the agent phase, so the delivery that triggered it was already
    scored against the recipe it satisfied; the new recipe applies from here
    on and is visible in this step's returned observation.
    """
    if not config.resample_recipe_on_delivery:
        return state

    new_recipe = sample_recipe(key, config)
    return state.replace(
        recipe=jnp.where(state.new_correct_delivery, new_recipe, state.recipe)
    )


def advance_dynamic_environment_systems(
    state: State, config: OvercookedV3Config
) -> State:
    """Advance walls, plates, conveyors, and barrier timers after agent actions."""
    if config.enable_moving_walls:
        state = move_moving_walls(state, config)

    if config.enable_pressure_plates:
        state = update_pressure_plates(state, config)

    if config.enable_item_conveyors:
        state = move_items_on_item_conveyors(state, config)

    if config.enable_player_conveyors:
        state = push_players_on_player_conveyors(state, config)

    return update_barrier_timers(state, config)

def advance_order_queue_and_add_queue_reward(
    key: Optional[chex.PRNGKey],
    state: State,
    reward: float,
    reward_breakdown: Dict[str, chex.Array],
    config: OvercookedV3Config,
) -> Tuple[State, float, Dict[str, chex.Array]]:
    """Generate and expire queued orders, adding any queue reward to the step reward."""
    if config.enable_order_queue:
        state, order_reward, order_breakdown = process_order_queue(state, key, config)
        reward = reward + order_reward
        reward_breakdown = merge_reward_breakdowns(reward_breakdown, order_breakdown)

    return state, reward, reward_breakdown

def advance_time_and_update_terminal_flag(
    state: State, config: OvercookedV3Config
) -> State:
    """Increment episode time and set the terminal flag for the new state."""
    state = state.replace(time=state.time + 1)
    done = is_terminal(state, config)
    return state.replace(terminal=done)

def build_step_env_return_values(
    state: State,
    reward: float,
    shaped_rewards: chex.Array,
    reward_breakdown: Dict[str, chex.Array],
    config: OvercookedV3Config,
) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
    """Build stopped-gradient observations, state, rewards, dones, and info."""
    obs = get_obs(state, config)
    done = state.terminal

    rewards = {f"agent_{i}": reward for i in range(config.num_agents)}
    shaped_rewards_dict = {
        f"agent_{i}": shaped_reward for i, shaped_reward in enumerate(shaped_rewards)
    }
    dones = {f"agent_{i}": done for i in range(config.num_agents)}
    dones["__all__"] = done

    return (
        lax.stop_gradient(obs),
        lax.stop_gradient(state),
        rewards,
        dones,
        {
            "shaped_reward": shaped_rewards_dict,
            # Per REWARD_COMPONENT_KEYS entry -> a (num_agents,) array,
            # itemizing what shaped_reward/reward summed. For diagnostics
            # (e.g. reward-hacking histograms), not used in training.
            "reward_breakdown": reward_breakdown,
        },
    )

def is_terminal(state: State, config: OvercookedV3Config) -> bool:
    """Return whether the state has reached the episode horizon or was terminal."""
    done_steps = state.time >= config.max_steps
    return done_steps | state.terminal
