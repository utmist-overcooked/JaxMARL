"""Functional observation construction helpers for Overcooked V3."""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import StaticObject
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS
from jaxmarl.environments.overcooked_v3.state import (
    ObservationMode,
    ObservationType,
    State,
)

def calculate_observation_shape(
    width: int,
    height: int,
    layout,
    observation_type,
    agent_view_size,
    observation_mode=ObservationMode.INDIVIDUAL,
    num_agents: int = 1,
) -> Tuple[int, ...]:
    """Calculate observation shape from static layout and observation settings.

    ``observation_mode`` selects the per-agent lens over the DEFAULT grid obs:
    ``INDIVIDUAL`` keeps each agent's own crop, ``CONCAT`` stacks every agent's
    crop under a leading agent axis (self-first), and ``FULL`` returns the whole
    grid, ignoring ``agent_view_size``. It has no effect on FEATURIZED obs.
    """
    observation_mode = ObservationMode(observation_mode)

    # FULL exposes the entire grid regardless of agent_view_size.
    if observation_mode == ObservationMode.FULL or not agent_view_size:
        view_width = width
        view_height = height
    else:
        view_size = agent_view_size * 2 + 1
        view_width = min(width, view_size)
        view_height = min(height, view_size)

    def _get_obs_shape_single(obs_type):
        if obs_type == ObservationType.DEFAULT:
            num_ingredients = layout.num_ingredients
            # Layers breakdown:
            # - agent_layer: 1 (pos) + 4 (dir) + (2 + num_ing) (inv) = 7 + num_ing
            # - other_agent_layers: same = 7 + num_ing
            # - static_layers: 11
            # - ingredient_pile_layers: num_ing
            # - ingredients_layers: 2 + num_ing
            # - recipe_layers: 2 + num_ing
            # - extra_layers: 1 (pot timer)
            # Total: 30 + 5 * num_ingredients
            num_layers = 30 + 5 * num_ingredients
            # CONCAT prepends a leading agent axis holding every agent's crop.
            if observation_mode == ObservationMode.CONCAT:
                return (num_agents, view_height, view_width, num_layers)
            return (view_height, view_width, num_layers)
        if obs_type == ObservationType.FEATURIZED:
            return (64,)
        raise ValueError(f"Invalid observation type: {obs_type}")

    if isinstance(observation_type, list):
        return [_get_obs_shape_single(obs_type) for obs_type in observation_type]

    return _get_obs_shape_single(observation_type)

def get_obs(state: State, config: OvercookedV3Config) -> Dict[str, chex.Array]:
    """Get observations for all agents using the configured observation type."""
    if not isinstance(config.observation_type, list):
        return get_obs_for_type(state, config.observation_type, config)

    all_obs = {}
    for i, obs_type in enumerate(config.observation_type):
        obs = get_obs_for_type(state, obs_type, config)
        key = f"agent_{i}"
        all_obs[key] = obs[key]
    return all_obs

def get_obs_for_type(
    state: State, obs_type: ObservationType, config: OvercookedV3Config
) -> Dict[str, chex.Array]:
    """Get observations for all agents for one observation encoding.

    Applies the configured ``observation_mode`` lens to the DEFAULT grid obs:
    ``INDIVIDUAL`` returns each agent's own crop, ``CONCAT`` returns every
    agent's crop stacked self-first under a leading agent axis, and ``FULL``
    returns the whole uncropped grid. FEATURIZED obs ignores the mode.
    """
    if obs_type == ObservationType.FEATURIZED:
        # Flat per-agent stub; observation_mode does not apply (the constructor
        # forbids non-individual modes together with FEATURIZED observations).
        all_obs = jnp.zeros((config.num_agents,) + config.obs_shape)
        return {f"agent_{i}": obs for i, obs in enumerate(all_obs)}
    if obs_type != ObservationType.DEFAULT:
        raise ValueError(f"Invalid observation type: {obs_type}")

    observation_mode = ObservationMode(config.observation_mode)

    # Full uncropped grid obs for every agent: (num_agents, H, W, L).
    all_obs = get_obs_default(state, config)

    # FULL exposes the whole grid to every agent; agent_view_size is ignored.
    if observation_mode == ObservationMode.FULL:
        return {f"agent_{i}": obs for i, obs in enumerate(all_obs)}

    def _mask_obs(obs, agent):
        """Crop one agent's full-grid obs to its (2k+1) view window.

        The slice size is derived locally from agent_view_size rather than from
        config.obs_shape, which under CONCAT carries an extra agent axis.
        """
        view_size = config.agent_view_size
        window = view_size * 2 + 1
        view_height = min(config.height, window)
        view_width = min(config.width, window)
        pos = agent.pos

        padded_obs = jnp.pad(
            obs,
            ((view_size, view_size), (view_size, view_size), (0, 0)),
            mode="constant",
            constant_values=0,
        )

        return jax.lax.dynamic_slice(
            padded_obs,
            (pos.y, pos.x, 0),
            (view_height, view_width, obs.shape[-1]),
        )

    # INDIVIDUAL/CONCAT with a finite window: crop each agent to its own view.
    if config.agent_view_size is not None:
        all_obs = jax.vmap(_mask_obs)(all_obs, state.agents)

    if observation_mode == ObservationMode.CONCAT:
        # Self-first cyclic gather: stacked[i, 0] is agent i's own crop, then
        # the other agents' crops in cyclic order. Shared-parameter policies
        # therefore always read "my view" in slot 0.
        n = config.num_agents
        idx = (jnp.arange(n)[:, None] + jnp.arange(n)[None, :]) % n
        all_obs = all_obs[idx]  # (num_agents, num_agents, ...)

    return {f"agent_{i}": obs for i, obs in enumerate(all_obs)}

def get_obs_default(state: State, config: OvercookedV3Config) -> chex.Array:
    """Build default grid-based observations for every agent."""
    width = config.width
    height = config.height
    num_ingredients = config.layout.num_ingredients

    static_objects = state.grid[:, :, 0]
    ingredients = state.grid[:, :, 1]
    static_encoding = jnp.array(
        [
            StaticObject.WALL,
            StaticObject.GOAL,
            StaticObject.POT,
            StaticObject.RECIPE_INDICATOR,
            StaticObject.PLATE_PILE,
            StaticObject.ITEM_CONVEYOR,
            StaticObject.PLAYER_CONVEYOR,
            StaticObject.MOVING_WALL,
            StaticObject.BUTTON,
            StaticObject.BARRIER,
            StaticObject.PRESSURE_PLATE,
        ]
    )
    static_layers = static_objects[..., None] == static_encoding

    def _ingredient_layers(ingredients):
        shift = jnp.array([0, 1] + [2 * (i + 1) for i in range(num_ingredients)])
        mask = jnp.array([0x1, 0x1] + [0x3] * num_ingredients)

        layers = ingredients[..., None] >> shift
        layers = layers & mask
        return layers

    recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
    recipe_ingredients = jnp.where(recipe_indicator_mask, state.recipe, 0)

    pot_timer_layer = jnp.zeros((height, width), dtype=jnp.int32)
    for i in range(MAX_POTS):
        y, x = state.pot_positions[i]
        timer = state.pot_cooking_timer[i]
        is_active = state.pot_active_mask[i]
        pot_timer_layer = jax.lax.select(
            is_active, pot_timer_layer.at[y, x].set(timer), pot_timer_layer
        )

    extra_layers = jnp.stack([pot_timer_layer], axis=-1)

    def _agent_layers(agent):
        pos = agent.pos
        direction = agent.dir
        inv = agent.inventory

        pos_layers = (
            jnp.zeros((height, width, 1), dtype=jnp.uint8)
            .at[pos.y, pos.x, 0]
            .set(1)
        )
        dir_layers = (
            jnp.zeros((height, width, 4), dtype=jnp.uint8)
            .at[pos.y, pos.x, direction]
            .set(1)
        )
        inv_grid = jnp.zeros_like(ingredients).at[pos.y, pos.x].set(inv)
        inv_layers = _ingredient_layers(inv_grid)

        return jnp.concatenate([pos_layers, dir_layers, inv_layers], axis=-1)

    def _agent_obs(agent_id):
        agent_layers = jax.vmap(_agent_layers)(state.agents)
        agent_layer = agent_layers[agent_id]
        all_agent_layers = jnp.sum(agent_layers, axis=0)
        other_agent_layers = all_agent_layers - agent_layer

        ingredients_layers = _ingredient_layers(ingredients)
        recipe_layers = _ingredient_layers(recipe_ingredients)

        ingredient_pile_encoding = jnp.array(
            [
                StaticObject.INGREDIENT_PILE_BASE + i
                for i in range(num_ingredients)
            ]
        )
        ingredient_pile_layers = (
            static_objects[..., None] == ingredient_pile_encoding
        )

        return jnp.concatenate(
            [
                agent_layer,
                other_agent_layers,
                static_layers,
                ingredient_pile_layers,
                ingredients_layers,
                recipe_layers,
                extra_layers,
            ],
            axis=-1,
        )

    return jax.vmap(_agent_obs)(jnp.arange(config.num_agents))
