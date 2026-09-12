"""Functional observation construction helpers for Overcooked V3."""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import StaticObject
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS
from jaxmarl.environments.overcooked_v3.state import ObservationType, State

def calculate_observation_shape(
    width: int,
    height: int,
    layout,
    observation_type,
    agent_view_size,
    enable_order_queue: bool = False,
    max_orders: int = 1,
) -> Tuple[int, ...]:
    """Calculate observation shape from static layout and observation settings."""
    if agent_view_size:
        view_size = agent_view_size * 2 + 1
        view_width = min(width, view_size)
        view_height = min(height, view_size)
    else:
        view_width = width
        view_height = height

    def _get_obs_shape_single(obs_type):
        if obs_type == ObservationType.DEFAULT:
            num_ingredients = layout.num_ingredients
            # Layers breakdown:
            # - agent_layer: 1 (pos) + 4 (dir) + (2 + num_ing) (inv) = 7 + num_ing
            # - other_agent_layers: same = 7 + num_ing
            # - static_layers: 11
            # - ingredient_pile_layers: num_ing
            # - ingredients_layers: 2 + num_ing
            # - recipe_layers: (2 + num_ing) per visible recipe slot
            # - extra_layers: 1 (pot timer)
            # Queue-off uses one slot; queue-on exposes all max_orders slots.
            num_recipe_slots = max_orders if enable_order_queue else 1
            num_layers = (
                28
                + 4 * num_ingredients
                + num_recipe_slots * (2 + num_ingredients)
            )
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
    """Get observations for all agents for one observation encoding."""
    if obs_type == ObservationType.DEFAULT:
        all_obs = get_obs_default(state, config)
    elif obs_type == ObservationType.FEATURIZED:
        all_obs = jnp.zeros((config.num_agents,) + config.obs_shape)
    else:
        raise ValueError(f"Invalid observation type: {obs_type}")

    def _mask_obs(obs, agent):
        view_size = config.agent_view_size
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
            config.obs_shape,
        )

    if config.agent_view_size is not None:
        all_obs = jax.vmap(_mask_obs)(all_obs, state.agents)

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
    has_recipe_indicator = jnp.any(recipe_indicator_mask)
    # An R tile spatially gates recipe information. Layouts without one expose
    # the current recipe (or the complete queue) at every visible grid cell.
    recipe_visible_mask = recipe_indicator_mask | ~has_recipe_indicator

    if config.enable_order_queue:
        safe_order_types = jnp.clip(
            state.order_types,
            0,
            config.order_recipe_encodings.shape[0] - 1,
        )
        visible_recipes = jnp.where(
            state.order_active_mask,
            config.order_recipe_encodings[safe_order_types],
            0,
        )
    else:
        visible_recipes = state.recipe[None]

    recipe_ingredients = jnp.where(
        recipe_visible_mask[..., None],
        visible_recipes[None, None, :],
        0,
    )

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
        recipe_layers = jnp.concatenate(
            [
                _ingredient_layers(recipe_ingredients[..., slot_idx])
                for slot_idx in range(recipe_ingredients.shape[-1])
            ],
            axis=-1,
        )

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
