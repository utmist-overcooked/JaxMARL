"""Functional observation construction helpers for Overcooked V3."""

from typing import Dict, Tuple

import chex
import jax
import jax.numpy as jnp

from jaxmarl.environments.overcooked_v3.common import (
    StaticObject,
    DIRTY_PLATE_BIT_SHIFT,
)
from jaxmarl.environments.overcooked_v3.config import OvercookedV3Config
from jaxmarl.environments.overcooked_v3.settings import MAX_POTS
from jaxmarl.environments.overcooked_v3.state import ObservationType, State

def calculate_observation_shape(
    width: int,
    height: int,
    layout,
    observation_type,
    agent_view_size,
    has_prep_stations: bool = False,
    enable_dish_washing: bool = False,
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
            # - recipe_layers: 2 + num_ing
            # - extra_layers: 1 (pot timer)
            # Total: 30 + 5 * num_ingredients
            # Layouts with prep stations add 3 static layers (cutting
            # board, grill, blender) and 1 extra layer (prep progress).
            num_layers = 30 + 5 * num_ingredients
            if has_prep_stations:
                num_layers += 4
            # Dish washing adds 2 static layers (sink, dirty pile), 1 extra
            # layer (plate stack / dirty pile counts) and a dirty-plate bit
            # in each of the 4 item-encoding blocks.
            if enable_dish_washing:
                num_layers += 7
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
    static_encoding_list = [
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
    if config.has_prep_stations:
        static_encoding_list += [
            StaticObject.CUTTING_BOARD,
            StaticObject.GRILL,
            StaticObject.BLENDER,
        ]
    if config.enable_dish_washing:
        static_encoding_list += [
            StaticObject.SINK,
            StaticObject.DIRTY_PLATE_PILE,
        ]
    static_encoding = jnp.array(static_encoding_list)
    static_layers = static_objects[..., None] == static_encoding

    def _ingredient_layers(ingredients):
        shifts = [0, 1] + [2 * (i + 1) for i in range(num_ingredients)]
        masks = [0x1, 0x1] + [0x3] * num_ingredients
        if config.enable_dish_washing:
            # Dirty plates must be distinguishable from clean ones.
            shifts = shifts + [DIRTY_PLATE_BIT_SHIFT]
            masks = masks + [0x1]
        shift = jnp.array(shifts)
        mask = jnp.array(masks)

        layers = ingredients[..., None] >> shift
        layers = layers & mask
        return layers

    recipe_indicator_mask = static_objects == StaticObject.RECIPE_INDICATOR
    has_recipe_indicator = jnp.any(recipe_indicator_mask)
    # Order queues can change the target recipe mid-episode. Some older
    # layouts, including around_the_island, have no R tile because they used
    # a fixed recipe; broadcast the active order in the existing recipe
    # channels so the policy can observe what should be delivered.
    recipe_visible_mask = recipe_indicator_mask | (
        config.enable_order_queue & ~has_recipe_indicator
    )
    recipe_ingredients = jnp.where(recipe_visible_mask, state.recipe, 0)

    pot_timer_layer = jnp.zeros((height, width), dtype=jnp.int32)
    for i in range(MAX_POTS):
        y, x = state.pot_positions[i]
        timer = state.pot_cooking_timer[i]
        is_active = state.pot_active_mask[i]
        pot_timer_layer = jax.lax.select(
            is_active, pot_timer_layer.at[y, x].set(timer), pot_timer_layer
        )

    extra_layer_list = [pot_timer_layer]
    if config.enable_dish_washing:
        # Surface both plate counters spatially: the clean stack sits on the
        # plate pile tiles, the dirty backlog on the dirty pile tiles.
        plate_count_layer = jnp.where(
            static_objects == StaticObject.PLATE_PILE,
            state.plate_stack_count,
            0,
        )
        plate_count_layer = jnp.where(
            static_objects == StaticObject.DIRTY_PLATE_PILE,
            state.dirty_pile_count,
            plate_count_layer,
        )
        extra_layer_list.append(plate_count_layer)
    if config.has_prep_stations:
        # Chop progress / grill timer / blender timer on station tiles
        prep_progress_layer = jnp.where(
            StaticObject.is_prep_station(static_objects),
            state.grid[:, :, 2],
            0,
        )
        extra_layer_list.append(prep_progress_layer)
    extra_layers = jnp.stack(extra_layer_list, axis=-1)

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
