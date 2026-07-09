"""Macro-action wrapper for Overcooked V3.

The base Overcooked V3 environment is unchanged: rewards, objects, timers,
conveyors, buttons, barriers, and collision handling all come from
``OvercookedV3``. This module only changes the action interface. Each macro
action emits one primitive Overcooked V3 action per environment step until the
macro terminates, following the style of WeihaoTan's macro Overcooked env.
"""

from collections import deque
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from jaxmarl.environments import spaces
from jaxmarl.environments.overcooked_v3.common import (
    DIR_TO_VEC,
    MAX_INGREDIENTS,
    Actions,
    Agent,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.layouts import (
    Layout,
    overcooked_v3_layouts,
)
from jaxmarl.environments.overcooked_v3.overcooked import (
    ObservationType,
    OvercookedV3,
    State as OvercookedV3State,
)


INF_DISTANCE = np.int32(1_000_000)


class MacroActions(IntEnum):
    """Available macro actions exposed to policies."""

    wait = 0
    get_ingredient_0 = 1
    get_ingredient_1 = 2
    get_ingredient_2 = 3
    get_plate = 4
    put_ingredient_in_nearest_pot = 5
    get_soup_from_nearest_pot = 6
    deliver = 7
    drop_on_nearest_counter = 8
    pickup_from_nearest_counter = 9
    press_nearest_button = 10
    stand_on_nearest_pressure_plate = 11
    wait_for_nearest_pot = 12


MACRO_ACTION_NAMES: Tuple[str, ...] = tuple(action.name for action in MacroActions)


@chex.dataclass
class State:
    """Overcooked V3 state plus macro-action bookkeeping."""

    agents: Agent
    grid: chex.Array

    pot_positions: chex.Array
    pot_cooking_timer: chex.Array
    pot_active_mask: chex.Array

    order_types: chex.Array
    order_expirations: chex.Array
    order_active_mask: chex.Array

    item_conveyor_positions: chex.Array
    item_conveyor_directions: chex.Array
    item_conveyor_active_mask: chex.Array

    player_conveyor_positions: chex.Array
    player_conveyor_directions: chex.Array
    player_conveyor_active_mask: chex.Array

    moving_wall_positions: chex.Array
    moving_wall_directions: chex.Array
    moving_wall_active_mask: chex.Array
    moving_wall_paused: chex.Array
    moving_wall_bounce: chex.Array

    button_positions: chex.Array
    button_target_idxs: chex.Array
    button_target_mask: chex.Array
    button_action_type: chex.Array
    button_active_mask: chex.Array
    button_toggled: chex.Array

    barrier_positions: chex.Array
    barrier_active: chex.Array
    barrier_active_mask: chex.Array
    barrier_timer: chex.Array
    barrier_duration: chex.Array

    pressure_plate_positions: chex.Array
    pressure_plate_linked_barrier: chex.Array
    pressure_plate_action_type: chex.Array
    pressure_plate_active_mask: chex.Array
    pressure_plate_toggled: chex.Array

    time: chex.Array
    terminal: bool
    recipe: int
    new_correct_delivery: bool

    current_macro_actions: chex.Array
    macro_action_done: chex.Array
    macro_step_count: chex.Array


class OvercookedV3Macro(OvercookedV3):
    """Overcooked V3 with temporally extended macro actions."""

    def __init__(
        self,
        layout: Union[str, Layout] = "cramped_room",
        max_steps: int = 400,
        observation_type: Union[
            ObservationType, List[ObservationType]
        ] = ObservationType.DEFAULT,
        agent_view_size: Optional[int] = None,
        max_macro_steps: int = 80,
        **kwargs,
    ):
        super().__init__(
            layout=layout,
            max_steps=max_steps,
            observation_type=observation_type,
            agent_view_size=agent_view_size,
            **kwargs,
        )
        self.max_macro_steps = max_macro_steps
        self.macro_action_set = jnp.array(list(MacroActions), dtype=jnp.int32)
        self.macro_action_names = MACRO_ACTION_NAMES
        self.num_macro_actions = len(MacroActions)

        (
            walkable_mask,
            next_action_table,
            distance_table,
        ) = self._build_navigation_tables(self.layout)
        self._walkable_mask = jnp.array(walkable_mask)
        self._next_action_table = jnp.array(next_action_table)
        self._distance_table = jnp.array(distance_table)

        y_grid, x_grid = np.meshgrid(
            np.arange(self.height, dtype=np.int32),
            np.arange(self.width, dtype=np.int32),
            indexing="ij",
        )
        self._grid_y = jnp.array(y_grid)
        self._grid_x = jnp.array(x_grid)

        self._dir_to_action = jnp.array(
            [Actions.up, Actions.down, Actions.right, Actions.left],
            dtype=jnp.int32,
        )
        self._dir_dx = DIR_TO_VEC[:, 0]
        self._dir_dy = DIR_TO_VEC[:, 1]
        self._move_actions = jnp.array(
            [Actions.right, Actions.down, Actions.left, Actions.up],
            dtype=jnp.int32,
        )
        self._move_dx = jnp.array([1, 0, -1, 0], dtype=jnp.int32)
        self._move_dy = jnp.array([0, 1, 0, -1], dtype=jnp.int32)

    # ------------------------------------------------------------------
    # Reset and Step
    # ------------------------------------------------------------------

    def reset(self, key: chex.PRNGKey) -> Tuple[Dict[str, chex.Array], State]:
        obs, base_state = super().reset(key)
        macro_state = self._add_macro_fields(base_state)
        return lax.stop_gradient(obs), lax.stop_gradient(macro_state)

    def step_env(
        self,
        key: chex.PRNGKey,
        state: State,
        actions: Dict[str, chex.Array],
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Translate macro actions into primitive actions and step Overcooked V3."""

        requested_macro_actions = jnp.array(
            [actions[f"agent_{i}"] for i in range(self.num_agents)],
            dtype=jnp.int32,
        )
        requested_macro_actions = jnp.clip(
            requested_macro_actions, 0, self.num_macro_actions - 1
        )

        current_macro_actions = jnp.where(
            state.macro_action_done,
            requested_macro_actions,
            state.current_macro_actions,
        )
        macro_step_count = jnp.where(
            state.macro_action_done, 0, state.macro_step_count
        )

        primitive_actions = self._macro_to_primitive_actions(
            state, current_macro_actions
        )
        primitive_action_dict = {
            f"agent_{i}": primitive_actions[i] for i in range(self.num_agents)
        }

        obs, next_state, rewards, dones, info = super().step_env(
            key, state, primitive_action_dict
        )

        next_macro_step_count = macro_step_count + 1
        macro_done = self._compute_macro_done(
            next_state,
            current_macro_actions,
            primitive_actions,
        )
        macro_done = (
            macro_done
            | (next_macro_step_count >= self.max_macro_steps)
            | dones["__all__"]
        )

        next_state = next_state.replace(
            current_macro_actions=current_macro_actions,
            macro_action_done=macro_done,
            macro_step_count=jnp.where(macro_done, 0, next_macro_step_count),
        )

        info = dict(info)
        info["current_macro_action"] = {
            f"agent_{i}": current_macro_actions[i] for i in range(self.num_agents)
        }
        info["macro_action_done"] = {
            f"agent_{i}": macro_done[i] for i in range(self.num_agents)
        }
        info["primitive_action"] = {
            f"agent_{i}": primitive_actions[i] for i in range(self.num_agents)
        }

        return obs, next_state, rewards, dones, info

    def _add_macro_fields(self, state: OvercookedV3State) -> State:
        return State(
            agents=state.agents,
            grid=state.grid,
            pot_positions=state.pot_positions,
            pot_cooking_timer=state.pot_cooking_timer,
            pot_active_mask=state.pot_active_mask,
            order_types=state.order_types,
            order_expirations=state.order_expirations,
            order_active_mask=state.order_active_mask,
            item_conveyor_positions=state.item_conveyor_positions,
            item_conveyor_directions=state.item_conveyor_directions,
            item_conveyor_active_mask=state.item_conveyor_active_mask,
            player_conveyor_positions=state.player_conveyor_positions,
            player_conveyor_directions=state.player_conveyor_directions,
            player_conveyor_active_mask=state.player_conveyor_active_mask,
            moving_wall_positions=state.moving_wall_positions,
            moving_wall_directions=state.moving_wall_directions,
            moving_wall_active_mask=state.moving_wall_active_mask,
            moving_wall_paused=state.moving_wall_paused,
            moving_wall_bounce=state.moving_wall_bounce,
            button_positions=state.button_positions,
            button_target_idxs=state.button_target_idxs,
            button_target_mask=state.button_target_mask,
            button_action_type=state.button_action_type,
            button_active_mask=state.button_active_mask,
            button_toggled=state.button_toggled,
            barrier_positions=state.barrier_positions,
            barrier_active=state.barrier_active,
            barrier_active_mask=state.barrier_active_mask,
            barrier_timer=state.barrier_timer,
            barrier_duration=state.barrier_duration,
            pressure_plate_positions=state.pressure_plate_positions,
            pressure_plate_linked_barrier=state.pressure_plate_linked_barrier,
            pressure_plate_action_type=state.pressure_plate_action_type,
            pressure_plate_active_mask=state.pressure_plate_active_mask,
            pressure_plate_toggled=state.pressure_plate_toggled,
            time=state.time,
            terminal=state.terminal,
            recipe=state.recipe,
            new_correct_delivery=state.new_correct_delivery,
            current_macro_actions=jnp.full(
                (self.num_agents,), MacroActions.wait, dtype=jnp.int32
            ),
            macro_action_done=jnp.ones((self.num_agents,), dtype=jnp.bool_),
            macro_step_count=jnp.zeros((self.num_agents,), dtype=jnp.int32),
        )

    # ------------------------------------------------------------------
    # Macro Translation
    # ------------------------------------------------------------------

    def _macro_to_primitive_actions(
        self, state: State, macro_actions: chex.Array
    ) -> chex.Array:
        agent_idxs = jnp.arange(self.num_agents)
        return jax.vmap(
            lambda agent_idx, macro_action: self._macro_to_primitive_action(
                state, agent_idx, macro_action
            )
        )(agent_idxs, macro_actions)

    def _macro_to_primitive_action(
        self, state: State, agent_idx: chex.Array, macro_action: chex.Array
    ) -> chex.Array:
        agent = self._agent_at(state, agent_idx)
        primitive_action = jnp.array(Actions.stay, dtype=jnp.int32)

        primitive_action = jnp.where(
            macro_action == MacroActions.get_ingredient_0,
            self._go_interact_with_ingredient(state, agent, 0),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.get_ingredient_1,
            self._go_interact_with_ingredient(state, agent, 1),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.get_ingredient_2,
            self._go_interact_with_ingredient(state, agent, 2),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.get_plate,
            self._go_interact_with_static_object(
                state, agent, state.grid[:, :, 0] == StaticObject.PLATE_PILE
            ),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.put_ingredient_in_nearest_pot,
            self._put_ingredient_in_nearest_pot(state, agent),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.get_soup_from_nearest_pot,
            self._get_soup_from_nearest_pot(state, agent),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.deliver,
            self._deliver_to_goal(state, agent),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.drop_on_nearest_counter,
            self._drop_on_nearest_counter(state, agent),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.pickup_from_nearest_counter,
            self._pickup_from_nearest_counter(state, agent),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.press_nearest_button,
            self._go_interact_with_static_object(
                state, agent, state.grid[:, :, 0] == StaticObject.BUTTON
            ),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.stand_on_nearest_pressure_plate,
            self._go_to_nearest_pressure_plate(state, agent),
            primitive_action,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.wait_for_nearest_pot,
            jnp.array(Actions.stay, dtype=jnp.int32),
            primitive_action,
        )
        return primitive_action.astype(jnp.int32)

    def _go_interact_with_ingredient(
        self, state: State, agent: Agent, ingredient_idx: int
    ) -> chex.Array:
        inventory_empty = agent.inventory == DynamicObject.EMPTY
        target_mask = state.grid[:, :, 0] == StaticObject.ingredient_pile(
            ingredient_idx
        )
        action = self._go_interact_with_static_object(state, agent, target_mask)
        return jnp.where(inventory_empty, action, Actions.stay)

    def _go_interact_with_static_object(
        self, state: State, agent: Agent, target_mask: chex.Array
    ) -> chex.Array:
        target_y, target_x, has_target = self._nearest_interactable_target(
            state, agent, target_mask
        )
        return self._action_to_interact_with_cell(
            state, agent, target_y, target_x, has_target
        )

    def _put_ingredient_in_nearest_pot(
        self, state: State, agent: Agent
    ) -> chex.Array:
        inventory = agent.inventory
        inventory_is_ingredient = DynamicObject.is_ingredient(inventory)
        pot_mask = self._valid_pot_placement_mask(state, inventory)
        action = self._go_interact_with_static_object(state, agent, pot_mask)
        return jnp.where(inventory_is_ingredient, action, Actions.stay)

    def _get_soup_from_nearest_pot(self, state: State, agent: Agent) -> chex.Array:
        inventory_is_plate = agent.inventory == DynamicObject.PLATE
        pot_mask = self._ready_recipe_pot_mask(state)
        action = self._go_interact_with_static_object(state, agent, pot_mask)
        return jnp.where(inventory_is_plate, action, Actions.stay)

    def _deliver_to_goal(self, state: State, agent: Agent) -> chex.Array:
        inventory_is_dish = (agent.inventory & DynamicObject.COOKED) != 0
        goal_mask = state.grid[:, :, 0] == StaticObject.GOAL
        action = self._go_interact_with_static_object(state, agent, goal_mask)
        return jnp.where(inventory_is_dish, action, Actions.stay)

    def _drop_on_nearest_counter(self, state: State, agent: Agent) -> chex.Array:
        can_drop = agent.inventory != DynamicObject.EMPTY
        counter_mask = self._counter_like_static_mask(state.grid[:, :, 0])
        target_mask = counter_mask & (state.grid[:, :, 1] == DynamicObject.EMPTY)
        action = self._go_interact_with_static_object(state, agent, target_mask)
        return jnp.where(can_drop, action, Actions.stay)

    def _pickup_from_nearest_counter(self, state: State, agent: Agent) -> chex.Array:
        can_pickup = agent.inventory == DynamicObject.EMPTY
        counter_mask = self._counter_like_static_mask(state.grid[:, :, 0])
        target_mask = counter_mask & (state.grid[:, :, 1] != DynamicObject.EMPTY)
        action = self._go_interact_with_static_object(state, agent, target_mask)
        return jnp.where(can_pickup, action, Actions.stay)

    def _go_to_nearest_pressure_plate(self, state: State, agent: Agent) -> chex.Array:
        target_mask = state.grid[:, :, 0] == StaticObject.PRESSURE_PLATE
        target_y, target_x, has_target = self._nearest_walkable_target(
            agent, target_mask
        )
        return self._action_to_walkable_cell(state, agent, target_y, target_x, has_target)

    def _compute_macro_done(
        self,
        state: State,
        macro_actions: chex.Array,
        primitive_actions: chex.Array,
    ) -> chex.Array:
        agent_idxs = jnp.arange(self.num_agents)
        return jax.vmap(
            lambda agent_idx, macro_action, primitive_action: self._macro_done_for_agent(
                state, agent_idx, macro_action, primitive_action
            )
        )(agent_idxs, macro_actions, primitive_actions)

    def _macro_done_for_agent(
        self,
        state: State,
        agent_idx: chex.Array,
        macro_action: chex.Array,
        primitive_action: chex.Array,
    ) -> chex.Array:
        agent = self._agent_at(state, agent_idx)
        inventory = agent.inventory

        done = macro_action == MacroActions.wait
        done = jnp.where(
            macro_action == MacroActions.get_ingredient_0,
            (inventory == DynamicObject.ingredient(0))
            | ((inventory != DynamicObject.EMPTY) & (inventory != DynamicObject.ingredient(0)))
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.ingredient_pile(0)),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.get_ingredient_1,
            (inventory == DynamicObject.ingredient(1))
            | ((inventory != DynamicObject.EMPTY) & (inventory != DynamicObject.ingredient(1)))
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.ingredient_pile(1)),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.get_ingredient_2,
            (inventory == DynamicObject.ingredient(2))
            | ((inventory != DynamicObject.EMPTY) & (inventory != DynamicObject.ingredient(2)))
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.ingredient_pile(2)),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.get_plate,
            (inventory == DynamicObject.PLATE)
            | ((inventory != DynamicObject.EMPTY) & (inventory != DynamicObject.PLATE))
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.PLATE_PILE),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.put_ingredient_in_nearest_pot,
            ~DynamicObject.is_ingredient(inventory)
            | ~jnp.any(self._valid_pot_placement_mask(state, inventory)),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.get_soup_from_nearest_pot,
            ((inventory & DynamicObject.COOKED) != 0)
            | (inventory != DynamicObject.PLATE)
            | ~jnp.any(self._ready_recipe_pot_mask(state)),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.deliver,
            ((inventory & DynamicObject.COOKED) == 0)
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.GOAL),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.drop_on_nearest_counter,
            (inventory == DynamicObject.EMPTY)
            | ~jnp.any(
                self._counter_like_static_mask(state.grid[:, :, 0])
                & (state.grid[:, :, 1] == DynamicObject.EMPTY)
            ),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.pickup_from_nearest_counter,
            (inventory != DynamicObject.EMPTY)
            | ~jnp.any(
                self._counter_like_static_mask(state.grid[:, :, 0])
                & (state.grid[:, :, 1] != DynamicObject.EMPTY)
            ),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.press_nearest_button,
            (primitive_action == Actions.interact)
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.BUTTON),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.stand_on_nearest_pressure_plate,
            self._agent_on_static_object(state, agent, StaticObject.PRESSURE_PLATE)
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.PRESSURE_PLATE),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.wait_for_nearest_pot,
            jnp.any(self._ready_recipe_pot_mask(state))
            | ~jnp.any(state.pot_cooking_timer > 0),
            done,
        )
        return done

    # ------------------------------------------------------------------
    # Target Selection and Navigation
    # ------------------------------------------------------------------

    def _agent_at(self, state: State, agent_idx: chex.Array) -> Agent:
        return Agent(
            pos=Position(
                x=state.agents.pos.x[agent_idx],
                y=state.agents.pos.y[agent_idx],
            ),
            dir=state.agents.dir[agent_idx],
            inventory=state.agents.inventory[agent_idx],
        )

    def _nearest_interactable_target(
        self, state: State, agent: Agent, target_mask: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        del state
        ay = agent.pos.y
        ax = agent.pos.x

        candidate_x = self._grid_x[:, :, None] - self._dir_dx[None, None, :]
        candidate_y = self._grid_y[:, :, None] - self._dir_dy[None, None, :]
        candidate_in_bounds = (
            (candidate_x >= 0)
            & (candidate_x < self.width)
            & (candidate_y >= 0)
            & (candidate_y < self.height)
        )
        safe_x = jnp.clip(candidate_x, 0, self.width - 1)
        safe_y = jnp.clip(candidate_y, 0, self.height - 1)
        candidate_walkable = self._walkable_mask[safe_y, safe_x]
        candidate_distances = self._distance_table[ay, ax, safe_y, safe_x]
        candidate_distances = jnp.where(
            candidate_in_bounds & candidate_walkable,
            candidate_distances,
            INF_DISTANCE,
        )
        best_distance = jnp.min(candidate_distances, axis=-1)

        scores = jnp.where(target_mask, best_distance, INF_DISTANCE)
        flat_scores = scores.reshape((-1,))
        flat_idx = jnp.argmin(flat_scores)
        has_target = flat_scores[flat_idx] < INF_DISTANCE
        target_y = flat_idx // self.width
        target_x = flat_idx % self.width
        return target_y.astype(jnp.int32), target_x.astype(jnp.int32), has_target

    def _nearest_walkable_target(
        self, agent: Agent, target_mask: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        ay = agent.pos.y
        ax = agent.pos.x
        distances = self._distance_table[ay, ax, self._grid_y, self._grid_x]
        scores = jnp.where(
            target_mask & self._walkable_mask, distances, INF_DISTANCE
        )
        flat_scores = scores.reshape((-1,))
        flat_idx = jnp.argmin(flat_scores)
        has_target = flat_scores[flat_idx] < INF_DISTANCE
        target_y = flat_idx // self.width
        target_x = flat_idx % self.width
        return target_y.astype(jnp.int32), target_x.astype(jnp.int32), has_target

    def _action_to_interact_with_cell(
        self,
        state: State,
        agent: Agent,
        target_y: chex.Array,
        target_x: chex.Array,
        has_target: chex.Array,
    ) -> chex.Array:
        ax = agent.pos.x
        ay = agent.pos.y
        dx = target_x - ax
        dy = target_y - ay
        adjacent = (jnp.abs(dx) + jnp.abs(dy)) == 1
        desired_dir = self._direction_from_delta(dx, dy)
        face_action = self._dir_to_action[desired_dir]
        interact_or_face = jnp.where(
            agent.dir == desired_dir, Actions.interact, face_action
        )
        move_action = self._action_to_adjacent_cell(state, agent, target_y, target_x)
        action = jnp.where(adjacent, interact_or_face, move_action)
        return jnp.where(has_target, action, Actions.stay).astype(jnp.int32)

    def _action_to_adjacent_cell(
        self, state: State, agent: Agent, target_y: chex.Array, target_x: chex.Array
    ) -> chex.Array:
        ax = agent.pos.x
        ay = agent.pos.y
        candidate_x = target_x - self._dir_dx
        candidate_y = target_y - self._dir_dy
        candidate_in_bounds = (
            (candidate_x >= 0)
            & (candidate_x < self.width)
            & (candidate_y >= 0)
            & (candidate_y < self.height)
        )
        safe_x = jnp.clip(candidate_x, 0, self.width - 1)
        safe_y = jnp.clip(candidate_y, 0, self.height - 1)
        candidate_walkable = self._walkable_mask[safe_y, safe_x]
        distances = self._distance_table[ay, ax, safe_y, safe_x]
        scores = jnp.where(
            candidate_in_bounds & candidate_walkable, distances, INF_DISTANCE
        )
        best_idx = jnp.argmin(scores)
        dest_y = safe_y[best_idx]
        dest_x = safe_x[best_idx]
        has_dest = scores[best_idx] < INF_DISTANCE
        action = self._next_action_avoiding_agents(state, agent, dest_y, dest_x)
        return jnp.where(has_dest, action, Actions.stay).astype(jnp.int32)

    def _action_to_walkable_cell(
        self,
        state: State,
        agent: Agent,
        target_y: chex.Array,
        target_x: chex.Array,
        has_target: chex.Array,
    ) -> chex.Array:
        ay = agent.pos.y
        ax = agent.pos.x
        already_there = (ay == target_y) & (ax == target_x)
        action = self._next_action_avoiding_agents(state, agent, target_y, target_x)
        action = jnp.where(already_there, Actions.stay, action)
        return jnp.where(has_target, action, Actions.stay).astype(jnp.int32)

    def _next_action_avoiding_agents(
        self, state: State, agent: Agent, target_y: chex.Array, target_x: chex.Array
    ) -> chex.Array:
        ax = agent.pos.x
        ay = agent.pos.y
        candidate_x = ax + self._move_dx
        candidate_y = ay + self._move_dy
        candidate_in_bounds = (
            (candidate_x >= 0)
            & (candidate_x < self.width)
            & (candidate_y >= 0)
            & (candidate_y < self.height)
        )
        safe_x = jnp.clip(candidate_x, 0, self.width - 1)
        safe_y = jnp.clip(candidate_y, 0, self.height - 1)
        candidate_walkable = self._walkable_mask[safe_y, safe_x]
        candidate_unoccupied = self._cell_unoccupied_by_other_agents(
            state, agent, safe_y, safe_x
        )
        candidate_distances = self._distance_table[safe_y, safe_x, target_y, target_x]
        scores = jnp.where(
            candidate_in_bounds & candidate_walkable & candidate_unoccupied,
            candidate_distances,
            INF_DISTANCE,
        )
        best_idx = jnp.argmin(scores)
        has_step = scores[best_idx] < INF_DISTANCE
        action = self._move_actions[best_idx]
        return jnp.where(has_step, action, Actions.stay).astype(jnp.int32)

    def _cell_unoccupied_by_other_agents(
        self,
        state: State,
        agent: Agent,
        cell_y: chex.Array,
        cell_x: chex.Array,
    ) -> chex.Array:
        occupied = (state.agents.pos.x[:, None] == cell_x[None, :]) & (
            state.agents.pos.y[:, None] == cell_y[None, :]
        )
        own_cell = (agent.pos.x == cell_x) & (agent.pos.y == cell_y)
        return ~jnp.any(occupied & ~own_cell[None, :], axis=0)

    def _direction_from_delta(self, dx: chex.Array, dy: chex.Array) -> chex.Array:
        return jnp.select(
            [dx == 1, dx == -1, dy == 1, dy == -1],
            [Direction.RIGHT, Direction.LEFT, Direction.DOWN, Direction.UP],
            default=Direction.UP,
        ).astype(jnp.int32)

    # ------------------------------------------------------------------
    # Object Masks and Completion Helpers
    # ------------------------------------------------------------------

    def _counter_like_static_mask(self, static_layer: chex.Array) -> chex.Array:
        return (
            (static_layer == StaticObject.WALL)
            | (static_layer == StaticObject.MOVING_WALL)
            | (static_layer == StaticObject.ITEM_CONVEYOR)
            | (static_layer == StaticObject.PLAYER_CONVEYOR)
        )

    def _valid_pot_placement_mask(
        self, state: State, inventory: chex.Array
    ) -> chex.Array:
        pot_mask = state.grid[:, :, 0] == StaticObject.POT
        pot_contents = state.grid[:, :, 1]
        ingredient_counts = jax.vmap(jax.vmap(DynamicObject.ingredient_count))(
            pot_contents
        )
        pot_ingredient_type = jax.vmap(jax.vmap(DynamicObject.get_ingredient_type))(
            pot_contents
        )
        inventory_type = DynamicObject.get_ingredient_type(inventory)
        same_type = (pot_ingredient_type == inventory_type) | (
            pot_contents == DynamicObject.EMPTY
        )
        pot_not_finished = (
            ((pot_contents & DynamicObject.COOKED) == 0)
            & ((pot_contents & DynamicObject.BURNED) == 0)
        )
        return (
            pot_mask
            & DynamicObject.is_ingredient(inventory)
            & (ingredient_counts < MAX_INGREDIENTS)
            & same_type
            & pot_not_finished
        )

    def _ready_recipe_pot_mask(self, state: State) -> chex.Array:
        pot_mask = state.grid[:, :, 0] == StaticObject.POT
        pot_contents = state.grid[:, :, 1]
        plated_recipe = state.recipe | DynamicObject.PLATE | DynamicObject.COOKED
        return pot_mask & ((pot_contents | DynamicObject.PLATE) == plated_recipe)

    def _agent_on_static_object(
        self, state: State, agent: Agent, static_object: int
    ) -> chex.Array:
        return state.grid[agent.pos.y, agent.pos.x, 0] == static_object

    # ------------------------------------------------------------------
    # Navigation Table Construction
    # ------------------------------------------------------------------

    @staticmethod
    def _build_navigation_tables(layout: Layout):
        static = np.asarray(layout.static_objects)
        height, width = static.shape
        walkable = np.isin(
            static,
            [
                StaticObject.EMPTY,
                StaticObject.PLAYER_CONVEYOR,
                StaticObject.PRESSURE_PLATE,
                StaticObject.BARRIER,
            ],
        )

        distance_table = np.full(
            (height, width, height, width), INF_DISTANCE, dtype=np.int32
        )
        next_action_table = np.full(
            (height, width, height, width),
            Actions.stay,
            dtype=np.int32,
        )

        moves = [
            (0, -1, Actions.up),
            (0, 1, Actions.down),
            (1, 0, Actions.right),
            (-1, 0, Actions.left),
        ]

        for dest_y in range(height):
            for dest_x in range(width):
                if not walkable[dest_y, dest_x]:
                    continue

                distances = np.full((height, width), INF_DISTANCE, dtype=np.int32)
                distances[dest_y, dest_x] = 0
                queue = deque([(dest_y, dest_x)])

                while queue:
                    y, x = queue.popleft()
                    for dx, dy, _action in moves:
                        ny = y - dy
                        nx = x - dx
                        if not (0 <= ny < height and 0 <= nx < width):
                            continue
                        if not walkable[ny, nx]:
                            continue
                        if distances[ny, nx] != INF_DISTANCE:
                            continue
                        distances[ny, nx] = distances[y, x] + 1
                        queue.append((ny, nx))

                for src_y in range(height):
                    for src_x in range(width):
                        distance_table[
                            src_y, src_x, dest_y, dest_x
                        ] = distances[src_y, src_x]
                        if distances[src_y, src_x] == 0:
                            next_action_table[
                                src_y, src_x, dest_y, dest_x
                            ] = Actions.stay
                            continue
                        best_distance = distances[src_y, src_x]
                        best_action = Actions.stay
                        for dx, dy, action in moves:
                            ny = src_y + dy
                            nx = src_x + dx
                            if not (0 <= ny < height and 0 <= nx < width):
                                continue
                            if distances[ny, nx] < best_distance:
                                best_distance = distances[ny, nx]
                                best_action = action
                        next_action_table[src_y, src_x, dest_y, dest_x] = best_action

        return walkable, next_action_table, distance_table

    # ------------------------------------------------------------------
    # Spaces and Metadata
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:
        return "Overcooked V3 Macro"

    @property
    def num_actions(self) -> int:
        return self.num_macro_actions

    def action_space(self, agent_id="") -> spaces.Discrete:
        return spaces.Discrete(self.num_macro_actions, dtype=jnp.uint32)

    def get_avail_actions(self, state: State) -> Dict[str, chex.Array]:
        del state
        return {
            f"agent_{i}": jnp.ones((self.num_macro_actions,), dtype=jnp.uint8)
            for i in range(self.num_agents)
        }
