"""Macro-action wrapper for Overcooked V3.

The base Overcooked V3 environment is unchanged: rewards, objects, timers,
conveyors, buttons, barriers, and collision handling all come from
``OvercookedV3``. This module only changes the action interface. Each macro
action emits one primitive Overcooked V3 action per environment step until the
macro terminates, following the style of WeihaoTan's macro Overcooked env.
"""

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
    """Available macro and one-step primitive actions exposed to policies."""

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
    # One macro per pressure-plate layout slot instead of a single ambiguous
    # "nearest plate" macro -- see PRESSURE_PLATE_MACROS below. Extending to
    # more plates means adding stand_on_pressure_plate_2, etc. here (bumping
    # wait_for_nearest_pot/up/down/left/right accordingly) and appending to
    # PRESSURE_PLATE_MACROS.
    stand_on_pressure_plate_0 = 11
    stand_on_pressure_plate_1 = 12
    wait_for_nearest_pot = 13
    up = 14
    down = 15
    left = 16
    right = 17


MACRO_ACTION_NAMES: Tuple[str, ...] = tuple(action.name for action in MacroActions)

# Macro index i always targets pressure_plate slot i (state.pressure_plate_positions[i]),
# in the row-major layout-string order pressure_plate_config indexes by. This
# fixed mapping is what lets a policy target a specific plate instead of
# whichever one happens to be nearest.
PRESSURE_PLATE_MACROS: Tuple[MacroActions, ...] = (
    MacroActions.stand_on_pressure_plate_0,
    MacroActions.stand_on_pressure_plate_1,
)


@chex.dataclass
class State:
    """Overcooked V3 state plus macro-action bookkeeping."""

    agents: Agent
    grid: chex.Array

    pot_positions: chex.Array
    pot_cooking_timer: chex.Array
    pot_active_mask: chex.Array
    pot_cook_durations: chex.Array

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

        replace_macro = self._macro_replacement_mask(
            state, requested_macro_actions
        )
        return self._step_with_macro_replacements(
            key, state, requested_macro_actions, replace_macro
        )

    def _macro_replacement_mask(
        self, state: State, requested_macro_actions: chex.Array
    ) -> chex.Array:
        """Select new macros only for agents whose current macro has ended."""
        del requested_macro_actions
        return state.macro_action_done

    def _step_with_macro_replacements(
        self,
        key: chex.PRNGKey,
        state: State,
        requested_macro_actions: chex.Array,
        replace_macro: chex.Array,
    ) -> Tuple[Dict[str, chex.Array], State, Dict[str, float], Dict[str, bool], Dict]:
        """Execute one primitive step with an explicit per-agent replacement mask."""

        current_macro_actions = jnp.where(
            replace_macro,
            requested_macro_actions,
            state.current_macro_actions,
        )
        macro_step_count = jnp.where(
            replace_macro, 0, state.macro_step_count
        )

        primitive_actions, macro_reachable = self._macro_to_primitive_actions(
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
            macro_reachable,
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
        info["macro_action_started"] = {
            f"agent_{i}": replace_macro[i] for i in range(self.num_agents)
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
            pot_cook_durations=state.pot_cook_durations,
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
    ) -> Tuple[chex.Array, chex.Array]:
        """Translate each macro into one primitive action and a reachability flag."""
        walkable_mask = self._current_walkable_mask(state)
        agent_idxs = jnp.arange(self.num_agents)
        return jax.vmap(
            lambda agent_idx, macro_action: self._macro_to_primitive_action(
                state, agent_idx, macro_action, walkable_mask
            )
        )(agent_idxs, macro_actions)

    def _macro_to_primitive_action(
        self,
        state: State,
        agent_idx: chex.Array,
        macro_action: chex.Array,
        walkable_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Plan one primitive action for one macro using one dynamic flood fill."""
        agent = self._agent_at(state, agent_idx)
        static_layer = state.grid[:, :, 0]
        dynamic_layer = state.grid[:, :, 1]
        counter_mask = self._counter_like_static_mask(static_layer)

        target_mask = jnp.zeros((self.height, self.width), dtype=jnp.bool_)
        target_mask = jnp.where(
            macro_action == MacroActions.get_ingredient_0,
            static_layer == StaticObject.ingredient_pile(0),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.get_ingredient_1,
            static_layer == StaticObject.ingredient_pile(1),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.get_ingredient_2,
            static_layer == StaticObject.ingredient_pile(2),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.get_plate,
            static_layer == StaticObject.PLATE_PILE,
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.put_ingredient_in_nearest_pot,
            self._valid_pot_placement_mask(state, agent.inventory),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.get_soup_from_nearest_pot,
            self._ready_recipe_pot_mask(state, agent),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.deliver,
            static_layer == StaticObject.GOAL,
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.drop_on_nearest_counter,
            counter_mask & (dynamic_layer == DynamicObject.EMPTY),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.pickup_from_nearest_counter,
            counter_mask & (dynamic_layer != DynamicObject.EMPTY),
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.press_nearest_button,
            static_layer == StaticObject.BUTTON,
            target_mask,
        )

        interaction_macro = (
            (macro_action >= MacroActions.get_ingredient_0)
            & (macro_action <= MacroActions.press_nearest_button)
        )
        pressure_plate_macro = jnp.zeros((), dtype=jnp.bool_)
        pressure_plate_goals = jnp.zeros(
            (self.height, self.width), dtype=jnp.bool_
        )
        for plate_idx, plate_macro in enumerate(PRESSURE_PLATE_MACROS):
            is_this_macro = macro_action == plate_macro
            pressure_plate_macro = pressure_plate_macro | is_this_macro
            plate_y = state.pressure_plate_positions[plate_idx, 0]
            plate_x = state.pressure_plate_positions[plate_idx, 1]
            this_plate_goal = (
                jnp.zeros((self.height, self.width), dtype=jnp.bool_)
                .at[plate_y, plate_x]
                .set(state.pressure_plate_active_mask[plate_idx])
            )
            pressure_plate_goals = jnp.where(
                is_this_macro, this_plate_goal, pressure_plate_goals
            )
        pressure_plate_goals &= walkable_mask
        navigation_macro = interaction_macro | pressure_plate_macro

        interaction_goals = (
            jnp.pad(target_mask[:-1, :], ((1, 0), (0, 0)))
            | jnp.pad(target_mask[1:, :], ((0, 1), (0, 0)))
            | jnp.pad(target_mask[:, :-1], ((0, 0), (1, 0)))
            | jnp.pad(target_mask[:, 1:], ((0, 0), (0, 1)))
        ) & walkable_mask
        goal_mask = jnp.where(
            interaction_macro, interaction_goals, pressure_plate_goals
        )
        goal_mask &= navigation_macro

        distances = self._distance_to_goals(walkable_mask, goal_mask)
        agent_distance = distances[agent.pos.y, agent.pos.x]
        has_path = agent_distance < INF_DISTANCE
        at_goal = agent_distance == 0

        move_action = self._next_action_avoiding_agents(
            state, agent, walkable_mask, distances
        )

        candidate_x = agent.pos.x + self._dir_dx
        candidate_y = agent.pos.y + self._dir_dy
        candidate_in_bounds = (
            (candidate_x >= 0)
            & (candidate_x < self.width)
            & (candidate_y >= 0)
            & (candidate_y < self.height)
        )
        safe_x = jnp.clip(candidate_x, 0, self.width - 1)
        safe_y = jnp.clip(candidate_y, 0, self.height - 1)
        adjacent_targets = candidate_in_bounds & target_mask[safe_y, safe_x]
        target_direction = jnp.argmax(adjacent_targets)
        face_action = self._dir_to_action[target_direction]
        interact_action = jnp.where(
            agent.dir == target_direction, Actions.interact, face_action
        )

        navigation_action = jnp.where(at_goal, interact_action, move_action)
        navigation_action = jnp.where(
            pressure_plate_macro & at_goal, Actions.stay, navigation_action
        )

        inventory_empty = agent.inventory == DynamicObject.EMPTY
        can_execute = jnp.ones((), dtype=jnp.bool_)
        can_execute = jnp.where(
            (macro_action >= MacroActions.get_ingredient_0)
            & (macro_action <= MacroActions.get_ingredient_2),
            inventory_empty,
            can_execute,
        )
        can_execute = jnp.where(
            macro_action == MacroActions.put_ingredient_in_nearest_pot,
            DynamicObject.is_ingredient(agent.inventory),
            can_execute,
        )
        can_execute = jnp.where(
            macro_action == MacroActions.get_soup_from_nearest_pot,
            agent.inventory == DynamicObject.PLATE,
            can_execute,
        )
        can_execute = jnp.where(
            macro_action == MacroActions.deliver,
            (agent.inventory & DynamicObject.COOKED) != 0,
            can_execute,
        )
        can_execute = jnp.where(
            macro_action == MacroActions.drop_on_nearest_counter,
            ~inventory_empty,
            can_execute,
        )
        can_execute = jnp.where(
            macro_action == MacroActions.pickup_from_nearest_counter,
            inventory_empty,
            can_execute,
        )

        primitive_action = jnp.where(
            navigation_macro & can_execute & has_path,
            navigation_action,
            Actions.stay,
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.up, Actions.up, primitive_action
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.down, Actions.down, primitive_action
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.left, Actions.left, primitive_action
        )
        primitive_action = jnp.where(
            macro_action == MacroActions.right, Actions.right, primitive_action
        )
        macro_reachable = ~navigation_macro | has_path
        return primitive_action.astype(jnp.int32), macro_reachable

    def _compute_macro_done(
        self,
        state: State,
        macro_actions: chex.Array,
        primitive_actions: chex.Array,
        macro_reachable: chex.Array,
    ) -> chex.Array:
        """Evaluate action-specific completion and unreachable navigation goals."""
        agent_idxs = jnp.arange(self.num_agents)
        return jax.vmap(
            lambda agent_idx, macro_action, primitive_action, reachable: (
                self._macro_done_for_agent(
                    state, agent_idx, macro_action, primitive_action, reachable
                )
            )
        )(agent_idxs, macro_actions, primitive_actions, macro_reachable)

    def _macro_done_for_agent(
        self,
        state: State,
        agent_idx: chex.Array,
        macro_action: chex.Array,
        primitive_action: chex.Array,
        macro_reachable: chex.Array,
    ) -> chex.Array:
        """Return whether one agent's macro has completed or become unreachable."""
        agent = self._agent_at(state, agent_idx)
        inventory = agent.inventory

        primitive_move = (macro_action >= MacroActions.up) & (
            macro_action <= MacroActions.right
        )
        done = (macro_action == MacroActions.wait) | primitive_move
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
            | ~jnp.any(self._ready_recipe_pot_mask(state, agent)),
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
        for plate_idx, plate_macro in enumerate(PRESSURE_PLATE_MACROS):
            plate_y = state.pressure_plate_positions[plate_idx, 0]
            plate_x = state.pressure_plate_positions[plate_idx, 1]
            on_this_plate = (agent.pos.y == plate_y) & (agent.pos.x == plate_x)
            done = jnp.where(
                macro_action == plate_macro,
                on_this_plate | ~state.pressure_plate_active_mask[plate_idx],
                done,
            )
        done = jnp.where(
            macro_action == MacroActions.wait_for_nearest_pot,
            jnp.any(self._ready_recipe_pot_mask(state, agent))
            | ~self._any_pot_cooking_visible(state, agent),
            done,
        )
        return done | ~macro_reachable

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

    def _current_walkable_mask(self, state: State) -> chex.Array:
        """Return walkability for the current grid and barrier state."""
        static_layer = state.grid[:, :, 0]
        walkable_mask = (
            (static_layer == StaticObject.EMPTY)
            | (static_layer == StaticObject.PLAYER_CONVEYOR)
            | (static_layer == StaticObject.PRESSURE_PLATE)
            | (static_layer == StaticObject.BARRIER)
        )

        if self.enable_pressure_plates:
            agent_on_plate = (
                state.pressure_plate_positions[:, 0, None]
                == state.agents.pos.y[None, :]
            ) & (
                state.pressure_plate_positions[:, 1, None]
                == state.agents.pos.x[None, :]
            )
            plate_pressed = state.pressure_plate_active_mask & jnp.any(
                agent_on_plate, axis=1
            )
            opened_by_plate = jnp.any(
                state.pressure_plate_linked_barrier & plate_pressed[:, None],
                axis=0,
            )
        else:
            opened_by_plate = jnp.zeros_like(state.barrier_active)

        blocked_barriers = (
            state.barrier_active_mask
            & state.barrier_active
            & ~opened_by_plate
        )
        blocked_cells = jnp.zeros(
            (self.height, self.width), dtype=jnp.int32
        ).at[
            state.barrier_positions[:, 0], state.barrier_positions[:, 1]
        ].add(blocked_barriers.astype(jnp.int32))
        return walkable_mask & (blocked_cells == 0)

    def _distance_to_goals(
        self, walkable_mask: chex.Array, goal_mask: chex.Array
    ) -> chex.Array:
        """Flood distances from all goals through the current walkable grid."""
        distances = jnp.where(goal_mask, 0, INF_DISTANCE).astype(jnp.int32)

        def relax(_iteration, current_distances):
            """Propagate known goal distances outward by one grid edge."""
            padded = jnp.pad(
                current_distances,
                ((1, 1), (1, 1)),
                constant_values=INF_DISTANCE,
            )
            nearest_neighbor = jnp.minimum(
                jnp.minimum(padded[:-2, 1:-1], padded[2:, 1:-1]),
                jnp.minimum(padded[1:-1, :-2], padded[1:-1, 2:]),
            )
            relaxed = jnp.minimum(current_distances, nearest_neighbor + 1)
            return jnp.where(walkable_mask, relaxed, INF_DISTANCE)

        return lax.fori_loop(
            0, self.height * self.width, relax, distances
        )

    def _next_action_avoiding_agents(
        self,
        state: State,
        agent: Agent,
        walkable_mask: chex.Array,
        distances: chex.Array,
    ) -> chex.Array:
        """Choose the free neighboring step with the lowest dynamic distance."""
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
        candidate_walkable = walkable_mask[safe_y, safe_x]
        candidate_unoccupied = self._cell_unoccupied_by_other_agents(
            state, agent, safe_y, safe_x
        )
        candidate_distances = distances[safe_y, safe_x]
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
        """Return which candidate cells are not occupied by another agent."""
        occupied = (state.agents.pos.x[:, None] == cell_x[None, :]) & (
            state.agents.pos.y[:, None] == cell_y[None, :]
        )
        own_cell = (agent.pos.x == cell_x) & (agent.pos.y == cell_y)
        return ~jnp.any(occupied & ~own_cell[None, :], axis=0)

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

    def _agent_visible_cell_mask(self, agent: Agent) -> chex.Array:
        """Grid cells inside this agent's observation window.

        Mirrors the cropping in observations.get_obs_for_type: agent_view_size=v
        yields a (2v+1)x(2v+1) window centred on the agent. With agent_view_size
        unset the agent observes the whole grid, so everything is visible.
        """
        if self.agent_view_size is None:
            return jnp.ones((self.height, self.width), dtype=jnp.bool_)
        yy, xx = jnp.meshgrid(
            jnp.arange(self.height), jnp.arange(self.width), indexing="ij"
        )
        return (jnp.abs(yy - agent.pos.y) <= self.agent_view_size) & (
            jnp.abs(xx - agent.pos.x) <= self.agent_view_size
        )

    def _visible_pot_slot_mask(self, state: State, agent: Agent) -> chex.Array:
        """Which pot slots (not grid cells) this agent can currently see."""
        if self.agent_view_size is None:
            return state.pot_active_mask
        dy = jnp.abs(state.pot_positions[:, 0] - agent.pos.y)
        dx = jnp.abs(state.pot_positions[:, 1] - agent.pos.x)
        return (
            state.pot_active_mask
            & (dy <= self.agent_view_size)
            & (dx <= self.agent_view_size)
        )

    def _any_pot_cooking_visible(self, state: State, agent: Agent) -> chex.Array:
        """Is a pot the agent can actually see still counting down?

        Used instead of a global `state.pot_cooking_timer > 0` check so
        wait_for_nearest_pot cannot terminate on knowledge of pots elsewhere.
        """
        return jnp.any(
            (state.pot_cooking_timer > 0) & self._visible_pot_slot_mask(state, agent)
        )

    def _ready_recipe_pot_mask(
        self, state: State, agent: Optional[Agent] = None
    ) -> chex.Array:
        """Pots holding the finished recipe.

        When `agent` is given the result is restricted to pots inside that
        agent's observation window, so macros cannot use pot readiness the
        agent has no way of perceiving. Passing agent=None keeps the original
        ground-truth behaviour and is only appropriate where a privileged view
        is intended.
        """
        pot_mask = state.grid[:, :, 0] == StaticObject.POT
        pot_contents = state.grid[:, :, 1]
        plated_recipe = state.recipe | DynamicObject.PLATE | DynamicObject.COOKED
        ready = pot_mask & ((pot_contents | DynamicObject.PLATE) == plated_recipe)
        if agent is None:
            return ready
        return ready & self._agent_visible_cell_mask(agent)

    def _agent_on_static_object(
        self, state: State, agent: Agent, static_object: int
    ) -> chex.Array:
        return state.grid[agent.pos.y, agent.pos.x, 0] == static_object

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
        """Mask macros that cannot make progress from the current state."""
        static_layer = state.grid[:, :, 0]
        dynamic_layer = state.grid[:, :, 1]
        counter_mask = self._counter_like_static_mask(static_layer)

        def agent_mask(agent_idx):
            inventory = state.agents.inventory[agent_idx]
            inventory_empty = inventory == DynamicObject.EMPTY
            # Pot-state availability is judged from this agent's own view, so
            # the mask never advertises a macro whose trigger the agent cannot
            # perceive (see _ready_recipe_pot_mask / _any_pot_cooking_visible).
            mask_agent = self._agent_at(state, agent_idx)
            mask = jnp.zeros((self.num_macro_actions,), dtype=jnp.bool_)
            mask = mask.at[MacroActions.wait].set(True)
            for action in (
                MacroActions.up,
                MacroActions.down,
                MacroActions.left,
                MacroActions.right,
            ):
                mask = mask.at[action].set(True)
            for ingredient_idx, action in enumerate(
                (
                    MacroActions.get_ingredient_0,
                    MacroActions.get_ingredient_1,
                    MacroActions.get_ingredient_2,
                )
            ):
                mask = mask.at[action].set(
                    inventory_empty
                    & jnp.any(
                        static_layer
                        == StaticObject.ingredient_pile(ingredient_idx)
                    )
                )
            mask = mask.at[MacroActions.get_plate].set(
                inventory_empty & jnp.any(static_layer == StaticObject.PLATE_PILE)
            )
            mask = mask.at[MacroActions.put_ingredient_in_nearest_pot].set(
                jnp.any(self._valid_pot_placement_mask(state, inventory))
            )
            mask = mask.at[MacroActions.get_soup_from_nearest_pot].set(
                (inventory == DynamicObject.PLATE)
                & jnp.any(self._ready_recipe_pot_mask(state, mask_agent))
            )
            mask = mask.at[MacroActions.deliver].set(
                ((inventory & DynamicObject.COOKED) != 0)
                & jnp.any(static_layer == StaticObject.GOAL)
            )
            mask = mask.at[MacroActions.drop_on_nearest_counter].set(
                ~inventory_empty
                & jnp.any(counter_mask & (dynamic_layer == DynamicObject.EMPTY))
            )
            mask = mask.at[MacroActions.pickup_from_nearest_counter].set(
                inventory_empty
                & jnp.any(counter_mask & (dynamic_layer != DynamicObject.EMPTY))
            )
            mask = mask.at[MacroActions.press_nearest_button].set(
                jnp.any(state.button_active_mask)
            )
            for plate_idx, plate_macro in enumerate(PRESSURE_PLATE_MACROS):
                mask = mask.at[plate_macro].set(
                    state.pressure_plate_active_mask[plate_idx]
                )
            mask = mask.at[MacroActions.wait_for_nearest_pot].set(
                self._any_pot_cooking_visible(state, mask_agent)
            )
            return mask.astype(jnp.uint8)

        masks = jax.vmap(agent_mask)(jnp.arange(self.num_agents))
        return {
            agent: masks[index] for index, agent in enumerate(self.agents)
        }


class OvercookedV3MacroInterruptible(OvercookedV3Macro):
    """Macro interface where changing the requested macro interrupts execution.

    Repeating the active macro means continue. Requesting a different macro
    replaces it immediately. This fixed action interface supports both the
    every-step and learned-replanning MAPPO baselines.
    """

    def _macro_replacement_mask(
        self, state: State, requested_macro_actions: chex.Array
    ) -> chex.Array:
        return state.macro_action_done | (
            requested_macro_actions != state.current_macro_actions
        )

    @property
    def name(self) -> str:
        return "Overcooked V3 Macro Interruptible"
