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
    stand_on_nearest_pressure_plate = 11
    wait_for_nearest_pot = 12
    up = 13
    down = 14
    left = 15
    right = 16


MACRO_ACTION_NAMES: Tuple[str, ...] = tuple(action.name for action in MacroActions)


@chex.dataclass
class State:
    """Overcooked V3 state plus macro-action bookkeeping."""

    agents: Agent
    grid: chex.Array

    pot_positions: chex.Array
    pot_cooking_timer: chex.Array
    pot_cook_durations: chex.Array
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

        primitive_actions, macro_nav_ok = (
            self._macro_to_primitive_actions(state, current_macro_actions)
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
            macro_nav_ok,
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

        return (
            lax.stop_gradient(obs),
            lax.stop_gradient(next_state),
            rewards,
            dones,
            info,
        )

    def _add_macro_fields(self, state: OvercookedV3State) -> State:
        return State(
            agents=state.agents,
            grid=state.grid,
            pot_positions=state.pot_positions,
            pot_cooking_timer=state.pot_cooking_timer,
            pot_cook_durations=state.pot_cook_durations,
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
    ) -> Tuple[chex.Array, chex.Array]:
        """Translate each macro into one primitive action and a reachability flag."""
        walkable_mask = self._current_walkable_mask(state)
        barrier_agnostic_mask = self._barrier_agnostic_walkable_mask(state)
        agent_idxs = jnp.arange(self.num_agents)
        return jax.vmap(
            lambda agent_idx, macro_action: self._macro_to_primitive_action(
                state,
                agent_idx,
                macro_action,
                walkable_mask,
                barrier_agnostic_mask,
            )
        )(agent_idxs, macro_actions)

    def _macro_to_primitive_action(
        self,
        state: State,
        agent_idx: chex.Array,
        macro_action: chex.Array,
        walkable_mask: chex.Array,
        barrier_agnostic_mask: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Plan one primitive action for one macro using two flood fills.

        `walkable_mask` is the current barrier-aware walkability;
        `barrier_agnostic_mask` treats closed barriers as open so a blocked but
        statically-reachable target still yields a distance gradient the agent
        can walk toward (approach-and-wait). Navigation targets are chosen by
        static object existence; the dynamic outcome is discovered only when the
        base interaction runs on arrival.
        """
        agent = self._agent_at(state, agent_idx)
        static_layer = state.grid[:, :, 0]
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
        # Navigation targets are chosen by STATIC existence only: any pot, any
        # counter. The agent walks to the nearest such object and discovers the
        # dynamic condition (pot full/ready, counter empty/occupied) only when it
        # arrives and the base interaction runs. This avoids leaking dynamic,
        # possibly out-of-view world state through navigation.
        target_mask = jnp.where(
            macro_action == MacroActions.put_ingredient_in_nearest_pot,
            static_layer == StaticObject.POT,
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.get_soup_from_nearest_pot,
            static_layer == StaticObject.POT,
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.deliver,
            static_layer == StaticObject.GOAL,
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.drop_on_nearest_counter,
            counter_mask,
            target_mask,
        )
        target_mask = jnp.where(
            macro_action == MacroActions.pickup_from_nearest_counter,
            counter_mask,
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
        pressure_plate_macro = (
            macro_action == MacroActions.stand_on_nearest_pressure_plate
        )
        navigation_macro = interaction_macro | pressure_plate_macro

        # Base goal cells (before intersecting with any walkability): for an
        # interaction macro, every cell orthogonally adjacent to a target; for
        # the pressure-plate macro, the plate cells themselves.
        interaction_goal_cells = (
            jnp.pad(target_mask[:-1, :], ((1, 0), (0, 0)))
            | jnp.pad(target_mask[1:, :], ((0, 1), (0, 0)))
            | jnp.pad(target_mask[:, :-1], ((0, 0), (1, 0)))
            | jnp.pad(target_mask[:, 1:], ((0, 0), (0, 1)))
        )
        pressure_plate_cells = static_layer == StaticObject.PRESSURE_PLATE
        base_goal_cells = jnp.where(
            interaction_macro, interaction_goal_cells, pressure_plate_cells
        )
        base_goal_cells &= navigation_macro

        # Two flood fields. `distances_open` uses current (barrier-aware)
        # walkability and drives normal movement, including detours around any
        # currently-open route. `distances_all` treats closed barriers as open,
        # so a target that is only transiently blocked still yields a gradient
        # to walk toward and to judge static reachability by.
        goal_mask_open = base_goal_cells & walkable_mask
        goal_mask_all = base_goal_cells & barrier_agnostic_mask
        distances_open = self._distance_to_goals(walkable_mask, goal_mask_open)
        distances_all = self._distance_to_goals(
            barrier_agnostic_mask, goal_mask_all
        )

        agent_distance = distances_open[agent.pos.y, agent.pos.x]
        has_path = agent_distance < INF_DISTANCE
        at_goal = agent_distance == 0

        # When an open route exists, move along it (detour-aware). Otherwise walk
        # up to the block along the barrier-agnostic gradient; that step returns
        # `stay` once the agent can no longer advance (it has reached the closest
        # reachable tile and is stuck at the block).
        move_open = self._next_action_avoiding_agents(
            state, agent, walkable_mask, distances_open
        )
        move_blocked = self._step_up_to_blocked_object(
            state, agent, walkable_mask, distances_all
        )
        move_action = jnp.where(has_path, move_open, move_blocked)

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

        # Note: the stay gate no longer requires `has_path`. When the target is
        # transiently blocked, `navigation_action` is the blocked-approach step
        # (which self-stays when it cannot advance), so gating on `has_path` here
        # would stop the agent from walking up to the block.
        primitive_action = jnp.where(
            navigation_macro & can_execute,
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
        # Walk up to the block, then hand control back. A navigation macro is
        # "stuck at a block" when it has no open route and its blocked-approach
        # step could not advance (it reached the closest reachable tile and can
        # only `stay`). This covers permanent walls / nonexistent targets too
        # (stuck from the first tick), while an approach still in progress keeps
        # moving. `macro_nav_ok` is True while the macro should keep running.
        blocked_and_stuck = (~has_path) & (move_blocked == Actions.stay)
        macro_nav_ok = ~navigation_macro | ~blocked_and_stuck
        return primitive_action.astype(jnp.int32), macro_nav_ok

    def _compute_macro_done(
        self,
        state: State,
        macro_actions: chex.Array,
        primitive_actions: chex.Array,
        macro_nav_ok: chex.Array,
    ) -> chex.Array:
        """Evaluate action-specific completion plus navigation give-up (stuck)."""
        agent_idxs = jnp.arange(self.num_agents)
        return jax.vmap(
            lambda agent_idx, macro_action, primitive_action, reachable: (
                self._macro_done_for_agent(
                    state, agent_idx, macro_action, primitive_action, reachable
                )
            )
        )(
            agent_idxs,
            macro_actions,
            primitive_actions,
            macro_nav_ok,
        )

    def _macro_done_for_agent(
        self,
        state: State,
        agent_idx: chex.Array,
        macro_action: chex.Array,
        primitive_action: chex.Array,
        macro_nav_ok: chex.Array,
    ) -> chex.Array:
        """Return whether one agent's macro has completed or given up (stuck)."""
        agent = self._agent_at(state, agent_idx)
        inventory = agent.inventory
        counter_mask_static = self._counter_like_static_mask(state.grid[:, :, 0])

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
        # For the navigation-to-object macros, completion is "arrived and
        # attempted once" (`primitive_action == interact`), the inventory
        # precondition no longer holding, or the static object being gone. The
        # attempt signal is what stops the macro after the agent reaches the
        # nearest object and finds out whether the interaction was valid.
        done = jnp.where(
            macro_action == MacroActions.put_ingredient_in_nearest_pot,
            (primitive_action == Actions.interact)
            | ~DynamicObject.is_ingredient(inventory)
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.POT),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.get_soup_from_nearest_pot,
            (primitive_action == Actions.interact)
            | (inventory != DynamicObject.PLATE)
            | ~jnp.any(state.grid[:, :, 0] == StaticObject.POT),
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
            (primitive_action == Actions.interact)
            | (inventory == DynamicObject.EMPTY)
            | ~jnp.any(counter_mask_static),
            done,
        )
        done = jnp.where(
            macro_action == MacroActions.pickup_from_nearest_counter,
            (primitive_action == Actions.interact)
            | (inventory != DynamicObject.EMPTY)
            | ~jnp.any(counter_mask_static),
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
        # `wait_for_nearest_pot` legitimately needs dynamic pot state to know when
        # to stop waiting, but only for pots the agent can actually see. Restrict
        # the read to the agent's observation window: keep waiting only while an
        # in-view pot is cooking-but-not-ready; finish when one becomes ready,
        # leaves the window, or there was none in view to begin with (a noop).
        in_view = self._in_view_mask(agent)
        ready_in_view = jnp.any(self._ready_recipe_pot_mask(state) & in_view)
        cooking_in_view = jnp.any(self._cooking_pot_cell_mask(state) & in_view)
        done = jnp.where(
            macro_action == MacroActions.wait_for_nearest_pot,
            ready_in_view | ~cooking_in_view,
            done,
        )
        # ...and a navigation macro also ends when it is stuck at a block: it has
        # walked as far as it can and cannot advance (`~macro_nav_ok`). This hands
        # control back to the policy instead of waiting for the block to clear.
        return done | ~macro_nav_ok

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

    def _barrier_agnostic_walkable_mask(self, state: State) -> chex.Array:
        """Return walkability treating every barrier tile as open.

        Identical to the base walkability of `_current_walkable_mask` but without
        subtracting currently-closed barriers, so the flood fill retains a
        gradient through timed barriers. Permanent walls stay non-walkable. It is
        purely static (no dynamic reads), so it is trivially JIT/vmap-safe.
        """
        static_layer = state.grid[:, :, 0]
        return (
            (static_layer == StaticObject.EMPTY)
            | (static_layer == StaticObject.PLAYER_CONVEYOR)
            | (static_layer == StaticObject.PRESSURE_PLATE)
            | (static_layer == StaticObject.BARRIER)
        )

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

    def _step_up_to_blocked_object(
        self,
        state: State,
        agent: Agent,
        walkable_mask: chex.Array,
        distances_all: chex.Array,
    ) -> chex.Array:
        """Step toward a transiently-blocked object, then wait at the block.

        Used only when no currently-open route to the target exists. The chosen
        direction is the neighbor with the smallest barrier-agnostic distance to
        the object, scored over ALL in-bounds cells (including closed barrier
        cells) so it points straight at the target. The agent actually moves only
        when that downhill neighbor is currently walkable, unoccupied, and
        strictly closer than its own cell; otherwise it stays. This walks the
        agent up to the blocking barrier and holds there until the barrier opens
        (at which point `has_path` becomes true and detour-aware movement resumes).
        """
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
        candidate_distances = jnp.where(
            candidate_in_bounds, distances_all[safe_y, safe_x], INF_DISTANCE
        )
        best_idx = jnp.argmin(candidate_distances)
        best_distance = candidate_distances[best_idx]
        best_walkable = walkable_mask[safe_y[best_idx], safe_x[best_idx]]
        best_unoccupied = self._cell_unoccupied_by_other_agents(
            state, agent, safe_y, safe_x
        )[best_idx]
        making_progress = best_distance < distances_all[ay, ax]
        do_move = (
            making_progress
            & best_walkable
            & best_unoccupied
            & (best_distance < INF_DISTANCE)
        )
        action = self._move_actions[best_idx]
        return jnp.where(do_move, action, Actions.stay).astype(jnp.int32)

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

    def _in_view_mask(self, agent: Agent) -> chex.Array:
        """Return a boolean grid of cells inside the agent's observation window.

        Mirrors the egocentric square window used to build partial observations
        (`overcooked_v3/observations.py`): a Chebyshev radius of
        ``self.agent_view_size`` centered on the agent. Under full observability
        (``agent_view_size`` is None) it returns all-True, so view-gated macros
        behave exactly as they did before partial observability. The None check
        is on a static attribute, so it is resolved at trace time.
        """
        if self.agent_view_size is None:
            return jnp.ones((self.height, self.width), dtype=jnp.bool_)
        ys = jnp.arange(self.height)[:, None]
        xs = jnp.arange(self.width)[None, :]
        within_rows = jnp.abs(ys - agent.pos.y) <= self.agent_view_size
        within_cols = jnp.abs(xs - agent.pos.x) <= self.agent_view_size
        return within_rows & within_cols

    def _cooking_pot_cell_mask(self, state: State) -> chex.Array:
        """Return a boolean grid marking pot cells that are currently cooking.

        Scatters each active pot's 'cooking' flag onto its grid cell (mirroring
        the pot-timer layer built in observations), so completion of
        `wait_for_nearest_pot` can be gated by what is inside the view window.
        """
        cooking = state.pot_active_mask & (state.pot_cooking_timer > 0)
        cooking_cells = jnp.zeros(
            (self.height, self.width), dtype=jnp.int32
        ).at[
            state.pot_positions[:, 0], state.pot_positions[:, 1]
        ].add(cooking.astype(jnp.int32))
        return cooking_cells > 0

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
        """Mask macros the agent knows it cannot start, without leaking state.

        Gates use only the agent's own inventory and STATIC object existence.
        They deliberately never read dynamic world state (pot contents, counter
        contents, cook timers), so the mask cannot reveal privileged, possibly
        out-of-view information through the action space. An action whose dynamic
        precondition turns out to be false is still available and simply becomes
        a no-op that terminates during execution.
        """
        static_layer = state.grid[:, :, 0]
        counter_mask = self._counter_like_static_mask(static_layer)
        pot_exists = jnp.any(static_layer == StaticObject.POT)

        def agent_mask(agent_idx):
            inventory = state.agents.inventory[agent_idx]
            inventory_empty = inventory == DynamicObject.EMPTY
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
            # Existence + inventory only; no dynamic pot/counter/timer reads.
            mask = mask.at[MacroActions.put_ingredient_in_nearest_pot].set(
                DynamicObject.is_ingredient(inventory) & pot_exists
            )
            mask = mask.at[MacroActions.get_soup_from_nearest_pot].set(
                (inventory == DynamicObject.PLATE) & pot_exists
            )
            mask = mask.at[MacroActions.deliver].set(
                ((inventory & DynamicObject.COOKED) != 0)
                & jnp.any(static_layer == StaticObject.GOAL)
            )
            mask = mask.at[MacroActions.drop_on_nearest_counter].set(
                ~inventory_empty & jnp.any(counter_mask)
            )
            mask = mask.at[MacroActions.pickup_from_nearest_counter].set(
                inventory_empty & jnp.any(counter_mask)
            )
            mask = mask.at[MacroActions.press_nearest_button].set(
                jnp.any(state.button_active_mask)
            )
            mask = mask.at[MacroActions.stand_on_nearest_pressure_plate].set(
                jnp.any(state.pressure_plate_active_mask)
            )
            mask = mask.at[MacroActions.wait_for_nearest_pot].set(pot_exists)
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
