"""Tests for the Overcooked V3 dish washing mechanic.

With dish washing enabled the kitchen owns a finite number of plates:
    plate pile --take--> clean plate --serve--> delivered
    delivered  --------> dirty pile --take--> dirty plate --sink--> clean plate

Plates are conserved: the clean stack, the dirty pile, every inventory and every
plate lying on the grid always sum to num_plates.
"""

import jax
import jax.numpy as jnp
import pytest

from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.environments.overcooked_v3.settings import SHAPED_REWARDS

DIRTY_PLATE = int(DynamicObject.PLATE | DynamicObject.DIRTY)
CLEAN_PLATE = int(DynamicObject.PLATE)

DISH_LAYOUTS = [
    "dish_washing_room",
    "dish_washing_kitchen",
    "dish_washing_handoff",
    "prep_dish_kitchen",
]

# dish_washing_room geometry (x, y):
#   WWPWW      pot (2,0)
#   0A AS      onion (0,1)  sink (4,1)
#   W   D      dirty pile (4,2)
#   WBWXW      plate pile (1,3)  goal (3,3)
POT_XY = (2, 0)
SINK_XY = (4, 1)
DIRTY_XY = (4, 2)
PLATE_PILE_XY = (1, 3)
GOAL_XY = (3, 3)


def _make(**kwargs):
    kwargs.setdefault("layout", "dish_washing_room")
    kwargs.setdefault("enable_dish_washing", True)
    kwargs.setdefault("num_plates", 3)
    return OvercookedV3(**kwargs)


def _place(state, x, y, direction, inventory=None):
    """Put agent 0 at (x, y) facing `direction`; park agent 1 out of the way."""
    agents = state.agents
    pos = Position(
        x=agents.pos.x.at[0].set(x).at[1].set(1),
        y=agents.pos.y.at[0].set(y).at[1].set(1),
    )
    inv = agents.inventory if inventory is None else agents.inventory.at[0].set(inventory)
    return state.replace(
        agents=agents.replace(pos=pos, dir=agents.dir.at[0].set(direction), inventory=inv)
    )


def _interact(env, state, key):
    key, subkey = jax.random.split(key)
    actions = {"agent_0": jnp.array(Actions.interact), "agent_1": jnp.array(Actions.stay)}
    obs, state, rewards, dones, info = env.step(subkey, state, actions)
    return state, rewards, info, key


def _total_plates(state):
    """Every plate in the kitchen, in any form or location."""
    held = jnp.sum(DynamicObject.counts_as_plate(state.agents.inventory))
    on_grid = jnp.sum(DynamicObject.counts_as_plate(state.grid[:, :, 1]))
    return int(state.plate_stack_count + state.dirty_pile_count + held + on_grid)


class TestDishWashingToggle:
    def test_disabled_by_default(self):
        env = OvercookedV3(layout="dish_washing_room")
        assert not env.enable_dish_washing

    def test_disabled_keeps_original_observation_schema(self):
        """A dish layout with washing off must match an equivalent plain kitchen."""
        off = OvercookedV3(layout="dish_washing_room")
        plain = OvercookedV3(layout="cramped_room")
        assert off.obs_shape == plain.obs_shape == (4, 5, 35)

    def test_disabled_turns_sink_and_pile_into_counters(self):
        """No sink and no dirty pile exist in frame when the toggle is off."""
        env = OvercookedV3(layout="dish_washing_room")
        obs, state = env.reset(jax.random.PRNGKey(0))
        statics = state.grid[:, :, 0]
        assert jnp.sum(statics == StaticObject.SINK) == 0
        assert jnp.sum(statics == StaticObject.DIRTY_PLATE_PILE) == 0
        # They became ordinary counters rather than vanishing
        assert statics[SINK_XY[1], SINK_XY[0]] == StaticObject.WALL
        assert statics[DIRTY_XY[1], DIRTY_XY[0]] == StaticObject.WALL

    def test_disabled_does_not_mutate_shared_layout(self):
        """Layouts are shared module-level objects; the remap must copy."""
        OvercookedV3(layout="dish_washing_room")
        statics = overcooked_v3_layouts["dish_washing_room"].static_objects
        assert (statics == StaticObject.SINK).sum() == 1
        assert (statics == StaticObject.DIRTY_PLATE_PILE).sum() == 1

    def test_disabled_plate_pile_is_infinite(self):
        env = OvercookedV3(layout="dish_washing_room")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        for _ in range(5):
            state = _place(state, *PLATE_PILE_XY[:1], PLATE_PILE_XY[1] - 1, Direction.DOWN, inventory=0)
            state, rewards, info, key = _interact(env, state, key)
            assert state.agents.inventory[0] == CLEAN_PLATE
            assert state.plate_stack_count == 0  # unused when disabled

    def test_enabled_adds_seven_layers(self):
        off = OvercookedV3(layout="dish_washing_room")
        on = _make()
        assert on.obs_shape[2] == off.obs_shape[2] + 7

    def test_enabled_requires_sink_and_dirty_pile(self):
        with pytest.raises(ValueError, match="sink"):
            OvercookedV3(layout="cramped_room", enable_dish_washing=True)

    def test_rejects_zero_plates(self):
        with pytest.raises(ValueError, match="num_plates"):
            _make(num_plates=0)

    @pytest.mark.parametrize("layout", DISH_LAYOUTS)
    def test_dish_layouts_build_both_ways(self, layout):
        OvercookedV3(layout=layout)
        env = _make(layout=layout)
        obs, state = env.reset(jax.random.PRNGKey(0))
        assert state.plate_stack_count == env.num_plates
        assert state.dirty_pile_count == 0


class TestDishWashingCycle:
    def test_initial_counts(self):
        env = _make(num_plates=3)
        obs, state = env.reset(jax.random.PRNGKey(0))
        assert state.plate_stack_count == 3
        assert state.dirty_pile_count == 0
        assert _total_plates(state) == 3

    def test_taking_a_plate_shrinks_the_stack(self):
        env = _make(num_plates=3)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state = _place(state, PLATE_PILE_XY[0], PLATE_PILE_XY[1] - 1, Direction.DOWN, inventory=0)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == CLEAN_PLATE
        assert state.plate_stack_count == 2
        assert _total_plates(state) == 3

    def test_empty_stack_yields_nothing(self):
        env = _make(num_plates=1)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state = _place(state, PLATE_PILE_XY[0], PLATE_PILE_XY[1] - 1, Direction.DOWN, inventory=0)
        state, rewards, info, key = _interact(env, state, key)
        assert state.plate_stack_count == 0

        # Stack is empty: a second empty-handed interact must hand back nothing
        state = _place(state, PLATE_PILE_XY[0], PLATE_PILE_XY[1] - 1, Direction.DOWN, inventory=0)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == 0
        assert state.plate_stack_count == 0

    def test_delivery_sends_plate_to_dirty_pile(self):
        env = _make(num_plates=3)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        soup = int(state.recipe) | int(DynamicObject.PLATE) | int(DynamicObject.COOKED)
        state = _place(state, GOAL_XY[0], GOAL_XY[1] - 1, Direction.DOWN, inventory=soup)
        # The handed-in dish carries a plate, so account for it leaving the stack
        # or the conservation check would see a fourth plate appear from nowhere.
        state = state.replace(plate_stack_count=state.plate_stack_count - 1)
        state, rewards, info, key = _interact(env, state, key)

        assert rewards["agent_0"] == env.delivery_reward
        assert state.dirty_pile_count == 1
        assert state.agents.inventory[0] == 0
        assert _total_plates(state) == 3

    def test_full_wash_cycle_conserves_plates(self):
        env = _make(num_plates=3)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # take a plate
        state = _place(state, PLATE_PILE_XY[0], PLATE_PILE_XY[1] - 1, Direction.DOWN, inventory=0)
        state, rewards, info, key = _interact(env, state, key)
        assert _total_plates(state) == 3

        # deliver a plated soup
        soup = int(state.recipe) | int(DynamicObject.PLATE) | int(DynamicObject.COOKED)
        state = _place(state, GOAL_XY[0], GOAL_XY[1] - 1, Direction.DOWN, inventory=soup)
        state, rewards, info, key = _interact(env, state, key)
        assert state.dirty_pile_count == 1
        assert _total_plates(state) == 3

        # collect the dirty plate
        state = _place(state, DIRTY_XY[0] - 1, DIRTY_XY[1], Direction.RIGHT, inventory=0)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == DIRTY_PLATE
        assert DynamicObject.is_dirty_plate(state.agents.inventory[0])
        assert state.dirty_pile_count == 0
        assert info["event/dirty_pickup"][0] == 1
        assert _total_plates(state) == 3

        # wash it
        state = _place(state, SINK_XY[0] - 1, SINK_XY[1], Direction.RIGHT)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == CLEAN_PLATE
        assert info["event/plate_wash"][0] == 1
        assert _total_plates(state) == 3

        # stack it back
        state = _place(state, PLATE_PILE_XY[0], PLATE_PILE_XY[1] - 1, Direction.DOWN)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == 0
        assert state.plate_stack_count == 3
        assert info["event/plate_return"][0] == 1
        assert _total_plates(state) == 3

    def test_empty_dirty_pile_yields_nothing(self):
        env = _make()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state = _place(state, DIRTY_XY[0] - 1, DIRTY_XY[1], Direction.RIGHT, inventory=0)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == 0
        assert state.dirty_pile_count == 0

    def test_sink_does_nothing_without_a_dirty_plate(self):
        env = _make()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state = _place(state, SINK_XY[0] - 1, SINK_XY[1], Direction.RIGHT, inventory=CLEAN_PLATE)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == CLEAN_PLATE
        assert info["event/plate_wash"][0] == 0

    def test_washing_is_not_farmable(self):
        """Repeated sink interacts pay once; re-dirtying requires a delivery."""
        env = _make()
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state = _place(state, SINK_XY[0] - 1, SINK_XY[1], Direction.RIGHT, inventory=DIRTY_PLATE)

        state, rewards, info, key = _interact(env, state, key)
        assert info["shaped_reward"]["agent_0"] == SHAPED_REWARDS["PLATE_WASH"]
        for _ in range(5):
            state, rewards, info, key = _interact(env, state, key)
            assert info["shaped_reward"]["agent_0"] == 0.0


class TestDirtyPlateIsUnusable:
    def test_dirty_plate_cannot_serve_soup(self):
        env = _make(pot_cook_time=2, pot_burn_time=0)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Cook a full pot
        pot_x, pot_y = POT_XY
        full_pot = int(DynamicObject.ingredient(0)) * 3
        state = state.replace(
            grid=state.grid.at[pot_y, pot_x, 1].set(full_pot),
            pot_cooking_timer=state.pot_cooking_timer.at[0].set(2),
        )
        for _ in range(3):
            state = _place(state, pot_x, pot_y + 1, Direction.UP, inventory=0)
            state, rewards, info, key = _interact(env, state, key)
        assert (state.grid[pot_y, pot_x, 1] & DynamicObject.COOKED) != 0

        # A dirty plate must not scoop it
        state = _place(state, pot_x, pot_y + 1, Direction.UP, inventory=DIRTY_PLATE)
        state, rewards, info, key = _interact(env, state, key)
        assert state.agents.inventory[0] == DIRTY_PLATE
        assert state.grid[pot_y, pot_x, 1] != 0
        assert info["event/dish_pickup"][0] == 0

    def test_dirty_plate_is_not_an_ingredient(self):
        assert not DynamicObject.is_ingredient(jnp.array(DIRTY_PLATE))
        assert DynamicObject.is_dirty_plate(jnp.array(DIRTY_PLATE))
        assert not DynamicObject.is_dirty_plate(jnp.array(CLEAN_PLATE))
        assert DynamicObject.counts_as_plate(jnp.array(DIRTY_PLATE))


class TestDishWashingRollout:
    @pytest.mark.parametrize("layout", DISH_LAYOUTS)
    def test_random_rollout_conserves_plates(self, layout):
        """Plate count is invariant under arbitrary play, including drops."""
        env = _make(layout=layout, num_plates=3)
        key = jax.random.PRNGKey(7)
        obs, state = env.reset(key)
        step = jax.jit(env.step_env)
        for _ in range(120):
            key, k_act, k_step = jax.random.split(key, 3)
            actions = {
                agent: jax.random.randint(jax.random.fold_in(k_act, i), (), 0, 6)
                for i, agent in enumerate(env.agents)
            }
            obs, state, rewards, dones, info = step(k_step, state, actions)
            assert _total_plates(state) == 3
            assert state.plate_stack_count >= 0
            assert state.dirty_pile_count >= 0

    def test_jit_and_vmap(self):
        env = _make()
        keys = jax.random.split(jax.random.PRNGKey(0), 4)
        obs, states = jax.vmap(env.reset)(keys)
        actions = {a: jnp.zeros((4,), dtype=jnp.int32) for a in env.agents}
        obs, states, rewards, dones, info = jax.vmap(
            jax.jit(env.step_env)
        )(keys, states, actions)
        assert states.plate_stack_count.shape == (4,)
