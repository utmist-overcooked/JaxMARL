"""Tests for Overcooked V3 prep stations (cutting board, grill, blender).

Prep chains:
    lettuce (2) -> cutting board -> chopped lettuce (5)
    meat (3)    -> grill         -> grilled meat (6)
    carrot (4)  -> blender       -> carrot puree (7)
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
from jaxmarl.environments.overcooked_v3.settings import BURN_PENALTY

RAW_LETTUCE = int(DynamicObject.ingredient(2))
RAW_MEAT = int(DynamicObject.ingredient(3))
RAW_CARROT = int(DynamicObject.ingredient(4))
CHOPPED_LETTUCE = int(DynamicObject.ingredient(5))
GRILLED_MEAT = int(DynamicObject.ingredient(6))
CARROT_PUREE = int(DynamicObject.ingredient(7))

# Station tile position shared by the three *_room layouts:
#   WWPWW
#   \dA AS   (\d = ingredient pile, S = station)
#   W   W
#   WBWXW
STATION_YX = (1, 4)


def _step(env, state, key, a0=Actions.stay, a1=Actions.stay):
    actions = {"agent_0": jnp.array(a0), "agent_1": jnp.array(a1)}
    key, subkey = jax.random.split(key)
    obs, state, rewards, dones, info = env.step(subkey, state, actions)
    return state, rewards, info, key


def _place_on_station(env, state, key):
    """Drive agent 0 from its start to place the raw ingredient on the station.

    Works for all three *_room layouts (identical geometry).
    """
    script = [
        (Actions.stay, Actions.down),   # agent 1 clears the tile next to the station
        (Actions.left, Actions.stay),   # agent 0 faces the ingredient pile
        (Actions.interact, Actions.stay),  # pick up raw ingredient
        (Actions.right, Actions.stay),
        (Actions.right, Actions.stay),
        (Actions.right, Actions.stay),  # blocked by station tile; turns to face it
        (Actions.interact, Actions.stay),  # place on station
    ]
    for a0, a1 in script:
        state, rewards, info, key = _step(env, state, key, a0, a1)
    return state, info, key


class TestPrepLayouts:
    def test_new_layouts_registered(self):
        for name in ["cutting_board_room", "grill_room", "blender_room", "prep_kitchen"]:
            assert name in overcooked_v3_layouts
            # Constructor runs validate_playable
            env = OvercookedV3(layout=name)
            assert env.has_prep_stations

    def test_num_ingredients_covers_processed_types(self):
        assert overcooked_v3_layouts["cutting_board_room"].num_ingredients == 6
        assert overcooked_v3_layouts["grill_room"].num_ingredients == 7
        assert overcooked_v3_layouts["blender_room"].num_ingredients == 8
        assert overcooked_v3_layouts["prep_kitchen"].num_ingredients == 8

    def test_station_tiles_parsed(self):
        layout = overcooked_v3_layouts["prep_kitchen"]
        statics = layout.static_objects
        assert (statics == StaticObject.CUTTING_BOARD).sum() == 1
        assert (statics == StaticObject.GRILL).sum() == 1
        assert (statics == StaticObject.BLENDER).sum() == 1

    def test_to_string_roundtrip(self):
        layout = overcooked_v3_layouts["prep_kitchen"]
        reparsed = Layout.from_string(
            layout.to_string(), possible_recipes=layout.possible_recipes
        )
        assert (reparsed.static_objects == layout.static_objects).all()
        assert reparsed.num_ingredients == layout.num_ingredients

    def test_processed_recipe_requires_station(self):
        # Recipe needs chopped lettuce but there is no cutting board
        no_station = """
WWPWW
2A AW
W   W
WBWXW
"""
        layout = Layout.from_string(no_station, possible_recipes=[[5, 5, 5]])
        is_playable, messages = layout.validate_playable()
        assert not is_playable
        assert any("CUTTING_BOARD" in m for m in messages)
        with pytest.raises(ValueError):
            OvercookedV3(layout=layout)

    def test_processed_recipe_requires_raw_pile(self):
        # Cutting board exists but the raw lettuce pile is missing
        no_pile = """
WWPWW
0A AC
W   W
WBWXW
"""
        layout = Layout.from_string(no_pile, possible_recipes=[[5, 5, 5]])
        is_playable, messages = layout.validate_playable()
        assert not is_playable
        assert any("missing ingredient piles: [2]" in m for m in messages)

    def test_layouts_without_stations_unchanged(self):
        env = OvercookedV3(layout="cramped_room")
        assert not env.has_prep_stations
        # Observation schema must stay identical to pre-station builds
        assert env.obs_shape == (4, 5, 35)


class TestCuttingBoard:
    def test_chop_cycle(self):
        env = OvercookedV3(layout="cutting_board_room")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        assert state.grid[y, x, 1] == RAW_LETTUCE
        assert state.grid[y, x, 2] == 0
        assert info["event/prep_placement"][0] == 1

        # Chop chop_stages - 1 times: raw item, progress advances
        for expected_progress in range(1, env.chop_stages):
            state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
            assert state.grid[y, x, 1] == RAW_LETTUCE
            assert state.grid[y, x, 2] == expected_progress
            assert info["event/prep_action"][0] == 1

        # Final chop converts the item and resets progress
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert state.grid[y, x, 1] == CHOPPED_LETTUCE
        assert state.grid[y, x, 2] == 0

        # Empty-handed interact now picks up the processed item
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert state.agents.inventory[0] == CHOPPED_LETTUCE
        assert state.grid[y, x, 1] == 0
        assert info["event/prep_pickup"][0] == 1

    def test_board_does_not_tick_over_time(self):
        env = OvercookedV3(layout="cutting_board_room")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        for _ in range(10):
            state, rewards, info, key = _step(env, state, key)
        assert state.grid[y, x, 1] == RAW_LETTUCE
        assert state.grid[y, x, 2] == 0

    def test_cannot_place_while_station_occupied(self):
        env = OvercookedV3(layout="cutting_board_room")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        # Give the agent another raw lettuce and try to place again
        state = state.replace(
            agents=state.agents.replace(
                inventory=state.agents.inventory.at[0].set(RAW_LETTUCE)
            )
        )
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        y, x = STATION_YX
        assert state.grid[y, x, 1] == RAW_LETTUCE  # unchanged, single unit
        assert state.agents.inventory[0] == RAW_LETTUCE  # still holding


class TestGrill:
    def test_grill_cooks_automatically(self):
        env = OvercookedV3(layout="grill_room", grill_cook_time=5, grill_burn_time=4)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        # Placement step already ticked the timer once
        assert state.grid[y, x, 2] == 5 + 4 - 1
        assert state.grid[y, x, 1] == RAW_MEAT

        # 4 more steps: still cooking
        for _ in range(4):
            state, rewards, info, key = _step(env, state, key)
        assert state.grid[y, x, 1] == GRILLED_MEAT

        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert state.agents.inventory[0] == GRILLED_MEAT
        assert state.grid[y, x, 2] == 0
        assert info["event/prep_pickup"][0] == 1

    def test_early_pickup_returns_raw(self):
        env = OvercookedV3(layout="grill_room", grill_cook_time=10, grill_burn_time=5)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        y, x = STATION_YX
        assert state.agents.inventory[0] == RAW_MEAT
        assert state.grid[y, x, 1] == 0
        assert state.grid[y, x, 2] == 0  # timer reset on pickup

    def test_grill_burns_item(self):
        env = OvercookedV3(layout="grill_room", grill_cook_time=3, grill_burn_time=2)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        # Timer is 4 after placement; run it down to 1
        for _ in range(3):
            state, rewards, info, key = _step(env, state, key)
        assert state.grid[y, x, 1] == GRILLED_MEAT
        assert state.grid[y, x, 2] == 1

        state, rewards, info, key = _step(env, state, key)
        assert state.grid[y, x, 1] == 0  # destroyed
        assert rewards["agent_0"] == BURN_PENALTY
        assert info["event/prep_burn"][0] == 1

    def test_grill_burn_disabled(self):
        env = OvercookedV3(layout="grill_room", grill_cook_time=3, grill_burn_time=0)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        for _ in range(20):
            state, rewards, info, key = _step(env, state, key)
        assert state.grid[y, x, 1] == GRILLED_MEAT  # ready forever
        assert info["event/prep_burn"][0] == 0


class TestBlender:
    def test_blender_requires_manual_start(self):
        env = OvercookedV3(layout="blender_room", blend_time=4)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        for _ in range(10):
            state, rewards, info, key = _step(env, state, key)
        # No auto-start: still raw, no timer
        assert state.grid[y, x, 1] == RAW_CARROT
        assert state.grid[y, x, 2] == 0

    def test_blend_cycle(self):
        env = OvercookedV3(layout="blender_room", blend_time=4)
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)
        state, info, key = _place_on_station(env, state, key)

        y, x = STATION_YX
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert info["event/prep_action"][0] == 1
        assert state.grid[y, x, 2] == 3  # started at 4, ticked once

        for _ in range(3):
            assert state.grid[y, x, 1] == RAW_CARROT
            state, rewards, info, key = _step(env, state, key)
        assert state.grid[y, x, 1] == CARROT_PUREE
        assert state.grid[y, x, 2] == 0

        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert state.agents.inventory[0] == CARROT_PUREE


class TestProcessedIngredientsInPot:
    def _move_agent(self, state, agent_idx, x, y, direction):
        agents = state.agents
        new_pos = Position(
            x=agents.pos.x.at[agent_idx].set(x),
            y=agents.pos.y.at[agent_idx].set(y),
        )
        return state.replace(
            agents=agents.replace(
                pos=new_pos, dir=agents.dir.at[agent_idx].set(direction)
            )
        )

    def _set_inventory(self, state, agent_idx, value):
        return state.replace(
            agents=state.agents.replace(
                inventory=state.agents.inventory.at[agent_idx].set(value)
            )
        )

    def test_pot_accepts_and_cooks_processed_ingredients(self):
        env = OvercookedV3(
            layout="cutting_board_room", pot_cook_time=5, pot_burn_time=0
        )
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Agent 0 below the pot (pot at y=0, x=2), facing up, holding chopped
        # lettuce; drop three units to trigger auto-cook.
        state = self._move_agent(state, 0, x=2, y=1, direction=Direction.UP)
        for i in range(3):
            state = self._set_inventory(state, 0, CHOPPED_LETTUCE)
            state, rewards, info, key = _step(env, state, key, a0=Actions.interact)

        pot_y, pot_x = state.pot_positions[0]
        assert DynamicObject.ingredient_count(state.grid[pot_y, pot_x, 1]) == 3
        assert state.pot_cooking_timer[0] > 0

        for _ in range(5):
            state, rewards, info, key = _step(env, state, key)
        assert (state.grid[pot_y, pot_x, 1] & DynamicObject.COOKED) != 0

        # Plate the soup and deliver it (goal at y=3, x=3)
        state = self._set_inventory(state, 0, DynamicObject.PLATE)
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        expected_dish = (
            CHOPPED_LETTUCE * 3 | DynamicObject.PLATE | DynamicObject.COOKED
        )
        assert state.agents.inventory[0] == expected_dish

        state = self._move_agent(state, 0, x=3, y=2, direction=Direction.DOWN)
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert rewards["agent_0"] == env.delivery_reward
        assert info["event/delivery"][0] == 1


class TestPrepObservations:
    def test_prep_layout_obs_shape(self):
        env = OvercookedV3(layout="cutting_board_room")
        # 30 + 5 * 6 ingredients + 3 station layers + 1 progress layer
        assert env.obs_shape == (4, 5, 64)
        obs, state = env.reset(jax.random.PRNGKey(0))
        assert obs["agent_0"].shape == env.obs_shape

    def test_station_layers_present(self):
        env = OvercookedV3(layout="prep_kitchen")
        obs, state = env.reset(jax.random.PRNGKey(0))
        agent_obs = obs["agent_0"]
        # Static layers start after the two agent blocks (2 * (7 + num_ing))
        num_ing = env.layout.num_ingredients
        static_start = 2 * (7 + num_ing)
        # The three station layers are the last of the 14 static layers
        board_layer = agent_obs[:, :, static_start + 11]
        grill_layer = agent_obs[:, :, static_start + 12]
        blender_layer = agent_obs[:, :, static_start + 13]
        statics = state.grid[:, :, 0]
        assert (board_layer == (statics == StaticObject.CUTTING_BOARD)).all()
        assert (grill_layer == (statics == StaticObject.GRILL)).all()
        assert (blender_layer == (statics == StaticObject.BLENDER)).all()

    def test_random_rollout_completes(self):
        env = OvercookedV3(layout="prep_kitchen")
        key = jax.random.PRNGKey(42)
        obs, state = env.reset(key)
        step = jax.jit(env.step_env)
        for _ in range(50):
            key, k_act, k_step = jax.random.split(key, 3)
            actions = {
                agent: jax.random.randint(jax.random.fold_in(k_act, i), (), 0, 6)
                for i, agent in enumerate(env.agents)
            }
            obs, state, rewards, dones, info = step(k_step, state, actions)
        assert state.time == 50


class TestHandoffLayouts:
    HANDOFF_LAYOUTS = [
        "cutting_board_handoff",
        "grill_handoff",
        "blender_handoff",
        "prep_kitchen_handoff",
    ]

    def test_registered_and_valid(self):
        for name in self.HANDOFF_LAYOUTS:
            assert name in overcooked_v3_layouts
            env = OvercookedV3(layout=name)
            assert env.has_prep_stations

    def test_agents_in_separate_regions(self):
        """The counter obstacle fully separates prep and cook sides."""
        for name in self.HANDOFF_LAYOUTS:
            env = OvercookedV3(layout=name)
            (x0, y0), (x1, y1) = env.layout.agent_positions
            assert env.enclosed_spaces[y0, x0] != env.enclosed_spaces[y1, x1], name

    def test_counter_handoff(self):
        """Prep agent passes a processed item over the middle counter."""
        env = OvercookedV3(layout="cutting_board_handoff")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Agent 0 (prep side, at x=1) faces the middle counter column (x=2)
        # holding chopped lettuce and drops it there.
        agents = state.agents
        state = state.replace(
            agents=agents.replace(
                dir=agents.dir.at[0].set(Direction.RIGHT),
                inventory=agents.inventory.at[0].set(CHOPPED_LETTUCE),
            )
        )
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        counter_y, counter_x = 2, 2
        assert state.grid[counter_y, counter_x, 1] == CHOPPED_LETTUCE
        assert state.agents.inventory[0] == 0

        # Agent 1 (cook side, at x=3) faces the counter and picks it up.
        agents = state.agents
        state = state.replace(
            agents=agents.replace(dir=agents.dir.at[1].set(Direction.LEFT))
        )
        state, rewards, info, key = _step(env, state, key, a1=Actions.interact)
        assert state.agents.inventory[1] == CHOPPED_LETTUCE
        assert state.grid[counter_y, counter_x, 1] == 0


class TestPrepShapedRewards:
    def test_placement_and_chop_shaped_rewards(self):
        env = OvercookedV3(layout="cutting_board_room")
        key = jax.random.PRNGKey(0)
        obs, state = env.reset(key)

        # Recipe is [[5,5,5]] so the chain is always needed at the start
        script = [
            (Actions.stay, Actions.down),
            (Actions.left, Actions.stay),
        ]
        for a0, a1 in script:
            state, rewards, info, key = _step(env, state, key, a0, a1)

        # Raw lettuce pile pickup is shaped (demand comes from processed form)
        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert info["shaped_reward"]["agent_0"] > 0

        for a0 in [Actions.right, Actions.right, Actions.right]:
            state, rewards, info, key = _step(env, state, key, a0=a0)

        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert info["shaped_reward"]["agent_0"] > 0  # placement

        state, rewards, info, key = _step(env, state, key, a0=Actions.interact)
        assert info["shaped_reward"]["agent_0"] > 0  # chop
