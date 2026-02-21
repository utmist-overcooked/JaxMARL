"""Tests for pressure plate controls in Overcooked V3."""

import jax
import textwrap
from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import ButtonAction
from jaxmarl.environments.overcooked_v3.layouts import Layout


class TestPressurePlate:
    """Verify pressure plates trigger linked actions on occupancy changes."""

    def test_pressure_plate_toggles_barrier_on_enter_and_exit(self):
        layout_str = textwrap.dedent(
            """
            WWWWW
            W A W
            W _#W
            W 0XW
            WWWWW
            """
        )

        layout = Layout.from_string(
            layout_str,
            possible_recipes=[[0, 0, 0]],
            barrier_config=[False],
            button_config=[(0, ButtonAction.TOGGLE_BARRIER)],
        )

        env = make("overcooked_v3", layout=layout, enable_buttons=True)

        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key)

        assert not state.barrier_active[0], "Barrier should start inactive"

        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, {"agent_0": 1})
        assert state.barrier_active[0], "Stepping onto pressure plate should toggle barrier on"

        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, {"agent_0": 3})
        assert not state.barrier_active[0], "Stepping off pressure plate should toggle barrier off"
