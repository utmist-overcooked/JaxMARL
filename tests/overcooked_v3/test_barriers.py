"""Tests for the barrier system in Overcooked V3."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import Actions


class TestBarriers:
    """Test that barriers block movement when active and allow it when inactive."""

    def _make_env_and_reset(self, layout, **kwargs):
        """Helper: create barrier env with given layout and reset it."""
        env = make("overcooked_v3", layout=layout, **kwargs)
        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        return env, key, obs, state

    def test_inactive_barrier_allows_movement(self):
        """Agent can move through an inactive barrier."""
        env, key, obs, state = self._make_env_and_reset("barrier_demo")

        # Explicitly set barriers to inactive
        state = state.replace(
            barrier_active=jnp.zeros_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Move agent_0 right to the cell adjacent to the barrier, then through it.
        actions_right = {"agent_0": int(Actions.right), "agent_1": int(Actions.stay)}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        key, step_key = jax.random.split(key)
        _, new_state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert new_state.agents.pos.x[0] == 3
        assert new_state.agents.pos.y[0] == 1

    def test_active_barrier_blocks_movement(self):
        """Agent cannot move through an active barrier."""
        env, key, obs, state = self._make_env_and_reset("barrier_demo")

        # Activate all barriers
        state = state.replace(
            barrier_active=jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Move agent_0 right twice to get adjacent to barrier, then try to cross
        actions_right = {"agent_0": 0, "agent_1": 4}
        for _ in range(3):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step_env(
                step_key, state, actions_right
            )

        # Record position and try one more move into the barrier
        pos_before_x = state.agents.pos.x[0]
        pos_before_y = state.agents.pos.y[0]

        key, step_key = jax.random.split(key)
        obs, state, rewards, dones, info = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] == pos_before_x, (
            "Agent x-position should not change when blocked by active barrier"
        )
        assert state.agents.pos.y[0] == pos_before_y, (
            "Agent y-position should not change when blocked by active barrier"
        )

    def test_deactivated_barrier_allows_movement(self):
        """After deactivating a barrier, agent can move through it."""
        env, key, obs, state = self._make_env_and_reset("barrier_demo")

        # Activate barriers, then deactivate them
        state = state.replace(
            barrier_active=jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
        )
        state = state.replace(
            barrier_active=jnp.zeros_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Move agent_0 right toward barrier position
        actions_right = {"agent_0": 0, "agent_1": 4}
        positions = []
        for _ in range(5):
            key, step_key = jax.random.split(key)
            obs, state, rewards, dones, info = env.step_env(
                step_key, state, actions_right
            )
            positions.append(int(state.agents.pos.x[0]))

        # Agent should have moved at least once (not stuck)
        assert len(set(positions)) > 1, (
            "Agent should be able to move through deactivated barrier"
        )


class TestTimedBarriers:
    """Test timed barrier functionality with button interaction."""

    BARRIER_DURATION = 5

    def _make_env_and_reset(self):
        """Helper: create timed barrier env and reset it."""
        env = make(
            "overcooked_v3",
            layout="timed_barrier_demo",
            barrier_duration=self.BARRIER_DURATION,
        )
        key = jax.random.PRNGKey(0)
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        return env, key, obs, state

    def _navigate_to_button_and_press(self, env, key, state):
        """Helper: move agent to button and press it. Returns updated key and state."""
        # Move toward barrier first
        actions_right = {"agent_0": 0, "agent_1": 4}
        for _ in range(2):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        # Move down to button
        actions_down = {"agent_0": 1, "agent_1": 4}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_down)

        # Press button
        actions_interact = {"agent_0": 5, "agent_1": 4}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_interact)

        return key, state

    def test_active_barrier_blocks(self):
        """Active timed barrier blocks movement."""
        env, key, obs, state = self._make_env_and_reset()

        # Explicitly set barrier to active
        state = state.replace(
            barrier_active=jnp.ones_like(state.barrier_active, dtype=jnp.bool_)
        )

        # Try moving toward barrier
        actions_right = {"agent_0": 0, "agent_1": 4}
        for _ in range(2):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        pos_before_x = state.agents.pos.x[0]
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] == pos_before_x, (
            "Agent should be blocked by active timed barrier"
        )

    def test_button_deactivates_barrier_and_sets_timer(self):
        """Pressing button deactivates barrier and starts countdown timer."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        assert not state.barrier_active[0], (
            "Barrier should be inactive after button press"
        )

        expected_timer = int(state.barrier_duration[0]) - 1
        assert state.barrier_timer[0] == expected_timer, (
            f"Timer should be barrier_duration - 1 = {expected_timer} "
            f"(decremented on same step), got {state.barrier_timer[0]}"
        )

    def test_deactivated_barrier_allows_movement(self):
        """Agent can move through a deactivated timed barrier."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        pos_before_x = state.agents.pos.x[0]
        actions_right = {"agent_0": 0, "agent_1": 4}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] != pos_before_x, (
            "Agent should move through deactivated barrier"
        )

    def test_barrier_reactivates_after_timer_expires(self):
        """Barrier reactivates once the countdown timer reaches zero."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        assert not state.barrier_active[0], "Barrier should be inactive after press"

        # Step until timer expires
        steps_to_simulate = int(state.barrier_timer[0])
        actions_stay = {"agent_0": 4, "agent_1": 4}
        for _ in range(steps_to_simulate):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_stay)

        assert state.barrier_active[0], "Barrier should reactivate after timer expires"
        assert state.barrier_timer[0] == 0, "Timer should be 0 after expiration"

    def test_reactivated_barrier_blocks(self):
        """After reactivation, barrier blocks movement again."""
        env, key, obs, state = self._make_env_and_reset()

        key, state = self._navigate_to_button_and_press(env, key, state)

        # Wait for timer to expire
        steps_to_simulate = int(state.barrier_timer[0])
        actions_stay = {"agent_0": 4, "agent_1": 4}
        for _ in range(steps_to_simulate):
            key, step_key = jax.random.split(key)
            _, state, _, _, _ = env.step_env(step_key, state, actions_stay)

        assert state.barrier_active[0], "Barrier should have reactivated"

        # Agent is adjacent to the reactivated barrier; trying to enter it is blocked.
        pos_before_x = state.agents.pos.x[0]
        pos_before_y = state.agents.pos.y[0]
        actions_right = {"agent_0": int(Actions.right), "agent_1": int(Actions.stay)}
        key, step_key = jax.random.split(key)
        _, state, _, _, _ = env.step_env(step_key, state, actions_right)

        assert state.agents.pos.x[0] == pos_before_x, (
            "Agent should be blocked by reactivated barrier"
        )
        assert state.agents.pos.y[0] == pos_before_y, (
            "Agent should be blocked by reactivated barrier"
        )


class TestPressurePlates:
    """Pressure plates (TOGGLE_BARRIER) open linked barriers while an agent
    stands on them and re-close them the moment every agent steps off."""

    def _make(self, layout="pressure_plate_demo", **kwargs):
        env = make("overcooked_v3", layout=layout, **kwargs)
        obs, state = env.reset(jax.random.PRNGKey(0))
        return env, state

    def _place(self, state, idx, y, x):
        """Teleport agent `idx` to (y, x) without going through movement/collision."""
        pos = state.agents.pos
        pos = pos.replace(x=pos.x.at[idx].set(x), y=pos.y.at[idx].set(y))
        return state.replace(agents=state.agents.replace(pos=pos))

    def test_walking_onto_plate_opens_then_off_recloses(self):
        """End-to-end: agent_0 starts adjacent to plate 0, which is linked to
        barrier 1. Stepping on opens it; stepping off re-closes it."""
        env, state = self._make("pressure_plate_demo")

        # Plate 0 links to barrier 1 (see pressure_plate_config in layouts.py).
        assert bool(state.barrier_active[1]), "barrier 1 starts closed"

        # agent_0 starts at (y=1, x=4); plate 0 is the floor cell to its left.
        _, state, _, _, _ = env.step_env(
            jax.random.PRNGKey(1), state, {"agent_0": int(Actions.left), "agent_1": 4}
        )
        assert bool(state.pressure_plate_toggled[0]), "plate 0 should read as pressed"
        assert not bool(state.barrier_active[1]), "barrier 1 should open while pressed"

        # Step back off the plate.
        _, state, _, _, _ = env.step_env(
            jax.random.PRNGKey(2), state, {"agent_0": int(Actions.right), "agent_1": 4}
        )
        assert not bool(state.pressure_plate_toggled[0]), "plate 0 no longer pressed"
        assert bool(state.barrier_active[1]), "barrier 1 should re-close on release"

    def test_plate_press_and_release_via_processor(self):
        """Unit test of _process_pressure_plates: press opens, release closes."""
        env, state = self._make("pressure_plate_demo")
        home = (int(state.agents.pos.y[0]), int(state.agents.pos.x[0]))
        py, px = (int(v) for v in np.array(state.pressure_plate_positions[0]))
        linked = np.array(state.pressure_plate_linked_barrier[0])

        pressed = env._process_pressure_plates(self._place(state, 0, py, px))
        assert not np.array(pressed.barrier_active)[linked].any(), "linked barrier opens"

        released = env._process_pressure_plates(self._place(pressed, 0, *home))
        mask = np.array(released.barrier_active_mask)
        assert np.array(released.barrier_active)[linked & mask].all(), (
            "linked barrier re-closes once the agent steps off"
        )

    def test_multi_target_plate_opens_all_linked_barriers(self):
        """A single plate linked to several barriers opens all of them at once,
        and leaves unlinked barriers untouched."""
        env, state = self._make("pressure_gated_zones")
        py, px = (int(v) for v in np.array(state.pressure_plate_positions[0]))
        linked = np.array(state.pressure_plate_linked_barrier[0])
        mask = np.array(state.barrier_active_mask)
        assert linked.sum() > 1, "this layout's plate 0 should target multiple barriers"

        # Park both agents on plate 0 so it is the only plate pressed.
        state = self._place(state, 0, py, px)
        if state.agents.pos.x.shape[0] > 1:
            state = self._place(state, 1, py, px)
        state = env._process_pressure_plates(state)

        active = np.array(state.barrier_active)
        assert not active[linked].any(), "all linked barriers open"
        assert active[mask & ~linked].all(), "unlinked barriers stay closed"

    def test_shared_barrier_stays_open_until_all_plates_released(self):
        """A barrier targeted by two plates stays open while either is pressed,
        and only re-closes once both are released."""
        env, state = self._make("pressure_gated_zones")
        l0 = np.array(state.pressure_plate_linked_barrier[0])
        l1 = np.array(state.pressure_plate_linked_barrier[1])
        shared = l0 & l1
        only0 = l0 & ~l1
        assert shared.any() and only0.any(), "layout must have shared + plate0-only barriers"

        p0 = tuple(int(v) for v in np.array(state.pressure_plate_positions[0]))
        p1 = tuple(int(v) for v in np.array(state.pressure_plate_positions[1]))

        # Both plates pressed.
        state = self._place(state, 0, *p0)
        state = self._place(state, 1, *p1)
        state = env._process_pressure_plates(state)
        assert not np.array(state.barrier_active)[l0 | l1].any(), "all open while both pressed"

        # Release plate 0 (move agent_0 onto plate 1's cell). Plate 1 still held.
        state = self._place(state, 0, *p1)
        state = env._process_pressure_plates(state)
        active = np.array(state.barrier_active)
        assert not active[shared].any(), "shared barriers stay open while plate 1 held"
        assert active[only0].all(), "barriers unique to plate 0 re-close on its release"

    def test_plates_inert_when_disabled(self):
        """With pressure plates disabled, standing on a plate does nothing."""
        with pytest.warns(UserWarning, match="Pressure plates will be inert"):
            env, state = self._make("pressure_plate_demo", enable_pressure_plates=False)
        py, px = (int(v) for v in np.array(state.pressure_plate_positions[0]))
        before = np.array(state.barrier_active).copy()

        state = env._process_pressure_plates(self._place(state, 0, py, px))
        assert np.array_equal(np.array(state.barrier_active), before), (
            "disabled plates must not change barrier state"
        )

    def test_toggle_barrier_does_not_close_on_occupying_agent(self):
        """A TOGGLE barrier stays open while an agent stands on it, even with the
        plate released, so an agent crossing as the plate releases can't be
        trapped inside an active barrier (Codex P2 #1)."""
        env, state = self._make("pressure_plate_demo")
        linked = np.array(state.pressure_plate_linked_barrier[0])
        bidx = int(np.flatnonzero(linked)[0])
        by, bx = (int(v) for v in np.array(state.barrier_positions[bidx]))
        assert bool(state.barrier_active[bidx]), "linked barrier starts closed"

        # Agent on the barrier tile, no agent on the plate (plate released):
        # the barrier must NOT re-close on top of the agent.
        occupied = self._place(state, 0, by, bx)
        occupied = self._place(occupied, 1, by, bx)
        result = env._process_pressure_plates(occupied)
        assert not bool(result.barrier_active[bidx]), (
            "barrier must stay open while an agent occupies its tile"
        )

        # Once the tile is clear (and no plate pressed), it re-closes.
        cleared = self._place(result, 0, 0, 0)
        cleared = self._place(cleared, 1, 0, 0)
        result2 = env._process_pressure_plates(cleared)
        assert bool(result2.barrier_active[bidx]), (
            "barrier re-closes once the occupying agent steps off"
        )

    def test_timed_barrier_reactivation_deferred_while_occupied(self):
        """A timed barrier about to expire does not reactivate onto an agent; the
        timer is held until the tile clears, then it reactivates."""
        env, state = self._make("pressure_plate_demo")
        bidx = int(np.flatnonzero(np.array(state.barrier_active_mask))[0])
        by, bx = (int(v) for v in np.array(state.barrier_positions[bidx]))

        # Arm barrier as open with its timer one step from expiry.
        state = state.replace(
            barrier_active=state.barrier_active.at[bidx].set(False),
            barrier_timer=state.barrier_timer.at[bidx].set(1),
        )

        # Agent on the barrier tile -> reactivation deferred, timer held at 1.
        occupied = self._place(state, 0, by, bx)
        occupied = self._place(occupied, 1, 0, 0)
        held = env._process_barrier_timers(occupied)
        assert not bool(held.barrier_active[bidx]), "deferred while occupied"
        assert int(held.barrier_timer[bidx]) == 1, "timer held at 1 while occupied"

        # Clear the tile -> reactivates and timer ticks to 0.
        cleared = self._place(held, 0, 0, 0)
        done = env._process_barrier_timers(cleared)
        assert bool(done.barrier_active[bidx]), "reactivates once the tile is clear"
        assert int(done.barrier_timer[bidx]) == 0, "timer reaches 0 after reactivation"
