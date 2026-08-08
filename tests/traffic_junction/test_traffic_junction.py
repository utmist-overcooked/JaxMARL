import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl.environments.traffic_junction.traffic_junction import State, TrafficJunction
from jaxmarl.environments.traffic_junction.demo_tj import step_for_demo


def make_state(position):
    return State(
        p_pos=jnp.array([position], dtype=jnp.int32),
        p_dir=jnp.zeros((1,), dtype=jnp.int32),
        path_idx=jnp.zeros((1,), dtype=jnp.int32),
        active=jnp.ones((1,), dtype=bool),
        path_type=jnp.zeros((1,), dtype=jnp.int32),
        step=0,
    )


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        (
            (3, 3),
            [
                [0, 0, 0],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ),
        (
            (0, 2),
            [
                [-1, -1, -1],
                [0, 1, 0],
                [0, 0, 0],
            ],
        ),
    ],
)
def test_local_observation_is_centered(position, expected):
    env = TrafficJunction(max_agents=1, view_size=3, grid_size=8)

    observation = env.get_obs(make_state(position))["car_0"].reshape((3, 3))

    np.testing.assert_array_equal(np.asarray(observation), np.asarray(expected))


@pytest.mark.parametrize(
    ("positions", "expected"),
    [
        ([[4, 5], [4, 4]], [[4, 6], [4, 5]]),
        ([[4, 4], [4, 5]], [[4, 5], [4, 6]]),
    ],
)
def test_adjacent_cars_advance_into_vacated_cell(positions, expected):
    env = TrafficJunction(max_agents=2, spawn_prob=0.0, grid_size=8)
    state = State(
        p_pos=jnp.array(positions, dtype=jnp.int32),
        p_dir=jnp.zeros((2,), dtype=jnp.int32),
        path_idx=jnp.ones((2,), dtype=jnp.int32),
        active=jnp.ones((2,), dtype=bool),
        path_type=jnp.zeros((2,), dtype=jnp.int32),
        step=0,
    )
    actions = {agent: jnp.int32(1) for agent in env.agents}

    _, next_state, rewards, _, info = env.step_env(
        jax.random.PRNGKey(0), state, actions
    )

    np.testing.assert_array_equal(np.asarray(next_state.p_pos), np.asarray(expected))
    assert all(int(info[agent]) == 0 for agent in env.agents)
    assert all(float(rewards[agent]) == pytest.approx(-0.01) for agent in env.agents)


def test_convoy_stays_put_when_leader_is_blocked():
    env = TrafficJunction(max_agents=3, spawn_prob=0.0, grid_size=8)
    state = State(
        p_pos=jnp.array([[4, 4], [4, 5], [4, 6]], dtype=jnp.int32),
        p_dir=jnp.zeros((3,), dtype=jnp.int32),
        path_idx=jnp.ones((3,), dtype=jnp.int32),
        active=jnp.ones((3,), dtype=bool),
        path_type=jnp.zeros((3,), dtype=jnp.int32),
        step=0,
    )
    actions = {
        "car_0": jnp.int32(1),
        "car_1": jnp.int32(1),
        "car_2": jnp.int32(0),
    }

    _, next_state, _, _, info = env.step_env(
        jax.random.PRNGKey(0), state, actions
    )

    np.testing.assert_array_equal(np.asarray(next_state.p_pos), np.asarray(state.p_pos))
    assert int(info["car_0"]) == 1
    assert int(info["car_1"]) == 1
    assert int(info["car_2"]) == 0


def test_demo_step_preserves_terminal_state():
    env = TrafficJunction(max_agents=1, spawn_prob=0.0, max_steps=1, grid_size=8)
    key = jax.random.PRNGKey(0)
    _, state = env.reset(key)
    actions = {"car_0": jnp.int32(0)}

    _, terminal_state, _, dones, _ = step_for_demo(env, key, state, actions)
    _, autoreset_state, _, _, _ = env.step(key, state, actions)

    assert bool(dones["__all__"])
    assert int(terminal_state.step) == 1
    assert int(autoreset_state.step) == 0
