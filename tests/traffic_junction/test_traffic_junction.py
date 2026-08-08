import jax.numpy as jnp
import numpy as np
import pytest

from jaxmarl.environments.traffic_junction.traffic_junction import State, TrafficJunction


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
