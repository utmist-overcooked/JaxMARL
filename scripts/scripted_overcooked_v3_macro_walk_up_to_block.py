"""Scripted demos of "walk up to a block and try to interact" macro behavior.

Each scenario drives a single macro-action agent and renders a GIF (with the
flood-fill planner panel) to make the new behavior visible:

1. ``empty_pot`` -- ``get_soup_from_nearest_pot`` on an empty pot. Navigation
   targets pots by static existence, so the agent walks up to the pot, attempts
   the interaction once, discovers there is no soup, and the macro terminates.

2. ``barrier_closes`` -- a barrier that is OPEN when the macro starts and CLOSES
   partway through execution. The agent walks up to the barrier and, once it is
   stuck there, the macro ends and hands control back to the policy, which
   switches to a different macro. It walks up to the block, then gives up rather
   than waiting for the barrier to reopen.

Reuses the flood-fill panel / GIF machinery from
``scripted_overcooked_v3_macro_cramped_room``.

Examples::

    python scripts/scripted_overcooked_v3_macro_walk_up_to_block.py --scenario empty_pot
    python scripts/scripted_overcooked_v3_macro_walk_up_to_block.py --scenario barrier_closes
    python scripts/scripted_overcooked_v3_macro_walk_up_to_block.py --scenario all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp

# Reuse the rendering helpers from the sibling scripted demo.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from scripted_overcooked_v3_macro_cramped_room import save_gif  # noqa: E402

from jaxmarl.environments.overcooked_v3.common import DynamicObject
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.environments.overcooked_v3_macro import MacroActions, OvercookedV3Macro


def _make_env(rows, barrier_config=None, max_macro_steps=40):
    """Build a one-agent macro env from grid rows plus a plate/pot/goal header."""
    width = len(rows[0])
    service_objects = "WBPX" + "W" * (width - 4)
    service_access = "W   " + "W" * (width - 4)
    layout = Layout.from_string(
        "\n".join([service_objects, service_access, *rows]),
        possible_recipes=[[0, 0, 0]],
        barrier_config=barrier_config,
    )
    return OvercookedV3Macro(layout=layout, max_macro_steps=max_macro_steps)


def _step_fn(env):
    """Return a jitted single-agent step (agent 1 always waits)."""

    @jax.jit
    def step_fn(step_key, step_state, action0):
        """Advance one macro tick for agent 0 while agent 1 waits."""
        return env.step_env(
            step_key,
            step_state,
            {
                "agent_0": action0,
                "agent_1": jnp.array(MacroActions.wait, dtype=jnp.int32),
            },
        )

    return step_fn


def empty_pot_pickup_rollout(seed: int = 0):
    """Run get_soup_from_nearest_pot against an empty pot in cramped_room."""
    env = OvercookedV3Macro(layout="cramped_room", max_macro_steps=20)
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)

    # Hand agent 0 a plate; the pot is empty at reset, so the pickup will fail.
    state = state.replace(
        agents=state.agents.replace(
            inventory=state.agents.inventory.at[0].set(DynamicObject.PLATE)
        )
    )
    action = MacroActions.get_soup_from_nearest_pot
    step_fn = _step_fn(env)

    states = [state]
    labels = [action]
    counts = [0]
    for _ in range(env.max_macro_steps):
        key, step_key = jax.random.split(key)
        _, state, _, dones, _ = step_fn(
            step_key, state, jnp.array(action, dtype=jnp.int32)
        )
        states.append(state)
        labels.append(action)
        counts.append(0)
        if bool(jax.device_get(state.macro_action_done[0])) or bool(
            jax.device_get(dones["__all__"])
        ):
            break
    return env, states, labels, counts


def barrier_closes_rollout(seed: int = 0, close_at: int = 3):
    """Open a barrier at macro start, close it mid-approach; agent gives up.

    The ingredient pile is walled off except through a single barrier tile. The
    barrier is open when ``get_ingredient_0`` starts and closes mid-approach and
    stays closed. The agent walks up to the barrier and, once stuck, the macro
    ends and hands control back to a tiny policy, which then switches to a
    different macro (``get_plate``) — showing that control returned rather than
    the agent waiting at the barrier forever.
    """
    rows = [
        "W    WWW",
        "WA   #0W",
        "W    WWW",
        "WWWWWWWW",
    ]
    env = _make_env(rows, barrier_config=[False], max_macro_steps=40)
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)
    # Freeze the barrier timer so its open/closed state is ours to control.
    state = state.replace(barrier_timer=jnp.zeros_like(state.barrier_timer))
    action = MacroActions.get_ingredient_0
    step_fn = _step_fn(env)

    states = [state]
    labels = [action]
    counts = [0]
    for t in range(30):
        # Barrier open before close_at, then closed and staying closed. Hold the
        # timer at 0 so the planner sees exactly the grid we intend.
        closed = jnp.array(t >= close_at, dtype=state.barrier_active.dtype)
        state = state.replace(
            barrier_active=jnp.full_like(state.barrier_active, closed),
            barrier_timer=jnp.zeros_like(state.barrier_timer),
        )
        key, step_key = jax.random.split(key)
        _, state, _, dones, _ = step_fn(
            step_key, state, jnp.array(action, dtype=jnp.int32)
        )
        states.append(state)
        labels.append(action)  # the macro that produced this state
        counts.append(0)

        macro_done = bool(jax.device_get(state.macro_action_done[0]))
        holding_plate = bool(
            jax.device_get(state.agents.inventory[0] == DynamicObject.PLATE)
        )
        empty_handed = bool(
            jax.device_get(state.agents.inventory[0] == DynamicObject.EMPTY)
        )
        # Policy hand-back: the ingredient macro gave up at the block (done, still
        # empty-handed) -> the policy chooses a different macro instead.
        if (
            action == MacroActions.get_ingredient_0
            and macro_done
            and empty_handed
        ):
            action = MacroActions.get_plate
        # Stop once the fallback macro succeeds or the episode ends.
        if holding_plate or bool(jax.device_get(dones["__all__"])):
            break
    return env, states, labels, counts


SCENARIOS = {
    "empty_pot": (
        empty_pot_pickup_rollout,
        "overcooked_v3_macro_walk_up_empty_pot.gif",
    ),
    "barrier_closes": (
        barrier_closes_rollout,
        "overcooked_v3_macro_walk_up_barrier_closes.gif",
    ),
}


def parse_args() -> argparse.Namespace:
    """Parse command-line options for the walk-up-to-block demos."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS.keys(), "all"],
        default="all",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--frame-ms", type=int, default=350)
    return parser.parse_args()


def main() -> None:
    """Render the requested walk-up-to-block scenario GIFs."""
    args = parse_args()
    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        rollout_fn, filename = SCENARIOS[name]
        env, states, labels, counts = rollout_fn(args.seed)
        output = args.output_dir / filename
        save_gif(
            env,
            states,
            labels,
            counts,
            output,
            args.tile_size,
            args.frame_ms,
            1,
            True,
            None,
        )
        print(f"[{name}] wrote {output} ({len(states)} frames)")


if __name__ == "__main__":
    main()
