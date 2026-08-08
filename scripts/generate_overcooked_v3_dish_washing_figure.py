#!/usr/bin/env python3
"""Render the explanatory dish-washing figures for the Overcooked V3 docs.

Two figures:

* ``cycle``      - 3 panels on ``dish_washing_kitchen`` walking the plate through
                   clean stack -> delivery -> dirty pile -> sink.
* ``plate_loop`` - 2 panels on ``dish_washing_handoff`` contrasting a full stack
                   with a mid-service backlog.

Panels are assembled directly rather than played out over a full service, so
every panel is checked against the environment's plate-conservation invariant
(stack + dirty pile + carried + on-grid == num_plates). A panel that could not
occur during real play fails the assert instead of being drawn.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.environments.overcooked_v3.common import (
    DynamicObject,
    Direction,
    Position,
)
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

NUM_PLATES = 3
CLEAN_PLATE = int(DynamicObject.PLATE)
DIRTY_PLATE = int(DynamicObject.PLATE | DynamicObject.DIRTY)
DEFAULT_DIR = "docs/imgs/overcooked_v3_dish_washing"


def count_plates(state) -> int:
    """Mirror of the env invariant: stack + dirty pile + carried + on-grid."""
    held = int(jnp.sum(DynamicObject.counts_as_plate(state.agents.inventory)))
    on_grid = int(jnp.sum(DynamicObject.counts_as_plate(state.grid[:, :, 1])))
    return int(state.plate_stack_count) + int(state.dirty_pile_count) + held + on_grid


def make_state(base, *, stack, dirty, agents=()):
    """Derive a panel state; `agents` is (idx, (y, x), direction, inventory)."""
    pos, dirs, inv = base.agents.pos, base.agents.dir, base.agents.inventory
    ys, xs = pos.y, pos.x
    for idx, (y, x), facing, item in agents:
        ys, xs = ys.at[idx].set(y), xs.at[idx].set(x)
        dirs = dirs.at[idx].set(int(facing))
        inv = inv.at[idx].set(item)

    state = base.replace(
        agents=base.agents.replace(pos=Position(y=ys, x=xs), dir=dirs, inventory=inv),
        plate_stack_count=jnp.array(stack, dtype=jnp.int32),
        dirty_pile_count=jnp.array(dirty, dtype=jnp.int32),
    )
    total = count_plates(state)
    assert total == NUM_PLATES, f"panel holds {total} plates, expected {NUM_PLATES}"
    return state


def build_env(layout: str, seed: int):
    env = OvercookedV3(
        layout=layout,
        enable_dish_washing=True,
        num_plates=NUM_PLATES,
        random_reset=False,
    )
    _, state = env.reset(jax.random.PRNGKey(seed))
    return env, state


def figure_cycle(seed: int):
    """dish_washing_kitchen: sink at (1,6), dirty pile at (3,6), stack at (3,0)."""
    env, start = build_env("dish_washing_kitchen", seed)
    panels = [
        (make_state(start, stack=3, dirty=0),
         "1. start\nstack=3 clean, dirty=0"),
        (make_state(start, stack=1, dirty=2),
         "2. after 2 deliveries\nstack=1, dirty=2 waiting"),
        (make_state(start, stack=0, dirty=2,
                    agents=[(0, (1, 5), Direction.RIGHT, DIRTY_PLATE)]),
         "3. washing\nstack=0, agent carries a dirty plate to the sink"),
    ]
    title = ("The plate cycle - plates are conserved: "
             "stack + dirty pile + carried + on-grid = num_plates")
    return env, panels, title


def figure_plate_loop(seed: int):
    """dish_washing_handoff: sink at (3,6), dirty pile at (2,6), stack at (2,0)."""
    env, start = build_env("dish_washing_handoff", seed)
    panels = [
        (make_state(start, stack=3, dirty=0),
         "start: 3 clean plates, nothing dirty"),
        (make_state(start, stack=0, dirty=2,
                    agents=[(1, (3, 5), Direction.RIGHT, DIRTY_PLATE)]),
         "mid-service: stack empty, 2 dirty waiting,\n"
         "serving-side agent carrying one to the sink"),
    ]
    return env, panels, "the plate loop under load"


FIGURES = {
    "cycle": (figure_cycle, "dish_washing_cycle.png"),
    "plate_loop": (figure_plate_loop, "dish_washing_handoff_plate_loop.png"),
}


def render(builder, out_path: Path, tile_size: int, seed: int) -> None:
    env, panels, title = builder(seed)
    viz = OvercookedV3Visualizer(env, tile_size=tile_size)

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 5.2))
    for ax, (state, caption) in zip(np.atleast_1d(axes), panels):
        ax.imshow(np.asarray(viz.render_state(state)))
        ax.set_title(caption, fontsize=11)
        ax.axis("off")
    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("figures", nargs="*", choices=list(FIGURES) + [[]],
                       help="Figures to render (default: all).")
    parser.add_argument("--output-dir", default=DEFAULT_DIR)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    for name in args.figures or list(FIGURES):
        builder, filename = FIGURES[name]
        render(builder, output_dir / filename, args.tile_size, args.seed)


if __name__ == "__main__":
    main()
