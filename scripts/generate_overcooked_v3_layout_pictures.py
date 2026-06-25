#!/usr/bin/env python3
"""Generate preview PNGs for registered Overcooked V3 layouts."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import numpy as np
from PIL import Image

from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Overcooked V3 layout preview PNGs."
    )
    parser.add_argument(
        "layouts",
        nargs="*",
        help="Layout names to render. Defaults to registered layouts missing PNGs.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Render every registered Overcooked V3 layout.",
    )
    parser.add_argument(
        "--output-dir",
        default="layout_pictures",
        help="Directory where PNGs are written.",
    )
    parser.add_argument(
        "--tile-size",
        type=int,
        default=32,
        help="Tile size in pixels.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="PRNG seed used for environment reset.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PNGs.",
    )
    return parser.parse_args()


def select_layouts(args: argparse.Namespace, output_dir: Path) -> list[str]:
    registered = list(overcooked_v3_layouts)

    if args.layouts and args.all:
        raise SystemExit("Use either explicit layout names or --all, not both.")

    unknown = sorted(set(args.layouts) - set(registered))
    if unknown:
        raise SystemExit(f"Unknown Overcooked V3 layout(s): {', '.join(unknown)}")

    if args.layouts:
        return list(args.layouts)

    if args.all:
        return registered

    existing = {path.stem for path in output_dir.glob("*.png")}
    return [name for name in registered if name not in existing]


def render_layout(layout_name: str, output_path: Path, tile_size: int, seed: int) -> None:
    env = OvercookedV3(layout=layout_name)
    _, state = env.reset(jax.random.PRNGKey(seed))

    viz = OvercookedV3Visualizer(env, tile_size=tile_size)
    img = np.asarray(viz.render_state(state))
    Image.fromarray(img).save(output_path)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    layout_names = select_layouts(args, output_dir)
    if not layout_names:
        print("No layout pictures to generate.")
        return

    for layout_name in layout_names:
        output_path = output_dir / f"{layout_name}.png"
        if output_path.exists() and not args.overwrite:
            print(f"skip existing {output_path}")
            continue

        render_layout(layout_name, output_path, args.tile_size, args.seed)
        print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
