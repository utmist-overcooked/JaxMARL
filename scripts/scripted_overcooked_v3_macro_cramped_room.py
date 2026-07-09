"""Scripted Overcooked V3 macro-action rollout for cramped_room.

Agent 0 repeatedly cooks and delivers onion soup using macro actions while
agent 1 waits. The generated GIF annotates each frame with the macro action
currently being executed by agent 0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaxmarl.environments.overcooked_v3_macro import MacroActions, OvercookedV3Macro
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


SOUP_PROGRAM = (
    MacroActions.get_ingredient_0,
    MacroActions.put_ingredient_in_nearest_pot,
    MacroActions.get_ingredient_0,
    MacroActions.put_ingredient_in_nearest_pot,
    MacroActions.get_ingredient_0,
    MacroActions.put_ingredient_in_nearest_pot,
    MacroActions.get_plate,
    MacroActions.wait_for_nearest_pot,
    MacroActions.get_soup_from_nearest_pot,
    MacroActions.deliver,
)


class SingleAgentSoupPolicy:
    """A tiny finite-state controller over macro actions."""

    def __init__(self):
        self.program_index = 0

    @property
    def current_action(self) -> MacroActions:
        return SOUP_PROGRAM[self.program_index]

    def observe(self, state) -> None:
        if bool(jax.device_get(state.macro_action_done[0])):
            self.program_index = (self.program_index + 1) % len(SOUP_PROGRAM)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overcooked_v3_macro_cramped_room_scripted.gif"),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--max-macro-steps", type=int, default=80)
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--frame-ms", type=int, default=120)
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Write every Nth environment frame to the GIF.",
    )
    return parser.parse_args()


def rollout(env: OvercookedV3Macro, seed: int):
    policy = SingleAgentSoupPolicy()
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)

    @jax.jit
    def step_fn(step_key, step_state, action0):
        return env.step_env(
            step_key,
            step_state,
            {
                "agent_0": action0,
                "agent_1": jnp.array(MacroActions.wait, dtype=jnp.int32),
            },
        )

    states = [state]
    macro_labels = ["reset"]
    delivery_counts = [0]
    deliveries = 0

    for _ in range(env.max_steps):
        action = policy.current_action
        key, step_key = jax.random.split(key)
        _, state, rewards, dones, info = step_fn(
            step_key, state, jnp.array(action, dtype=jnp.int32)
        )

        macro_idx = int(jax.device_get(info["current_macro_action"]["agent_0"]))
        deliveries += int(float(jax.device_get(rewards["agent_0"])) > 0.0)
        macro_labels.append(env.macro_action_names[macro_idx])
        delivery_counts.append(deliveries)
        states.append(state)
        policy.observe(state)

        if bool(jax.device_get(dones["__all__"])):
            break

    return states, macro_labels, delivery_counts, deliveries


def draw_header(
    frame: np.ndarray, label: str, step: int, deliveries: int
) -> Image.Image:
    image = Image.fromarray(frame)
    header_h = 58
    canvas = Image.new("RGB", (image.width, image.height + header_h), (24, 24, 28))
    canvas.paste(image.convert("RGB"), (0, header_h))

    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (10, 8),
        f"t={step:03d} | deliveries={deliveries}",
        fill=(235, 235, 235),
        font=font,
    )
    draw.text(
        (10, 30),
        f"agent_0 macro: {label}",
        fill=(255, 226, 120),
        font=font,
    )
    return canvas


def save_gif(
    env: OvercookedV3Macro,
    states,
    macro_labels,
    delivery_counts,
    output: Path,
    tile_size: int,
    frame_ms: int,
    frame_skip: int,
) -> None:
    visualizer = OvercookedV3Visualizer(env, tile_size=tile_size)
    frames = []

    for step, (state, label, deliveries) in enumerate(
        zip(states, macro_labels, delivery_counts)
    ):
        if step % frame_skip != 0 and step != len(states) - 1:
            continue
        frame = np.asarray(jax.device_get(visualizer.render_state(state)))
        frames.append(draw_header(frame, label, step, deliveries))

    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=frame_ms,
        loop=0,
        optimize=True,
    )


def main() -> None:
    args = parse_args()
    env = OvercookedV3Macro(
        layout="cramped_room",
        max_steps=args.max_steps,
        max_macro_steps=args.max_macro_steps,
    )

    states, macro_labels, delivery_counts, deliveries = rollout(env, args.seed)
    save_gif(
        env,
        states,
        macro_labels,
        delivery_counts,
        args.output,
        args.tile_size,
        args.frame_ms,
        max(args.frame_skip, 1),
    )

    print(f"Wrote {args.output}")
    print(f"Frames: {len(states)}")
    print(f"Deliveries: {deliveries}")
    print(f"Final time: {int(jax.device_get(states[-1].time))}")


if __name__ == "__main__":
    main()
