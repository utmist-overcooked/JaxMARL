"""Render the out-of-order queue delivery behavior as a comparison GIF."""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    Direction,
    DynamicObject,
    Position,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3.layouts import Layout
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


BACKGROUND = "#111827"
PANEL = "#1f2937"
TEXT = "#f9fafb"
MUTED = "#cbd5e1"
ONION = "#facc15"
TOMATO = "#ef4444"
SUCCESS = "#22c55e"


def parse_args() -> argparse.Namespace:
    """Parse output and rendering options for the scripted comparison."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/overcooked_v3_out_of_order_delivery.gif"),
    )
    parser.add_argument("--tile-size", type=int, default=64)
    parser.add_argument("--frame-ms", type=int, default=1400)
    return parser.parse_args()


def build_environment() -> OvercookedV3:
    """Create a compact two-recipe kitchen for the delivery demonstration."""
    layout = Layout.from_string(
        """
WWWWWWWW
W0A1P RW
WB X A W
WWWWWWWW
""",
        possible_recipes=[[0, 0, 0], [1, 1, 1]],
    )
    return OvercookedV3(
        layout=layout,
        recipe_mode="alternating",
        enable_order_queue=True,
        max_orders=4,
        order_generation_rate=0.0,
        order_expiration_time=100,
    )


def put_tomato_delivery_at_goal(state):
    """Place agent 0 beside the goal holding a cooked tomato dish."""
    static = np.asarray(state.grid[:, :, 0])
    goal_y, goal_x = np.argwhere(static == int(StaticObject.GOAL))[0]
    candidates = (
        (goal_y, goal_x - 1, Direction.RIGHT),
        (goal_y, goal_x + 1, Direction.LEFT),
        (goal_y - 1, goal_x, Direction.DOWN),
        (goal_y + 1, goal_x, Direction.UP),
    )
    for stand_y, stand_x, direction in candidates:
        if (
            0 <= stand_y < static.shape[0]
            and 0 <= stand_x < static.shape[1]
            and static[stand_y, stand_x] == int(StaticObject.EMPTY)
        ):
            tomato_recipe = DynamicObject.get_recipe_encoding(
                jnp.array([1, 1, 1], dtype=jnp.int32)
            )
            plated_tomato = (
                tomato_recipe | DynamicObject.PLATE | DynamicObject.COOKED
            )
            agents = state.agents.replace(
                pos=Position(
                    x=state.agents.pos.x.at[0].set(stand_x),
                    y=state.agents.pos.y.at[0].set(stand_y),
                ),
                dir=state.agents.dir.at[0].set(direction),
                inventory=state.agents.inventory.at[0].set(plated_tomato),
            )
            return state.replace(agents=agents)
    raise ValueError("Demo layout has no empty tile adjacent to its goal")


def prepare_demo_state(env: OvercookedV3):
    """Seed onion, old tomato, and new tomato orders with distinct timers."""
    _, state = env.reset(jax.random.PRNGKey(0))
    onion_recipe = DynamicObject.get_recipe_encoding(
        jnp.array([0, 0, 0], dtype=jnp.int32)
    )
    state = state.replace(
        recipe=onion_recipe,
        order_types=jnp.array([1, 2, 2, 0], dtype=jnp.int32),
        order_expirations=jnp.array([30, 60, 90, 0], dtype=jnp.int32),
        order_active_mask=jnp.array([True, True, True, False]),
    )
    return put_tomato_delivery_at_goal(state)


def run_scripted_delivery(env: OvercookedV3, state):
    """Have agent 0 deliver tomato while agent 1 waits."""
    actions = {
        "agent_0": jnp.array(Actions.interact, dtype=jnp.int32),
        "agent_1": jnp.array(Actions.stay, dtype=jnp.int32),
    }
    _, new_state, rewards, _, _ = env.step_env(
        jax.random.PRNGKey(1),
        state,
        actions,
    )
    assert new_state.order_types.tolist() == [1, 2, 0, 0]
    assert new_state.order_expirations.tolist() == [29, 89, 0, 0]
    assert float(rewards["agent_0"]) == env.delivery_reward
    return new_state


def simulate_previous_front_only_result(state):
    """Build the prior front-only outcome for visual comparison."""
    agents = state.agents.replace(
        inventory=state.agents.inventory.at[0].set(DynamicObject.EMPTY)
    )
    return state.replace(
        agents=agents,
        order_expirations=jnp.array([29, 59, 89, 0], dtype=jnp.int32),
        time=state.time + 1,
        new_correct_delivery=False,
        new_correct_delivery_types=jnp.zeros_like(
            state.new_correct_delivery_types
        ),
    )


def load_fonts() -> tuple[ImageFont.ImageFont, ImageFont.ImageFont]:
    """Load readable fonts with a Pillow-default fallback."""
    try:
        return (
            ImageFont.truetype("DejaVuSans-Bold.ttf", 25),
            ImageFont.truetype("DejaVuSans.ttf", 19),
        )
    except OSError:
        default = ImageFont.load_default()
        return default, default


def draw_queue(
    draw: ImageDraw.ImageDraw,
    origin: tuple[int, int],
    order_types: list[int],
    expirations: list[int],
    body_font: ImageFont.ImageFont,
) -> None:
    """Draw active order cards in oldest-to-newest order."""
    x, y = origin
    for slot, (order_type, expiration) in enumerate(
        zip(order_types, expirations)
    ):
        if order_type == 0:
            continue
        name = "Onion" if order_type == 1 else "Tomato"
        color = ONION if order_type == 1 else TOMATO
        width = 126
        draw.rounded_rectangle(
            (x, y, x + width, y + 48),
            radius=10,
            fill=color,
        )
        draw.text(
            (x + 10, y + 5),
            f"{slot + 1}. {name}",
            font=body_font,
            fill="#111827",
        )
        draw.text(
            (x + 10, y + 26),
            f"timer {expiration}",
            font=body_font,
            fill="#111827",
        )
        x += width + 10


def render_comparison_frame(
    visualizer: OvercookedV3Visualizer,
    left_state,
    right_state,
    *,
    before: bool,
) -> Image.Image:
    """Compose prior and new behavior panels for one animation frame."""
    title_font, body_font = load_fonts()
    left_game = Image.fromarray(
        np.asarray(jax.device_get(visualizer.render_state(left_state)))
    )
    right_game = Image.fromarray(
        np.asarray(jax.device_get(visualizer.render_state(right_state)))
    )
    panel_width = max(left_game.width, 560)
    panel_height = left_game.height + 180
    canvas = Image.new(
        "RGB",
        (panel_width * 2 + 36, panel_height + 128),
        BACKGROUND,
    )
    draw = ImageDraw.Draw(canvas)
    heading = (
        "Tomato is delivered while Onion remains the oldest order"
        if before
        else "One action, two different queue outcomes"
    )
    draw.text((24, 20), heading, font=title_font, fill=TEXT)

    panels = (
        (18, "Previous: front-only", left_state, left_game),
        (panel_width + 36, "New: oldest matching", right_state, right_game),
    )
    for panel_x, label, state, game in panels:
        panel_y = 64
        draw.rounded_rectangle(
            (panel_x, panel_y, panel_x + panel_width, panel_y + panel_height),
            radius=16,
            fill=PANEL,
        )
        draw.text((panel_x + 16, panel_y + 14), label, font=title_font, fill=TEXT)
        game_x = panel_x + (panel_width - game.width) // 2
        canvas.paste(game, (game_x, panel_y + 52))
        queue_y = panel_y + 62 + game.height
        draw.text(
            (panel_x + 16, queue_y),
            "Queue (oldest first)",
            font=body_font,
            fill=MUTED,
        )
        draw_queue(
            draw,
            (panel_x + 16, queue_y + 28),
            state.order_types.tolist(),
            state.order_expirations.tolist(),
            body_font,
        )

    if before:
        footer = "Scripted action: agent 0 INTERACTS with a cooked Tomato dish"
        footer_color = MUTED
    else:
        footer = (
            "Previous: reward 0, queue unchanged    |    "
            "New: reward +20, oldest Tomato removed; newer Tomato remains"
        )
        footer_color = SUCCESS
    draw.text(
        (24, panel_height + 84),
        footer,
        font=body_font,
        fill=footer_color,
    )
    return canvas


def main() -> None:
    """Run the policy and save the before/after comparison GIF."""
    args = parse_args()
    env = build_environment()
    before_state = prepare_demo_state(env)
    after_state = run_scripted_delivery(env, before_state)
    previous_state = simulate_previous_front_only_result(before_state)
    visualizer = OvercookedV3Visualizer(env, tile_size=args.tile_size)

    before_frame = render_comparison_frame(
        visualizer,
        before_state,
        before_state,
        before=True,
    )
    after_frame = render_comparison_frame(
        visualizer,
        previous_state,
        after_state,
        before=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    before_frame.save(
        args.output,
        save_all=True,
        append_images=[after_frame],
        duration=[args.frame_ms, args.frame_ms * 2],
        loop=0,
        disposal=2,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
