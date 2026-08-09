"""Scripted Overcooked V3 macro-action rollout with a flood-fill view.

Agent 0 repeatedly cooks and delivers onion soup using macro actions while
agent 1 waits. The default layout is cramped_room; other compatible two-agent
layouts can be selected to inspect barrier-aware navigation.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from jaxmarl.environments.overcooked_v3.common import (
    Actions,
    DynamicObject,
    StaticObject,
)
from jaxmarl.environments.overcooked_v3_macro import (
    PRESSURE_PLATE_MACROS,
    MacroActions,
    OvercookedV3Macro,
)
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
    parser.add_argument("--layout", default="cramped_room")
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
    parser.add_argument(
        "--flood-fill",
        action="store_true",
        help="Add a synchronized panel showing macro goals and flood distances.",
    )
    parser.add_argument(
        "--cooperative-barrier-demo",
        action="store_true",
        help=(
            "Use timed_barrier_demo: agent 1 opens the plate gate for agent 0."
        ),
    )
    parser.add_argument(
        "--barrier-duration",
        type=int,
        default=30,
        help="Open-gate duration used by the cooperative barrier demo.",
    )
    return parser.parse_args()


def rollout(env: OvercookedV3Macro, seed: int):
    """Run the scripted soup policy and retain states aligned to the next macro."""
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
    macro_actions = [policy.current_action]
    delivery_counts = [0]
    deliveries = 0

    for _ in range(env.max_steps):
        action = policy.current_action
        key, step_key = jax.random.split(key)
        _, state, rewards, dones, info = step_fn(
            step_key, state, jnp.array(action, dtype=jnp.int32)
        )

        deliveries += int(float(jax.device_get(rewards["agent_0"])) > 0.0)
        delivery_counts.append(deliveries)
        states.append(state)
        policy.observe(state)
        macro_actions.append(policy.current_action)

        if bool(jax.device_get(dones["__all__"])):
            break

    return states, macro_actions, delivery_counts, deliveries


def cooperative_barrier_rollout(env: OvercookedV3Macro, seed: int):
    """Show agent 1 opening a timed gate so agent 0 can reach the plate."""
    key = jax.random.PRNGKey(seed)
    _, state = env.reset(key)
    agent_0_action = MacroActions.get_plate
    agent_1_action = MacroActions.press_nearest_button

    @jax.jit
    def step_fn(step_key, step_state, action0, action1):
        """Advance both cooperative demo agents by one compiled macro tick."""
        return env.step_env(
            step_key,
            step_state,
            {"agent_0": action0, "agent_1": action1},
        )

    states = [state]
    agent_0_actions = [agent_0_action]
    agent_1_actions = [agent_1_action]
    delivery_counts = [0]

    for _ in range(env.max_steps):
        key, step_key = jax.random.split(key)
        _, state, _, dones, info = step_fn(
            step_key,
            state,
            jnp.array(agent_0_action, dtype=jnp.int32),
            jnp.array(agent_1_action, dtype=jnp.int32),
        )

        if (
            agent_1_action == MacroActions.press_nearest_button
            and bool(jax.device_get(info["macro_action_done"]["agent_1"]))
        ):
            agent_1_action = MacroActions.wait
        if bool(
            jax.device_get(state.agents.inventory[0] == DynamicObject.PLATE)
        ):
            agent_0_action = MacroActions.wait

        states.append(state)
        agent_0_actions.append(agent_0_action)
        agent_1_actions.append(agent_1_action)
        delivery_counts.append(0)

        if (
            agent_0_action == MacroActions.wait
            or bool(jax.device_get(dones["__all__"]))
        ):
            break

    return states, agent_0_actions, agent_1_actions, delivery_counts, 0


def navigation_snapshot(
    env: OvercookedV3Macro, state, macro_action: MacroActions, agent_idx: int = 0
):
    """Return host-side target, goal, distance, and action data for one macro."""
    static_layer = state.grid[:, :, 0]
    dynamic_layer = state.grid[:, :, 1]
    agent = env._agent_at(state, jnp.array(agent_idx, dtype=jnp.int32))
    walkable = env._current_walkable_mask(state)

    target_mask = jnp.zeros((env.height, env.width), dtype=jnp.bool_)
    if macro_action == MacroActions.get_ingredient_0:
        target_mask = static_layer == StaticObject.ingredient_pile(0)
    elif macro_action == MacroActions.get_ingredient_1:
        target_mask = static_layer == StaticObject.ingredient_pile(1)
    elif macro_action == MacroActions.get_ingredient_2:
        target_mask = static_layer == StaticObject.ingredient_pile(2)
    elif macro_action == MacroActions.get_plate:
        target_mask = static_layer == StaticObject.PLATE_PILE
    elif macro_action == MacroActions.put_ingredient_in_nearest_pot:
        target_mask = env._valid_pot_placement_mask(state, agent.inventory)
    elif macro_action == MacroActions.get_soup_from_nearest_pot:
        target_mask = env._ready_recipe_pot_mask(state)
    elif macro_action == MacroActions.deliver:
        target_mask = static_layer == StaticObject.GOAL
    elif macro_action == MacroActions.drop_on_nearest_counter:
        target_mask = env._counter_like_static_mask(static_layer) & (
            dynamic_layer == DynamicObject.EMPTY
        )
    elif macro_action == MacroActions.pickup_from_nearest_counter:
        target_mask = env._counter_like_static_mask(static_layer) & (
            dynamic_layer != DynamicObject.EMPTY
        )
    elif macro_action == MacroActions.press_nearest_button:
        target_mask = static_layer == StaticObject.BUTTON

    interaction_macro = (
        MacroActions.get_ingredient_0
        <= macro_action
        <= MacroActions.press_nearest_button
    )
    pressure_plate_macro = macro_action in PRESSURE_PLATE_MACROS
    navigation_macro = interaction_macro or pressure_plate_macro

    interaction_goals = (
        jnp.pad(target_mask[:-1, :], ((1, 0), (0, 0)))
        | jnp.pad(target_mask[1:, :], ((0, 1), (0, 0)))
        | jnp.pad(target_mask[:, :-1], ((0, 0), (1, 0)))
        | jnp.pad(target_mask[:, 1:], ((0, 0), (0, 1)))
    ) & walkable
    pressure_plate_goals = (
        (static_layer == StaticObject.PRESSURE_PLATE) & walkable
    )
    goal_mask = interaction_goals if interaction_macro else pressure_plate_goals
    goal_mask = goal_mask & navigation_macro
    distances = env._distance_to_goals(walkable, goal_mask)

    all_macros = jnp.full(
        (env.num_agents,), MacroActions.wait, dtype=jnp.int32
    ).at[agent_idx].set(macro_action)
    primitive_actions, reachable = env._macro_to_primitive_actions(
        state, all_macros
    )
    valid_barriers = state.barrier_active_mask
    closed_barriers = valid_barriers & state.barrier_active
    open_timers = jnp.where(
        valid_barriers & ~state.barrier_active,
        state.barrier_timer,
        0,
    )

    return {
        "walkable": np.asarray(jax.device_get(walkable)),
        "targets": np.asarray(jax.device_get(target_mask)),
        "goals": np.asarray(jax.device_get(goal_mask)),
        "distances": np.asarray(jax.device_get(distances)),
        "agent_x": int(jax.device_get(agent.pos.x)),
        "agent_y": int(jax.device_get(agent.pos.y)),
        "primitive_action": int(jax.device_get(primitive_actions[agent_idx])),
        "reachable": bool(jax.device_get(reachable[agent_idx])),
        "navigation_macro": navigation_macro,
        "static_layer": np.asarray(jax.device_get(static_layer)),
        "closed_barriers": int(jax.device_get(jnp.sum(closed_barriers))),
        "barrier_count": int(jax.device_get(jnp.sum(valid_barriers))),
        "open_timer": int(jax.device_get(jnp.max(open_timers))),
    }


def draw_flood_fill_panel(snapshot, tile_size: int) -> Image.Image:
    """Render a planner distance field as a tile-aligned Pillow image."""
    distances = snapshot["distances"]
    walkable = snapshot["walkable"]
    goals = snapshot["goals"]
    targets = snapshot["targets"]
    static_layer = snapshot["static_layer"]
    height, width = distances.shape
    panel = Image.new(
        "RGB", (width * tile_size, height * tile_size), (28, 29, 34)
    )
    draw = ImageDraw.Draw(panel)
    font = ImageFont.load_default()
    finite = distances < int(1_000_000)
    max_distance = max(int(distances[finite].max()) if finite.any() else 1, 1)

    for y in range(height):
        for x in range(width):
            x0 = x * tile_size
            y0 = y * tile_size
            x1 = x0 + tile_size - 1
            y1 = y0 + tile_size - 1

            if not walkable[y, x]:
                fill = (48, 49, 55)
            elif not finite[y, x]:
                fill = (82, 84, 92)
            else:
                closeness = 1.0 - float(distances[y, x]) / max_distance
                fill = (
                    int(42 + 35 * closeness),
                    int(82 + 95 * closeness),
                    int(128 + 105 * closeness),
                )

            if (
                static_layer[y, x] == StaticObject.BARRIER
                and not walkable[y, x]
            ):
                fill = (145, 48, 48)
            if goals[y, x]:
                fill = (42, 150, 83)

            draw.rectangle((x0, y0, x1, y1), fill=fill, outline=(25, 26, 30))

            if targets[y, x]:
                draw.rectangle(
                    (x0 + 3, y0 + 3, x1 - 3, y1 - 3),
                    outline=(205, 100, 230),
                    width=max(2, tile_size // 18),
                )
                draw.text((x0 + 6, y0 + 5), "T", fill=(245, 210, 255), font=font)

            if goals[y, x]:
                draw.text((x0 + 6, y0 + 5), "G", fill=(235, 255, 235), font=font)
            if finite[y, x] and walkable[y, x]:
                label = str(int(distances[y, x]))
                box = draw.textbbox((0, 0), label, font=font)
                draw.text(
                    (
                        x0 + (tile_size - (box[2] - box[0])) // 2,
                        y0 + (tile_size - (box[3] - box[1])) // 2,
                    ),
                    label,
                    fill=(250, 250, 250),
                    font=font,
                )
            elif walkable[y, x] and snapshot["navigation_macro"]:
                draw.text(
                    (x0 + tile_size // 2 - 9, y0 + tile_size // 2 - 5),
                    "INF",
                    fill=(225, 225, 225),
                    font=font,
                )

    agent_x = snapshot["agent_x"]
    agent_y = snapshot["agent_y"]
    ax0 = agent_x * tile_size
    ay0 = agent_y * tile_size
    draw.rectangle(
        (ax0 + 2, ay0 + 2, ax0 + tile_size - 3, ay0 + tile_size - 3),
        outline=(255, 221, 64),
        width=max(3, tile_size // 12),
    )
    draw.text((ax0 + 6, ay0 + tile_size - 16), "A0", fill=(255, 240, 125), font=font)
    return panel


def draw_header(
    frame: np.ndarray,
    label: str,
    step: int,
    deliveries: int,
    flood_snapshot=None,
    tile_size: int = 64,
    agent_1_label: str | None = None,
) -> Image.Image:
    image = Image.fromarray(frame)
    panel = (
        draw_flood_fill_panel(flood_snapshot, tile_size)
        if flood_snapshot is not None
        else None
    )
    gap = 8 if panel is not None else 0
    content_width = image.width + gap + (panel.width if panel is not None else 0)
    header_h = 98 if panel is not None and agent_1_label is not None else 78
    if panel is None:
        header_h = 78 if agent_1_label is not None else 58
    canvas = Image.new(
        "RGB", (content_width, image.height + header_h), (24, 24, 28)
    )
    canvas.paste(image.convert("RGB"), (0, header_h))
    if panel is not None:
        canvas.paste(panel, (image.width + gap, header_h))

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
    detail_y = 52
    if agent_1_label is not None:
        draw.text(
            (10, 52),
            f"agent_1 macro: {agent_1_label}",
            fill=(170, 245, 185),
            font=font,
        )
        detail_y = 74
    if flood_snapshot is not None:
        primitive_name = Actions(flood_snapshot["primitive_action"]).name
        draw.text(
            (10, detail_y),
            f"primitive: {primitive_name} | reachable: {flood_snapshot['reachable']}"
            f" | closed gates: {flood_snapshot['closed_barriers']}/"
            f"{flood_snapshot['barrier_count']}"
            f" | open timer: {flood_snapshot['open_timer']}",
            fill=(185, 220, 255),
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
    show_flood_fill: bool,
    agent_1_macro_actions=None,
) -> None:
    """Render rollout states and optionally append planner-distance panels."""
    visualizer = OvercookedV3Visualizer(env, tile_size=tile_size)
    frames = []

    if agent_1_macro_actions is None:
        agent_1_macro_actions = [None] * len(states)

    for step, (state, macro_action, agent_1_action, deliveries) in enumerate(
        zip(states, macro_labels, agent_1_macro_actions, delivery_counts)
    ):
        if step % frame_skip != 0 and step != len(states) - 1:
            continue
        frame = np.asarray(jax.device_get(visualizer.render_state(state)))
        snapshot = (
            navigation_snapshot(env, state, macro_action)
            if show_flood_fill
            else None
        )
        frames.append(
            draw_header(
                frame,
                macro_action.name,
                step,
                deliveries,
                snapshot,
                tile_size,
                agent_1_action.name if agent_1_action is not None else None,
            )
        )

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
    layout = "timed_barrier_demo" if args.cooperative_barrier_demo else args.layout
    env = OvercookedV3Macro(
        layout=layout,
        max_steps=args.max_steps,
        max_macro_steps=args.max_macro_steps,
        barrier_duration=args.barrier_duration,
    )

    if args.cooperative_barrier_demo:
        (
            states,
            macro_actions,
            agent_1_actions,
            delivery_counts,
            deliveries,
        ) = cooperative_barrier_rollout(env, args.seed)
    else:
        states, macro_actions, delivery_counts, deliveries = rollout(
            env, args.seed
        )
        agent_1_actions = None
    save_gif(
        env,
        states,
        macro_actions,
        delivery_counts,
        args.output,
        args.tile_size,
        args.frame_ms,
        max(args.frame_skip, 1),
        args.flood_fill,
        agent_1_actions,
    )

    print(f"Wrote {args.output}")
    print(f"Frames: {len(states)}")
    print(f"Deliveries: {deliveries}")
    print(f"Final time: {int(jax.device_get(states[-1].time))}")


if __name__ == "__main__":
    main()
