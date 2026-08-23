#!/usr/bin/env python3
"""Manually control both agents in the Overcooked V3 macro-action environment.

Unlike ``play_overcooked_v3.py`` (primitive WASD/arrow control), this tool
drives the *macro* interface: each agent is commanded with one of the 17
temporally extended macro actions by clicking an on-screen button. The
environment auto-runs at a fixed frame rate so you watch each macro play out
live (navigation, interaction, waiting) one primitive step at a time.

Layout is chosen at launch with ``--layout`` (use ``--list`` to see options).

Modes (``--mode``) mirror the three macro MAPPO baselines in ``baselines/``:

    boundary    Committed macros (env ``overcooked_v3_macro``). A click is
                *queued*: the agent finishes its current macro, then adopts
                your standing selection at the macro boundary. This matches
                ``mappo_macro_boundary.py``, which only lets a new macro start
                at a boundary.

    every_step  Interruptible macros (env ``overcooked_v3_macro_interruptible``).
                A click takes effect immediately, interrupting the running
                macro. Matches ``mappo_macro_every_step.py``.

    replan      Same interruptible env as ``every_step``. The learned
                CONTINUE/REPLAN gate of ``mappo_macro_replan.py`` is played by
                you: re-issuing (not clicking) is CONTINUE, clicking a new
                macro is REPLAN. Behaves like ``every_step`` under manual
                control; the HUD shows the CONTINUE/REPLAN gate each tick.

Controls:
    Mouse       Click a button in an agent's row to command that agent.
    SPACE       Pause / resume the auto-run.
    . (period)  Advance a single tick while paused.
    R           Reset the current layout.
    N / P       Next / previous layout (cycles the registered list).
    - / +       Slower / faster auto-run.
    Q / ESC     Quit.

When an agent's macro completes and you have not queued a new command, the
agent reverts to ``wait`` (idle) until you click again.
"""

import argparse

import jax
import numpy as np
import pygame

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import DynamicObject
from jaxmarl.environments.overcooked_v3.layouts import (
    load_layouts_from_json,
    overcooked_v3_layouts,
)
from jaxmarl.environments.overcooked_v3_macro.overcooked import MacroActions
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Env registration names per mode. boundary uses the committed variant so a
# click only takes effect at a macro boundary; the other two use the
# interruptible variant so a click interrupts immediately.
MODE_TO_ENV = {
    "boundary": "overcooked_v3_macro",
    "every_step": "overcooked_v3_macro_interruptible",
    "replan": "overcooked_v3_macro_interruptible",
}

# Short, human-friendly labels for each macro action, indexed by MacroActions.
MACRO_LABELS = {
    MacroActions.wait: "wait",
    MacroActions.get_ingredient_0: "get ing0",
    MacroActions.get_ingredient_1: "get ing1",
    MacroActions.get_ingredient_2: "get ing2",
    MacroActions.get_plate: "get plate",
    MacroActions.put_ingredient_in_nearest_pot: "put in pot",
    MacroActions.get_soup_from_nearest_pot: "get soup",
    MacroActions.deliver: "deliver",
    MacroActions.drop_on_nearest_counter: "drop",
    MacroActions.pickup_from_nearest_counter: "pickup",
    MacroActions.press_nearest_button: "press btn",
    MacroActions.stand_on_nearest_pressure_plate: "stand plate",
    MacroActions.wait_for_nearest_pot: "wait pot",
    MacroActions.up: "up",
    MacroActions.down: "down",
    MacroActions.left: "left",
    MacroActions.right: "right",
}

TILE = 42
NUM_MACROS = len(MacroActions)

# Button-row geometry. One row of NUM_MACROS buttons per agent.
BTN_W = 82
BTN_H = 46
BTN_GAP = 4
ROW_LABEL_H = 20
MARGIN = 10
HUD_H = 74

AGENT_COLORS = [(70, 130, 220), (70, 190, 110)]  # blue = agent_0, green = agent_1


def build_env(env_name, layout_name):
    """Create the macro env and its visualizer, auto-enabling layout features."""
    env = make(
        env_name,
        layout=layout_name,
        pot_cook_time=20,
        pot_cook_time_range=[15, 25],
        pot_burn_time=10,
        # None => auto-enable moving walls / buttons / plates from the layout.
        enable_moving_walls=None,
        enable_buttons=None,
        enable_pressure_plates=None,
    )
    viz = OvercookedV3Visualizer(env, tile_size=TILE)
    return env, viz


def describe_inventory(inventory):
    """Return a short readable description of an agent's inventory integer."""
    inventory = int(inventory)
    if inventory == DynamicObject.EMPTY:
        return "empty"
    parts = []
    if inventory & DynamicObject.PLATE:
        parts.append("plate")
    # Ingredient bits live above the plate/cooked/burned flags; count and name.
    ingredient_bits = inventory & ~(
        DynamicObject.PLATE | DynamicObject.COOKED | DynamicObject.BURNED
    )
    if ingredient_bits:
        count = int(DynamicObject.ingredient_count(np.int32(inventory)))
        kind = int(DynamicObject.get_ingredient_type(np.int32(inventory)))
        parts.append(f"ing{kind}x{count}" if count else "ing")
    if inventory & DynamicObject.COOKED:
        parts.append("cooked")
    if inventory & DynamicObject.BURNED:
        parts.append("burned")
    return "+".join(parts) if parts else f"0x{inventory:x}"


def wrap_label(font, text, max_width):
    """Split a button label into at most two lines that fit ``max_width``."""
    words = text.split()
    if len(words) <= 1 or font.size(text)[0] <= max_width:
        return [text]
    # Greedy two-line wrap; buttons never need more than two lines here.
    line1 = words[0]
    for word in words[1:]:
        if font.size(f"{line1} {word}")[0] <= max_width:
            line1 = f"{line1} {word}"
        else:
            return [line1, " ".join(words[len(line1.split()):])]
    return [line1]


def compute_geometry(env):
    """Return window/render dimensions and the button-row origin for ``env``."""
    render_w = env.width * TILE
    render_h = env.height * TILE
    panel_w = MARGIN + NUM_MACROS * (BTN_W + BTN_GAP) - BTN_GAP + MARGIN
    win_w = max(render_w, panel_w)
    rows_top = render_h + HUD_H
    win_h = rows_top + env.num_agents * (ROW_LABEL_H + BTN_H + MARGIN) + MARGIN
    return {
        "render_w": render_w,
        "render_h": render_h,
        "win_w": win_w,
        "win_h": win_h,
        "rows_top": rows_top,
        "render_x": (win_w - render_w) // 2,
    }


def button_rects(env, geom):
    """Return {(agent_idx, macro): pygame.Rect} for every command button."""
    rects = {}
    for agent_idx in range(env.num_agents):
        row_y = geom["rows_top"] + agent_idx * (ROW_LABEL_H + BTN_H + MARGIN)
        btn_y = row_y + ROW_LABEL_H
        for macro in range(NUM_MACROS):
            x = MARGIN + macro * (BTN_W + BTN_GAP)
            rects[(agent_idx, macro)] = pygame.Rect(x, btn_y, BTN_W, BTN_H)
    return rects


def draw_frame(screen, fonts, env, viz, state, geom, rects, ui):
    """Render one full frame: the world, the HUD, and both button rows.

    ``ui`` carries the transient interface state (commanded macros, gate
    labels, mode, score, step count, fps, paused) that the world state does not
    itself hold.
    """
    hud_font, btn_font, small_font = fonts
    screen.fill((30, 30, 34))

    # World render from the shared Overcooked V3 visualizer.
    img = np.array(viz.render_state(state))
    surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
    screen.blit(surf, (geom["render_x"], 0))

    avail = env.get_avail_actions(state)
    render_h = geom["render_h"]

    # HUD line 1: layout / mode / step / score / run status.
    status = "PAUSED" if ui["paused"] else f"{ui['fps']:.0f} fps"
    header = (
        f"{ui['layout']}  |  mode={ui['mode']}  |  step {ui['step_count']}"
        f"  |  score {ui['total_reward']:.0f}  |  {status}"
    )
    screen.blit(hud_font.render(header, True, (255, 255, 0)), (MARGIN, render_h + 6))

    # HUD line 2+: per-agent running macro, timer, inventory (and gate).
    for i in range(env.num_agents):
        inv = describe_inventory(state.agents.inventory[i])
        cur = MACRO_LABELS[MacroActions(int(state.current_macro_actions[i]))]
        done = bool(state.macro_action_done[i])
        steps_in = int(state.macro_step_count[i])
        line = (
            f"agent_{i}: running={cur} (t={steps_in}"
            f"{', done' if done else ''})  inv={inv}"
        )
        if ui["mode"] == "replan":
            line += f"  gate={ui['gate'][i]}"
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        screen.blit(
            small_font.render(line, True, color),
            (MARGIN + 6, render_h + 30 + i * 18),
        )

    # Button rows, one per agent.
    for i in range(env.num_agents):
        row_y = geom["rows_top"] + i * (ROW_LABEL_H + BTN_H + MARGIN)
        color = AGENT_COLORS[i] if i < len(AGENT_COLORS) else (200, 200, 200)
        tag = "(blue)" if i == 0 else "(green)" if i == 1 else ""
        screen.blit(
            small_font.render(f"agent_{i} {tag}", True, color), (MARGIN, row_y + 3)
        )

        agent_avail = np.array(avail[f"agent_{i}"]).astype(bool)
        for macro in range(NUM_MACROS):
            rect = rects[(i, macro)]
            is_current = int(state.current_macro_actions[i]) == macro
            is_commanded = ui["commanded"][i] == macro
            is_avail = bool(agent_avail[macro])

            # Fill: dim when the macro cannot make progress; highlight the
            # actively running macro brightest, the queued/commanded one darker.
            base = (58, 58, 66) if is_avail else (40, 40, 44)
            if is_current:
                base = tuple(min(255, c + 90) for c in color)
            elif is_commanded:
                base = tuple(min(255, int(c * 0.6) + 25) for c in color)
            pygame.draw.rect(screen, base, rect, border_radius=6)
            border = color if (is_current or is_commanded) else (90, 90, 100)
            pygame.draw.rect(screen, border, rect, width=2, border_radius=6)

            text_color = (255, 255, 255) if is_avail else (130, 130, 140)
            lines = wrap_label(btn_font, MACRO_LABELS[MacroActions(macro)], BTN_W - 8)
            ty = rect.y + (BTN_H - len(lines) * btn_font.get_height()) // 2
            for text_line in lines:
                text_surf = btn_font.render(text_line, True, text_color)
                screen.blit(
                    text_surf, (rect.x + (BTN_W - text_surf.get_width()) // 2, ty)
                )
                ty += btn_font.get_height()


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--layout",
        default="cramped_room",
        help="layout name to open (default: cramped_room; see --list)",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_ENV),
        default="boundary",
        help="macro decision mode matching the baseline trainers "
        "(default: boundary)",
    )
    parser.add_argument(
        "--layout-json",
        help="load and register generated layouts from this JSON file first",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=8.0,
        help="initial primitive steps per second (default: 8)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="list all registered layouts and exit",
    )
    args = parser.parse_args()

    json_layouts = (
        list(load_layouts_from_json(args.layout_json, register=True))
        if args.layout_json
        else []
    )
    del json_layouts  # Registration side effect is all we need before --list.

    if args.list:
        print("Registered overcooked_v3 layouts:")
        for name in sorted(overcooked_v3_layouts):
            print(f"  {name}")
        return

    if args.layout not in overcooked_v3_layouts:
        raise SystemExit(
            f"Unknown layout {args.layout!r}. Use --list to see options."
        )

    env_name = MODE_TO_ENV[args.mode]

    # Cycle order for N/P: put the selected layout first, then the rest.
    layouts = [args.layout] + [
        name for name in sorted(overcooked_v3_layouts) if name != args.layout
    ]
    layout_idx = 0

    print("=" * 60)
    print("  OVERCOOKED V3 MACRO - Manual Control")
    print("=" * 60)
    print(f"  mode = {args.mode}  (env: {env_name})")
    if args.mode == "boundary":
        print("  Clicks are QUEUED: the agent finishes its current macro,")
        print("  then adopts your selection at the next macro boundary.")
    else:
        print("  Clicks INTERRUPT the running macro immediately.")
    print("  Click a button in an agent's row to command that agent.")
    print("  SPACE pause/resume | . step | R reset | N/P layout | -/+ speed")
    print("=" * 60)

    pygame.init()
    key = jax.random.PRNGKey(0)

    def load(name):
        """Build env/viz for ``name`` and reset to a fresh episode."""
        nonlocal key
        env, viz = build_env(env_name, name)
        key, reset_key = jax.random.split(key)
        _, state = env.reset(reset_key)
        return env, viz, state

    env, viz, state = load(layouts[layout_idx])
    geom = compute_geometry(env)
    rects = button_rects(env, geom)

    commanded = [int(MacroActions.wait)] * env.num_agents
    gate = ["CONTINUE"] * env.num_agents

    screen = pygame.display.set_mode((geom["win_w"], geom["win_h"]))
    pygame.display.set_caption(
        f"Overcooked V3 Macro [{args.mode}] - {layouts[layout_idx]}"
    )
    clock = pygame.time.Clock()
    fonts = (
        pygame.font.Font(None, 24),
        pygame.font.Font(None, 20),
        pygame.font.Font(None, 18),
    )

    fps = max(1.0, float(args.fps))
    paused = False
    step_once = False
    total_reward = 0.0
    step_count = 0

    running = True
    while running:
        clicked_this_frame = [False] * env.num_agents

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                for (agent_idx, macro), rect in rects.items():
                    if rect.collidepoint(event.pos):
                        commanded[agent_idx] = macro
                        clicked_this_frame[agent_idx] = True
                        # A fresh click on a running macro is a REPLAN event.
                        if int(state.current_macro_actions[agent_idx]) != macro:
                            gate[agent_idx] = "REPLAN"
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_PERIOD:
                    step_once = True
                elif event.key == pygame.K_r:
                    key, reset_key = jax.random.split(key)
                    _, state = env.reset(reset_key)
                    commanded = [int(MacroActions.wait)] * env.num_agents
                    gate = ["CONTINUE"] * env.num_agents
                    total_reward, step_count = 0.0, 0
                elif event.key in (pygame.K_n, pygame.K_p):
                    layout_idx = (
                        layout_idx + (1 if event.key == pygame.K_n else -1)
                    ) % len(layouts)
                    env, viz, state = load(layouts[layout_idx])
                    geom = compute_geometry(env)
                    rects = button_rects(env, geom)
                    screen = pygame.display.set_mode((geom["win_w"], geom["win_h"]))
                    commanded = [int(MacroActions.wait)] * env.num_agents
                    gate = ["CONTINUE"] * env.num_agents
                    total_reward, step_count = 0.0, 0
                    pygame.display.set_caption(
                        f"Overcooked V3 Macro [{args.mode}] - {layouts[layout_idx]}"
                    )
                elif event.key == pygame.K_MINUS:
                    fps = max(1.0, fps - 1.0)
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS):
                    fps = min(60.0, fps + 1.0)

        # Advance one primitive step when running (or single-stepping while
        # paused). Unset agents default to wait.
        do_step = (not paused) or step_once
        step_once = False
        if do_step:
            actions = {
                f"agent_{i}": np.int32(commanded[i])
                for i in range(env.num_agents)
            }
            key, step_key = jax.random.split(key)
            _, state, rewards, dones, _ = env.step(step_key, state, actions)
            step_count += 1
            total_reward += float(rewards["agent_0"])

            # Idle-on-completion: an agent whose macro just finished with no
            # newer distinct command pending reverts to wait. In boundary mode
            # a queued click (commanded != the macro that just ran) survives so
            # it starts at the boundary.
            for i in range(env.num_agents):
                if bool(state.macro_action_done[i]) and not clicked_this_frame[i]:
                    if int(state.current_macro_actions[i]) == commanded[i]:
                        commanded[i] = int(MacroActions.wait)
                # The replan gate settles back to CONTINUE after each tick
                # unless the human interrupts again next frame.
                if not clicked_this_frame[i]:
                    gate[i] = "CONTINUE"

            if bool(dones["__all__"]):
                # env.step auto-resets; treat the returned state as episode 0.
                commanded = [int(MacroActions.wait)] * env.num_agents
                total_reward, step_count = 0.0, 0

        ui = {
            "commanded": commanded,
            "gate": gate,
            "mode": args.mode,
            "layout": layouts[layout_idx],
            "total_reward": total_reward,
            "step_count": step_count,
            "fps": fps,
            "paused": paused,
        }
        draw_frame(screen, fonts, env, viz, state, geom, rects, ui)
        pygame.display.flip()
        clock.tick(fps if not paused else 30)

    pygame.quit()
    print(f"Done. Final score {total_reward:.0f} after {step_count} steps.")


if __name__ == "__main__":
    main()
