#!/usr/bin/env python3
"""Interactive Overcooked V3 player with keyboard controls.

Usage:
    python scripts/play_overcooked_v3.py                # default layout
    python scripts/play_overcooked_v3.py --layout pressure_plate_demo
    python scripts/play_overcooked_v3.py --list         # list all layouts

In game:
    Agent 0 (Blue):  WASD to move, SPACE to interact
    Agent 1 (Green): Arrow keys to move, ENTER to interact
    N = next layout, P = previous layout (cycles the PLAYABLE list below)
    R = reset, Q/ESC = quit
"""

import argparse
import jax
import pygame
import numpy as np
from jaxmarl import make
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer
from jaxmarl.environments.overcooked_v3.layouts import overcooked_v3_layouts

# Curated list to cycle with N/P keys.
PLAYABLE = [
    "random_recipe_demo",
    "pressure_plate_demo",
    "pressure_gated_conveyor_access",
    "pressure_gated_circuit",
    "pressure_gated_zones",
    "twin_movement",
    "button_triple_gate",
    "button_gated_conveyor_access",
    "button_gated_circuit",
    "button_gated_zones",
]
DEFAULT_LAYOUT = PLAYABLE[0]

AGENT0_KEYS = {pygame.K_w: 3, pygame.K_s: 1, pygame.K_a: 2, pygame.K_d: 0, pygame.K_SPACE: 5}
AGENT1_KEYS = {pygame.K_UP: 3, pygame.K_DOWN: 1, pygame.K_LEFT: 2, pygame.K_RIGHT: 0, pygame.K_RETURN: 5}

TILE = 48
POT_COOK_TIME_RANGE = [15, 25]


def build(layout_name):
    """Create env + viz, auto-enabling all interactive features."""
    env_kwargs = {
        "layout": layout_name,
        "pot_cook_time": 20,
        "pot_cook_time_range": POT_COOK_TIME_RANGE,
        "pot_burn_time": 10,
        # None => auto-enable from the layout (walls/buttons/plates).
        "enable_moving_walls": None,
        "enable_buttons": None,
        "enable_pressure_plates": None,
    }
    if layout_name == "random_recipe_demo":
        env_kwargs.update(
            enable_random_recipe=True,
            recipe_probs=[0.5, 0.5],
        )
    env = make("overcooked_v3", **env_kwargs)
    viz = OvercookedV3Visualizer(env, tile_size=TILE)
    return env, viz


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", default=DEFAULT_LAYOUT)
    parser.add_argument("--list", action="store_true", help="list all registered layouts and exit")
    args = parser.parse_args()

    if args.list:
        print("Registered overcooked_v3 layouts:")
        for name in overcooked_v3_layouts:
            print(f"  {name}")
        return

    if args.layout not in overcooked_v3_layouts:
        raise SystemExit(f"Unknown layout {args.layout!r}. Use --list to see options.")

    # Cycle index: start the N/P cycle at the requested layout if it's in PLAYABLE.
    layouts = PLAYABLE if args.layout in PLAYABLE else [args.layout] + PLAYABLE
    idx = layouts.index(args.layout)

    print("=" * 56)
    print("  OVERCOOKED V3 - Interactive Mode")
    print("=" * 56)
    print("  Agent 0 (Blue):  WASD move, SPACE interact")
    print("  Agent 1 (Green): Arrows move, ENTER interact")
    print("  N/P = next/prev layout   R = reset   Q/ESC = quit")
    print(
        "  Pot timing: time until ready=random inclusive range "
        f"{POT_COOK_TIME_RANGE}; burn window=10 steps"
    )
    print("  A pot-start line prints when a full pot begins cooking.")
    print("=" * 56)

    pygame.init()
    key = jax.random.PRNGKey(42)

    def load(i):
        name = layouts[i]
        env, viz = build(name)
        nonlocal_key = jax.random.split(key)[1]
        obs, state = env.reset(nonlocal_key)
        screen = pygame.display.set_mode((env.width * TILE, env.height * TILE))
        pygame.display.set_caption(f"Overcooked V3 - {name}")
        print(f"\n=== Layout: {name}  ({env.width}x{env.height}, agents={len(env.agents)}) ===")
        print(f"    enable_moving_walls={env.enable_moving_walls} "
              f"enable_buttons={env.enable_buttons} "
              f"enable_pressure_plates={env.enable_pressure_plates}")
        if env.enable_random_recipe:
            print("    recipe target resamples between onion and tomato after delivery")
        return env, viz, state, screen, name

    env, viz, state, screen, name = load(idx)
    clock = pygame.time.Clock()
    font = None
    total_reward, step_count, running = 0, 0, True

    while running:
        a0, a1 = 4, 4
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    key, sk = jax.random.split(key)
                    obs, state = env.reset(sk)
                    total_reward, step_count = 0, 0
                    print("--- reset ---")
                elif event.key in (pygame.K_n, pygame.K_p):
                    idx = (idx + (1 if event.key == pygame.K_n else -1)) % len(layouts)
                    env, viz, state, screen, name = load(idx)
                    total_reward, step_count = 0, 0

        keys = pygame.key.get_pressed()
        for k, a in AGENT0_KEYS.items():
            if keys[k]:
                a0 = a
                break
        for k, a in AGENT1_KEYS.items():
            if keys[k]:
                a1 = a
                break

        actions = {"agent_0": a0}
        if "agent_1" in env.agents:
            actions["agent_1"] = a1

        previous_pot_timers = np.array(state.pot_cooking_timer)
        key, sk = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(sk, state, actions)
        current_pot_timers = np.array(state.pot_cooking_timer)
        current_pot_durations = np.array(state.pot_cook_durations)
        step_count += 1
        r = float(rewards["agent_0"])
        total_reward += r

        started_cooking = np.where(
            (previous_pot_timers == 0) & (current_pot_timers > 0)
        )[0]
        for pot_idx in started_cooking:
            sampled_ready_time = int(current_pot_durations[pot_idx])
            print(
                f"Pot {pot_idx} started cooking at step {step_count}: "
                f"sampled time until ready = {sampled_ready_time} steps"
            )

        if r > 0:
            print(f"DELIVERY! +{r:.0f} (total {total_reward:.0f})")

        img = np.array(viz.render_state(state))
        surf = pygame.surfarray.make_surface(img.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        if font is None:
            font = pygame.font.Font(None, 22)
        # Show barrier states if this layout has any (handy for plate testing).
        hud = f"{name}  step {step_count}  score {total_reward:.0f}"
        if hasattr(state, "barrier_active_mask"):
            n = int(np.array(state.barrier_active_mask).sum())
            if n:
                active = np.array(state.barrier_active)[:n].astype(int).tolist()
                hud += f"  barriers(active=1):{active}"
        screen.blit(font.render(hud, True, (255, 255, 0)), (5, 5))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print(f"\nDone. Final score {total_reward:.0f} in {step_count} steps.")


if __name__ == "__main__":
    main()
