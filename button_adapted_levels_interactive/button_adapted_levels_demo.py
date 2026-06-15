#!/usr/bin/env python3
"""Interactive demo for button-adapted Overcooked V3 layouts.

Controls:
  Agent 0 (Blue): WASD to move, SPACE to interact
  Agent 1 (Green): Arrows to move, ENTER to interact
  R = Reset
  Q/ESC = Quit
"""

import jax
import jax.numpy as jnp
import numpy as np
import pygame
from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

AGENT0_KEYS = {
    pygame.K_w: 3,  # up
    pygame.K_s: 1,  # down
    pygame.K_a: 2,  # left
    pygame.K_d: 0,  # right
    pygame.K_SPACE: 5,  # interact
}

AGENT1_KEYS = {
    pygame.K_UP: 3,  # up
    pygame.K_DOWN: 1,  # down
    pygame.K_LEFT: 2,  # left
    pygame.K_RIGHT: 0,  # right
    pygame.K_RETURN: 5,  # interact
}

"""
Button Adapted Layouts:
    (GOOD) button_triple_gate, 
    (GOOD) button_gated_conveyor_access, 
    button_gated_circuit, 
    (GOOD) button_gated_zones
"""
LAYOUT_NAME = "button_twin_movement" # Change this to try different layouts


def create_env(layout_name: str) -> OvercookedV3:
    return OvercookedV3(
        layout=layout_name,
        enable_buttons=True,
        enable_pressure_plates=False,
        enable_item_conveyors=True,
        enable_moving_walls=False,
        pot_cook_time=20,
        pot_burn_time=10,
    )


def main():
    layout_name = LAYOUT_NAME

    print("=" * 55)
    print("  OVERCOOKED V3 - Button Adapted Levels Demo")
    print("=" * 55)
    print("Controls:")
    print("  Agent 0 (Blue): WASD to move, SPACE to interact")
    print("  Agent 1 (Green): Arrows to move, ENTER to interact")
    print("  R = Reset, Q/ESC = Quit")
    print("=" * 55)

    env = create_env(layout_name)
    tile_size = 48
    viz = OvercookedV3Visualizer(env, tile_size=tile_size)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    pygame.init()
    width = env.width * tile_size
    height = env.height * tile_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Button Adapted Levels Demo - {layout_name}")
    clock = pygame.time.Clock()

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    obs, state = jit_reset(subkey)

    total_reward = 0.0
    step_count = 0
    running = True

    while running:
        agent0_action = 4
        agent1_action = 4

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    key, subkey = jax.random.split(key)
                    obs, state = jit_reset(subkey)
                    total_reward = 0.0
                    step_count = 0
                    print("\n--- Reset ---\n")

        keys = pygame.key.get_pressed()
        for k, action in AGENT0_KEYS.items():
            if keys[k]:
                agent0_action = action
                break
        for k, action in AGENT1_KEYS.items():
            if keys[k]:
                agent1_action = action
                break

        actions = {
            "agent_0": jnp.array(agent0_action),
            "agent_1": jnp.array(agent1_action),
        }
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = jit_step(subkey, state, actions)

        step_count += 1
        reward = float(rewards["agent_0"] + rewards["agent_1"])
        total_reward += reward

        img = viz.render_state(state)
        img_np = np.array(img)
        surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        font = pygame.font.Font(None, 24)
        hud = f"Step: {step_count}  Score: {total_reward:.0f}  Layout: {layout_name}"
        screen.blit(font.render(hud, True, (255, 255, 255)), (5, 5))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print(f"\nDone! Score: {total_reward:.0f} in {step_count} steps")


if __name__ == "__main__":
    main()
