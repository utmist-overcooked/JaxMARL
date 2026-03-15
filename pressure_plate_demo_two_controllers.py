#!/usr/bin/env python3
"""Interactive Overcooked V3 pressure-plate demo for two controllers.

Gamepad controls for each player:
  Left stick or D-pad: move
  Face/shoulder buttons: interact
  START: reset
  BACK/SELECT: quit

Keyboard controls (app-level only):
  R = Reset, Q/ESC = Quit
"""

import jax
import jax.numpy as jnp
import numpy as np
import pygame

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

ACTION_RIGHT = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_UP = 3
ACTION_STAY = 4
ACTION_INTERACT = 5

DEADZONE = 0.5
INTERACT_BUTTONS = (0, 1, 2, 3, 4, 5)  # A/B/X/Y/LB/RB on common layouts
BUTTON_BACK = 6
BUTTON_START = 7


def create_env(layout_name):
    """Create environment with moving walls and buttons enabled."""
    return OvercookedV3(
        layout=layout_name,
        enable_moving_walls=True,
        enable_buttons=True,
        enable_item_conveyors=True,
        pot_cook_time=20,
        pot_burn_time=10,
    )


def refresh_controllers(max_count=2):
    """Return up to `max_count` initialized pygame joystick objects."""
    controllers = []
    for i in range(min(pygame.joystick.get_count(), max_count)):
        joystick = pygame.joystick.Joystick(i)
        joystick.init()
        controllers.append(joystick)
    return controllers


def button_pressed(joystick, button_idx):
    return button_idx < joystick.get_numbuttons() and joystick.get_button(button_idx)


def controller_action(joystick):
    """Map one controller state to a single Overcooked discrete action."""
    if joystick is None:
        return ACTION_STAY

    if any(button_pressed(joystick, b) for b in INTERACT_BUTTONS):
        return ACTION_INTERACT

    if joystick.get_numhats() > 0:
        hat_x, hat_y = joystick.get_hat(0)
        if hat_x > 0:
            return ACTION_RIGHT
        if hat_x < 0:
            return ACTION_LEFT
        if hat_y > 0:
            return ACTION_UP
        if hat_y < 0:
            return ACTION_DOWN

    axis_x = joystick.get_axis(0) if joystick.get_numaxes() > 0 else 0.0
    axis_y = joystick.get_axis(1) if joystick.get_numaxes() > 1 else 0.0

    if abs(axis_x) < DEADZONE and abs(axis_y) < DEADZONE:
        return ACTION_STAY

    if abs(axis_x) >= abs(axis_y):
        return ACTION_RIGHT if axis_x > 0 else ACTION_LEFT
    return ACTION_DOWN if axis_y > 0 else ACTION_UP


def get_slot(controllers, index):
    return controllers[index] if index < len(controllers) else None


def main():
    # CHANGE LAYOUT NAME HERE:
    # pressure_gated_conveyor_access
    # pressure_gated_circuit
    # pressure_gated_zones
    # twin_movement
    layout_name = "pressure_plate_demo"  # "pressure_plate_demo"

    print("=" * 60)
    print("  OVERCOOKED V3 - Pressure Plates Demo (Two Controllers)")
    print("=" * 60)
    print("Controller controls (both players):")
    print("  Left stick / D-pad = Move")
    print("  Face buttons or shoulder buttons = Interact")
    print("  START = Reset, BACK/SELECT = Quit")
    print("Keyboard (app controls only):")
    print("  R = Reset, Q/ESC = Quit")
    print("=" * 60)

    env = create_env(layout_name)
    tile_size = 64
    viz = OvercookedV3Visualizer(env, tile_size=tile_size)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    pygame.init()
    pygame.joystick.init()

    grid_width = env.width * tile_size
    grid_height = env.height * tile_size

    display_info = pygame.display.Info()
    window_width = max(960, int(display_info.current_w * 0.9))
    window_height = max(540, int(display_info.current_h * 0.9))
    screen = pygame.display.set_mode((window_width, window_height), pygame.RESIZABLE)
    pygame.display.set_caption(f"Pressure Plate Demo - {layout_name} (2 Controllers)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 30)

    controllers = refresh_controllers(max_count=2)
    if len(controllers) < 2:
        print(f"Warning: detected {len(controllers)} controller(s). Connect two for full control.")
    for i, c in enumerate(controllers):
        print(f"Controller {i}: {c.get_name()}")

    key = jax.random.PRNGKey(42)
    key, subkey = jax.random.split(key)
    obs, state = jit_reset(subkey)

    total_reward = 0
    step_count = 0
    running = True

    while running:
        reset_requested = False

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type in (pygame.JOYDEVICEADDED, pygame.JOYDEVICEREMOVED):
                controllers = refresh_controllers(max_count=2)
                print(f"Controllers connected: {len(controllers)}")
                for i, c in enumerate(controllers):
                    print(f"Controller {i}: {c.get_name()}")
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_r:
                    reset_requested = True

        for joystick in controllers:
            if button_pressed(joystick, BUTTON_BACK):
                running = False
            if button_pressed(joystick, BUTTON_START):
                reset_requested = True

        if reset_requested:
            key, subkey = jax.random.split(key)
            obs, state = jit_reset(subkey)
            total_reward = 0
            step_count = 0
            print("\n--- Reset ---\n")

        agent0_action = controller_action(get_slot(controllers, 0))
        agent1_action = controller_action(get_slot(controllers, 1))

        actions = {
            "agent_0": jnp.array(agent0_action),
            "agent_1": jnp.array(agent1_action),
        }

        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = jit_step(subkey, state, actions)

        step_count += 1
        reward = rewards["agent_0"]
        total_reward += reward

        if reward > 0:
            print(f"DELIVERY! +{reward:.0f} (Total: {total_reward:.0f})")

        img = viz.render_state(state)
        img_np = np.array(img)
        surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        # Fit the full grid into the window while preserving aspect ratio.
        screen_width, screen_height = screen.get_size()
        scale = min(screen_width / grid_width, screen_height / grid_height)
        scaled_width = max(1, int(grid_width * scale))
        scaled_height = max(1, int(grid_height * scale))
        scaled_surf = pygame.transform.scale(surf, (scaled_width, scaled_height))

        # Letterbox with black bars when aspect ratios differ.
        offset_x = (screen_width - scaled_width) // 2
        offset_y = (screen_height - scaled_height) // 2
        screen.fill((0, 0, 0))
        screen.blit(scaled_surf, (offset_x, offset_y))

        hud = (
            f"Step: {step_count}  Score: {total_reward:.0f}  "
            f"Layout: {layout_name}  Controllers: {len(controllers)}/2"
        )
        hud_text = font.render(hud, True, (255, 255, 255))
        hud_bg = pygame.Surface((hud_text.get_width() + 16, hud_text.get_height() + 10), pygame.SRCALPHA)
        hud_bg.fill((0, 0, 0, 170))
        screen.blit(hud_bg, (8, 8))
        screen.blit(hud_text, (16, 13))

        pygame.display.flip()
        clock.tick(10)

    pygame.quit()
    print(f"\nDone! Score: {total_reward:.0f} in {step_count} steps")


if __name__ == "__main__":
    main()
