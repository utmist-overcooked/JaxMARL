#!/usr/bin/env python3
"""Interactive Overcooked V3 player with keyboard controls."""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import jax
import numpy as np
import pygame

from jaxmarl import make
from jaxmarl.environments.overcooked_v3.common import DynamicObject, StaticObject
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Keyboard mappings for Agent 0 (WASD + Space)
AGENT0_KEYS = {
    pygame.K_w: 3,      # up
    pygame.K_s: 1,      # down
    pygame.K_a: 2,      # left
    pygame.K_d: 0,      # right
    pygame.K_SPACE: 5,  # interact
}

# Keyboard mappings for Agent 1 (Arrow keys + Enter)
AGENT1_KEYS = {
    pygame.K_UP: 3,     # up
    pygame.K_DOWN: 1,   # down
    pygame.K_LEFT: 2,   # left
    pygame.K_RIGHT: 0,  # right
    pygame.K_RETURN: 5, # interact
}


def _parse_args():
    parser = argparse.ArgumentParser(description="Play and optionally record Overcooked V3 demonstrations.")
    parser.add_argument("--layout", type=str, default="player_conveyor_loop")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--pot-cook-time", type=int, default=20)
    parser.add_argument("--pot-burn-time", type=int, default=10)
    parser.add_argument("--tile-size", type=int, default=48)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--record-dir", type=str, default="")
    parser.add_argument(
        "--partner-mode",
        type=str,
        default="human",
        choices=("human", "scripted", "stay"),
        help="How to control the non-human partner. 'human' preserves two-player keyboard mode.",
    )
    parser.add_argument(
        "--human-agent",
        type=int,
        default=0,
        choices=(0, 1),
        help="Which agent the local player controls in solo modes.",
    )
    parser.add_argument(
        "--print-all-rewards",
        action="store_true",
        help="Print reward information every step instead of only non-zero events.",
    )
    parser.add_argument(
        "--disable-player-conveyors",
        action="store_true",
        help="Disable player conveyor interactions for layouts that do not use them.",
    )
    return parser.parse_args()


def _stack_obs(obs, agents):
    return np.stack([np.asarray(obs[agent], dtype=np.float32) for agent in agents], axis=0)


def _action_name(action: int) -> str:
    return ["right", "down", "left", "up", "stay", "interact"][int(action)]


def _decode_ingredient_counts(obj: int, max_types: int = 3) -> List[int]:
    return [int((obj >> (2 + 2 * idx)) & 0x3) for idx in range(max_types)]


def _walkable_static(static_value: int) -> bool:
    return static_value in {
        int(StaticObject.EMPTY),
        int(StaticObject.ITEM_CONVEYOR),
        int(StaticObject.PLAYER_CONVEYOR),
    }


def _direction_action(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    dx = to_pos[0] - from_pos[0]
    dy = to_pos[1] - from_pos[1]
    if dx == 1 and dy == 0:
        return 0
    if dx == -1 and dy == 0:
        return 2
    if dx == 0 and dy == 1:
        return 1
    if dx == 0 and dy == -1:
        return 3
    return 4


def _find_cells(static_grid: np.ndarray, target_value: int) -> List[Tuple[int, int]]:
    ys, xs = np.where(static_grid == target_value)
    return [(int(x), int(y)) for y, x in zip(ys, xs)]


def _adjacent_walkable_positions(
    static_grid: np.ndarray,
    target_positions: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    height, width = static_grid.shape
    adjacent: List[Tuple[int, int]] = []
    for target_x, target_y in target_positions:
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            x = target_x + dx
            y = target_y + dy
            if 0 <= x < width and 0 <= y < height and _walkable_static(int(static_grid[y, x])):
                adjacent.append((x, y))
    return adjacent


def _bfs_next_step(
    static_grid: np.ndarray,
    start: Tuple[int, int],
    goals: Sequence[Tuple[int, int]],
    blocked: Optional[Iterable[Tuple[int, int]]] = None,
) -> Optional[Tuple[int, int]]:
    goal_set = set(goals)
    if start in goal_set:
        return start

    blocked_set = set(blocked or [])
    blocked_set.discard(start)
    height, width = static_grid.shape
    frontier = [start]
    parents = {start: None}
    index = 0

    while index < len(frontier):
        x, y = frontier[index]
        index += 1
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx = x + dx
            ny = y + dy
            nxt = (nx, ny)
            if not (0 <= nx < width and 0 <= ny < height):
                continue
            if nxt in parents or nxt in blocked_set:
                continue
            if not _walkable_static(int(static_grid[ny, nx])):
                continue
            parents[nxt] = (x, y)
            if nxt in goal_set:
                current = nxt
                while parents[current] is not None and parents[current] != start:
                    current = parents[current]
                return current
            frontier.append(nxt)
    return None


def _navigate_to_interaction_target(
    static_grid: np.ndarray,
    agent_pos: Tuple[int, int],
    agent_dir: int,
    target_positions: Sequence[Tuple[int, int]],
    blocked: Optional[Iterable[Tuple[int, int]]] = None,
) -> int:
    target_set = set(target_positions)
    for target_x, target_y in target_positions:
        if abs(agent_pos[0] - target_x) + abs(agent_pos[1] - target_y) == 1:
            desired = _direction_action(agent_pos, (target_x, target_y))
            if int(agent_dir) == desired:
                return 5
            return desired

    approach_positions = _adjacent_walkable_positions(static_grid, target_positions)
    next_step = _bfs_next_step(static_grid, agent_pos, approach_positions, blocked=blocked)
    if next_step is None:
        if agent_pos in target_set:
            return 4
        return 4
    if next_step == agent_pos:
        for target_x, target_y in target_positions:
            if abs(agent_pos[0] - target_x) + abs(agent_pos[1] - target_y) == 1:
                return _direction_action(agent_pos, (target_x, target_y))
        return 4
    return _direction_action(agent_pos, next_step)


def _pot_ready(dynamic_obj: int) -> bool:
    return bool(dynamic_obj & int(DynamicObject.COOKED)) and not bool(dynamic_obj & int(DynamicObject.BURNED))


def _scripted_partner_action(state, partner_idx: int, env) -> int:
    static_grid = np.asarray(state.grid[..., 0], dtype=np.int32)
    dynamic_grid = np.asarray(state.grid[..., 1], dtype=np.int32)
    agent_x = int(np.asarray(state.agents.pos.x)[partner_idx])
    agent_y = int(np.asarray(state.agents.pos.y)[partner_idx])
    agent_pos = (agent_x, agent_y)
    agent_dir = int(np.asarray(state.agents.dir)[partner_idx])
    inventory = int(np.asarray(state.agents.inventory)[partner_idx])

    blocked_positions = []
    for idx in range(env.num_agents):
        if idx == partner_idx:
            continue
        blocked_positions.append(
            (
                int(np.asarray(state.agents.pos.x)[idx]),
                int(np.asarray(state.agents.pos.y)[idx]),
            )
        )

    pot_positions = _find_cells(static_grid, int(StaticObject.POT))
    goal_positions = _find_cells(static_grid, int(StaticObject.GOAL))
    plate_positions = _find_cells(static_grid, int(StaticObject.PLATE_PILE))

    ingredient_positions: Dict[int, List[Tuple[int, int]]] = {}
    for ingredient_idx in range(3):
        ingredient_positions[ingredient_idx] = _find_cells(
            static_grid, int(StaticObject.INGREDIENT_PILE_BASE) + ingredient_idx
        )

    recipe_counts = _decode_ingredient_counts(int(np.asarray(state.recipe)))
    pot_contents = [0, 0, 0]
    pot_is_ready = False
    if pot_positions:
        pot_x, pot_y = pot_positions[0]
        pot_dynamic = int(dynamic_grid[pot_y, pot_x])
        pot_contents = _decode_ingredient_counts(pot_dynamic)
        pot_is_ready = _pot_ready(pot_dynamic)

    needs_ingredients = any(pot_contents[idx] < recipe_counts[idx] for idx in range(len(recipe_counts))) and not pot_is_ready

    if inventory & int(DynamicObject.COOKED):
        return _navigate_to_interaction_target(static_grid, agent_pos, agent_dir, goal_positions, blocked_positions)

    if inventory == int(DynamicObject.PLATE):
        return _navigate_to_interaction_target(static_grid, agent_pos, agent_dir, pot_positions, blocked_positions)

    if bool(DynamicObject.is_ingredient(inventory)):
        return _navigate_to_interaction_target(static_grid, agent_pos, agent_dir, pot_positions, blocked_positions)

    if pot_is_ready:
        return _navigate_to_interaction_target(static_grid, agent_pos, agent_dir, plate_positions, blocked_positions)

    if needs_ingredients:
        for ingredient_idx, required in enumerate(recipe_counts):
            if required > pot_contents[ingredient_idx] and ingredient_positions[ingredient_idx]:
                return _navigate_to_interaction_target(
                    static_grid,
                    agent_pos,
                    agent_dir,
                    ingredient_positions[ingredient_idx],
                    blocked_positions,
                )

    if plate_positions:
        return _navigate_to_interaction_target(static_grid, agent_pos, agent_dir, plate_positions, blocked_positions)
    return 4


def _compute_partner_action(args, state, partner_idx: Optional[int], env) -> Optional[int]:
    if partner_idx is None:
        return None
    if args.partner_mode == "stay":
        return 4
    if args.partner_mode == "scripted":
        return _scripted_partner_action(state, partner_idx, env)
    return None


def _flush_episode(record_dir, layout, env_kwargs, episode_index, episode_buffer, reason, metadata_extra=None):
    if record_dir is None or not episode_buffer["actions"]:
        return episode_index

    obs = np.stack(episode_buffer["obs"], axis=0).astype(np.float32)
    dones = np.stack(episode_buffer["dones"], axis=0).astype(np.bool_)
    actions = np.stack(episode_buffer["actions"], axis=0).astype(np.int32)
    rewards = np.stack(episode_buffer["rewards"], axis=0).astype(np.float32)
    shaped_rewards = np.stack(episode_buffer["shaped_rewards"], axis=0).astype(np.float32)
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    episode_name = f"demo_{timestamp}_{episode_index:04d}.npz"
    metadata = {
        "layout": layout,
        "reason": reason,
        "episode_index": episode_index,
        "episode_length": int(actions.shape[0]),
        "num_agents": int(actions.shape[1]),
        "returns": rewards.sum(axis=0).tolist(),
        "shaped_returns": shaped_rewards.sum(axis=0).tolist(),
        "env_kwargs": env_kwargs,
    }
    if metadata_extra:
        metadata.update(metadata_extra)
    np.savez_compressed(
        record_dir / episode_name,
        obs=obs,
        dones=dones,
        actions=actions,
        rewards=rewards,
        shaped_rewards=shaped_rewards,
        metadata_json=np.array(json.dumps(metadata)),
    )
    print(
        f"Saved recording: {record_dir / episode_name} "
        f"({metadata['episode_length']} steps, returns={metadata['returns']}, shaped={metadata['shaped_returns']})"
    )
    return episode_index + 1


def _new_episode_buffer():
    return {"obs": [], "dones": [], "actions": [], "rewards": [], "shaped_rewards": []}


def main():
    args = _parse_args()

    print("=" * 50)
    print("  OVERCOOKED V3 - Interactive Mode")
    print("=" * 50)
    print()
    print("Controls:")
    if args.partner_mode == "human":
        print("  Agent 0 (Blue):  WASD to move, SPACE to interact")
        print("  Agent 1 (Green): Arrow keys to move, ENTER to interact")
    else:
        partner_idx = 1 - args.human_agent
        print(f"  Human controls Agent {args.human_agent} (WASD + SPACE)")
        print(f"  Agent {partner_idx} partner mode: {args.partner_mode}")
    print()
    print("  R = Reset")
    print("  Q or ESC = Quit")
    print()
    print("Goal: Pick up onions, put 3 in the pot, wait for cooking,")
    print("      then pick up soup with a plate and deliver to green zone!")
    print()
    print("Pot timing: Cook=20 steps, Burn window=10 steps")
    print("  - Green bar = cooking progress")
    print("  - Orange bar = burning window (pick up before it empties!)")
    print("  - HUD shows sparse reward, latest shaped reward, and last joint action")
    if args.record_dir:
        print()
        print(f"Recording enabled: {args.record_dir}")
    print("=" * 50)

    # Create environment
    env_kwargs = {
        "layout": args.layout,
        "max_steps": args.max_steps,
        "pot_cook_time": args.pot_cook_time,
        "pot_burn_time": args.pot_burn_time,
        "enable_player_conveyors": not args.disable_player_conveyors,
    }
    env = make("overcooked_v3", **env_kwargs)
    viz = OvercookedV3Visualizer(env, tile_size=args.tile_size)

    # Initialize pygame
    pygame.init()

    # Calculate window size
    width = env.width * args.tile_size
    height = env.height * args.tile_size
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Overcooked V3 - Press Q to quit")
    clock = pygame.time.Clock()

    record_dir = Path(args.record_dir).expanduser() if args.record_dir else None
    if record_dir is not None:
        record_dir.mkdir(parents=True, exist_ok=True)

    # Initialize game state
    key = jax.random.PRNGKey(args.seed)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey)

    total_reward = 0
    step_count = 0
    running = True
    episode_index = 0
    episode_buffer = _new_episode_buffer()
    prev_done = np.zeros((env.num_agents,), dtype=np.bool_)
    total_shaped_reward = np.zeros((env.num_agents,), dtype=np.float32)
    last_reward = 0.0
    last_shaped_reward = np.zeros((env.num_agents,), dtype=np.float32)
    last_action_arr = np.full((env.num_agents,), 4, dtype=np.int32)
    solo_mode = args.partner_mode != "human"
    partner_idx = None if not solo_mode else 1 - args.human_agent
    episode_metadata = {
        "partner_mode": args.partner_mode,
        "solo_mode": solo_mode,
        "human_agent": None if not solo_mode else int(args.human_agent),
        "partner_agent": None if partner_idx is None else int(partner_idx),
    }

    while running:
        # Handle events
        agent0_action = 4  # stay
        agent1_action = 4  # stay

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_r:
                    # Reset
                    episode_index = _flush_episode(
                        record_dir,
                        args.layout,
                        env_kwargs,
                        episode_index,
                        episode_buffer,
                        reason="manual_reset",
                        metadata_extra=episode_metadata,
                    )
                    episode_buffer = _new_episode_buffer()
                    key, subkey = jax.random.split(key)
                    obs, state = env.reset(subkey)
                    total_reward = 0
                    step_count = 0
                    prev_done = np.zeros((env.num_agents,), dtype=np.bool_)
                    total_shaped_reward = np.zeros((env.num_agents,), dtype=np.float32)
                    last_reward = 0.0
                    last_shaped_reward = np.zeros((env.num_agents,), dtype=np.float32)
                    last_action_arr = np.full((env.num_agents,), 4, dtype=np.int32)
                    print("\n--- Game Reset ---\n")

        # Get current key states for continuous input
        keys = pygame.key.get_pressed()

        if args.partner_mode == "human":
            for k, action in AGENT0_KEYS.items():
                if keys[k]:
                    agent0_action = action
                    break

            for k, action in AGENT1_KEYS.items():
                if keys[k]:
                    agent1_action = action
                    break
        else:
            human_action = 4
            for k, action in AGENT0_KEYS.items():
                if keys[k]:
                    human_action = action
                    break

            partner_action = _compute_partner_action(args, state, partner_idx, env)
            if args.human_agent == 0:
                agent0_action = human_action
                agent1_action = 4 if partner_action is None else partner_action
            else:
                agent0_action = 4 if partner_action is None else partner_action
                agent1_action = human_action

        # Step environment
        episode_buffer["obs"].append(_stack_obs(obs, env.agents))
        episode_buffer["dones"].append(prev_done.copy())

        action_arr = np.array([agent0_action, agent1_action], dtype=np.int32)
        episode_buffer["actions"].append(action_arr)

        actions = {"agent_0": agent0_action, "agent_1": agent1_action}
        key, subkey = jax.random.split(key)
        obs, state, rewards, dones, info = env.step(subkey, state, actions)

        reward_arr = np.array([rewards[a] for a in env.agents], dtype=np.float32)
        shaped_reward_arr = np.array([info["shaped_reward"][a] for a in env.agents], dtype=np.float32)
        episode_buffer["rewards"].append(reward_arr)
        episode_buffer["shaped_rewards"].append(shaped_reward_arr)
        prev_done = np.array([dones[a] for a in env.agents], dtype=np.bool_)

        step_count += 1
        reward = rewards["agent_0"]
        total_reward += reward
        total_shaped_reward += shaped_reward_arr
        last_reward = float(reward)
        last_shaped_reward = shaped_reward_arr
        last_action_arr = action_arr.copy()

        should_log_rewards = args.print_all_rewards or reward != 0 or np.any(np.abs(shaped_reward_arr) > 1e-6)
        if should_log_rewards:
            print(
                f"Step {step_count:03d} | sparse={float(reward):+.2f} | "
                f"shaped={[round(float(x), 2) for x in shaped_reward_arr.tolist()]} | "
                f"actions={[ _action_name(a) for a in action_arr.tolist() ]}"
            )

        if reward > 0:
            print(f"Delivery completed: +{reward:.0f} points (Total sparse score: {total_reward:.0f})")

        if dones["__all__"]:
            print(
                f"Episode complete. Sparse score: {total_reward:.0f} in {step_count} steps | "
                f"shaped totals={[round(float(x), 2) for x in total_shaped_reward.tolist()]}"
            )
            episode_index = _flush_episode(
                record_dir,
                args.layout,
                env_kwargs,
                episode_index,
                episode_buffer,
                reason="episode_done",
                metadata_extra=episode_metadata,
            )
            episode_buffer = _new_episode_buffer()
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey)
            total_reward = 0
            step_count = 0
            prev_done = np.zeros((env.num_agents,), dtype=np.bool_)
            total_shaped_reward = np.zeros((env.num_agents,), dtype=np.float32)
            last_reward = 0.0
            last_shaped_reward = np.zeros((env.num_agents,), dtype=np.float32)
            last_action_arr = np.full((env.num_agents,), 4, dtype=np.int32)

        # Render
        img = viz.render_state(state)
        img_np = np.array(img)

        # Convert to pygame surface (need to transpose for pygame)
        surf = pygame.surfarray.make_surface(img_np.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Draw HUD
        font = pygame.font.Font(None, 24)
        hud_text = f"Step: {step_count}  Sparse: {total_reward:.0f}  Last: {last_reward:+.1f}"
        text_surf = font.render(hud_text, True, (255, 255, 255))
        screen.blit(text_surf, (5, 5))
        shaped_text = font.render(
            f"Shaped A0/A1: {last_shaped_reward[0]:+.2f} / {last_shaped_reward[1]:+.2f}",
            True,
            (180, 255, 180),
        )
        screen.blit(shaped_text, (5, 28))
        action_text = font.render(
            f"Actions: A0={_action_name(last_action_arr[0])}  A1={_action_name(last_action_arr[1])}",
            True,
            (180, 220, 255),
        )
        screen.blit(action_text, (5, 51))
        if record_dir is not None:
            rec_text = font.render(f"REC {record_dir.name}  Ep {episode_index}", True, (255, 210, 80))
            screen.blit(rec_text, (5, 74))
        if solo_mode:
            solo_text = font.render(
                f"Solo: human=A{args.human_agent} partner={args.partner_mode}",
                True,
                (255, 220, 160),
            )
            screen.blit(solo_text, (5, 97))

        pygame.display.flip()
        clock.tick(args.fps)

    episode_index = _flush_episode(
        record_dir,
        args.layout,
        env_kwargs,
        episode_index,
        episode_buffer,
        reason="quit",
        metadata_extra=episode_metadata,
    )
    pygame.quit()
    print(
        f"\nGame Over! Final sparse score: {total_reward:.0f} in {step_count} steps | "
        f"shaped totals={[round(float(x), 2) for x in total_shaped_reward.tolist()]}"
    )


if __name__ == "__main__":
    main()
