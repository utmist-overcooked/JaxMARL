# eval.py
"""
Evaluation pipeline for IS-MADDPG on OvercookedV3.

Loads a checkpoint, runs greedy episodes, and saves a GIF of each episode.

Usage:
    python baselines/IS_MADDPG/eval.py \
        --checkpoint checkpoints/is_maddpg_cramped_room_step00200000.zip \
        --num_episodes 5 \
        --gif_dir gifs/

    # Run without GIF (faster, just metrics)
    python baselines/IS_MADDPG/eval.py \
        --checkpoint checkpoints/is_maddpg_cramped_room_step00200000.zip \
        --no_gif
"""

import argparse
import os
import pickle
import zipfile
from typing import Dict, List, Optional

import jax
import jax.numpy as jnp
import numpy as np

from jaxmarl.environments.overcooked_v3.overcooked import OvercookedV3
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

from networks import ISAgentNet
from loss import received_messages


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_checkpoint_zip(path: str) -> dict:
    """Load actor params and metadata from a .zip checkpoint."""
    with zipfile.ZipFile(path, "r") as zf:
        actor_params = pickle.loads(zf.read("actor_params.pkl"))
        meta         = pickle.loads(zf.read("metadata.pkl"))
    return {
        "actor_params": actor_params,
        "step":         meta["step"],
        "config":       meta["config"],
        "layout":       meta["layout"],
    }


def frames_to_gif(frames: List[np.ndarray], path: str, fps: int = 8) -> None:
    """Save a list of RGB frames as a GIF.

    Args:
        frames: list of (H, W, 3) uint8 numpy arrays
        path:   output .gif path
        fps:    frames per second (default 8 — readable speed)
    """
    try:
        from PIL import Image
        imgs = [Image.fromarray(f.astype(np.uint8)) for f in frames]
        imgs[0].save(
            path,
            save_all=True,
            append_images=imgs[1:],
            duration=int(1000 / fps),
            loop=0,
        )
        size_kb = os.path.getsize(path) / 1e3
        print(f"  GIF saved → {path}  ({size_kb:.0f} KB,  {len(frames)} frames)")
    except ImportError:
        print("  [WARNING] PIL not available — skipping GIF. Install with: pip install Pillow")


# ---------------------------------------------------------------------------
# Action selection (greedy, no epsilon)
# ---------------------------------------------------------------------------

def greedy_actions(
    actor_params: dict,
    actor:        ISAgentNet,
    obs_all:      np.ndarray,   # (1, N, obs_dim)
    prev_msgs:    np.ndarray,   # (1, N, msg_dim)
    rng,
    *,
    num_agents: int,
    act_dim:    int,
    gumbel_tau: float = 1.0,
) -> tuple:
    """Select greedy actions (argmax, no Gumbel noise) for all agents.

    Returns:
        actions_idx: (N,) int  — integer actions per agent
        msgs_out:    (1, N, msg_dim)
        rng:         updated key
    """
    obs_jax      = jnp.array(obs_all)
    prev_msgs_jax= jnp.array(prev_msgs)
    received     = received_messages(prev_msgs_jax)  # (1, N, N-1, msg_dim)

    actions_idx = np.zeros(num_agents, dtype=np.int32)
    msgs_out    = np.zeros_like(prev_msgs)

    for j in range(num_agents):
        rng, subkey = jax.random.split(rng)
        logits, _, msg, _ = actor.apply(
            actor_params,
            obs_jax[:, j, :],        # (1, obs_dim)
            received[:, j, :, :],    # (1, N-1, msg_dim)
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )
        # print("logits:", np.array(logits[0]))
        actions_idx[j] = int(jnp.argmax(logits[0]))
        msgs_out[0, j] = np.array(msg[0])

    return actions_idx, msgs_out, rng


def select_actions_eval(
    actor_params: dict,
    actor:        ISAgentNet,
    obs_all:      np.ndarray,   # (1, N, obs_dim)
    prev_msgs:    np.ndarray,   # (1, N, msg_dim)
    rng,
    *,
    num_agents:   int,
    act_dim:      int,
    gumbel_tau:   float = 1.0,
    epsilon:      float = 0.0,
) -> tuple:
    """Action selection with optional epsilon-greedy exploration for eval.

    Args:
        epsilon: 0.0 = fully greedy, >0 = adds random exploration
                 Set to match EPSILON_END from training config to
                 evaluate under the same conditions the policy was trained.
    """
    obs_jax       = jnp.array(obs_all)
    prev_msgs_jax = jnp.array(prev_msgs)
    received      = received_messages(prev_msgs_jax)

    actions_idx = np.zeros(num_agents, dtype=np.int32)
    msgs_out    = np.zeros_like(prev_msgs)

    for j in range(num_agents):
        rng, subkey = jax.random.split(rng)

        logits, _, msg, _ = actor.apply(
            actor_params,
            obs_jax[:, j, :],
            received[:, j, :, :],
            rng=subkey,
            gumbel_tau=gumbel_tau,
            gumbel_hard=True,
        )

        # Greedy action from raw logits
        greedy_act = int(jnp.argmax(logits[0]))

        # Epsilon-greedy
        if epsilon > 0 and np.random.random() < epsilon:
            rng, eps_key = jax.random.split(rng)
            act = int(jax.random.randint(eps_key, (), 0, act_dim))
        else:
            act = greedy_act

        actions_idx[j]    = act
        msgs_out[0, j, :] = np.array(msg[0])

    return actions_idx, msgs_out, rng



# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

# eval.py — replace the run_episode function and add visualizer import

def run_episode(
    actor_params: dict,
    actor:        ISAgentNet,
    env:          OvercookedV3,
    rng,
    *,
    config:       dict,
    render:       bool = True,
    episode_idx:  int  = 0,
    epsilon:      float = 0.0,
) -> dict:
    """Run one greedy episode and optionally save a GIF.

    Args:
        actor_params: loaded actor parameters
        actor:        ISAgentNet module
        env:          single OvercookedV3 instance
        rng:          PRNG key
        config:       config dict
        render:       if True, collect states for GIF rendering
        episode_idx:  episode number for logging
        epsilon: 0.0 = fully greedy, >0 = adds random exploration
                 Set to match EPSILON_END from training config to
                 evaluate under the same conditions the policy was trained.

    Returns:
        dict with episode stats and state_seq
    """
    num_agents = config["NUM_AGENTS"]
    obs_dim    = config["OBS_DIM"]
    msg_dim    = config["MSG_DIM"]
    act_dim    = config["ACT_DIM"]
    agent_ids  = [f"agent_{i}" for i in range(num_agents)]

    rng, reset_key = jax.random.split(rng)
    obs_dict, env_state = env.reset(reset_key)
    print(f"Recipe: {env_state.recipe}")
    # print(f"Recipe ingredient count: {DynamicObject.ingredient_count(env_state.recipe)}")
    print(f"Pot full threshold in process_interact: 2 (HARDCODED)")    

    prev_msgs  = np.zeros((1, num_agents, msg_dim), dtype=np.float32)
    ep_return  = 0.0
    deliveries = 0
    reward_events = {
        "ingredient_pickup": 0,
        "placement_in_pot": 0,
        "plate_pickup":     0,
        "soup_in_dish":     0,
        "delivery":         0,
    }

    # Collect state objects for rendering
    # We store as a list then stack into a pytree at the end
    # state_seq = [env_state]
    state_seq = [jax.tree_util.tree_map(lambda x: np.array(x).copy(), env_state)]

    max_steps = getattr(env, "max_steps", 400)

    for step in range(max_steps):
        obs_all = np.stack(
            [np.asarray(obs_dict[aid]).reshape(1, obs_dim) for aid in agent_ids],
            axis=1,
        ).astype(np.float32)

        # acts_idx, msgs, rng = greedy_actions(
        #     actor_params, actor, obs_all, prev_msgs, rng,
        #     num_agents=num_agents, act_dim=act_dim,
        # )
        acts_idx, msgs, rng = select_actions_eval(
            actor_params, actor, obs_all, prev_msgs, rng,
            num_agents=num_agents, act_dim=act_dim,
            gumbel_tau=config["GUMBEL_TAU"],
            epsilon=epsilon,   # pass from evaluate()
        )        

        action_dict = {f"agent_{i}": int(acts_idx[i]) for i in range(num_agents)}

        rng, step_key = jax.random.split(rng)
        obs_dict, env_state, rewards_dict, dones_dict, info = env.step_env(
            step_key, env_state, action_dict
        )

        if render:
            # state_seq.append(env_state)
            state_seq.append(
                jax.tree_util.tree_map(lambda x: np.array(x).copy(), env_state)
            )
        
        print("acts:", acts_idx)

        # Accumulate rewards
        raw_reward = sum(float(rewards_dict[aid]) for aid in agent_ids)      
        shaped_reward = 0.0
        if "shaped_reward" in info:
            aid0 = agent_ids[0]
            sv = float(jnp.array(info["shaped_reward"][aid0]))
            shaped_reward += sv
            if sv == 12:
                reward_events["soup_in_dish"] += 1
            elif sv == 6:
                reward_events["placement_in_pot"] += 1
            elif sv == 4:
                reward_events["plate_pickup"] += 1
            elif sv == 3:
                reward_events["ingredient_pickup"] += 1

        if raw_reward >= 20:
            deliveries += 1
            reward_events["delivery"] += 1
        if raw_reward != 0:
            print(f"  [step={step}] raw={raw_reward:.2f} shaped={shaped_reward:.2f} acts={acts_idx}", flush=True)              

        ep_return += raw_reward + shaped_reward
        prev_msgs  = msgs

        done = bool(dones_dict.get("__all__", False)) or \
               all(bool(dones_dict.get(aid, False)) for aid in agent_ids)
        if done:
            break

    print(
        f"  Episode {episode_idx+1:2d}: "
        f"return={ep_return:.1f}  "
        f"deliveries={deliveries}  "
        f"placements={reward_events['placement_in_pot']}  "
        f"pickups={reward_events['ingredient_pickup']}"
        f"plates={reward_events['plate_pickup']}  "
        f"soups={reward_events['soup_in_dish']}  "
        f"steps={step+1}"
    )

    return {
        "return":        ep_return,
        "deliveries":    deliveries,
        "reward_events": reward_events,
        "state_seq":     state_seq,  # list of State objects
        "steps":         step + 1,
    }


def save_episode_gif(
    state_seq: list,
    env:       OvercookedV3,
    path:      str,
    fps:       int = 4,
) -> None:
    """Stack state list into pytree and render GIF using OvercookedV3Visualizer.

    The visualizer's animate() uses jax.vmap over states, which requires
    the state sequence to be a single pytree of stacked arrays rather than
    a Python list of individual states.

    Args:
        state_seq: list of State objects from run_episode
        env:       OvercookedV3 instance (for pot timing config)
        path:      output .gif path
        fps:       frames per second
    """
    if not state_seq:
        print("  [WARNING] Empty state sequence — no GIF saved.")
        return

    # Stack list of State pytrees into a single batched State pytree
    # jax.tree_util.tree_map stacks each leaf array along a new axis 0
    state_batch = jax.tree_util.tree_map(
        lambda *arrays: jnp.stack(arrays, axis=0),
        *state_seq,
    )

    # Print shapes of agent positions in the batch
    # print("Batched pos.y shape:", state_batch.agents.pos.y.shape)
    # Should be (num_steps, num_agents)
    # If it's (num_agents,) the stacking failed silently

    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)

    viz = OvercookedV3Visualizer(env)
    viz.animate(
        state_batch,
        filename=path,
        agent_view_size=None,
    )

    # imageio's duration is in seconds per frame for GIF
    # The visualizer hardcodes duration=0.5 — patch it after saving
    # by re-reading and re-saving with correct fps if needed
    size_kb = os.path.getsize(path) / 1e3
    print(f"  GIF saved → {path}  ({size_kb:.0f} KB,  {len(state_seq)} frames @ {fps}fps)")



def add_step_counter(frame: np.ndarray, step: int, total_steps: int,
                     ep_return: float, deliveries: int) -> np.ndarray:
    """Add a step counter sidebar to a rendered frame.

    Args:
        frame:       (H, W, 3) uint8 numpy array from visualizer
        step:        current step number
        total_steps: total steps in episode
        ep_return:   cumulative return so far
        deliveries:  deliveries so far

    Returns:
        (H, W + sidebar_width, 3) uint8 array with sidebar appended
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        # PIL not available — return frame unchanged
        return frame

    H, W, _ = frame.shape
    SIDEBAR_W = 120

    # Create sidebar
    sidebar = Image.new("RGB", (SIDEBAR_W, H), (30, 30, 30))
    draw    = ImageDraw.Draw(sidebar)

    # Try to load a font, fall back to default
    try:
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
    except (IOError, OSError):
        font_large = ImageFont.load_default()
        font_small = font_large

    # Progress bar
    bar_x0, bar_y0 = 8, 20
    bar_w, bar_h   = SIDEBAR_W - 16, 10
    progress       = step / max(1, total_steps)

    # Background bar
    draw.rectangle([bar_x0, bar_y0, bar_x0 + bar_w, bar_y0 + bar_h],
                   fill=(80, 80, 80))
    # Filled portion
    filled_w = int(bar_w * progress)
    if filled_w > 0:
        draw.rectangle([bar_x0, bar_y0, bar_x0 + filled_w, bar_y0 + bar_h],
                       fill=(70, 180, 70))

    # Text entries
    entries = [
        ("STEP",      f"{step}/{total_steps}", (200, 200, 200), 42),
        ("RETURN",    f"{ep_return:.1f}",       (255, 220, 50),  68),
        ("DELIVERIES",f"{deliveries}",          (100, 220, 100), 94),
        ("PROGRESS",  f"{100*progress:.0f}%",   (150, 150, 255), 120),
    ]

    for label, value, color, y in entries:
        draw.text((8, y),      label, font=font_small, fill=(140, 140, 140))
        draw.text((8, y + 13), value, font=font_large, fill=color)

    # Divider line
    draw.line([(4, 16), (SIDEBAR_W - 4, 16)], fill=(60, 60, 60), width=1)

    sidebar_np = np.array(sidebar)

    # Concatenate frame + sidebar horizontally
    return np.concatenate([frame, sidebar_np], axis=1)

# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate(
    checkpoint_path: str,
    num_episodes:    int  = 1,
    gif_dir:         Optional[str] = "gifs",
    render:          bool = True,
    fps:             int  = 8,
    seed:            int  = 42,
    epsilon:         float = 0.4,
) -> None:
    """Load a checkpoint and run evaluation episodes.

    Args:
        checkpoint_path: path to .zip checkpoint
        num_episodes:    number of episodes to run
        gif_dir:         directory to save GIFs (None to skip)
        render:          whether to collect frames for GIF
        fps:             GIF playback speed
        seed:            random seed
    """
    print(f"\n{'='*60}")
    print(f"IS-MADDPG Evaluation")
    print(f"{'='*60}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"  Episodes   : {num_episodes}")
    print(f"  Render GIF : {render}")
    print(f"{'='*60}\n")

    # Load checkpoint (.pkl from run_overcooked_v3.save_checkpoint, or legacy .zip)
    if checkpoint_path.endswith(".zip"):
        ckpt   = load_checkpoint_zip(checkpoint_path)
        config = ckpt["config"]; layout = ckpt["layout"]; step = ckpt["step"]
    else:
        with open(checkpoint_path, "rb") as f:
            raw = pickle.load(f)
        config = raw["config"]
        layout = config.get("LAYOUT") or config.get("layout")
        step   = int(raw.get("step", 0))
        ckpt   = {"actor_params": raw["actor_params"], "config": config,
                  "layout": layout, "step": step}
    print(f"  Loaded checkpoint from step {step:,} — layout: {layout}\n")

    # Build actor module (stateless — just defines the architecture)
    actor = ISAgentNet(
        obs_dim=    config["OBS_DIM"],
        act_dim=    config["ACT_DIM"],
        msg_dim=    config["MSG_DIM"],
        hidden_dim= config["HIDDEN_DIM"],
        num_agents= config["NUM_AGENTS"],
        horizon_H=  config["HORIZON_H"],
    )

    actor_params = ckpt["actor_params"]
    # Build the env with the SAME kwargs training used (CTC/conveyor maps need the
    # order queue + conveyors, else the rendered policy misbehaves).
    env_kwargs = dict(layout=layout, shaped_rewards=True,
                      max_steps=config.get("MAX_STEPS", 400))
    if layout in ("coordinated_temporal_conveyor", "maze_conveyor_hell", "around_the_island"):
        env_kwargs.update(pot_cook_time=60, pot_burn_time=90, enable_order_queue=True,
                          max_orders=5, order_generation_rate=1.0, order_expiration_time=0,
                          order_queue_mode="alternating", plate_pickup_guard=1)
    if layout in ("coordinated_temporal_conveyor", "maze_conveyor_hell"):
        env_kwargs.update(enable_item_conveyors=True, enable_player_conveyors=False)
    env          = OvercookedV3(**env_kwargs)
    rng          = jax.random.PRNGKey(seed)

    # Run episodes
    all_results = []
    for ep in range(num_episodes):
        rng, ep_rng = jax.random.split(rng)
        result = run_episode(
            actor_params, actor, env, ep_rng,
            config=config, render=render, episode_idx=ep, epsilon=epsilon
        )
        all_results.append(result)

        # Save GIF using OvercookedV3Visualizer
        if render and gif_dir and result["state_seq"]:
            gif_path = os.path.join(
                gif_dir,
                f"eval_{layout}_step{step:08d}_ep{ep+1:02d}.gif"
            )
            save_episode_gif(result["state_seq"], env, gif_path, fps=fps)

    # Summary
    returns     = [r["return"]     for r in all_results]
    deliveries  = [r["deliveries"] for r in all_results]

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY  ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"  Mean return    : {np.mean(returns):.2f} ± {np.std(returns):.2f}")
    print(f"  Max return     : {np.max(returns):.2f}")
    print(f"  Mean deliveries: {np.mean(deliveries):.2f} ± {np.std(deliveries):.2f}")
    print(f"  Max deliveries : {int(np.max(deliveries))}")

    # Aggregate reward events
    all_events = {}
    for r in all_results:
        for k, v in r["reward_events"].items():
            all_events[k] = all_events.get(k, 0) + v
    print(f"\n  Reward events (total across {num_episodes} episodes):")
    for k, v in all_events.items():
        print(f"    {k:<20}: {v:>5}  ({v/num_episodes:.1f}/ep)")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate IS-MADDPG on OvercookedV3")
    parser.add_argument(
        "--checkpoint", type=str, required=True,
        help="Path to .zip checkpoint file"
    )
    parser.add_argument(
        "--num_episodes", type=int, default=10,
        help="Number of evaluation episodes"
    )
    parser.add_argument(
        "--gif_dir", type=str, default="gifs",
        help="Directory to save GIF files"
    )
    parser.add_argument(
        "--no_gif", action="store_true",
        help="Skip GIF rendering (faster, metrics only)"
    )
    parser.add_argument(
        "--fps", type=int, default=8,
        help="GIF playback speed in frames per second"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for evaluation"
    )
    parser.add_argument(
        "--epsilon", type=float, default=0.1,
        help="Exploration rate for eval (0=greedy, match corresponding epsilon value from training)"
    )    
    args = parser.parse_args()

    evaluate(
        checkpoint_path=args.checkpoint,
        num_episodes=   args.num_episodes,
        gif_dir=        None if args.no_gif else args.gif_dir,
        render=         not args.no_gif,
        fps=            args.fps,
        seed=           args.seed,
        epsilon=        args.epsilon,
    )


if __name__ == "__main__":
    main()