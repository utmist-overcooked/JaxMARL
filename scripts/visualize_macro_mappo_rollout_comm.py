"""Render a deterministic rollout from the comm-augmented macro MAPPO
policy: frozen macro actor + trained communication module.

Separate from the base render script because action selection now
includes a communication round, and the params live in two different
places (frozen macro actor path stored in the comm run's config.yaml,
comm module weights in the comm run's own final_actor.safetensors —
see note in chat about why that file holds comm params, not macro-actor
params, for this particular run).
"""

import argparse
from pathlib import Path
import sys


MAPPO_DIR = Path(__file__).parents[1] / "baselines" / "MAPPO"
sys.path.insert(0, str(MAPPO_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import load_params
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

from mappo_macro_common import Actor, build_env

# Adjust this import if your local training script filename differs.
from mappo_macro_every_step_comm import CommModule


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Comm training run directory (contains config.yaml and final_actor.safetensors, "
        "the latter holding the trained comm module weights for this run).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=150)
    parser.add_argument("--tile-size", type=int, default=40)
    parser.add_argument(
        "--render-chunk-size",
        type=int,
        default=25,
        help="Render this many frames per visualizer call instead of the whole "
        "trajectory at once. Lower this further if you still see OOM.",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Run this many separate episodes, each seeded with --seed + episode "
        "index, saving one GIF per episode.",
    )
    return parser.parse_args()


def episode_output_path(base: Path, index: int, total: int) -> Path:
    if total == 1:
        return base
    suffix = base.suffix if base.suffix else ".gif"
    return base.with_name(f"{base.stem}_ep{index}{suffix}")


def swap_two_agent_messages(message):
    """2 agents only: agent0 receives agent1's message and vice versa."""
    return message[::-1]


def select_actions_comm(actor, comm_module, actor_params, comm_params, obs, env):
    actor_obs = jnp.stack([obs[agent] for agent in env.agents])
    action_mask = obs["action_mask"].astype(jnp.bool_)

    message_logits = comm_module.apply(
        comm_params, actor_obs, method=comm_module.encode_message
    )
    message = jnp.argmax(message_logits, axis=-1)
    received_message = swap_two_agent_messages(message)

    logit_bias = comm_module.apply(
        comm_params, actor_obs, received_message, method=comm_module.correction
    )
    base_logits = actor.apply(actor_params, actor_obs)
    final_logits = base_logits + logit_bias

    # Built on the every_step macro variant (no macro_done boundary gating).
    actions = jnp.argmax(jnp.where(action_mask, final_logits, -1e9), axis=-1)
    return actions, message


def add_header(
    frame, checkpoint_label, step, action_names, message_pair, total_return
):
    header_height = 76
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    canvas = Image.new("RGB", (image.width, image.height + header_height), (18, 20, 28))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (7, 6),
        f"every_step_comm | {checkpoint_label} | primitive step {step}",
        fill=(245, 245, 245),
        font=font,
    )
    draw.text(
        (7, 24),
        f"A0: {action_names[0]}  A1: {action_names[1]}",
        fill=(150, 220, 255),
        font=font,
    )
    draw.text(
        (7, 42),
        f"msg A0->A1: {message_pair[1]}   A1->A0: {message_pair[0]}",
        fill=(255, 205, 130),
        font=font,
    )
    draw.text(
        (7, 60),
        f"sparse team return: {total_return:.1f}",
        fill=(170, 245, 185),
        font=font,
    )
    return np.asarray(canvas)


def run_episode(args, config, env, actor, comm_module, actor_params, comm_params, seed, output_path):
    key = jax.random.PRNGKey(seed)
    obs, log_state = env.reset(key)
    state = log_state.env_state
    rollout_env = env._env._env
    states = [state]
    action_labels = [("wait", "wait")]
    message_labels = [(0, 0)]
    returns = [0.0]
    total_return = 0.0
    max_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 400))

    for step in range(max_steps):
        actions, message = select_actions_comm(
            actor, comm_module, actor_params, comm_params, obs, env
        )
        action_names = tuple(
            env.macro_action_names[int(action)] for action in np.asarray(actions)
        )
        message_pair = tuple(int(m) for m in np.asarray(message))
        env_actions = {
            agent: actions[index] for index, agent in enumerate(env.agents)
        }
        key, step_key = jax.random.split(key)
        raw_obs, state, reward, done, info = rollout_env.step_env(
            step_key, state, env_actions
        )
        obs = env._env._augment(raw_obs, state)
        total_return += float(
            np.mean([np.asarray(reward[agent]) for agent in env.agents])
        )
        if step % 50==0:
            print(f"Step {step}: return={total_return:.1f}")
        states.append(state)
        action_labels.append(action_names)
        message_labels.append(message_pair)
        returns.append(total_return)
        if bool(np.asarray(done["__all__"])):
            break

    frame_indices = list(range(0, len(states), args.frame_skip))
    if frame_indices[-1] != len(states) - 1:
        frame_indices.append(len(states) - 1)
    selected_states = [states[index] for index in frame_indices]

    visualizer = OvercookedV3Visualizer(rollout_env, tile_size=args.tile_size)

    # Render in chunks rather than stacking the whole trajectory into one
    # batched call — that's the likely OOM source in the base script too.
    rendered = []
    chunk_size = max(1, args.render_chunk_size)
    for chunk_start in range(0, len(selected_states), chunk_size):
        chunk = selected_states[chunk_start : chunk_start + chunk_size]
        stacked_chunk = jax.tree.map(lambda *values: jnp.stack(values), *chunk)
        rendered_chunk = jax.device_get(visualizer.render_sequence(stacked_chunk))
        rendered.extend(rendered_chunk)
        del stacked_chunk, rendered_chunk

    frames = [
        add_header(
            frame,
            args.checkpoint_label,
            step,
            action_labels[step],
            message_labels[step],
            returns[step],
        )
        for frame, step in zip(rendered, frame_indices)
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = [Image.fromarray(frame) for frame in frames]
    gif_frames[0].save(
        output_path,
        format="GIF",
        save_all=True,
        append_images=gif_frames[1:],
        duration=args.frame_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved {output_path} ({len(frames)} frames, return={total_return:.1f})")
    return total_return, len(states) - 1


def main():
    args = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "config.yaml"), resolve=True
    )
    env = build_env(config)

    actor = Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
    comm_module = CommModule(
        hidden_size=int(config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"])),
        vocab_size=int(config["VOCAB_SIZE"]),
        action_dim=env.num_actions,
        message_embed_dim=int(config.get("MESSAGE_EMBED_DIM", 8)),
    )

    actor_params = load_params(Path(config["FROZEN_ACTOR_PATH"]))
    comm_params = load_params(args.run_dir / "best_actor.safetensors")

    episode_returns = []
    episode_lengths = []
    for episode_index in range(args.num_episodes):
        seed = args.seed + episode_index
        output_path = episode_output_path(args.output, episode_index, args.num_episodes)
        total_return, length = run_episode(
            args, config, env, actor, comm_module, actor_params, comm_params, seed, output_path
        )
        episode_returns.append(total_return)
        episode_lengths.append(length)

    if args.num_episodes > 1:
        returns_array = np.asarray(episode_returns)
        lengths_array = np.asarray(episode_lengths)
        print(
            f"\n{args.num_episodes} episodes: "
            f"return mean={returns_array.mean():.2f} std={returns_array.std():.2f} "
            f"min={returns_array.min():.2f} max={returns_array.max():.2f} | "
            f"length mean={lengths_array.mean():.1f}"
        )


if __name__ == "__main__":
    main()