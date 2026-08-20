"""Evaluate the comm-augmented macro MAPPO policy: frozen macro actor + trained
communication module. GIF rollout and/or reward/macro/message histograms.

Separate from the base render script because action selection now includes a
communication round, and the params live in two different places (frozen
macro actor path stored in the comm run's config.yaml, comm module weights
in the comm run's own final_actor.safetensors -- see note in chat about why
that file holds comm params, not macro-actor params, for this particular
run).

Runs deterministic eval episodes and can produce, independently:
  - a rendered GIF per episode (--gif-output)
  - a three-panel histogram figure pooled across all episodes: total reward
    collected by type, macro-action selection counts, and comm message
    symbols sent -- one bar per agent in each panel (--histogram-output)

Both read from the same rollout loop, so pass both flags to get a GIF you can
watch alongside the histograms that explain it.
"""

import argparse
from pathlib import Path
import sys


MAPPO_DIR = Path(__file__).parents[1] / "baselines" / "MAPPO"
sys.path.insert(0, str(MAPPO_DIR))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from jaxmarl.environments.overcooked_v3.settings import REWARD_COMPONENT_KEYS
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
    parser.add_argument(
        "--checkpoint-label",
        default=None,
        help="Label stamped on GIF frames only. Defaults to <run-dir>'s name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=1,
        help="Run this many separate episodes, each seeded with --seed + episode "
        "index. Histograms pool over all of them; GIFs are saved one per episode.",
    )
    parser.add_argument(
        "--gif-output",
        type=Path,
        default="outputs/with_comm/rollout.gif",
        help="If set, render and save a rollout GIF per episode to this path.",
    )
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
        "--histogram-output",
        type=Path,
        default="outputs/with_comm/histograms.png",
        help="If set, save a reward-type/macro-action/message histogram figure "
        "here, pooled across all --num-episodes episodes.",
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


def run_episode(
    args,
    config,
    env,
    actor,
    comm_module,
    actor_params,
    comm_params,
    seed,
    gif_output_path,
    reward_totals,
    macro_counts,
    message_counts,
):
    """Roll out one deterministic episode.

    Always accumulates into reward_totals/macro_counts/message_counts
    (num_agents, ...) arrays for the histogram output. Only renders/saves a
    GIF if gif_output_path is set.
    """
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
        if step % 50 == 0:
            print(f"Step {step}: return={total_return:.1f}")

        breakdown = info["reward_breakdown"]
        for component_idx, component_key in enumerate(REWARD_COMPONENT_KEYS):
            reward_totals[:, component_idx] += np.asarray(breakdown[component_key])
        for agent_idx, agent in enumerate(env.agents):
            if bool(np.asarray(info["macro_action_started"][agent])):
                macro_idx = int(np.asarray(info["current_macro_action"][agent]))
                macro_counts[agent_idx, macro_idx] += 1
        for agent_idx, symbol in enumerate(message_pair):
            message_counts[agent_idx, symbol] += 1

        if gif_output_path is not None:
            states.append(state)
            action_labels.append(action_names)
            message_labels.append(message_pair)
            returns.append(total_return)
        if bool(np.asarray(done["__all__"])):
            break

    episode_length = len(states) - 1

    if gif_output_path is not None:
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

        gif_output_path.parent.mkdir(parents=True, exist_ok=True)
        gif_frames = [Image.fromarray(frame) for frame in frames]
        gif_frames[0].save(
            gif_output_path,
            format="GIF",
            save_all=True,
            append_images=gif_frames[1:],
            duration=args.frame_ms,
            loop=0,
            optimize=False,
        )
        print(f"Saved {gif_output_path} ({len(frames)} frames, return={total_return:.1f})")

    return total_return, episode_length


AGENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_grouped_bars(ax, labels, values_by_agent, agent_labels, title, ylabel):
    num_groups = len(labels)
    num_agents = len(values_by_agent)
    bar_width = 0.8 / num_agents
    x = np.arange(num_groups)
    for agent_idx, values in enumerate(values_by_agent):
        offset = (agent_idx - (num_agents - 1) / 2) * bar_width
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            label=agent_labels[agent_idx],
            color=AGENT_COLORS[agent_idx % len(AGENT_COLORS)],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()


def save_histograms(args, env, vocab_size, reward_totals, macro_counts, message_counts):
    agent_labels = [f"agent_{i}" for i in range(env.num_agents)]
    fig, (ax_reward, ax_macro, ax_message) = plt.subplots(1, 3, figsize=(24, 6))
    plot_grouped_bars(
        ax_reward,
        list(REWARD_COMPONENT_KEYS),
        [reward_totals[i] for i in range(env.num_agents)],
        agent_labels,
        f"Total reward by type over {args.num_episodes} episode(s)",
        "total reward",
    )
    plot_grouped_bars(
        ax_macro,
        list(env.macro_action_names),
        [macro_counts[i] for i in range(env.num_agents)],
        agent_labels,
        f"Macro action selections over {args.num_episodes} episode(s)",
        "times selected",
    )
    plot_grouped_bars(
        ax_message,
        [str(symbol) for symbol in range(vocab_size)],
        [message_counts[i] for i in range(env.num_agents)],
        agent_labels,
        f"Comm messages sent over {args.num_episodes} episode(s)",
        "times sent",
    )
    fig.suptitle(f"every_step_comm | {args.checkpoint_label}")
    fig.tight_layout()
    args.histogram_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.histogram_output, dpi=150)
    print(f"Saved {args.histogram_output}")


def print_reward_macro_message_summary(env, vocab_size, reward_totals, macro_counts, message_counts):
    agent_labels = [f"agent_{i}" for i in range(env.num_agents)]
    print("\nReward totals by type:")
    for agent_idx, agent in enumerate(agent_labels):
        print(f"  {agent}:")
        for component_idx, component_key in enumerate(REWARD_COMPONENT_KEYS):
            print(f"    {component_key}: {reward_totals[agent_idx, component_idx]:.2f}")

    print("\nMacro action selection counts:")
    for agent_idx, agent in enumerate(agent_labels):
        print(f"  {agent}:")
        for macro_idx, name in enumerate(env.macro_action_names):
            count = int(macro_counts[agent_idx, macro_idx])
            if count > 0:
                print(f"    {name}: {count}")

    print("\nComm message symbol counts:")
    for agent_idx, agent in enumerate(agent_labels):
        print(f"  {agent}:")
        for symbol in range(vocab_size):
            count = int(message_counts[agent_idx, symbol])
            if count > 0:
                print(f"    symbol {symbol}: {count}")


def main():
    args = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "config.yaml"), resolve=True
    )
    if args.checkpoint_label is None:
        args.checkpoint_label = args.run_dir.name
    env = build_env(config)

    actor = Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
    vocab_size = int(config["VOCAB_SIZE"])
    comm_module = CommModule(
        hidden_size=int(config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"])),
        vocab_size=vocab_size,
        action_dim=env.num_actions,
        message_embed_dim=int(config.get("MESSAGE_EMBED_DIM", 8)),
    )

    actor_params = load_params(Path(config["FROZEN_ACTOR_PATH"]))
    comm_params = load_params(args.run_dir / "best_actor.safetensors")

    reward_totals = np.zeros((env.num_agents, len(REWARD_COMPONENT_KEYS)), dtype=np.float64)
    macro_counts = np.zeros((env.num_agents, env.num_actions), dtype=np.int64)
    message_counts = np.zeros((env.num_agents, vocab_size), dtype=np.int64)

    episode_returns = []
    episode_lengths = []
    for episode_index in range(args.num_episodes):
        seed = args.seed + episode_index
        gif_output_path = (
            episode_output_path(args.gif_output, episode_index, args.num_episodes)
            if args.gif_output is not None
            else None
        )
        total_return, length = run_episode(
            args, config, env, actor, comm_module, actor_params, comm_params, seed,
            gif_output_path, reward_totals, macro_counts, message_counts,
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

    print_reward_macro_message_summary(env, vocab_size, reward_totals, macro_counts, message_counts)
    if args.histogram_output is not None:
        save_histograms(args, env, vocab_size, reward_totals, macro_counts, message_counts)


if __name__ == "__main__":
    main()
