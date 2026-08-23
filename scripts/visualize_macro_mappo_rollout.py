"""Evaluate a trained macro MAPPO actor: GIF rollout and/or reward/macro histograms.

Runs deterministic eval episodes from a saved actor checkpoint and can produce,
independently:
  - a rendered GIF per episode (--gif-output)
  - a two-panel histogram figure pooled across all episodes, showing total
    reward collected by type and macro-action selection counts, one bar per
    agent (--histogram-output) -- useful for reward-hacking diagnostics and
    for seeing which macros a policy actually uses vs. e.g. spamming 'wait'

Both read from the same rollout loop, so pass both flags to get a GIF you can
watch alongside the histograms that explain it.

Run from the repo root (needed for the baselines.MAPPO.* imports to resolve):

    python -m scripts.visualize_macro_mappo_rollout \\
        --variant every_step \\
        --run-dir models/mappo_macro/mappo_macro_every_step/seed_0 \\
        --actor-path models/mappo_macro/mappo_macro_every_step/seed_0/final_actor.safetensors \\
        --gif-output outputs/without_comm/rollout.gif \\
        --histogram-output outputs/without_comm/histograms.png \\
        --num-episodes 1

What it reads:
    <run-dir>/config.yaml
    --actor-path, if given, else <run-dir>/final_actor.safetensors  (point
        this at a scripts/extract_actor_checkpoints.py output to eval one
        specific training step instead of the final policy)

What it writes (only for flags you actually pass -- both are optional; the
reward/macro text summary always prints regardless):
    --gif-output           one GIF for --num-episodes == 1, else
                            <stem>_ep0.gif, <stem>_ep1.gif, ... per episode
    --histogram-output      one figure, pooled across all --num-episodes

--variant must match how <run-dir> was trained (boundary/every_step/replan).
--checkpoint-label only labels GIF frames -- it does not select weights,
that's --actor-path -- and defaults to the actor file's stem.
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

from baselines.MAPPO.mappo_macro_common import (
    Actor,
    ActorRNN,
    ReplanActor,
    ScannedRNN,
    build_env,
)
from baselines.MAPPO.mappo_macro_replan import REPLAN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("boundary", "every_step", "replan"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--actor-path",
        type=Path,
        default=None,
        help="Safetensors file to load. Defaults to <run-dir>/final_actor.safetensors. "
        "Point this at a scripts/extract_actor_checkpoints.py output to eval one "
        "specific training step instead of the final policy.",
    )
    parser.add_argument(
        "--checkpoint-label",
        default=None,
        help="Label stamped on GIF frames only (does not select which weights "
        "load -- that's --actor-path). Defaults to the actor file's stem.",
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
        default="outputs/without_comm/rollout.gif",
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
        default="outputs/without_comm/histograms.png",
        help="If set, save a reward-type/macro-action histogram figure here, "
        "pooled across all --num-episodes episodes.",
    )
    return parser.parse_args()


def episode_output_path(base: Path, index: int, total: int) -> Path:
    if total == 1:
        return base
    suffix = base.suffix if base.suffix else ".gif"
    return base.with_name(f"{base.stem}_ep{index}{suffix}")


def select_actions_rnn(actor, params, hidden, last_done, obs, env):
    """Recurrent action selection: threads the GRU carry across the episode.

    A leading time axis of 1 is added because ScannedRNN scans over time.
    """
    actor_obs = jnp.stack([obs[agent] for agent in env.agents])
    action_mask = obs["action_mask"].astype(jnp.bool_)
    hidden, logits = actor.apply(
        params, hidden, (actor_obs[None, :], last_done[None, :])
    )
    logits = logits.squeeze(0)
    return hidden, jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1)


def select_actions(variant, actor, params, obs, env):
    actor_obs = jnp.stack([obs[agent] for agent in env.agents])
    action_mask = obs["action_mask"].astype(jnp.bool_)

    if variant != "replan":
        logits = actor.apply(params, actor_obs)
        proposed_actions = jnp.argmax(
            jnp.where(action_mask, logits, -1e9), axis=-1
        )
        if variant == "boundary":
            return jnp.where(
                obs["macro_done"], proposed_actions, obs["current_macro"]
            )
        return proposed_actions

    macro_done = obs["macro_done"]
    current_macro = obs["current_macro"]
    macro_logits, replan_logits = actor.apply(params, actor_obs)
    replacement_mask = action_mask & ~(
        (~macro_done)[:, None]
        & (jnp.arange(env.num_actions)[None, :] == current_macro[:, None])
    )
    macro_action = jnp.argmax(
        jnp.where(replacement_mask, macro_logits, -1e9), axis=-1
    )
    replan_action = jnp.argmax(replan_logits, axis=-1)
    replace = macro_done | (replan_action == REPLAN)
    return jnp.where(replace, macro_action, current_macro)


def add_header(
    frame, variant, checkpoint_label, step, action_names, total_return, shaped_return
):
    header_height = 76
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    canvas = Image.new("RGB", (image.width, image.height + header_height), (18, 20, 28))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text(
        (7, 6),
        f"{variant} | {checkpoint_label} | primitive step {step}",
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
        f"[sparse] team return: {total_return:.1f}",
        fill=(170, 245, 185),
        font=font,
    )
    draw.text(
        (7, 60),
        f"[shaped] team return: {shaped_return:.1f}",
        fill=(255, 205, 130),
        font=font,
    )
    return np.asarray(canvas)


def run_episode(
    args, config, env, actor, params, seed, gif_output_path, reward_totals, macro_counts
):
    """Roll out one deterministic episode.

    Always accumulates into reward_totals/macro_counts (num_agents, ...) arrays
    for the histogram output. Only renders/saves a GIF if gif_output_path is set.
    """
    key = jax.random.PRNGKey(seed)
    obs, log_state = env.reset(key)
    state = log_state.env_state
    rollout_env = env._env._env
    states = [state]
    action_labels = [("wait", "wait")]
    returns = [0.0]
    shaped_returns = [0.0]
    total_return = 0.0
    shaped_return = 0.0
    max_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 400))

    use_rnn = bool(config.get("USE_RNN", False))
    hidden = (
        ScannedRNN.initialize_carry(env.num_agents, int(config["HIDDEN_SIZE"]))
        if use_rnn
        else None
    )
    last_done = jnp.zeros((env.num_agents,), dtype=jnp.bool_)

    for step in range(max_steps):
        if use_rnn:
            hidden, actions = select_actions_rnn(
                actor, params, hidden, last_done, obs, env
            )
        else:
            actions = select_actions(args.variant, actor, params, obs, env)
        action_names = tuple(
            env.macro_action_names[int(action)] for action in np.asarray(actions)
        )
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
        shaped_return += float(
            np.mean([np.asarray(info["shaped_reward"][agent]) for agent in env.agents])
        )
        if step % 50 == 0:
            print(
                f"Step {step}: [sparse] return={total_return:.1f}  "
                f"[shaped] return={shaped_return:.1f}"
            )

        breakdown = info["reward_breakdown"]
        for component_idx, component_key in enumerate(REWARD_COMPONENT_KEYS):
            reward_totals[:, component_idx] += np.asarray(breakdown[component_key])
        for agent_idx, agent in enumerate(env.agents):
            if bool(np.asarray(info["macro_action_started"][agent])):
                macro_idx = int(np.asarray(info["current_macro_action"][agent]))
                macro_counts[agent_idx, macro_idx] += 1

        if gif_output_path is not None:
            states.append(state)
            action_labels.append(action_names)
            returns.append(total_return)
            shaped_returns.append(shaped_return)
        last_done = jnp.full((env.num_agents,), bool(np.asarray(done["__all__"])))
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
        # batched call — stacking up to max_steps frames at full tile resolution
        # and rendering them in a single shot is the likely OOM source.
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
                args.variant,
                args.checkpoint_label,
                step,
                action_labels[step],
                returns[step],
                shaped_returns[step],
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
        print(
            f"Saved {gif_output_path} ({len(frames)} frames, "
            f"[sparse] return={total_return:.1f} [shaped] return={shaped_return:.1f})"
        )

    return total_return, shaped_return, episode_length


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


def save_histograms(args, env, reward_totals, macro_counts):
    agent_labels = [f"agent_{i}" for i in range(env.num_agents)]
    fig, (ax_reward, ax_macro) = plt.subplots(1, 2, figsize=(18, 6))
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
    fig.suptitle(f"{args.variant} | {args.checkpoint_label}")
    fig.tight_layout()
    args.histogram_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.histogram_output, dpi=150)
    print(f"Saved {args.histogram_output}")


def print_reward_macro_summary(env, reward_totals, macro_counts):
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


def main():
    args = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "config.yaml"), resolve=True
    )
    env = build_env(config)
    if config.get("USE_RNN", False):
        # Must match how the run was trained -- the RNN and MLP parameter
        # trees are different, so loading one into the other will fail.
        actor = ActorRNN(env.num_actions, int(config["HIDDEN_SIZE"]))
    elif args.variant == "replan":
        actor = ReplanActor(env.num_actions, int(config["HIDDEN_SIZE"]))
    else:
        actor = Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
    actor_path = (args.run_dir / args.actor_path) if args.actor_path else (args.run_dir / "final_actor.safetensors")
    if args.checkpoint_label is None:
        args.checkpoint_label = actor_path.stem
    params = load_params(actor_path)

    reward_totals = np.zeros((env.num_agents, len(REWARD_COMPONENT_KEYS)), dtype=np.float64)
    macro_counts = np.zeros((env.num_agents, env.num_actions), dtype=np.int64)

    episode_returns = []
    episode_shaped_returns = []
    episode_lengths = []
    for episode_index in range(args.num_episodes):
        seed = args.seed + episode_index
        gif_output_path = (
            episode_output_path(args.gif_output, episode_index, args.num_episodes)
            if args.gif_output is not None
            else None
        )
        total_return, shaped_return, length = run_episode(
            args, config, env, actor, params, seed, gif_output_path,
            reward_totals, macro_counts,
        )
        episode_returns.append(total_return)
        episode_shaped_returns.append(shaped_return)
        episode_lengths.append(length)

    if args.num_episodes > 1:
        returns_array = np.asarray(episode_returns)
        shaped_returns_array = np.asarray(episode_shaped_returns)
        lengths_array = np.asarray(episode_lengths)
        print(
            f"\n{args.num_episodes} episodes: "
            f"[sparse] return mean={returns_array.mean():.2f} std={returns_array.std():.2f} "
            f"min={returns_array.min():.2f} max={returns_array.max():.2f} | "
            f"[shaped] return mean={shaped_returns_array.mean():.2f} "
            f"std={shaped_returns_array.std():.2f} | "
            f"length mean={lengths_array.mean():.1f}"
        )

    print_reward_macro_summary(env, reward_totals, macro_counts)
    if args.histogram_output is not None:
        save_histograms(args, env, reward_totals, macro_counts)


if __name__ == "__main__":
    main()
