"""Evaluate a JOINT macro+comm run from mappo_macro_boundary_joint_comm.py.

Separate from scripts/visualize_macro_mappo_rollout_comm.py because that script
assumes the two-stage setup: a frozen macro actor at config["FROZEN_ACTOR_PATH"]
plus comm weights in the run's own checkpoint. A joint run has neither -- the
actor and comm module were trained together and live in ONE checkpoint as a
single pytree {"actor": ..., "comm": ...}. Pointing the two-stage script at a
joint run fails on the missing FROZEN_ACTOR_PATH key.

Produces, independently:
  - a rendered GIF per episode                     (--gif-output)
  - reward-type / macro-action / message histograms (--histogram-output)
  - a counterfactual message-intervention report    (--intervene)
  - a message->recipe decoding report               (--decode)

The last two are the ones that let you attribute credit to the protocol rather
than to the extra parameters, which return curves alone cannot do:

  --intervene  is POSITIVE LISTENING. It replays each decision point under
      every possible received symbol, holding observation and both carries
      fixed, and reports how often the chosen macro actually changes. A
      protocol the listener ignores scores 0 here no matter how good the
      return looks. The reported logit swing is directly comparable to the
      base logit gap the correction head has to overcome.

  --decode     is POSITIVE SIGNALING. It fits the empirical-optimal
      symbol->recipe decoder (exhaustive for a small vocab, so no probe
      training) and reports accuracy against the majority-class baseline plus
      mutual information in bits. A speaker that emits a constant scores at
      baseline with 0 bits.

Both are counterfactual probes evaluated off to the side: they never alter the
trajectory, so the GIF and histograms still show true behavior.

Run from the repo root (this script puts baselines/MAPPO on sys.path itself):

    python -m scripts.visualize_macro_mappo_rollout_joint_comm \
        --run-dir models/mappo_macro/mappo_macro_boundary_joint_comm/seed_0 \
        --gif-output outputs/joint_comm/rollout.gif \
        --histogram-output outputs/joint_comm/histograms.png \
        --intervene --decode \
        --num-episodes 1

What it reads:
    <run-dir>/config.yaml                    -- env, USE_RNN, COMM_USE_MEMORY,
                                                VOCAB_SIZE, COMM_MODE
    <run-dir>/best_actor.safetensors         -- {"actor", "comm"} in one tree,
                                                or --actor-path for a specific
                                                checkpoint (see
                                                scripts/extract_actor_checkpoints.py)

COMM_MODE defaults to the run's own value so eval reproduces training. Override
it with --comm-mode to run a severed-channel control against the SAME weights;
that measures how much the trained policy actually depends on the channel,
which is a different question from training a separate control run.

--checkpoint-label only labels GIF frames; it does not select weights.
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

from mappo_macro_common import Actor, ActorRNN, ScannedRNN, build_env
from mappo_macro_every_step_comm import CommModule
from mappo_macro_boundary_joint_comm import COMM_MODES, route_messages


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir", type=Path, required=True,
        help="Joint comm run directory (contains config.yaml and "
             "best_actor.safetensors holding both actor and comm params).",
    )
    parser.add_argument(
        "--actor-path", type=Path, default=None,
        help="Specific checkpoint holding the joint {actor, comm} params. "
             "Defaults to <run-dir>/best_actor.safetensors.",
    )
    parser.add_argument(
        "--checkpoint-label", default=None,
        help="Label stamped on GIF frames only. Defaults to <run-dir>'s name.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--num-episodes", type=int, default=1,
        help="Episodes to run, each seeded --seed + index. Histograms and the "
             "intervention/decoding reports pool over all of them; GIFs are "
             "one per episode. Use >=8 for the diagnostics to mean anything.",
    )
    parser.add_argument(
        "--comm-mode", default=None, choices=list(COMM_MODES),
        help="Override the run's COMM_MODE for this eval. Defaults to the "
             "trained value so eval reproduces training.",
    )
    parser.add_argument(
        "--intervene", action="store_true",
        help="Report positive listening: how often the chosen macro changes "
             "when the received symbol is swapped, holding all else fixed.",
    )
    parser.add_argument(
        "--decode", action="store_true",
        help="Report positive signaling: accuracy of the best symbol->recipe "
             "decoder vs the majority-class baseline, plus mutual information.",
    )
    parser.add_argument(
        "--gif-output", type=Path, default=None,
        help="If set, render and save a rollout GIF per episode.",
    )
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=150)
    parser.add_argument("--tile-size", type=int, default=40)
    parser.add_argument(
        "--render-chunk-size", type=int, default=25,
        help="Frames rendered per visualizer call; lower this if you hit OOM.",
    )
    parser.add_argument(
        "--histogram-output", type=Path, default=None,
        help="If set, save the reward/macro/message histogram figure here.",
    )
    return parser.parse_args()


def episode_output_path(base: Path, index: int, total: int) -> Path:
    if total == 1:
        return base
    suffix = base.suffix if base.suffix else ".gif"
    return base.with_name(f"{base.stem}_ep{index}{suffix}")


class JointPolicy:
    """Deterministic action/message selection for a jointly trained run.

    Mirrors mappo_macro_boundary_joint_comm.py's rollout exactly: the message
    is re-emitted only at a macro boundary (otherwise the previous one
    persists), and a new macro is committed only at a boundary. Evaluating
    without that gating would not reproduce training.

    Holds the architecture flags so the intervention probe can re-run just the
    listening half of a step without re-running the speaker or advancing carries.
    """

    def __init__(self, actor, comm_module, params, config, num_agents, comm_mode):
        self.actor = actor
        self.comm_module = comm_module
        self.actor_params = params["actor"]
        self.comm_params = params["comm"]
        self.use_rnn = bool(config.get("USE_RNN", False))
        self.comm_use_memory = bool(config.get("COMM_USE_MEMORY", False))
        # Changes the actor's input width, so it must mirror the training run.
        self.comm_injection = config.get("COMM_INJECTION", "concat")
        self.num_agents = num_agents
        self.comm_mode = comm_mode
        self.hidden_size = int(config["HIDDEN_SIZE"])
        self.comm_hidden_size = int(
            config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"])
        )

    def initial_carries(self):
        """Zeroed GRU carries; None for the memoryless variants."""
        actor_hidden = (
            ScannedRNN.initialize_carry(self.num_agents, self.hidden_size)
            if self.use_rnn else None
        )
        comm_hidden = (
            ScannedRNN.initialize_carry(self.num_agents, self.comm_hidden_size)
            if self.comm_use_memory else None
        )
        return actor_hidden, comm_hidden

    def speak(self, obs_stack, comm_hidden, last_done, last_message, macro_done):
        """-> (comm_hidden, summary, message). Boundary-gated, deterministic."""
        if self.comm_use_memory:
            comm_hidden, summary, logits = self.comm_module.apply(
                self.comm_params, comm_hidden, obs_stack[None, :],
                last_done[None, :], method=self.comm_module.encode_message_recurrent,
            )
            logits = logits.squeeze(0)
        else:
            summary = None
            logits = self.comm_module.apply(
                self.comm_params, obs_stack, method=self.comm_module.encode_message
            )
        message = jnp.where(macro_done, jnp.argmax(logits, axis=-1), last_message)
        return comm_hidden, summary, message

    def listen(self, obs_stack, actor_hidden, summary, last_done, received):
        """-> (actor_hidden, final_logits) for a given received message.

        Pure in `received`, which is what makes the intervention probe valid:
        calling it again with a different symbol and the SAME carries isolates
        the channel's effect from everything else in the step. That holds for
        both injections -- under "concat" the message enters the actor's input,
        under "bias" it enters as an added logit -- and in neither case does it
        touch a carry, so a single-step swap captures the full causal effect.
        """
        if self.comm_injection == "concat":
            embed = self.comm_module.apply(
                self.comm_params, received, method=self.comm_module.embed_message
            )
            actor_input = jnp.concatenate((obs_stack, embed), axis=-1)
        else:
            actor_input = obs_stack

        if self.use_rnn:
            actor_hidden, logits = self.actor.apply(
                self.actor_params, actor_hidden,
                (actor_input[None, :], last_done[None, :]),
            )
            logits = logits.squeeze(0)
        else:
            logits = self.actor.apply(self.actor_params, actor_input)

        if self.comm_injection == "concat":
            return actor_hidden, logits

        if self.comm_use_memory:
            bias = self.comm_module.apply(
                self.comm_params, summary, obs_stack[None, :], received[None, :],
                method=self.comm_module.correction_recurrent,
            ).squeeze(0)
        else:
            bias = self.comm_module.apply(
                self.comm_params, obs_stack, received,
                method=self.comm_module.correction,
            )
        return actor_hidden, logits + bias

    def step(self, obs, env, actor_hidden, comm_hidden, last_done, last_message, key):
        """One deterministic decision. Returns everything the probes need."""
        obs_stack = jnp.stack([obs[agent] for agent in env.agents])
        action_mask = obs["action_mask"].astype(jnp.bool_)
        macro_done = obs["macro_done"]

        comm_hidden, summary, message = self.speak(
            obs_stack, comm_hidden, last_done, last_message, macro_done
        )
        # One env at eval, so route over num_envs=1. Note "shuffled" is a no-op
        # here -- there is no other environment to draw a message from; main()
        # warns rather than silently reporting it as a control.
        received = route_messages(message, key, 1, self.comm_mode)

        actor_hidden, final_logits = self.listen(
            obs_stack, actor_hidden, summary, last_done, received
        )
        actions = jnp.where(
            macro_done,
            jnp.argmax(jnp.where(action_mask, final_logits, -1e9), axis=-1),
            obs["current_macro"],
        )
        return {
            "actions": actions,
            "message": message,
            "received": received,
            "actor_hidden": actor_hidden,
            "comm_hidden": comm_hidden,
            # frozen inputs the intervention probe replays against
            "obs_stack": obs_stack,
            "action_mask": action_mask,
            "macro_done": macro_done,
            "summary": summary,
        }


def probe_intervention(policy, obs_stack, actor_hidden, summary, last_done,
                       action_mask, macro_done, vocab_size):
    """Counterfactual: what would each agent choose under every other symbol?

    Replays only the listening half with the pre-step carries, so the returned
    trajectory is untouched. Returns per-agent (changed, logit_swing) where
    `changed` is True if any symbol yields a different masked argmax and
    `logit_swing` is the max spread of any action's logit across symbols --
    directly comparable to the base logit gap the correction must overcome.
    """
    per_symbol_logits = []
    per_symbol_actions = []
    for symbol in range(vocab_size):
        forced = jnp.full((policy.num_agents,), symbol, dtype=jnp.int32)
        _, logits = policy.listen(obs_stack, actor_hidden, summary, last_done, forced)
        per_symbol_logits.append(np.asarray(logits))
        per_symbol_actions.append(
            np.asarray(jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1))
        )
    stacked_logits = np.stack(per_symbol_logits)      # (vocab, agents, actions)
    stacked_actions = np.stack(per_symbol_actions)    # (vocab, agents)
    changed = (stacked_actions != stacked_actions[0]).any(axis=0)
    swing = (stacked_logits.max(axis=0) - stacked_logits.min(axis=0)).max(axis=-1)
    # Only decisions at a macro boundary are real choices; elsewhere the agent
    # is committed to a running macro and the logits are not consulted.
    boundary = np.asarray(macro_done).astype(bool)
    return changed & boundary, swing, boundary


def decoding_report(messages, recipes, vocab_size):
    """Positive signaling: best symbol->recipe decoder vs majority baseline.

    With a small vocabulary the optimal deterministic decoder is just "map each
    symbol to its most common co-occurring recipe", so this is exhaustive
    rather than a trained probe -- no fitting or held-out split needed to
    upper-bound what the symbol carries. Also returns mutual information in
    bits, which unlike accuracy is not flattered by a skewed recipe
    distribution.
    """
    messages = np.asarray(messages)
    recipes = np.asarray(recipes)
    if messages.size == 0:
        return None
    recipe_values, recipe_index = np.unique(recipes, return_inverse=True)
    joint = np.zeros((vocab_size, len(recipe_values)), dtype=np.float64)
    np.add.at(joint, (messages, recipe_index), 1.0)
    total = joint.sum()

    # Optimal deterministic decoder: each symbol votes for its modal recipe.
    accuracy = joint.max(axis=1).sum() / total
    baseline = joint.sum(axis=0).max() / total

    p_joint = joint / total
    p_msg = p_joint.sum(axis=1, keepdims=True)
    p_recipe = p_joint.sum(axis=0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = p_joint * np.log2(p_joint / (p_msg * p_recipe))
    return {
        "accuracy": float(accuracy),
        "baseline": float(baseline),
        "mutual_information_bits": float(np.nansum(terms)),
        "num_samples": int(total),
        "recipe_values": recipe_values,
        "joint": joint,
    }


def add_header(frame, label, step, action_names, message_pair, total_return,
               shaped_return):
    header_height = 94
    image = Image.fromarray(np.asarray(frame, dtype=np.uint8))
    canvas = Image.new("RGB", (image.width, image.height + header_height), (18, 20, 28))
    canvas.paste(image, (0, header_height))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((7, 6), f"joint_comm | {label} | primitive step {step}",
              fill=(245, 245, 245), font=font)
    draw.text((7, 24), f"A0: {action_names[0]}  A1: {action_names[1]}",
              fill=(150, 220, 255), font=font)
    draw.text((7, 42), f"msg A0->A1: {message_pair[0]}   A1->A0: {message_pair[1]}",
              fill=(255, 205, 130), font=font)
    draw.text((7, 60), f"[sparse] team return: {total_return:.1f}",
              fill=(170, 245, 185), font=font)
    draw.text((7, 78), f"[shaped] team return: {shaped_return:.1f}",
              fill=(255, 205, 130), font=font)
    return np.asarray(canvas)


def run_episode(args, config, env, policy, seed, gif_output_path, stats):
    """Roll out one deterministic episode, accumulating into `stats`."""
    key = jax.random.PRNGKey(seed)
    obs, log_state = env.reset(key)
    state = log_state.env_state
    rollout_env = env._env._env
    states = [state]
    action_labels = [("wait", "wait")]
    message_labels = [(0, 0)]
    returns = [0.0]
    shaped_returns = [0.0]
    total_return = 0.0
    shaped_return = 0.0
    max_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 800))
    vocab_size = int(config["VOCAB_SIZE"])

    actor_hidden, comm_hidden = policy.initial_carries()
    last_message = jnp.zeros((env.num_agents,), dtype=jnp.int32)
    last_done = jnp.zeros((env.num_agents,), dtype=jnp.bool_)
    segment_symbols = [[] for _ in range(env.num_agents)]
    names = list(env.macro_action_names)
    ingredient_macros = {
        index for index, name in enumerate(names) if name.startswith("get_ingredient_")
    }

    for step in range(max_steps):
        key, route_key, step_key = jax.random.split(key, 3)
        # Kept because the intervention probe must replay the listening half
        # from the PRE-step carry, not the one the real step already advanced.
        pre_actor_hidden = actor_hidden
        out = policy.step(
            obs, env, actor_hidden, comm_hidden, last_done, last_message, route_key
        )
        actor_hidden = out["actor_hidden"]
        comm_hidden = out["comm_hidden"]
        last_message = out["message"]
        actions = out["actions"]

        if args.intervene:
            changed, swing, boundary = probe_intervention(
                policy, out["obs_stack"], pre_actor_hidden, out["summary"],
                last_done, out["action_mask"], out["macro_done"], vocab_size,
            )
            stats["intervention_changed"] += changed.astype(np.int64)
            stats["intervention_swing"] += swing * boundary
            stats["intervention_decisions"] += boundary.astype(np.int64)

        recipe = int(np.asarray(state.recipe))
        if args.decode:
            # One (symbol, recipe) sample per agent per macro boundary -- the
            # only moments at which a fresh message is actually emitted.
            for agent_idx in range(env.num_agents):
                if bool(np.asarray(out["macro_done"])[agent_idx]):
                    stats["decode_messages"][agent_idx].append(
                        int(np.asarray(out["message"])[agent_idx])
                    )
                    stats["decode_recipes"][agent_idx].append(recipe)
                    segment_symbols[agent_idx].append(
                        int(np.asarray(out["message"])[agent_idx])
                    )

        # Ingredient choice vs the TRUE recipe. This is the number the training
        # DELIVERY curve cannot give you: that reward is
        # (delivery rate) x (fraction correct), so a doubling is equally
        # consistent with a working protocol and with an unchanged protocol
        # that simply delivers twice as often. Conditioning the choice on the
        # recipe divides the rate out.
        for agent_idx in range(env.num_agents):
            if bool(np.asarray(out["macro_done"])[agent_idx]):
                chosen = int(np.asarray(out["actions"])[agent_idx])
                if chosen in ingredient_macros:
                    stats["ingredient_choices"].append((recipe, chosen))

        action_names = tuple(
            env.macro_action_names[int(a)] for a in np.asarray(actions)
        )
        message_pair = tuple(int(m) for m in np.asarray(out["message"]))
        env_actions = {a: actions[i] for i, a in enumerate(env.agents)}
        raw_obs, state, reward, done, info = rollout_env.step_env(
            step_key, state, env_actions
        )
        obs = env._env._augment(raw_obs, state)
        # Feeds the GRUs next step so carries reset at episode ends, matching
        # how the trainer threads prev_done.
        last_done = jnp.full(
            (env.num_agents,), bool(np.asarray(done["__all__"])), dtype=jnp.bool_
        )
        total_return += float(np.mean([np.asarray(reward[a]) for a in env.agents]))
        shaped_return += float(
            np.mean([np.asarray(info["shaped_reward"][a]) for a in env.agents])
        )
        # Also fire on the LAST step: otherwise the progress stream stops
        # up to 49 steps early and a late delivery never appears in it,
        # even though it is counted in the totals below.
        if step % 50 == 0 or step == max_steps - 1 or bool(np.asarray(done["__all__"])):
            print(f"Step {step}: [sparse] return={total_return:.1f}  "
                  f"[shaped] return={shaped_return:.1f}")

        breakdown = info["reward_breakdown"]
        for idx, comp in enumerate(REWARD_COMPONENT_KEYS):
            stats["reward_totals"][:, idx] += np.asarray(breakdown[comp])
        for agent_idx, agent in enumerate(env.agents):
            if bool(np.asarray(info["macro_action_started"][agent])):
                macro_idx = int(np.asarray(info["current_macro_action"][agent]))
                stats["macro_counts"][agent_idx, macro_idx] += 1
        for agent_idx, symbol in enumerate(message_pair):
            stats["message_counts"][agent_idx, symbol] += 1

        new_recipe = int(np.asarray(state.recipe))
        if new_recipe != recipe:
            # A recipe run just ended: emit ONE decoding sample per agent for it.
            for agent_idx in range(env.num_agents):
                if segment_symbols[agent_idx]:
                    values, counts = np.unique(
                        segment_symbols[agent_idx], return_counts=True
                    )
                    stats["segment_messages"][agent_idx].append(
                        int(values[counts.argmax()])
                    )
                    stats["segment_recipes"][agent_idx].append(recipe)
                segment_symbols[agent_idx] = []
        stats["correct_deliveries"] += float(
            np.sum(np.asarray(info["reward_breakdown"]["DELIVERY"]))
        ) / 20.0

        if gif_output_path is not None:
            states.append(state)
            action_labels.append(action_names)
            message_labels.append(message_pair)
            returns.append(total_return)
            shaped_returns.append(shaped_return)
        if bool(np.asarray(done["__all__"])):
            break

    for agent_idx in range(env.num_agents):
        if segment_symbols[agent_idx]:
            values, counts = np.unique(segment_symbols[agent_idx], return_counts=True)
            stats["segment_messages"][agent_idx].append(int(values[counts.argmax()]))
            stats["segment_recipes"][agent_idx].append(int(np.asarray(state.recipe)))

    episode_length = len(states) - 1

    if gif_output_path is not None:
        frame_indices = list(range(0, len(states), args.frame_skip))
        if frame_indices[-1] != len(states) - 1:
            frame_indices.append(len(states) - 1)
        selected = [states[i] for i in frame_indices]
        visualizer = OvercookedV3Visualizer(rollout_env, tile_size=args.tile_size)
        # Render in chunks rather than stacking the whole trajectory at once.
        rendered = []
        chunk_size = max(1, args.render_chunk_size)
        for start in range(0, len(selected), chunk_size):
            chunk = selected[start:start + chunk_size]
            stacked = jax.tree.map(lambda *v: jnp.stack(v), *chunk)
            rendered.extend(jax.device_get(visualizer.render_sequence(stacked)))
            del stacked
        frames = [
            add_header(frame, args.checkpoint_label, s, action_labels[s],
                       message_labels[s], returns[s], shaped_returns[s])
            for frame, s in zip(rendered, frame_indices)
        ]
        gif_output_path.parent.mkdir(parents=True, exist_ok=True)
        gif_frames = [Image.fromarray(f) for f in frames]
        gif_frames[0].save(
            gif_output_path, format="GIF", save_all=True,
            append_images=gif_frames[1:], duration=args.frame_ms,
            loop=0, optimize=False,
        )
        print(f"Saved {gif_output_path} ({len(frames)} frames, "
              f"[sparse] return={total_return:.1f} "
              f"[shaped] return={shaped_return:.1f})")

    return total_return, shaped_return, episode_length


AGENT_COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def plot_grouped_bars(ax, labels, values_by_agent, agent_labels, title, ylabel):
    """Grouped bar chart with one bar per agent in each labelled group."""
    num_agents = len(values_by_agent)
    bar_width = 0.8 / num_agents
    x = np.arange(len(labels))
    for agent_idx, values in enumerate(values_by_agent):
        offset = (agent_idx - (num_agents - 1) / 2) * bar_width
        ax.bar(x + offset, values, width=bar_width,
               label=agent_labels[agent_idx],
               color=AGENT_COLORS[agent_idx % len(AGENT_COLORS)])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.legend()


def save_histograms(args, env, vocab_size, stats):
    """Three-panel figure: reward by type, macro selections, symbols sent."""
    agent_labels = [f"agent_{i}" for i in range(env.num_agents)]
    fig, (ax_reward, ax_macro, ax_message) = plt.subplots(1, 3, figsize=(24, 6))
    plot_grouped_bars(
        ax_reward, list(REWARD_COMPONENT_KEYS),
        [stats["reward_totals"][i] for i in range(env.num_agents)], agent_labels,
        f"Total reward by type over {args.num_episodes} episode(s)", "total reward",
    )
    plot_grouped_bars(
        ax_macro, list(env.macro_action_names),
        [stats["macro_counts"][i] for i in range(env.num_agents)], agent_labels,
        f"Macro action selections over {args.num_episodes} episode(s)",
        "times selected",
    )
    plot_grouped_bars(
        ax_message, [str(s) for s in range(vocab_size)],
        [stats["message_counts"][i] for i in range(env.num_agents)], agent_labels,
        f"Comm messages sent over {args.num_episodes} episode(s)", "times sent",
    )
    fig.suptitle(f"joint_comm | {args.checkpoint_label} | COMM_MODE={args.comm_mode}")
    fig.tight_layout()
    args.histogram_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.histogram_output, dpi=150)
    print(f"Saved {args.histogram_output}")


def print_summary(args, env, vocab_size, stats):
    """Reward/macro/message tallies plus, when requested, the comm diagnostics."""
    agent_labels = [f"agent_{i}" for i in range(env.num_agents)]

    print("\nReward totals by type:")
    for agent_idx, agent in enumerate(agent_labels):
        print(f"  {agent}:")
        for idx, comp in enumerate(REWARD_COMPONENT_KEYS):
            print(f"    {comp}: {stats['reward_totals'][agent_idx, idx]:.2f}")

    print("\nMacro action selection counts:")
    for agent_idx, agent in enumerate(agent_labels):
        print(f"  {agent}:")
        for macro_idx, name in enumerate(env.macro_action_names):
            count = int(stats["macro_counts"][agent_idx, macro_idx])
            if count > 0:
                print(f"    {name}: {count}")

    print("\nComm message symbol counts:")
    for agent_idx, agent in enumerate(agent_labels):
        counts = stats["message_counts"][agent_idx]
        print(f"  {agent}:")
        for symbol in range(vocab_size):
            if int(counts[symbol]) > 0:
                print(f"    symbol {symbol}: {int(counts[symbol])}")
        # A channel pinned to one symbol carries nothing, whatever the return is.
        if int(counts.sum()) > 0 and int((counts > 0).sum()) <= 1:
            print("    WARNING: channel collapsed to a single symbol "
                  "-- this protocol transmits zero information.")

    if args.intervene:
        print("\nPositive listening (counterfactual message intervention):")
        print("  Fraction of macro decisions whose chosen action changes when "
              "the received symbol is swapped.")
        for agent_idx, agent in enumerate(agent_labels):
            decisions = int(stats["intervention_decisions"][agent_idx])
            if decisions == 0:
                print(f"  {agent}: no macro decisions recorded")
                continue
            changed = int(stats["intervention_changed"][agent_idx])
            swing = stats["intervention_swing"][agent_idx] / decisions
            print(f"  {agent}: flip_rate={changed / decisions:.4f} "
                  f"({changed}/{decisions})  mean_logit_swing={swing:.4f}")
        print("  flip_rate == 0 means the listener ignores the channel "
              "entirely; any return advantage came from elsewhere.")

    if args.decode:
        print("\nPositive signaling (message -> recipe decoding):")
        for agent_idx, agent in enumerate(agent_labels):
            step_report = decoding_report(
                stats["decode_messages"][agent_idx],
                stats["decode_recipes"][agent_idx], vocab_size,
            )
            segment_report = decoding_report(
                stats["segment_messages"][agent_idx],
                stats["segment_recipes"][agent_idx], vocab_size,
            )
            if step_report is None:
                print(f"  {agent}: no messages recorded")
                continue
            print(f"  {agent}:")
            print(f"    per-step    n={step_report['num_samples']:5d}  "
                  f"acc={step_report['accuracy']:.4f} "
                  f"base={step_report['baseline']:.4f}  "
                  f"MI={step_report['mutual_information_bits']:.4f} bits")
            if segment_report is not None:
                print(f"    per-SEGMENT n={segment_report['num_samples']:5d}  "
                      f"acc={segment_report['accuracy']:.4f} "
                      f"base={segment_report['baseline']:.4f}  "
                      f"MI={segment_report['mutual_information_bits']:.4f} bits"
                      "   <- independent samples")
        # Per-step rows are sampled at every macro boundary, so whichever recipe
        # happened to stay active longer dominates them and the baseline they
        # print is NOT the recipe prior. Per-segment takes one sample per
        # contiguous recipe run.
        print("  Quote the per-SEGMENT row. Accuracy at baseline with ~0 bits "
              "means the speaker never encoded the recipe.")

    # The headline number: accuracy with the delivery RATE divided out.
    print("\nIngredient choice vs the true recipe (protocol ACCURACY):")
    choices = np.asarray(stats["ingredient_choices"], dtype=np.int64)
    if choices.size == 0:
        print("  no ingredient macros were selected")
    else:
        macro_names = list(env.macro_action_names)
        ingredient_names = [n for n in macro_names if n.startswith("get_ingredient_")]
        print("  recipe  " + "".join(f"{n:>20s}" for n in ingredient_names))
        for recipe_value in np.unique(choices[:, 0]):
            rows = choices[choices[:, 0] == recipe_value]
            counts = [
                int((rows[:, 1] == macro_names.index(n)).sum())
                for n in ingredient_names
            ]
            total = max(sum(counts), 1)
            cells = "".join(f"{c:>12d} ({c / total:4.0%})" for c in counts)
            print(f"  {recipe_value:<8d}" + cells)
        print("  Same split across recipes => the doer is NOT conditioning on "
              "the recipe; any DELIVERY gain came from a higher delivery rate, "
              "not from communication.")
    print(f"\nCorrect deliveries over {args.num_episodes} episode(s): "
          f"{stats['correct_deliveries']:.0f} "
          f"({stats['correct_deliveries'] / max(args.num_episodes, 1):.2f} per episode)")


def main():
    """Load a joint run, roll out episodes, and emit the requested outputs."""
    args = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "config.yaml"), resolve=True
    )
    if args.checkpoint_label is None:
        args.checkpoint_label = args.run_dir.name

    # Fail with an explanation rather than a bare KeyError deeper down. A joint
    # run is identified by having VOCAB_SIZE but NOT FROZEN_ACTOR_PATH; the
    # latter marks the two-stage runs that the other comm script handles.
    if "VOCAB_SIZE" not in config:
        raise ValueError(
            f"{args.run_dir} is not a comm run (no VOCAB_SIZE in config.yaml). "
            "For a non-comm macro run use scripts/visualize_macro_mappo_rollout.py."
        )
    if "FROZEN_ACTOR_PATH" in config:
        raise ValueError(
            f"{args.run_dir} is a two-stage comm run (it has FROZEN_ACTOR_PATH), "
            "whose checkpoint holds comm params only. Use "
            "scripts/visualize_macro_mappo_rollout_comm.py for it. This script "
            "evaluates joint runs from mappo_macro_boundary_joint_comm.py, whose "
            "checkpoint holds both actor and comm params in one tree."
        )

    if args.comm_mode is None:
        args.comm_mode = config.get("COMM_MODE", "normal")
    # Eval runs one env at a time, so there is no other environment to draw a
    # message from -- "shuffled" silently degenerates to "normal" here.
    if args.comm_mode == "shuffled":
        print("WARNING: COMM_MODE=shuffled is a no-op at eval (single "
              "environment, nothing to shuffle against). It behaves as "
              "'normal'. Use --comm-mode self or constant for a severed-channel "
              "control against these weights.")

    env = build_env(config)

    # Both flags change parameter shapes, so they must mirror the training run.
    use_rnn = bool(config.get("USE_RNN", False))
    actor = (
        ActorRNN(env.num_actions, int(config["HIDDEN_SIZE"]))
        if use_rnn else Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
    )
    vocab_size = int(config["VOCAB_SIZE"])
    comm_module = CommModule(
        hidden_size=int(config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"])),
        vocab_size=vocab_size,
        action_dim=env.num_actions,
        message_embed_dim=int(config.get("MESSAGE_EMBED_DIM", 8)),
        use_memory=bool(config.get("COMM_USE_MEMORY", False)),
    )

    actor_path = args.actor_path or (args.run_dir / "best_actor.safetensors")
    params = load_params(actor_path)
    if not ("actor" in params and "comm" in params):
        # The common cause is a run directory whose config.yaml was copied in
        # by hand over weights produced by a different trainer, so say what the
        # file actually contains rather than just that it is not what we want.
        looks_like_bare_network = list(params) == ["params"]
        raise ValueError(
            f"{actor_path} holds top-level keys {sorted(params)}, not a joint "
            "{'actor', 'comm'} tree.\n"
            + (
                "It contains a single bare network -- an actor with no comm "
                "module attached, i.e. weights from mappo_macro_boundary.py "
                "(no-comm) or the comm-only checkpoint of "
                "mappo_macro_boundary_comm.py. A config.yaml naming a joint run "
                "does not make the weights joint; check that the weights and "
                "the config in this directory came from the same run (compare "
                "their timestamps), then train with "
                "baselines/MAPPO/mappo_macro_boundary_joint_comm.py.\n"
                if looks_like_bare_network else ""
            )
            + "Joint runs save the actor and comm halves together in one "
            "checkpoint."
        )
    print(f"Loaded joint params from {actor_path} (COMM_MODE={args.comm_mode})")

    stats = {
        "reward_totals": np.zeros(
            (env.num_agents, len(REWARD_COMPONENT_KEYS)), dtype=np.float64),
        "macro_counts": np.zeros((env.num_agents, env.num_actions), dtype=np.int64),
        "message_counts": np.zeros((env.num_agents, vocab_size), dtype=np.int64),
        "intervention_changed": np.zeros((env.num_agents,), dtype=np.int64),
        "intervention_swing": np.zeros((env.num_agents,), dtype=np.float64),
        "intervention_decisions": np.zeros((env.num_agents,), dtype=np.int64),
        "decode_messages": [[] for _ in range(env.num_agents)],
        "decode_recipes": [[] for _ in range(env.num_agents)],
        # One entry per contiguous recipe run, so decoding statistics are not
        # weighted by how long a recipe happened to stay active.
        "segment_messages": [[] for _ in range(env.num_agents)],
        "segment_recipes": [[] for _ in range(env.num_agents)],
        # (recipe, chosen ingredient macro) at every ingredient decision --
        # accuracy with the delivery RATE divided out.
        "ingredient_choices": [],
        "correct_deliveries": 0,
    }

    policy = JointPolicy(
        actor, comm_module, params, config, env.num_agents, args.comm_mode
    )

    episode_returns, episode_shaped, episode_lengths = [], [], []
    for episode_index in range(args.num_episodes):
        gif_path = (
            episode_output_path(args.gif_output, episode_index, args.num_episodes)
            if args.gif_output is not None else None
        )
        total_return, shaped_return, length = run_episode(
            args, config, env, policy, args.seed + episode_index, gif_path, stats
        )
        episode_returns.append(total_return)
        episode_shaped.append(shaped_return)
        episode_lengths.append(length)

    if args.num_episodes > 1:
        returns = np.asarray(episode_returns)
        shaped = np.asarray(episode_shaped)
        lengths = np.asarray(episode_lengths)
        print(f"\n{args.num_episodes} episodes: "
              f"[sparse] return mean={returns.mean():.2f} std={returns.std():.2f} "
              f"min={returns.min():.2f} max={returns.max():.2f} | "
              f"[shaped] return mean={shaped.mean():.2f} std={shaped.std():.2f} | "
              f"length mean={lengths.mean():.1f}")

    print_summary(args, env, vocab_size, stats)
    if args.histogram_output is not None:
        save_histograms(args, env, vocab_size, stats)


if __name__ == "__main__":
    main()
