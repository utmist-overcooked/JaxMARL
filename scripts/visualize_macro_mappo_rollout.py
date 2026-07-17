"""Render a deterministic rollout from a trained macro MAPPO actor."""

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

from mappo_macro_common import Actor, ReplanActor, build_env
from mappo_macro_replan import REPLAN


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("boundary", "every_step", "replan"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-label", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame-skip", type=int, default=1)
    parser.add_argument("--frame-ms", type=int, default=150)
    parser.add_argument("--tile-size", type=int, default=40)
    return parser.parse_args()


def select_actions(variant, actor, params, obs, env):
    actor_obs = jnp.stack([obs[agent] for agent in env.agents])
    action_mask = obs["action_mask"].astype(jnp.bool_)

    if variant != "replan":
        logits = actor.apply(params, actor_obs)
        return jnp.argmax(jnp.where(action_mask, logits, -1e9), axis=-1)

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


def add_header(frame, variant, checkpoint_label, step, action_names, total_return):
    header_height = 58
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
        f"sparse team return: {total_return:.1f}",
        fill=(170, 245, 185),
        font=font,
    )
    return np.asarray(canvas)


def main():
    args = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "config.yaml"), resolve=True
    )
    env = build_env(config)
    actor = (
        ReplanActor(env.num_actions, int(config["HIDDEN_SIZE"]))
        if args.variant == "replan"
        else Actor(env.num_actions, int(config["HIDDEN_SIZE"]))
    )
    params = load_params(args.run_dir / "final_actor.safetensors")

    key = jax.random.PRNGKey(args.seed)
    obs, state = env.reset(key)
    states = [state.env_state]
    action_labels = [("wait", "wait")]
    returns = [0.0]
    total_return = 0.0
    max_steps = int(config.get("ENV_KWARGS", {}).get("max_steps", 400))

    for _ in range(max_steps):
        actions = select_actions(args.variant, actor, params, obs, env)
        action_names = tuple(
            env.macro_action_names[int(action)] for action in np.asarray(actions)
        )
        env_actions = {
            agent: actions[index] for index, agent in enumerate(env.agents)
        }
        key, step_key = jax.random.split(key)
        obs, state, reward, _, _ = env.step(step_key, state, env_actions)
        total_return += float(
            np.mean([np.asarray(reward[agent]) for agent in env.agents])
        )
        states.append(state.env_state)
        action_labels.append(action_names)
        returns.append(total_return)

    frame_indices = list(range(0, len(states), args.frame_skip))
    if frame_indices[-1] != len(states) - 1:
        frame_indices.append(len(states) - 1)
    selected_states = [states[index] for index in frame_indices]
    stacked_states = jax.tree.map(lambda *values: jnp.stack(values), *selected_states)
    base_env = env._env._env
    visualizer = OvercookedV3Visualizer(base_env, tile_size=args.tile_size)
    rendered = jax.device_get(visualizer.render_sequence(stacked_states))
    frames = [
        add_header(
            frame,
            args.variant,
            args.checkpoint_label,
            step,
            action_labels[step],
            returns[step],
        )
        for frame, step in zip(rendered, frame_indices)
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    gif_frames = [Image.fromarray(frame) for frame in frames]
    gif_frames[0].save(
        args.output,
        save_all=True,
        append_images=gif_frames[1:],
        duration=args.frame_ms,
        loop=0,
        optimize=False,
    )
    print(f"Saved {args.output} ({len(frames)} frames, return={total_return:.1f})")


if __name__ == "__main__":
    main()
