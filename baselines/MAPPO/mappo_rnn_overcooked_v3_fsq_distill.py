"""
MAPPO RNN for Overcooked V3 with FSQ communication and full-observation
teacher distillation.

Student actors see partial observations and exchange quantized messages.
The critic remains centralized over concatenated partial observations only.
"""

import datetime
import inspect
import shutil
import time
from pathlib import Path
import jax
import jax.api_util
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Callable, Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import jaxmarl
from jaxmarl.wrappers.baselines import LogWrapper, save_params
from jaxmarl.environments.overcooked_v3 import OvercookedV3, overcooked_v3_layouts
from jaxmarl.environments.overcooked_v3.common import DynamicObject
import hydra
from omegaconf import OmegaConf
import copy
import os
import sys
import wandb
import functools

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
try:
    from .fsq import FSQ
    from .fsq_viewer import (
        build_viewer_data,
        index_to_coord,
        state_summary,
        write_viewer_artifacts,
    )
except ImportError:
    from fsq import FSQ
    from fsq_viewer import (
        build_viewer_data,
        index_to_coord,
        state_summary,
        write_viewer_artifacts,
    )
from jaxmarl.wrappers.baselines import load_params

try:
    from utils.monitor import TrainingMonitor

    _MONITOR_AVAILABLE = True
except ImportError:
    _MONITOR_AVAILABLE = False


def load_actor_params(path):
    params = load_params(path)
    if "params" in params:
        return {"params": params["params"]}
    return {"params": params}


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x

        new_carry = self.initialize_carry(ins.shape[0], ins.shape[1])

        rnn_state = jnp.where(
            resets[:, np.newaxis],
            new_carry,
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=ins.shape[1])(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        cell = nn.GRUCell(features=hidden_size)
        return cell.initialize_carry(jax.random.PRNGKey(0), (batch_size, hidden_size))


class CNN(nn.Module):
    output_size: int = 64
    activation: Callable[..., Any] = nn.relu

    @nn.compact
    def __call__(self, x, train=False):
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=128,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)
        x = nn.Conv(
            features=8,
            kernel_size=(1, 1),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=16,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = nn.Conv(
            features=32,
            kernel_size=(3, 3),
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        x = x.reshape((x.shape[0], -1))

        x = nn.Dense(
            features=self.output_size,
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(x)
        x = self.activation(x)

        return x


class CommActorRNN(nn.Module):
    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = CNN(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = jax.vmap(embed_model)(obs)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        fsq = FSQ(levels=tuple(self.config["FSQ_LEVELS"]))
        time_steps, batch_size = embedding.shape[:2]
        if self.config.get("DISABLE_FSQ_COMM", False):
            msg_codes = jnp.zeros(
                (time_steps, batch_size, fsq.num_dimensions), dtype=embedding.dtype
            )
            msg_indices = jnp.zeros((time_steps, batch_size), dtype=jnp.int32)
            actor_input = embedding
        else:
            msg_logits = nn.Dense(
                fsq.num_dimensions,
                kernel_init=orthogonal(0.01),
                bias_init=constant(0.0),
            )(embedding)
            msg_codes, msg_indices = fsq.quantize_and_index(msg_logits)

            num_agents = self.config["NUM_AGENTS"]
            num_envs = batch_size // num_agents
            msg_by_agent = msg_codes.reshape(
                time_steps, num_agents, num_envs, fsq.num_dimensions
            )
            partner_msg = jnp.flip(msg_by_agent, axis=1).reshape(
                time_steps, batch_size, fsq.num_dimensions
            )
            actor_input = jnp.concatenate([embedding, partner_msg], axis=-1)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(actor_input)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return hidden, pi, msg_codes, msg_indices


class TeacherActorRNN(nn.Module):
    """Actor architecture used by mappo_rnn_overcooked_v3_full_obs.py."""

    action_dim: Sequence[int]
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.config["ACTIVATION"] == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        embed_model = nn.vmap(
            CNN,
            variable_axes={"params": None},
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )(
            output_size=self.config["GRU_HIDDEN_DIM"],
            activation=activation,
        )
        embedding = embed_model(obs)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)

        return hidden, pi


class CriticRNN(nn.Module):
    config: Dict

    @nn.compact
    def __call__(self, hidden, x):
        world_state, dones = x

        embedding = nn.Dense(
            self.config["GRU_HIDDEN_DIM"],
            kernel_init=orthogonal(jnp.sqrt(2)),
            bias_init=constant(0.0),
        )(world_state)
        embedding = nn.relu(embedding)

        embedding = nn.LayerNorm()(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        critic = nn.Dense(
            self.config["FC_DIM_SIZE"],
            kernel_init=orthogonal(2),
            bias_init=constant(0.0),
        )(embedding)
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, jnp.squeeze(critic, axis=-1)


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    teacher_logits: jnp.ndarray
    distill_weight: jnp.ndarray
    comm_code: jnp.ndarray
    comm_index: jnp.ndarray
    info: jnp.ndarray


def batchify(x: dict, agent_list, num_actors):
    x = jnp.stack([x[a] for a in agent_list])
    return x.reshape((num_actors, -1))


def unbatchify(x: jnp.ndarray, agent_list, num_envs, num_actors):
    x = x.reshape((num_actors, num_envs, -1))
    return {a: x[i] for i, a in enumerate(agent_list)}


def _overcooked_env_kwargs(config: Dict) -> Dict:
    return dict(config["ENV_KWARGS"])


def _filter_overcooked_env_kwargs(env_kwargs: Dict) -> Dict:
    signature = inspect.signature(OvercookedV3.__init__)
    allowed = set(signature.parameters) - {"self"}
    return {key: value for key, value in env_kwargs.items() if key in allowed}


def _sanitize_name(value: str) -> str:
    safe = "".join(char if char.isalnum() or char in "._-" else "_" for char in value)
    return safe.strip("._") or "run"


def _checkpoint_gif_namespace(config: Dict) -> str:
    return (
        config.get("CHECKPOINT_GIF_NAMESPACE")
        or "checkpoint_rollouts_by_recipe"
    ).strip("/")


def _recipe_label(recipe: Sequence[int] | None) -> str:
    if recipe is None:
        return "sampled"
    return "-".join(str(int(ingredient)) for ingredient in recipe)


def _checkpoint_recipe_variants(config: Dict) -> list[dict[str, Any]]:
    env_kwargs = _filter_overcooked_env_kwargs(_overcooked_env_kwargs(config))
    order_queue_enabled = bool(env_kwargs.get("enable_order_queue", False))
    if order_queue_enabled:
        return [{"recipe": None, "recipe_index": None, "recipe_label": "sampled"}]

    env = OvercookedV3(**env_kwargs)
    recipes = [
        tuple(int(ingredient) for ingredient in recipe)
        for recipe in env.layout.possible_recipes
    ]
    if len(recipes) <= 1:
        return [{"recipe": None, "recipe_index": None, "recipe_label": "sampled"}]

    return [
        {
            "recipe": recipe,
            "recipe_index": recipe_index,
            "recipe_label": _recipe_label(recipe),
        }
        for recipe_index, recipe in enumerate(recipes)
    ]


def _checkpoint_policy_step(network: CommActorRNN, action_dim: int):
    @jax.jit
    def policy_step(params, hstate, obs_batch, done_batch, rng, epsilon):
        ac_in = (obs_batch[None, :], done_batch[None, :])
        hstate, pi, comm_code, comm_index = network.apply(params, hstate, ac_in)
        greedy_action = jnp.argmax(pi.logits, axis=-1).squeeze(0)
        rng_random, rng_mask = jax.random.split(rng)
        random_action = jax.random.randint(
            rng_random,
            shape=greedy_action.shape,
            minval=0,
            maxval=action_dim,
        )
        explore = jax.random.uniform(rng_mask, greedy_action.shape) < epsilon
        action = jnp.where(explore, random_action, greedy_action)
        return hstate, action, comm_code.squeeze(axis=0), comm_index.squeeze(axis=0)

    return policy_step


def checkpoint_updates(num_updates: int, checkpoint_count: int) -> tuple[int, ...]:
    if num_updates <= 0 or checkpoint_count <= 0:
        return ()

    count = min(num_updates, checkpoint_count)
    updates = np.rint(np.linspace(1, num_updates, count)).astype(np.int32)
    return tuple(sorted({int(update) for update in updates}))


def checkpoint_actor_paths(
    *,
    wandb_dir: str,
    run_name: str,
    updates: Sequence[int],
) -> list[tuple[int, Path]]:
    wanted_updates = set(int(update) for update in updates)
    models_dir = Path(wandb_dir) / "models"
    if not models_dir.exists():
        return []

    paths = []
    for ckpt_dir in models_dir.iterdir():
        if not ckpt_dir.is_dir() or not ckpt_dir.name.startswith(f"{run_name}_"):
            continue
        for actor_path in ckpt_dir.glob("*_actor.safetensors"):
            try:
                update = int(actor_path.name.split("_", 1)[0])
            except ValueError:
                continue
            if update in wanted_updates:
                paths.append((update, actor_path))

    return sorted(paths, key=lambda item: item[0])


def _annotate_checkpoint_frame(
    frame,
    *,
    layout: str,
    update: int,
    step: int,
    reward: float,
    epsilon: float,
    recipe_label: str | None = None,
):
    from PIL import Image, ImageDraw

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    recipe_text = f" | recipe {recipe_label}" if recipe_label else ""
    text = (
        f"{layout} | update {update} | step {step:03d} | "
        f"eps {epsilon:g} | reward {reward:.0f}{recipe_text}"
    )
    bbox = draw.textbbox((0, 0), text)
    x1 = min(image.width, bbox[2] + 12)
    y1 = min(image.height, bbox[3] + 10)
    draw.rectangle((4, 4, x1, y1), fill=(0, 0, 0))
    draw.text((8, 7), text, fill=(255, 255, 255))
    return image


def _checkpoint_gif_grid_shape(count: int) -> tuple[int, int]:
    if count <= 0:
        raise ValueError("Cannot tile zero checkpoint GIFs")
    columns = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / columns))
    return rows, columns


def _combine_checkpoint_gifs(
    *,
    gif_paths: Sequence[os.PathLike | str],
    output_file: os.PathLike | str,
    fps: int,
) -> dict[str, Any]:
    from PIL import Image, ImageSequence

    if not gif_paths:
        raise ValueError("Cannot combine an empty checkpoint GIF list")

    gif_frames = []
    cell_width = 0
    cell_height = 0
    max_frame_count = 0
    for gif_path in gif_paths:
        with Image.open(gif_path) as image:
            frames = [
                frame.convert("RGB").copy()
                for frame in ImageSequence.Iterator(image)
            ]
        if not frames:
            raise ValueError(f"Checkpoint GIF has no frames: {gif_path}")
        gif_frames.append(frames)
        max_frame_count = max(max_frame_count, len(frames))
        cell_width = max(cell_width, max(frame.width for frame in frames))
        cell_height = max(cell_height, max(frame.height for frame in frames))

    rows, columns = _checkpoint_gif_grid_shape(len(gif_frames))
    combined_frames = []
    for frame_index in range(max_frame_count):
        canvas = Image.new(
            "RGB",
            (columns * cell_width, rows * cell_height),
            color=(0, 0, 0),
        )
        for gif_index, frames in enumerate(gif_frames):
            row = gif_index // columns
            column = gif_index % columns
            frame = frames[min(frame_index, len(frames) - 1)]
            canvas.paste(frame, (column * cell_width, row * cell_height))
        combined_frames.append(canvas)

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    combined_frames[0].save(
        output_file,
        save_all=True,
        append_images=combined_frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )
    return {
        "gif": str(output_file),
        "rows": rows,
        "columns": columns,
        "frame_count": max_frame_count,
        "cell_width": cell_width,
        "cell_height": cell_height,
    }


def render_checkpoint_gif(
    *,
    actor_params: Dict,
    config: Dict,
    update: int,
    output_file: os.PathLike | str,
    run_name: str | None = None,
    forced_recipe: Sequence[int] | None = None,
    recipe_index: int | None = None,
    fsq_output_dir: os.PathLike | str | None = None,
    actor_path: os.PathLike | str | None = None,
    config_path: os.PathLike | str | None = None,
) -> dict[str, Any]:
    from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

    env_kwargs = _overcooked_env_kwargs(config)
    layout = str(config["ENV_KWARGS"]["layout"])
    max_steps = int(
        config.get("CHECKPOINT_GIF_MAX_STEPS")
        or env_kwargs.get("max_steps")
        or 300
    )
    epsilon = float(config.get("CHECKPOINT_GIF_EPSILON", 0.0))
    seed = int(config.get("CHECKPOINT_GIF_SEED", 0))
    fps = int(config.get("CHECKPOINT_GIF_FPS", 8))
    tile_size = int(config.get("CHECKPOINT_GIF_TILE_SIZE", 32))

    env_kwargs["max_steps"] = max_steps
    env_kwargs = _filter_overcooked_env_kwargs(env_kwargs)
    env = OvercookedV3(**env_kwargs)
    action_dim = env.action_space(env.agents[0]).n
    network_config = {
        "GRU_HIDDEN_DIM": int(config["GRU_HIDDEN_DIM"]),
        "FC_DIM_SIZE": int(config["FC_DIM_SIZE"]),
        "ACTIVATION": config["ACTIVATION"],
        "FSQ_LEVELS": tuple(config["FSQ_LEVELS"]),
        "NUM_AGENTS": env.num_agents,
        "DISABLE_FSQ_COMM": bool(config.get("DISABLE_FSQ_COMM", False)),
    }
    network = CommActorRNN(action_dim, config=network_config)
    policy_step = _checkpoint_policy_step(network, action_dim)

    rng = jax.random.PRNGKey(seed + update)
    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng)
    recipe_label = _recipe_label(forced_recipe) if forced_recipe is not None else None
    if forced_recipe is not None:
        recipe_encoding = DynamicObject.get_recipe_encoding(
            jnp.asarray(forced_recipe, dtype=jnp.int32)
        )
        state = state.replace(recipe=recipe_encoding)
        obs = env.get_obs(state)
    hstate = ScannedRNN.initialize_carry(
        env.num_agents,
        network_config["GRU_HIDDEN_DIM"],
    )
    done_batch = jnp.zeros(env.num_agents, dtype=bool)
    viz = OvercookedV3Visualizer(env, tile_size=tile_size)

    frames = []
    total_reward = 0.0
    initial_frame = np.array(viz.render_state(state))
    frames.append(
        _annotate_checkpoint_frame(
            initial_frame,
            layout=layout,
            update=update,
            step=0,
            reward=total_reward,
            epsilon=epsilon,
            recipe_label=recipe_label,
        )
    )

    fsq_enabled = (
        bool(config.get("CHECKPOINT_FSQ_VIEWER", True))
        and not bool(config.get("DISABLE_FSQ_COMM", False))
        and fsq_output_dir is not None
    )
    fsq = FSQ(levels=tuple(config["FSQ_LEVELS"]))
    fsq_counts = np.zeros((fsq.codebook_size,), dtype=np.int64)
    fsq_dim_counts = np.zeros(
        (fsq.num_dimensions, max(config["FSQ_LEVELS"])), dtype=np.int64
    )
    fsq_examples = {i: [] for i in range(fsq.codebook_size)}
    fsq_examples_dir = None
    if fsq_enabled:
        fsq_output_dir = Path(fsq_output_dir)
        fsq_examples_dir = fsq_output_dir / "examples"
        fsq_examples_dir.mkdir(parents=True, exist_ok=True)

    steps = 0
    for step in range(1, max_steps + 1):
        pre_action_state = state
        pre_action_frame = np.array(viz.render_state(pre_action_state))
        message_step = step - 1
        obs_batch = jnp.stack([obs[agent] for agent in env.agents])
        rng, act_rng = jax.random.split(rng)
        hstate, action, comm_code, comm_index = policy_step(
            actor_params,
            hstate,
            obs_batch,
            done_batch,
            act_rng,
            jnp.asarray(epsilon, dtype=jnp.float32),
        )
        env_action = {agent: int(action[idx]) for idx, agent in enumerate(env.agents)}

        if fsq_enabled and fsq_examples_dir is not None:
            comm_code_np = np.asarray(comm_code)
            comm_index_np = np.asarray(comm_index).astype(int)
            action_np = np.asarray(action).astype(int)
            for agent_idx, code_index in enumerate(comm_index_np.tolist()):
                fsq_counts[code_index] += 1
                coord = index_to_coord(code_index, list(config["FSQ_LEVELS"]))
                for dim, value in enumerate(coord):
                    fsq_dim_counts[dim, value] += 1
                image_name = (
                    f"code_{code_index:03d}_ep000_"
                    f"step{message_step:04d}_agent{agent_idx}.png"
                )
                image_path = fsq_examples_dir / image_name
                from PIL import Image

                Image.fromarray(pre_action_frame).save(image_path)
                fsq_examples[code_index].append(
                    {
                        "episode": 0,
                        "step": message_step,
                        "agent": int(agent_idx),
                        "image": f"examples/{image_name}",
                        "summary": state_summary(
                            pre_action_state,
                            agent_idx,
                            int(action_np[agent_idx]),
                        ),
                        "raw_code": comm_code_np[agent_idx].astype(float).tolist(),
                    }
                )

        rng, step_rng = jax.random.split(rng)
        obs, state, rewards, dones, _ = env.step(step_rng, state, env_action)
        done_batch = jnp.array([dones[agent] for agent in env.agents])
        total_reward += float(rewards[env.agents[0]])
        steps = step

        frame = np.array(viz.render_state(state))
        frames.append(
            _annotate_checkpoint_frame(
                frame,
                layout=layout,
                update=update,
                step=step,
                reward=total_reward,
                epsilon=epsilon,
                recipe_label=recipe_label,
            )
        )

        if bool(dones["__all__"]):
            break

    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=int(1000 / fps),
        loop=0,
    )

    row = {
        "update": update,
        "steps": steps,
        "reward": total_reward,
        "gif": str(output_file),
        "layout": layout,
        "max_steps": max_steps,
        "epsilon": epsilon,
        "recipe": recipe_label or "sampled",
        "recipe_index": recipe_index,
    }

    if fsq_enabled and fsq_output_dir is not None:
        fsq_output_dir = Path(fsq_output_dir)
        gif_copy = fsq_output_dir / output_file.name
        if gif_copy.resolve() != output_file.resolve():
            shutil.copy2(output_file, gif_copy)
        metadata = {
            "schema_version": 1,
            "artifact_type": "checkpoint_fsq_viewer",
            "run_name": run_name,
            "checkpoint_update": int(update),
            "actor_path": None if actor_path is None else str(actor_path),
            "config_path": None if config_path is None else str(config_path),
            "gif": gif_copy.name,
            "layout": layout,
            "max_steps": int(max_steps),
            "steps": int(steps),
            "reward": float(total_reward),
            "epsilon": float(epsilon),
            "seed": int(seed),
            "recipe": recipe_label or "sampled",
            "recipe_index": recipe_index,
            "forced_recipe": (
                None
                if forced_recipe is None
                else [int(ingredient) for ingredient in forced_recipe]
            ),
            "fsq_levels": [int(level) for level in config["FSQ_LEVELS"]],
        }
        data = build_viewer_data(
            layout=layout,
            levels=tuple(config["FSQ_LEVELS"]),
            codebook=np.asarray(fsq.codebook),
            counts=fsq_counts,
            examples=fsq_examples,
            dim_counts=fsq_dim_counts,
            metadata=metadata,
        )
        viewer_paths = write_viewer_artifacts(fsq_output_dir, data)
        row.update(
            {
                "fsq_viewer_dir": viewer_paths["viewer_dir"],
                "fsq_usage_json": viewer_paths["usage_json"],
                "fsq_index_html": viewer_paths["index_html"],
                "fsq_total_samples": int(fsq_counts.sum()),
                "fsq_nonzero_codes": int(np.count_nonzero(fsq_counts)),
            }
        )

    return row


def render_and_log_checkpoint_gif(
    *,
    actor_params: Dict,
    config: Dict,
    update: int,
    run_name: str,
    checkpoint_interval: int,
    rollout_index: int | None = None,
    actor_path: os.PathLike | str | None = None,
    config_path: os.PathLike | str | None = None,
) -> None:
    checkpoint_gif = config.get("CHECKPOINT_GIF", False)
    assert isinstance(checkpoint_gif, bool), (
        f"CHECKPOINT_GIF must be true/false, got {checkpoint_gif!r}"
    )
    if not checkpoint_gif:
        return

    output_root = config.get("CHECKPOINT_GIF_OUTPUT_DIR") or os.path.join(
        config["WANDB_DIR"],
        "checkpoint_rollouts",
    )
    epsilon = float(config.get("CHECKPOINT_GIF_EPSILON", 0.0))
    max_steps = int(config.get("CHECKPOINT_GIF_MAX_STEPS", 300))
    recipe_variants = _checkpoint_recipe_variants(config)

    if rollout_index is None:
        rollout_index = max(update // checkpoint_interval - 1, 0)
    namespace = _checkpoint_gif_namespace(config)
    media_key = config.get("CHECKPOINT_GIF_MEDIA_KEY") or f"{namespace}/rollout"

    rows = []
    for variant in recipe_variants:
        recipe_suffix = ""
        if variant["recipe"] is not None:
            recipe_suffix = (
                f"_recipe{variant['recipe_index']}_"
                f"{_sanitize_name(variant['recipe_label'])}"
            )
        output_file = (
            Path(output_root)
            / _sanitize_name(run_name)
            / f"update_{update:06d}{recipe_suffix}_eps{epsilon:g}_{max_steps}steps.gif"
        )
        fsq_output_dir = None
        if bool(config.get("CHECKPOINT_FSQ_VIEWER", True)) and not bool(
            config.get("DISABLE_FSQ_COMM", False)
        ):
            fsq_output_dir = output_file.parent / f"{output_file.stem}_fsq"

        row = render_checkpoint_gif(
            actor_params=actor_params,
            config=config,
            update=update,
            output_file=output_file,
            run_name=run_name,
            forced_recipe=variant["recipe"],
            recipe_index=variant["recipe_index"],
            fsq_output_dir=fsq_output_dir,
            actor_path=actor_path,
            config_path=config_path,
        )
        print(
            "Checkpoint GIF saved: "
            f"update={row['update']} recipe={row['recipe']} "
            f"reward={row['reward']:.1f} steps={row['steps']} gif={row['gif']}"
        )
        if "fsq_viewer_dir" in row:
            print(
                "FSQ viewer saved: "
                f"samples={row['fsq_total_samples']} "
                f"nonzero_codes={row['fsq_nonzero_codes']} "
                f"dir={row['fsq_viewer_dir']}"
            )

        rows.append(row)

    if not rows:
        return

    combined_output_file = (
        Path(output_root)
        / _sanitize_name(run_name)
        / f"update_{update:06d}_combined_eps{epsilon:g}_{max_steps}steps.gif"
    )
    fps = int(config.get("CHECKPOINT_GIF_FPS", 8))
    combined = _combine_checkpoint_gifs(
        gif_paths=[row["gif"] for row in rows],
        output_file=combined_output_file,
        fps=fps,
    )
    print(
        "Combined checkpoint GIF saved: "
        f"update={update} variants={len(rows)} "
        f"grid={combined['columns']}x{combined['rows']} gif={combined['gif']}"
    )

    if config["WANDB_MODE"] == "disabled" or wandb.run is None:
        return

    wandb.log(
        {
            media_key: wandb.Video(combined["gif"], format="gif"),
            f"{namespace}/rollout_index": rollout_index,
            f"{namespace}/checkpoint_update": int(update),
            f"{namespace}/episode_reward": float(
                np.mean([row["reward"] for row in rows])
            ),
            f"{namespace}/episode_steps": int(max(row["steps"] for row in rows)),
            f"{namespace}/layout": rows[0]["layout"],
            f"{namespace}/max_steps": int(rows[0]["max_steps"]),
            f"{namespace}/epsilon": float(rows[0]["epsilon"]),
            f"{namespace}/recipe": ",".join(str(row["recipe"]) for row in rows),
            f"{namespace}/recipe_index": -1,
            f"{namespace}/variant_count": len(rows),
            f"{namespace}/grid_rows": int(combined["rows"]),
            f"{namespace}/grid_columns": int(combined["columns"]),
            f"{namespace}/fsq_viewer_dir": "",
            f"{namespace}/fsq_total_samples": int(
                sum(row.get("fsq_total_samples", 0) for row in rows)
            ),
            f"{namespace}/fsq_nonzero_codes": int(
                max(row.get("fsq_nonzero_codes", 0) for row in rows)
            ),
        }
    )


def make_train(config, monitor=None):
    env = OvercookedV3(**config["ENV_KWARGS"])
    teacher_env_kwargs = copy.deepcopy(config["ENV_KWARGS"])
    teacher_env_kwargs["agent_view_size"] = None
    teacher_env = OvercookedV3(**teacher_env_kwargs)

    if env.num_agents != 2:
        raise ValueError("FSQ communication distillation currently supports 2 agents.")

    config["NUM_AGENTS"] = env.num_agents
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    if config["NUM_ENVS"] % config["NUM_MINIBATCHES"] != 0:
        raise ValueError("NUM_ENVS must be divisible by NUM_MINIBATCHES.")

    world_state_size = env.num_agents * int(np.prod(env.observation_space().shape))
    teacher_actor_path = config.get("TEACHER_ACTOR_PATH", "")
    if not teacher_actor_path:
        raise ValueError("TEACHER_ACTOR_PATH must point to a full-observation actor.")
    teacher_actor_params = load_actor_params(teacher_actor_path)
    fsq = FSQ(levels=tuple(config["FSQ_LEVELS"]))
    fsq_levels = jnp.asarray(config["FSQ_LEVELS"], dtype=jnp.int32)
    fsq_max_level = int(max(config["FSQ_LEVELS"]))

    env = LogWrapper(env, replace_info=False)

    def create_learning_rate_fn():
        base_learning_rate = config["LR"]

        lr_warmup = config["LR_WARMUP"]
        update_steps = config["NUM_UPDATES"]
        warmup_steps = int(lr_warmup * update_steps)

        steps_per_epoch = config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]

        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=base_learning_rate,
            transition_steps=warmup_steps * steps_per_epoch,
        )
        cosine_epochs = max(update_steps - warmup_steps, 1)

        cosine_fn = optax.cosine_decay_schedule(
            init_value=base_learning_rate, decay_steps=cosine_epochs * steps_per_epoch
        )
        schedule_fn = optax.join_schedules(
            schedules=[warmup_fn, cosine_fn],
            boundaries=[warmup_steps * steps_per_epoch],
        )
        return schedule_fn

    rew_shaping_anneal = optax.linear_schedule(
        init_value=1.0, end_value=0.0, transition_steps=config["REW_SHAPING_HORIZON"]
    )

    def distill_weight_schedule(env_step):
        progress = env_step / config["TOTAL_TIMESTEPS"]
        decay_fraction = config["DISTILL_DECAY_FRACTION"]
        phase = jnp.clip(progress / decay_fraction, 0.0, 1.0)
        weight = config["DISTILL_COEF"] * 0.5 * (1.0 + jnp.cos(jnp.pi * phase))
        return jnp.where(progress < decay_fraction, weight, 0.0)

    def full_obs_batch_from_state(log_env_state):
        full_obs = jax.vmap(teacher_env.get_obs)(log_env_state.env_state)
        return jnp.stack([full_obs[a] for a in env.agents]).reshape(
            -1, *teacher_env.observation_space().shape
        )

    def fsq_code_metrics(comm_index, comm_code):
        code_counts = jnp.bincount(
            comm_index.reshape(-1).astype(jnp.int32), length=fsq.codebook_size
        )
        total = jnp.maximum(code_counts.sum(), 1)
        probs = code_counts / total
        entropy = -jnp.sum(jnp.where(probs > 0, probs * jnp.log(probs), 0.0))
        unique_codes = (code_counts > 0).sum()
        max_frac = code_counts.max() / total

        level_ids = jnp.rint(
            comm_code * (fsq_levels // 2) + (fsq_levels // 2)
        ).astype(jnp.int32)
        dim_hists = []
        for dim, level in enumerate(config["FSQ_LEVELS"]):
            hist = jnp.bincount(
                level_ids[..., dim].reshape(-1), length=fsq_max_level
            )
            dim_hists.append(jnp.where(jnp.arange(fsq_max_level) < level, hist, 0))
        dim_hist = jnp.stack(dim_hists, axis=0)
        return code_counts, dim_hist, unique_codes, entropy, max_frac

    checkpoint_count = int(config.get("CHECKPOINT_GIF_COUNT", 10))
    checkpoint_update_tuple = checkpoint_updates(
        int(config["NUM_UPDATES"]), checkpoint_count
    )
    checkpoint_update_set = set(checkpoint_update_tuple)
    checkpoint_update_to_rollout_index = {
        update: idx for idx, update in enumerate(checkpoint_update_tuple)
    }
    checkpoint_dir = os.path.join(config["WANDB_DIR"], "models")
    layout_name = config["ENV_KWARGS"]["layout"]

    def train(rng):
        original_seed = rng[0]

        # INIT NETWORKS
        actor_network = CommActorRNN(env.action_space(env.agents[0]).n, config=config)
        critic_network = CriticRNN(config=config)
        teacher_actor_network = TeacherActorRNN(
            teacher_env.action_space(teacher_env.agents[0]).n, config=config
        )

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)

        # Actor init: grid observations
        ac_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], *env.observation_space().shape)),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        actor_network_params = actor_network.init(
            _rng_actor, ac_init_hstate, ac_init_x
        )

        # Critic init: flat world state
        cr_init_x = (
            jnp.zeros((1, config["NUM_ACTORS"], world_state_size)),
            jnp.zeros((1, config["NUM_ACTORS"])),
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        teacher_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        critic_network_params = critic_network.init(
            _rng_critic, cr_init_hstate, cr_init_x
        )

        if config["ANNEAL_LR"]:
            lr_schedule = create_learning_rate_fn()
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=critic_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)
        ac_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )
        cr_init_hstate = ScannedRNN.initialize_carry(
            config["NUM_ACTORS"], config["GRU_HIDDEN_DIM"]
        )

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                (
                    train_states,
                    env_state,
                    last_obs,
                    last_done,
                    update_step,
                    hstates,
                    rng,
                ) = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                    -1, *env.observation_space().shape
                )
                ac_in = (
                    obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )

                ac_hstate, pi, comm_code, comm_index = actor_network.apply(
                    train_states[0].params, hstates[0], ac_in
                )
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                full_obs_batch = full_obs_batch_from_state(env_state)
                teacher_in = (
                    full_obs_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                teacher_hstate, teacher_pi = teacher_actor_network.apply(
                    teacher_actor_params, hstates[2], teacher_in
                )
                teacher_logits = teacher_pi.logits

                # WORLD STATE for critic
                obs_flat = obs_batch.reshape(env.num_agents, config["NUM_ENVS"], -1)
                world_state_per_env = jnp.concatenate(
                    [obs_flat[i] for i in range(env.num_agents)], axis=-1
                )
                world_state_batch = jnp.tile(
                    world_state_per_env, (env.num_agents, 1)
                )

                cr_in = (
                    world_state_batch[np.newaxis, :],
                    last_done[np.newaxis, :],
                )
                cr_hstate, value = critic_network.apply(
                    train_states[1].params, hstates[1], cr_in
                )

                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )
                env_act = {k: v.flatten() for k, v in env_act.items()}

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)
                original_reward = jnp.array([reward[a] for a in env.agents])

                current_timestep = (
                    update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
                )
                anneal_factor = rew_shaping_anneal(current_timestep)
                distill_weight = distill_weight_schedule(current_timestep)
                reward = jax.tree_util.tree_map(
                    lambda x, y: x + y * anneal_factor * config["SHAPED_REWARD_SCALE"],
                    reward,
                    info["shaped_reward"],
                )

                for info_key, info_value in tuple(info.items()):
                    if isinstance(info_value, dict):
                        info[info_key] = jnp.array(
                            [info_value[a] for a in env.agents]
                        )

                shaped_reward = info["shaped_reward"]
                combined_reward = jnp.array([reward[a] for a in env.agents])

                info["shaped_reward"] = shaped_reward
                info["original_reward"] = original_reward
                info["anneal_factor"] = jnp.full_like(shaped_reward, anneal_factor)
                info["combined_reward"] = combined_reward

                info = jax.tree_util.tree_map(
                    lambda x: x.reshape((config["NUM_ACTORS"])), info
                )
                done_batch = batchify(done, env.agents, config["NUM_ACTORS"]).squeeze()
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    action.squeeze(),
                    value.squeeze(),
                    batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob.squeeze(),
                    obs_batch,
                    world_state_batch,
                    teacher_logits.squeeze(axis=0),
                    jnp.full((config["NUM_ACTORS"],), distill_weight),
                    comm_code.squeeze(axis=0),
                    comm_index.squeeze(axis=0),
                    info,
                )
                runner_state = (
                    train_states,
                    env_state,
                    obsv,
                    done_batch,
                    update_step,
                    (ac_hstate, cr_hstate, teacher_hstate),
                    rng,
                )
                return runner_state, transition

            initial_hstates = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, update_step, hstates, rng = (
                runner_state
            )

            # Build world state for last obs
            last_obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(
                -1, *env.observation_space().shape
            )
            last_obs_flat = last_obs_batch.reshape(
                env.num_agents, config["NUM_ENVS"], -1
            )
            last_world_state = jnp.concatenate(
                [last_obs_flat[i] for i in range(env.num_agents)], axis=-1
            )
            last_world_state_batch = jnp.tile(
                last_world_state, (env.num_agents, 1)
            )

            cr_in = (
                last_world_state_batch[np.newaxis, :],
                last_done[np.newaxis, :],
            )
            _, last_val = critic_network.apply(
                train_states[1].params, hstates[1], cr_in
            )
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch_info):
                    actor_train_state, critic_train_state = train_states
                    ac_init_hstate, cr_init_hstate, traj_batch, advantages, targets = (
                        batch_info
                    )

                    def from_env_major(x):
                        axes = (0, 2, 1) + tuple(range(3, x.ndim))
                        x = jnp.transpose(x, axes)
                        return jnp.reshape(
                            x,
                            (x.shape[0], x.shape[1] * x.shape[2]) + x.shape[3:],
                        )

                    ac_init_hstate = from_env_major(ac_init_hstate)
                    cr_init_hstate = from_env_major(cr_init_hstate)
                    traj_batch = jax.tree_util.tree_map(from_env_major, traj_batch)
                    advantages = from_env_major(advantages)
                    targets = from_env_major(targets)

                    def _actor_loss_fn(actor_params, init_hstate, traj_batch, gae):
                        # RERUN ACTOR
                        _, pi, _, _ = actor_network.apply(
                            actor_params,
                            init_hstate.squeeze(),
                            (traj_batch.obs, traj_batch.done),
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        # Diagnostic metrics
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(
                            jnp.abs(ratio - 1.0) > config["CLIP_EPS"]
                        )

                        temperature = config["DISTILL_TEMPERATURE"]
                        teacher_logits = traj_batch.teacher_logits / temperature
                        student_logits = pi.logits / temperature
                        teacher_log_probs = jax.nn.log_softmax(
                            teacher_logits, axis=-1
                        )
                        teacher_probs = jax.nn.softmax(teacher_logits, axis=-1)
                        student_log_probs = jax.nn.log_softmax(
                            student_logits, axis=-1
                        )
                        teacher_kl = jnp.sum(
                            teacher_probs * (teacher_log_probs - student_log_probs),
                            axis=-1,
                        )
                        distill_loss = (
                            traj_batch.distill_weight * teacher_kl
                        ).mean() * (temperature**2)

                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                            + distill_loss
                        )
                        return actor_loss, (
                            loss_actor,
                            entropy,
                            approx_kl,
                            clip_frac,
                            teacher_kl.mean(),
                            distill_loss,
                            traj_batch.distill_weight.mean(),
                        )

                    def _critic_loss_fn(
                        critic_params, init_hstate, traj_batch, targets
                    ):
                        # RERUN CRITIC
                        _, value = critic_network.apply(
                            critic_params,
                            init_hstate.squeeze(),
                            (traj_batch.world_state, traj_batch.done),
                        )

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(
                            value_pred_clipped - targets
                        )
                        value_loss = (
                            0.5
                            * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, value_loss

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params,
                        ac_init_hstate,
                        traj_batch,
                        advantages,
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params,
                        cr_init_hstate,
                        traj_batch,
                        targets,
                    )

                    actor_train_state = actor_train_state.apply_gradients(
                        grads=actor_grads
                    )
                    critic_train_state = critic_train_state.apply_gradients(
                        grads=critic_grads
                    )

                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[1][0],
                        "value_loss": critic_loss[1],
                        "entropy": actor_loss[1][1],
                        "approx_kl": actor_loss[1][2],
                        "clip_frac": actor_loss[1][3],
                        "teacher_kl": actor_loss[1][4],
                        "distill_loss": actor_loss[1][5],
                        "distill_weight": actor_loss[1][6],
                    }

                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                raw_init_hstates = init_hstates
                init_hstates = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, (1, config["NUM_ACTORS"], -1)),
                    init_hstates[:2],
                )

                batch = (
                    init_hstates[0],
                    init_hstates[1],
                    traj_batch,
                    advantages,
                    targets,
                )

                num_agents = env.num_agents
                num_envs = config["NUM_ENVS"]
                num_minibatches = config["NUM_MINIBATCHES"]
                envs_per_minibatch = num_envs // num_minibatches
                permutation = jax.random.permutation(_rng, num_envs)

                def to_env_major(x):
                    x = jnp.reshape(
                        x,
                        (x.shape[0], num_agents, num_envs) + x.shape[2:],
                    )
                    axes = (0, 2, 1) + tuple(range(3, x.ndim))
                    return jnp.transpose(x, axes)

                def make_minibatches(x):
                    x = to_env_major(x)
                    x = jnp.take(x, permutation, axis=1)
                    x = jnp.reshape(
                        x,
                        (x.shape[0], num_minibatches, envs_per_minibatch, num_agents)
                        + x.shape[3:],
                    )
                    return jnp.swapaxes(x, 0, 1)

                minibatches = jax.tree_util.tree_map(make_minibatches, batch)

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    raw_init_hstates,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                initial_hstates,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            (
                fsq_code_counts,
                fsq_dim_hist,
                fsq_unique_codes,
                fsq_code_entropy,
                fsq_usage_max_frac,
            ) = fsq_code_metrics(traj_batch.comm_index, traj_batch.comm_code)

            def callback(metric, original_seed, actor_params, critic_params):
                step = int(metric["env_step"])
                updates = int(metric["update_step"])
                num_updates = int(config["NUM_UPDATES"])
                ret = float(metric.get("returned_episode_returns", 0.0))

                if monitor is not None:
                    monitor.update(
                        step=updates,
                        metrics={
                            "env_step": step,
                            "update": f"{updates}/{num_updates}",
                            "train_return": ret,
                            "shaped_reward": float(metric.get("shaped_reward", 0.0)),
                            "original_reward": float(
                                metric.get("original_reward", 0.0)
                            ),
                            "anneal_factor": float(metric.get("anneal_factor", 0.0)),
                            "fsq_unique_codes": float(
                                metric.get("fsq_unique_codes", 0.0)
                            ),
                            "distill_weight": float(
                                metric.get("distill_weight", 0.0)
                            ),
                        },
                        seed=int(original_seed),
                    )

                if config["WANDB_MODE"] != "disabled":
                    wandb_metric = dict(metric)
                    code_counts = np.asarray(
                        wandb_metric.pop("fsq_code_counts", np.array([]))
                    )
                    dim_hist = np.asarray(
                        wandb_metric.pop("fsq_dim_hist", np.array([]))
                    )
                    for i, count in enumerate(code_counts.tolist()):
                        wandb_metric[f"fsq/code_{i:03d}_count"] = int(count)
                    for dim in range(dim_hist.shape[0]):
                        for level in range(dim_hist.shape[1]):
                            wandb_metric[
                                f"fsq/dim_{dim}_level_{level}_count"
                            ] = int(dim_hist[dim, level])
                    wandb.log(wandb_metric)

                # Periodic checkpointing
                if (
                    not config.get("DISABLE_CHECKPOINTS", False)
                    and updates in checkpoint_update_set
                ):
                    run_name = wandb.run.name if wandb.run else "offline"
                    date_str = datetime.datetime.now().strftime("%Y%m%d")
                    ckpt_subdir = os.path.join(checkpoint_dir, f"{run_name}_{date_str}")
                    os.makedirs(ckpt_subdir, exist_ok=True)
                    actor_checkpoint_path = os.path.join(
                        ckpt_subdir, f"{updates}_actor.safetensors"
                    )
                    critic_checkpoint_path = os.path.join(
                        ckpt_subdir, f"{updates}_critic.safetensors"
                    )
                    save_params(actor_params, actor_checkpoint_path)
                    save_params(critic_params, critic_checkpoint_path)
                    print(f"Checkpoint saved: {ckpt_subdir}/{updates}_*.safetensors")
                    render_and_log_checkpoint_gif(
                        actor_params=actor_params,
                        config=config,
                        update=updates,
                        run_name=run_name,
                        checkpoint_interval=max(int(config["NUM_UPDATES"]), 1),
                        rollout_index=checkpoint_update_to_rollout_index[updates],
                        actor_path=actor_checkpoint_path,
                    )

            update_step = update_step + 1
            loss_info = jax.tree_util.tree_map(lambda x: x.mean(), loss_info)
            metric = jax.tree_util.tree_map(lambda x: x.mean(), metric)
            metric["total_loss"] = loss_info["total_loss"]
            metric["value_loss"] = loss_info["value_loss"]
            metric["actor_loss"] = loss_info["actor_loss"]
            metric["entropy"] = loss_info["entropy"]
            metric["approx_kl"] = loss_info["approx_kl"]
            metric["clip_frac"] = loss_info["clip_frac"]
            metric["teacher_kl"] = loss_info["teacher_kl"]
            metric["distill_loss"] = loss_info["distill_loss"]
            metric["distill_weight"] = loss_info["distill_weight"]
            metric["fsq_unique_codes"] = fsq_unique_codes
            metric["fsq_code_entropy"] = fsq_code_entropy
            metric["fsq_usage_max_frac"] = fsq_usage_max_frac
            metric["fsq_code_counts"] = fsq_code_counts
            metric["fsq_dim_hist"] = fsq_dim_hist
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            if (
                monitor is not None
                or config["WANDB_MODE"] != "disabled"
                or not config.get("DISABLE_CHECKPOINTS", False)
            ):
                jax.debug.callback(
                    callback,
                    metric,
                    original_seed,
                    train_states[0].params,
                    train_states[1].params,
                )

            runner_state = (
                train_states,
                env_state,
                last_obs,
                last_done,
                update_step,
                hstates,
                rng,
            )
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            0,
            (ac_init_hstate, cr_init_hstate, teacher_init_hstate),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


def single_run(config):
    """Execute a single training run."""
    layout_name = config["ENV_KWARGS"]["layout"]
    num_seeds = config["NUM_SEEDS"]
    checkpoint_gif = config.get("CHECKPOINT_GIF", False)
    assert isinstance(checkpoint_gif, bool), (
        f"CHECKPOINT_GIF must be true/false, got {checkpoint_gif!r}"
    )

    wandb_dir = config["WANDB_DIR"]
    os.makedirs(wandb_dir, exist_ok=True)

    wandb.init(
        dir=wandb_dir,
        entity=config["ENTITY"],
        project=config["PROJECT"],
        tags=[
            "MAPPO",
            "RNN",
            "OvercookedV3",
            "Distillation",
            "NoFSQ" if config.get("DISABLE_FSQ_COMM", False) else "FSQ",
        ],
        config=copy.deepcopy(config),
        mode=config["WANDB_MODE"],
        name=config["WANDB_RUN_NAME"]
        or f"mappo_rnn_overcooked_v3_fsq_distill_{layout_name}",
    )
    if checkpoint_gif and config["WANDB_MODE"] != "disabled":
        checkpoint_gif_namespace = _checkpoint_gif_namespace(config)
        wandb.define_metric(f"{checkpoint_gif_namespace}/rollout_index")
        wandb.define_metric(
            f"{checkpoint_gif_namespace}/*",
            step_metric=f"{checkpoint_gif_namespace}/rollout_index",
        )

    num_updates = int(
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    use_monitor = config.get("USE_RICH_MONITOR", True) and _MONITOR_AVAILABLE
    monitor = None
    if use_monitor:
        monitor = TrainingMonitor(
            total_updates=num_updates,
            config_dict={
                "env": "overcooked_v3",
                "algo": "MAPPO",
                "layout": layout_name,
                "total_timesteps": int(config["TOTAL_TIMESTEPS"]),
                "num_updates": num_updates,
                "num_envs": config["NUM_ENVS"],
                "num_seeds": num_seeds,
                "lr": config["LR"],
                "gamma": config["GAMMA"],
            },
            title=f"MAPPO-RNN - OvercookedV3 ({layout_name})",
        )

    with jax.disable_jit(False):
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, num_seeds)
        train_jit = jax.jit(make_train(config, monitor=monitor))
        if monitor is not None:
            with monitor:
                out = jax.block_until_ready(jax.vmap(train_jit)(rngs))
        else:
            out = jax.vmap(train_jit)(rngs)

    # Save final model params
    save_dir = os.path.join(wandb_dir, "models")
    os.makedirs(save_dir, exist_ok=True)

    actor_state, critic_state = out["runner_state"][0]
    run_name = wandb.run.name if wandb.run else "offline"
    OmegaConf.save(
        config,
        os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_fsq_distill_{layout_name}_seed{config['SEED']}_config.yaml",
        ),
    )

    for i, rng in enumerate(rngs):
        actor_params = jax.tree.map(lambda x: x[i], actor_state.params)
        critic_params = jax.tree.map(lambda x: x[i], critic_state.params)
        actor_path = os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_fsq_distill_{layout_name}_seed{config['SEED']}_vmap{i}_actor.safetensors",
        )
        critic_path = os.path.join(
            save_dir,
            f"mappo_rnn_overcooked_v3_fsq_distill_{layout_name}_seed{config['SEED']}_vmap{i}_critic.safetensors",
        )
        save_params(actor_params, actor_path)
        save_params(critic_params, critic_path)
        print(f"Saved actor params to {actor_path}")
        print(f"Saved critic params to {critic_path}")


def tune(config):
    """Hyperparameter sweep with CARBS."""
    from carbs_sweep import CARBSSweep

    layout_name = config["ENV_KWARGS"]["layout"]
    sweep = CARBSSweep(config)

    print(f"Starting CARBS sweep: {sweep.num_trials} trials, layout={layout_name}")

    for trial in range(sweep.num_trials):
        suggestion = sweep.suggest()
        trial_config = sweep.apply_suggestion(suggestion)
        trial_config["WANDB_MODE"] = "disabled"

        print(f"\n{'='*60}")
        print(f"Trial {trial+1}/{sweep.num_trials}")
        print(f"  {CARBSSweep.format_suggestion(suggestion)}")

        start_time = time.time()
        try:
            rng = jax.random.PRNGKey(trial_config["SEED"])
            rngs = jax.random.split(rng, trial_config["NUM_SEEDS"])
            train_fn = make_train(trial_config, monitor=None)
            outs = jax.block_until_ready(jax.jit(jax.vmap(train_fn))(rngs))

            final_return = float(
                outs["metrics"]["returned_episode_returns"][:, -1].mean()
            )
            elapsed = time.time() - start_time

            sweep.observe(suggestion, output=final_return, cost=elapsed)
            print(
                f"  Return: {final_return:.2f}  Time: {elapsed:.1f}s  "
                f"Best: {sweep.best_return:.2f}"
            )

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  FAILED: {e}")
            sweep.observe_failure(suggestion, cost=elapsed)

    sweep.print_summary()


@hydra.main(
    version_base=None,
    config_path="config",
    config_name="mappo_rnn_overcooked_v3_fsq_distill",
)
def main(config):
    config = OmegaConf.to_container(config, resolve=True)
    if config.get("TUNE", False):
        tune(config)
    else:
        single_run(config)


if __name__ == "__main__":
    main()
