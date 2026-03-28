"""Behavior cloning pretraining for Overcooked V3 IPPO policies."""

import glob
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from hydra.utils import to_absolute_path
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import jaxmarl
from baselines.IPPO.ippo_rnn_overcooked_v3 import (
    ActorCriticRNN,
    _restore_model_params,
    _save_model_params,
)


def _resolve_demo_files(patterns) -> List[Path]:
    if isinstance(patterns, str):
        patterns = [patterns]
    demo_files: List[Path] = []
    for pattern in patterns:
        resolved_pattern = to_absolute_path(pattern)
        matches = sorted(Path(match) for match in glob.glob(resolved_pattern, recursive=True))
        demo_files.extend(matches)
    unique_files = sorted(dict.fromkeys(demo_files))
    if not unique_files:
        raise FileNotFoundError(f"No demonstration files matched: {patterns}")
    return unique_files


def _load_dataset(files: List[Path]) -> List[Dict[str, Any]]:
    dataset = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            obs = data["obs"].astype(np.float32)
            dones = data["dones"].astype(np.bool_)
            actions = data["actions"].astype(np.int32)
            rewards = data["rewards"].astype(np.float32) if "rewards" in data else None
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"]))

        if obs.ndim < 3:
            raise ValueError(f"Expected obs with shape [T, A, ...], got {obs.shape} from {path}")
        if dones.shape != actions.shape:
            raise ValueError(f"Expected dones/actions shapes to match in {path}, got {dones.shape} and {actions.shape}")
        if obs.shape[:2] != actions.shape:
            raise ValueError(f"Expected obs[:2] to match actions in {path}, got {obs.shape[:2]} and {actions.shape}")

        dataset.append(
            {
                "path": str(path),
                "obs": obs,
                "dones": dones,
                "actions": actions,
                "rewards": rewards,
                "metadata": metadata,
            }
        )
    return dataset


def _split_dataset(dataset, val_fraction, seed):
    indices = np.arange(len(dataset))
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = int(round(len(indices) * val_fraction))
    if val_fraction > 0.0:
        val_count = max(1, val_count)
    val_indices = set(indices[:val_count].tolist())
    train_set = [dataset[idx] for idx in range(len(dataset)) if idx not in val_indices]
    val_set = [dataset[idx] for idx in range(len(dataset)) if idx in val_indices]
    if not train_set:
        raise ValueError("Validation split consumed the full dataset; lower VAL_FRACTION or add more demos.")
    return train_set, val_set


def _make_padded_batch(episodes):
    max_len = max(episode["obs"].shape[0] for episode in episodes)
    obs_shape = episodes[0]["obs"].shape[2:]
    num_agents = episodes[0]["obs"].shape[1]
    batch_size = len(episodes)
    total_actors = batch_size * num_agents

    obs = np.zeros((max_len, total_actors, *obs_shape), dtype=np.float32)
    dones = np.ones((max_len, total_actors), dtype=np.bool_)
    actions = np.zeros((max_len, total_actors), dtype=np.int32)
    mask = np.zeros((max_len, total_actors), dtype=np.float32)

    for batch_idx, episode in enumerate(episodes):
        length = episode["obs"].shape[0]
        actor_slice = slice(batch_idx * num_agents, (batch_idx + 1) * num_agents)
        obs[:length, actor_slice] = episode["obs"]
        dones[:length, actor_slice] = episode["dones"]
        actions[:length, actor_slice] = episode["actions"]
        mask[:length, actor_slice] = 1.0

    return {
        "obs": jnp.asarray(obs),
        "dones": jnp.asarray(dones),
        "actions": jnp.asarray(actions),
        "mask": jnp.asarray(mask),
    }


def _dataset_summary(dataset):
    total_steps = int(sum(item["actions"].shape[0] for item in dataset))
    total_agent_steps = int(sum(item["actions"].size for item in dataset))
    mean_length = float(np.mean([item["actions"].shape[0] for item in dataset]))
    mean_return = None
    if dataset and dataset[0]["rewards"] is not None:
        mean_return = float(np.mean([item["rewards"].sum() for item in dataset]))
    return {
        "episodes": len(dataset),
        "timesteps": total_steps,
        "agent_steps": total_agent_steps,
        "mean_episode_length": mean_length,
        "mean_total_return": mean_return,
    }


def _iterate_minibatches(dataset, batch_size, rng):
    order = rng.permutation(len(dataset))
    for start in range(0, len(order), batch_size):
        batch_indices = order[start:start + batch_size]
        episodes = [dataset[idx] for idx in batch_indices]
        yield _make_padded_batch(episodes)


def _scalarize_metrics(metrics):
    return {key: float(np.asarray(value)) for key, value in metrics.items()}


@hydra.main(version_base=None, config_path="config", config_name="behavior_clone_overcooked_v3")
def main(config):
    config = OmegaConf.to_container(config, resolve=True)

    demo_files = _resolve_demo_files(config["DEMO_GLOB"])
    dataset = _load_dataset(demo_files)
    train_set, val_set = _split_dataset(dataset, config.get("VAL_FRACTION", 0.0), config.get("SEED", 0))

    print(f"Loaded {len(dataset)} demo files")
    print(f"Train split: {_dataset_summary(train_set)}")
    if val_set:
        print(f"Validation split: {_dataset_summary(val_set)}")

    env = jaxmarl.make(config["ENV_NAME"], **config.get("ENV_KWARGS", {}))
    obs_shape = env.observation_space(env.agents[0]).shape
    action_dim = env.action_space(env.agents[0]).n
    hidden_dim = config.get("GRU_HIDDEN_DIM", 128)
    network = ActorCriticRNN(action_dim, config=config)

    rng = jax.random.PRNGKey(config.get("SEED", 0))
    rng, init_rng = jax.random.split(rng)
    init_obs = jnp.zeros((1, 1, *obs_shape), dtype=jnp.float32)
    init_dones = jnp.zeros((1, 1), dtype=jnp.bool_)
    init_hidden = jnp.zeros((1, hidden_dim), dtype=jnp.float32)
    params = network.init(init_rng, init_hidden, (init_obs, init_dones))

    init_checkpoint = config.get("INIT_CHECKPOINT")
    if init_checkpoint:
        init_checkpoint = to_absolute_path(init_checkpoint)
        params = _restore_model_params(init_checkpoint, params)
        print(f"Loaded initialization checkpoint: {init_checkpoint}")

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"]),
    )
    train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            init_hidden = jnp.zeros((batch["obs"].shape[1], hidden_dim), dtype=jnp.float32)
            _, pi, _ = network.apply(params, init_hidden, (batch["obs"], batch["dones"]))
            log_prob = pi.log_prob(batch["actions"])
            mask = batch["mask"]
            denom = jnp.maximum(mask.sum(), 1.0)
            loss = -(log_prob * mask).sum() / denom
            predictions = pi.mode()
            accuracy = ((predictions == batch["actions"]) * mask).sum() / denom
            entropy = (pi.entropy() * mask).sum() / denom
            return loss, {
                "loss": loss,
                "accuracy": accuracy,
                "entropy": entropy,
            }

        (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, metrics

    @jax.jit
    def eval_step(params, batch):
        init_hidden = jnp.zeros((batch["obs"].shape[1], hidden_dim), dtype=jnp.float32)
        _, pi, _ = network.apply(params, init_hidden, (batch["obs"], batch["dones"]))
        log_prob = pi.log_prob(batch["actions"])
        mask = batch["mask"]
        denom = jnp.maximum(mask.sum(), 1.0)
        loss = -(log_prob * mask).sum() / denom
        predictions = pi.mode()
        accuracy = ((predictions == batch["actions"]) * mask).sum() / denom
        entropy = (pi.entropy() * mask).sum() / denom
        return {
            "loss": loss,
            "accuracy": accuracy,
            "entropy": entropy,
        }

    save_path = Path(to_absolute_path(config.get("SAVE_PATH", "checkpoints/behavior_clone_overcooked_v3")))
    save_path.mkdir(parents=True, exist_ok=True)
    best_metric = math.inf
    history = []
    np_rng = np.random.default_rng(config.get("SEED", 0))

    for epoch in range(1, config["NUM_EPOCHS"] + 1):
        train_metrics = []
        for batch in _iterate_minibatches(train_set, config["BATCH_SIZE_EPISODES"], np_rng):
            train_state, metrics = train_step(train_state, batch)
            train_metrics.append(_scalarize_metrics(metrics))

        epoch_train = {
            key: float(np.mean([metric[key] for metric in train_metrics]))
            for key in train_metrics[0]
        }

        epoch_val = None
        if val_set:
            val_metrics = []
            for batch in _iterate_minibatches(val_set, config["BATCH_SIZE_EPISODES"], np_rng):
                val_metrics.append(_scalarize_metrics(eval_step(train_state.params, batch)))
            epoch_val = {
                key: float(np.mean([metric[key] for metric in val_metrics]))
                for key in val_metrics[0]
            }

        tracked_metric = epoch_val["loss"] if epoch_val is not None else epoch_train["loss"]
        if tracked_metric < best_metric:
            best_metric = tracked_metric
            best_path = _save_model_params(train_state.params, str(save_path / "best"))
        else:
            best_path = str(save_path / "best" / "model.msgpack")

        final_path = _save_model_params(train_state.params, str(save_path / "latest"))
        record = {
            "epoch": epoch,
            "train": epoch_train,
            "val": epoch_val,
            "best_metric": best_metric,
            "best_checkpoint": best_path,
            "latest_checkpoint": final_path,
        }
        history.append(record)

        print(
            f"epoch={epoch} train_loss={epoch_train['loss']:.4f} train_acc={epoch_train['accuracy']:.4f} "
            + (
                f"val_loss={epoch_val['loss']:.4f} val_acc={epoch_val['accuracy']:.4f} "
                if epoch_val is not None
                else ""
            )
            + f"best_metric={best_metric:.4f}"
        )

    summary = {
        "config": config,
        "demo_files": [str(path) for path in demo_files],
        "train_summary": _dataset_summary(train_set),
        "val_summary": _dataset_summary(val_set) if val_set else None,
        "history": history,
        "best_checkpoint": str(save_path / "best" / "model.msgpack"),
        "latest_checkpoint": str(save_path / "latest" / "model.msgpack"),
    }
    with open(save_path / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()