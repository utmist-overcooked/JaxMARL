#!/usr/bin/env python3
"""Evaluate a behavior-cloned Overcooked V3 checkpoint on hold-out demos and roll-outs."""

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jaxmarl
from baselines.IPPO.ippo_rnn_overcooked_v3 import ActorCriticRNN, _restore_model_params
from jaxmarl.wrappers.baselines import LogWrapper


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--layout", type=str, default="cramped_room")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=20, help="Number of roll-out episodes to run.")
    parser.add_argument("--demo-glob", type=str, default="", help="Glob for hold-out demonstration .npz files.")
    parser.add_argument("--batch-size-episodes", type=int, default=8)
    parser.add_argument("--sample-actions", action="store_true", help="Sample actions instead of greedy argmax during roll-outs.")
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def resolve_demo_files(pattern):
    if not pattern:
        return []
    return sorted(Path(match) for match in glob.glob(pattern, recursive=True))


def load_demo_dataset(files):
    dataset = []
    for path in files:
        with np.load(path, allow_pickle=False) as data:
            metadata = {}
            if "metadata_json" in data:
                metadata = json.loads(str(data["metadata_json"]))
            dataset.append(
                {
                    "path": str(path),
                    "obs": data["obs"].astype(np.float32),
                    "dones": data["dones"].astype(np.bool_),
                    "actions": data["actions"].astype(np.int32),
                    "metadata": metadata,
                }
            )
    return dataset


def make_padded_batch(episodes):
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


def infer_network_config(params):
    hidden_dim = 128
    fc_dim = 128
    if isinstance(params, dict) and "gru_Wh_z" in params:
        hidden_dim = int(params["gru_Wh_z"].shape[0])
    if isinstance(params, dict):
        for key, value in params.items():
            if key.startswith("Dense_") and isinstance(value, dict) and "kernel" in value:
                fc_dim = int(value["kernel"].shape[1])
                break
    return {
        "GRU_HIDDEN_DIM": hidden_dim,
        "FC_DIM_SIZE": fc_dim,
        "ACTIVATION": "relu",
    }


def load_policy(checkpoint_path, env):
    raw_params = _restore_model_params(checkpoint_path)
    if isinstance(raw_params, dict) and "params" in raw_params and isinstance(raw_params["params"], dict):
        raw_params = raw_params["params"]
    config = infer_network_config(raw_params)
    network = ActorCriticRNN(env.action_space(env.agents[0]).n, config=config)

    obs_shape = env.observation_space(env.agents[0]).shape
    init_hidden = jnp.zeros((1, config["GRU_HIDDEN_DIM"]), dtype=jnp.float32)
    init_obs = jnp.zeros((1, 1, *obs_shape), dtype=jnp.float32)
    init_dones = jnp.zeros((1, 1), dtype=jnp.bool_)
    template_params = network.init(jax.random.PRNGKey(0), init_hidden, (init_obs, init_dones))
    params = _restore_model_params(checkpoint_path, template_params)
    return network, {"params": params}, config


def eval_holdout_batch(network, variables, hidden_dim, batch):
    init_hidden = jnp.zeros((batch["obs"].shape[1], hidden_dim), dtype=jnp.float32)
    _, pi, _ = network.apply(variables, init_hidden, (batch["obs"], batch["dones"]))
    log_prob = pi.log_prob(batch["actions"])
    mask = batch["mask"]
    denom = jnp.maximum(mask.sum(), 1.0)
    predictions = pi.mode()
    return {
        "loss": -(log_prob * mask).sum() / denom,
        "accuracy": ((predictions == batch["actions"]) * mask).sum() / denom,
        "entropy": (pi.entropy() * mask).sum() / denom,
        "agent_steps": mask.sum(),
    }


def evaluate_holdout(network, variables, hidden_dim, dataset, batch_size_episodes):
    if not dataset:
        return None

    metrics = []
    for start in range(0, len(dataset), batch_size_episodes):
        episodes = dataset[start:start + batch_size_episodes]
        batch = make_padded_batch(episodes)
        metrics.append(eval_holdout_batch(network, variables, hidden_dim, batch))

    weights = np.asarray([np.asarray(metric["agent_steps"]) for metric in metrics], dtype=np.float64)
    total_weight = max(1.0, float(weights.sum()))
    result = {}
    for key in ("loss", "accuracy", "entropy"):
        values = np.asarray([np.asarray(metric[key]) for metric in metrics], dtype=np.float64)
        result[key] = float((values * weights).sum() / total_weight)
    result["agent_steps"] = float(total_weight)
    result["episodes"] = len(dataset)
    result["files"] = [item["path"] for item in dataset]
    return result


def evaluate_rollouts(network, variables, hidden_dim, env, episodes, seed, sample_actions):
    returns = []
    deliveries = []
    lengths = []
    action_counts = np.zeros((env.num_agents, env.action_space(env.agents[0]).n), dtype=np.int64)

    key = jax.random.PRNGKey(seed)
    for _ in range(episodes):
        key, reset_key = jax.random.split(key)
        obs, state = env.reset(reset_key)
        hidden = jnp.zeros((env.num_agents, hidden_dim), dtype=jnp.float32)
        done_vec = jnp.zeros((env.num_agents,), dtype=jnp.bool_)

        while True:
            obs_batch = jnp.stack([obs[agent] for agent in env.agents]).reshape(
                env.num_agents, *env.observation_space(env.agents[0]).shape
            )
            hidden, pi, _ = network.apply(variables, hidden, (obs_batch[jnp.newaxis, :], done_vec[jnp.newaxis, :]))
            if sample_actions:
                key, action_key = jax.random.split(key)
                action = np.asarray(pi.sample(seed=action_key).squeeze(0))
            else:
                action = np.asarray(pi.mode().squeeze(0))

            for idx in range(env.num_agents):
                action_counts[idx, int(action[idx])] += 1

            actions = {agent: int(action[idx]) for idx, agent in enumerate(env.agents)}
            key, step_key = jax.random.split(key)
            obs, state, reward, done, info = env.step(step_key, state, actions)
            done_vec = jnp.asarray([done[agent] for agent in env.agents], dtype=jnp.bool_)

            if bool(done["__all__"]):
                returns.append(float(np.asarray(info["returned_episode_returns"]).mean()))
                deliveries.append(float(np.asarray(info["returned_episode_deliveries"]).mean()))
                lengths.append(float(np.asarray(info["returned_episode_lengths"]).mean()))
                break

    total_actions = max(1, int(action_counts.sum()))
    return {
        "episodes": episodes,
        "mean_return": float(np.mean(returns)),
        "std_return": float(np.std(returns)),
        "mean_deliveries": float(np.mean(deliveries)),
        "mean_length": float(np.mean(lengths)),
        "stay_ratio": float(action_counts[:, 4].sum() / total_actions),
        "action_counts": action_counts.tolist(),
    }


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    demo_files = resolve_demo_files(args.demo_glob)
    holdout_dataset = load_demo_dataset(demo_files)

    base_env = jaxmarl.make("overcooked_v3", layout=args.layout, max_steps=args.max_steps)
    network, variables, config = load_policy(str(checkpoint_path), base_env)
    hidden_dim = config["GRU_HIDDEN_DIM"]

    holdout_metrics = evaluate_holdout(
        network,
        variables,
        hidden_dim,
        holdout_dataset,
        args.batch_size_episodes,
    )

    rollout_env = LogWrapper(jaxmarl.make("overcooked_v3", layout=args.layout, max_steps=args.max_steps), replace_info=False)
    rollout_metrics = evaluate_rollouts(
        network,
        variables,
        hidden_dim,
        rollout_env,
        args.episodes,
        args.seed,
        args.sample_actions,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "layout": args.layout,
        "max_steps": args.max_steps,
        "policy_config": config,
        "holdout": holdout_metrics,
        "rollout": rollout_metrics,
    }

    print(json.dumps(summary, indent=2))

    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"Saved evaluation summary to: {output_path}")


if __name__ == "__main__":
    main()