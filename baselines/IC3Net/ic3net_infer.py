"""IC3Net-family inference script for JaxMARL.

Evaluate trained IC3Net/CommNet/IC/IRIC policies.
Supports headless batch evaluation and animated GIF visualization.

Usage:
  # Batch eval (headless)
  python ic3net_infer.py --config-name=ic3net_pp_medium_infer

  # Save animated GIF
  python ic3net_infer.py --config-name=ic3net_pp_medium_infer SAVE_GIF=/tmp/ic3net_pp.gif
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import jax
import jax.numpy as jnp
import numpy as np
from flax import serialization
from typing import Dict
import jaxmarl
import hydra
from omegaconf import OmegaConf
import time

from baselines.IC3Net.models import IndependentMLP, CommNetDiscrete, IndependentLSTM, CommNetLSTM


def _build_network(config, num_agents, action_dim):
    """Build network based on baseline type and recurrence setting."""
    baseline = config.get("BASELINE", "ic3net")
    recurrent = config.get("RECURRENT", False)
    hidden_dim = config.get("HIDDEN_DIM", 64)

    if baseline in ("ic", "iric"):
        if recurrent:
            network = IndependentLSTM(action_dim=action_dim, hidden_dim=hidden_dim)
        else:
            network = IndependentMLP(action_dim=action_dim, hidden_dim=hidden_dim)
        has_talk = False
    else:
        hard_attn = (baseline == "ic3net")
        kw = dict(
            num_agents=num_agents, action_dim=action_dim, hidden_dim=hidden_dim,
            comm_passes=config.get("COMM_PASSES", 1),
            comm_mode=config.get("COMM_MODE", "avg"),
            hard_attn=hard_attn,
            share_weights=config.get("SHARE_WEIGHTS", False),
            encoder_layers=config.get("ENCODER_LAYERS", 1),
        )
        network = CommNetLSTM(**kw) if recurrent else CommNetDiscrete(**kw)
        has_talk = hard_attn
    return network, has_talk


def _stack_obs(obs, agents, num_agents):
    """Stack obs dict -> (1, N, obs_dim)."""
    obs_list = []
    for a in agents:
        o = obs[a]
        if len(o.shape) > 1:
            o = o.reshape(-1)
        obs_list.append(o)
    return jnp.stack(obs_list, axis=0)[None]


def run_episode(env, network, params, config, has_talk, rng):
    """Run a single episode with a Python loop, collecting state_seq for viz."""
    num_agents = env.num_agents
    recurrent = config.get("RECURRENT", False)
    hidden_dim = config.get("HIDDEN_DIM", 64)
    max_steps = config.get("MAX_STEPS", 50)
    deterministic = config.get("DETERMINISTIC", True)

    rng, _rng = jax.random.split(rng)
    obs, env_state = env.reset(_rng)

    comm_action = jnp.zeros((1, num_agents), dtype=jnp.int32)
    if recurrent:
        hstate = (jnp.zeros((1, num_agents, hidden_dim)),
                  jnp.zeros((1, num_agents, hidden_dim)))
    else:
        hstate = None

    state_seq = [env_state]
    episode_return = 0.0

    for step in range(max_steps):
        obs_batch = _stack_obs(obs, env.agents, num_agents)

        # Forward pass
        if recurrent:
            if has_talk:
                logits, value, talk_logits, hstate = network.apply(
                    params, obs_batch, carry=hstate, comm_action=comm_action)
            else:
                logits, value, _, hstate = network.apply(
                    params, obs_batch, carry=hstate)
                talk_logits = None
        else:
            if has_talk:
                logits, value, talk_logits = network.apply(
                    params, obs_batch, comm_action=comm_action)
            else:
                logits, value = network.apply(params, obs_batch)
                talk_logits = None

        # Action selection: deterministic (argmax) or stochastic (sample)
        if deterministic:
            action = jnp.argmax(logits[0], axis=-1)
        else:
            rng, _rng = jax.random.split(rng)
            action = jax.random.categorical(_rng, logits[0], axis=-1)

        # Update comm_action
        if has_talk and talk_logits is not None:
            action_talk = jnp.argmax(talk_logits[0], axis=-1)
            if config.get("COMM_ACTION_ONE", False):
                comm_action = jnp.ones((1, num_agents), dtype=jnp.int32)
            else:
                comm_action = jnp.expand_dims(action_talk, 0)

        action_dict = {a: action[i] for i, a in enumerate(env.agents)}
        rng, _rng = jax.random.split(rng)
        obs, env_state, reward, done, info = env.step(_rng, env_state, action_dict)

        state_seq.append(env_state)
        total_reward = sum(float(reward[a]) for a in env.agents)
        episode_return += total_reward

        if done["__all__"]:
            break

    return state_seq, episode_return


@hydra.main(version_base=None, config_path="config", config_name="ic3net_mpe_infer")
def main(config):
    """Main inference entry point."""
    config = OmegaConf.to_container(config)

    # Load model
    if not config.get("MODEL_PATH"):
        raise ValueError("MODEL_PATH must be specified for inference")

    model_path = config["MODEL_PATH"]
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        params = serialization.from_bytes(None, f.read())

    # Setup environment
    env_kwargs = dict(config.get("ENV_KWARGS", {}))
    if config["ENV_NAME"] == "overcooked" and "layout" in env_kwargs:
        layout_name = env_kwargs["layout"]
        if isinstance(layout_name, str):
            from jaxmarl.environments.overcooked.layouts import overcooked_layouts
            from flax.core.frozen_dict import FrozenDict
            env_kwargs["layout"] = FrozenDict(overcooked_layouts[layout_name])

    env = jaxmarl.make(config["ENV_NAME"], **env_kwargs)
    num_agents = env.num_agents
    action_dim = env.action_space(env.agents[0]).n

    network, has_talk = _build_network(config, num_agents, action_dim)

    print(f"Env: {config['ENV_NAME']}, {num_agents} agents, action_dim={action_dim}")
    print(f"Model: {config.get('BASELINE', 'ic3net').upper()}, "
          f"recurrent={config.get('RECURRENT', False)}, has_talk={has_talk}")

    # Run episodes
    num_episodes = config.get("NUM_EPISODES", 10)
    rng = jax.random.PRNGKey(config.get("SEED", 0))

    all_returns = []
    best_seq = None
    best_return = -float("inf")

    for ep in range(num_episodes):
        rng, _rng = jax.random.split(rng)
        state_seq, ep_return = run_episode(env, network, params, config, has_talk, _rng)
        all_returns.append(ep_return)
        print(f"  Episode {ep + 1}/{num_episodes}: return={ep_return:.2f}, "
              f"steps={len(state_seq) - 1}")
        if ep_return > best_return:
            best_return = ep_return
            best_seq = state_seq

    mean_ret = np.mean(all_returns)
    std_ret = np.std(all_returns)
    print(f"\nMean return: {mean_ret:.2f} +/- {std_ret:.2f}")

    # Save GIF if requested
    save_gif = config.get("SAVE_GIF", None)
    if save_gif and best_seq is not None:
        print(f"\nCreating animation ({len(best_seq)} frames) ...")
        if config["ENV_NAME"].startswith("MPE_"):
            from jaxmarl.environments.mpe.mpe_visualizer import MPEVisualizer
            viz = MPEVisualizer(env, best_seq)
            viz.animate(save_fname=save_gif, view=False)
        elif config["ENV_NAME"] == "overcooked":
            from jaxmarl.viz.overcooked_visualizer import OvercookedVisualizer
            viz = OvercookedVisualizer()
            viz.animate(best_seq, env.agent_view_size, filename=save_gif)
        else:
            from jaxmarl.viz.visualizer import Visualizer
            viz = Visualizer(env, best_seq)
            viz.animate(save_fname=save_gif, view=False)
        print(f"Saved to {save_gif}")
        print(f"View with:  xdg-open {save_gif}")


if __name__ == "__main__":
    main()
