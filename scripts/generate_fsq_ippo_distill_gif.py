#!/usr/bin/env python3
"""Render an inference GIF for the FSQ-comm MAPPO student trained by
mappo_rnn_overcooked_v3_fsq_ippo_distill.py.

The training script does not emit a GIF (it only saves actor/critic params), so
this rolls out one full episode with the trained partial-obs FSQ actor and
animates it with the OvercookedV3 visualizer. The env is rebuilt from the run's
saved config yaml so the rendered policy behaves exactly as in training.

Example:
  python scripts/generate_fsq_ippo_distill_gif.py \
    --checkpoint outputs/mappo_fsq_ippo_distill_ctc_20260705/models/mappo_rnn_overcooked_v3_fsq_ippo_distill_coordinated_temporal_conveyor_seed0_vmap0_actor.safetensors \
    --config     outputs/mappo_fsq_ippo_distill_ctc_20260705/models/mappo_rnn_overcooked_v3_fsq_ippo_distill_coordinated_temporal_conveyor_seed0_config.yaml \
    --output     outputs/mappo_fsq_ippo_distill_ctc_20260705.gif --seed 0
"""
import argparse
import os
import sys

# The training module does `from FSQ import FSQ`, which needs baselines/MAPPO on
# the path when imported from elsewhere.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines", "MAPPO"))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.wrappers.baselines import load_params
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer

# Import the exact student network so the rollout matches the trained weights.
from mappo_rnn_overcooked_v3_fsq_ippo_distill import ActorRNN, ScannedRNN


def run_episode(checkpoint, config_path, output, seed, deterministic):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    env_kwargs = dict(cfg["ENV_KWARGS"])
    env = OvercookedV3(**env_kwargs)

    net_config = {
        "NUM_AGENTS": env.num_agents,
        "GRU_HIDDEN_DIM": cfg.get("GRU_HIDDEN_DIM", 128),
        "FC_DIM_SIZE": cfg.get("FC_DIM_SIZE", 64),
        "FSQ_LEVELS": cfg.get("FSQ_LEVELS", [5, 5, 5]),
        "ACTIVATION": cfg.get("ACTIVATION", "relu"),
    }

    network = ActorRNN(env.action_space(env.agents[0]).n, config=net_config)
    params = load_params(checkpoint)  # {"params": {...}} as saved during training

    key = jax.random.PRNGKey(seed)
    key, key_r = jax.random.split(key)
    obs, state = env.reset(key_r)
    hidden = ScannedRNN.initialize_carry(env.num_agents, net_config["GRU_HIDDEN_DIM"])
    done_batch = jnp.zeros((env.num_agents,), dtype=bool)

    state_seq = [state]
    deliveries = 0
    done = False
    while not done:
        key, key_a, key_s = jax.random.split(key, 3)
        obs_batch = jnp.stack([obs[a] for a in env.agents]).reshape(
            -1, *env.observation_space().shape
        )
        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        hidden, pi, _ = network.apply(params, hidden, ac_in)
        if deterministic:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            action = pi.sample(seed=key_a)
        action = action.squeeze(0)
        env_act = {a: action[i] for i, a in enumerate(env.agents)}
        obs, state, reward, dones, info = env.step(key_s, state, env_act)
        deliveries += int(info.get("event/delivery", jnp.array(0)).sum()) if "event/delivery" in info else 0
        done_batch = jnp.array([dones[a] for a in env.agents])
        done = bool(dones["__all__"])
        state_seq.append(state)

    print(f"Episode length: {len(state_seq)} steps; deliveries observed: {deliveries}", flush=True)
    # The visualizer indexes a stacked pytree along a leading time axis.
    stacked = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *state_seq)
    viz = OvercookedV3Visualizer(env)
    viz.animate(stacked, filename=str(output))
    print(f"Saved GIF to: {output}", flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="..._vmap0_actor.safetensors")
    p.add_argument("--config", required=True, help="..._config.yaml saved with the run")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--deterministic", action="store_true",
                   help="argmax actions instead of sampling")
    args = p.parse_args()
    run_episode(args.checkpoint, args.config, args.output, args.seed, args.deterministic)


if __name__ == "__main__":
    main()
