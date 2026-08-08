#!/usr/bin/env python3
"""Batched greedy delivery eval for FSQ-distilled students on the common CTC env.

Runs N parallel independent episodes per student (one full episode each, greedy
argmax actions) and reports mean +/- std deliveries per episode. All three
distilled students share the same partial-obs env (from each run's saved config),
so the numbers are directly comparable.
"""
import argparse
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "baselines", "MAPPO"))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from jaxmarl.environments.overcooked_v3 import OvercookedV3
from jaxmarl.wrappers.baselines import load_params
from mappo_rnn_overcooked_v3_fsq_ippo_distill import ActorRNN, ScannedRNN


def eval_student(actor_ckpt, config_path, n_envs, seed, deterministic, random_start=False):
    cfg = OmegaConf.to_container(OmegaConf.load(config_path), resolve=True)
    env_kwargs = dict(cfg["ENV_KWARGS"])
    if random_start:
        # Force episode diversity: the students trained with a fixed start, so a
        # deterministic env replays one identical episode. Randomizing agent start
        # positions gives a distribution (tests generalization; slightly OOD).
        env_kwargs["random_agent_positions"] = True
    env = OvercookedV3(**env_kwargs)
    max_steps = cfg["ENV_KWARGS"]["max_steps"]
    num_agents = env.num_agents
    num_actors = num_agents * n_envs
    obs_shape = env.observation_space().shape

    net_config = {
        "NUM_AGENTS": num_agents,
        "GRU_HIDDEN_DIM": cfg.get("GRU_HIDDEN_DIM", 128),
        "FC_DIM_SIZE": cfg.get("FC_DIM_SIZE", 64),
        "FSQ_LEVELS": cfg.get("FSQ_LEVELS", [5, 5, 5]),
        "ACTIVATION": cfg.get("ACTIVATION", "relu"),
    }
    net = ActorRNN(env.action_space(env.agents[0]).n, config=net_config)
    params = load_params(actor_ckpt)

    key = jax.random.PRNGKey(seed)
    key, rk = jax.random.split(key)
    obsv, state = jax.vmap(env.reset)(jax.random.split(rk, n_envs))
    hidden = ScannedRNN.initialize_carry(num_actors, net_config["GRU_HIDDEN_DIM"])
    done_batch = jnp.zeros((num_actors,), dtype=bool)

    def step(carry, _):
        obsv, state, hidden, done_batch, key = carry
        obs_batch = jnp.stack([obsv[a] for a in env.agents]).reshape(-1, *obs_shape)
        ac_in = (obs_batch[np.newaxis, :], done_batch[np.newaxis, :])
        hidden, pi, _ = net.apply(params, hidden, ac_in)
        if deterministic:
            action = jnp.argmax(pi.logits, axis=-1)
        else:
            key, ak = jax.random.split(key)
            action = pi.sample(seed=ak)
        action = action.squeeze(0).reshape(num_agents, n_envs)
        env_act = {a: action[i] for i, a in enumerate(env.agents)}
        key, sk = jax.random.split(key)
        obsv, state, reward, dones, info = jax.vmap(env.step)(
            jax.random.split(sk, n_envs), state, env_act
        )
        d = info["event/delivery"]
        while d.ndim > 1:
            d = d.sum(axis=-1)
        done_batch = jnp.stack([dones[a] for a in env.agents]).reshape(num_actors)
        return (obsv, state, hidden, done_batch, key), d

    (_, _, _, _, _), deliv_per_step = jax.lax.scan(
        step, (obsv, state, hidden, done_batch, key), None, max_steps
    )
    # deliv_per_step: (max_steps, n_envs) -> per-episode totals
    per_ep = np.asarray(deliv_per_step.sum(axis=0))
    return per_ep


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-envs", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sampled", action="store_true", help="sample instead of greedy")
    p.add_argument("--random-start", action="store_true",
                   help="randomize agent start positions for episode diversity")
    args = p.parse_args()

    base = "outputs"
    arms = [
        ("IPPO-teacher", f"{base}/mappo_fsq_ippo_distill_ctc_qmixenv_20260705/models/mappo_rnn_overcooked_v3_fsq_ippo_distill_coordinated_temporal_conveyor_seed0"),
        ("QMIX-teacher", f"{base}/mappo_fsq_qmix_distill_ctc_20260705/models/mappo_rnn_overcooked_v3_fsq_qmix_distill_coordinated_temporal_conveyor_seed0"),
        ("MAPPO-teacher", f"{base}/mappo_fsq_mappo_distill_ctc_20260705/models/mappo_rnn_overcooked_v3_fsq_mappo_distill_coordinated_temporal_conveyor_seed0"),
    ]
    mode = "sampled" if args.sampled else "greedy"
    start = "random-start" if args.random_start else "fixed-start"
    print(f"\nDeliveries/episode over {args.n_envs} episodes ({mode}, {start}, common env: 400 steps, pots 60/90)\n")
    print(f"{'arm':16} {'mean':>7} {'std':>7} {'min':>5} {'max':>5} {'%>0':>6}")
    for label, stem in arms:
        ckpt = f"{stem}_vmap0_actor.safetensors"
        cfg = f"{stem}_config.yaml"
        per_ep = eval_student(ckpt, cfg, args.n_envs, args.seed, not args.sampled,
                              random_start=args.random_start)
        print(f"{label:16} {per_ep.mean():7.2f} {per_ep.std():7.2f} "
              f"{per_ep.min():5.0f} {per_ep.max():5.0f} {100*(per_ep>0).mean():5.1f}%")


if __name__ == "__main__":
    main()
