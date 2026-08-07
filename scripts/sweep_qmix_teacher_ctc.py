#!/usr/bin/env python3
"""wandb Bayesian sweep for the QMIX CTC teacher — hyperparams only, env FIXED.

The environment (ENV_KWARGS) is held byte-identical to the common CTC teacher env
(coordinated_temporal_conveyor, max_steps=400, pots 60/90, full obs, alternating
queue, conveyors). The full-episode-BPTT regime (NUM_STEPS=400=max_steps, NUM_EPOCHS=8,
HIDDEN_SIZE=256, Huber loss, mixer dims) is also fixed. Only training hyperparameters
that plausibly unlock better CTC delivery are swept.

Metric: test_returned_episode_returns (greedy eval during training), maximised.
Each trial runs a shortened budget so many configs can be evaluated on one GPU.

Run:
  cd /student/brownd58/dev/JaxMARL
  export PATH=.../cuda_nvcc/bin:$PATH ; export PYTHONPATH=/student/brownd58/dev/JaxMARL
  python scripts/sweep_qmix_teacher_ctc.py --count 16 --budget 6000000
"""
import argparse
import copy
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, REPO_ROOT)

import jax
import wandb
from omegaconf import OmegaConf

from baselines.QLearning.qmix_rnn import make_train, env_from_config

CFG_DIR = os.path.join(REPO_ROOT, "baselines", "QLearning", "config")

# Common CTC teacher env — held EXACTLY fixed across all trials.
COMMON_ENV = {
    "layout": "coordinated_temporal_conveyor",
    "agent_view_size": None,
    "max_steps": 400,
    "pot_cook_time": 60,
    "pot_burn_time": 90,
    "enable_order_queue": True,
    "max_orders": 5,
    "order_generation_rate": 1.0,
    "order_expiration_time": 0,
    "order_queue_mode": "alternating",
    "plate_pickup_guard": 1,
    "enable_item_conveyors": True,
    "enable_player_conveyors": False,
}

# Swept hyperparameters (env untouched). Values chosen around the urm30dyu baseline.
SWEEP_PARAMETERS = {
    # dense-shaping strength — the biggest suspected lever (baseline was 1.0, weak)
    "SHAPED_REWARD_COEFF": {"values": [1.0, 5.0, 15.0, 30.0]},
    # shaping anneal floor (baseline held constant at 1.0)
    "REW_SHAPING_MIN_COEFF": {"values": [0.1, 0.5, 1.0]},
    # parallel rollouts — exploration/discovery (baseline 4, memory-bounded)
    "NUM_ENVS": {"values": [4, 8]},
    "LR": {"values": [0.00005, 0.0001, 0.00025]},
    # exploration horizon (fraction of updates to anneal epsilon)
    "EPS_DECAY": {"values": [0.2, 0.4, 0.6]},
    "TARGET_UPDATE_INTERVAL": {"values": [10, 50, 200]},
}


def build_base_config(budget):
    top = OmegaConf.load(os.path.join(CFG_DIR, "config.yaml"))
    alg = OmegaConf.load(os.path.join(CFG_DIR, "alg", "ql_rnn_overcooked_v3.yaml"))
    cfg = {**OmegaConf.to_container(top, resolve=True),
           **OmegaConf.to_container(alg, resolve=True)}
    cfg["ENV_KWARGS"] = dict(COMMON_ENV)

    # Fixed full-episode-BPTT regime (urm30dyu recipe), NOT swept.
    cfg.update({
        "NUM_STEPS": 400,             # == max_steps (full-episode BPTT)
        "BUFFER_SIZE": 512,
        "BUFFER_BATCH_SIZE": 32,
        "HIDDEN_SIZE": 256,
        "MIXER_EMBEDDING_DIM": 64,
        "MIXER_HYPERNET_HIDDEN_DIM": 256,
        "MIXER_INIT_SCALE": 0.001,
        "EPS_START": 1.0,
        "EPS_FINISH": 0.05,
        "MAX_GRAD_NORM": 10,
        "TAU": 1.0,
        "NUM_EPOCHS": 8,
        "LEARNING_STARTS": 10000,
        "LR_LINEAR_DECAY": False,
        "GAMMA": 0.99,
        "REW_SCALE": 1.0,
        "TOTAL_TIMESTEPS": budget,
        "REW_SHAPING_HORIZON": budget,
        # greedy eval => metric reflects real delivery performance
        "TEST_DURING_TRAINING": True,
        "TEST_INTERVAL": 0.1,
        "TEST_NUM_STEPS": 400,        # one full episode
        "TEST_NUM_ENVS": 128,
        "NUM_SEEDS": 1,
        "SEED": 0,
        "SAVE_PATH": None,            # don't save every trial
    })
    return cfg


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--count", type=int, default=16, help="number of trials")
    p.add_argument("--budget", type=int, default=6_000_000, help="timesteps/trial")
    p.add_argument("--project", default="ocv3_qmix_ctc_teacher_sweep")
    p.add_argument("--entity", default="zacharytang24-")
    p.add_argument("--sweep-id", default=None, help="attach an agent to an existing sweep")
    args = p.parse_args()

    base_config = build_base_config(args.budget)
    base_config["PROJECT"] = args.project
    base_config["ENTITY"] = args.entity
    base_config["WANDB_MODE"] = "online"

    # env built ONCE — identical for every trial (only hyperparams vary)
    env, env_name = env_from_config(copy.deepcopy(base_config))

    def run_trial():
        wandb.init(project=args.project, entity=args.entity)
        config = copy.deepcopy(base_config)
        for k, v in dict(wandb.config).items():
            config[k] = v
        print("[sweep] trial config overrides:", dict(wandb.config), flush=True)
        rng = jax.random.PRNGKey(config["SEED"])
        rngs = jax.random.split(rng, config["NUM_SEEDS"])
        train_vjit = jax.jit(jax.vmap(make_train(config, env)))
        jax.block_until_ready(train_vjit(rngs))

    if args.sweep_id:
        sweep_id = args.sweep_id
    else:
        sweep_config = {
            "name": f"qmix_ctc_teacher_{env_name}",
            "method": "bayes",
            "metric": {"name": "test_returned_episode_returns", "goal": "maximize"},
            "parameters": SWEEP_PARAMETERS,
        }
        wandb.login()
        sweep_id = wandb.sweep(sweep_config, entity=args.entity, project=args.project)
        print(f"[sweep] created sweep {sweep_id}", flush=True)

    wandb.agent(sweep_id, run_trial, count=args.count,
                entity=args.entity, project=args.project)


if __name__ == "__main__":
    main()
