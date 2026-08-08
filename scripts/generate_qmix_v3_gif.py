#!/usr/bin/env python3
"""Render a GIF from a trained QMIX (qmix_rnn) overcooked_v3 checkpoint.

Reuses the env construction and greedy rollout from baselines/QLearning/qmix_rnn.py
so the rendered policy matches training exactly. Lets you pick a different GIF seed.
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import jax.numpy as jnp
from jaxmarl.wrappers.baselines import load_params
from baselines.QLearning.qmix_rnn import env_from_config, get_greedy_rollout


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="path to *.safetensors")
    p.add_argument("--layout", default="coordinated_temporal_conveyor")
    p.add_argument("--output", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=400)
    p.add_argument("--hidden-size", type=int, default=256)
    args = p.parse_args()

    config = {
        "ENV_NAME": "overcooked_v3",
        "HIDDEN_SIZE": args.hidden_size,
        "SEED": args.seed,
        "GIF_SEED": args.seed,
        "ENV_KWARGS": {
            "layout": args.layout,
            "agent_view_size": None,
            "max_steps": args.max_steps,
            "pot_cook_time": 60,
            "pot_burn_time": 90,
            "enable_order_queue": True,
            "max_orders": 5,
            "order_generation_rate": 1.0,
            "order_expiration_time": 0,
            "recipe_mode": "alternating",
            "plate_pickup_guard": 1,
            "enable_item_conveyors": True,
            "enable_player_conveyors": False,
        },
    }

    env, _ = env_from_config({**config, "ENV_KWARGS": dict(config["ENV_KWARGS"])})
    params = load_params(args.checkpoint)
    agent_params = params["agent"] if "agent" in params else params

    state_seq, viz = get_greedy_rollout(agent_params, config, env, max_steps=args.max_steps)
    viz.animate(state_seq, filename=args.output)
    print(f"Saved GIF (seed={args.seed}) to: {args.output}", flush=True)


if __name__ == "__main__":
    main()
