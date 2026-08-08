#!/usr/bin/env python3
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import wandb
from flax import serialization

from baselines.IPPO.ippo_rnn_overcooked_v3 import make_train


def main():
    wandb.init(mode="disabled")

    map_name = sys.argv[1] if len(sys.argv) > 1 else "around_the_island"
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 4_000_000
    save_path = Path(sys.argv[3]) if len(sys.argv) > 3 else Path("checkpoints/ippo_v3_validate_fix")

    config = {
        "ENV_NAME": "overcooked_v3",
        "ENV_KWARGS": {"layout": map_name, "max_steps": 200},
        "GRU_HIDDEN_DIM": 128,
        "FC_DIM_SIZE": 128,
        "ACTIVATION": "relu",
        "TOTAL_TIMESTEPS": total,
        "NUM_ENVS": 128,
        "NUM_STEPS": 200,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "ANNEAL_LR": True,
        "REW_SHAPING_HORIZON": total,
        "SHAPED_REWARD_COEFF": 40.0,
        "LR": 0.0003,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.06,
        "ENT_COEF_MIN": 0.02,
        "ENTROPY_FLOOR": 0.8,
        "ENTROPY_FLOOR_COEF": 0.1,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "WANDB_MODE": "disabled",
        "SEED": 42,
    }

    print(f"Map: {map_name} | TOTAL_TIMESTEPS: {total}")
    train_fn = make_train(config)
    print(f"NUM_UPDATES={config['NUM_UPDATES']}, NUM_ACTORS={config['NUM_ACTORS']}")

    rng = jax.random.PRNGKey(42)
    out = jax.jit(train_fn)(rng)

    train_state = out["runner_state"][0]
    params = train_state.params

    save_path.mkdir(parents=True, exist_ok=True)
    model_path = save_path / "model.msgpack"
    model_path.write_bytes(serialization.to_bytes({"params": params}))

    print(f"Saved checkpoint: {model_path.resolve()}")


if __name__ == "__main__":
    main()
