"""Standalone IPPO v3 smoke test — no Hydra, no W&B."""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.9"

import jax
import jax.numpy as jnp
import numpy as np
import wandb

# Initialize wandb in disabled mode before importing ippo module
wandb.init(mode="disabled")

from baselines.IPPO.ippo_rnn_overcooked_v3 import make_train

MAP = sys.argv[1] if len(sys.argv) > 1 else "cramped_room"
TOTAL = int(sys.argv[2]) if len(sys.argv) > 2 else 2_000_000

config = {
    "ENV_NAME": "overcooked_v3",
    "ENV_KWARGS": {"layout": MAP, "max_steps": 200},
    "GRU_HIDDEN_DIM": 128,
    "FC_DIM_SIZE": 128,
    "ACTIVATION": "relu",
    "TOTAL_TIMESTEPS": TOTAL,
    "NUM_ENVS": 128,
    "NUM_STEPS": 200,
    "UPDATE_EPOCHS": 4,
    "NUM_MINIBATCHES": 8,
    "ANNEAL_LR": True,
    "REW_SHAPING_HORIZON": TOTAL,
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

print(f"Map: {MAP} | TOTAL_TIMESTEPS: {TOTAL}")
print(f"  SHAPED_REWARD_COEFF={config['SHAPED_REWARD_COEFF']}, ENT_COEF={config['ENT_COEF']}, NUM_ENVS={config['NUM_ENVS']}")
print("Building training function...")
train_fn = make_train(config)
print(f"NUM_UPDATES={config['NUM_UPDATES']}, NUM_ACTORS={config['NUM_ACTORS']}")

print("JIT compiling + running training...")
rng = jax.random.PRNGKey(42)
train_jit = jax.jit(train_fn)
out = train_jit(rng)

# Extract metrics (arrays of shape (NUM_UPDATES,))
metrics = out["metrics"]
ep_returns = np.array(metrics["returned_episode_returns"])
mean_rew = np.array(metrics["mean_reward"])
max_rew = np.array(metrics["max_reward"])

print(f"\nTraining complete! {len(ep_returns)} updates on {MAP}")
print(f"Final EpRet:     {ep_returns[-1]:.4f}")
print(f"Final MeanRew:   {mean_rew[-1]:.6f}")
print(f"Max MeanRew:     {mean_rew.max():.6f}")
print(f"Max MaxRew:      {max_rew.max():.4f}")

n = len(ep_returns)
samples = [0, n//10, n//4, n//2, 3*n//4, n-1]
for i in samples:
    print(f"  Update {i+1}: EpRet={ep_returns[i]:.4f} MeanRew={mean_rew[i]:.6f} MaxRew={max_rew[i]:.4f}")
