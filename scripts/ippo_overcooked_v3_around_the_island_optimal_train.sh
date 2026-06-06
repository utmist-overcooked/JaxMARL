#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
export PYTHONPATH="$ROOT_DIR:${PYTHONPATH:-}"

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"

# Prefer the CUDA toolkit bundled with jax[cuda12] when it is installed. This
# avoids accidentally using an older system ptxas that cannot compile CUDA 12
# PTX for the JAX CUDA plugin.
CUDA_NVCC_BIN="$("$PYTHON_BIN" - <<'PY'
from pathlib import Path
import site

for base in site.getsitepackages():
    candidate = Path(base) / "nvidia" / "cuda_nvcc" / "bin"
    if (candidate / "ptxas").exists():
        print(candidate)
        break
PY
)"
if [[ -n "$CUDA_NVCC_BIN" ]]; then
  export PATH="$CUDA_NVCC_BIN:$PATH"
fi

CURRENT_TIME=$(date +"%Y%m%d_%H%M%S")
LAYOUT="${LAYOUT:-around_the_island}"
ENABLE_ITEM_CONVEYORS="${ENABLE_ITEM_CONVEYORS:-false}"
ENABLE_PLAYER_CONVEYORS="${ENABLE_PLAYER_CONVEYORS:-false}"
POT_COOK_TIME="${POT_COOK_TIME:-20}"
POT_BURN_TIME="${POT_BURN_TIME:-$((POT_COOK_TIME * 2))}"
ENABLE_ORDER_QUEUE="${ENABLE_ORDER_QUEUE:-true}"
MAX_ORDERS="${MAX_ORDERS:-5}"
ORDER_GENERATION_RATE="${ORDER_GENERATION_RATE:-1.0}"
ORDER_EXPIRATION_TIME="${ORDER_EXPIRATION_TIME:-0}"
ORDER_QUEUE_MODE="${ORDER_QUEUE_MODE:-alternating}"

# W&B identifiers. Override these from the shell when running controlled
# experiments so each run has a descriptive project name.
WANDB_MODE="${WANDB_MODE:-online}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3_ippo_cnn_${LAYOUT}_order_visible_plateguard_burn${POT_BURN_TIME}_${CURRENT_TIME}}"
WANDB_ENTITY="${WANDB_ENTITY:-dannyb3334-university-of-toronto}"
WANDB_NAME="${WANDB_NAME:-ippo_cnn_overcooked_v3_${LAYOUT}_order_visible_plateguard_burn${POT_BURN_TIME}}"

# Training budget and rollout shape. One PPO update consumes
# NUM_ENVS * NUM_STEPS environment transitions. The current default matches the
# latest successful RNN/CNN comparison setup: 15M requested steps, 128x200
# rollout batches, and full-horizon shaping with a small non-zero floor.
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-15000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-$TOTAL_TIMESTEPS}"
REW_SHAPING_MIN_COEFF="${REW_SHAPING_MIN_COEFF:-0.10}"
MAX_STEPS="${MAX_STEPS:-1000}"
NUM_ENVS="${NUM_ENVS:-128}"
NUM_STEPS="${NUM_STEPS:-200}"

# PPO optimizer defaults. These are the original IPPO CNN dynamics we used for
# the strongest dish_pickup run; reward-shaping choices are the main remaining
# experiment knob.
LR="${LR:-0.0005}"
GAMMA="${GAMMA:-0.99}"
GAE_LAMBDA="${GAE_LAMBDA:-0.95}"
ANNEAL_LR="${ANNEAL_LR:-true}"
CNN_CHANNELS="${CNN_CHANNELS:-128}"
CNN_EMBED_DIM="${CNN_EMBED_DIM:-128}"
FC_DIM_SIZE="${FC_DIM_SIZE:-128}"

# Output locations. The Python trainer saves a msgpack checkpoint plus one GIF
# rollout when training completes normally. Interrupted runs will not have these.
SAVE_PATH="${SAVE_PATH:-$ROOT_DIR/checkpoints/ippo_cnn_v3_${LAYOUT}_${CURRENT_TIME}}"
SAVE_GIF_PATH="${SAVE_GIF_PATH:-$ROOT_DIR/outputs/ippo_cnn_v3_${LAYOUT}_${CURRENT_TIME}.gif}"
EXPECTED_UPDATES=$((TOTAL_TIMESTEPS / NUM_ENVS / NUM_STEPS))
COMPLETED_TIMESTEPS=$((EXPECTED_UPDATES * NUM_ENVS * NUM_STEPS))

mkdir -p "$SAVE_PATH" "$(dirname "$SAVE_GIF_PATH")"

echo "Starting optimal IPPO CNN training for $LAYOUT"
echo "W&B project: $WANDB_PROJECT"
echo "Save path: $SAVE_PATH"
echo "GIF path: $SAVE_GIF_PATH"
echo "Layout: $LAYOUT"
echo "Episode max steps: $MAX_STEPS"
echo "Conveyors: item=$ENABLE_ITEM_CONVEYORS; player=$ENABLE_PLAYER_CONVEYORS"
echo "Pot timing: cook=$POT_COOK_TIME; burn_expiry=$POT_BURN_TIME"
echo "Order queue: enabled=$ENABLE_ORDER_QUEUE; mode=$ORDER_QUEUE_MODE; rate=$ORDER_GENERATION_RATE; expiry=$ORDER_EXPIRATION_TIME; max=$MAX_ORDERS"
echo "Reward shaping horizon: $REW_SHAPING_HORIZON; floor: $REW_SHAPING_MIN_COEFF"
echo "CNN widths: channels=$CNN_CHANNELS; embed=$CNN_EMBED_DIM; fc=$FC_DIM_SIZE"
echo "JAX platforms: ${JAX_PLATFORMS:-auto}"
echo "ptxas: $(command -v ptxas || echo unavailable)"
echo "Expected PPO updates: $EXPECTED_UPDATES; W&B global Step uses env_step up to $COMPLETED_TIMESTEPS"

# Keep the v1-like cook timer and use a burn window twice as long as cooking.
"$PYTHON_BIN" "$ROOT_DIR/baselines/IPPO/ippo_cnn_overcooked_v3.py" \
  --config-name=ippo_cnn_overcooked_v3 \
  WANDB_MODE="$WANDB_MODE" \
  PROJECT="$WANDB_PROJECT" \
  ENTITY="$WANDB_ENTITY" \
  WANDB_NAME="$WANDB_NAME" \
  ENV_KWARGS.layout="$LAYOUT" \
  ENV_KWARGS.max_steps="$MAX_STEPS" \
  ENV_KWARGS.pot_cook_time="$POT_COOK_TIME" \
  ENV_KWARGS.pot_burn_time="$POT_BURN_TIME" \
  ENV_KWARGS.enable_order_queue="$ENABLE_ORDER_QUEUE" \
  ENV_KWARGS.order_generation_rate="$ORDER_GENERATION_RATE" \
  ENV_KWARGS.order_expiration_time="$ORDER_EXPIRATION_TIME" \
  +ENV_KWARGS.max_orders="$MAX_ORDERS" \
  +ENV_KWARGS.order_queue_mode="$ORDER_QUEUE_MODE" \
  +ENV_KWARGS.enable_item_conveyors="$ENABLE_ITEM_CONVEYORS" \
  +ENV_KWARGS.enable_player_conveyors="$ENABLE_PLAYER_CONVEYORS" \
  TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
  NUM_ENVS="$NUM_ENVS" \
  NUM_STEPS="$NUM_STEPS" \
  UPDATE_EPOCHS=4 \
  NUM_MINIBATCHES=8 \
  LR="$LR" \
  GAMMA="$GAMMA" \
  GAE_LAMBDA="$GAE_LAMBDA" \
  CLIP_EPS=0.2 \
  VF_COEF=0.5 \
  ENT_COEF=0.01 \
  MAX_GRAD_NORM=0.5 \
  CNN_CHANNELS="$CNN_CHANNELS" \
  CNN_EMBED_DIM="$CNN_EMBED_DIM" \
  FC_DIM_SIZE="$FC_DIM_SIZE" \
  ACTIVATION=relu \
  ANNEAL_LR="$ANNEAL_LR" \
  REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
  REW_SHAPING_MIN_COEFF="$REW_SHAPING_MIN_COEFF" \
  SHAPED_REWARD_COEFF=30.0 \
  SAVE_CHECKPOINT_PATH="$SAVE_PATH" \
  SAVE_GIF_PATH="$SAVE_GIF_PATH"

echo "Training finished. Model should be in: $SAVE_PATH/model.msgpack"
echo "GIF should be in: $SAVE_GIF_PATH"
