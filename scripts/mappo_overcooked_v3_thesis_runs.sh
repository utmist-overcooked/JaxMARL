#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/JaxMARL
PY=${PY:-/workspace/JaxMARL/.venv-jaxcuda/bin/python}
STAMP="$(date +%Y%m%d_%H%M%S)"
PROJECT_BASE="${PROJECT_BASE:-overcookedv3_mappo_thesis}"
PROJECT="${PROJECT_BASE}_${STAMP}"
ENTITY="${ENTITY:-dannyb3334-university-of-toronto}"
OUT_DIR="$ROOT/outputs/mappo_overcooked_v3_${STAMP}"
LOG_DIR="$OUT_DIR/logs"
CKPT_DIR="$OUT_DIR/checkpoints"
mkdir -p "$LOG_DIR" "$CKPT_DIR"

if [ -f "$ROOT/.env" ]; then
  set -a
  source "$ROOT/.env"
  set +a
fi
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_API_KEY="${WANDB_API_KEY%\"}"
WANDB_API_KEY="${WANDB_API_KEY#\"}"
if [ -n "$WANDB_API_KEY" ]; then
  export WANDB_API_KEY
else
  echo "WANDB_API_KEY not exported; using existing wandb CLI credentials if available"
fi
export WANDB_MODE=online
export WANDB_DIR="$ROOT/wandb"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

MAPS=(single_file around_the_island middle_conveyor)

echo "MAPPO W&B project: $PROJECT"
echo "Output dir: $OUT_DIR"

for MAP in "${MAPS[@]}"; do
  RUN_NAME="mappo_overcooked_v3_${MAP}_${STAMP}"
  LOG_PATH="$LOG_DIR/${RUN_NAME}.log"
  SAVE_PATH="$CKPT_DIR/$MAP"
  mkdir -p "$SAVE_PATH"

  case "$MAP" in
    single_file)
      TOTAL_TIMESTEPS=15000000
      NUM_ENVS=32
      ENT_COEF=0.01
      ;;
    around_the_island)
      TOTAL_TIMESTEPS=25000000
      NUM_ENVS=24
      ENT_COEF=0.02
      ;;
    middle_conveyor)
      TOTAL_TIMESTEPS=25000000
      NUM_ENVS=24
      ENT_COEF=0.02
      ;;
  esac

  echo "--- MAPPO training on $MAP ---"
  (
    cd "$ROOT"
    "$PY" baselines/MAPPO/mappo_rnn_overcooked_v3.py \
      --config-name=mappo_homogenous_rnn_overcooked_v3 \
      ENV_KWARGS.layout="$MAP" \
      TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
      NUM_ENVS="$NUM_ENVS" \
      ENT_COEF="$ENT_COEF" \
      PROJECT="$PROJECT" \
      ENTITY="$ENTITY" \
      WANDB_MODE=online \
      WANDB_NAME="$RUN_NAME" \
      SAVE_PATH="$SAVE_PATH"
  ) 2>&1 | tee "$LOG_PATH"
done

echo "MAPPO thesis runs complete. Logs: $LOG_DIR"