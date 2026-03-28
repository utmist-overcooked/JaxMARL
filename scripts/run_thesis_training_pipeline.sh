#!/usr/bin/env bash
set -euo pipefail

ROOT=/workspace/JaxMARL
PY=${PY:-$ROOT/.venv-jaxcuda/bin/python}
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT/outputs/thesis_pipeline_${STAMP}"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"
ENTITY="${ENTITY:-dannyb3334-university-of-toronto}"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export WANDB_DIR="$ROOT/wandb"
export WANDB_MODE=online
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.85

set -a
source "$ROOT/.env"
set +a
WANDB_API_KEY="${WANDB_API_KEY%\"}"
WANDB_API_KEY="${WANDB_API_KEY#\"}"
export WANDB_API_KEY

echo "Pipeline output dir: $OUT_DIR"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,cuda_version --format=csv,noheader > "$OUT_DIR/cuda_info.csv" 2>&1 || true
fi

echo "[smoke] MAPPO"
"$PY" "$ROOT/baselines/MAPPO/mappo_rnn_overcooked_v3.py" \
  --config-name=mappo_homogenous_rnn_overcooked_v3 \
  WANDB_MODE=disabled \
  TOTAL_TIMESTEPS=256 \
  NUM_ENVS=1 \
  NUM_STEPS=32 \
  NUM_MINIBATCHES=1 \
  UPDATE_EPOCHS=1 \
  NUM_TEST_ENVS=1 \
  TEST_INTERVAL=1.0 \
  ENV_KWARGS.layout=single_file \
  SAVE_PATH="$OUT_DIR/mappo_smoke_ckpt" \
  > "$LOG_DIR/mappo_smoke.log" 2>&1

echo "[smoke] IC3Net"
"$PY" "$ROOT/baselines/IC3Net/ic3net_train.py" \
  --config-name=ic3net_overcooked_v3_single_file \
  WANDB_MODE=disabled \
  TOTAL_TIMESTEPS=256 \
  NUM_ENVS=1 \
  NUM_STEPS=16 \
  LOG_EVERY=1 \
  ENV_KWARGS.max_steps=32 \
  SAVE_PATH="$OUT_DIR/ic3net_smoke_ckpt" \
  > "$LOG_DIR/ic3net_smoke.log" 2>&1

echo "[full] IC3Net thesis runs"
bash "$ROOT/scripts/ic3net_overcooked_v3_thesis_runs.sh" \
  > "$LOG_DIR/ic3net_full.log" 2>&1

if ls "$ROOT"/outputs/ic3net_overcooked_v3_* >/dev/null 2>&1; then
  IC3_PROJECT=$(basename "$(ls -dt "$ROOT"/outputs/ic3net_overcooked_v3_* | head -n1)" | sed 's/^ic3net_overcooked_v3_/overcookedv3_ic3net_thesis_/')
  "$PY" "$ROOT/get_wandb_run_logs.py" --entity "$ENTITY" --project "$IC3_PROJECT" --output-dir "$OUT_DIR/wandb_ic3net" \
    > "$LOG_DIR/ic3net_wandb_export.log" 2>&1 || true
fi

echo "[full] MAPPO thesis runs"
bash "$ROOT/scripts/mappo_overcooked_v3_thesis_runs.sh" \
  > "$LOG_DIR/mappo_full.log" 2>&1

if ls "$ROOT"/outputs/mappo_overcooked_v3_* >/dev/null 2>&1; then
  MAPPO_PROJECT=$(basename "$(ls -dt "$ROOT"/outputs/mappo_overcooked_v3_* | head -n1)" | sed 's/^mappo_overcooked_v3_/overcookedv3_mappo_thesis_/')
  "$PY" "$ROOT/get_wandb_run_logs.py" --entity "$ENTITY" --project "$MAPPO_PROJECT" --output-dir "$OUT_DIR/wandb_mappo" \
    > "$LOG_DIR/mappo_wandb_export.log" 2>&1 || true
fi

echo "Pipeline complete"