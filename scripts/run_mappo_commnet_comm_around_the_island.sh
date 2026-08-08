#!/usr/bin/env bash
# MAPPO-RNN + CommNet communication (no distillation) on around_the_island.
#
# Agents are partially observed (agent_view_size=2) and communicate through a
# continuous CommNet channel: each agent reads the mean of the other agents'
# hidden states, folded back in over 2 rounds of  h <- tanh(H h + C c).
# The critic stays centralized over the concatenated partial observations.
#
# Companion to run_mappo_fsq_comm_around_the_island.sh, which is identical
# apart from using the discrete FSQ codebook - so the two are comparable.
#
# The GPU on this host is currently unusable (kernel module 535 vs userspace
# libs 580 - needs a root module reload, see scripts/fix_nvidia_driver.sh), so
# this script WAITS for a working GPU and launches the moment one appears.
# Set FSQ_WAIT_GPU=0 to skip waiting and run immediately on whatever backend
# is available.
set -uo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=mappo_commnet_comm_around_the_island_10m
LOG=$REPO/outputs/${NAME}_train.log
WAIT_GPU=${FSQ_WAIT_GPU:-1}
POLL_SECONDS=${FSQ_POLL_SECONDS:-300}

cd "$REPO"
export PATH=$REPO/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99
export PYTHONUNBUFFERED=1
mkdir -p "$REPO/outputs"

gpu_ready() {
  "$REPO/.venv/bin/python" - <<'PY' 2>/dev/null
import sys
try:
    import jax
    sys.exit(0 if any(d.platform == "gpu" for d in jax.devices()) else 1)
except Exception:
    sys.exit(1)
PY
}

if [ "$WAIT_GPU" = "1" ]; then
  echo "[$(date +%F_%T)] waiting for a working GPU (polling every ${POLL_SECONDS}s)..." | tee -a "$LOG"
  until gpu_ready; do sleep "$POLL_SECONDS"; done
  echo "[$(date +%F_%T)] GPU is available - starting training" | tee -a "$LOG"
fi

"$REPO/.venv/bin/python" "$REPO/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq.py" \
  ENV_KWARGS.layout=around_the_island \
  ENV_KWARGS.agent_view_size=2 \
  ENV_KWARGS.max_steps=400 \
  COMM_TYPE=commnet COMMNET_ROUNDS=2 \
  DISABLE_FSQ_COMM=False \
  TOTAL_TIMESTEPS=1e7 \
  REW_SHAPING_HORIZON=5e6 \
  NUM_ENVS=256 NUM_STEPS=256 UPDATE_EPOCHS=4 NUM_MINIBATCHES=64 \
  LR=0.00025 ANNEAL_LR=True ENT_COEF=0.04 \
  GRU_HIDDEN_DIM=128 FC_DIM_SIZE=64 \
  SEED=0 NUM_SEEDS=1 \
  WANDB_MODE=online PROJECT=ocv3_mappo_commnet_comm ENTITY=zacharytang24- \
  WANDB_RUN_NAME=$NAME \
  USE_RICH_MONITOR=False \
  DISABLE_CHECKPOINTS=False \
  CHECKPOINT_GIF=False CHECKPOINT_FSQ_VIEWER=False \
  >> "$LOG" 2>&1

echo "[$(date +%F_%T)] finished (exit $?)" >> "$LOG"
