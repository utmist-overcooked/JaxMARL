#!/usr/bin/env bash
# TransfQMix (transformer QMIX) teacher on coordinated_temporal_conveyor, on the SAME
# common env as the IPPO / MAPPO / QMIX teachers (max_steps=400, pots 60/90, full obs).
#
# Entities = grid cells: the (H,W,C) full-obs grid is tokenized cell-wise into H*W tokens,
# each carrying its channel features + normalized (row,col) coords, so the transformer
# attends over the kitchen layout (see jaxmarl/wrappers/transformers.py).
#
# Reward setup: same env reward table as the other teachers; shaping strength/anneal taken
# from the QMIX sweep winner (SHAPED_REWARD_COEFF=5, REW_SHAPING_MIN_COEFF=0.5), the best
# known value-based recipe on this map.
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=transf_qmix_ctc_15m_20260707

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PYTHONPATH=${REPO}:\${PYTHONPATH:-} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 && \
mkdir -p ${REPO}/outputs ${REPO}/checkpoints && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python ${REPO}/baselines/QLearning/transf_qmix.py \
  +alg=transf_qmix_overcooked_v3 \
  WANDB_MODE=online PROJECT=ocv3_ctc_comparison ENTITY=zacharytang24- \
  +alg.WANDB_NAME=${NAME} SEED=42 \
  SAVE_PATH=${REPO}/checkpoints/${NAME} \
  > ${REPO}/outputs/${NAME}_train.log 2>&1"

echo "launched tmux session jaxmarl_${NAME}; log -> outputs/${NAME}_train.log"
