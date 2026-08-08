#!/usr/bin/env bash
# FSQ communication distillation on coordinated_temporal_conveyor.
# Teacher: the trained full-observation IPPO checkpoint
#   checkpoints/ippo_ctc_15m_20260703 (wandb ocv3_ctc_comparison/ix4u8fte).
# Student: partially-observed (agent_view_size=2) MAPPO actor with an FSQ comm
#   channel, distilled toward the teacher's action logits (cosine-decayed KL).
# Env mirrors the IPPO teacher's CTC training env exactly so the teacher stays
# in-distribution; the only difference is the student's partial view.
# Same wandb project as the IPPO/QMIX CTC comparison runs.
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=mappo_fsq_ippo_distill_ctc_20260705

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python \
  ${REPO}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_ippo_distill.py \
  --config-name=mappo_rnn_overcooked_v3_fsq_ippo_distill \
  WANDB_MODE=online PROJECT=ocv3_ctc_comparison ENTITY=zacharytang24- \
  WANDB_RUN_NAME=${NAME} USE_RICH_MONITOR=False \
  TEACHER_ACTOR_PATH=${REPO}/checkpoints/ippo_ctc_15m_20260703 \
  WANDB_DIR=${REPO}/outputs/${NAME} \
  TOTAL_TIMESTEPS=3e7 NUM_ENVS=256 NUM_STEPS=256 NUM_MINIBATCHES=64 \
  SEED=0 NUM_SEEDS=1 \
  > ${REPO}/outputs/${NAME}_train.log 2>&1"

echo "launched tmux session jaxmarl_${NAME}; log -> outputs/${NAME}_train.log"
