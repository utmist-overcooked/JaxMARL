#!/usr/bin/env bash
# FSQ communication distillation on coordinated_temporal_conveyor.
# Teacher: the trained full-observation QMIX agent Q-network
#   checkpoints/qmix_ctc_15m_handoff_20260702/.../seed42_vmap0.safetensors
#   (wandb ocv3_ctc_comparison/urm30dyu). Real greedy policy delivers ~4/episode.
# Student: partially-observed (agent_view_size=2) MAPPO actor with an FSQ comm
#   channel, distilled toward the teacher's action ranking. QMIX Q-values have a
#   tiny spread, so we standardize them per action-vector (TEACHER_Q_STANDARDIZE)
#   and use DISTILL_TEMPERATURE=0.5 so the KL target is meaningfully peaked.
# Env mirrors the QMIX teacher's CTC training env exactly (pots 60/90, max_steps
# 400) so the teacher stays in-distribution; only the student's view is partial.
# Companion to run_mappo_fsq_ippo_distill_ctc_20260705.sh (same project).
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=mappo_fsq_qmix_distill_ctc_20260705
CK=${REPO}/checkpoints/qmix_ctc_15m_handoff_20260702/overcooked_v3_coordinated_temporal_conveyor/qmix_rnn_overcooked_v3_coordinated_temporal_conveyor_seed42_vmap0.safetensors

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python \
  ${REPO}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_qmix_distill.py \
  --config-name=mappo_rnn_overcooked_v3_fsq_qmix_distill \
  WANDB_MODE=online PROJECT=ocv3_ctc_comparison ENTITY=zacharytang24- \
  WANDB_RUN_NAME=${NAME} USE_RICH_MONITOR=False \
  TEACHER_ACTOR_PATH=${CK} \
  WANDB_DIR=${REPO}/outputs/${NAME} \
  TOTAL_TIMESTEPS=3e7 NUM_ENVS=256 NUM_STEPS=256 NUM_MINIBATCHES=64 \
  SEED=0 NUM_SEEDS=1 \
  > ${REPO}/outputs/${NAME}_train.log 2>&1"

echo "launched tmux session jaxmarl_${NAME}; log -> outputs/${NAME}_train.log"
