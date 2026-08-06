#!/usr/bin/env bash
# FSQ communication distillation on coordinated_temporal_conveyor, from the IPPO
# teacher RETRAINED in the QMIX-matched env (checkpoints/ippo_ctc_15m_qmixenv_20260705).
# This makes the IPPO-teacher and QMIX-teacher distilled students directly comparable:
# both run in the SAME env (max_steps=400, pot_cook=60, pot_burn=90).
# Only difference vs run_mappo_fsq_ippo_distill_ctc_20260705.sh: the new teacher path
# and the env timings overridden to match it (config defaults were 800 / 20 / 40).
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=mappo_fsq_ippo_distill_ctc_qmixenv_20260705

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python \
  ${REPO}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_ippo_distill.py \
  --config-name=mappo_rnn_overcooked_v3_fsq_ippo_distill \
  WANDB_MODE=online PROJECT=ocv3_ctc_comparison ENTITY=zacharytang24- \
  WANDB_RUN_NAME=${NAME} USE_RICH_MONITOR=False \
  TEACHER_ACTOR_PATH=${REPO}/checkpoints/ippo_ctc_15m_qmixenv_20260705 \
  ENV_KWARGS.max_steps=400 ENV_KWARGS.pot_cook_time=60 ENV_KWARGS.pot_burn_time=90 \
  WANDB_DIR=${REPO}/outputs/${NAME} \
  TOTAL_TIMESTEPS=3e7 NUM_ENVS=256 NUM_STEPS=256 NUM_MINIBATCHES=64 \
  SEED=0 NUM_SEEDS=1 \
  > ${REPO}/outputs/${NAME}_train.log 2>&1"

echo "launched tmux session jaxmarl_${NAME}; log -> outputs/${NAME}_train.log"
