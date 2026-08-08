#!/usr/bin/env bash
# FSQ communication distillation on coordinated_temporal_conveyor, from the full-obs
# MAPPO teacher (outputs/mappo_ctc_15m_qmixenv_20260705, returns ~60 => ~2-3 deliv/ep).
# Third arm of the teacher comparison — all three distilled students now share the
# same common env (max_steps=400, pots 60/90): IPPO-teacher, QMIX-teacher, MAPPO-teacher.
# Teacher is a policy, so we distill on its logits directly (DISTILL_TEMPERATURE=1.0,
# no Q-standardization), same as the IPPO-teacher arm.
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=mappo_fsq_mappo_distill_ctc_20260705
TEACHER=${REPO}/outputs/mappo_ctc_15m_qmixenv_20260705/models/mappo_rnn_overcooked_v3_full_obs_coordinated_temporal_conveyor_seed42_vmap0_actor.safetensors

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python \
  ${REPO}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_mappo_distill.py \
  --config-name=mappo_rnn_overcooked_v3_fsq_mappo_distill \
  WANDB_MODE=online PROJECT=ocv3_ctc_comparison ENTITY=zacharytang24- \
  WANDB_RUN_NAME=${NAME} USE_RICH_MONITOR=False \
  TEACHER_ACTOR_PATH=${TEACHER} \
  WANDB_DIR=${REPO}/outputs/${NAME} \
  TOTAL_TIMESTEPS=3e7 NUM_ENVS=256 NUM_STEPS=256 NUM_MINIBATCHES=64 \
  SEED=0 NUM_SEEDS=1 \
  > ${REPO}/outputs/${NAME}_train.log 2>&1"

echo "launched tmux session jaxmarl_${NAME}; log -> outputs/${NAME}_train.log"
