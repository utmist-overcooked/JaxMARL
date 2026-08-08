#!/usr/bin/env bash
# wandb Bayesian sweep for the QMIX CTC teacher — hyperparams only, env held FIXED
# (common CTC teacher env: coordinated_temporal_conveyor, max_steps=400, pots 60/90,
# full obs, alternating queue, conveyors). Full-episode-BPTT regime is fixed too;
# swept: SHAPED_REWARD_COEFF, REW_SHAPING_MIN_COEFF, NUM_ENVS, LR, EPS_DECAY,
# TARGET_UPDATE_INTERVAL. Metric: test_returned_episode_returns (greedy), maximised.
# ~16 trials x 6M steps => long (~10-13h) sequential run on one GPU.
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=qmix_ctc_teacher_sweep_20260706
COUNT=${1:-16}
BUDGET=${2:-6000000}

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export PYTHONPATH=${REPO} && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python ${REPO}/scripts/sweep_qmix_teacher_ctc.py \
  --count ${COUNT} --budget ${BUDGET} \
  --project ocv3_qmix_ctc_teacher_sweep --entity zacharytang24- \
  > ${REPO}/outputs/${NAME}.log 2>&1"

echo "launched tmux session jaxmarl_${NAME} (count=${COUNT}, budget=${BUDGET}); log -> outputs/${NAME}.log"
