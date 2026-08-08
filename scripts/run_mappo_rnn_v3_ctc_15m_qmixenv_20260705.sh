#!/usr/bin/env bash
# MAPPO-RNN (full-observation) teacher on coordinated_temporal_conveyor, trained in
# the SAME env as the QMIX and (retrained) IPPO teachers so all three teachers — and
# their FSQ-distilled students — are directly comparable.
#   env (QMIX-matched): full obs (agent_view_size=null), conveyors ON, alternating
#     queue, gen_rate=1.0, expiration=0, max_orders=5, plate_pickup_guard=1,
#     max_steps=400, pot_cook=60, pot_burn=90.
#   MAPPO uses SHAPED_REWARD_SCALE the way IPPO uses SHAPED_REWARD_COEFF
#     (reward + shaped*anneal*SCALE). CTC needs a strong dense signal, so SCALE=20
#     mirrors the proven IPPO CTC recipe (POT_START_COOKING 2.0 -> effective 40,
#     under the farm-trap threshold). Default SCALE=1.0 is ~20x too weak for CTC.
#   Network / PPO hyperparameters are the MAPPO full_obs config defaults.
# ONE GPU: launch only after the IPPO teacher retrain (jaxmarl_ippo_ctc_15m_qmixenv_
# 20260705) finishes. Runs are sequential.
set -euo pipefail

REPO=/student/brownd58/dev/JaxMARL
NAME=mappo_ctc_15m_qmixenv_20260705

tmux new-session -d -s jaxmarl_${NAME} "cd ${REPO} && \
export PATH=${REPO}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && \
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && \
PYTHONUNBUFFERED=1 ${REPO}/.venv/bin/python \
  ${REPO}/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py \
  --config-name=mappo_rnn_overcooked_v3_full_obs \
  WANDB_MODE=online PROJECT=ocv3_ctc_comparison ENTITY=zacharytang24- \
  WANDB_RUN_NAME=${NAME} USE_RICH_MONITOR=False \
  ENV_KWARGS.layout=coordinated_temporal_conveyor \
  ENV_KWARGS.agent_view_size=null \
  ENV_KWARGS.max_steps=400 \
  ENV_KWARGS.random_agent_positions=false \
  +ENV_KWARGS.pot_cook_time=60 \
  +ENV_KWARGS.pot_burn_time=90 \
  +ENV_KWARGS.enable_order_queue=true \
  +ENV_KWARGS.max_orders=5 \
  +ENV_KWARGS.order_generation_rate=1.0 \
  +ENV_KWARGS.order_expiration_time=0 \
  +ENV_KWARGS.order_queue_mode=alternating \
  +ENV_KWARGS.plate_pickup_guard=1 \
  +ENV_KWARGS.enable_item_conveyors=true \
  +ENV_KWARGS.enable_player_conveyors=false \
  TOTAL_TIMESTEPS=15000000 REW_SHAPING_HORIZON=15000000 SHAPED_REWARD_SCALE=20 \
  NUM_ENVS=256 NUM_STEPS=256 UPDATE_EPOCHS=4 NUM_MINIBATCHES=64 \
  LR=0.00025 ANNEAL_LR=true GAMMA=0.99 GAE_LAMBDA=0.95 CLIP_EPS=0.2 \
  ENT_COEF=0.04 VF_COEF=0.5 MAX_GRAD_NORM=0.5 \
  GRU_HIDDEN_DIM=128 FC_DIM_SIZE=64 ACTIVATION=relu \
  SEED=42 NUM_SEEDS=1 \
  WANDB_DIR=${REPO}/outputs/${NAME} \
  > ${REPO}/outputs/${NAME}_train.log 2>&1"

echo "launched tmux session jaxmarl_${NAME}; log -> outputs/${NAME}_train.log"
