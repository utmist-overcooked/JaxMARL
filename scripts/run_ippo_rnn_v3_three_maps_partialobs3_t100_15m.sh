#!/usr/bin/env bash
set -euo pipefail

cd /student/brownd58/dev/JaxMARL

RUN_ID=$(date +"%Y%m%d_%H%M%S")
WANDB_PROJECT="overcookedv3_ippo_rnn_three_maps_partialobs3_T100_15m_${RUN_ID}"
LOG_DIR=/student/brownd58/dev/JaxMARL/outputs
CKPT_DIR=/student/brownd58/dev/JaxMARL/checkpoints

mkdir -p "$LOG_DIR" "$CKPT_DIR"

export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:$PATH
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9

for LAYOUT in middle_conveyor coordinated_temporal_conveyor maze_conveyor_hell; do
  SLUG="ippo_rnn_v3_${LAYOUT}_partialobs3_T100_15m_${RUN_ID}"
  echo "[$(date +"%F %T")] launching ${LAYOUT} -> ${SLUG}"

  .venv/bin/python baselines/IPPO/ippo_rnn_overcooked_v3.py \
    ENV_KWARGS.layout=${LAYOUT} \
    +ENV_KWARGS.agent_view_size=3 \
    ENV_KWARGS.max_steps=100 \
    ENV_KWARGS.pot_cook_time=20 \
    ENV_KWARGS.pot_burn_time=40 \
    ENV_KWARGS.enable_order_queue=true \
    ENV_KWARGS.max_orders=5 \
    ENV_KWARGS.order_generation_rate=0.1 \
    ENV_KWARGS.order_expiration_time=200 \
    ENV_KWARGS.recipe_mode=fixed \
    ENV_KWARGS.plate_pickup_guard=1 \
    ENV_KWARGS.enable_item_conveyors=true \
    ENV_KWARGS.enable_player_conveyors=false \
    TOTAL_TIMESTEPS=15000000 \
    NUM_ENVS=128 \
    NUM_STEPS=200 \
    UPDATE_EPOCHS=4 \
    NUM_MINIBATCHES=8 \
    LR=0.0005 \
    GAMMA=0.99 \
    GAE_LAMBDA=0.95 \
    CLIP_EPS=0.2 \
    VF_COEF=0.5 \
    ENT_COEF=0.1 \
    ENT_COEF_MIN=0.1 \
    ENTROPY_FLOOR=0.1 \
    ENTROPY_FLOOR_COEF=0.01 \
    MAX_GRAD_NORM=0.5 \
    GRU_HIDDEN_DIM=128 \
    FC_DIM_SIZE=128 \
    ACTIVATION=relu \
    ANNEAL_LR=true \
    REW_SHAPING_HORIZON=15000000 \
    REW_SHAPING_MIN_COEFF=0.10 \
    SHAPED_REWARD_COEFF=30.0 \
    WANDB_MODE=online \
    +WANDB_LOG_HISTORY_TABLE=true \
    ENTITY=zacharytang24- \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_NAME=ippo_rnn_overcooked_v3_${LAYOUT}_partialobs3_T100_${RUN_ID} \
    SAVE_PATH="$CKPT_DIR/$SLUG" \
    SAVE_GIF_PATH="$LOG_DIR/$SLUG.gif" \
    SEED=42 \
    > "$LOG_DIR/$SLUG.log" 2>&1

  echo "[$(date +"%F %T")] finished ${LAYOUT}"
done
