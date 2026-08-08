#!/usr/bin/env bash
# IPPO-RNN on prep_kitchen_handoff: all three prep chains (cutting board, grill,
# blender) on one side of a full counter wall, pot/plates/delivery on the other,
# so processed ingredients must be handed over the middle counter.
#   env: no order queue (recipe indicator samples from [[5,5,5],[6,6,6],[7,7,7]]),
#       no conveyors, max_steps=400, cook=20, burn=40, default prep timings
#       (chop_stages=3, grill 15+30, blend 10), full obs (74 channels).
#   hyperparams: the proven CTC handoff recipe - SHAPED_REWARD_COEFF=20, MIN_COEFF=0.1,
#       ENT_COEF=0.1 -> 0.01 (handoff maps need the extra exploration), otherwise the
#       working around_the_island IPPO settings.
set -euo pipefail

tmux new-session -d -s jaxmarl_ippo_prep_kitchen_handoff_15m_20260726 "cd /student/brownd58/dev/JaxMARL && export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && PYTHONUNBUFFERED=1 /student/brownd58/dev/JaxMARL/.venv/bin/python /student/brownd58/dev/JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v3.py --config-name=ippo_rnn_overcooked_v3 WANDB_MODE=online WANDB_PROJECT=ocv3_prep_stations WANDB_NAME=ippo_prep_kitchen_handoff_15m_20260726 ENTITY=zacharytang24- ENV_KWARGS.layout=prep_kitchen_handoff ENV_KWARGS.max_steps=400 ENV_KWARGS.pot_cook_time=20 ENV_KWARGS.pot_burn_time=40 ENV_KWARGS.enable_order_queue=false ENV_KWARGS.enable_item_conveyors=false ENV_KWARGS.enable_player_conveyors=false TOTAL_TIMESTEPS=15000000 NUM_ENVS=128 NUM_STEPS=200 UPDATE_EPOCHS=4 NUM_MINIBATCHES=8 LR=0.0005 GAMMA=0.99 GAE_LAMBDA=0.95 CLIP_EPS=0.2 VF_COEF=0.5 ENT_COEF=0.1 ENT_COEF_MIN=0.01 ENTROPY_FLOOR=0.0 ENTROPY_FLOOR_COEF=0.0 MAX_GRAD_NORM=0.5 GRU_HIDDEN_DIM=128 FC_DIM_SIZE=128 ACTIVATION=relu ANNEAL_LR=true REW_SHAPING_HORIZON=15000000 REW_SHAPING_MIN_COEFF=0.1 SHAPED_REWARD_COEFF=20.0 SEED=42 SAVE_PATH=/student/brownd58/dev/JaxMARL/checkpoints/ippo_prep_kitchen_handoff_15m_20260726 SAVE_GIF_PATH=/student/brownd58/dev/JaxMARL/outputs/ippo_prep_kitchen_handoff_15m_20260726.gif LOG_EVERY=1 CHECKPOINT_EVERY=5000 > /student/brownd58/dev/JaxMARL/outputs/ippo_prep_kitchen_handoff_15m_20260726_train.log 2>&1"
