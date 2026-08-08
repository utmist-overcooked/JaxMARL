#!/usr/bin/env bash
# IPPO-RNN teacher on coordinated_temporal_conveyor, retrained in the SAME env as
# the QMIX teacher (qmix_ctc_15m_handoff_20260702) so the two teachers — and their
# FSQ-distilled students — are directly comparable. Only the env TIMINGS differ from
# run_ippo_rnn_v3_ctc_15m_20260703.sh: max_steps 800->400, pot_cook 20->60,
# pot_burn 40->90 (matching the QMIX run). IPPO training hyperparameters unchanged
# (proven around_the_island/CTC recipe: SHAPED_REWARD_COEFF=20.0). Same wandb project.
set -euo pipefail

tmux new-session -d -s jaxmarl_ippo_ctc_15m_qmixenv_20260705 "cd /student/brownd58/dev/JaxMARL && export PATH=/student/brownd58/dev/JaxMARL/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && PYTHONUNBUFFERED=1 /student/brownd58/dev/JaxMARL/.venv/bin/python /student/brownd58/dev/JaxMARL/baselines/IPPO/ippo_rnn_overcooked_v3.py --config-name=ippo_rnn_overcooked_v3 WANDB_MODE=online WANDB_PROJECT=ocv3_ctc_comparison WANDB_NAME=ippo_ctc_15m_qmixenv_20260705 ENTITY=zacharytang24- ENV_KWARGS.layout=coordinated_temporal_conveyor ENV_KWARGS.max_steps=400 ENV_KWARGS.pot_cook_time=60 ENV_KWARGS.pot_burn_time=90 ENV_KWARGS.enable_order_queue=true ENV_KWARGS.max_orders=5 ENV_KWARGS.order_generation_rate=1.0 ENV_KWARGS.order_expiration_time=0 ENV_KWARGS.recipe_mode=fixed ENV_KWARGS.plate_pickup_guard=1 ENV_KWARGS.enable_item_conveyors=true ENV_KWARGS.enable_player_conveyors=false TOTAL_TIMESTEPS=15000000 NUM_ENVS=128 NUM_STEPS=200 UPDATE_EPOCHS=4 NUM_MINIBATCHES=8 LR=0.0005 GAMMA=0.99 GAE_LAMBDA=0.95 CLIP_EPS=0.2 VF_COEF=0.5 ENT_COEF=0.1 ENT_COEF_MIN=0.01 ENTROPY_FLOOR=0.0 ENTROPY_FLOOR_COEF=0.0 MAX_GRAD_NORM=0.5 GRU_HIDDEN_DIM=128 FC_DIM_SIZE=128 ACTIVATION=relu ANNEAL_LR=true REW_SHAPING_HORIZON=15000000 REW_SHAPING_MIN_COEFF=0.0 SHAPED_REWARD_COEFF=20.0 SEED=42 SAVE_PATH=/student/brownd58/dev/JaxMARL/checkpoints/ippo_ctc_15m_qmixenv_20260705 SAVE_GIF_PATH=/student/brownd58/dev/JaxMARL/outputs/ippo_ctc_15m_qmixenv_20260705.gif LOG_EVERY=1 CHECKPOINT_EVERY=5000 > /student/brownd58/dev/JaxMARL/outputs/ippo_ctc_15m_qmixenv_20260705_train.log 2>&1"

echo "launched tmux session jaxmarl_ippo_ctc_15m_qmixenv_20260705; log -> outputs/ippo_ctc_15m_qmixenv_20260705_train.log"
