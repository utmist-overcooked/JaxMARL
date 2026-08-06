#!/usr/bin/env bash
# IS-MADDPG on coordinated_temporal_conveyor (CTC), mirroring the QMIX CTC 15M
# handoff recipe (scripts/run_qmix_rnn_v3_ctc_15m_handoff_20260702.sh) so a
# QMIX-vs-MADDPG gap on CTC points at MADDPG itself, not a hyperparameter mismatch.
#
# WHAT IS MIRRORED (edited into make_overcooked_config in run_overcooked_v3.py):
#   HIDDEN_DIM=256 (QMIX HIDDEN_SIZE=256), ACTOR_LR=CRITIC_LR=5e-5 (QMIX LR=5e-5),
#   GRAD_CLIP=10 (QMIX MAX_GRAD_NORM=10), LEARNING_STARTS=10000, GAMMA=0.99,
#   EPSILON_END=0.05 (EPS_FINISH), EPSILON_DECAY=0.2*total (EPS_DECAY=0.2).
#   Reward shaping is ALREADY identical: both read the global settings.py
#   SHAPED_REWARDS; QMIX uses SHAPED_REWARD_COEFF=1.0 / no-anneal / REW_SCALE=1.0,
#   which equals IS-MADDPG's `reward = raw + shaped`. Env kwargs (pot 60/90,
#   alternating order queue, conveyors, plate_pickup_guard=1) also match — set in main().
#
# NOT MIRRORED (architectural — can't map cleanly, left at MADDPG defaults):
#   * Target update: QMIX HARD copy every 10 rollouts (TAU=1.0) vs MADDPG soft
#     TAU=0.01 every env step.
#   * Update cadence / NUM_EPOCHS: MADDPG trains EVERY env step (2 grad epochs)
#     vs QMIX once per 400-step rollout (8 epochs) — mirroring NUM_EPOCHS=8 would
#     quadruple an already-much-larger gradient budget.
#   * Buffer: 200k flat transitions (memory-bound, ~4.6GB obs on 16GB GPU) vs
#     QMIX 512 episode sequences.
#
# GIF: --save_gif_path enables a greedy rollout rendered every 1/10th of training
#   (+ a final one), filenames get _step<N> spliced in. Watch these to see whether
#   the policy cook-burn-farms or never crosses the conveyor handoff.
set -euo pipefail

NAME=ismaddpg_ctc_15m_mirror_20260705
ROOT=/student/brownd58/dev/JaxMARL

tmux new-session -d -s jaxmarl_${NAME} "cd ${ROOT} && export PYTHONPATH=${ROOT}:\${PYTHONPATH:-} && export PATH=${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && mkdir -p ${ROOT}/outputs ${ROOT}/checkpoints && PYTHONUNBUFFERED=1 ${ROOT}/.venv/bin/python ${ROOT}/baselines/IS_MADDPG/run_overcooked_v3.py --layout coordinated_temporal_conveyor --total_timesteps 15000000 --num_envs 4 --max_steps 400 --seed 42 --save_path ${ROOT}/checkpoints/${NAME} --save_gif_path ${ROOT}/outputs/${NAME}.gif --wandb --wandb_entity dannyb3334-university-of-toronto > ${ROOT}/outputs/${NAME}_train.log 2>&1"
