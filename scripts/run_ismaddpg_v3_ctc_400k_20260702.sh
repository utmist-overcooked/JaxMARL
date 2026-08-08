#!/usr/bin/env bash
# IS-MADDPG on coordinated_temporal_conveyor (CTC), mirroring the QMIX CTC 3M runs.
#
# NOTE: unlike QMIX (Hydra, +alg.* keys), IS-MADDPG/run_overcooked_v3.py uses argparse.
# Only --layout/--total_timesteps/--num_envs/--max_steps/--seed/--save_path/--wandb/
# --wandb_entity are exposed; everything else (BUFFER_SIZE=200k, BATCH=512, ACTOR/CRITIC_LR,
# GAMMA, etc.) is hardcoded in make_overcooked_config(). Env kwargs (recipe_mode,
# max_orders, conveyor flags) are NOT passable on the CLI — OvercookedV3(layout=CTC) only
# auto-enables item conveyors from the layout; the order-queue/alternating config the QMIX
# scripts pass is left at env defaults here. Edit make_overcooked_config to change those.
#
# Memory: the flat replay buffer is 200k *transitions* (not episode sequences). CTC obs is
# (6,12,40)=2880 → ~4.6 GB of obs alone on the 16 GB GPU (the dominant consumer; num_envs
# does NOT change buffer size). If it OOMs, lower BUFFER_SIZE in make_overcooked_config.
# Training runs as a CHUNKED lax.scan (LOG_INTERVAL=10k env steps/chunk): each chunk
# compiles once then reused, and drops to Python to wandb.log + print and checkpoint every
# CKPT_INTERVAL=250k steps (+ a _final.pkl). BATCH_SIZE=1024. Plots/CSV saved at the end.
#   Smoke-tested on CPU (JAX_PLATFORMS=cpu): chunks + checkpoints + stitch all work.
set -euo pipefail

NAME=ismaddpg_ctc_400k_20260702
ROOT=/student/brownd58/dev/JaxMARL

tmux new-session -d -s jaxmarl_${NAME} "cd ${ROOT} && export PYTHONPATH=${ROOT}:\${PYTHONPATH:-} && export PATH=${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && mkdir -p ${ROOT}/outputs ${ROOT}/checkpoints && PYTHONUNBUFFERED=1 ${ROOT}/.venv/bin/python ${ROOT}/baselines/IS_MADDPG/run_overcooked_v3.py --layout coordinated_temporal_conveyor --total_timesteps 400000 --num_envs 8 --max_steps 400 --seed 42 --save_path ${ROOT}/checkpoints/${NAME} --wandb --wandb_entity dannyb3334-university-of-toronto > ${ROOT}/outputs/${NAME}_train.log 2>&1"
