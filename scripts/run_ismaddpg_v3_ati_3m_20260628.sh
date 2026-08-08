#!/usr/bin/env bash
# IS-MADDPG on around_the_island (ATI) — the solvable single-region map (IPPO solves it).
#
# argparse interface (NOT Hydra): only --layout/--total_timesteps/--num_envs/--max_steps/
# --seed/--save_path/--wandb/--wandb_entity. Env config is set in run_overcooked_v3.py's
# main(): around_the_island gets the same alternating order-queue + guard=1 the proven
# IPPO/QMIX recipes use, with conveyor flags auto-detected from the layout (ATI is not a
# conveyor map). Hyperparams (BUFFER_SIZE=200k, BATCH=1024, LR, GAMMA) hardcoded in
# make_overcooked_config; edit there to change.
#
# Memory: 200k-transition flat buffer; ATI obs (7,10,40)=2800 → ~4.5 GB obs (~14.7 GB total
# on the 16 GB GPU). num_envs does NOT change buffer size; if OOM lower BUFFER_SIZE.
# Training is a CHUNKED lax.scan: per-chunk (10k env steps) wandb.log + print, checkpoint
# every 250k steps + _final.pkl. Plots/CSV at end.  CPU-smoke-verified.
set -euo pipefail

NAME=ismaddpg_ati_3m_20260628
ROOT=/student/brownd58/dev/JaxMARL

tmux new-session -d -s jaxmarl_${NAME} "cd ${ROOT} && export PYTHONPATH=${ROOT}:\${PYTHONPATH:-} && export PATH=${ROOT}/.venv/lib/python3.10/site-packages/nvidia/cuda_nvcc/bin:\$PATH && export XLA_PYTHON_CLIENT_MEM_FRACTION=0.99 && mkdir -p ${ROOT}/outputs ${ROOT}/checkpoints && PYTHONUNBUFFERED=1 ${ROOT}/.venv/bin/python ${ROOT}/baselines/IS_MADDPG/run_overcooked_v3.py --layout around_the_island --total_timesteps 3000000 --num_envs 8 --max_steps 400 --seed 42 --save_path ${ROOT}/checkpoints/${NAME} --wandb --wandb_entity zacharytang24- > ${ROOT}/outputs/${NAME}_train.log 2>&1"
