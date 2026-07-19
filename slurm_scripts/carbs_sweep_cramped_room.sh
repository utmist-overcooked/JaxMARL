#!/bin/bash
#SBATCH --job-name=carbs-cramped-room
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=12:00:00
#SBATCH --account=rrg-cglee
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/carbs_sweep_cramped_room_%j.out

cd /project/rrg-cglee/zachtang/JaxMARL
source venv/bin/activate

export PYTHONUNBUFFERED=1

python baselines/IPPO/ippo_rnn_overcooked_v3.py \
    --config-name ippo_rnn_overcooked_v3 \
    TUNE=True \
    CARBS_NUM_TRIALS=20 \
    TOTAL_TIMESTEPS=3e7 \
    ENV_KWARGS.layout=cramped_room \
    ENV_KWARGS.agent_view_size=2 \
    WANDB_MODE=disabled \
    USE_RICH_MONITOR=False
