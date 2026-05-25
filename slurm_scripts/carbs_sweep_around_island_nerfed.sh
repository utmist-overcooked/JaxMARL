#!/bin/bash
#SBATCH --job-name=carbs-around-nerfed
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=6:00:00
#SBATCH --account=rrg-cglee
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/carbs_sweep_around_nerfed_%j.out

cd /project/rrg-cglee/zachtang/JaxMARL
source venv/bin/activate

export PYTHONUNBUFFERED=1

python baselines/IPPO/ippo_rnn_overcooked_v3.py \
    --config-name ippo_rnn_overcooked_v3 \
    TUNE=True \
    CARBS_NUM_TRIALS=20 \
    TOTAL_TIMESTEPS=5e6 \
    REW_SHAPING_HORIZON=2.5e6 \
    ENV_KWARGS.layout=around_the_island_nerfed \
    ENV_KWARGS.agent_view_size=null \
    WANDB_MODE=disabled \
    USE_RICH_MONITOR=False
