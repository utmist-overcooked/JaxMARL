#!/bin/bash
#SBATCH --job-name=carbs-sweep-test
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=1:00:00
#SBATCH --account=rrg-cglee
#SBATCH --output=/scratch/zachtang/jaxmarl/logs/carbs_sweep_test_%j.out

# Pre-create directories (no mkdir available on compute nodes)
# Run `mkdir -p /scratch/zachtang/jaxmarl/carbs_sweep/checkpoints` on login node first.

cd /project/rrg-cglee/zachtang/JaxMARL
source venv/bin/activate

python baselines/IPPO/ippo_rnn_overcooked_v3.py \
    --config-name ippo_rnn_overcooked_v3 \
    TUNE=True \
    CARBS_NUM_TRIALS=3 \
    TOTAL_TIMESTEPS=1e6 \
    NUM_ENVS=64 \
    ENV_KWARGS.layout=cramped_room \
    ENV_KWARGS.agent_view_size=2 \
    WANDB_MODE=disabled \
    USE_RICH_MONITOR=False
