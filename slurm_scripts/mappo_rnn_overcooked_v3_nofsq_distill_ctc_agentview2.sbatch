#!/bin/bash
#SBATCH -J nofsq_ctc
#SBATCH -o /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%j.out
#SBATCH -e /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%j.err
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill}"
CTC_PROJECT_DIR="${CTC_PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/ctc-mappo-fullobs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/tangzach/jaxmarl/nofsq_ctc_harder_agentview2}"
TEACHER_ACTOR_PATH="${TEACHER_ACTOR_PATH:-${CTC_PROJECT_DIR}/outputs/overcookedv3-mappo-full-obs/models/ctc_harder_exp3_late_entropy_anneal_5m_20260607/70_actor.safetensors}"

cd "$PROJECT_DIR"

# shellcheck disable=SC1091
source "${PROJECT_DIR}/venv/bin/activate"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "${PROJECT_DIR}/venv/bin/python" ]]; then
    PYTHON_BIN="${PROJECT_DIR}/venv/bin/python"
else
    PYTHON_BIN="python"
fi

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE="${XLA_PYTHON_CLIENT_PREALLOCATE:-false}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cuda,cpu}"
export PYTHONPATH="${CTC_PROJECT_DIR}:${PROJECT_DIR}${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-${OUTPUT_ROOT}/wandb-cache}"
export WANDB_CONFIG_DIR="${WANDB_CONFIG_DIR:-${OUTPUT_ROOT}/wandb-config}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-${OUTPUT_ROOT}/matplotlib}"

WANDB_ENTITY="${WANDB_ENTITY:-zacharytang24-}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-zac-distillation}"

LAYOUT="${LAYOUT:-coordinated_temporal_conveyor_harder}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
SHAPED_REWARD_SCALE="${SHAPED_REWARD_SCALE:-5.0}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-64}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
SEED="${SEED:-0}"
AGENT_VIEW_SIZE="${AGENT_VIEW_SIZE:-2}"
DISTILL_COEF="${DISTILL_COEF:-2.0}"
DISTILL_DECAY_FRACTION="${DISTILL_DECAY_FRACTION:-0.60}"
DISTILL_TEMPERATURE="${DISTILL_TEMPERATURE:-1.0}"
FSQ_LEVELS="${FSQ_LEVELS:-[5,5,5]}"
CHECKPOINT_GIF="${CHECKPOINT_GIF:-True}"
CHECKPOINT_GIF_COUNT="${CHECKPOINT_GIF_COUNT:-10}"
CHECKPOINT_GIF_MAX_STEPS="${CHECKPOINT_GIF_MAX_STEPS:-150}"
CHECKPOINT_GIF_EPSILON="${CHECKPOINT_GIF_EPSILON:-0}"
CHECKPOINT_GIF_SEED="${CHECKPOINT_GIF_SEED:-0}"
CHECKPOINT_GIF_FPS="${CHECKPOINT_GIF_FPS:-8}"
CHECKPOINT_GIF_TILE_SIZE="${CHECKPOINT_GIF_TILE_SIZE:-32}"

RUN_NAME="${WANDB_RUN_NAME:-nofsq_ctc_harder_agentview2_strong_long_seed${SEED}_5m}"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${SLURM_JOB_ID:-manual}}"
RUN_DIR="${WANDB_DIR}/runs/${RUN_NAME}"
CHECKPOINT_GIF_OUTPUT_DIR="${CHECKPOINT_GIF_OUTPUT_DIR:-${WANDB_DIR}/checkpoint_rollouts}"

mkdir -p \
    "${PROJECT_DIR}/outputs/logs" \
    "${OUTPUT_ROOT}" \
    "${WANDB_DIR}/models" \
    "$CHECKPOINT_GIF_OUTPUT_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR" \
    "$MPLCONFIGDIR"

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "project_dir=$PROJECT_DIR"
echo "ctc_project_dir=$CTC_PROJECT_DIR"
echo "teacher_actor_path=$TEACHER_ACTOR_PATH"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_run_name=$RUN_NAME"
echo "wandb_dir=$WANDB_DIR"
echo "layout=$LAYOUT"
echo "total_timesteps=$TOTAL_TIMESTEPS"
echo "seed=$SEED"
echo "agent_view_size=$AGENT_VIEW_SIZE"
echo "disable_fsq_comm=True"
echo "distill_coef=$DISTILL_COEF"
echo "distill_decay_fraction=$DISTILL_DECAY_FRACTION"
echo "distill_temperature=$DISTILL_TEMPERATURE"
echo "checkpoint_gif=$CHECKPOINT_GIF"
echo "checkpoint_gif_count=$CHECKPOINT_GIF_COUNT"

if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader || true
else
    echo "nvidia-smi not found"
fi

env -u LD_LIBRARY_PATH "$PYTHON_BIN" -u "${PROJECT_DIR}/baselines/MAPPO/mappo_rnn_overcooked_v3_fsq_distill.py" \
    hydra.run.dir="$RUN_DIR" \
    hydra.output_subdir=null \
    ENV_KWARGS.layout="$LAYOUT" \
    ENV_KWARGS.agent_view_size="$AGENT_VIEW_SIZE" \
    ENV_KWARGS.random_agent_positions=False \
    "++ENV_KWARGS.delivery_reward=100.0" \
    "++ENV_KWARGS.wrong_delivery_reward=20.0" \
    "++ENV_KWARGS.enforce_handoff_roles=True" \
    "++ENV_KWARGS.enable_item_conveyors=True" \
    "++ENV_KWARGS.enable_player_conveyors=True" \
    TEACHER_ACTOR_PATH="$TEACHER_ACTOR_PATH" \
    DISABLE_FSQ_COMM=True \
    "FSQ_LEVELS=$FSQ_LEVELS" \
    DISTILL_COEF="$DISTILL_COEF" \
    DISTILL_DECAY_FRACTION="$DISTILL_DECAY_FRACTION" \
    DISTILL_TEMPERATURE="$DISTILL_TEMPERATURE" \
    TOTAL_TIMESTEPS="$TOTAL_TIMESTEPS" \
    REW_SHAPING_HORIZON="$REW_SHAPING_HORIZON" \
    SHAPED_REWARD_SCALE="$SHAPED_REWARD_SCALE" \
    NUM_ENVS="$NUM_ENVS" \
    NUM_STEPS="$NUM_STEPS" \
    NUM_MINIBATCHES="$NUM_MINIBATCHES" \
    UPDATE_EPOCHS="$UPDATE_EPOCHS" \
    SEED="$SEED" \
    WANDB_MODE=online \
    ENTITY="$WANDB_ENTITY" \
    PROJECT="$WANDB_PROJECT" \
    WANDB_DIR="$WANDB_DIR" \
    WANDB_RUN_NAME="$RUN_NAME" \
    USE_RICH_MONITOR=False \
    "++DISABLE_CHECKPOINTS=False" \
    CHECKPOINT_GIF="$CHECKPOINT_GIF" \
    CHECKPOINT_GIF_COUNT="$CHECKPOINT_GIF_COUNT" \
    CHECKPOINT_GIF_OUTPUT_DIR="$CHECKPOINT_GIF_OUTPUT_DIR" \
    CHECKPOINT_GIF_MAX_STEPS="$CHECKPOINT_GIF_MAX_STEPS" \
    CHECKPOINT_GIF_EPSILON="$CHECKPOINT_GIF_EPSILON" \
    CHECKPOINT_GIF_SEED="$CHECKPOINT_GIF_SEED" \
    CHECKPOINT_GIF_FPS="$CHECKPOINT_GIF_FPS" \
    CHECKPOINT_GIF_TILE_SIZE="$CHECKPOINT_GIF_TILE_SIZE"

echo "END: $(date)"
