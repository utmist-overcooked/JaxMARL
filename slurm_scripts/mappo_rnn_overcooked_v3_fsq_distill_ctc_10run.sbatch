#!/bin/bash
#SBATCH -J fsq_ctc_10
#SBATCH -o /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%A_%a.out
#SBATCH -e /home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill/outputs/logs/%x_%A_%a.err
#SBATCH -p base_suma_rtx3090
#SBATCH -q base_qos
#SBATCH --array=0-9%10
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=3:00:00
#SBATCH --exclude=node19,node13,node16,node08,node10,node21,node14,node04,node05

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/overcooked-fsq-distill}"
CTC_PROJECT_DIR="${CTC_PROJECT_DIR:-/home/tangzach/JaxMARL/.worktrees/codex/ctc-mappo-fullobs}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/tangzach/jaxmarl/fsq_ctc_10run}"
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

WANDB_ENTITY="${WANDB_ENTITY:-zacharytang24-}"
WANDB_PROJECT="${WANDB_PROJECT:-overcookedv3-mappo-full-obs}"

LAYOUT="${LAYOUT:-coordinated_temporal_conveyor_harder}"
TOTAL_TIMESTEPS="${TOTAL_TIMESTEPS:-5000000}"
REW_SHAPING_HORIZON="${REW_SHAPING_HORIZON:-5000000}"
SHAPED_REWARD_SCALE="${SHAPED_REWARD_SCALE:-5.0}"
NUM_ENVS="${NUM_ENVS:-256}"
NUM_STEPS="${NUM_STEPS:-256}"
NUM_MINIBATCHES="${NUM_MINIBATCHES:-64}"
UPDATE_EPOCHS="${UPDATE_EPOCHS:-4}"
CHECKPOINT_GIF="${CHECKPOINT_GIF:-True}"
CHECKPOINT_GIF_COUNT="${CHECKPOINT_GIF_COUNT:-10}"
CHECKPOINT_GIF_MAX_STEPS="${CHECKPOINT_GIF_MAX_STEPS:-300}"
CHECKPOINT_GIF_EPSILON="${CHECKPOINT_GIF_EPSILON:-0}"
CHECKPOINT_GIF_SEED="${CHECKPOINT_GIF_SEED:-0}"
CHECKPOINT_GIF_FPS="${CHECKPOINT_GIF_FPS:-8}"
CHECKPOINT_GIF_TILE_SIZE="${CHECKPOINT_GIF_TILE_SIZE:-32}"

TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"

case "$TASK_ID" in
    0)
        VARIANT="baseline_seed0"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.30
        DISTILL_TEMPERATURE=1.0
        ;;
    1)
        VARIANT="baseline_seed1"
        SEED=1
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.30
        DISTILL_TEMPERATURE=1.0
        ;;
    2)
        VARIANT="stronger_distill"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=2.0
        DISTILL_DECAY_FRACTION=0.30
        DISTILL_TEMPERATURE=1.0
        ;;
    3)
        VARIANT="longer_distill"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.60
        DISTILL_TEMPERATURE=1.0
        ;;
    4)
        VARIANT="strong_long_distill"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=2.0
        DISTILL_DECAY_FRACTION=0.60
        DISTILL_TEMPERATURE=1.0
        ;;
    5)
        VARIANT="soft_teacher"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.60
        DISTILL_TEMPERATURE=2.0
        ;;
    6)
        VARIANT="tiny_channel"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[3,3,3]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.60
        DISTILL_TEMPERATURE=1.0
        ;;
    7)
        VARIANT="big_channel"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[7,7,7]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.60
        DISTILL_TEMPERATURE=1.0
        ;;
    8)
        VARIANT="larger_partial_view"
        SEED=0
        AGENT_VIEW_SIZE=3
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=1.0
        DISTILL_DECAY_FRACTION=0.60
        DISTILL_TEMPERATURE=1.0
        ;;
    9)
        VARIANT="no_distill_control"
        SEED=0
        AGENT_VIEW_SIZE=2
        FSQ_LEVELS="[5,5,5]"
        DISTILL_COEF=0.0
        DISTILL_DECAY_FRACTION=0.30
        DISTILL_TEMPERATURE=1.0
        ;;
    *)
        echo "No variant configured for SLURM_ARRAY_TASK_ID=$TASK_ID"
        exit 1
        ;;
esac

RUN_NAME="${WANDB_RUN_NAME_PREFIX:-fsq_ctc_harder}_${TASK_ID}_${VARIANT}_5m"
WANDB_DIR="${WANDB_DIR:-${OUTPUT_ROOT}/${RUN_NAME}_${SLURM_ARRAY_JOB_ID:-manual}_${TASK_ID}}"
RUN_DIR="${WANDB_DIR}/runs/${RUN_NAME}"
CHECKPOINT_GIF_OUTPUT_DIR="${CHECKPOINT_GIF_OUTPUT_DIR:-${WANDB_DIR}/checkpoint_rollouts}"

mkdir -p \
    "${PROJECT_DIR}/outputs/logs" \
    "${OUTPUT_ROOT}" \
    "${WANDB_DIR}/models" \
    "$CHECKPOINT_GIF_OUTPUT_DIR" \
    "$WANDB_CACHE_DIR" \
    "$WANDB_CONFIG_DIR"

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "task_id=$TASK_ID"
echo "variant=$VARIANT"
echo "project_dir=$PROJECT_DIR"
echo "ctc_project_dir=$CTC_PROJECT_DIR"
echo "python=$PYTHON_BIN"
echo "pythonpath=$PYTHONPATH"
echo "teacher_actor_path=$TEACHER_ACTOR_PATH"
echo "wandb_entity=$WANDB_ENTITY"
echo "wandb_project=$WANDB_PROJECT"
echo "wandb_run_name=$RUN_NAME"
echo "wandb_dir=$WANDB_DIR"
echo "layout=$LAYOUT"
echo "total_timesteps=$TOTAL_TIMESTEPS"
echo "num_envs=$NUM_ENVS"
echo "num_steps=$NUM_STEPS"
echo "num_minibatches=$NUM_MINIBATCHES"
echo "update_epochs=$UPDATE_EPOCHS"
echo "seed=$SEED"
echo "agent_view_size=$AGENT_VIEW_SIZE"
echo "fsq_levels=$FSQ_LEVELS"
echo "distill_coef=$DISTILL_COEF"
echo "distill_decay_fraction=$DISTILL_DECAY_FRACTION"
echo "distill_temperature=$DISTILL_TEMPERATURE"
echo "checkpoint_gif=$CHECKPOINT_GIF"
echo "checkpoint_gif_count=$CHECKPOINT_GIF_COUNT"
echo "checkpoint_gif_output_dir=$CHECKPOINT_GIF_OUTPUT_DIR"
echo "checkpoint_gif_max_steps=$CHECKPOINT_GIF_MAX_STEPS"
echo "checkpoint_gif_epsilon=$CHECKPOINT_GIF_EPSILON"
echo "checkpoint_gif_seed=$CHECKPOINT_GIF_SEED"
echo "checkpoint_gif_fps=$CHECKPOINT_GIF_FPS"
echo "checkpoint_gif_tile_size=$CHECKPOINT_GIF_TILE_SIZE"

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
