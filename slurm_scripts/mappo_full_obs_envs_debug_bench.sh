#!/bin/bash
set -u

PROJECT_DIR=$HOME/links/projects/rrg-cglee/zachtang/JaxMARL
BENCH_DIR=${FULL_OBS_BENCH_DIR:-$SCRATCH/jaxmarl/full_obs_env_bench_debug}

if [[ ! -d "$BENCH_DIR" ]]; then
    echo "Missing benchmark directory: $BENCH_DIR"
    echo "Create it on the login node before launching the debug job."
    exit 1
fi

source ${PROJECT_DIR}/venv/bin/activate

cd "$SCRATCH"

export PYTHONUNBUFFERED=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export WANDB_MODE=disabled

LAYOUTS=("cramped_room" "asymm_advantages_recipes_right")
ENV_COUNTS=(512 1024 1536 2048 3072 4096 6144 8192)

echo "Benchmark dir: $BENCH_DIR"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free --format=csv,noheader

run_case() {
    local layout="$1"
    local envs="$2"
    local log="$BENCH_DIR/${layout}_${envs}.log"
    local max_mem_file="$BENCH_DIR/${layout}_${envs}.max_mem"

    echo
    echo "=== layout=${layout} NUM_ENVS=${envs} ==="
    rm -f "$max_mem_file"

    (
        max_mem=0
        while true; do
            used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')
            if [[ "$used" =~ ^[0-9]+$ && "$used" -gt "$max_mem" ]]; then
                max_mem="$used"
                echo "$max_mem" > "$max_mem_file"
            fi
            sleep 2
        done
    ) &
    local sampler_pid=$!

    BENCH_PROJECT_DIR="$PROJECT_DIR" \
    BENCH_DIR="$BENCH_DIR" \
    BENCH_LAYOUT="$layout" \
    BENCH_NUM_ENVS="$envs" \
    JAX_PLATFORMS=cuda,cpu timeout 35m python - <<'PY' >"$log" 2>&1
import importlib.util
import os
from pathlib import Path

import jax
from omegaconf import OmegaConf

project_dir = Path(os.environ["BENCH_PROJECT_DIR"])
bench_dir = os.environ["BENCH_DIR"]
layout = os.environ["BENCH_LAYOUT"]
num_envs = int(os.environ["BENCH_NUM_ENVS"])

spec = importlib.util.spec_from_file_location(
    "mappo_full_obs",
    project_dir / "baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py",
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

config = OmegaConf.load(
    project_dir / "baselines/MAPPO/config/mappo_rnn_overcooked_v3_full_obs.yaml"
)
config = OmegaConf.to_container(config, resolve=False)
config["ENV_KWARGS"]["layout"] = layout
config["NUM_ENVS"] = num_envs
config["NUM_STEPS"] = 256
config["NUM_MINIBATCHES"] = 64
config["TOTAL_TIMESTEPS"] = num_envs * config["NUM_STEPS"]
config["LR"] = 0.002
config["WANDB_MODE"] = "disabled"
config["ENTITY"] = None
config["PROJECT"] = None
config["WANDB_DIR"] = bench_dir
config["WANDB_RUN_NAME"] = f"debug_full_obs_{layout}_{num_envs}"
config["USE_RICH_MONITOR"] = False
config["DISABLE_CHECKPOINTS"] = True

train_fn = module.make_train(config, monitor=None)
out = jax.block_until_ready(jax.jit(train_fn)(jax.random.PRNGKey(config["SEED"])))
ret = float(out["metrics"]["returned_episode_returns"][-1])
print(f"completed layout={layout} num_envs={num_envs} final_return={ret:.3f}")
PY
    local status=$?

    kill "$sampler_pid" >/dev/null 2>&1 || true
    wait "$sampler_pid" >/dev/null 2>&1 || true

    local max_mem="unknown"
    if [[ -f "$max_mem_file" ]]; then
        max_mem=$(cat "$max_mem_file")
    fi

    if [[ "$status" -eq 0 ]]; then
        echo "PASS layout=${layout} NUM_ENVS=${envs} max_gpu_mem_mib=${max_mem}"
    else
        echo "FAIL layout=${layout} NUM_ENVS=${envs} status=${status} max_gpu_mem_mib=${max_mem}"
        tail -n 40 "$log"
        return 1
    fi
}

for layout in "${LAYOUTS[@]}"; do
    best=0
    for envs in "${ENV_COUNTS[@]}"; do
        if run_case "$layout" "$envs"; then
            best="$envs"
        else
            break
        fi
    done
    echo "BEST layout=${layout} NUM_ENVS=${best}"
done

echo
echo "Benchmark complete. Logs: $BENCH_DIR"
