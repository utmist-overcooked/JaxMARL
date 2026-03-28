#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
  echo "usage: $0 <python_bin> <log_path> <command...>" >&2
  exit 1
fi

PY="$1"
LOG_PATH="$2"
shift 2

mkdir -p "$(dirname "$LOG_PATH")"

{
  echo "[$(date -Iseconds)] waiting for JAX GPU runtime"
  while true; do
    if JAX_PLATFORMS=cuda "$PY" -c 'import jax; assert jax.default_backend() == "gpu"; print(jax.devices())' >/tmp/jax_gpu_ready.$$ 2>&1; then
      echo "[$(date -Iseconds)] JAX GPU ready"
      cat /tmp/jax_gpu_ready.$$
      rm -f /tmp/jax_gpu_ready.$$
      break
    fi
    sleep 15
  done

  echo "[$(date -Iseconds)] starting command: $*"
  exec "$@"
} >> "$LOG_PATH" 2>&1