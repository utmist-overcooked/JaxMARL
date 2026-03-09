# IC3Net Predator-Prey (Hard) Setup Runbook

This runbook documents the exact setup and launch steps used on Linux to run IC3Net training with the `ic3net_pp_hard` config on branch `ic3net_and_family`.

## 1) Go to repo and confirm branch

```bash
cd /home/barracuda/Desktop/test/JaxMARL
git checkout ic3net_and_family
```

## 2) Use system Python 3.10

Commands below use:

```bash
/usr/bin/python3
```

## 3) Install algorithm dependencies

Documented command:

```bash
/usr/bin/python3 -m pip install '.[algs]'
```

On this branch, pip may show packaging quirks (e.g., `UNKNOWN` metadata). If that happens or training later fails on missing modules, install required packages explicitly.

## 4) Install explicit runtime dependencies (resolved in-session)

```bash
/usr/bin/python3 -m pip install jax==0.4.38 jaxlib==0.4.38
/usr/bin/python3 -m pip install optax distrax flax hydra-core omegaconf wandb rich gymnax tqdm matplotlib pillow
/usr/bin/python3 -m pip install brax==0.10.3 mujoco==3.1.3 pygame safetensors
```

Notes:
- `brax` is required because `jaxmarl` imports environment modules transitively during startup.
- pip may upgrade `jax`/`jaxlib` while resolving dependencies; this was acceptable for launching training in this session.

## 5) Launch IC3Net PP-hard training

```bash
cd /home/barracuda/Desktop/test/JaxMARL
/usr/bin/python3 baselines/IC3Net/ic3net_train.py --config-name=ic3net_pp_hard
```

## 6) Expected startup output

You should see logs similar to:

- `Starting REINFORCE training: IC3NET | ...`
- `JIT-compiling update step (first call will be slow)...`
- Live monitor with metrics (`Update`, `EpRet`, `Loss`, `Pi Loss`, `V Loss`, `Entropy`)

## 7) Errors seen and fixes

- `ModuleNotFoundError: No module named 'jax'`
  - Fix: install `jax` + `jaxlib` into `/usr/bin/python3`.
- `ModuleNotFoundError: No module named 'optax'`
  - Fix: install algorithm stack (`optax`, `distrax`, `flax`, `hydra-core`, `omegaconf`, etc.).
- `ModuleNotFoundError: No module named 'brax'`
  - Fix: install `brax==0.10.3` and `mujoco==3.1.3`.

## 8) Config used

Config file:

- `baselines/IC3Net/config/ic3net_pp_hard.yaml`

Run override used:

```bash
--config-name=ic3net_pp_hard
```

## 9) This PC (workspace `/workspace/JaxMARL`) — what actually worked

The commands above were originally written for a different machine (`/home/barracuda/...`, Python 3.10).
On this PC, the working setup was:

- Repo path: `/workspace/JaxMARL`
- Branch: `ic3net_and_family`
- Python: `/usr/bin/python3` -> **Python 3.12.3**
- GPU: NVIDIA present (`nvidia-smi` works)

### 9.1 Branch + interpreter verification

```bash
git -C /workspace/JaxMARL checkout -B ic3net_and_family origin/ic3net_and_family
git -C /workspace/JaxMARL branch --show-current
/usr/bin/python3 --version
```

### 9.2 Dependency installation sequence that worked here

```bash
/usr/bin/python3 -m pip install '/workspace/JaxMARL[algs]'
/usr/bin/python3 -m pip install jax==0.4.38 jaxlib==0.4.38
/usr/bin/python3 -m pip install optax distrax flax hydra-core omegaconf wandb rich gymnax tqdm matplotlib pillow
/usr/bin/python3 -m pip install brax==0.10.3 mujoco==3.1.3 pygame safetensors
```

When `brax` install hit a Debian package conflict (`blinker` uninstall issue), this resolved it:

```bash
/usr/bin/python3 -m pip install --ignore-installed blinker==1.9.0
/usr/bin/python3 -m pip install --no-deps brax==0.10.3 mujoco==3.1.3
/usr/bin/python3 -m pip install dm-env flask flask-cors jaxopt ml-collections tensorboardX trimesh gym glfw pyopengl pytinyrenderer
```

### 9.3 CUDA/JAX fix on this machine

On this PC, CPU-only JAX initially loaded. CUDA required the plugin + CUDA wheel stack:

```bash
/usr/bin/python3 -m pip install -U jax-cuda12-plugin jax-cuda12-pjrt
/usr/bin/python3 -m pip install -U "jax[cuda12]==0.9.1"
```

Verification command:

```bash
JAX_PLATFORMS='' /usr/bin/python3 -c "import jax; print(jax.default_backend()); print(jax.devices())"
```

Expected on this PC:

- `gpu`
- `[CudaDevice(id=0)]`

### 9.4 Launch command that worked on this PC

```bash
JAX_PLATFORMS=cuda WANDB_MODE=online /usr/bin/python3 /workspace/JaxMARL/baselines/IC3Net/ic3net_train.py --config-name=ic3net_pp_hard WANDB_MODE=online
```

If W&B prompts for account selection/login, complete that interactively once per environment.

### 9.5 Smoke-test checkpoint save (validated)

The following smoke run successfully saved checkpoints under `/workspace/JaxMARL/checkpoints`:

```bash
JAX_PLATFORMS=cuda WANDB_MODE=disabled /usr/bin/python3 /workspace/JaxMARL/baselines/IC3Net/ic3net_train.py \
  --config-name=ic3net_pp_hard \
  WANDB_MODE=disabled \
  SAVE_PATH=/workspace/JaxMARL/checkpoints \
  +CHECKPOINT_EVERY=1 \
  LOG_EVERY=1 \
  NUM_ENVS=1 \
  NUM_STEPS=8 \
  TOTAL_TIMESTEPS=8 \
  ENV_KWARGS.max_steps=8
```

Artifacts confirmed:

- `/workspace/JaxMARL/checkpoints/model.msgpack`
- `/workspace/JaxMARL/checkpoints/model_update_1.msgpack`
- `/workspace/JaxMARL/checkpoints/run_config.json`
