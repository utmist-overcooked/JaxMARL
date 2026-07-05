# QMIX on PyMARL2 + SMACv2: Windows Setup and AI-Agent Handoff

This repository is a QMIX implementation built on PyMARL2 and SMACv2. This
guide documents a working Windows setup for training, evaluating, and
replaying its QMIX agents through Docker Desktop. It also records the
compatibility issues an AI agent should check when reproducing the
environment.

## Working architecture

- Windows hosts the repository, Docker Desktop, and W&B credentials.
- A Linux Docker container runs PyTorch, PyMARL2, SMACv2, and StarCraft II.
- The Linux StarCraft II 4.10 client is stored in
  `3rdparty/StarCraftII`.
- Experiment output is written through the bind mount to `results/`.
- The Windows StarCraft II installation is not used for training. Linux
  containers cannot execute the Windows game binaries.

The local Python virtual environment is optional and is not used for the
Docker training jobs.

Running `python src/main.py` with the system Python is not equivalent to
running the configured QMIX environment. For example, a clean Windows Python
installation may fail immediately with `ModuleNotFoundError: sacred`, and
newer Python, PyTorch, or SMACv2 versions may be incompatible with saved
models. Use the repository's Docker image, which provides the tested Python
3.8 and PyTorch 1.11 stack.

## Prerequisites

Install:

1. Git for Windows, including Git Bash.
2. Docker Desktop configured to use Linux containers.
3. At least 16 GB of available memory for the default QMIX configuration.
4. Enough disk space for the Docker image, the approximately 4 GB SC2
   download, its extracted files, checkpoints, and logs.
5. A Weights & Biases account and API key.

An NVIDIA GPU and NVIDIA Container Toolkit support are optional. On a machine
with only Intel or AMD integrated graphics, use CPU mode.

Check the environment from PowerShell:

```powershell
docker version
docker info
git --version
Get-CimInstance Win32_VideoController | Select-Object Name, DriverVersion
```

Start Docker Desktop before proceeding if `docker info` cannot connect to the
Linux engine.

## Repository setup

Clone the repository if needed:

```bash
git clone https://github.com/benellis3/pymarl2.git
cd pymarl2
```

This working copy contains Windows compatibility changes in:

- `docker/Dockerfile`
- `install_dependencies.sh`
- `install_sc2.sh`
- `run_docker.sh`
- `run_exp.sh`
- `src/config/default.yaml`

Important changes compared with the original upstream scripts include:

- The SC2 installer resolves the repository from its own location instead of
  assuming `~/pymarl`.
- Installation is idempotent and checks for
  `Versions/Base75689/SC2_x64`, not merely an existing directory.
- Docker handles CRLF line endings in `install_dependencies.sh`.
- Dependency installation stops on failure.
- W&B is pinned to `0.21.4`, which has a Python 3.8-compatible Linux wheel.
- The Docker bind mount is based on the script location and disables Git
  Bash/MSYS path conversion.
- The container working directory is explicitly `/source`.
- CPU execution is the default, while NVIDIA GPU IDs remain supported.
- W&B uses the authenticated account's default entity.

## Install Linux StarCraft II and SMACv2 maps

Run from Git Bash:

```bash
./install_sc2.sh
```

This downloads the headless Linux SC2 4.10 client and installs the SMACv2 map
pack. The archive is approximately 4 GB. Extraction can be slow when the
repository is inside OneDrive because SC2 contains many small cache files.

Verify the required assets:

```bash
test -x 3rdparty/StarCraftII/Versions/Base75689/SC2_x64
test -f 3rdparty/StarCraftII/Maps/SMAC_Maps/32x32_flat.SC2Map
```

Do not substitute an existing Windows SC2 installation for these files.

## Build the Docker image

From the repository root:

```bash
docker build -t pymarl2:ben_smac \
  -f docker/Dockerfile \
  --build-arg UID=1000 .
```

Verify the Python stack and mounted SC2 assets:

```bash
REPO_DIR="$(pwd -W)"
MSYS_NO_PATHCONV=1 docker run --rm \
  --mount "type=bind,source=$REPO_DIR,target=/source" \
  --workdir /source \
  pymarl2:ben_smac \
  python3 -c "import torch, smacv2, wandb; print(torch.__version__, wandb.__version__)"
```

Expected core versions are PyTorch `1.11.0` and W&B `0.21.4`.

## Configure W&B securely

Never put the API key in the repository or in this Markdown file.

The launcher reads the key from the file named by
`WANDB_API_KEY_FILE`. A suitable persistent Windows user environment value is:

```text
/c/Users/<windows-user>/.config/wandb/api-key
```

Set the persistent variable from PowerShell:

```powershell
$value = "/c/Users/$env:USERNAME/.config/wandb/api-key"
[Environment]::SetEnvironmentVariable("WANDB_API_KEY_FILE", $value, "User")
```

Open a new Git Bash window, then create the key file without putting the key
in shell history:

```bash
mkdir -p ~/.config/wandb
read -rsp "W&B API key: " WANDB_KEY
printf '%s' "$WANDB_KEY" > ~/.config/wandb/api-key
unset WANDB_KEY
echo
```

The default W&B project is `smacv2-qmix`. Change `project` in
`src/config/default.yaml` if desired. `entity: null` tells W&B to use the
default entity associated with the API key.

## Optional local virtual environment

The tested local environment uses Python 3.10:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

The directory is ignored by Git. Use Docker for actual SMACv2 training because
the configured SC2 runtime and legacy PyTorch stack are Linux-based.

## Run a smoke test

Before launching a baseline-sized job, run one short CPU episode:

```bash
./run_docker.sh cpu \
  python3 src/main.py \
  --config=qmix \
  --env-config=sc2_gen_protoss \
  with \
  env_args.capability_config.n_units=2 \
  env_args.capability_config.n_enemies=2 \
  batch_size_run=1 \
  batch_size=1 \
  buffer_size=2 \
  use_wandb=False \
  use_cuda=False \
  t_max=1 \
  test_nepisode=1 \
  save_model=False
```

Success means the logs show SC2 4.10 starting, a game loading, one episode
completing, and a clean shutdown.

## Run the baseline

The requested baseline defaults are:

```text
td_lambda = 0.4
epsilon_anneal_time = 100000
```

Launch from Git Bash:

```bash
./run_exp.sh qmix <tag>
```

For example:

```bash
./run_exp.sh qmix cpu-baseline
```

The default script runs:

- 3 environment configurations: Protoss, Terran, and Zerg
- 3 unit counts: 10, 5, and 20
- 3 random runs for each combination
- 27 experiments total
- 1 experiment at a time

Each experiment uses `t_max: 10050000`, or 10,050,000 environment steps.
The complete default command therefore requests 271,350,000 environment steps.

On the tested integrated-graphics machine, one CPU experiment was initially
estimated at approximately 1.5 to 2 days. All 27 sequential runs may require
roughly 40 to 55 days. This estimate varies with unit count, episode length,
CPU thermal limits, Docker resources, and OneDrive activity.

To monitor a running job:

```bash
docker ps
docker logs --tail 100 <container-name>
docker stats <container-name> --no-stream
```

Look for messages in this form:

```text
t_env: 70965 / 10050000
```

## Run a smaller experiment

For iteration, call the container runner directly and override `t_max`:

```bash
./run_docker.sh cpu \
  python3 src/main.py \
  --config=qmix \
  --env-config=sc2_gen_protoss \
  with \
  group=qmix-protoss-test \
  env_args.capability_config.n_units=5 \
  env_args.capability_config.n_enemies=5 \
  use_wandb=True \
  use_cuda=False \
  td_lambda=0.4 \
  epsilon_anneal_time=100000 \
  t_max=100000 \
  save_model=True
```

Do not increase `run_exp.sh`'s thread count on a memory-limited CPU machine.
The default QMIX runner already launches four SC2 environments inside one
experiment and consumed approximately 13 GB during testing.

## Checkpoints and outputs

`run_exp.sh` passes `save_model=True`.

Models are stored under:

```text
results/models/qmix__<timestamp>/<environment-step>/
```

A QMIX checkpoint normally contains:

```text
agent.th
mixer.th
opt.th
```

The current configuration saves:

- once after the first completed episode;
- subsequently at intervals of at least 2,000,000 environment steps.

There is no unconditional final-save call. With a 10,050,000-step budget, a
checkpoint should normally be produced near 10 million steps, shortly before
training ends.

Sacred metadata is stored below `results/sacred/`, and metrics are sent to
W&B. The current code does not upload model checkpoint files as W&B artifacts.

## Evaluate a checkpoint

Pass the checkpoint directory that contains the numeric timestep
subdirectories. With `load_step=0`, PyMARL2 selects the largest available
timestep:

```bash
./run_docker.sh cpu \
  python3 src/main.py \
  --config=qmix \
  --env-config=sc2_gen_protoss \
  with \
  checkpoint_path=/source/results/models/qmix__<timestamp> \
  load_step=0 \
  evaluate=True \
  test_nepisode=10 \
  use_wandb=False \
  use_cuda=False \
  env_args.capability_config.n_units=10 \
  env_args.capability_config.n_enemies=10
```

The environment configuration, unit count, enemy count, and model
architecture must match the checkpoint's training run.

This match is strict. A checkpoint trained with `sc2_gen_protoss` cannot be
loaded under `sc2_gen_terran`, even when both use the same number of agents.
The races produce different observation widths and therefore different QMIX
network tensor shapes. The number of enemies also determines the action
count. A mismatch produces errors such as:

```text
RuntimeError: Error(s) in loading state_dict for NRNNAgent:
    size mismatch for fc1.weight
    size mismatch for fc2.weight
```

Use the Sacred or W&B run metadata to recover the original command-line
arguments. In a local W&B run, `wandb-metadata.json` records `args`, while
`logs/debug.log` records the merged configuration. Do not infer the
environment from the checkpoint directory name: the default name
`qmix__<timestamp>` does not include the race or unit counts.

## Replays and visualization

SMACv2 can request a native `.SC2Replay` from StarCraft II. This repository's
`EpisodeRunner` implements replay saving, but `ParallelRunner.save_replay()` is
a no-op. QMIX defaults to the parallel runner, so replay evaluation must
override it:

```bash
./run_docker.sh cpu \
  python3 src/main.py \
  --config=qmix \
  --env-config=sc2_gen_protoss \
  with \
  checkpoint_path=/source/results/models/qmix__<timestamp> \
  load_step=0 \
  runner=episode \
  batch_size_run=1 \
  evaluate=True \
  save_replay=True \
  test_nepisode=1 \
  env_args.replay_dir=/source/results/replays \
  env_args.replay_prefix=qmix-protoss \
  use_wandb=False \
  use_cuda=False \
  env_args.capability_config.n_units=10 \
  env_args.capability_config.n_enemies=10
```

The replay is then persisted on the Windows host under:

```text
results/replays/*.SC2Replay
```

Without `runner=episode`, evaluation may complete successfully but produce no
replay because `ParallelRunner.save_replay()` is empty. Without a replay
directory below `/source`, `docker run --rm` may delete the replay with the
container.

### Open a generated replay in Windows StarCraft II

Install StarCraft II for Windows, then double-click the `.SC2Replay` file or
open it from PowerShell:

```powershell
Invoke-Item ".\results\replays\<replay-name>.SC2Replay"
```

The generated replay references the underlying SMACv2 arena map:

```text
SMAC_Maps/32x32_flat.SC2Map
```

It does **not** reference `10gen_protoss.SC2Map`, even when the replay filename
or SMACv2 scenario is named `10gen_protoss`. Copy the arena map into the
Windows SC2 installation. Open PowerShell as Administrator and run these
commands from the repository root:

```powershell
$destination = "C:\Program Files (x86)\StarCraft II\Maps\SMAC_Maps"
New-Item -ItemType Directory -Force $destination
Copy-Item `
  ".\3rdparty\StarCraftII\Maps\SMAC_Maps\32x32_flat.SC2Map" `
  $destination
```

If PowerShell is not currently in the repository, use an absolute source path:

```powershell
Copy-Item `
  "C:\path\to\pymarl2\3rdparty\StarCraftII\Maps\SMAC_Maps\32x32_flat.SC2Map" `
  "C:\Program Files (x86)\StarCraft II\Maps\SMAC_Maps\"
```

Fully close and restart StarCraft II after copying the map, then reopen the
replay.

SC2 replays are version-dependent. The Docker environment creates version
4.10 (`Base75689`) replays. A current Windows installation may only contain
5.0.16 (`Base97425`). If the map is in the correct location but playback still
fails, inspect the error separately from the missing-map problem: the Windows
client may not have the older `Base75689` runtime required by the replay.
Practical visualization choices are:

1. Play the `.SC2Replay` using a matching SC2 4.10 runtime.
2. Use PySC2 replay playback with the matching runtime.
3. Add a version-independent Python visualizer that records unit position,
   health, movement, attacks, and rewards during evaluation and exports an
   animation or MP4.

Very early checkpoints, such as one saved after only a few hundred steps,
will behave almost randomly and are not useful demonstrations.

## Common failures

### `python3: can't open file 'src/main.py'`

Cause: Git Bash/MSYS rewrote the Docker bind-mount arguments, or the launcher
mounted the shell's current directory instead of the repository.

Required launcher behavior:

- resolve the repository using `BASH_SOURCE[0]`;
- convert it using `cygpath -m` on Git Bash;
- launch Docker with `MSYS_NO_PATHCONV=1`;
- use `--mount type=bind,...,target=/source`;
- set `--workdir /source`.

The `run_docker.sh` in this working copy includes these changes.

### `/bin/bash^M: bad interpreter`

Cause: Windows CRLF line endings were copied into the Linux image.

The Dockerfile normalizes `install_dependencies.sh` with:

```dockerfile
RUN sed -i 's/\r$//' /source/install_dependencies.sh
```

### W&B tries to compile from source and cannot find Go

Cause: the latest unpinned W&B release no longer provides the required wheel
for the image's Python 3.8 environment.

Use:

```text
wandb==0.21.4
```

Also ensure `install_dependencies.sh` uses `set -euo pipefail`; otherwise pip
can fail while the Docker build continues with a broken image.

### `WANDB_API_KEY_FILE must point to a non-empty file`

Open a new Git Bash terminal after setting the persistent Windows environment
variable. Check:

```bash
echo "$WANDB_API_KEY_FILE"
test -s "$WANDB_API_KEY_FILE" && echo ready
```

### Docker rejects `--gpus`

The machine does not have an NVIDIA GPU available to Docker. Use `cpu` as the
first `run_docker.sh` argument. PyTorch will automatically run on CPU.

### SC2 directory exists but the game does not launch

An earlier failed installation may have created only the map directory.
Check the executable itself:

```bash
test -x 3rdparty/StarCraftII/Versions/Base75689/SC2_x64
```

Rerun `./install_sc2.sh` if it is missing.

### Sacred/GitPython reports dubious ownership

The bind-mounted Windows repository is owned differently inside the
container. The Docker image configures `/source` as a safe Git directory.

## AI-agent execution checklist

An AI agent reproducing this environment should:

1. Inspect the repository and preserve unrelated user changes.
2. Check Docker Desktop state and verify the Linux engine responds.
3. Detect available GPUs instead of assuming NVIDIA CUDA support.
4. Check for an existing non-empty W&B key file without printing its content.
5. Confirm that Windows SC2 cannot replace the Linux container runtime.
6. Make installation scripts location-independent and idempotent.
7. Account for CRLF line endings in Docker build inputs.
8. Pin legacy dependencies that no longer resolve on Python 3.8.
9. Use a Git Bash-safe bind mount and explicit `/source` working directory.
10. Build the image and verify imports from inside the final image.
11. Verify the exact SC2 executable and `32x32_flat.SC2Map`.
12. Run a one-episode, W&B-disabled smoke test before starting training.
13. Confirm that SC2 starts, the map loads, an episode completes, and the
    container exits cleanly.
14. Explain that the default command launches 27 sequential 10.05-million-step
    experiments and obtain confirmation before starting it when runtime or
    compute cost matters.
15. Never expose the W&B API key in logs, commands, patches, or documentation.

## References

- [SMACv2 repository](https://github.com/oxwhirl/smacv2)
- [PySC2 environment and replay documentation](https://github.com/google-deepmind/pysc2)
- [PyMARL2 repository used by this project](https://github.com/benellis3/pymarl2)
