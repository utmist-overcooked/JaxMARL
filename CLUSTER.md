
## 0. Mental model

Use the cluster like this:

1. **SSH into a login server** for editing, submitting jobs, checking queues, moving files, and light debugging.
2. **Never run serious training on the login server.** Request GPU/CPU resources through **SLURM**.
3. Use **`srun` only for short interactive debugging**, and **`sbatch` for real jobs**. The official precautions explicitly recommend `sbatch`; `srun` sessions that are left open waste resources and may be auto-cancelled after 6 hours. ([Yonsei University College of AI][1])
4. Store code/envs in `/home`, temporary datasets/checkpoints in `/scratch*`, and long-term data in `/lustre`.
5. Do not manually pick GPU IDs. SLURM assigns GPUs and sets `CUDA_VISIBLE_DEVICES`; hard-coding it can collide with other users’ jobs. ([Yonsei University College of AI][1])

---

## 1. Login servers

From your notes:

| Server |                IP | Best use                              |
| ------ | ----------------: | ------------------------------------- |
| login1 |  `165.132.142.82` | node assignment / job submission only |
| login3 | `165.132.142.207` | VSCode connection + simple debugging  |
| login4 | `165.132.142.208` | VSCode connection + simple debugging  |
| login5 |  `165.132.143.45` | general login                         |
| login6 |  `165.132.143.46` | general login                         |

The official guide says the cluster is accessible only from inside campus; from outside campus you need Yonsei VPN, and the precautions say to use VPN even on Yonsei Wi-Fi. ([Yonsei University College of AI][2])

Basic SSH:

```bash
ssh <id>@165.132.142.207
```

For X11 forwarding:

```bash
ssh -XCY <id>@165.132.142.207
```

The guide suggests MobaXterm on Windows because it supports SSH, SCP, and X11 forwarding in one place; macOS needs XQuartz for X11 forwarding. ([Yonsei University College of AI][2])

Useful `~/.ssh/config`:

```sshconfig
Host y-login3
    HostName 165.132.142.207
    User <id>
    ServerAliveInterval 60
    ServerAliveCountMax 5

Host y-login4
    HostName 165.132.142.208
    User <id>
    ServerAliveInterval 60
    ServerAliveCountMax 5

Host y-login5
    HostName 165.132.143.45
    User <id>
    ServerAliveInterval 60
    ServerAliveCountMax 5

Host y-login6
    HostName 165.132.143.46
    User <id>
    ServerAliveInterval 60
    ServerAliveCountMax 5
```

Then:

```bash
ssh y-login3
```

To check load across login servers after logging in:

```bash
for I in $(seq -w 1 6); do
    echo $I
    ssh login$I uptime
done
```

The FAQ explains that `uptime` shows current time, uptime, number of logged-in users, and 1/5/15-minute CPU load averages. ([Yonsei University College of AI][3])

---

## 2. What not to do on login servers

Login servers are shared. Use them for:

```text
editing code
git pull / git clone
small tests
submitting SLURM jobs
checking queue/status
light VSCode debugging on login3/login4
file transfer/downloads
```

Do **not** use them for:

```text
training
long CPU jobs
large preprocessing
large memory jobs
multi-GPU runs
large evaluation loops
```

The precautions page says admins may kill login-server processes without notifying users: CPU ≥200% may be killed immediately; memory ≥20% may be killed immediately; long-running high-CPU/high-memory processes may also be killed. ([Yonsei University College of AI][1])

Check your own login-server load:

```bash
top -u $USER
# or
htop -u $USER
```

---

## 3. Storage: where to put things

From the official storage page, `/home` has a default quota of **200 GB** per account, and extra storage should go to NAS mounted at `/scratch` and `/scratch2`. ([Yonsei University College of AI][4]) Your note also says `/lustre/<id>` is persistent with a 1 TB limit.

Use this rule:

| Path             | Use for                                                          | Risk                                  |
| ---------------- | ---------------------------------------------------------------- | ------------------------------------- |
| `/home/<id>`     | code, scripts, small config files, conda metadata, small outputs | quota fills quickly                   |
| `/scratch/<id>`  | datasets, temporary checkpoints, large intermediate files        | auto-deleted if not used for >30 days |
| `/scratch2/<id>` | same as `/scratch`                                               | auto-deleted if not used for >30 days |
| `/lustre/<id>`   | important personal data, long-term datasets/checkpoints          | 1 TB limit from your note             |

Create your scratch directories:

```bash
mkdir -p /scratch/<id>
mkdir -p /scratch2/<id>
mkdir -p /lustre/<id>
```

Important: `/scratch` and `/scratch2` deletion is based on access time, but simply reading files may not refresh access time over the network. The official storage/precautions pages recommend touching recursively with `find ... | xargs touch`; they also say to do this immediately after downloading because downloaded files may not get the expected access time. ([Yonsei University College of AI][4])

```bash
find /scratch/<id>/folder_name -print | xargs touch
find /scratch2/<id>/folder_name -print | xargs touch
```

Check usage:

```bash
du -h -d 1 ~
du -h -d 1 /scratch/<id>
du -h -d 1 /scratch2/<id>
du -h -d 1 /lustre/<id>
```

The FAQ says `du -h -d 1 ./` is the current way to check folder usage; old `/home` quota commands may no longer apply after `/home` was moved to SOL-NAS. ([Yonsei University College of AI][5])

---

## 4. File transfer and downloads

For small transfers from your machine:

```bash
scp local_file.txt <id>@165.132.142.207:/scratch/<id>/
scp -r local_folder <id>@165.132.142.207:/scratch/<id>/
```

For large transfers, prefer `rsync` because it can resume after interruption:

```bash
rsync -avzh local_folder/ <id>@165.132.142.207:/scratch/<id>/local_folder/
```

The official file-transfer guide recommends SCP for small/simple transfers and `rsync` for large or many files because it resumes interrupted transfers. ([Yonsei University College of AI][6])

Downloads should be run from a **login server**, not a compute/GPU node, and downloaded data should be saved to NAS rather than `/home`; the precautions page says downloading on compute nodes or into `/home` can bottleneck the cluster. ([Yonsei University College of AI][1])

Good pattern:

```bash
cd /scratch/<id>
wget <dataset-url>
find /scratch/<id>/<downloaded_folder> -print | xargs touch
```

---

## 5. SLURM basics

The cluster uses **SLURM** as the resource manager; the official current cluster page lists the environment as OpenHPC Cluster, Rocky9, and SLURM. ([Yonsei University College of AI][7])

Useful commands:

```bash
# cluster/partition status
sinfo
sinfolong

# your queue
squeue -u $USER
squeuelong -u $USER

# cancel a job
scancel <JOBID>

# GPU visibility inside an allocated job
echo $CUDA_VISIBLE_DEVICES

# GPU status
gpustat
nvidia-smi
```

The official guide lists `sinfo`, `squeue`, `srun`, `sbatch`, `scancel`, and `scontrol` as key SLURM commands, and explains that `idle`, `mix`, `alloc`, and `down` represent node states. ([Yonsei University College of AI][8])

---

## 6. Interactive debugging with `srun`

Use `srun` only for short debugging.

One GPU for 1 hour:

```bash
srun --gres=gpu:1 --time=1:00:00 --pty bash -i
```

For larger GPU requests, the guide shows using the big partition/QoS:

```bash
srun -p big -q big --gres=gpu:10 --time=1:00:00 --pty bash -i
```

After allocation:

```bash
echo $CUDA_VISIBLE_DEVICES
gpustat
nvidia-smi
python train.py --small_debug_run
exit
```

The guide recommends specifying a time limit for `srun`, and says interactive jobs are useful for real-time debugging but can waste resources because the allocation does not automatically end just because your command finished. ([Yonsei University College of AI][9])

For X11 inside allocated nodes, the FAQ says to add `--x11`:

```bash
srun -p base_suma_rtx3090 --gres=gpu:1 --x11 --pty bash
```

([Yonsei University College of AI][3])

---

## 7. Real jobs with `sbatch`

A minimal GPU job:

```bash
#!/bin/bash
#SBATCH -J myjob
#SBATCH -o logs/%x.%j.out
#SBATCH -e logs/%x.%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

echo "START: $(date)"
echo "HOST: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

source ~/.bashrc
conda activate myenv

python -u train.py \
    --data_dir /scratch/<id>/dataset \
    --output_dir /lustre/<id>/runs/myjob

echo "END: $(date)"
```

Submit:

```bash
mkdir -p logs
sbatch run.sh
```

For more than the normal/base allocation, use the big partition/QoS only when you really need it:

```bash
sbatch -p big -q big --gres=gpu:8 --time=12:00:00 run.sh
```

The official guide recommends giving `--time` because it can improve backfill scheduling, and shows `sbatch --gres=gpu:1 --time=10:00 script.sh` as the basic pattern. ([Yonsei University College of AI][9])

For CPU threads:

```bash
#SBATCH --cpus-per-task=8
```

The FAQ says `--cpus-per-task=<thread-count>` requests the desired number of CPU threads. ([Yonsei University College of AI][3])

For real-time Python output in `.out` files, use `python -u`; the FAQ notes this for `sbatch` print buffering. ([Yonsei University College of AI][5])

---

## 8. Partitions, QoS, and preemption

The FAQ describes two broad GPU usage modes:

```text
base_q: fair sharing, generally 4 GPUs per user
big_q: larger GPU use, but lower priority / can be preempted
```

The FAQ says `big_q` jobs may be preempted after a `base_q` job has waited for 3 hours, and preemption appears as a SLURM cancellation message. ([Yonsei University College of AI][3])

Practical advice:

```text
Use base/default for normal 1–4 GPU runs.
Use big only for large multi-GPU jobs.
Checkpoint often if using big_q.
Make jobs restartable.
Do not assume a big job will run uninterrupted.
```

Example checkpoint flags:

```bash
python -u train.py \
  --resume auto \
  --save_every_steps 1000 \
  --output_dir /lustre/<id>/runs/exp1
```

---

## 9. Python, CUDA, modules, conda

The guide says the environment can be configured using Anaconda, environment modules, or Singularity. ([Yonsei University College of AI][10])

Common module commands:

```bash
module list
module av
module av | grep -i cuda
module load cuda/12.1
module swap CUDA/11.2 CUDA/11.6
module purge
```

The FAQ says CUDA versions are provided through modules; `module list` checks loaded CUDA, `module av` checks available versions, and `module swap CUDA/11.2 CUDA/11.6` swaps versions. It also warns that the CUDA version shown by `nvidia-smi` is the driver-supported CUDA version, not necessarily the CUDA toolkit you loaded. ([Yonsei University College of AI][3])

Conda in `sbatch`:

```bash
source ~/.bashrc
conda activate myenv
```

FAQ item 18 says `srun` works directly after Anaconda install, but `sbatch` scripts need `source .bashrc`. ([Yonsei University College of AI][3])

Recommended cache setup to avoid `/tmp` and `/home` filling:

```bash
mkdir -p /scratch/<id>/.cache/huggingface
mkdir -p /scratch/<id>/.cache/torch
mkdir -p /scratch/<id>/.cache/pip

export HF_HOME=/scratch/<id>/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/<id>/.cache/huggingface
export TORCH_HOME=/scratch/<id>/.cache/torch
export PIP_CACHE_DIR=/scratch/<id>/.cache/pip
```

The FAQ says “no space left on device” can come from quota being full or `/tmp` being full, especially if Hugging Face defaults to `/tmp/.cache`; it recommends checking usage and moving cache locations. ([Yonsei University College of AI][3])

---

## 10. VSCode Remote SSH

Use **login3** or **login4** for VSCode, based on your notes.

In VSCode Remote SSH, connect to:

```text
y-login3
```

or:

```text
<id>@165.132.142.207
```

Do not use VSCode to run heavy training on the login server. Use the VSCode terminal to submit `sbatch` jobs, inspect logs, and do short debugging.

Important VSCode fixes from FAQ:

```bash
# If VSCode server gets corrupted or quota problems happen:
rm -rf ~/.vscode-server
```

The FAQ says if VSCode fails with disk quota errors, free space via SSH, check usage, and try removing `~/.vscode-server`. ([Yonsei University College of AI][3])

After the OS upgrade/migration, FAQ item 26 says VSCode may fail until you remove the local `known_hosts` entry/file and retry; it also notes that manually installing `vscode-server-linux-x64` solved some cases. Verify host-key changes with the admin if you are unsure, because blindly deleting host keys can hide a real security warning. ([Yonsei University College of AI][3])

On Windows, the FAQ/guide also mention a MobaXterm bug when the Windows username is Korean; the workaround edits `~/.bashrc` in MobaXterm to fix an empty SSH `User ""` entry. ([Yonsei University College of AI][2])

---

## 11. Jupyter Lab

The FAQ gives two ways: X11 forwarding with Firefox from the login node, or SSH SOCKS proxy with `ssh -D`. The Jupyter process itself should run on an allocated compute node, and the FAQ notes using `jupyter lab --ip=0.0.0.0` after getting a node. ([Yonsei University College of AI][5])

Safer practical flow:

```bash
# 1. SSH into login server
ssh y-login3

# 2. Request a GPU node
srun --gres=gpu:1 --time=2:00:00 --pty bash -i

# 3. On allocated node
source ~/.bashrc
conda activate myenv
jupyter lab --ip=0.0.0.0 --no-browser
```

Then use the method recommended by the cluster FAQ: X11 browser or SOCKS proxy. Do not start long Jupyter sessions directly on a login server.

---

## 12. Distributed training / DDP

For RTX4090 or A6000 with PyTorch DDP using NCCL, the FAQ says to add:

```bash
export NCCL_P2P_DISABLE=1
```

or:

```bash
NCCL_P2P_DISABLE=1 torchrun --nnodes=1 --nproc_per_node=4 ddp.py
```

The FAQ says NCCL may try P2P communication on RTX4090/A6000 systems and hang, so disabling P2P avoids that issue. ([Yonsei University College of AI][3])

---

## 13. Singularity / containers

Prefer Singularity over Docker on the cluster. The official resource-allocation guide marks Docker as not recommended and shows Singularity examples using `singularity exec --nv`. ([Yonsei University College of AI][9])

Basic pattern:

```bash
module purge
module load singularity

singularity exec --nv image.sif python train.py
```

If datasets under a mounted path do not appear inside Singularity, the FAQ says to bind the path, for example:

```bash
singularity exec --nv -B /datasets image.sif python train.py
```

([Yonsei University College of AI][3])

For your own scratch/lustre paths:

```bash
singularity exec --nv \
  -B /scratch/<id>:/scratch/<id> \
  -B /lustre/<id>:/lustre/<id> \
  image.sif python train.py
```

---

## 14. Common problems and fixes

| Problem                          | What to do                                                                                                                                                              |
| -------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Job pending forever              | Check `squeue -u $USER`; wrong QoS/partition can leave jobs pending without obvious errors. ([Yonsei University College of AI][5])                                      |
| Need to stop job                 | `scancel <JOBID>` ([Yonsei University College of AI][5])                                                                                                                |
| “No space left on device”        | Run `du -h -d 1 ./`; check `/tmp` on the node; remove your `/tmp/.cache`; move Hugging Face/cache dirs to `/scratch`. ([Yonsei University College of AI][3])            |
| VSCode fails                     | Free disk space, `rm -rf ~/.vscode-server`, retry; after OS upgrade, remove stale local `known_hosts` entry/file if appropriate. ([Yonsei University College of AI][3]) |
| Job stops / conflicts            | Make sure neither your code nor imported code hard-codes `CUDA_VISIBLE_DEVICES`. ([Yonsei University College of AI][5])                                                 |
| Need different CUDA              | Use `module av`, `module list`, `module swap`. ([Yonsei University College of AI][3])                                                                                   |
| `sbatch` cannot see conda        | Add `source ~/.bashrc` before `conda activate`. ([Yonsei University College of AI][3])                                                                                  |
| Prints don’t show until job ends | Use `python -u train.py`. ([Yonsei University College of AI][5])                                                                                                        |
| Need X11 on compute node         | Add `--x11` to `srun`. ([Yonsei University College of AI][3])                                                                                                           |

---

## 15. First-day checklist

Run these after your first login:

```bash
whoami
hostname
pwd
echo $HOME

# login server health
w
uptime
top -u $USER

# storage
mkdir -p /scratch/<id> /scratch2/<id> /lustre/<id>
du -h -d 1 ~

# SLURM visibility
sinfo
squeue -u $USER

# environment
module av | head
module av | grep -i cuda
```

Then create a tiny test job:

```bash
cat > ~/test_gpu.sh <<'EOF'
#!/bin/bash
#SBATCH -J test_gpu
#SBATCH -o test_gpu.%j.out
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00

echo "HOST=$(hostname)"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())
PY
EOF

sbatch ~/test_gpu.sh
squeue -u $USER
```

Check output:

```bash
ls -lh test_gpu.*.out
cat test_gpu.*.out
```

---

## 16. My recommended default workflow

```bash
# login
ssh y-login3

# go to project
cd /lustre/<id>/projects/my_project

# update code
git pull

# keep data in scratch
mkdir -p /scratch/<id>/datasets
mkdir -p /lustre/<id>/runs

# submit real job
sbatch run.sh

# monitor
squeue -u $USER
tail -f logs/myjob.<JOBID>.out
```

Use `/scratch/<id>` for large fast-changing data, `/lustre/<id>` for anything you would be upset to lose, and `sbatch` for nearly everything that uses meaningful CPU/GPU time.

[1]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_3_j "주의사항 1 페이지 | 연세대학교 인공지능융합대학"
[2]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_3_b "사용법 안내 1 페이지 | 연세대학교 인공지능융합대학"
[3]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_4 "FAQ 1 페이지 | 연세대학교 인공지능융합대학"
[4]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_3_i "스토리지 1 페이지 | 연세대학교 인공지능융합대학"
[5]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_4&page=2 "FAQ 2 페이지 | 연세대학교 인공지능융합대학"
[6]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_3_e "사용법 안내 1 페이지 | 연세대학교 인공지능융합대학"
[7]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_2_b "소개 1 페이지 | 연세대학교 인공지능융합대학"
[8]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=resoureceManager "사용법 안내 1 페이지 | 연세대학교 인공지능융합대학"
[9]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=resouceAlloc "사용법 안내 1 페이지 | 연세대학교 인공지능융합대학"
[10]: https://computing.yonsei.ac.kr/bbs/board.php?bo_table=sub6_3_c "사용법 안내 1 페이지 | 연세대학교 인공지능융합대학"
