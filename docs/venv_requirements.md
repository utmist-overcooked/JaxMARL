# `.venv` requirements & trim log

`/student/brownd58/dev/JaxMARL/.venv` — Python 3.10.12, single ~16 GB GPU, JAX 0.4.38 (cu12).

## What the venv actually needs
Authoritative dependency lists already live in the repo:
- **`pyproject.toml`** — abstract deps (`jax<=0.4.38`, `jaxlib`, `flax`, `optax`, `distrax`,
  `flashbax==0.1.0`, `brax==0.10.3`, `mujoco==3.1.6`, `wandb`, `hydra-core`, `chex`,
  `safetensors`, `pygame`, `matplotlib`, `scipy<=1.12`, `pettingzoo`, `gymnax`, …).
- **`requirements.txt`** — the full pinned freeze (exact versions of every transitive dep,
  including the `nvidia-*-cu12` CUDA wheels).

GPU JAX is provided by `jaxlib==0.4.38` + `jax-cuda12-plugin==0.4.38` + `jax-cuda12-pjrt==0.4.38`,
which pull the `nvidia-*-cu12` CUDA runtime wheels. Pip resolved those to the **latest 12.9**
releases, which are far larger than what jaxlib was built against — this is the entire source
of venv bloat (was 5.4 GB, of which 3.9 GB was `nvidia/`).

## Trim performed (2026-07-11) — no reinstall, 5.4 GB → 3.9 GB
Removed CUDA libraries that this repo's workloads (IPPO, `qmix_rnn`, VDN, IS-MADDPG, FSQ
distillation on overcooked_v3 — all single-GPU MLP+GRU) never call. Verified by running the
IPPO training smoke test (`test_ippo_v3_smoke.py cramped_room 200000`) after removal: 7 updates
complete, GPU, exit 0.

| Package (`nvidia/…`)     | Size  | Removed? | Reason |
|--------------------------|-------|----------|--------|
| `nccl`                   | 394M  | ✅ removed | multi-GPU collectives; unused on one GPU (no `pmap`/`psum`/`shard_map` in repo) |
| `cufft`                  | 281M  | ✅ removed | FFT; no `jnp.fft` anywhere in repo |
| `cusparse`               | 465M  | ✅ removed | sparse ops; no `experimental.sparse`/`BCOO` in repo |
| `cusolver`               | 473M  | ✅ removed | dense linalg (QR/eig/solve). Only caller is `baselines/QLearning/transf_qmix.py` with `use_fast_attention=True` (transformer-QMIX, not in our pipeline) |
| `cudnn`                  | 1.1G  | ❌ **kept** | jaxlib 0.4.38 **eagerly initializes the DNN library at startup**; without it even `jnp.array(...)` raises `FAILED_PRECONDITION: DNN library initialization failed`. Required. |
| `cublas`                 | 817M  | ❌ kept | every matmul |
| `cuda_nvrtc`             | 217M  | ❌ kept | runtime kernel compilation |
| `cuda_nvcc` (`ptxas`)    | 95M   | ❌ kept | ptxas — the PATH fix in CLAUDE.md points here |
| `nvjitlink`,`cupti`,`cuda_runtime` | ~145M | ❌ kept | core runtime |

### If a removed lib is ever needed again
Reinstall just that wheel (versions from `requirements.txt`):
```
.venv/bin/pip install nvidia-cusolver-cu12==11.7.5.82   # for transf_qmix + fast attention
.venv/bin/pip install nvidia-cufft-cu12==11.4.1.4
.venv/bin/pip install nvidia-cusparse-cu12==12.5.10.65
.venv/bin/pip install nvidia-nccl-cu12==2.30.4          # multi-GPU only
```

## Further shrink (would require reinstall — not done)
The remaining `cudnn`/`cublas`/`nvrtc` (2.1 GB) are the 12.9 wheels. Pinning the older CUDA
libs jaxlib 0.4.38 was tested against (roughly cu12.3–12.4) would cut these ~2–3× but requires
uninstalling/reinstalling the `nvidia-*-cu12` wheels, so it's out of scope for an in-place trim.
