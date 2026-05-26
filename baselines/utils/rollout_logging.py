"""Rollout GIF scheduling and logging helpers for baseline trainers."""

from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Sequence


def rollout_milestones(num_updates: int, count: int) -> tuple[int, ...]:
    """Return evenly spaced update milestones, including the final update."""
    num_updates = int(num_updates)
    count = int(count)
    if num_updates <= 0 or count <= 0:
        return ()

    num_milestones = min(num_updates, count)
    milestones = {
        int(math.ceil(num_updates * idx / num_milestones))
        for idx in range(1, num_milestones + 1)
    }
    return tuple(sorted(milestones))


def sanitize_run_name(run_name: str | None) -> str:
    """Make a wandb/run name safe for a local directory name."""
    if not run_name:
        return "run"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("._")
    return safe_name or "run"


def rollout_gif_path(
    root_dir: str | os.PathLike[str],
    run_name: str | None,
    update_step: int,
    env_step: int,
    seed: int,
) -> Path:
    """Build the local path for a rollout GIF and ensure its directory exists."""
    run_dir = Path(root_dir) / sanitize_run_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / (
        f"rollout_update{int(update_step):06d}_"
        f"envstep{int(env_step):010d}_seed{int(seed)}.gif"
    )


def stack_state_sequence(states: Sequence):
    """Stack a Python sequence of JAX pytrees along a leading time axis."""
    import jax
    import jax.numpy as jnp

    return jax.tree.map(lambda *xs: jnp.stack(xs), *states)


def log_gif_to_wandb(
    gif_path: str | os.PathLike[str],
    key: str,
    step: int,
    wandb_mode: str | None,
) -> bool:
    """Log a GIF to wandb when a run is active; return whether upload was attempted."""
    if wandb_mode == "disabled":
        return False

    try:
        import wandb
    except ImportError:
        return False

    if wandb.run is None:
        return False

    wandb.log({key: wandb.Video(str(gif_path), format="gif")}, step=int(step))
    return True
