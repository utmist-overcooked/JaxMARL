"""Rendering, local paths, and optional W&B upload for V3 rollouts."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from baselines.overcooked_v3.rollout import RolloutEpisode
from jaxmarl.viz.overcooked_v3_visualizer import OvercookedV3Visualizer


def checkpoint_gif_indices(
    num_checkpoints: int,
    gif_count: int,
) -> tuple[int, ...]:
    """Select evenly spaced one-based checkpoints, including the final one."""

    num_checkpoints = int(num_checkpoints)
    gif_count = int(gif_count)
    if gif_count == 0:
        return ()
    if num_checkpoints <= 0:
        raise ValueError("num_checkpoints must be positive when GIFs are enabled")
    if gif_count < 0:
        raise ValueError("gif_count cannot be negative")
    if num_checkpoints % gif_count != 0:
        raise ValueError(
            "The number of checkpoints must be divisible by the requested "
            f"GIF count; received {num_checkpoints} checkpoints and "
            f"{gif_count} GIFs"
        )

    stride = num_checkpoints // gif_count
    return tuple(range(stride, num_checkpoints + 1, stride))


def saved_checkpoint_updates(
    num_updates: int,
    checkpoint_interval: int,
) -> tuple[int, ...]:
    """Return update numbers that save checkpoints, including the final update."""

    num_updates = int(num_updates)
    checkpoint_interval = int(checkpoint_interval)
    if num_updates <= 0 or checkpoint_interval <= 0:
        return ()
    updates = list(
        range(checkpoint_interval, num_updates + 1, checkpoint_interval)
    )
    if not updates or updates[-1] != num_updates:
        updates.append(num_updates)
    return tuple(updates)


def rollout_milestones(num_updates: int, count: int) -> tuple[int, ...]:
    """Compatibility alias treating each supplied update as a checkpoint."""

    return checkpoint_gif_indices(num_updates, count)


def sanitize_run_name(run_name: str | None) -> str:
    """Make a W&B/run name safe for a local directory name."""

    if not run_name:
        return "run"
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(run_name)).strip("._")
    return safe_name or "run"


def rollout_gif_path(
    root_dir: str | os.PathLike[str],
    run_name: str | None,
    checkpoint_index: int,
    update_step: int,
    env_step: int,
    seed: int,
    rollout_seed: int | None = None,
) -> Path:
    """Build a GIF path and ensure its run directory exists.

    The optional ``rollout_seed`` distinguishes the fixed environment seed from
    the training seed. Omitting it preserves the original filename format.
    """

    run_dir = Path(root_dir) / sanitize_run_name(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_suffix = f"seed{int(seed)}"
    if rollout_seed is not None:
        seed_suffix = f"trainseed{int(seed)}_rolloutseed{int(rollout_seed)}"
    return run_dir / (
        f"rollout_checkpoint{int(checkpoint_index):06d}_"
        f"update{int(update_step):06d}_"
        f"envstep{int(env_step):010d}_{seed_suffix}.gif"
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
    """Log a GIF to W&B when a run is active."""

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


@dataclass(frozen=True)
class GifLogResult:
    """Describe a locally rendered rollout and its optional W&B upload."""

    path: Path
    uploaded: bool
    episode_length: int


def save_rollout_gif(
    episode: RolloutEpisode,
    env,
    *,
    root_dir: str | os.PathLike[str],
    run_name: str | None,
    checkpoint_index: int,
    update_step: int,
    env_step: int,
    training_seed: int,
    rollout_seed: int,
    wandb_mode: str | None,
    wandb_key: str = "rollouts/policy",
) -> GifLogResult:
    """Render a recorded episode, save it locally, and optionally upload it."""

    gif_path = rollout_gif_path(
        root_dir,
        run_name,
        checkpoint_index,
        update_step,
        env_step,
        training_seed,
        rollout_seed,
    )
    visualizer = OvercookedV3Visualizer(env)
    visualizer.animate(
        stack_state_sequence(episode.states),
        filename=str(gif_path),
    )
    uploaded = log_gif_to_wandb(
        gif_path,
        wandb_key,
        env_step,
        wandb_mode,
    )
    return GifLogResult(gif_path, uploaded, episode.length)
