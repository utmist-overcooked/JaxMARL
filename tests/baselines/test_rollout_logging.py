"""Unit tests for checkpoint-based Overcooked V3 GIF selection."""

from pathlib import Path

import pytest

from baselines.utils.rollout_logging import (
    checkpoint_gif_indices,
    log_gif_to_wandb,
    rollout_gif_path,
    saved_checkpoint_updates,
    sanitize_run_name,
)


def test_twenty_checkpoints_selects_every_other_for_ten_gifs():
    """Select checkpoints 2, 4, ..., 20 for the motivating example."""

    assert checkpoint_gif_indices(20, 10) == tuple(range(2, 21, 2))


def test_checkpoint_selection_includes_final_checkpoint():
    """Always include the final checkpoint when the counts divide evenly."""

    assert checkpoint_gif_indices(12, 3) == (4, 8, 12)


def test_checkpoint_selection_rejects_non_integer_spacing():
    """Reject checkpoint and GIF counts that do not divide evenly."""

    with pytest.raises(ValueError, match="must be divisible"):
        checkpoint_gif_indices(19, 10)


def test_checkpoint_selection_handles_disabled_gifs():
    """Allow zero requested GIFs without requiring saved checkpoints."""

    assert checkpoint_gif_indices(0, 0) == ()


def test_saved_checkpoint_updates_include_non_interval_final_update():
    """Model the checkpoint condition used by host-driven trainers."""

    assert saved_checkpoint_updates(20, 6) == (6, 12, 18, 20)


def test_rollout_gif_path_uses_checkpoint_and_safe_run_directory(tmp_path):
    """Include the checkpoint identity and sanitize the run directory."""

    path = rollout_gif_path(
        tmp_path,
        "run/name with spaces",
        checkpoint_index=2,
        update_step=7,
        env_step=1024,
        seed=123,
    )

    assert path.parent == Path(tmp_path) / "run_name_with_spaces"
    assert path.name == (
        "rollout_checkpoint000002_update000007_"
        "envstep0000001024_seed123.gif"
    )
    assert path.parent.exists()


def test_rollout_gif_path_distinguishes_training_and_environment_seeds(tmp_path):
    """Name both seeds so checkpoint GIFs remain reproducible."""

    path = rollout_gif_path(
        tmp_path,
        "experiment",
        checkpoint_index=1,
        update_step=2,
        env_step=128,
        seed=123,
        rollout_seed=7,
    )

    assert path.name == (
        "rollout_checkpoint000001_update000002_envstep0000000128_"
        "trainseed123_rolloutseed7.gif"
    )


def test_sanitize_run_name_has_stable_fallback():
    """Use a stable directory when no usable run name is supplied."""

    assert sanitize_run_name("") == "run"
    assert sanitize_run_name("...") == "run"


def test_disabled_wandb_mode_does_not_attempt_upload(tmp_path):
    """Skip W&B imports and uploads when logging is disabled."""

    assert (
        log_gif_to_wandb(
            tmp_path / "missing.gif",
            "rollouts/policy",
            1,
            "disabled",
        )
        is False
    )
