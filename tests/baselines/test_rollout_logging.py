from pathlib import Path

from baselines.utils.rollout_logging import (
    log_gif_to_wandb,
    rollout_gif_path,
    rollout_milestones,
    sanitize_run_name,
)


def test_rollout_milestones_are_ten_percent_steps():
    assert rollout_milestones(100, 10) == (
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    )


def test_rollout_milestones_include_final_for_non_divisible_updates():
    assert rollout_milestones(19, 10) == (2, 4, 6, 8, 10, 12, 14, 16, 18, 19)


def test_rollout_milestones_do_not_duplicate_tiny_runs():
    assert rollout_milestones(3, 10) == (1, 2, 3)


def test_rollout_milestones_handle_disabled_inputs():
    assert rollout_milestones(0, 10) == ()
    assert rollout_milestones(100, 0) == ()


def test_rollout_gif_path_uses_safe_run_directory(tmp_path):
    path = rollout_gif_path(tmp_path, "run/name with spaces", 7, 1024, 123)

    assert path.parent == Path(tmp_path) / "run_name_with_spaces"
    assert path.name == "rollout_update000007_envstep0000001024_seed123.gif"
    assert path.parent.exists()


def test_sanitize_run_name_has_stable_fallback():
    assert sanitize_run_name("") == "run"
    assert sanitize_run_name("...") == "run"


def test_disabled_wandb_mode_does_not_attempt_upload(tmp_path):
    assert (
        log_gif_to_wandb(
            tmp_path / "missing.gif",
            "rollouts/greedy",
            1,
            "disabled",
        )
        is False
    )
