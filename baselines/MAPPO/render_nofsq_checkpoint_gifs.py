"""Render and upload no-FSQ checkpoint rollout GIFs for completed runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import load_params
from mappo_rnn_overcooked_v3_fsq_distill import render_and_log_checkpoint_gif


RUNS = [
    {
        "run_id": "1uh1b9p0",
        "run_name": "nofsq_ctc_harder_agentview2_strong_long_seed0_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/nofsq_ctc_harder_agentview2/"
            "nofsq_ctc_harder_agentview2_strong_long_seed0_5m_1669286"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/nofsq_ctc_harder_agentview2/"
            "nofsq_ctc_harder_agentview2_strong_long_seed0_5m_1669286/models/"
            "nofsq_ctc_harder_agentview2_strong_long_seed0_5m_20260615"
        ),
    },
    {
        "run_id": "6yf2krh9",
        "run_name": "nofsq_ctc_harder_agentview2_distill05_seed0_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/nofsq_ctc_harder_agentview2_distill05/"
            "nofsq_ctc_harder_agentview2_distill05_seed0_5m_1669291"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/nofsq_ctc_harder_agentview2_distill05/"
            "nofsq_ctc_harder_agentview2_distill05_seed0_5m_1669291/models/"
            "nofsq_ctc_harder_agentview2_distill05_seed0_5m_20260615"
        ),
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="zacharytang24-")
    parser.add_argument("--project", default="overcookedv3-zac-distillation")
    parser.add_argument("--max-steps", type=int, default=150)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--tile-size", type=int, default=32)
    parser.add_argument("--mode", default="online")
    return parser.parse_args()


def checkpoint_actor_paths(checkpoint_dir: Path) -> list[tuple[int, Path]]:
    paths = []
    for path in checkpoint_dir.glob("*_actor.safetensors"):
        update = int(path.name.split("_", 1)[0])
        paths.append((update, path))
    return sorted(paths)


def load_run_config(run_dir: Path, max_steps: int, fps: int, tile_size: int) -> dict:
    config_paths = sorted(run_dir.glob("models/*_config.yaml"))
    if len(config_paths) != 1:
        raise RuntimeError(f"Expected one config under {run_dir}/models, got {config_paths}")
    config = OmegaConf.to_container(OmegaConf.load(config_paths[0]), resolve=False)
    config["DISABLE_FSQ_COMM"] = True
    config["CHECKPOINT_GIF_MAX_STEPS"] = max_steps
    config["CHECKPOINT_GIF_EPSILON"] = 0.0
    config["CHECKPOINT_GIF_SEED"] = 0
    config["CHECKPOINT_GIF_FPS"] = fps
    config["CHECKPOINT_GIF_TILE_SIZE"] = tile_size
    config["CHECKPOINT_GIF"] = True
    config["CHECKPOINT_GIF_MEDIA_KEY"] = "checkpoint_rollouts/rollout"
    return config


def main() -> None:
    args = parse_args()
    for run_info in RUNS:
        run_dir = run_info["run_dir"]
        run_name = run_info["run_name"]
        run_id = run_info["run_id"]
        output_dir = run_dir / "checkpoint_rollouts" / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        config = load_run_config(run_dir, args.max_steps, args.fps, args.tile_size)
        config["CHECKPOINT_GIF_OUTPUT_DIR"] = str(run_dir / "checkpoint_rollouts")
        config["WANDB_MODE"] = args.mode
        config["WANDB_DIR"] = str(run_dir)

        checkpoint_paths = checkpoint_actor_paths(run_info["checkpoint_dir"])
        if not checkpoint_paths:
            raise RuntimeError(f"No checkpoint actor files found in {run_info['checkpoint_dir']}")

        run = wandb.init(
            entity=args.entity,
            project=args.project,
            id=run_id,
            name=run_name,
            resume="allow",
            mode=args.mode,
            dir=str(run_dir),
        )
        wandb.define_metric("checkpoint_rollouts/rollout_index")
        wandb.define_metric(
            "checkpoint_rollouts/*",
            step_metric="checkpoint_rollouts/rollout_index",
        )

        try:
            for rollout_index, (update, actor_path) in enumerate(checkpoint_paths):
                print(
                    f"{run_name}: rendering update={update} "
                    f"actor={actor_path}"
                )
                render_and_log_checkpoint_gif(
                    actor_params=load_params(str(actor_path)),
                    config=config,
                    update=update,
                    run_name=run_name,
                    checkpoint_interval=max(len(checkpoint_paths), 1),
                    rollout_index=rollout_index,
                )
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
