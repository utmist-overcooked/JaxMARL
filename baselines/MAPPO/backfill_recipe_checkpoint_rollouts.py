"""Backfill recipe-specific checkpoint rollout media for completed CTC runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import wandb
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import load_params
from mappo_rnn_overcooked_v3_fsq_distill import (
    _checkpoint_gif_namespace,
    render_and_log_checkpoint_gif,
)


RUNS = [
    {
        "run_id": "pzq6nu8f",
        "run_name": "fsq_ctc_harder_0_baseline_seed0_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_0_baseline_seed0_5m_1648282_0"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_0_baseline_seed0_5m_1648282_0/models/"
            "fsq_ctc_harder_0_baseline_seed0_5m_20260612"
        ),
    },
    {
        "run_id": "7xpzyl76",
        "run_name": "fsq_ctc_harder_1_baseline_seed1_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_1_baseline_seed1_5m_1648282_1"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_1_baseline_seed1_5m_1648282_1/models/"
            "fsq_ctc_harder_1_baseline_seed1_5m_20260612"
        ),
    },
    {
        "run_id": "572saciq",
        "run_name": "fsq_ctc_harder_2_stronger_distill_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_2_stronger_distill_5m_1648282_2"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_2_stronger_distill_5m_1648282_2/models/"
            "fsq_ctc_harder_2_stronger_distill_5m_20260612"
        ),
    },
    {
        "run_id": "iyu7fa78",
        "run_name": "fsq_ctc_harder_3_longer_distill_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_3_longer_distill_5m_1648282_3"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_3_longer_distill_5m_1648282_3/models/"
            "fsq_ctc_harder_3_longer_distill_5m_20260612"
        ),
    },
    {
        "run_id": "bm4od9su",
        "run_name": "fsq_ctc_harder_4_strong_long_distill_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_4_strong_long_distill_5m_1648282_4"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_4_strong_long_distill_5m_1648282_4/models/"
            "fsq_ctc_harder_4_strong_long_distill_5m_20260612"
        ),
    },
    {
        "run_id": "p02k9hbh",
        "run_name": "fsq_ctc_harder_5_soft_teacher_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_5_soft_teacher_5m_1648282_5"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_5_soft_teacher_5m_1648282_5/models/"
            "fsq_ctc_harder_5_soft_teacher_5m_20260612"
        ),
    },
    {
        "run_id": "fq089q7m",
        "run_name": "fsq_ctc_harder_6_tiny_channel_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_6_tiny_channel_5m_1648282_6"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_6_tiny_channel_5m_1648282_6/models/"
            "fsq_ctc_harder_6_tiny_channel_5m_20260612"
        ),
    },
    {
        "run_id": "q2jy53we",
        "run_name": "fsq_ctc_harder_7_big_channel_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_7_big_channel_5m_1648282_7"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_7_big_channel_5m_1648282_7/models/"
            "fsq_ctc_harder_7_big_channel_5m_20260612"
        ),
    },
    {
        "run_id": "g8ckvggq",
        "run_name": "fsq_ctc_harder_8_larger_partial_view_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_8_larger_partial_view_5m_1648282_8"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_8_larger_partial_view_5m_1648282_8/models/"
            "fsq_ctc_harder_8_larger_partial_view_5m_20260612"
        ),
    },
    {
        "run_id": "4fghqtdr",
        "run_name": "fsq_ctc_harder_9_no_distill_control_5m",
        "run_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_9_no_distill_control_5m_1648282_9"
        ),
        "checkpoint_dir": Path(
            "/scratch/tangzach/jaxmarl/fsq_ctc_10run/"
            "fsq_ctc_harder_9_no_distill_control_5m_1648282_9/models/"
            "fsq_ctc_harder_9_no_distill_control_5m_20260612"
        ),
    },
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
        "disable_fsq_comm": True,
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
        "disable_fsq_comm": True,
    },
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--entity", default="zacharytang24-")
    parser.add_argument("--project", default="overcookedv3-zac-distillation")
    parser.add_argument("--namespace", default="checkpoint_rollouts_by_recipe")
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


def load_run_config(
    run_info: dict,
    *,
    max_steps: int,
    fps: int,
    tile_size: int,
    namespace: str,
    mode: str,
) -> tuple[dict, Path]:
    config_paths = sorted(run_info["run_dir"].glob("models/*_config.yaml"))
    if len(config_paths) != 1:
        raise RuntimeError(
            f"Expected one config under {run_info['run_dir']}/models, got {config_paths}"
        )
    config = OmegaConf.to_container(OmegaConf.load(config_paths[0]), resolve=False)
    if run_info.get("disable_fsq_comm") is not None:
        config["DISABLE_FSQ_COMM"] = bool(run_info["disable_fsq_comm"])
    else:
        config["DISABLE_FSQ_COMM"] = bool(config.get("DISABLE_FSQ_COMM", False))
    config["CHECKPOINT_GIF"] = True
    config["CHECKPOINT_GIF_NAMESPACE"] = namespace
    config["CHECKPOINT_GIF_MEDIA_KEY"] = f"{namespace}/rollout"
    config["CHECKPOINT_GIF_OUTPUT_DIR"] = str(run_info["run_dir"] / namespace)
    config["CHECKPOINT_GIF_MAX_STEPS"] = max_steps
    config["CHECKPOINT_GIF_EPSILON"] = 0.0
    config["CHECKPOINT_GIF_SEED"] = 0
    config["CHECKPOINT_GIF_FPS"] = fps
    config["CHECKPOINT_GIF_TILE_SIZE"] = tile_size
    config["CHECKPOINT_FSQ_VIEWER"] = True
    config["WANDB_MODE"] = mode
    config["WANDB_DIR"] = str(run_info["run_dir"])
    return config, config_paths[0]


def main() -> None:
    args = parse_args()
    for run_info in RUNS:
        run_dir = run_info["run_dir"]
        run_name = run_info["run_name"]
        run_id = run_info["run_id"]
        checkpoint_paths = checkpoint_actor_paths(run_info["checkpoint_dir"])
        if not checkpoint_paths:
            raise RuntimeError(
                f"No checkpoint actor files found in {run_info['checkpoint_dir']}"
            )

        config, config_path = load_run_config(
            run_info,
            max_steps=args.max_steps,
            fps=args.fps,
            tile_size=args.tile_size,
            namespace=args.namespace,
            mode=args.mode,
        )
        output_dir = run_dir / args.namespace / run_name
        output_dir.mkdir(parents=True, exist_ok=True)
        namespace = _checkpoint_gif_namespace(config)

        print(
            f"RUN {run_name} ({run_id}): checkpoints={len(checkpoint_paths)} "
            f"disable_fsq={config['DISABLE_FSQ_COMM']} namespace={namespace}"
        )
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            id=run_id,
            name=run_name,
            resume="allow",
            mode=args.mode,
            dir=str(run_dir),
        )
        wandb.define_metric(f"{namespace}/rollout_index")
        wandb.define_metric(f"{namespace}/*", step_metric=f"{namespace}/rollout_index")

        try:
            for rollout_index, (update, actor_path) in enumerate(checkpoint_paths):
                print(f"{run_name}: rendering update={update} actor={actor_path}")
                render_and_log_checkpoint_gif(
                    actor_params=load_params(str(actor_path)),
                    config=config,
                    update=update,
                    run_name=run_name,
                    checkpoint_interval=max(len(checkpoint_paths), 1),
                    rollout_index=rollout_index,
                    actor_path=actor_path,
                    config_path=config_path,
                )
        finally:
            wandb.finish()


if __name__ == "__main__":
    main()
