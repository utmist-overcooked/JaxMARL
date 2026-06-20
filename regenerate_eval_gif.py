"""
Standalone recovery script: regenerate the eval gif for a completed training run
without retraining, using the actor checkpoint already saved to disk.

Usage:
    pip install imageio --break-system-packages   # if not already installed
    python regenerate_eval_gif.py \
        --actor_path /path/to/..._vmap0_actor.safetensors \
        --config_yaml /path/to/..._config.yaml \
        --seed_index 0 \
        --out_dir /path/to/models

This re-imports the training script as a module to reuse run_eval_episode,
ActorRNN, OvercookedV3, and OvercookedV3Visualizer exactly as defined there,
so the eval logic stays in sync with whatever's in the main script.

python regenerate_eval_gif.py \
        --actor_path /home/raiyan/JaxMARL/jaxmarl/mappo_overcooked_v3_full_obs/models/button_gated_zones_mappo_full_obs_20260617/1098_actor.safetensors \
        --config_yaml /home/raiyan/JaxMARL/jaxmarl/mappo_overcooked_v3_full_obs/wandb/latest-run/files/config.yaml \
        --seed_index 0 \
        --out_dir /home/raiyan/JaxMARL/jaxmarl/mappo_overcooked_v3_full_obs/models
"""

import argparse
import importlib.util
import os

import jax
from flax.traverse_util import unflatten_dict
from omegaconf import OmegaConf
from safetensors.flax import load_file


def load_params(filename):
    flat = load_file(filename)
    return unflatten_dict(flat, sep=",")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script_path", default="/home/raiyan/JaxMARL/baselines/MAPPO/mappo_rnn_overcooked_v3_full_obs.py")
    parser.add_argument("--actor_path", required=True)
    parser.add_argument("--config_yaml", required=True)
    parser.add_argument("--seed_index", type=int, default=0)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="If set, log the gif to the existing wandb run via wandb.Api/run resume "
        "instead of just writing it to disk.",
    )
    parser.add_argument("--wandb_run_path", default=None, help="entity/project/run_id")
    args = parser.parse_args()

    print(f"[1/6] loading training script module from {args.script_path}...", flush=True)
    spec = importlib.util.spec_from_file_location("trainscript", args.script_path)
    trainscript = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trainscript)
    print("[1/6] training script module loaded", flush=True)

    print(f"[2/6] loading config from {args.config_yaml}...", flush=True)
    config = OmegaConf.load(args.config_yaml)
    config = OmegaConf.to_container(config, resolve=True)

    # Unwrap W&B 'value' nesting
    config = {
        k: (v["value"] if isinstance(v, dict) and "value" in v else v) 
        for k, v in config.items()
    }

    print(f"[3/6] loading actor params from {args.actor_path}...", flush=True)
    actor_params = load_params(args.actor_path)
    print("[3/6] actor params loaded", flush=True)

    print("[4/6] building unwrapped eval env and actor network...", flush=True)
    eval_env = trainscript.OvercookedV3(**config["ENV_KWARGS"])
    actor_network = trainscript.ActorRNN(
        eval_env.action_space(eval_env.agents[0]).n, config=config
    )
    print("[4/6] constructing OvercookedV3Visualizer (this is the step most likely "
          "to hang on a headless machine if a display backend is required)...", flush=True)
    viz = trainscript.OvercookedV3Visualizer(eval_env)
    print("[4/6] visualizer constructed", flush=True)

    print("[5/6] running eval rollout...", flush=True)
    eval_rng = jax.random.PRNGKey(config["SEED"] + 1000 + args.seed_index)
    state_seq = trainscript.run_eval_episode(
        actor_params,
        actor_network,
        eval_env,
        eval_rng,
        config,
        max_steps=config["ENV_KWARGS"]["max_steps"],
    )
    print(f"[5/6] eval rollout finished, {len(state_seq)} states collected", flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    gif_path = os.path.join(args.out_dir, f"final_episode_seed{args.seed_index}.gif")
    print("[6/6] stacking states into one pytree...", flush=True)
    stacked_state_seq = jax.tree_util.tree_map(
        lambda *leaves: jax.numpy.stack(leaves), *state_seq
    )
    print("[6/6] states stacked, calling viz.animate() now "
          "(if this is the last line you see, the hang is inside animate() itself: "
          "either frame rendering or the imageio gif-encoding step)...", flush=True)
    viz.animate(stacked_state_seq, filename=gif_path)
    print(f"[6/6] Wrote gif: {gif_path}", flush=True)

    if args.log_to_wandb:
        import wandb

        if args.wandb_run_path:
            api = wandb.Api()
            run = api.run(args.wandb_run_path)
            run.upload_file(gif_path)
            print(f"Uploaded {gif_path} as a file to existing run {args.wandb_run_path}")
        else:
            print(
                "Skipped wandb upload: --wandb_run_path not given. "
                "Pass entity/project/run_id to attach the gif to the original run."
            )


if __name__ == "__main__":
    main()