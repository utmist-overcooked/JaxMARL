"""Extract actor-only safetensors from full-runner training checkpoints.

`maybe_checkpoint` in mappo_macro_common.py dumps the *entire* training
runner (actor_state, critic_state, env_state, obs, rng) as a flat list of
array leaves to `checkpoints/checkpoint_{step:08d}.npz`. There is no
actor-only file per step -- only `final_actor.safetensors` (end of training)
and `best_actor.safetensors` (best eval score so far) are saved standalone.

This script rebuilds the same runner pytree structure used at training
init (without actually training), uses it to unflatten each checkpoint
file, and saves just the actor params as a safetensors file per step so
they can be loaded with `jaxmarl.wrappers.baselines.load_params` the same
way `visualize_macro_mappo_rollout.py` loads `final_actor.safetensors`.
"""

import argparse
from pathlib import Path
import re
import sys

MAPPO_DIR = Path(__file__).parents[1] / "baselines" / "MAPPO"
sys.path.insert(0, str(MAPPO_DIR))

import jax
import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import save_params

from baselines.MAPPO.mappo_macro_common import (
    Actor,
    Critic,
    ReplanActor,
    build_env,
    initialize_actor_critic,
    initialize_config,
)

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.npz$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--variant", choices=("boundary", "every_step", "replan"), required=True
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/actors",
    )
    parser.add_argument(
        "--steps",
        type=int,
        nargs="*",
        default=None,
        help="Only extract these completed-update steps. Defaults to every "
        "checkpoint_*.npz found in <run-dir>/checkpoints.",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_initial_runner(config, variant, seed):
    """Reconstruct the exact runner pytree structure used at training init.

    Only shapes matter here -- initialize_actor_critic and env.reset just
    need to produce a pytree with the same structure/shapes as the one
    maybe_checkpoint flattened, so the saved leaves can be unflattened
    back into it. The actual values are immediately discarded and
    replaced by the checkpoint's arrays.
    """
    env = build_env(config)
    config = initialize_config(config, env)

    actor_cls = ReplanActor if variant == "replan" else Actor
    actor = actor_cls(env.num_actions, int(config["HIDDEN_SIZE"]))
    critic = Critic(int(config["HIDDEN_SIZE"]))

    rng = jax.random.PRNGKey(seed)
    rng, actor_state, critic_state = initialize_actor_critic(
        actor,
        critic,
        jnp.zeros((1, env.observation_space(env.agents[0]).shape[0])),
        jnp.zeros((1, env.world_state_size())),
        rng,
        config,
    )

    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, int(config["NUM_ENVS"]))
    obs, env_state = jax.vmap(env.reset)(reset_keys)

    return (actor_state, critic_state, env_state, obs, rng)


def discover_checkpoints(checkpoint_dir: Path, steps):
    found = {}
    for path in checkpoint_dir.glob("checkpoint_*.npz"):
        match = CHECKPOINT_RE.match(path.name)
        if match:
            found[int(match.group(1))] = path
    if steps is not None:
        missing = sorted(set(steps) - set(found))
        if missing:
            raise ValueError(
                f"Requested steps not found in {checkpoint_dir}: {missing}. "
                f"Available: {sorted(found)}"
            )
        return {step: found[step] for step in steps}
    return dict(sorted(found.items()))


def extract_actor_params(checkpoint_path: Path, target_leaves, tree_definition):
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        restored_leaves = [archive[f"arr_{i}"] for i in range(len(archive.files))]
    if len(restored_leaves) != len(target_leaves):
        raise ValueError(
            f"{checkpoint_path.name}: checkpoint structure does not match the "
            "reconstructed runner (wrong --variant, or config.yaml doesn't "
            "match the run that produced this checkpoint)"
        )
    for restored_leaf, target_leaf in zip(restored_leaves, target_leaves):
        if restored_leaf.shape != np.asarray(target_leaf).shape:
            raise ValueError(
                f"{checkpoint_path.name}: checkpoint array shapes do not match "
                "the reconstructed runner"
            )
    restored = jax.tree.unflatten(tree_definition, restored_leaves)
    actor_state = restored[0]
    return actor_state.params


def main():
    args = parse_args()
    config = OmegaConf.to_container(
        OmegaConf.load(args.run_dir / "config.yaml"), resolve=True
    )

    checkpoint_dir = args.run_dir / "checkpoints"
    checkpoints = discover_checkpoints(checkpoint_dir, args.steps)
    if not checkpoints:
        raise ValueError(f"No checkpoint_*.npz files found in {checkpoint_dir}")

    output_dir = args.output_dir or (checkpoint_dir / "actors")
    output_dir.mkdir(parents=True, exist_ok=True)

    initial_runner = build_initial_runner(config, args.variant, args.seed)
    target_leaves, tree_definition = jax.tree.flatten(initial_runner)

    for step, checkpoint_path in checkpoints.items():
        actor_params = extract_actor_params(
            checkpoint_path, target_leaves, tree_definition
        )
        output_path = output_dir / f"actor_{step:08d}.safetensors"
        save_params(actor_params, output_path)
        print(f"{checkpoint_path.name} -> {output_path}")

    print(f"\nExtracted {len(checkpoints)} actor checkpoint(s) to {output_dir}")


if __name__ == "__main__":
    main()
