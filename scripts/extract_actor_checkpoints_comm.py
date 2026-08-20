"""Extract comm-module-only safetensors from full-runner comm training checkpoints.

Comm variant of scripts/extract_actor_checkpoints.py. `maybe_checkpoint` in
mappo_macro_common.py, called from mappo_macro_every_step_comm.py, dumps the
entire training runner -- (comm_state, frozen_critic_state, env_state, obs,
rng), NOT (actor_state, critic_state, ...) like the plain macro trainers --
as a flat list of array leaves to `checkpoints/checkpoint_{step:08d}.npz`.

For this variant, "the actor" is comm_state.params: the trained message
encoder + correction head. The frozen macro actor/critic never change during
comm training (they're loaded once from FROZEN_ACTOR_PATH/FROZEN_CRITIC_PATH
and held fixed), so there's nothing per-step to extract for them -- only
comm_state.params varies across checkpoints, matching what
visualize_macro_mappo_rollout_comm.py already loads from best_actor.safetensors
/final_actor.safetensors for this run type.

This script rebuilds the same runner pytree structure used at comm-training
init (without actually training), uses it to unflatten each checkpoint file,
and saves just comm_state.params as a safetensors file per step.
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
import optax
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

from jaxmarl.wrappers.baselines import save_params

from baselines.MAPPO.mappo_macro_common import Actor, Critic, build_env, initialize_config
from baselines.MAPPO.mappo_macro_every_step_comm import (
    CommModule,
    load_frozen_macro_params,
)

CHECKPOINT_RE = re.compile(r"checkpoint_(\d+)\.npz$")


def parse_args():
    parser = argparse.ArgumentParser()
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


def build_initial_runner(config, seed):
    """Reconstruct the exact runner pytree structure used at comm-training init.

    Only shapes matter here -- comm_module.init, the frozen TrainStates, and
    env.reset just need to produce a pytree with the same structure/shapes as
    the one maybe_checkpoint flattened, so the saved leaves can be unflattened
    back into it. The actual values are immediately discarded and replaced by
    the checkpoint's arrays.
    """
    env = build_env(config)
    config = initialize_config(config, env)

    if len(env.agents) != 2:
        raise ValueError("mappo_macro_every_step_comm.py only supports exactly 2 agents.")

    critic = Critic(int(config["HIDDEN_SIZE"]))
    comm_module = CommModule(
        hidden_size=int(config.get("COMM_HIDDEN_SIZE", config["HIDDEN_SIZE"])),
        vocab_size=int(config["VOCAB_SIZE"]),
        action_dim=env.num_actions,
        message_embed_dim=int(config.get("MESSAGE_EMBED_DIM", 8)),
    )

    # Real weights (needed for correct array shapes/dtypes), but never
    # trained here -- only used to build a frozen_critic_state to unflatten
    # into, matching training exactly.
    _frozen_actor_params, frozen_critic_params = load_frozen_macro_params(config)

    obs_dim = env.observation_space(env.agents[0]).shape[0]
    rng = jax.random.PRNGKey(seed)
    rng, comm_rng = jax.random.split(rng)
    dummy_obs = jnp.zeros((1, obs_dim))
    dummy_message = jnp.zeros((1,), dtype=jnp.int32)
    comm_params = comm_module.init(comm_rng, dummy_obs, dummy_message)

    comm_state = TrainState.create(
        apply_fn=comm_module.apply,
        params=comm_params,
        tx=optax.chain(
            optax.clip_by_global_norm(config.get("MAX_GRAD_NORM", 0.5)),
            optax.adam(config["LR"], eps=1e-5),
        ),
    )
    frozen_critic_state = TrainState.create(
        apply_fn=critic.apply,
        params=frozen_critic_params,
        tx=optax.set_to_zero(),
    )

    rng, reset_rng = jax.random.split(rng)
    reset_keys = jax.random.split(reset_rng, int(config["NUM_ENVS"]))
    obs, env_state = jax.vmap(env.reset)(reset_keys)

    return (comm_state, frozen_critic_state, env_state, obs, rng)


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


def extract_comm_params(checkpoint_path: Path, target_leaves, tree_definition):
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        restored_leaves = [archive[f"arr_{i}"] for i in range(len(archive.files))]
    if len(restored_leaves) != len(target_leaves):
        raise ValueError(
            f"{checkpoint_path.name}: checkpoint structure does not match the "
            "reconstructed runner (config.yaml doesn't match the run that "
            "produced this checkpoint, or FROZEN_ACTOR_PATH/FROZEN_CRITIC_PATH "
            "point at different weights than the ones this run trained against)"
        )
    for restored_leaf, target_leaf in zip(restored_leaves, target_leaves):
        if restored_leaf.shape != np.asarray(target_leaf).shape:
            raise ValueError(
                f"{checkpoint_path.name}: checkpoint array shapes do not match "
                "the reconstructed runner"
            )
    restored = jax.tree.unflatten(tree_definition, restored_leaves)
    comm_state = restored[0]
    return comm_state.params


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

    initial_runner = build_initial_runner(config, args.seed)
    target_leaves, tree_definition = jax.tree.flatten(initial_runner)

    for step, checkpoint_path in checkpoints.items():
        comm_params = extract_comm_params(checkpoint_path, target_leaves, tree_definition)
        output_path = output_dir / f"actor_{step:08d}.safetensors"
        save_params(comm_params, output_path)
        print(f"{checkpoint_path.name} -> {output_path}")

    print(f"\nExtracted {len(checkpoints)} comm-module checkpoint(s) to {output_dir}")


if __name__ == "__main__":
    main()
