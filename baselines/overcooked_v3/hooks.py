"""Training hook for milestone Overcooked V3 rollout GIFs."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

import jaxmarl
from baselines.overcooked_v3.gif_logging import (
    GifLogResult,
    checkpoint_gif_indices,
    save_rollout_gif,
)
from baselines.overcooked_v3.policy import RolloutPolicy
from baselines.overcooked_v3.rollout import rollout_episode


PolicyFactory = Callable[[Any], RolloutPolicy]
EnvironmentFactory = Callable[[Mapping[str, Any]], Any]


def select_seed_params(params, original_seed, seed_index: int):
    """Select one seed from values that may have been batched by ``vmap``."""

    seed_values = np.asarray(original_seed).reshape(-1)
    if seed_values.size <= 1:
        return params, int(seed_values[0])

    selected_index = min(max(int(seed_index), 0), seed_values.size - 1)
    selected_params = jax.tree.map(
        lambda x: x[selected_index]
        if hasattr(x, "shape") and x.ndim > 0 and x.shape[0] == seed_values.size
        else x,
        params,
    )
    return selected_params, int(seed_values[selected_index])


@dataclass(frozen=True)
class RolloutGifHook:
    """Schedule host-side rollout GIFs from a JIT-compiled training loop."""

    config: Mapping[str, Any]
    policy_factory: PolicyFactory
    environment_factory: EnvironmentFactory | None = None
    checkpoint_indices: tuple[int, ...] = field(init=False)
    target_training_seed: int = field(init=False)
    training_seed_index: int = field(init=False)
    rollout_seed: int = field(init=False)

    def __post_init__(self):
        """Validate checkpoint selection and vectorized-seed metadata."""

        if (
            self.config.get("ENV_NAME") != "overcooked_v3"
            and self.environment_factory is None
        ):
            raise ValueError(
                "Non-primitive V3 environments require an environment factory"
            )

        checkpoint_indices = ()
        if self.config.get("ROLLOUT_GIF_ENABLED", False):
            checkpoint_indices = checkpoint_gif_indices(
                self.config.get("NUM_CHECKPOINTS", 0),
                self.config.get("ROLLOUT_GIF_COUNT", 0),
            )
        num_seeds = max(int(self.config.get("NUM_SEEDS", 1)), 1)
        seed_index = min(
            max(int(self.config.get("ROLLOUT_GIF_SEED_INDEX", 0)), 0),
            num_seeds - 1,
        )
        rngs = jax.random.split(
            jax.random.PRNGKey(int(self.config["SEED"])),
            num_seeds,
        )

        object.__setattr__(self, "checkpoint_indices", checkpoint_indices)
        object.__setattr__(self, "training_seed_index", seed_index)
        object.__setattr__(self, "target_training_seed", int(rngs[seed_index][0]))
        object.__setattr__(
            self,
            "rollout_seed",
            int(self.config.get("ROLLOUT_GIF_ENV_SEED", 0)),
        )

    @property
    def enabled(self) -> bool:
        """Return whether this run has at least one enabled GIF milestone."""

        return bool(self.checkpoint_indices)

    def should_record(self, checkpoint_index: int) -> bool:
        """Return whether this one-based saved checkpoint needs a GIF."""

        return self.enabled and int(checkpoint_index) in self.checkpoint_indices

    def record(
        self,
        params,
        *,
        checkpoint_index: int,
        update_step: int,
        env_step: int,
        training_seed: int,
        run_name: str | None = None,
    ) -> GifLogResult:
        """Record a GIF directly; useful for tests and offline evaluation."""

        env = (
            self.environment_factory(self.config)
            if self.environment_factory is not None
            else jaxmarl.make(
                self.config["ENV_NAME"],
                **self.config.get("ENV_KWARGS", {}),
            )
        )
        policy = self.policy_factory(env)
        max_steps = int(
            self.config.get(
                "ROLLOUT_GIF_MAX_STEPS",
                self.config.get("ENV_KWARGS", {}).get("max_steps", 400),
            )
        )
        episode = rollout_episode(
            env,
            policy,
            params,
            seed=self.rollout_seed,
            max_steps=max_steps,
        )
        result = save_rollout_gif(
            episode,
            env,
            root_dir=self.config.get("ROLLOUT_GIF_DIR", "rollouts"),
            run_name=(
                run_name
                or self.config.get("RUN_NAME")
                or self.config.get("WANDB_NAME")
            ),
            checkpoint_index=checkpoint_index,
            update_step=update_step,
            env_step=env_step,
            training_seed=training_seed,
            rollout_seed=self.rollout_seed,
            wandb_mode=self.config.get("WANDB_MODE"),
            wandb_key=self.config.get("ROLLOUT_GIF_WANDB_KEY", "rollouts/policy"),
        )
        upload_message = " and uploaded to W&B" if result.uploaded else ""
        print(
            f"Saved {result.episode_length}-step rollout GIF{upload_message}: "
            f"{result.path}"
        )
        return result

    def _record_callback(
        self,
        checkpoint_index,
        update_step,
        env_step,
        original_seed,
        params,
    ):
        """Move callback values to the host and render the selected seed."""

        try:
            selected_params, training_seed = select_seed_params(
                params,
                original_seed,
                self.training_seed_index,
            )
            if int(self.config.get("NUM_SEEDS", 1)) > 1 and (
                training_seed != self.target_training_seed
            ):
                return
            if int(self.config.get("NUM_SEEDS", 1)) == 1:
                training_seed = int(self.config["SEED"])

            run_name = self.config.get("RUN_NAME") or self.config.get("WANDB_NAME")
            if self.config.get("WANDB_MODE") != "disabled":
                try:
                    import wandb

                    if wandb.run is not None:
                        run_name = wandb.run.name
                except ImportError:
                    pass

            self.record(
                selected_params,
                checkpoint_index=int(
                    np.asarray(checkpoint_index).reshape(-1)[0]
                ),
                update_step=int(np.asarray(update_step).reshape(-1)[0]),
                env_step=int(np.asarray(env_step).reshape(-1)[0]),
                training_seed=training_seed,
                run_name=run_name,
            )
        except Exception as exc:
            if self.config.get("ROLLOUT_GIF_STRICT", False):
                raise
            print(f"Failed to save rollout GIF: {exc}")

    def maybe_record(
        self,
        checkpoint_index,
        update_step,
        env_step,
        original_seed,
        params,
    ):
        """Emit a host callback for a selected checkpoint save event."""

        if not self.enabled:
            return

        checkpoint_array = jnp.asarray(
            self.checkpoint_indices,
            dtype=jnp.int32,
        )

        def record_rollout(_):
            """Schedule the host-side renderer for a matching milestone."""

            jax.debug.callback(
                self._record_callback,
                checkpoint_index,
                update_step,
                env_step,
                original_seed,
                params,
            )
            return jnp.array(0, dtype=jnp.int32)

        jax.lax.cond(
            jnp.any(checkpoint_index == checkpoint_array),
            record_rollout,
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=None,
        )
