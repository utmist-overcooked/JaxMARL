"""Common checkpoint callback for Overcooked V3 training runs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from baselines.overcooked_v3.gif_logging import GifLogResult
from baselines.overcooked_v3.hooks import (
    EnvironmentFactory,
    PolicyFactory,
    RolloutGifHook,
)


@dataclass(frozen=True)
class OvercookedV3Training:
    """Attach shared GIF behavior to a model's checkpoint save events."""

    config: Mapping[str, Any]
    policy_factory: PolicyFactory
    environment_factory: EnvironmentFactory | None = None
    gif_hook: RolloutGifHook = field(init=False)

    def __post_init__(self):
        """Create and validate the checkpoint GIF schedule immediately."""

        object.__setattr__(
            self,
            "gif_hook",
            RolloutGifHook(
                self.config,
                self.policy_factory,
                self.environment_factory,
            ),
        )

    def checkpoint_saved(
        self,
        params,
        *,
        checkpoint_index: int,
        update_step: int,
        env_step: int,
        training_seed: int,
        run_name: str | None = None,
    ) -> GifLogResult | None:
        """Render a GIF when this saved checkpoint is selected by the schedule."""

        if not self.gif_hook.should_record(checkpoint_index):
            return None
        try:
            return self.gif_hook.record(
                params,
                checkpoint_index=checkpoint_index,
                update_step=update_step,
                env_step=env_step,
                training_seed=training_seed,
                run_name=run_name,
            )
        except Exception as exc:
            if self.config.get("ROLLOUT_GIF_STRICT", False):
                raise
            print(f"Failed to save rollout GIF: {exc}")
            return None

    def checkpoint_saved_jax(
        self,
        params,
        *,
        checkpoint_index,
        update_step,
        env_step,
        original_seed,
    ):
        """Schedule the same checkpoint callback from compiled JAX code."""

        self.gif_hook.maybe_record(
            checkpoint_index,
            update_step,
            env_step,
            original_seed,
            params,
        )
