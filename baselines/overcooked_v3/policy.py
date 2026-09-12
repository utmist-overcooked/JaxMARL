"""Environment-facing policy contract for Overcooked V3 rollouts."""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import chex


class RolloutPolicy(Protocol):
    """Small adapter required by the shared rollout runner.

    Network modules may return environment actions directly or return a
    distribution. The shared runner uses direct actions unchanged and calls
    ``mode()`` only when the returned object is a distribution.
    """

    def initial_state(self, env) -> Any:
        """Return the policy state used at the start of an episode."""

    def act(
        self,
        params,
        obs: Mapping[str, chex.Array],
        policy_state: Any,
        dones: Mapping[str, chex.Array],
        rng: chex.PRNGKey,
    ) -> tuple[Any, Any]:
        """Return actions or a distribution, plus the next policy state."""
