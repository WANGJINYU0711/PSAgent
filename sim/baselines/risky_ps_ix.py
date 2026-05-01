"""Exp3-IX shared-leaf variant of Risky-PS.

This variant preserves the current barrier-style Risky-PS behavior and only
changes the shared leaf importance-weighted estimator from ``1 / path_prob`` to
``1 / (path_prob + gamma_shared)``.
"""

from __future__ import annotations

from typing import Any

from fixed_tree_env import EpisodeResult
from risky_ps import RiskyPSPolicy


class RiskyPSIXPolicy(RiskyPSPolicy):
    """Risky-PS with an Exp3-IX style shared-leaf estimator."""

    def __init__(
        self,
        seed: int = 0,
        eta: float = 0.2,
        eta_shared: float | None = 0.05,
        epsilon: float = 0.1,
        loss_clip: float | None = None,
        prob_floor: float = 0.0,
        gamma_shared: float = 0.005,
    ) -> None:
        super().__init__(
            seed=seed,
            eta=eta,
            eta_shared=eta_shared,
            epsilon=epsilon,
            loss_clip=loss_clip,
            prob_floor=prob_floor,
        )
        self.gamma_shared = max(0.0, float(gamma_shared))

    @property
    def name(self) -> str:
        return "risky_ps_ix"

    def _shared_leaf_estimated_loss(
        self,
        episode_result: EpisodeResult,
        *,
        observed_loss: float | None = None,
    ) -> float:
        return self._importance_weighted_loss(
            observed_loss=(
                observed_loss
                if observed_loss is not None
                else self._observed_loss(episode_result)
            ),
            denominator=self.last_path_prob + self.gamma_shared,
        )

    def update(self, episode_result: EpisodeResult) -> None:
        super().update(episode_result)
        self.last_update_info = {
            **self.last_update_info,
            "update_type": "risky_ps_ix_layered_barrier_theta_loss",
            "shared_estimator_variant": "exp3_ix",
            "gamma_shared": self.gamma_shared,
            "shared_leaf_sampling_prob": self.last_path_prob,
            "shared_leaf_estimated_loss": self.last_update_info.get(
                "shared_leaf_estimated_loss"
            ),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state.update(
            {
                "update_type": "risky_ps_ix_layered_barrier_theta_loss",
                "gamma_shared": self.gamma_shared,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info.update(
            {
                "update_type": "risky_ps_ix_layered_barrier_theta_loss",
                "gamma_shared": self.gamma_shared,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return info
