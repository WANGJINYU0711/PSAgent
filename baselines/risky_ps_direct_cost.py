"""Direct-observed-cost shared-leaf ablation for Risky-PS.

This variant preserves Risky-PS selection, risky edge updates, aggregate-mass
safe-prefix sampling, and barrier-limited delta propagation. It only changes
the shared leaf estimator from an importance-weighted loss to the observed
episode loss:

``loss_hat = loss``
"""

from __future__ import annotations

from typing import Any

from fixed_tree_env import EpisodeResult
from risky_ps import RiskyPSPolicy


class RiskyPSDirectCostPolicy(RiskyPSPolicy):
    """Risky-PS direct-cost shared update ablation."""

    shared_estimator_variant = "direct_cost"
    shared_denominator_mode = "none_observed_cost"

    @property
    def name(self) -> str:
        return "risky_ps_direct_cost"

    def _shared_leaf_estimated_loss(
        self,
        episode_result: EpisodeResult,
        *,
        observed_loss: float | None = None,
    ) -> float:
        observed = (
            observed_loss
            if observed_loss is not None
            else self._observed_loss(episode_result)
        )
        estimated_loss = max(0.0, float(observed))
        self._last_shared_estimator_info = {
            "shared_estimator_variant": self.shared_estimator_variant,
            "shared_denominator_mode": self.shared_denominator_mode,
            "shared_leaf_global_prob": float(self.last_path_prob),
            "shared_leaf_observed_loss": estimated_loss,
            "shared_leaf_estimated_loss": estimated_loss,
            "shared_denominator": None,
        }
        return estimated_loss

    def update(self, episode_result: EpisodeResult) -> None:
        self._last_shared_estimator_info = {}
        super().update(episode_result)
        self.last_update_info = {
            **self.last_update_info,
            "update_type": "risky_ps_direct_cost_layered_barrier_theta_loss",
            "shared_estimator_variant": self.shared_estimator_variant,
            "shared_denominator_mode": self.shared_denominator_mode,
            **getattr(self, "_last_shared_estimator_info", {}),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state.update(
            {
                "update_type": "risky_ps_direct_cost_layered_barrier_theta_loss",
                "shared_estimator_variant": self.shared_estimator_variant,
                "shared_denominator_mode": self.shared_denominator_mode,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info.update(
            {
                "update_type": "risky_ps_direct_cost_layered_barrier_theta_loss",
                "shared_estimator_variant": self.shared_estimator_variant,
                "shared_denominator_mode": self.shared_denominator_mode,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return info
