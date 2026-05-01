"""Safe-subtree conditional shared-estimator variants for Risky-PS.

These variants preserve Risky-PS selection, risky edge updates, aggregate-mass
safe-prefix sampling, and barrier-limited delta propagation. They only change
the shared leaf estimator denominator:

- ``RiskyPSSafeConditionalPolicy`` uses the selected leaf probability
  conditional on the highest upload-reachable safe subtree root.
- ``RiskyPSSafeConditionalIXPolicy`` adds an Exp3-IX smoothing term to that
  conditional probability.
"""

from __future__ import annotations

from typing import Any

from fixed_tree_env import EpisodeResult
from risky_ps import LeafKey, PrefixKey, RiskyPSPolicy


class RiskyPSSafeConditionalPolicy(RiskyPSPolicy):
    """Risky-PS with safe-subtree-local shared leaf estimation."""

    shared_estimator_variant = "safe_conditional"
    shared_denominator_mode = "safe_subtree_conditional_prob"

    @property
    def name(self) -> str:
        return "risky_ps_safe_conditional"

    def _safe_subtree_root_for_leaf(self, leaf: LeafKey) -> tuple[PrefixKey, str]:
        """Return the highest upload-reachable safe prefix for the selected leaf.

        The prefix must be a parent on one of the leaf's upload edges. Among all
        such safe prefixes, the shortest prefix is the highest safe subtree root.
        If none is available, the estimator falls back to the global path
        probability and records this explicitly.
        """

        candidates = [
            prefix
            for prefix, _child_prefix in self.upload_edges_by_leaf.get(leaf, [])
            if self.safe_prefixes.get(prefix, False)
        ]
        if not candidates:
            return (), "fallback_global_no_upload_reachable_safe_prefix"
        return min(candidates, key=len), "highest_upload_reachable_safe_prefix"

    def _prefix_reach_prob(self, prefix: PrefixKey) -> float:
        if not prefix:
            return 1.0
        for edge_info in self.last_sampled_edges:
            if edge_info.get("prefix") == prefix:
                return float(
                    edge_info.get(
                        "prefix_reach_prob",
                        edge_info.get("path_prob_so_far", 1.0),
                    )
                )
        # If the prefix is the leaf itself, all sampled edges have already been
        # multiplied into last_path_prob.
        selected_leaf = tuple(self.last_sampled_edges[-1]["child_prefix"]) if self.last_sampled_edges else ()
        if prefix == selected_leaf:
            return float(self.last_path_prob)
        return 1.0

    def _shared_conditional_denominator(self, leaf: LeafKey) -> dict[str, Any]:
        safe_root, root_mode = self._safe_subtree_root_for_leaf(leaf)
        global_prob = max(float(self.last_path_prob), 1e-12)
        root_reach_prob = max(self._prefix_reach_prob(safe_root), 1e-12)
        if root_mode.startswith("fallback_global"):
            conditional_prob = global_prob
        else:
            conditional_prob = max(global_prob / root_reach_prob, 1e-12)
        return {
            "shared_estimator_variant": self.shared_estimator_variant,
            "shared_denominator_mode": self.shared_denominator_mode,
            "shared_leaf_global_prob": global_prob,
            "shared_leaf_conditional_prob": conditional_prob,
            "safe_subtree_root": list(safe_root),
            "safe_subtree_root_mode": root_mode,
            "safe_subtree_root_reach_prob": root_reach_prob,
            "shared_denominator": conditional_prob,
        }

    def _shared_leaf_estimated_loss(
        self,
        episode_result: EpisodeResult,
        *,
        observed_loss: float | None = None,
    ) -> float:
        leaf = tuple(episode_result.selected_path)
        estimator_info = self._shared_conditional_denominator(leaf)
        estimated_loss = self._importance_weighted_loss(
            observed_loss=(
                observed_loss
                if observed_loss is not None
                else self._observed_loss(episode_result)
            ),
            denominator=float(estimator_info["shared_denominator"]),
        )
        self._last_shared_estimator_info = {
            **estimator_info,
            "shared_leaf_estimated_loss": estimated_loss,
        }
        return estimated_loss

    def update(self, episode_result: EpisodeResult) -> None:
        self._last_shared_estimator_info = {}
        super().update(episode_result)
        self.last_update_info = {
            **self.last_update_info,
            "update_type": "risky_ps_safe_conditional_layered_barrier_theta_loss",
            "shared_estimator_variant": self.shared_estimator_variant,
            "shared_denominator_mode": self.shared_denominator_mode,
            **getattr(self, "_last_shared_estimator_info", {}),
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state.update(
            {
                "update_type": "risky_ps_safe_conditional_layered_barrier_theta_loss",
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
                "update_type": "risky_ps_safe_conditional_layered_barrier_theta_loss",
                "shared_estimator_variant": self.shared_estimator_variant,
                "shared_denominator_mode": self.shared_denominator_mode,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return info


class RiskyPSSafeConditionalIXPolicy(RiskyPSSafeConditionalPolicy):
    """Safe-conditional Risky-PS with Exp3-IX smoothing."""

    shared_estimator_variant = "safe_conditional_ix"
    shared_denominator_mode = "safe_subtree_conditional_prob_plus_gamma"

    def __init__(
        self,
        seed: int = 0,
        eta: float = 0.2,
        eta_shared: float | None = 0.05,
        epsilon: float = 0.1,
        loss_clip: float | None = None,
        prob_floor: float = 0.0,
        gamma_shared: float = 0.0005,
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
        return "risky_ps_safe_conditional_ix"

    def _shared_conditional_denominator(self, leaf: LeafKey) -> dict[str, Any]:
        estimator_info = super()._shared_conditional_denominator(leaf)
        conditional_prob = float(estimator_info["shared_leaf_conditional_prob"])
        estimator_info.update(
            {
                "shared_estimator_variant": self.shared_estimator_variant,
                "shared_denominator_mode": self.shared_denominator_mode,
                "gamma_shared": self.gamma_shared,
                "shared_denominator": conditional_prob + self.gamma_shared,
            }
        )
        return estimator_info

    def update(self, episode_result: EpisodeResult) -> None:
        super().update(episode_result)
        self.last_update_info = {
            **self.last_update_info,
            "update_type": "risky_ps_safe_conditional_ix_layered_barrier_theta_loss",
            "shared_estimator_variant": self.shared_estimator_variant,
            "shared_denominator_mode": self.shared_denominator_mode,
            "gamma_shared": self.gamma_shared,
        }

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state.update(
            {
                "update_type": "risky_ps_safe_conditional_ix_layered_barrier_theta_loss",
                "gamma_shared": self.gamma_shared,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info.update(
            {
                "update_type": "risky_ps_safe_conditional_ix_layered_barrier_theta_loss",
                "gamma_shared": self.gamma_shared,
                "last_update_info": dict(self.last_update_info),
            }
        )
        return info
