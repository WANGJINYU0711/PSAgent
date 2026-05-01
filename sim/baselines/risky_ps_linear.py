"""Linear shared-mass upload variant of Risky-PS.

This variant keeps the risky / mixed-prefix logic identical to ``risky_ps`` and
only changes the full-share safe suffix semantics:

- each shared leaf maintains a positive linear mass ``Theta_l``
- safe-prefix child probabilities are proportional to summed descendant mass
- shared updates propagate additive mass deltas instead of exponentiated deltas
"""

from __future__ import annotations

from typing import Any

from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from risky_ps import EdgeKey, LeafKey, PrefixKey, RiskyPSPolicy


class RiskyPSLinearPolicy(RiskyPSPolicy):
    """Risky-PS with linear shared-safe mass propagation."""

    def __init__(
        self,
        seed: int = 0,
        eta: float = 0.2,
        eta_shared: float | None = 0.05,
        epsilon: float = 0.1,
        loss_clip: float | None = None,
        prob_floor: float = 0.0,
        shared_leaf_init: float = 1.0,
        shared_leaf_floor: float = 1e-12,
    ) -> None:
        super().__init__(
            seed=seed,
            eta=eta,
            eta_shared=eta_shared,
            epsilon=epsilon,
            loss_clip=loss_clip,
            prob_floor=prob_floor,
        )
        self.shared_leaf_init = max(shared_leaf_floor, float(shared_leaf_init))
        self.shared_leaf_floor = max(0.0, float(shared_leaf_floor))

    @property
    def name(self) -> str:
        return "risky_ps_linear"

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)

        # Reinitialize shared-leaf state as linear positive mass, then rebuild
        # every upload-reachable edge aggregate under the same barrier topology.
        for leaf in list(self.theta):
            self.theta[leaf] = (
                self.shared_leaf_init if self.leaf_types.get(leaf) == "shared" else 0.0
            )

        recomputed_edge_mass: dict[EdgeKey, float] = {}
        for leaf, upload_edges in self.upload_edges_by_leaf.items():
            if self.leaf_types.get(leaf) != "shared":
                continue
            leaf_mass = self._shared_leaf_weight(leaf)
            for edge in upload_edges:
                recomputed_edge_mass[edge] = recomputed_edge_mass.get(edge, 0.0) + leaf_mass

        for edge in list(self.shared_edge_mass):
            aggregate_mass = float(recomputed_edge_mass.get(edge, 0.0))
            self.shared_edge_mass[edge] = aggregate_mass
            self.shared_edge_weight[edge] = aggregate_mass

    def _shared_leaf_weight(self, leaf: LeafKey) -> float:
        return max(self.shared_leaf_floor, float(self.theta.get(leaf, self.shared_leaf_floor)))

    def _shared_edge_theta(self, edge: EdgeKey) -> float:
        return float(self.shared_edge_mass.get(edge, self.shared_edge_weight.get(edge, 0.0)))

    def _apply_shared_leaf_update(self, leaf: LeafKey, estimated_loss: float) -> float:
        old_mass = self._shared_leaf_weight(leaf)
        new_mass = max(self.shared_leaf_floor, old_mass - (self.eta_shared * estimated_loss))
        self.theta[leaf] = new_mass
        return new_mass - old_mass

    def _propagate_barrier_limited_shared_delta(
        self,
        leaf: LeafKey,
        delta: float,
    ) -> tuple[list[EdgeKey], PrefixKey | None]:
        touched_edges = list(self.upload_edges_by_leaf.get(leaf, []))
        for edge in touched_edges:
            updated_mass = max(self.shared_leaf_floor, self.shared_edge_mass.get(edge, 0.0) + delta)
            self.shared_edge_mass[edge] = updated_mass
            self.shared_edge_weight[edge] = updated_mass
        return touched_edges, self.barrier_stop_prefix_by_leaf.get(leaf)

    def update(self, episode_result: EpisodeResult) -> None:
        super().update(episode_result)
        self.last_update_info["update_type"] = "risky_ps_linear_layered_barrier_theta_loss"
        self.last_update_info["shared_mass_semantics"] = "linear_leaf_mass"
        self.last_update_info["shared_leaf_init"] = self.shared_leaf_init
        self.last_update_info["shared_leaf_floor"] = self.shared_leaf_floor

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["update_type"] = "risky_ps_linear_layered_barrier_theta_loss"
        state["shared_mass_semantics"] = "linear_leaf_mass"
        state["shared_leaf_init"] = self.shared_leaf_init
        state["shared_leaf_floor"] = self.shared_leaf_floor
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info["update_type"] = "risky_ps_linear_layered_barrier_theta_loss"
        info["shared_mass_semantics"] = "linear_leaf_mass"
        return info
