"""Equal-child shared-init PS variant with parent-conserved shared edge mass."""

from __future__ import annotations

from typing import Any

from risky_ps import EdgeKey, PrefixKey
from risky_ps_const_init import RiskyPSConstInitPolicy


class RiskyPSConstInitConservePolicy(RiskyPSConstInitPolicy):
    """Constant-init Risky-PS with barrier-limited delta and parent conservation.

    This keeps the existing sampled-leaf ``delta_t`` upload semantics: only the
    sampled shared leaf uploads along its barrier-limited path.  After each
    selected parent-child edge receives that delta, the sibling masses under the
    same parent are rescaled back to the equal-child initial total.  This avoids
    the all-children-to-floor failure mode without adding extra leaf visibility.
    """

    def __init__(
        self,
        *args: Any,
        conserve_parent_mass: bool = True,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("shared_edge_floor", 0.001)
        super().__init__(*args, **kwargs)
        self.conserve_parent_mass = bool(conserve_parent_mass)
        self.last_conservation_updates: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "risky_ps_const_init_conserve"

    def _parent_child_edges(self, prefix: PrefixKey) -> list[EdgeKey]:
        return sorted(
            edge
            for edge in self.shared_edge_mass
            if edge[0] == prefix and self.shared_reachable_leaf_count.get(edge, 0) > 0
        )

    def _target_parent_mass_total(self, child_edges: list[EdgeKey]) -> float:
        return len(child_edges) * self._constant_shared_edge_mass()

    def _renormalize_parent_mass(self, prefix: PrefixKey) -> dict[str, Any] | None:
        child_edges = self._parent_child_edges(prefix)
        if not child_edges:
            return None

        floor = self.shared_edge_floor
        before = {edge: float(self.shared_edge_mass.get(edge, 0.0)) for edge in child_edges}
        floored = {edge: max(floor, value) for edge, value in before.items()}
        current_total = sum(floored.values())
        target_total = self._target_parent_mass_total(child_edges)
        min_total = floor * len(child_edges)
        if target_total < min_total:
            target_total = min_total

        if current_total <= 0.0:
            equal_mass = target_total / len(child_edges)
            after = {edge: equal_mass for edge in child_edges}
        else:
            scale = target_total / current_total
            after = {edge: max(floor, value * scale) for edge, value in floored.items()}
            after_total = sum(after.values())
            if after_total > 0.0 and abs(after_total - target_total) > 1e-12:
                rescale = target_total / after_total
                after = {edge: max(floor, value * rescale) for edge, value in after.items()}

        for edge, value in after.items():
            self.shared_edge_mass[edge] = value
            self.shared_edge_weight[edge] = value

        return {
            "prefix": list(prefix),
            "child_count": len(child_edges),
            "target_total": target_total,
            "floor": floor,
            "before_total": sum(before.values()),
            "after_total": sum(after.values()),
            "child_masses_before": [
                {"child_prefix": list(edge[1]), "mass": before[edge]}
                for edge in child_edges
            ],
            "child_masses_after": [
                {"child_prefix": list(edge[1]), "mass": after[edge]}
                for edge in child_edges
            ],
        }

    def _propagate_barrier_limited_shared_delta(
        self,
        leaf: tuple[str, ...],
        delta: float,
    ) -> tuple[list[EdgeKey], PrefixKey | None]:
        touched_edges = list(self.upload_edges_by_leaf.get(leaf, []))
        self.last_conservation_updates = []
        floor = self.shared_edge_floor

        for edge in touched_edges:
            old_mass = float(self.shared_edge_mass.get(edge, 0.0))
            updated_mass = max(floor, old_mass + delta)
            self.shared_edge_mass[edge] = updated_mass
            self.shared_edge_weight[edge] = updated_mass
            if self.conserve_parent_mass:
                info = self._renormalize_parent_mass(edge[0])
                if info is not None:
                    info["selected_child_prefix"] = list(edge[1])
                    info["selected_mass_after_delta_before_renorm"] = updated_mass
                    self.last_conservation_updates.append(info)

        return touched_edges, self.barrier_stop_prefix_by_leaf.get(leaf)

    def update(self, episode_result: Any) -> None:
        super().update(episode_result)
        if isinstance(self.last_update_info, dict):
            self.last_update_info["shared_parent_conservation_enabled"] = self.conserve_parent_mass
            self.last_update_info["shared_edge_floor"] = self.shared_edge_floor
            self.last_update_info["shared_parent_conservation_updates"] = list(
                self.last_conservation_updates
            )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["shared_parent_conservation_enabled"] = self.conserve_parent_mass
        state["shared_edge_floor"] = self.shared_edge_floor
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info["shared_parent_conservation_enabled"] = self.conserve_parent_mass
        info["shared_edge_floor"] = self.shared_edge_floor
        return info
