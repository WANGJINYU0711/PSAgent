"""Equal-child shared-init PS variant with natural shared edge decay.

This variant keeps the constant shared-edge initialization from
``RiskyPSConstInitPolicy`` but removes fixed-total parent conservation. Shared
edge masses are allowed to decay naturally under the sampled leaf delta. If a
sibling group gets close to numerical underflow, all masses in that group are
rescaled by the same positive factor, preserving child probabilities exactly up
to floating point precision.
"""

from __future__ import annotations

from typing import Any

from risky_ps import EdgeKey, PrefixKey
from risky_ps_const_init import RiskyPSConstInitPolicy


class RiskyPSConstInitNaturalDecayPolicy(RiskyPSConstInitPolicy):
    """Constant-init Risky-PS with ratio-preserving group rescale only."""

    def __init__(
        self,
        *args: Any,
        shared_edge_floor: float = 1e-300,
        shared_group_rescale_trigger: float = 1e-80,
        shared_group_rescale_target: float = 1.0,
        **kwargs: Any,
    ) -> None:
        kwargs.setdefault("shared_edge_floor", shared_edge_floor)
        super().__init__(*args, **kwargs)
        self.shared_group_rescale_trigger = float(shared_group_rescale_trigger)
        self.shared_group_rescale_target = float(shared_group_rescale_target)
        self.last_group_rescale_updates: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return "risky_ps_const_init_natural_decay"

    def _parent_child_edges(self, prefix: PrefixKey) -> list[EdgeKey]:
        return sorted(
            edge
            for edge in self.shared_edge_mass
            if edge[0] == prefix and self.shared_reachable_leaf_count.get(edge, 0) > 0
        )

    def _rescale_parent_group_if_needed(self, prefix: PrefixKey) -> dict[str, Any] | None:
        child_edges = self._parent_child_edges(prefix)
        if not child_edges:
            return None

        before = {edge: float(self.shared_edge_mass.get(edge, 0.0)) for edge in child_edges}
        positive_values = [value for value in before.values() if value > 0.0]
        floor = max(0.0, float(self.shared_edge_floor))

        if not positive_values:
            reset_mass = max(floor, float(self.shared_group_rescale_target))
            after = {edge: reset_mass for edge in child_edges}
            for edge, value in after.items():
                self.shared_edge_mass[edge] = value
                self.shared_edge_weight[edge] = value
            return {
                "prefix": list(prefix),
                "mode": "reset_equal_after_all_zero",
                "child_count": len(child_edges),
                "floor": floor,
                "rescale_trigger": self.shared_group_rescale_trigger,
                "rescale_target": self.shared_group_rescale_target,
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

        max_mass = max(positive_values)
        if max_mass >= float(self.shared_group_rescale_trigger):
            return None

        scale = float(self.shared_group_rescale_target) / max_mass
        after = {
            edge: max(floor, float(before[edge]) * scale)
            for edge in child_edges
        }
        for edge, value in after.items():
            self.shared_edge_mass[edge] = value
            self.shared_edge_weight[edge] = value

        return {
            "prefix": list(prefix),
            "mode": "ratio_preserving_rescale",
            "child_count": len(child_edges),
            "scale": scale,
            "floor": floor,
            "rescale_trigger": self.shared_group_rescale_trigger,
            "rescale_target": self.shared_group_rescale_target,
            "max_before": max_mass,
            "max_after": max(after.values(), default=0.0),
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
        self.last_group_rescale_updates = []
        floor = max(0.0, float(self.shared_edge_floor))

        for edge in touched_edges:
            old_mass = float(self.shared_edge_mass.get(edge, 0.0))
            updated_mass = max(floor, old_mass + float(delta))
            self.shared_edge_mass[edge] = updated_mass
            self.shared_edge_weight[edge] = updated_mass
            info = self._rescale_parent_group_if_needed(edge[0])
            if info is not None:
                info["selected_child_prefix"] = list(edge[1])
                info["selected_mass_after_delta_before_rescale"] = updated_mass
                self.last_group_rescale_updates.append(info)

        return touched_edges, self.barrier_stop_prefix_by_leaf.get(leaf)

    def update(self, episode_result: Any) -> None:
        super().update(episode_result)
        if isinstance(self.last_update_info, dict):
            self.last_update_info["shared_parent_conservation_enabled"] = False
            self.last_update_info["shared_natural_decay_enabled"] = True
            self.last_update_info["shared_group_rescale_trigger"] = (
                self.shared_group_rescale_trigger
            )
            self.last_update_info["shared_group_rescale_target"] = (
                self.shared_group_rescale_target
            )
            self.last_update_info["shared_group_rescale_updates"] = list(
                self.last_group_rescale_updates
            )

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["shared_parent_conservation_enabled"] = False
        state["shared_natural_decay_enabled"] = True
        state["shared_group_rescale_trigger"] = self.shared_group_rescale_trigger
        state["shared_group_rescale_target"] = self.shared_group_rescale_target
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info["shared_parent_conservation_enabled"] = False
        info["shared_natural_decay_enabled"] = True
        info["shared_group_rescale_trigger"] = self.shared_group_rescale_trigger
        info["shared_group_rescale_target"] = self.shared_group_rescale_target
        return info
