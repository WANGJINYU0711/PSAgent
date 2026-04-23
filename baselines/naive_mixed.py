"""Naive mixed baseline with only direct parent-child cost ordering.

This policy does not use any shared/unshared structure and does not maintain
whole-path arms. Each prefix locally stores the latest observed end-to-end cost
for each selected direct child edge. Selection greedily chooses the legal child
with the smallest stored edge cost, using deterministic prefix ordering to break
ties. There is no epsilon exploration, no historical averaging, no theta/EXP3
update, and no shared delta propagation.
"""

from __future__ import annotations

from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import enumerate_all_paths


class NaiveMixedPolicy(BasePolicy):
    """Prefix-local naive cost-ordering baseline without shared propagation."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")
        self.paths: list[list[str]] = []
        self.edge_costs: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}

    @property
    def name(self) -> str:
        return "naive_mixed"

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.paths = enumerate_all_paths(env)

    def _edge_cost(
        self,
        edge_key: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> float:
        return float(self.edge_costs.get(edge_key, 0.0))

    def _child_prefixes(
        self,
        current_prefix: tuple[str, ...],
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> list[tuple[str, ...]]:
        expected_depth = len(current_prefix)
        if expected_depth >= len(env.STAGE_NAMES):
            return []
        if env.STAGE_NAMES[expected_depth] != stage_name:
            raise ValueError(
                f"Prefix depth {expected_depth} expects stage {env.STAGE_NAMES[expected_depth]}, "
                f"but got {stage_name}."
            )

        child_prefixes: set[tuple[str, ...]] = set()
        for path in self.paths:
            leaf = tuple(path)
            if leaf[:expected_depth] != current_prefix:
                continue
            if len(leaf) <= expected_depth:
                continue
            child_prefixes.add(tuple(leaf[: expected_depth + 1]))
        return sorted(child_prefixes)

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.paths:
            self.bind_env(env)

        current_prefix: tuple[str, ...] = ()
        selected_path: list[str] = []
        for stage_name in env.STAGE_NAMES:
            child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
            if not child_prefixes:
                raise RuntimeError(
                    "NaiveMixedPolicy found no legal child prefixes. "
                    f"current_prefix={list(current_prefix)} stage_name={stage_name}"
                )

            chosen_child = min(
                child_prefixes,
                key=lambda child_prefix: (
                    self._edge_cost((current_prefix, child_prefix)),
                    child_prefix,
                ),
            )

            selected_path.append(chosen_child[-1])
            current_prefix = chosen_child
        return selected_path

    def update(self, episode_result: EpisodeResult) -> None:
        leaf = tuple(episode_result.selected_path)
        observed_cost = float(episode_result.total_cost)
        for depth in range(len(leaf)):
            prefix = tuple(leaf[:depth])
            child_prefix = tuple(leaf[: depth + 1])
            edge_key = (prefix, child_prefix)
            self.edge_costs[edge_key] = observed_cost

    def get_state(self) -> dict[str, Any]:
        return {
            "protocol_mode": self.protocol_mode,
            "selection_rule": "prefix_local_min_latest_edge_cost",
            "update_rule": "overwrite_selected_edges_with_observed_total_cost",
            "num_paths": len(self.paths),
            "tracked_edges": len(self.edge_costs),
        }
