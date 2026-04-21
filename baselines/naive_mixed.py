"""Naive mixed baseline with only direct parent-child feedback.

This policy does not use any shared/unshared structure and does not maintain
whole-path arms. Instead, each prefix locally learns average end-to-end cost
for the child edge it selected. Every parent therefore only updates its own
chosen child interface; there is no non-local shared delta propagation.
"""

from __future__ import annotations

from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import enumerate_all_paths


class NaiveMixedPolicy(BasePolicy):
    """Prefix-local epsilon-greedy baseline without shared propagation."""

    def __init__(self, seed: int = 0, epsilon: float = 0.2) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")
        self.epsilon = epsilon
        self.paths: list[list[str]] = []
        self.counts: dict[tuple[tuple[str, ...], tuple[str, ...]], int] = {}
        self.total_costs: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}

    @property
    def name(self) -> str:
        return "naive_mixed"

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.paths = enumerate_all_paths(env)

    def _mean_cost(
        self,
        edge_key: tuple[tuple[str, ...], tuple[str, ...]],
    ) -> float:
        count = self.counts.get(edge_key, 0)
        if count == 0:
            return 0.0
        return self.total_costs[edge_key] / count

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

            if self.rng.random() < self.epsilon:
                chosen_child = self.rng.choice(child_prefixes)
            else:
                chosen_child = min(
                    child_prefixes,
                    key=lambda child_prefix: (
                        self._mean_cost((current_prefix, child_prefix)),
                        self.counts.get((current_prefix, child_prefix), 0),
                    ),
                )

            selected_path.append(chosen_child[-1])
            current_prefix = chosen_child
        return selected_path

    def update(self, episode_result: EpisodeResult) -> None:
        leaf = tuple(episode_result.selected_path)
        for depth in range(len(leaf)):
            prefix = tuple(leaf[:depth])
            child_prefix = tuple(leaf[: depth + 1])
            edge_key = (prefix, child_prefix)
            self.counts[edge_key] = self.counts.get(edge_key, 0) + 1
            self.total_costs[edge_key] = self.total_costs.get(edge_key, 0.0) + episode_result.total_cost

    def get_state(self) -> dict[str, Any]:
        return {
            "protocol_mode": self.protocol_mode,
            "epsilon": self.epsilon,
            "num_paths": len(self.paths),
            "visited_edges": len(self.counts),
        }
