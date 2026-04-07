"""Structure-unaware whole-path learner for mixed disclosure catalogs."""

from __future__ import annotations

from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import enumerate_all_paths


class NaiveMixedPolicy(BasePolicy):
    """Treat each full path as an arm and learn without tree awareness."""

    def __init__(self, seed: int = 0, epsilon: float = 0.2) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")
        self.epsilon = epsilon
        self.paths: list[list[str]] = []
        self.counts: dict[tuple[str, ...], int] = {}
        self.total_costs: dict[tuple[str, ...], float] = {}

    @property
    def name(self) -> str:
        return "naive_mixed"

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.paths = enumerate_all_paths(env)

    def _mean_cost(self, path_key: tuple[str, ...]) -> float:
        count = self.counts.get(path_key, 0)
        if count == 0:
            return 0.0
        return self.total_costs[path_key] / count

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.paths:
            self.bind_env(env)
        if self.rng.random() < self.epsilon:
            return list(self.rng.choice(self.paths))

        best_path = min(self.paths, key=lambda path: (self._mean_cost(tuple(path)), self.counts.get(tuple(path), 0)))
        return list(best_path)

    def update(self, episode_result: EpisodeResult) -> None:
        key = tuple(episode_result.selected_path)
        self.counts[key] = self.counts.get(key, 0) + 1
        self.total_costs[key] = self.total_costs.get(key, 0.0) + episode_result.total_cost

    def get_state(self) -> dict[str, Any]:
        return {
            "protocol_mode": self.protocol_mode,
            "epsilon": self.epsilon,
            "num_paths": len(self.paths),
            "visited_paths": len(self.counts),
        }
