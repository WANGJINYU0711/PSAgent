"""Oracle baseline that selects the best fixed path per instance."""

from __future__ import annotations

from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import find_best_fixed_path


class OraclePolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")
        self.last_best_cost: float | None = None

    @property
    def name(self) -> str:
        return "oracle"

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        if not self.stage_agent_ids:
            self.bind_env(env)
        best_path, best_result = find_best_fixed_path(instance, env)
        self.last_best_cost = float(best_result.total_cost)
        return list(best_path)

    def update(self, episode_result: EpisodeResult) -> None:
        del episode_result

    def get_state(self) -> dict[str, Any]:
        return {
            "protocol_mode": self.protocol_mode,
            "last_best_cost": self.last_best_cost,
        }
