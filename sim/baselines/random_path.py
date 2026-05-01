"""Uniform random-path baseline."""

from __future__ import annotations

from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment


class RandomPathPolicy(BasePolicy):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")

    @property
    def name(self) -> str:
        return "random_path"

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.stage_agent_ids:
            self.bind_env(env)
        path: list[str] = []
        current_prefix: tuple[str, ...] = ()
        for stage_name in env.STAGE_NAMES:
            legal_agent_ids = self._legal_agent_ids_for_prefix(current_prefix, stage_name, env)
            chosen = self.rng.choice(legal_agent_ids)
            path.append(chosen)
            current_prefix = current_prefix + (chosen,)
        return path

    def update(self, episode_result: EpisodeResult) -> None:
        del episode_result
