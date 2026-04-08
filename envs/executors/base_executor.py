"""Base executor interface for family-mode environments."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from tree_family.specs import AgentSpec, TaskDescriptor


class BaseExecutor(ABC):
    def __init__(self, stages: list[str], seed: int = 0) -> None:
        self.stages = stages
        self.seed = seed

    @abstractmethod
    def run_path(
        self,
        task: TaskDescriptor,
        path: list[str],
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        ...
