"""Core reusable dataclasses for tree-family experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TaskDescriptor:
    task_id: str
    attribute_weights: dict[int, float]
    stage_difficulty: dict[str, float]


@dataclass
class AgentSpec:
    agent_id: str
    g: int
    base_cost: float
    competence_level: str
    scope_level: str
    stability_level: str
    attribute_skill: dict[int, float]


@dataclass
class FamilySpec:
    family_name: str
    stages: list[str]
    stage_agents: dict[str, list[str]]
