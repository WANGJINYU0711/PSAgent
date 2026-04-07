"""Catalog helpers for fixed-tree experiments.

The catalog is kept separate from baseline implementations so that disclosure
attribute ``g`` remains a property of the environment configuration rather than
an implicit proxy for agent quality.
"""

from __future__ import annotations

from typing import Iterable

from fixed_tree_env import AgentSpec, default_agent_catalog


def load_catalog(preset: str = "day3_default") -> list[AgentSpec]:
    """Return an agent catalog preset.

    Presets currently only change the disclosure attribute ``g`` while keeping
    the same agent identities, costs, and rule/simulated behavior.
    """

    base_catalog = default_agent_catalog()
    if preset in {"day3_default", "mixed"}:
        return base_catalog
    if preset == "all_share":
        return [_replace_g(agent, 0) for agent in base_catalog]
    if preset == "all_unshare":
        return [_replace_g(agent, 1) for agent in base_catalog]
    raise ValueError(f"Unknown catalog preset: {preset}")


def _replace_g(agent: AgentSpec, new_g: int) -> AgentSpec:
    return AgentSpec(
        agent_id=agent.agent_id,
        stage_name=agent.stage_name,
        g=new_g,
        kind=agent.kind,
        cost=agent.cost,
    )


def group_agent_ids_by_stage(agent_catalog: Iterable[AgentSpec]) -> dict[str, list[str]]:
    """Group agent ids by stage while preserving input order."""

    grouped: dict[str, list[str]] = {}
    for agent in agent_catalog:
        grouped.setdefault(agent.stage_name, []).append(agent.agent_id)
    return grouped
