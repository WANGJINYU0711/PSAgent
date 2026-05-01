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
    if preset == "mixed_v2_richer":
        return _build_richer_catalog("mixed")
    if preset == "all_share_v2_richer":
        return _build_richer_catalog("all_share")
    if preset == "all_unshare_v2_richer":
        return _build_richer_catalog("all_unshare")
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


def _build_richer_catalog(mode: str) -> list[AgentSpec]:
    stage_templates = {
        "stage1": [
            ("ground_oracle_g0", "rule", 0.20, 0),
            ("ground_weak_g0", "simulated", 0.15, 0),
            ("ground_specialist_g1", "rule", 0.30, 1),
            ("ground_noisy_g1", "simulated", 0.40, 1),
        ],
        "stage2": [
            ("resolve_oracle_g0", "rule", 0.20, 0),
            ("resolve_weak_g0", "simulated", 0.18, 0),
            ("resolve_specialist_g1", "rule", 0.35, 1),
            ("resolve_noisy_g1", "simulated", 0.50, 1),
        ],
        "stage3": [
            ("feature_oracle_g0", "rule", 0.20, 0),
            ("feature_weak_g0", "simulated", 0.18, 0),
            ("feature_specialist_g1", "rule", 0.32, 1),
            ("feature_noisy_g1", "simulated", 0.50, 1),
        ],
        "stage4": [
            ("adjudicate_oracle_g0", "rule", 0.20, 0),
            ("adjudicate_weak_g0", "simulated", 0.20, 0),
            ("adjudicate_specialist_g1", "rule", 0.34, 1),
            ("adjudicate_noisy_g1", "simulated", 0.50, 1),
        ],
        "stage5": [
            ("execute_oracle_g0", "rule", 0.20, 0),
            ("execute_weak_g0", "simulated", 0.22, 0),
            ("execute_specialist_g1", "rule", 0.36, 1),
            ("execute_noisy_g1", "simulated", 0.50, 1),
        ],
    }

    catalog: list[AgentSpec] = []
    for stage_name, rows in stage_templates.items():
        for agent_id, kind, cost, g in rows:
            if mode == "all_share":
                g = 0
            elif mode == "all_unshare":
                g = 1
            catalog.append(
                AgentSpec(
                    agent_id=agent_id,
                    stage_name=stage_name,
                    g=g,
                    kind=kind,
                    cost=cost,
                )
            )
    return catalog
