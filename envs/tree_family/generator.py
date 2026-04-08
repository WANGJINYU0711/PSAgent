"""Family generator and validators for reusable tree families."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from .presets import (
    build_moderate_family_spec,
    build_neutral_family_spec,
    build_strong_family_spec,
)
from .specs import AgentSpec, FamilySpec


ATTRIBUTE_IDS = list(range(1, 11))
STAGE_FOCUS = {
    "stage1": [1, 7, 8, 9],
    "stage2": [2, 6, 8],
    "stage3": [3, 4, 9],
    "stage4": [4, 5, 9],
    "stage5": [5, 10, 6],
}


class TreeFamilyGenerator:
    def build_family(self, kind: str, seed: int = 0) -> tuple[FamilySpec, dict[str, AgentSpec]]:
        config = self._load_preset(kind)
        rng = random.Random(seed)

        stages = list(config["stages"])
        stage_agents: dict[str, list[str]] = {}
        agent_map: dict[str, AgentSpec] = {}

        for stage_name in stages:
            specs = self._build_stage_agents(stage_name, config, rng)
            stage_agents[stage_name] = [spec.agent_id for spec in specs]
            for spec in specs:
                agent_map[spec.agent_id] = spec

        family_spec = FamilySpec(
            family_name=f"{kind}_seed_{seed}",
            stages=stages,
            stage_agents=stage_agents,
        )
        return family_spec, agent_map

    def validate_family(self, family_spec: FamilySpec, agent_map: dict[str, AgentSpec]) -> list[str]:
        errors: list[str] = []
        if not family_spec.stages:
            errors.append("Family has no stages.")
        for stage_name in family_spec.stages:
            stage_agent_ids = family_spec.stage_agents.get(stage_name, [])
            if not stage_agent_ids:
                errors.append(f"Stage {stage_name} has no agents.")
            for agent_id in stage_agent_ids:
                if agent_id not in agent_map:
                    errors.append(f"Agent {agent_id} missing from agent_map.")
                    continue
                spec = agent_map[agent_id]
                if spec.g not in {0, 1}:
                    errors.append(f"Agent {agent_id} has invalid g={spec.g}.")
                if not spec.attribute_skill:
                    errors.append(f"Agent {agent_id} has empty attribute_skill.")
                for key, value in spec.attribute_skill.items():
                    if not isinstance(key, int):
                        errors.append(f"Agent {agent_id} has non-int attribute key {key!r}.")
                    if not isinstance(value, (int, float)):
                        errors.append(f"Agent {agent_id} has non-numeric skill value {value!r}.")
        return errors

    def describe_family(self, family_spec: FamilySpec, agent_map: dict[str, AgentSpec]) -> dict[str, Any]:
        competence = Counter()
        scope = Counter()
        stability = Counter()
        g_ratio_per_stage: dict[str, float] = {}
        num_agents_per_stage: dict[str, int] = {}
        dangerous_child_count = 0
        risky_depth = 0

        for depth, stage_name in enumerate(family_spec.stages, start=1):
            stage_ids = family_spec.stage_agents[stage_name]
            num_agents_per_stage[stage_name] = len(stage_ids)
            g1_count = 0
            for agent_id in stage_ids:
                spec = agent_map[agent_id]
                competence[spec.competence_level] += 1
                scope[spec.scope_level] += 1
                stability[spec.stability_level] += 1
                if spec.g == 1:
                    g1_count += 1
            g_ratio = g1_count / max(1, len(stage_ids))
            g_ratio_per_stage[stage_name] = g_ratio
            dangerous_child_count += g1_count
            if g1_count > 0:
                risky_depth = depth

        return {
            "family_name": family_spec.family_name,
            "num_stages": len(family_spec.stages),
            "num_agents_per_stage": num_agents_per_stage,
            "g1_ratio_per_stage": g_ratio_per_stage,
            "competence_counts": dict(competence),
            "scope_counts": dict(scope),
            "stability_counts": dict(stability),
            "estimated_risky_depth": risky_depth,
            "estimated_dangerous_child_count": dangerous_child_count,
        }

    def _load_preset(self, kind: str) -> dict[str, Any]:
        if kind == "neutral":
            return build_neutral_family_spec()
        if kind == "moderate":
            return build_moderate_family_spec()
        if kind == "strong":
            return build_strong_family_spec()
        raise ValueError(f"Unknown family kind: {kind}")

    def _build_stage_agents(
        self,
        stage_name: str,
        config: dict[str, Any],
        rng: random.Random,
    ) -> list[AgentSpec]:
        num_agents = config["num_agents_per_stage"]
        g1_count = config["g1_per_stage"][stage_name]
        competence_levels = self._expand_counts(config["competence_per_stage"], num_agents)
        scope_levels = self._expand_counts(config["scope_per_stage"], num_agents)
        stability_levels = self._expand_counts(config["stability_per_stage"], num_agents)

        rng.shuffle(competence_levels)
        rng.shuffle(scope_levels)
        rng.shuffle(stability_levels)

        g_layout = [1] * g1_count + [0] * (num_agents - g1_count)
        rng.shuffle(g_layout)

        specs: list[AgentSpec] = []
        for idx in range(num_agents):
            competence = competence_levels[idx]
            scope = scope_levels[idx]
            stability = stability_levels[idx]
            g = g_layout[idx]
            agent_id = f"{stage_name}_{scope}_{competence}_{stability}_g{g}_{idx}"
            attribute_skill = self._build_attribute_skill(
                stage_name=stage_name,
                scope_level=scope,
                competence_level=competence,
                config=config,
                rng=rng,
            )
            base_cost = self._build_base_cost(
                g=g,
                scope_level=scope,
                stability_level=stability,
                config=config,
                rng=rng,
            )
            specs.append(
                AgentSpec(
                    agent_id=agent_id,
                    g=g,
                    base_cost=round(base_cost, 3),
                    competence_level=competence,
                    scope_level=scope,
                    stability_level=stability,
                    attribute_skill=attribute_skill,
                )
            )
        return specs

    def _expand_counts(self, count_map: dict[str, int], total: int) -> list[str]:
        items: list[str] = []
        for key, count in count_map.items():
            items.extend([key] * count)
        if len(items) != total:
            raise ValueError(f"Preset count mismatch: expected {total}, got {len(items)}.")
        return items

    def _build_attribute_skill(
        self,
        stage_name: str,
        scope_level: str,
        competence_level: str,
        config: dict[str, Any],
        rng: random.Random,
    ) -> dict[int, float]:
        skill_ranges = config["skill_ranges"]
        focus = set(STAGE_FOCUS[stage_name])
        values: dict[int, float] = {}
        if scope_level == "broad":
            lo, hi = skill_ranges["broad"]
            for attr_id in ATTRIBUTE_IDS:
                values[attr_id] = rng.uniform(lo, hi)
        else:
            focus_lo, focus_hi = skill_ranges["narrow_focus"]
            other_lo, other_hi = skill_ranges["narrow_other"]
            extra_focus = set(rng.sample(ATTRIBUTE_IDS, k=2))
            effective_focus = focus | extra_focus
            for attr_id in ATTRIBUTE_IDS:
                if attr_id in effective_focus:
                    values[attr_id] = rng.uniform(focus_lo, focus_hi)
                else:
                    values[attr_id] = rng.uniform(other_lo, other_hi)

        if competence_level == "high":
            bonus = skill_ranges["high_bonus"]
            for attr_id in values:
                values[attr_id] = min(1.0, values[attr_id] + bonus)

        return {attr_id: round(score, 3) for attr_id, score in values.items()}

    def _build_base_cost(
        self,
        g: int,
        scope_level: str,
        stability_level: str,
        config: dict[str, Any],
        rng: random.Random,
    ) -> float:
        if g == 0 and scope_level == "broad" and stability_level == "stable":
            lo, hi = config["cost_ranges"]["safe"]
        else:
            lo, hi = config["cost_ranges"]["special"]
        return rng.uniform(lo, hi)
