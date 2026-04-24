"""Family generator and validators for reusable tree families."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from .presets import (
    build_moderate_family_spec,
    build_neutral_family_spec,
    build_shared_basin_strong_2of5_gonly_family_spec,
    build_shared_basin_strong_all_share_gonly_family_spec,
    build_shared_basin_strong_all_unshare_gonly_family_spec,
    build_shared_basin_strong_family_spec,
    build_shared_basin_strong_prefix_dedup_family_spec,
    build_shared_basin_strong_prefix_dedup_profile_switch_family_spec,
    build_strong_family_spec,
)
from .specs import AgentSpec, CAPABILITY_NAMES, FamilySpec


STAGE_FOCUS = {
    "stage1": [
        "user_grounding",
        "account_lookup",
        "line_resolution",
        "verification",
    ],
    "stage2": [
        "account_lookup",
        "line_resolution",
        "roaming_diagnosis",
    ],
    "stage3": [
        "network_diagnosis",
        "permission_diagnosis",
        "apn_diagnosis",
        "roaming_diagnosis",
    ],
    "stage4": [
        "network_diagnosis",
        "permission_diagnosis",
        "apn_diagnosis",
        "repair_execution",
    ],
    "stage5": [
        "repair_execution",
        "verification",
        "terminal_decision",
    ],
}


class TreeFamilyGenerator:
    def build_family(self, kind: str, seed: int = 0) -> tuple[FamilySpec, dict[str, AgentSpec]]:
        config = self._load_preset(kind)
        rng = random.Random(seed)
        generation_mode = config.get("generation_mode")

        if generation_mode == "capability_shared_basin_prefix_dedup":
            return self._build_prefix_dedup_family(kind=kind, config=config, rng=rng, seed=seed)

        stages = list(config["stages"])
        stage_agents: dict[str, list[str]] = {}
        agent_map: dict[str, AgentSpec] = {}
        stage_profile_by_agent: dict[str, dict[str, Any]] = {}

        for stage_name in stages:
            specs = self._build_stage_agents(stage_name, config, rng)
            stage_agents[stage_name] = [spec.agent_id for spec in specs]
            for spec in specs:
                agent_map[spec.agent_id] = spec
            if config.get("generation_mode") == "capability_shared_basin":
                profiles = config["stage_profiles"][stage_name]
                for idx, spec in enumerate(specs):
                    stage_profile_by_agent[spec.agent_id] = dict(profiles[idx])

        allowed_children = None
        if config.get("generation_mode") == "capability_shared_basin":
            allowed_children = self._build_allowed_children(
                stages=stages,
                stage_agents=stage_agents,
                stage_profile_by_agent=stage_profile_by_agent,
            )

        family_spec = FamilySpec(
            family_name=f"{kind}_seed_{seed}",
            stages=stages,
            stage_agents=stage_agents,
            allowed_children=allowed_children,
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
                    if not isinstance(key, str):
                        errors.append(f"Agent {agent_id} has non-string capability key {key!r}.")
                    elif key not in CAPABILITY_NAMES:
                        errors.append(f"Agent {agent_id} has unknown capability key {key!r}.")
                    if not isinstance(value, (int, float)):
                        errors.append(f"Agent {agent_id} has non-numeric skill value {value!r}.")
        allowed_children = family_spec.allowed_children or {}
        for prefix, child_ids in allowed_children.items():
            expected_depth = len(prefix)
            if expected_depth >= len(family_spec.stages):
                errors.append(f"Continuation prefix {prefix!r} is deeper than the family stages.")
                continue
            expected_stage = family_spec.stages[expected_depth]
            for agent_id in child_ids:
                spec = agent_map.get(agent_id)
                if spec is None:
                    errors.append(f"Continuation child {agent_id!r} missing from agent_map.")
                    continue
                if spec.agent_id not in family_spec.stage_agents.get(expected_stage, []):
                    errors.append(
                        f"Continuation child {agent_id!r} is not registered for expected stage {expected_stage}."
                    )
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
        if kind == "shared_basin_strong":
            return build_shared_basin_strong_family_spec()
        if kind == "shared_basin_strong_prefix_dedup":
            return build_shared_basin_strong_prefix_dedup_family_spec()
        if kind == "shared_basin_strong_prefix_dedup_profile_switch":
            return build_shared_basin_strong_prefix_dedup_profile_switch_family_spec()
        if kind == "shared_basin_strong_2of5_gonly":
            return build_shared_basin_strong_2of5_gonly_family_spec()
        if kind == "shared_basin_strong_all_share_gonly":
            return build_shared_basin_strong_all_share_gonly_family_spec()
        if kind == "shared_basin_strong_all_unshare_gonly":
            return build_shared_basin_strong_all_unshare_gonly_family_spec()
        raise ValueError(f"Unknown family kind: {kind}")

    def _build_stage_agents(
        self,
        stage_name: str,
        config: dict[str, Any],
        rng: random.Random,
    ) -> list[AgentSpec]:
        if config.get("generation_mode") == "capability_shared_basin":
            return self._build_shared_basin_stage_agents(stage_name, config, rng)

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
                    deliberation_mode=self._legacy_deliberation_mode(
                        competence_level=competence,
                        scope_level=scope,
                    ),
                )
            )
        return specs

    def _build_shared_basin_stage_agents(
        self,
        stage_name: str,
        config: dict[str, Any],
        rng: random.Random,
    ) -> list[AgentSpec]:
        profiles = config["stage_profiles"][stage_name]
        fields = config["profile_fields"]
        specs: list[AgentSpec] = []
        for idx, profile in enumerate(profiles):
            agent_id = f"{stage_name}_{profile['role']}_g{profile['g']}_{idx}"
            attribute_skill = self._build_shared_basin_attribute_skill(stage_name, profile, config, rng)
            base_cost = self._build_shared_basin_base_cost(config, rng, profile=profile)
            specs.append(
                AgentSpec(
                    agent_id=agent_id,
                    g=profile["g"],
                    base_cost=round(base_cost, 3),
                    competence_level=fields["competence_level"],
                    scope_level=fields["scope_level"],
                    stability_level=fields["stability_level"],
                    attribute_skill=attribute_skill,
                    deliberation_mode=self._shared_basin_deliberation_mode(
                        stage_name=stage_name,
                        profile=profile,
                    ),
                    node_semantic=str(profile.get("node_semantic", "mixed_shared")),
                    route_label=str(profile.get("route_label", "")),
                )
            )
        return specs

    def _build_prefix_dedup_family(
        self,
        *,
        kind: str,
        config: dict[str, Any],
        rng: random.Random,
        seed: int,
    ) -> tuple[FamilySpec, dict[str, AgentSpec]]:
        stages = list(config["stages"])
        topology_path = Path(str(config["prefix_dedup_topology_spec_path"]))
        topology = json.loads(topology_path.read_text(encoding="utf-8"))
        topology_stages = list(topology.get("stages", []))
        if topology_stages != stages:
            raise ValueError(
                "Prefix-dedup topology stages do not match preset stages: "
                f"preset={stages} topology={topology_stages}"
            )

        base_profile_by_alias: dict[str, dict[str, dict[str, Any]]] = {}
        for stage_name in stages:
            stage_profiles = list(config["stage_profiles"][stage_name])
            stage_alias_map: dict[str, dict[str, Any]] = {}
            for idx, profile in enumerate(stage_profiles, start=1):
                stage_alias_map[f"{stage_name}_n{idx}"] = dict(profile)
            base_profile_by_alias[stage_name] = stage_alias_map

        fields = config["profile_fields"]
        stage_agents: dict[str, list[str]] = {stage_name: [] for stage_name in stages}
        agent_map: dict[str, AgentSpec] = {}
        allowed_children: dict[tuple[str, ...], list[str]] = {(): []}
        alias_to_prefix: dict[str, tuple[str, ...]] = {}

        for stage_index, stage_name in enumerate(stages):
            stage_nodes = list(topology.get("nodes", {}).get(stage_name, []))
            for node in stage_nodes:
                agent_id = str(node["agent_id"])
                base_alias = str(node["base_alias"])
                profile = base_profile_by_alias[stage_name].get(base_alias)
                if profile is None:
                    raise ValueError(
                        f"Missing base profile for prefix-dedup node {agent_id!r} "
                        f"with base_alias={base_alias!r}."
                    )
                attribute_skill = self._build_shared_basin_attribute_skill(
                    stage_name,
                    profile,
                    config,
                    rng,
                )
                base_cost = self._build_shared_basin_base_cost(config, rng, profile=profile)
                agent_map[agent_id] = AgentSpec(
                    agent_id=agent_id,
                    g=int(node.get("g", profile["g"])),
                    base_cost=round(base_cost, 3),
                    competence_level=fields["competence_level"],
                    scope_level=fields["scope_level"],
                    stability_level=fields["stability_level"],
                    attribute_skill=attribute_skill,
                    deliberation_mode=self._shared_basin_deliberation_mode(
                        stage_name=stage_name,
                        profile=profile,
                    ),
                    node_semantic=str(profile.get("node_semantic", "mixed_shared")),
                    route_label=str(profile.get("route_label", "")),
                )
                stage_agents[stage_name].append(agent_id)

                parent_alias = str(node.get("parent_alias", "ROOT"))
                if stage_index == 0:
                    if parent_alias != "ROOT":
                        raise ValueError(
                            f"Stage1 prefix-dedup node {agent_id!r} must parent to ROOT, "
                            f"got {parent_alias!r}."
                        )
                    allowed_children[()].append(agent_id)
                    alias_to_prefix[agent_id] = (agent_id,)
                    continue

                parent_prefix = alias_to_prefix.get(parent_alias)
                if parent_prefix is None:
                    raise ValueError(
                        f"Prefix-dedup parent alias {parent_alias!r} for node {agent_id!r} "
                        "was not built before its child."
                    )
                allowed_children.setdefault(parent_prefix, []).append(agent_id)
                alias_to_prefix[agent_id] = parent_prefix + (agent_id,)

        family_spec = FamilySpec(
            family_name=f"{kind}_seed_{seed}",
            stages=stages,
            stage_agents=stage_agents,
            allowed_children=allowed_children,
        )
        return family_spec, agent_map

    def _build_allowed_children(
        self,
        *,
        stages: list[str],
        stage_agents: dict[str, list[str]],
        stage_profile_by_agent: dict[str, dict[str, Any]],
    ) -> dict[tuple[str, ...], list[str]]:
        """Build explicit continuation topology for family variants with routed basins.

        The shared-basin family needs a topology stronger than a plain stagewise
        cartesian product: some shared prefixes should close into fully shared
        subtrees, while mixed/specialist branches should continue to expose risky
        suffixes. We encode this as an explicit prefix-to-children map.
        """

        allowed_children: dict[tuple[str, ...], list[str]] = {(): list(stage_agents[stages[0]])}
        frontier: list[tuple[str, ...]] = [()]

        for depth, stage_name in enumerate(stages[:-1]):
            next_stage = stages[depth + 1]
            next_candidates = list(stage_agents[next_stage])
            next_by_label: dict[str, list[str]] = {}
            for agent_id in next_candidates:
                label = str(stage_profile_by_agent.get(agent_id, {}).get("route_label", ""))
                next_by_label.setdefault(label, []).append(agent_id)

            next_frontier: list[tuple[str, ...]] = []
            for prefix in frontier:
                current_children = allowed_children.get(prefix, [])
                for agent_id in current_children:
                    current_prefix = prefix + (agent_id,)
                    profile = stage_profile_by_agent.get(agent_id, {})
                    allowed_labels = profile.get("allowed_next_labels")
                    if not allowed_labels:
                        child_ids = list(next_candidates)
                    else:
                        child_ids = []
                        for label in allowed_labels:
                            child_ids.extend(next_by_label.get(str(label), []))
                    # Preserve preset ordering and remove duplicates.
                    deduped_child_ids = [
                        next_agent_id
                        for next_agent_id in next_candidates
                        if next_agent_id in child_ids
                    ]
                    allowed_children[current_prefix] = deduped_child_ids
                    next_frontier.append(current_prefix)
            frontier = next_frontier

        return allowed_children

    def _legacy_deliberation_mode(
        self,
        *,
        competence_level: str,
        scope_level: str,
    ) -> str:
        if competence_level == "high" or scope_level == "broad":
            return "deep"
        return "fast"

    def _shared_basin_deliberation_mode(
        self,
        *,
        stage_name: str,
        profile: dict[str, Any],
    ) -> str:
        explicit = profile.get("deliberation_mode")
        if explicit in {"fast", "deep"}:
            return explicit

        role = str(profile.get("role", ""))
        if stage_name in {"stage3", "stage5"}:
            return "deep"
        if profile.get("profile_kind") == "specialist":
            return "deep"
        if stage_name == "stage1":
            return "fast" if "lookup" in role else "deep"
        if stage_name == "stage2":
            return "fast" if "line_core" in role else "deep"
        if stage_name == "stage4":
            return "fast" if "network" in role else "deep"
        return "deep"

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
    ) -> dict[str, float]:
        skill_ranges = config["skill_ranges"]
        focus = set(STAGE_FOCUS[stage_name])
        values: dict[str, float] = {}
        if scope_level == "broad":
            lo, hi = skill_ranges["broad"]
            for capability_name in CAPABILITY_NAMES:
                values[capability_name] = rng.uniform(lo, hi)
        else:
            focus_lo, focus_hi = skill_ranges["narrow_focus"]
            other_lo, other_hi = skill_ranges["narrow_other"]
            extra_focus = set(rng.sample(list(CAPABILITY_NAMES), k=2))
            effective_focus = focus | extra_focus
            for capability_name in CAPABILITY_NAMES:
                if capability_name in effective_focus:
                    values[capability_name] = rng.uniform(focus_lo, focus_hi)
                else:
                    values[capability_name] = rng.uniform(other_lo, other_hi)

        if competence_level == "high":
            bonus = skill_ranges["high_bonus"]
            for capability_name in values:
                values[capability_name] = min(1.0, values[capability_name] + bonus)

        return {capability_name: round(score, 3) for capability_name, score in values.items()}

    def _build_shared_basin_attribute_skill(
        self,
        stage_name: str,
        profile: dict[str, Any],
        config: dict[str, Any],
        rng: random.Random,
    ) -> dict[str, float]:
        focus = set(STAGE_FOCUS[stage_name])
        anchors = set(profile.get("anchor_caps", []))
        supports = set(profile.get("support_caps", []))
        node_semantic = str(profile.get("node_semantic", "mixed_shared"))
        ranges = config["semantic_skill_ranges"][node_semantic]
        focus_fallback = ranges.get("focus_fallback", ranges["support"])

        values: dict[str, float] = {}
        for capability_name in CAPABILITY_NAMES:
            if capability_name in anchors:
                lo, hi = ranges["anchor"]
            elif capability_name in supports:
                lo, hi = ranges["support"]
            elif capability_name in focus:
                lo, hi = focus_fallback
            else:
                lo, hi = ranges["background"]
            sampled = rng.uniform(lo, hi)
            if capability_name in anchors:
                sampled = min(1.0, sampled + float(profile.get("anchor_boost", 0.0)))
            values[capability_name] = round(sampled, 3)
        return values

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

    def _build_shared_basin_base_cost(
        self,
        config: dict[str, Any],
        rng: random.Random,
        profile: dict[str, Any] | None = None,
    ) -> float:
        if profile is not None:
            explicit_range = profile.get("base_cost_range")
            if (
                isinstance(explicit_range, (list, tuple))
                and len(explicit_range) == 2
            ):
                lo = float(explicit_range[0])
                hi = float(explicit_range[1])
                if lo > hi:
                    lo, hi = hi, lo
                return rng.uniform(lo, hi)
        lo, hi = config["cost_ranges"]["uniform"]
        return rng.uniform(lo, hi)
