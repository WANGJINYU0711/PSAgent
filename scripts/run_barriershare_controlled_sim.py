"""Theory-aligned non-LLM controlled simulation for BarrierShare."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import statistics
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Any, Iterable, Sequence


ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
    ROOT / "baselines",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fixed_tree_env import (  # noqa: E402
    AgentSpec,
    EpisodeResult,
    compute_shared_upload_edges,
    compute_shared_upload_stop_prefix,
    leaf_starts_shared_upload,
)
from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402
from risky_ps_old import RiskyPSOldPolicy  # noqa: E402
from risky_ps_ix import RiskyPSIXPolicy  # noqa: E402
from risky_ps_direct_cost import RiskyPSDirectCostPolicy  # noqa: E402
from risky_ps_safe_conditional import (  # noqa: E402
    RiskyPSSafeConditionalIXPolicy,
    RiskyPSSafeConditionalPolicy,
)


METHODS = {
    "risky_ps_old": RiskyPSOldPolicy,
    "risky_ps": RiskyPSPolicy,
    "risky_ps_ix": RiskyPSIXPolicy,
    "direct_multistage_exp3": DirectMultiStageExp3Policy,
    "epsilon_exp3": EpsilonExp3Policy,
    "random_path": RandomPathPolicy,
    "naive_mixed": NaiveMixedPolicy,
    "risky_ps_safe_conditional": RiskyPSSafeConditionalPolicy,
    "risky_ps_safe_conditional_ix": RiskyPSSafeConditionalIXPolicy,
    "risky_ps_direct_cost": RiskyPSDirectCostPolicy,
}
COMMON_ETA_METHODS = frozenset(
    {
        "direct_multistage_exp3",
        "epsilon_exp3",
        "risky_ps_old",
        "risky_ps",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    }
)
COMMON_EPSILON_METHODS = frozenset(
    {
        "epsilon_exp3",
        "risky_ps_old",
        "risky_ps",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    }
)
MAIN_VARIANTS = ("all_share", "partial_4of5", "partial_2of5", "all_unshare")
SHARED_CORE_A = "shared_core_a"
SHARED_CORE_B = "shared_core_b"
PRIVATE_CORE = "private_core"
REFERENCE_METHODS = (
    "risky_ps",
    "epsilon_exp3",
    "direct_multistage_exp3",
    "naive_mixed",
    "random_path",
)
TREE_SPEC_ROLE_MODES = ("spec_or_agent_id", "agent_id", "subtree_local", "base_alias")
TREE_SPEC_COST_MODES = ("default", "ps_favored_trap")
PS_FAVORED_TRAP_BASE_ALIASES = (
    "stage1_n4",
    "stage2_n5",
    "stage3_n5",
    "stage4_n5",
    "stage5_n5",
)
PS_FAVORED_TRAP_BASIN_STAGE1 = "stage1_n4"
PS_FAVORED_TRAP_BASIN_STAGE2 = frozenset({"stage2_n4", "stage2_n5"})
PS_FAVORED_SAFE_SUFFIX_STAGE3 = frozenset({"stage3_n1", "stage3_n2"})
PS_FAVORED_SAFE_SUFFIX_STAGE4 = "stage4_n1"
PS_FAVORED_SAFE_SUFFIX_STAGE5 = frozenset({"stage5_n1", "stage5_n2"})
DEFAULT_IX_ETA_SHARED_VALUES = (0.005, 0.01, 0.02, 0.05)
DEFAULT_IX_GAMMA_SHARED_VALUES = (0.0005, 0.001, 0.002, 0.005)
DENOMINATOR_ABLATION_METHODS = (
    "risky_ps",
    "risky_ps_ix",
    "risky_ps_safe_conditional",
    "risky_ps_safe_conditional_ix",
    "epsilon_exp3",
    "direct_multistage_exp3",
    "naive_mixed",
    "random_path",
)
DENOMINATOR_METHODS = (
    "risky_ps",
    "risky_ps_ix",
    "risky_ps_safe_conditional",
    "risky_ps_safe_conditional_ix",
)
DIRECT_COST_ABLATION_METHODS = (
    "risky_ps",
    "risky_ps_ix",
    "risky_ps_safe_conditional",
    "risky_ps_direct_cost",
    "epsilon_exp3",
    "direct_multistage_exp3",
    "naive_mixed",
    "random_path",
)
DIRECT_COST_METHODS = (
    "risky_ps",
    "risky_ps_ix",
    "risky_ps_safe_conditional",
    "risky_ps_direct_cost",
)
TAIL_WINDOW_SIZE_DEFAULT = 20


@dataclass(frozen=True)
class ControlledInstance:
    instance_id: str
    episode_index: int
    is_specialist: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "episode_index": self.episode_index,
            "is_specialist": self.is_specialist,
        }


@dataclass(frozen=True)
class PathProfile:
    visible_path: tuple[str, ...]
    latent_roles: tuple[str, ...]
    family_label: str
    normal_cost: float
    specialist_cost: float
    base_aliases: tuple[str, ...] = ()
    gates: tuple[int, ...] = ()


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.fmean(rows) if rows else 0.0


def stdev(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.stdev(rows) if len(rows) > 1 else 0.0


def parse_float_list(value: str) -> list[float]:
    rows = [row.strip() for row in value.split(",")]
    return [float(row) for row in rows if row]


def format_float_for_label(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def stable_unit_interval(*parts: Any) -> float:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    value = int.from_bytes(digest[:8], "big")
    return value / float(2**64 - 1)


def stable_noise(scale: float, *parts: Any) -> float:
    if scale <= 0:
        return 0.0
    u1 = max(stable_unit_interval("u1", *parts), 1e-12)
    u2 = stable_unit_interval("u2", *parts)
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return scale * z


def stable_centered_hash_noise(scale: float, *parts: Any) -> float:
    if scale <= 0:
        return 0.0
    return (stable_unit_interval("centered_noise", *parts) - 0.5) * scale


def stable_positive_hash_noise(scale: float, *parts: Any) -> float:
    if scale <= 0:
        return 0.0
    return stable_unit_interval("positive_noise", *parts) * scale


def stable_permutation(items: Sequence[str], *parts: Any) -> list[str]:
    return sorted(items, key=lambda item: stable_unit_interval("perm", *parts, item))


def default_shared_estimator_variant(method: str) -> str | None:
    if method == "risky_ps_ix":
        return "exp3_ix"
    if method == "risky_ps_direct_cost":
        return "direct_cost"
    if method == "risky_ps_safe_conditional":
        return "safe_conditional"
    if method == "risky_ps_safe_conditional_ix":
        return "safe_conditional_ix"
    if method in {"risky_ps", "risky_ps_old"}:
        return "path_importance_weighted"
    return None


def default_shared_denominator_mode(method: str) -> str | None:
    if method == "risky_ps_ix":
        return "path_prob_plus_gamma"
    if method == "risky_ps_direct_cost":
        return "none_observed_cost"
    if method == "risky_ps_safe_conditional":
        return "safe_subtree_conditional_prob"
    if method == "risky_ps_safe_conditional_ix":
        return "safe_subtree_conditional_prob_plus_gamma"
    if method in {"risky_ps", "risky_ps_old"}:
        return "path_prob"
    return None


def build_instances(
    *,
    horizon: int,
    seed: int,
    specialist_fraction: float,
) -> list[dict[str, Any]]:
    rng = random.Random(seed + 100_003)
    return [
        ControlledInstance(
            instance_id=f"ctx_{seed}_{idx}",
            episode_index=idx,
            is_specialist=rng.random() < specialist_fraction,
        ).to_dict()
        | {"horizon": horizon}
        for idx in range(horizon)
    ]


def latent_role_from_base_alias(base_alias: str) -> str:
    try:
        local_idx = int(str(base_alias).rsplit("n", 1)[1])
    except (IndexError, ValueError):
        return f"spec_role_{base_alias}"
    if local_idx == 1:
        return SHARED_CORE_A
    if local_idx == 2:
        return SHARED_CORE_B
    if local_idx == 5:
        return PRIVATE_CORE
    return f"filler_{max(0, local_idx - 3):02d}"


def resolve_tree_spec_cost_role(
    node: dict[str, Any],
    *,
    role_mode: str,
) -> tuple[str, str]:
    for field in ("cost_role", "synthetic_role", "latent_role"):
        value = node.get(field)
        if value not in (None, ""):
            return str(value), f"spec:{field}"
    agent_id = str(node.get("agent_id", node["alias"]))
    if role_mode in {"spec_or_agent_id", "agent_id", "subtree_local"}:
        return agent_id, "agent_id"
    if role_mode == "base_alias":
        base_alias = str(node.get("base_alias", node["alias"]))
        return latent_role_from_base_alias(base_alias), "base_alias"
    raise ValueError(f"Unknown tree-spec role mode: {role_mode}")


class ControlledTreeEnv:
    """Minimal FixedTreeEnvironment-compatible synthetic tree."""

    def __init__(
        self,
        *,
        setting_name: str,
        variant: str,
        depth: int,
        branching: int,
        seed: int,
        cost_noise: float,
        specialist_fraction: float,
        sharing_scheme: str,
        risky_depth: int | None = None,
    ) -> None:
        self.setting_name = setting_name
        self.variant = variant
        self.depth = depth
        self.branching = branching
        self.seed = seed
        self.cost_noise = cost_noise
        self.specialist_fraction = specialist_fraction
        self.sharing_scheme = sharing_scheme
        self.risky_depth = depth if risky_depth is None else risky_depth
        self.topology = "full"
        self.family_depth = min(3, depth)
        self.shared_core_roles = (SHARED_CORE_A, SHARED_CORE_B)
        self.private_core_role = PRIVATE_CORE
        self.latent_roles = self._build_latent_roles(branching)
        if sharing_scheme == "main_variant":
            self.shared_roles_main = self._shared_roles_for_variant(variant)
        else:
            self.shared_roles_main = set()
        self.STAGE_NAMES = [f"stage{i + 1}" for i in range(depth)]
        self.visible_role_map_by_stage: dict[str, dict[str, str]] = {}
        self.role_permutation_by_stage: dict[str, list[dict[str, Any]]] = {}
        self.stage_visible_ids: dict[str, list[str]] = {}
        self.agents_by_stage: dict[str, list[AgentSpec]] = {}
        self.agent_catalog: dict[str, AgentSpec] = {}
        for stage_idx, stage_name in enumerate(self.STAGE_NAMES):
            permutation = stable_permutation(
                self.latent_roles,
                setting_name,
                seed,
                stage_name,
            )
            stage_rows: list[AgentSpec] = []
            stage_map: dict[str, str] = {}
            stage_visible_ids: list[str] = []
            stage_perm_rows: list[dict[str, Any]] = []
            for visible_idx, latent_role in enumerate(permutation):
                agent_id = f"{stage_name}_child_{visible_idx:02d}"
                g = self._g_for_stage_role(stage_idx, latent_role)
                agent = AgentSpec(
                    agent_id=agent_id,
                    stage_name=stage_name,
                    g=g,
                    kind="synthetic",
                    cost=0.0,
                )
                stage_rows.append(agent)
                stage_visible_ids.append(agent_id)
                stage_map[agent_id] = latent_role
                stage_perm_rows.append(
                    {
                        "visible_child": agent_id,
                        "latent_role": latent_role,
                        "g": g,
                    }
                )
                self.agent_catalog[agent.agent_id] = agent
            self.agents_by_stage[stage_name] = stage_rows
            self.stage_visible_ids[stage_name] = stage_visible_ids
            self.visible_role_map_by_stage[stage_name] = stage_map
            self.role_permutation_by_stage[stage_name] = stage_perm_rows

        stage_agents = {
            stage_name: list(agent_ids)
            for stage_name, agent_ids in self.stage_visible_ids.items()
        }
        self.family_spec = SimpleNamespace(
            family_name=f"controlled_{setting_name}_seed_{seed}",
            stages=list(self.STAGE_NAMES),
            stage_agents=stage_agents,
            allowed_children=self._build_allowed_children(stage_agents),
        )
        self.path_profiles = self._build_path_profiles()
        self.num_paths = len(self.path_profiles)
        self.current_instance: dict[str, Any] | None = None

    def _build_latent_roles(self, branching: int) -> list[str]:
        base_roles = [SHARED_CORE_A, SHARED_CORE_B, PRIVATE_CORE]
        if branching <= len(base_roles):
            return base_roles[:branching]
        filler_count = branching - len(base_roles)
        fillers = [f"filler_{idx:02d}" for idx in range(filler_count)]
        return base_roles + fillers

    def _share_count_for_variant(self, variant: str) -> int:
        if variant == "all_share":
            return self.branching
        if variant == "partial_4of5":
            return max(1, int(round(0.8 * self.branching)))
        if variant == "partial_2of5":
            return max(1, int(round(0.4 * self.branching)))
        if variant == "all_unshare":
            return 0
        raise ValueError(f"Unknown controlled variant: {variant}")

    def _shared_roles_for_variant(self, variant: str) -> set[str]:
        share_count = self._share_count_for_variant(variant)
        if share_count >= self.branching:
            return set(self.latent_roles)
        if share_count <= 0:
            return set()
        priority = [
            role for role in (SHARED_CORE_A, SHARED_CORE_B) if role in self.latent_roles
        ]
        priority.extend(
            role
            for role in self.latent_roles
            if role not in {SHARED_CORE_A, SHARED_CORE_B, PRIVATE_CORE}
        )
        if PRIVATE_CORE in self.latent_roles:
            priority.append(PRIVATE_CORE)
        return set(priority[:share_count])

    def _g_for_stage_role(self, stage_idx: int, latent_role: str) -> int:
        if self.sharing_scheme == "main_variant":
            return 0 if latent_role in self.shared_roles_main else 1
        if self.sharing_scheme == "safe_suffix":
            return 1 if stage_idx < self.risky_depth else 0
        raise ValueError(f"Unknown sharing_scheme: {self.sharing_scheme}")

    def _build_allowed_children(
        self,
        stage_agents: dict[str, list[str]],
    ) -> dict[tuple[str, ...], list[str]]:
        allowed: dict[tuple[str, ...], list[str]] = {}
        frontier: list[tuple[str, ...]] = [()]
        for depth_idx, stage_name in enumerate(self.STAGE_NAMES):
            child_ids = list(stage_agents[stage_name])
            next_frontier: list[tuple[str, ...]] = []
            for prefix in frontier:
                allowed[prefix] = child_ids
                for child_id in child_ids:
                    next_frontier.append(prefix + (child_id,))
            frontier = next_frontier
        return allowed

    def _shared_template_set(self) -> set[tuple[str, ...]]:
        if self.family_depth == 1:
            return {(SHARED_CORE_A,), (SHARED_CORE_B,)}
        if self.family_depth == 2:
            return {
                (SHARED_CORE_A, SHARED_CORE_B),
                (SHARED_CORE_B, SHARED_CORE_A),
            }
        return {
            (SHARED_CORE_A, SHARED_CORE_B, SHARED_CORE_A),
            (SHARED_CORE_B, SHARED_CORE_A, SHARED_CORE_B),
        }

    def _classify_family(self, latent_roles: tuple[str, ...]) -> str:
        suffix = latent_roles[-self.family_depth :]
        shared_templates = self._shared_template_set()
        private_template = tuple([PRIVATE_CORE] * self.family_depth)
        shared_core_hits = sum(role in self.shared_core_roles for role in suffix)
        private_hits = sum(role == PRIVATE_CORE for role in suffix)
        if suffix in shared_templates:
            return "shared_template"
        if suffix == private_template:
            return "private_template"
        if shared_core_hits == self.family_depth:
            return "shared_core_family"
        if private_hits >= max(1, self.family_depth - 1):
            return "private_family"
        return "mixed_family"

    def _base_costs(self, family_label: str) -> tuple[float, float]:
        if family_label == "shared_template":
            return 0.04, 0.18
        if family_label == "shared_core_family":
            return 0.10, 0.24
        if family_label == "private_template":
            return 0.44, 0.03
        if family_label == "private_family":
            return 0.52, 0.10
        return 0.76, 0.68

    def _leaf_bias(self, visible_path: tuple[str, ...]) -> float:
        unit = stable_unit_interval("leaf_bias", self.setting_name, self.seed, *visible_path)
        return (unit - 0.5) * 0.03

    def _build_path_profiles(self) -> dict[tuple[str, ...], PathProfile]:
        stage_lists = [self.stage_visible_ids[stage_name] for stage_name in self.STAGE_NAMES]
        profiles: dict[tuple[str, ...], PathProfile] = {}
        for visible_path in itertools.product(*stage_lists):
            latent_roles = tuple(
                self.visible_role_map_by_stage[self.STAGE_NAMES[depth_idx]][agent_id]
                for depth_idx, agent_id in enumerate(visible_path)
            )
            family_label = self._classify_family(latent_roles)
            normal_base, specialist_base = self._base_costs(family_label)
            bias = self._leaf_bias(tuple(visible_path))
            profiles[tuple(visible_path)] = PathProfile(
                visible_path=tuple(visible_path),
                latent_roles=latent_roles,
                family_label=family_label,
                normal_cost=clamp01(normal_base + bias),
                specialist_cost=clamp01(specialist_base + bias),
            )
        return profiles

    def reset(self, instance: dict[str, Any]) -> None:
        self.current_instance = dict(instance)

    def describe_role_permutation(self) -> dict[str, Any]:
        return {
            "setting_name": self.setting_name,
            "seed": self.seed,
            "stages": self.role_permutation_by_stage,
        }

    def expected_cost(self, instance: dict[str, Any], path: Sequence[str]) -> float:
        profile = self.path_profiles[tuple(path)]
        if bool(instance.get("is_specialist", False)):
            return profile.specialist_cost
        return profile.normal_cost

    def observed_cost(
        self,
        *,
        expected: float,
        episode_index: int,
        visible_path: tuple[str, ...],
    ) -> float:
        noise = stable_noise(self.cost_noise, self.seed, episode_index, *visible_path)
        return clamp01(expected + noise)

    def oracle_reference(self, instances: list[dict[str, Any]]) -> dict[str, Any]:
        specialist_count = sum(1 for instance in instances if instance.get("is_specialist"))
        normal_count = len(instances) - specialist_count
        best_path: tuple[str, ...] | None = None
        best_profile: PathProfile | None = None
        best_cumulative = float("inf")
        for path, profile in self.path_profiles.items():
            cumulative = sum(self.expected_cost(instance, path) for instance in instances)
            if cumulative < best_cumulative:
                best_cumulative = cumulative
                best_path = path
                best_profile = profile
        if best_path is None or best_profile is None:
            raise RuntimeError("No legal paths found for controlled oracle.")
        oracle_episode_costs = [
            self.expected_cost(instance, best_path)
            for instance in instances
        ]
        return {
            "true_best_leaf": list(best_path),
            "true_best_cost": best_cumulative / max(1, len(instances)),
            "oracle_cumulative_cost": best_cumulative,
            "oracle_episode_costs": oracle_episode_costs,
            "oracle_family_label": best_profile.family_label,
            "oracle_latent_roles": list(best_profile.latent_roles),
            "num_paths": self.num_paths,
        }

    def run_path(self, path: list[str]) -> EpisodeResult:
        if self.current_instance is None:
            raise RuntimeError("ControlledTreeEnv.run_path called before reset().")
        visible_path = tuple(path)
        profile = self.path_profiles[visible_path]
        expected = self.expected_cost(self.current_instance, visible_path)
        episode_index = int(self.current_instance.get("episode_index", 0))
        observed = self.observed_cost(
            expected=expected,
            episode_index=episode_index,
            visible_path=visible_path,
        )
        leaf_type = self.compute_leaf_type(list(visible_path))
        first_private = self._first_private_barrier_stage_label(visible_path)
        stop_prefix = self.compute_shared_upload_stop_prefix(list(visible_path))
        episode_log = {
            "instance_id": self.current_instance.get("instance_id"),
            "selected_path": list(visible_path),
            "latent_roles": list(profile.latent_roles),
            "family_label": profile.family_label,
            "leaf_type": leaf_type,
            "terminal_cost": observed,
            "expected_cost": expected,
            "first_private_barrier_stage": first_private,
            "barrier_stop_depth": len(stop_prefix) if stop_prefix is not None else None,
            "candidate_count_per_stage": [
                len(self.agents_by_stage[stage_name]) for stage_name in self.STAGE_NAMES
            ],
            "legal_child_count_per_stage": [
                len(self.family_spec.allowed_children[tuple(visible_path[:depth_idx])])
                for depth_idx in range(len(visible_path))
            ],
        }
        return EpisodeResult(
            instance_id=str(self.current_instance.get("instance_id", "")),
            selected_path=list(visible_path),
            leaf_type=leaf_type,
            stage_outputs={},
            final_action="synthetic",
            oracle_action="synthetic",
            terminal_cost=observed,
            raw_terminal_penalty=observed,
            raw_path_cost_component=0.0,
            raw_reasoning_cost_component=0.0,
            raw_total_cost=observed,
            normalized_terminal_penalty=observed,
            success=False,
            path_agent_cost=0.0,
            reasoning_cost=0.0,
            total_cost=observed,
            total_cost_upper_bound=1.0,
            cost_scale_version="controlled_theory_aligned_v2",
            raw_outcome_penalty=observed,
            raw_policy_penalty=0.0,
            episode_log=episode_log,
        )

    def compute_leaf_type(self, path: list[str]) -> str:
        return "shared" if self.leaf_starts_shared_upload(path) else "unshared"

    def leaf_starts_shared_upload(self, path: list[str]) -> bool:
        return leaf_starts_shared_upload(path, self.agent_catalog)

    def compute_shared_upload_edges(self, path: list[str]) -> list[tuple[tuple[str, ...], tuple[str, ...]]]:
        return compute_shared_upload_edges(path, self.agent_catalog)

    def compute_shared_upload_stop_prefix(self, path: list[str]) -> tuple[str, ...] | None:
        return compute_shared_upload_stop_prefix(path, self.agent_catalog)

    def _first_private_barrier_stage_label(self, path: Sequence[str]) -> str | None:
        for depth_idx, agent_id in enumerate(path):
            if self.agent_catalog[agent_id].g == 1:
                return self.STAGE_NAMES[depth_idx]
        return None


class SpecBackedControlledTreeEnv(ControlledTreeEnv):
    """Controlled synthetic cost landscape on top of an external tree spec."""

    def __init__(
        self,
        *,
        spec_path: Path,
        seed: int,
        cost_noise: float,
        specialist_fraction: float,
        tree_spec_role_mode: str = "spec_or_agent_id",
        tree_spec_cost_mode: str = "default",
    ) -> None:
        if tree_spec_cost_mode not in TREE_SPEC_COST_MODES:
            raise ValueError(f"Unknown tree-spec cost mode: {tree_spec_cost_mode}")
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.spec_path = spec_path
        self.spec = spec
        self.tree_spec_role_mode = tree_spec_role_mode
        self.tree_spec_cost_mode = tree_spec_cost_mode
        self.setting_name = str(spec.get("tree_name", spec_path.stem))
        self.variant = str(spec.get("tree_name", spec_path.stem))
        self.depth = int(spec["depth"])
        self.branching = len(spec["edges_by_node_alias"].get("ROOT", []))
        self.seed = seed
        self.cost_noise = cost_noise
        self.specialist_fraction = specialist_fraction
        self.sharing_scheme = "external_tree_spec"
        self.risky_depth = None
        self.topology = "external_tree_spec"
        self.family_depth = min(3, self.depth)
        self.shared_core_roles = (SHARED_CORE_A, SHARED_CORE_B)
        self.private_core_role = PRIVATE_CORE
        self.latent_roles = [SHARED_CORE_A, SHARED_CORE_B, PRIVATE_CORE, "filler_00", "filler_01"]
        self.STAGE_NAMES = list(spec["stages"])
        self.visible_role_map_by_stage: dict[str, dict[str, str]] = {}
        self.role_permutation_by_stage: dict[str, list[dict[str, Any]]] = {}
        self.stage_visible_ids: dict[str, list[str]] = {}
        self.agents_by_stage: dict[str, list[AgentSpec]] = {}
        self.agent_catalog: dict[str, AgentSpec] = {}
        self.base_alias_by_agent: dict[str, str] = {}
        self.cost_role_by_agent: dict[str, str] = {}
        self.cost_role_source_counts: Counter[str] = Counter()

        for stage_name in self.STAGE_NAMES:
            stage_rows: list[AgentSpec] = []
            stage_map: dict[str, str] = {}
            stage_visible_ids: list[str] = []
            stage_perm_rows: list[dict[str, Any]] = []
            for node in spec["nodes"].get(stage_name, []):
                agent_id = str(node.get("agent_id", node["alias"]))
                base_alias = str(node.get("base_alias", node["alias"]))
                latent_role, cost_role_source = resolve_tree_spec_cost_role(
                    node,
                    role_mode=tree_spec_role_mode,
                )
                g = int(node["g"])
                agent = AgentSpec(
                    agent_id=agent_id,
                    stage_name=stage_name,
                    g=g,
                    kind="synthetic",
                    cost=0.0,
                )
                stage_rows.append(agent)
                stage_visible_ids.append(agent_id)
                stage_map[agent_id] = latent_role
                stage_perm_rows.append(
                    {
                        "visible_child": agent_id,
                        "latent_role": latent_role,
                        "cost_role": latent_role,
                        "cost_role_source": cost_role_source,
                        "g": g,
                        "base_alias": base_alias,
                    }
                )
                self.agent_catalog[agent.agent_id] = agent
                self.base_alias_by_agent[agent.agent_id] = base_alias
                self.cost_role_by_agent[agent.agent_id] = latent_role
                self.cost_role_source_counts[cost_role_source] += 1
            self.agents_by_stage[stage_name] = stage_rows
            self.stage_visible_ids[stage_name] = stage_visible_ids
            self.visible_role_map_by_stage[stage_name] = stage_map
            self.role_permutation_by_stage[stage_name] = stage_perm_rows
        self.latent_roles = sorted(set(self.cost_role_by_agent.values()))

        stage_agents = {
            stage_name: list(agent_ids)
            for stage_name, agent_ids in self.stage_visible_ids.items()
        }
        self.family_spec = SimpleNamespace(
            family_name=f"controlled_{self.setting_name}_seed_{seed}",
            stages=list(self.STAGE_NAMES),
            stage_agents=stage_agents,
            allowed_children=self._build_allowed_children_from_spec(
                spec["edges_by_node_alias"]
            ),
        )
        self.path_profiles = self._build_path_profiles()
        self.num_paths = len(self.path_profiles)
        self.current_instance: dict[str, Any] | None = None

    def _build_allowed_children_from_spec(
        self,
        edges_by_node_alias: dict[str, list[str]],
    ) -> dict[tuple[str, ...], list[str]]:
        allowed: dict[tuple[str, ...], list[str]] = {}
        frontier: list[tuple[str, ...]] = [()]
        while frontier:
            prefix = frontier.pop()
            parent_alias = "ROOT" if not prefix else prefix[-1]
            child_ids = list(edges_by_node_alias.get(parent_alias, []))
            if not child_ids:
                continue
            allowed[prefix] = child_ids
            for child_id in child_ids:
                frontier.append(prefix + (child_id,))
        return allowed

    def _build_path_profiles(self) -> dict[tuple[str, ...], PathProfile]:
        profiles: dict[tuple[str, ...], PathProfile] = {}
        frontier: list[tuple[str, ...]] = [()]
        while frontier:
            prefix = frontier.pop()
            child_ids = self.family_spec.allowed_children.get(prefix, [])
            if not child_ids:
                if len(prefix) != self.depth:
                    continue
                latent_roles = tuple(
                    self.visible_role_map_by_stage[self.STAGE_NAMES[depth_idx]][agent_id]
                    for depth_idx, agent_id in enumerate(prefix)
                )
                base_aliases = tuple(self.base_alias_by_agent[agent_id] for agent_id in prefix)
                gates = tuple(int(self.agent_catalog[agent_id].g) for agent_id in prefix)
                if self.tree_spec_cost_mode == "ps_favored_trap":
                    family_label, normal_base, specialist_base = self._base_costs_ps_favored_trap(
                        tuple(prefix),
                        base_aliases,
                        gates,
                    )
                elif self.tree_spec_role_mode == "base_alias":
                    family_label = self._classify_family(latent_roles)
                    normal_base, specialist_base = self._base_costs(family_label)
                elif self.tree_spec_role_mode == "subtree_local":
                    normal_base, specialist_base, family_label = self._base_costs_subtree_local(
                        tuple(prefix),
                        latent_roles,
                    )
                else:
                    family_label, normal_base, specialist_base = self._base_costs_unbound(
                        tuple(prefix),
                        latent_roles,
                    )
                bias = 0.0 if self.tree_spec_cost_mode == "ps_favored_trap" else self._leaf_bias(tuple(prefix))
                profiles[tuple(prefix)] = PathProfile(
                    visible_path=tuple(prefix),
                    latent_roles=latent_roles,
                    family_label=family_label,
                    normal_cost=clamp01(normal_base + bias),
                    specialist_cost=clamp01(specialist_base + bias),
                    base_aliases=base_aliases,
                    gates=gates,
                )
                continue
            for child_id in child_ids:
                frontier.append(prefix + (child_id,))
        return profiles

    def _base_costs_unbound(
        self,
        visible_path: tuple[str, ...],
        cost_roles: tuple[str, ...],
    ) -> tuple[str, float, float]:
        suffix_path = visible_path[-self.family_depth :]
        suffix_roles = cost_roles[-self.family_depth :]
        suffix_g = tuple(int(self.agent_catalog[agent_id].g) for agent_id in suffix_path)
        barrier_hits = sum(g == 1 for g in suffix_g)
        shared_hits = len(suffix_g) - barrier_hits
        signature = hashlib.sha256("|".join(suffix_roles).encode("utf-8")).hexdigest()[:10]
        family_label = f"unbound_s{shared_hits}_b{barrier_hits}_{signature}"
        normal_noise = stable_unit_interval(
            "external_tree_unbound_normal",
            self.setting_name,
            self.seed,
            *suffix_roles,
        )
        specialist_noise = stable_unit_interval(
            "external_tree_unbound_special",
            self.setting_name,
            self.seed,
            *suffix_roles,
        )
        normal_base = clamp01(0.04 + 0.16 * barrier_hits + 0.08 * normal_noise)
        specialist_base = clamp01(0.04 + 0.16 * shared_hits + 0.08 * specialist_noise)
        return family_label, normal_base, specialist_base

    def _base_costs_subtree_local(
        self,
        visible_path: tuple[str, ...],
        cost_roles: tuple[str, ...],
    ) -> tuple[float, float, str]:
        normal_cost = 0.02
        specialist_cost = 0.02
        barrier_hits = 0
        shared_hits = 0
        stage_scales = [0.11, 0.10, 0.09, 0.08, 0.07]
        for depth_idx, (agent_id, cost_role) in enumerate(zip(visible_path, cost_roles)):
            g = int(self.agent_catalog[agent_id].g)
            if g == 1:
                barrier_hits += 1
            else:
                shared_hits += 1
            stage_scale = stage_scales[min(depth_idx, len(stage_scales) - 1)]
            normal_noise = stable_unit_interval(
                "subtree_local_normal_node",
                self.setting_name,
                self.seed,
                cost_role,
            )
            specialist_noise = stable_unit_interval(
                "subtree_local_specialist_node",
                self.setting_name,
                self.seed,
                cost_role,
            )
            normal_term = 0.22 + 0.38 * normal_noise + (0.22 if g == 1 else -0.04)
            specialist_term = 0.18 + 0.38 * specialist_noise + (-0.06 if g == 1 else 0.18)
            normal_cost += stage_scale * normal_term
            specialist_cost += stage_scale * specialist_term
        normal_cost += 0.01 * barrier_hits
        specialist_cost += 0.01 * max(0, shared_hits - barrier_hits)
        prefix_signature = hashlib.sha256(
            "|".join(cost_roles[: self.family_depth]).encode("utf-8")
        ).hexdigest()[:10]
        family_label = (
            f"subtree_local_prefix_{prefix_signature}_"
            f"s{shared_hits}_b{barrier_hits}"
        )
        return clamp01(normal_cost), clamp01(specialist_cost), family_label

    def _is_ps_favored_safe_corridor(self, base_aliases: tuple[str, ...]) -> bool:
        return (
            len(base_aliases) == 5
            and base_aliases[1] == "stage2_n1"
            and self._is_ps_favored_safe_suffix(base_aliases, (0, 0, 0, 0, 0))
        )

    def _is_ps_favored_safe_suffix(
        self,
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            len(base_aliases) == 5
            and base_aliases[2] in PS_FAVORED_SAFE_SUFFIX_STAGE3
            and base_aliases[3] == PS_FAVORED_SAFE_SUFFIX_STAGE4
            and base_aliases[4] in PS_FAVORED_SAFE_SUFFIX_STAGE5
            and len(gates) == 5
            and all(int(gate) == 0 for gate in gates[2:5])
        )

    def _ps_favored_safe_suffix_signature(self, base_aliases: tuple[str, ...]) -> tuple[str, ...]:
        return base_aliases[2:5]

    def _is_ps_favored_candidate_safe_subtree(
        self,
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return self._is_ps_favored_target_subtree(base_aliases, gates)

    def _ps_favored_candidate_corridor_label(
        self,
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> str | None:
        if len(base_aliases) != 5 or len(gates) != 5:
            return None
        if not all(int(gate) == 0 for gate in gates):
            return None
        b1, b2, b3, b4, b5 = base_aliases
        del b1
        if (
            b2 == "stage2_n1"
            and b3 in {"stage3_n1", "stage3_n2"}
            and b4 == "stage4_n1"
            and b5 in {"stage5_n1", "stage5_n2"}
        ):
            return "C1_stage2_n1_stage4_n1"
        if (
            b2 == "stage2_n2"
            and b3 == "stage3_n2"
            and b4 == "stage4_n1"
            and b5 in {"stage5_n1", "stage5_n2"}
        ):
            return "C2_stage2_n2_stage4_n1"
        if (
            b2 == "stage2_n3"
            and b3 == "stage3_n3"
            and b4 in {"stage4_n2", "stage4_n3"}
            and b5 in {"stage5_n1", "stage5_n2"}
        ):
            return f"C3_stage2_n3_{b4}"
        if (
            b2 == "stage2_n3"
            and b3 == "stage3_n4"
            and b4 == "stage4_n3"
            and b5 in {"stage5_n3", "stage5_n4"}
        ):
            return "C4_stage2_n3_stage4_n3"
        return None

    def _is_ps_favored_target_subtree(
        self,
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            len(base_aliases) == 5
            and len(gates) == 5
            and base_aliases[1] == "stage2_n1"
            and base_aliases[2] in {"stage3_n1", "stage3_n2"}
            and base_aliases[3] == "stage4_n1"
            and base_aliases[4] in {"stage5_n1", "stage5_n2"}
            and all(int(gate) == 0 for gate in gates[1:5])
        )

    def _is_ps_favored_decoy_branch(
        self,
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            len(base_aliases) == 5
            and len(gates) == 5
            and base_aliases[0] in {"stage1_n1", "stage1_n2", "stage1_n3"}
            and base_aliases[1] in {"stage2_n2", "stage2_n3"}
            and all(int(gate) == 0 for gate in gates)
            and not self._is_ps_favored_target_subtree(base_aliases, gates)
        )

    def _iter_ps_favored_leaf_paths(self) -> Iterable[tuple[str, ...]]:
        frontier: list[tuple[str, ...]] = [()]
        while frontier:
            prefix = frontier.pop()
            child_ids = self.family_spec.allowed_children.get(prefix, [])
            if not child_ids:
                if len(prefix) == self.depth:
                    yield prefix
                continue
            for child_id in child_ids:
                frontier.append(prefix + (child_id,))

    def _ps_favored_base_aliases_for_path(self, visible_path: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(self.base_alias_by_agent[agent_id] for agent_id in visible_path)

    def _ps_favored_gates_for_path(self, visible_path: tuple[str, ...]) -> tuple[int, ...]:
        return tuple(int(self.agent_catalog[agent_id].g) for agent_id in visible_path)

    def _ps_favored_candidate_safe_subtree_paths(self) -> tuple[tuple[str, ...], ...]:
        cached = getattr(self, "_ps_favored_candidate_safe_subtree_paths_cache", None)
        if cached is not None:
            return cached
        paths = tuple(
            path
            for path in self._iter_ps_favored_leaf_paths()
            if self._is_ps_favored_candidate_safe_subtree(
                self._ps_favored_base_aliases_for_path(path),
                self._ps_favored_gates_for_path(path),
            )
        )
        self._ps_favored_candidate_safe_subtree_paths_cache = paths
        return paths

    def _ps_favored_selected_good_leaf_paths(self) -> frozenset[tuple[str, ...]]:
        cached = getattr(self, "_ps_favored_selected_good_leaf_paths_cache", None)
        if cached is not None:
            return cached
        candidate_paths = list(self._ps_favored_candidate_safe_subtree_paths())
        selected = sorted(
            candidate_paths,
            key=lambda group_key: stable_unit_interval(
                "ps_favored_v7_target_good_rank",
                self.setting_name,
                *group_key,
            ),
        )[: min(4, len(candidate_paths))]
        cached = frozenset(selected)
        self._ps_favored_selected_good_leaf_paths_cache = cached
        return cached

    def _ps_favored_local_decoy_leaf_paths(self) -> frozenset[tuple[str, ...]]:
        cached = getattr(self, "_ps_favored_local_decoy_leaf_paths_cache", None)
        if cached is not None:
            return cached
        decoy_paths = [
            path
            for path in self._iter_ps_favored_leaf_paths()
            if self._is_ps_favored_decoy_branch(
                self._ps_favored_base_aliases_for_path(path),
                self._ps_favored_gates_for_path(path),
            )
        ]
        selected = sorted(
            decoy_paths,
            key=lambda path: stable_unit_interval(
                "ps_favored_v7_local_decoy_rank",
                self.setting_name,
                *self._ps_favored_base_aliases_for_path(path),
                *path,
            ),
        )[: min(8, len(decoy_paths))]
        cached = frozenset(selected)
        self._ps_favored_local_decoy_leaf_paths_cache = cached
        return cached

    def _is_ps_favored_balancing_decoy_candidate(
        self,
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            len(base_aliases) == 5
            and len(gates) == 5
            and base_aliases[0] == "stage1_n3"
            and base_aliases[1] in {"stage2_n2", "stage2_n3"}
            and all(int(gate) == 0 for gate in gates)
        )

    def _ps_favored_balancing_decoy_leaf_paths(self) -> frozenset[tuple[str, ...]]:
        return frozenset(self._ps_favored_calibration_plan()["balancing_decoy_paths"])

    def _ps_favored_group_paths(
        self,
        *,
        b1: str,
        b2: str,
    ) -> list[PathProfile]:
        return [
            profile for profile in self.path_profiles.values()
            if len(profile.base_aliases) == 5
            and profile.base_aliases[0] == b1
            and profile.base_aliases[1] == b2
        ]

    def _ps_favored_balancing_probability(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
    ) -> float:
        probability = 0.20 + stable_positive_hash_noise(
            0.08,
            "ps_favored_v9_calibrated_decoy",
            self.setting_name,
            *base_aliases,
            *visible_path,
        )
        return clamp01(probability)

    def _ps_favored_adjusted_target_bad_probability(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
    ) -> float:
        probability = 0.90 + stable_positive_hash_noise(
            0.06,
            "ps_favored_v9_g1_target_bad_adjusted",
            self.setting_name,
            *base_aliases,
            *visible_path,
        )
        return clamp01(probability)

    def _ps_favored_calibration_plan(self) -> dict[str, Any]:
        cached = getattr(self, "_ps_favored_calibration_plan_cache", None)
        if cached is not None:
            return cached
        if not hasattr(self, "path_profiles"):
            return {
                "balancing_decoy_paths": frozenset(),
                "g1_target_bad_adjusted": False,
                "pre_stage1_n3_marginal": {},
                "post_stage1_n3_marginal": {},
                "g1_target_bad_adjusted_p_range": None,
            }

        target_mean = 0.52
        low, high = 0.49, 0.56

        def base_probability(profile: PathProfile) -> float:
            return self._ps_favored_trap_base_probability(
                visible_path=profile.visible_path,
                base_aliases=profile.base_aliases,
                gates=profile.gates,
                episode_index=None,
                horizon=None,
            )[0]

        groups = {
            b2: self._ps_favored_group_paths(b1="stage1_n3", b2=b2)
            for b2 in ("stage2_n1", "stage2_n2", "stage2_n3")
        }
        base_by_path = {
            profile.visible_path: base_probability(profile)
            for profile in self.path_profiles.values()
        }

        def mean_for_profiles(
            profiles: list[PathProfile],
            *,
            decoy_paths: set[tuple[str, ...]] | frozenset[tuple[str, ...]] = frozenset(),
            adjust_g1_target_bad: bool = False,
        ) -> float | None:
            if not profiles:
                return None
            values: list[float] = []
            for profile in profiles:
                if profile.visible_path in decoy_paths:
                    values.append(
                        self._ps_favored_balancing_probability(
                            profile.visible_path,
                            profile.base_aliases,
                        )
                    )
                elif (
                    adjust_g1_target_bad
                    and profile.base_aliases[0] == "stage1_n3"
                    and profile.base_aliases[1] == "stage2_n1"
                    and self._is_ps_favored_target_bad(
                        profile.visible_path,
                        profile.base_aliases,
                        profile.gates,
                    )
                ):
                    values.append(
                        self._ps_favored_adjusted_target_bad_probability(
                            profile.visible_path,
                            profile.base_aliases,
                        )
                    )
                else:
                    values.append(base_by_path[profile.visible_path])
            return mean(values)

        pre_marginal = {
            b2: mean_for_profiles(profiles)
            for b2, profiles in groups.items()
        }

        selected_decoys: set[tuple[str, ...]] = set()
        decoy_counts: dict[str, int] = {}
        for b2 in ("stage2_n2", "stage2_n3"):
            profiles = groups[b2]
            candidates = [
                profile for profile in profiles
                if self._is_ps_favored_balancing_decoy_candidate(
                    profile.base_aliases,
                    profile.gates,
                )
            ]
            candidates = sorted(
                candidates,
                key=lambda profile: stable_unit_interval(
                    "ps_favored_v9_calibration_rank",
                    self.setting_name,
                    b2,
                    *profile.base_aliases,
                    *profile.visible_path,
                ),
            )
            best_count = 0
            best_error = float("inf")
            best_in_range = False
            for count in range(len(candidates) + 1):
                trial_decoys = selected_decoys | {
                    profile.visible_path for profile in candidates[:count]
                }
                marginal = mean_for_profiles(profiles, decoy_paths=trial_decoys)
                if marginal is None:
                    continue
                in_range = low <= marginal <= high
                error = abs(marginal - target_mean)
                if (
                    (in_range and not best_in_range)
                    or (in_range == best_in_range and error < best_error)
                ):
                    best_count = count
                    best_error = error
                    best_in_range = in_range
            selected_decoys.update(profile.visible_path for profile in candidates[:best_count])
            decoy_counts[b2] = best_count

        g1_pre_after_decoys = mean_for_profiles(
            groups["stage2_n1"],
            decoy_paths=selected_decoys,
        )
        g1_target_bad_adjusted = (
            g1_pre_after_decoys is not None
            and g1_pre_after_decoys < low
        )
        post_marginal = {
            b2: mean_for_profiles(
                profiles,
                decoy_paths=selected_decoys,
                adjust_g1_target_bad=g1_target_bad_adjusted,
            )
            for b2, profiles in groups.items()
        }
        adjusted_target_bad_probabilities = [
            self._ps_favored_adjusted_target_bad_probability(
                profile.visible_path,
                profile.base_aliases,
            )
            for profile in groups["stage2_n1"]
            if self._is_ps_favored_target_bad(
                profile.visible_path,
                profile.base_aliases,
                profile.gates,
            )
        ] if g1_target_bad_adjusted else []

        cached = {
            "balancing_decoy_paths": frozenset(selected_decoys),
            "g2_decoy_count": decoy_counts.get("stage2_n2", 0),
            "g3_decoy_count": decoy_counts.get("stage2_n3", 0),
            "g1_target_bad_adjusted": g1_target_bad_adjusted,
            "pre_stage1_n3_marginal": pre_marginal,
            "post_stage1_n3_marginal": post_marginal,
            "g1_target_bad_adjusted_p_range": (
                {
                    "min": min(adjusted_target_bad_probabilities),
                    "max": max(adjusted_target_bad_probabilities),
                }
                if adjusted_target_bad_probabilities
                else None
            ),
        }
        self._ps_favored_calibration_plan_cache = cached
        return cached

    def _ps_favored_exact_best_path(self) -> tuple[str, ...] | None:
        cached = getattr(self, "_ps_favored_exact_best_path_cache", None)
        if cached is not None:
            return cached
        good_paths = sorted(
            self._ps_favored_selected_good_leaf_paths(),
            key=lambda path: stable_unit_interval(
                "ps_favored_exact_best_rank",
                self.setting_name,
                *self._ps_favored_base_aliases_for_path(path),
                *path,
            ),
        )
        cached = good_paths[0] if good_paths else None
        self._ps_favored_exact_best_path_cache = cached
        return cached

    def _is_ps_favored_near_best_good(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            self._is_ps_favored_candidate_safe_subtree(base_aliases, gates)
            and visible_path in self._ps_favored_selected_good_leaf_paths()
        )

    def _is_ps_favored_target_bad(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            self._is_ps_favored_target_subtree(base_aliases, gates)
            and visible_path not in self._ps_favored_selected_good_leaf_paths()
        )

    def _is_ps_favored_local_decoy(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            self._is_ps_favored_decoy_branch(base_aliases, gates)
            and visible_path in self._ps_favored_local_decoy_leaf_paths()
        )

    def _is_ps_favored_balancing_decoy(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> bool:
        return (
            self._is_ps_favored_balancing_decoy_candidate(base_aliases, gates)
            and visible_path in self._ps_favored_balancing_decoy_leaf_paths()
        )

    def _is_ps_favored_trap_basin(self, base_aliases: tuple[str, ...]) -> bool:
        return (
            len(base_aliases) == 5
            and base_aliases[0] == PS_FAVORED_TRAP_BASIN_STAGE1
            and base_aliases[1] in PS_FAVORED_TRAP_BASIN_STAGE2
        )

    def _is_ps_favored_exact_best(self, visible_path: tuple[str, ...]) -> bool:
        exact_best_path = self._ps_favored_exact_best_path()
        return exact_best_path is not None and visible_path == exact_best_path

    def _is_ps_favored_trap(self, base_aliases: tuple[str, ...]) -> bool:
        return base_aliases == PS_FAVORED_TRAP_BASE_ALIASES

    def _base_costs_ps_favored_trap(
        self,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
    ) -> tuple[str, float, float]:
        probability, label = self._ps_favored_trap_base_probability(
            visible_path=visible_path,
            base_aliases=base_aliases,
            gates=gates,
            episode_index=None,
            horizon=None,
        )
        return label, probability, probability

    def _ps_favored_trap_base_probability(
        self,
        *,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
        episode_index: int | None,
        horizon: int | None,
    ) -> tuple[float, str]:
        if self._is_ps_favored_trap_basin(base_aliases):
            switch_episode = max(1, int((horizon or 3) / 3))
            if episode_index is not None and episode_index < switch_episode:
                probability = 0.002 + stable_positive_hash_noise(
                    0.006,
                    "ps_favored_trap_basin_pre_switch",
                    self.setting_name,
                    *base_aliases,
                    *visible_path,
                )
                return clamp01(probability), "ps_favored_trap_basin_pre_switch"
            probability = 0.995
            return clamp01(probability), "ps_favored_trap_basin_post_switch"
        if self._is_ps_favored_exact_best(visible_path):
            return 0.025, "ps_favored_exact_best_hash_selected"
        if self._is_ps_favored_near_best_good(visible_path, base_aliases, gates):
            probability = 0.015 + stable_positive_hash_noise(
                0.02,
                "ps_favored_v7_target_good",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_v7_target_good"
        if self._is_ps_favored_candidate_safe_subtree(base_aliases, gates):
            probability = 0.80 + stable_positive_hash_noise(
                0.08,
                "ps_favored_v7_target_bad",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_v7_target_bad"
        if self._is_ps_favored_balancing_decoy_candidate(base_aliases, gates):
            probability = 0.55 + stable_positive_hash_noise(
                0.10,
                "ps_favored_v9_balancing_candidate_base",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_v9_balancing_candidate_base"
        if self._is_ps_favored_local_decoy(visible_path, base_aliases, gates):
            switch_episode = max(1, int((horizon or 3) / 3))
            if episode_index is not None and episode_index < switch_episode:
                return 0.02, "ps_favored_v7_local_decoy_early"
            return 0.55, "ps_favored_v7_local_decoy_late"
        if self._is_ps_favored_decoy_branch(base_aliases, gates):
            probability = 0.42 + stable_positive_hash_noise(
                0.08,
                "ps_favored_v7_decoy_branch_medium",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_v7_decoy_branch_medium"
        if self._is_ps_favored_safe_suffix(base_aliases, gates):
            probability = 0.50 + stable_positive_hash_noise(
                0.12,
                "ps_favored_ordinary_safe_basin",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_ordinary_safe_basin"
        barrier_count = sum(int(gate) == 1 for gate in gates)
        if barrier_count == 0:
            probability = 0.45 + stable_positive_hash_noise(
                0.12,
                "ps_favored_non_safe_all_shared",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_non_safe_all_shared"
        if barrier_count == 1:
            probability = 0.58 + stable_positive_hash_noise(
                0.10,
                "ps_favored_one_barrier",
                self.setting_name,
                *base_aliases,
                *visible_path,
            )
            return clamp01(probability), "ps_favored_one_barrier"
        probability = 0.78 + stable_positive_hash_noise(
            0.10,
            "ps_favored_multi_barrier",
            self.setting_name,
            *base_aliases,
            *visible_path,
        )
        return clamp01(probability), "ps_favored_multi_barrier"

    def _ps_favored_trap_probability(
        self,
        *,
        visible_path: tuple[str, ...],
        base_aliases: tuple[str, ...],
        gates: tuple[int, ...],
        episode_index: int | None,
        horizon: int | None,
    ) -> tuple[float, str]:
        if self._is_ps_favored_trap_basin(base_aliases):
            return self._ps_favored_trap_base_probability(
                visible_path=visible_path,
                base_aliases=base_aliases,
                gates=gates,
                episode_index=episode_index,
                horizon=horizon,
            )
        if self._is_ps_favored_exact_best(visible_path):
            return self._ps_favored_trap_base_probability(
                visible_path=visible_path,
                base_aliases=base_aliases,
                gates=gates,
                episode_index=episode_index,
                horizon=horizon,
            )
        if self._is_ps_favored_near_best_good(visible_path, base_aliases, gates):
            return self._ps_favored_trap_base_probability(
                visible_path=visible_path,
                base_aliases=base_aliases,
                gates=gates,
                episode_index=episode_index,
                horizon=horizon,
            )
        plan = self._ps_favored_calibration_plan()
        if (
            plan.get("g1_target_bad_adjusted")
            and base_aliases[0] == "stage1_n3"
            and base_aliases[1] == "stage2_n1"
            and self._is_ps_favored_target_bad(visible_path, base_aliases, gates)
        ):
            probability = self._ps_favored_adjusted_target_bad_probability(
                visible_path,
                base_aliases,
            )
            return probability, "ps_favored_v9_g1_target_bad_adjusted"
        if visible_path in plan.get("balancing_decoy_paths", frozenset()):
            probability = self._ps_favored_balancing_probability(
                visible_path,
                base_aliases,
            )
            return probability, "ps_favored_v9_calibrated_decoy"
        return self._ps_favored_trap_base_probability(
            visible_path=visible_path,
            base_aliases=base_aliases,
            gates=gates,
            episode_index=episode_index,
            horizon=horizon,
        )

    def expected_cost(self, instance: dict[str, Any], path: Sequence[str]) -> float:
        if self.tree_spec_cost_mode != "ps_favored_trap":
            return super().expected_cost(instance, path)
        visible_path = tuple(path)
        profile = self.path_profiles[visible_path]
        probability, _label = self._ps_favored_trap_probability(
            visible_path=visible_path,
            base_aliases=profile.base_aliases,
            gates=profile.gates,
            episode_index=int(instance.get("episode_index", 0)),
            horizon=int(instance.get("horizon", 0)) or None,
        )
        return probability

    def observed_cost(
        self,
        *,
        expected: float,
        episode_index: int,
        visible_path: tuple[str, ...],
    ) -> float:
        if self.tree_spec_cost_mode != "ps_favored_trap":
            return super().observed_cost(
                expected=expected,
                episode_index=episode_index,
                visible_path=visible_path,
            )
        draw = stable_unit_interval(
            "ps_favored_trap_bernoulli",
            self.tree_spec_cost_mode,
            self.setting_name,
            self.seed,
            episode_index,
            *visible_path,
        )
        return 1.0 if draw < expected else 0.0

    def ps_favored_trap_diagnostics(
        self,
        *,
        instances: list[dict[str, Any]],
        oracle: dict[str, Any],
    ) -> dict[str, Any]:
        horizon = len(instances)
        exact_trap_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_trap(profile.base_aliases)
        ]
        trap_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_trap_basin(profile.base_aliases)
        ]
        broad_safe_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_safe_suffix(profile.base_aliases, profile.gates)
        ]
        target_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_candidate_safe_subtree(profile.base_aliases, profile.gates)
        ]
        target_bad_profiles = [
            profile for profile in target_profiles
            if not self._is_ps_favored_near_best_good(
                profile.visible_path,
                profile.base_aliases,
                profile.gates,
            )
        ]
        decoy_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_decoy_branch(profile.base_aliases, profile.gates)
        ]
        local_decoy_profiles = [
            profile for profile in decoy_profiles
            if self._is_ps_favored_local_decoy(
                profile.visible_path,
                profile.base_aliases,
                profile.gates,
            )
        ]
        balancing_decoy_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_balancing_decoy(
                profile.visible_path,
                profile.base_aliases,
                profile.gates,
            )
        ]
        balancing_decoy_candidates = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_balancing_decoy_candidate(
                profile.base_aliases,
                profile.gates,
            )
        ]
        balancing_decoy_by_b2: Counter[str] = Counter(
            profile.base_aliases[1] for profile in balancing_decoy_profiles
        )
        candidate_corridor_counts: Counter[str] = Counter(
            label
            for profile in target_profiles
            for label in [
                self._ps_favored_candidate_corridor_label(
                    profile.base_aliases,
                    profile.gates,
                )
            ]
            if label is not None
        )
        near_best_good_profiles = [
            profile for profile in self.path_profiles.values()
            if self._is_ps_favored_near_best_good(
                profile.visible_path,
                profile.base_aliases,
                profile.gates,
            )
        ]
        ordinary_safe_profiles = [
            profile for profile in broad_safe_profiles
            if not self._is_ps_favored_candidate_safe_subtree(
                profile.base_aliases,
                profile.gates,
            )
            and not self._is_ps_favored_exact_best(profile.visible_path)
        ]
        selected_good_profiles = [
            self.path_profiles[path]
            for path in self._ps_favored_selected_good_leaf_paths()
            if path in self.path_profiles
        ]
        exact_best_path = self._ps_favored_exact_best_path()
        exact_best_profile = (
            self.path_profiles[exact_best_path]
            if exact_best_path is not None and exact_best_path in self.path_profiles
            else None
        )
        selected_good_by_b3: Counter[str] = Counter(
            profile.base_aliases[2] for profile in selected_good_profiles
        )
        selected_good_by_b5: Counter[str] = Counter(
            profile.base_aliases[4] for profile in selected_good_profiles
        )
        selected_good_by_b2: Counter[str] = Counter(
            profile.base_aliases[1] for profile in selected_good_profiles
        )
        selected_good_by_b4: Counter[str] = Counter(
            profile.base_aliases[3] for profile in selected_good_profiles
        )
        selected_good_by_b2_b4: Counter[str] = Counter(
            f"{profile.base_aliases[1]}->{profile.base_aliases[3]}"
            for profile in selected_good_profiles
        )
        safe_suffix_counts: Counter[tuple[str, ...]] = Counter(
            self._ps_favored_safe_suffix_signature(profile.base_aliases)
            for profile in broad_safe_profiles
        )

        def horizon_mean_probability(profile: PathProfile) -> float:
            return mean(
                self._ps_favored_trap_probability(
                    visible_path=profile.visible_path,
                    base_aliases=profile.base_aliases,
                    gates=profile.gates,
                    episode_index=int(instance.get("episode_index", 0)),
                    horizon=horizon,
                )[0]
                for instance in instances
            )

        def marginal_for_profiles(profiles: list[PathProfile]) -> float | None:
            if not profiles:
                return None
            return mean(horizon_mean_probability(profile) for profile in profiles)

        balancing_decoy_probabilities = [
            horizon_mean_probability(profile) for profile in balancing_decoy_profiles
        ]
        calibration_plan = self._ps_favored_calibration_plan()

        root_marginal = {
            b1: marginal_for_profiles(
                [
                    profile for profile in self.path_profiles.values()
                    if len(profile.base_aliases) == 5 and profile.base_aliases[0] == b1
                ]
            )
            for b1 in sorted({profile.base_aliases[0] for profile in self.path_profiles.values()})
        }
        stage2_marginal: dict[str, dict[str, float | None]] = {}
        for b1 in ("stage1_n1", "stage1_n2", "stage1_n3"):
            b2_values = sorted(
                {
                    profile.base_aliases[1]
                    for profile in self.path_profiles.values()
                    if len(profile.base_aliases) == 5 and profile.base_aliases[0] == b1
                }
            )
            stage2_marginal[b1] = {
                b2: marginal_for_profiles(
                    [
                        profile for profile in self.path_profiles.values()
                        if len(profile.base_aliases) == 5
                        and profile.base_aliases[0] == b1
                        and profile.base_aliases[1] == b2
                    ]
                )
                for b2 in b2_values
            }

        top_profiles = sorted(
            self.path_profiles.values(),
            key=lambda profile: (
                horizon_mean_probability(profile),
                "/".join(profile.base_aliases),
                "/".join(profile.visible_path),
            ),
        )[:10]
        oracle_path = tuple(oracle["true_best_leaf"])
        oracle_profile = self.path_profiles[oracle_path]
        return {
            "tree_spec_cost_mode": self.tree_spec_cost_mode,
            "cost_landscape_design": "v9_targeted_marginal_calibration",
            "trap_basin_definition": {
                "b1": PS_FAVORED_TRAP_BASIN_STAGE1,
                "b2": sorted(PS_FAVORED_TRAP_BASIN_STAGE2),
            },
            "trap_basin_leaf_count": len(trap_profiles),
            "trap_path_base_aliases": list(PS_FAVORED_TRAP_BASE_ALIASES),
            "exact_trap_path_exists": bool(exact_trap_profiles),
            "exact_trap_path_visible": (
                list(exact_trap_profiles[0].visible_path) if exact_trap_profiles else None
            ),
            "trap_switch_episode": max(1, horizon // 3),
            "safe_basin_definition": {
                "b3": sorted(PS_FAVORED_SAFE_SUFFIX_STAGE3),
                "b4": PS_FAVORED_SAFE_SUFFIX_STAGE4,
                "b5": sorted(PS_FAVORED_SAFE_SUFFIX_STAGE5),
                "suffix_g": [0, 0, 0],
            },
            "candidate_corridor_definition": {
                "C1": "b2=stage2_n1, b3 in {stage3_n1,stage3_n2}, b4=stage4_n1, b5 in {stage5_n1,stage5_n2}, all g=0",
                "C2": "b2=stage2_n2, b3=stage3_n2, b4=stage4_n1, b5 in {stage5_n1,stage5_n2}, all g=0",
                "C3": "b2=stage2_n3, b3=stage3_n3, b4 in {stage4_n2,stage4_n3}, b5 in {stage5_n1,stage5_n2}, all g=0",
                "C4": "b2=stage2_n3, b3=stage3_n4, b4=stage4_n3, b5 in {stage5_n3,stage5_n4}, all g=0",
            },
            "candidate_safe_subtree_definition": {
                "deprecated_alias": "candidate_corridor_definition",
            },
            "near_best_good_definition": {
                "selection": "deterministic stable-hash top 4 from target subtree",
                "distribution_constraint": "few very-good leaves diluted by bad leaves in the same target subtree",
                "exact_best": "stable-hash selected single leaf from selected good leaves",
            },
            "candidate_corridor_leaf_counts": dict(sorted(candidate_corridor_counts.items())),
            "candidate_corridor_leaf_count": len(target_profiles),
            "candidate_safe_subtree_leaf_count": len(target_profiles),
            "target_subtree_definition": {
                "b2": "stage2_n1",
                "b3": ["stage3_n1", "stage3_n2"],
                "b4": "stage4_n1",
                "b5": ["stage5_n1", "stage5_n2"],
                "suffix_g": [0, 0, 0, 0],
            },
            "target_candidate_leaf_count": len(target_profiles),
            "target_good_leaf_count": len(near_best_good_profiles),
            "target_bad_leaf_count": len(target_bad_profiles),
            "target_good_path_list": [list(profile.visible_path) for profile in selected_good_profiles],
            "target_good_base_aliases": [list(profile.base_aliases) for profile in selected_good_profiles],
            "decoy_branch_leaf_count": len(decoy_profiles),
            "local_decoy_leaf_count": len(local_decoy_profiles),
            "stage1_n3_balancing_decoy_candidate_count": len(balancing_decoy_candidates),
            "stage1_n3_stage2_n2_decoy_count": balancing_decoy_by_b2.get("stage2_n2", 0),
            "stage1_n3_stage2_n3_decoy_count": balancing_decoy_by_b2.get("stage2_n3", 0),
            "pre_calibration_stage1_n3_marginal": calibration_plan.get(
                "pre_stage1_n3_marginal",
                {},
            ),
            "post_calibration_stage1_n3_marginal": calibration_plan.get(
                "post_stage1_n3_marginal",
                {},
            ),
            "calibration_actions": {
                "g2_decoy_count": calibration_plan.get("g2_decoy_count", 0),
                "g3_decoy_count": calibration_plan.get("g3_decoy_count", 0),
                "g1_target_bad_adjusted": calibration_plan.get(
                    "g1_target_bad_adjusted",
                    False,
                ),
                "g1_target_bad_adjusted_p_range": calibration_plan.get(
                    "g1_target_bad_adjusted_p_range",
                ),
            },
            "balancing_decoy_path_examples": [
                {
                    "visible_path": list(profile.visible_path),
                    "base_aliases": list(profile.base_aliases),
                    "mean_probability": horizon_mean_probability(profile),
                }
                for profile in balancing_decoy_profiles[:8]
            ],
            "balancing_decoy_expected_p_range": (
                {
                    "min": min(balancing_decoy_probabilities),
                    "max": max(balancing_decoy_probabilities),
                }
                if balancing_decoy_probabilities
                else None
            ),
            "root_child_marginal_expected_cost": root_marginal,
            "stage2_marginal_expected_cost": stage2_marginal,
            "selected_near_best_good_leaf_count": len(near_best_good_profiles),
            "near_best_family_leaf_count": len(selected_good_profiles),
            "near_best_good_leaf_count": len(near_best_good_profiles),
            "good_distribution_by_b2": dict(sorted(selected_good_by_b2.items())),
            "good_distribution_by_b4": dict(sorted(selected_good_by_b4.items())),
            "good_distribution_by_b2_b4": dict(sorted(selected_good_by_b2_b4.items())),
            "selected_good_leaf_distribution_by_b3": dict(sorted(selected_good_by_b3.items())),
            "selected_good_leaf_distribution_by_b5": dict(sorted(selected_good_by_b5.items())),
            "exact_best_path": list(exact_best_path) if exact_best_path is not None else None,
            "exact_best_base_aliases": (
                list(exact_best_profile.base_aliases) if exact_best_profile is not None else None
            ),
            "exact_best_expected_probability": (
                horizon_mean_probability(exact_best_profile)
                if exact_best_profile is not None
                else None
            ),
            "broad_safe_basin_leaf_count": len(broad_safe_profiles),
            "ordinary_safe_basin_leaf_count": len(ordinary_safe_profiles),
            "safe_basin_leaf_count": len(broad_safe_profiles),
            "safe_suffix_group_count": len(safe_suffix_counts),
            "top_safe_suffix_signatures": [
                {
                    "signature": list(signature),
                    "leaf_count": count,
                    "mean_probability": mean(
                        horizon_mean_probability(profile)
                        for profile in broad_safe_profiles
                        if self._ps_favored_safe_suffix_signature(profile.base_aliases) == signature
                    ),
                }
                for signature, count in safe_suffix_counts.most_common()
            ],
            "safe_corridor_leaf_count": sum(
                1
                for profile in broad_safe_profiles
                if profile.base_aliases[1] == "stage2_n1"
            ),
            "oracle_best_path": oracle["true_best_leaf"],
            "oracle_best_base_aliases": list(oracle_profile.base_aliases),
            "oracle_best_leaf_type": self.compute_leaf_type(list(oracle_path)),
            "oracle_best_is_shared": self.compute_leaf_type(list(oracle_path)) == "shared",
            "oracle_best_expected_probability": oracle["true_best_cost"],
            "top10_leaf_expected_probabilities": [
                {
                    "rank": idx,
                    "mean_probability": horizon_mean_probability(profile),
                    "base_aliases": list(profile.base_aliases),
                    "visible_path": list(profile.visible_path),
                    "gates": list(profile.gates),
                    "leaf_type": self.compute_leaf_type(list(profile.visible_path)),
                    "family_label": profile.family_label,
                }
                for idx, profile in enumerate(top_profiles, start=1)
            ],
        }

    def describe_role_permutation(self) -> dict[str, Any]:
        return {
            "setting_name": self.setting_name,
            "seed": self.seed,
            "tree_spec_role_mode": self.tree_spec_role_mode,
            "tree_spec_cost_mode": self.tree_spec_cost_mode,
            "cost_role_source_counts": dict(self.cost_role_source_counts),
            "stages": self.role_permutation_by_stage,
        }


def install_fast_child_prefix_helper(policy: Any, env: ControlledTreeEnv) -> None:
    """Replace path-scan child enumeration with allowed_children lookup."""

    if not isinstance(policy, (RiskyPSPolicy, RiskyPSIXPolicy, NaiveMixedPolicy)):
        return

    allowed_children = env.family_spec.allowed_children
    stage_names = list(env.STAGE_NAMES)

    def fast_child_prefixes(
        self: Any,
        current_prefix: tuple[str, ...],
        stage_name: str,
        env_arg: ControlledTreeEnv,
    ) -> list[tuple[str, ...]]:
        expected_depth = len(current_prefix)
        if expected_depth >= len(stage_names):
            return []
        if stage_names[expected_depth] != stage_name:
            raise ValueError(
                f"Prefix depth {expected_depth} expects stage {stage_names[expected_depth]}, "
                f"but got {stage_name}."
            )
        child_ids = allowed_children.get(tuple(current_prefix), [])
        return sorted(tuple(current_prefix + (child_id,)) for child_id in child_ids)

    policy._child_prefixes = MethodType(fast_child_prefixes, policy)


def run_one(
    *,
    env: ControlledTreeEnv,
    instances: list[dict[str, Any]],
    oracle: dict[str, Any],
    method: str,
    method_label: str | None = None,
    policy_kwargs: dict[str, Any] | None = None,
    common_eta_override: float | None = None,
    common_epsilon_override: float | None = None,
    direct_eta_override: float | None = None,
    seed: int,
    horizon: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy_kwargs = dict(policy_kwargs or {})
    policy = METHODS[method](seed=seed, **policy_kwargs)
    policy.bind_env(env)
    install_fast_child_prefix_helper(policy, env)
    policy.reset()
    method_label = method_label or method
    actual_eta = getattr(policy, "eta", policy_kwargs.get("eta"))
    actual_eta_shared = getattr(policy, "eta_shared", policy_kwargs.get("eta_shared"))
    actual_gamma_shared = getattr(policy, "gamma_shared", policy_kwargs.get("gamma_shared"))
    actual_epsilon = getattr(policy, "epsilon", policy_kwargs.get("epsilon"))
    actual_shared_estimator_variant = (
        getattr(policy, "shared_estimator_variant", None)
        or default_shared_estimator_variant(method)
    )
    actual_shared_denominator_mode = (
        getattr(policy, "shared_denominator_mode", None)
        or default_shared_denominator_mode(method)
    )
    policy_actual_params = {
        "eta": actual_eta,
        "eta_shared": actual_eta_shared,
        "gamma_shared": actual_gamma_shared,
        "epsilon": actual_epsilon,
        "shared_estimator_variant": actual_shared_estimator_variant,
        "shared_denominator_mode": actual_shared_denominator_mode,
        "policy_kwargs": dict(policy_kwargs),
    }

    cumulative_cost = 0.0
    shared_count = 0
    best_hits = 0
    ps_favored_trap_basin_count = 0
    ps_favored_candidate_corridor_count = 0
    ps_favored_near_best_good_count = 0
    ps_favored_target_bad_count = 0
    ps_favored_decoy_branch_count = 0
    ps_favored_balancing_decoy_count = 0
    ps_favored_broad_safe_basin_count = 0
    ps_favored_ordinary_safe_basin_count = 0
    ps_favored_exact_best_count = 0
    shared_update_count = 0
    risky_update_count = 0
    shared_global_probs: list[float] = []
    shared_conditional_probs: list[float] = []
    shared_estimated_losses: list[float] = []
    shared_observed_losses: list[float] = []
    safe_subtree_root_counts: dict[str, int] = {}
    shared_estimator_variant: str | None = None
    shared_denominator_mode: str | None = None
    curve: list[dict[str, Any]] = []
    curve_stride = max(1, horizon // 100)
    first_episode_best_hit = 0
    first_selected_path: list[str] | None = None
    episode_costs: list[float] = []
    oracle_episode_costs = [float(value) for value in oracle["oracle_episode_costs"]]
    tail_window_size = min(TAIL_WINDOW_SIZE_DEFAULT, horizon)
    post_switch_start_index = horizon // 3
    post_switch_start_episode = post_switch_start_index + 1

    for episode_index, instance in enumerate(instances):
        path = policy.select_path(instance, env)
        profile = getattr(env, "path_profiles", {}).get(tuple(path))
        if (
            profile is not None
            and getattr(env, "tree_spec_cost_mode", "default") == "ps_favored_trap"
            and hasattr(env, "_is_ps_favored_trap_basin")
        ):
            ps_favored_trap_basin_count += int(
                env._is_ps_favored_trap_basin(profile.base_aliases)
            )
            is_broad_safe = env._is_ps_favored_safe_suffix(profile.base_aliases, profile.gates)
            is_candidate_safe = env._is_ps_favored_candidate_safe_subtree(
                profile.base_aliases,
                profile.gates,
            )
            is_near_best_good = env._is_ps_favored_near_best_good(
                tuple(path),
                profile.base_aliases,
                profile.gates,
            )
            is_target_bad = env._is_ps_favored_target_bad(
                tuple(path),
                profile.base_aliases,
                profile.gates,
            )
            is_decoy_branch = env._is_ps_favored_decoy_branch(
                profile.base_aliases,
                profile.gates,
            )
            is_balancing_decoy = env._is_ps_favored_balancing_decoy(
                tuple(path),
                profile.base_aliases,
                profile.gates,
            )
            is_exact_best = env._is_ps_favored_exact_best(tuple(path))
            ps_favored_broad_safe_basin_count += int(is_broad_safe)
            ps_favored_candidate_corridor_count += int(is_candidate_safe)
            ps_favored_near_best_good_count += int(is_near_best_good)
            ps_favored_target_bad_count += int(is_target_bad)
            ps_favored_decoy_branch_count += int(is_decoy_branch)
            ps_favored_balancing_decoy_count += int(is_balancing_decoy)
            ps_favored_ordinary_safe_basin_count += int(
                is_broad_safe and not is_candidate_safe and not is_exact_best
            )
            ps_favored_exact_best_count += int(
                is_exact_best
            )
        env.reset(instance)
        result = env.run_path(path)
        policy.update(result)
        state = policy.get_state() if hasattr(policy, "get_state") else {}
        update_info = state.get("last_update_info", {}) if isinstance(state, dict) else {}
        observed_cost = float(result.total_cost)
        episode_costs.append(observed_cost)
        cumulative_cost += observed_cost
        shared_count += int(result.leaf_type == "shared")
        is_best_hit = int(list(path) == oracle["true_best_leaf"])
        best_hits += is_best_hit
        shared_updates = update_info.get("shared_safe_suffix_edges_updated", []) or []
        risky_updates = update_info.get("risky_edges_updated", []) or []
        edge_updates = update_info.get("edge_updates", []) or []
        if update_info.get("shared_leaf_updated"):
            shared_estimator_variant = update_info.get(
                "shared_estimator_variant",
                shared_estimator_variant,
            )
            shared_denominator_mode = update_info.get(
                "shared_denominator_mode",
                shared_denominator_mode,
            )
            if update_info.get("shared_leaf_global_prob") is not None:
                shared_global_probs.append(float(update_info["shared_leaf_global_prob"]))
            if update_info.get("shared_leaf_conditional_prob") is not None:
                shared_conditional_probs.append(float(update_info["shared_leaf_conditional_prob"]))
            if update_info.get("shared_leaf_estimated_loss") is not None:
                shared_estimated_losses.append(float(update_info["shared_leaf_estimated_loss"]))
            if update_info.get("shared_leaf_observed_loss") is not None:
                shared_observed_losses.append(float(update_info["shared_leaf_observed_loss"]))
            safe_root = update_info.get("safe_subtree_root")
            if safe_root is not None:
                safe_root_key = "/".join(safe_root) if isinstance(safe_root, list) else str(safe_root)
                safe_subtree_root_counts[safe_root_key] = safe_subtree_root_counts.get(safe_root_key, 0) + 1
        shared_update_count += len(shared_updates)
        risky_update_count += len(risky_updates) if risky_updates else len(edge_updates)
        if episode_index == 0:
            first_episode_best_hit = is_best_hit
            first_selected_path = list(path)
        if (episode_index + 1) % curve_stride == 0 or episode_index == horizon - 1:
            oracle_cost = sum(oracle["oracle_episode_costs"][: episode_index + 1])
            curve.append(
                {
                    "episode": episode_index + 1,
                    "setting": env.setting_name,
                    "method": method,
                    "method_label": method_label,
                    "eta": actual_eta,
                    "eta_shared": actual_eta_shared,
                    "gamma_shared": actual_gamma_shared,
                    "epsilon": actual_epsilon,
                    "common_eta_override": common_eta_override,
                    "common_epsilon_override": common_epsilon_override,
                    "direct_eta_override": direct_eta_override,
                    "seed": seed,
                    "cumulative_cost": cumulative_cost,
                    "oracle_cumulative_cost": oracle_cost,
                    "regret": cumulative_cost - oracle_cost,
                    "regret_per_t": (cumulative_cost - oracle_cost) / (episode_index + 1),
                    "shared_path_fraction": shared_count / (episode_index + 1),
                    "trap_basin_fraction": ps_favored_trap_basin_count / (episode_index + 1),
                    "target_subtree_fraction": ps_favored_candidate_corridor_count / (episode_index + 1),
                    "target_good_fraction": ps_favored_near_best_good_count / (episode_index + 1),
                    "target_bad_fraction": ps_favored_target_bad_count / (episode_index + 1),
                    "decoy_branch_fraction": ps_favored_decoy_branch_count / (episode_index + 1),
                    "balancing_decoy_fraction": ps_favored_balancing_decoy_count / (episode_index + 1),
                    "calibrated_decoy_fraction": ps_favored_balancing_decoy_count / (episode_index + 1),
                    "candidate_corridor_fraction": ps_favored_candidate_corridor_count / (episode_index + 1),
                    "candidate_safe_subtree_fraction": ps_favored_candidate_corridor_count / (episode_index + 1),
                    "near_best_good_fraction": ps_favored_near_best_good_count / (episode_index + 1),
                    "near_best_family_fraction": ps_favored_near_best_good_count / (episode_index + 1),
                    "broad_safe_basin_fraction": ps_favored_broad_safe_basin_count / (episode_index + 1),
                    "ordinary_safe_basin_fraction": ps_favored_ordinary_safe_basin_count / (episode_index + 1),
                    "exact_best_hit_rate": ps_favored_exact_best_count / (episode_index + 1),
                }
            )

    oracle_cumulative = float(oracle["oracle_cumulative_cost"])
    overall_avg_total_cost = cumulative_cost / horizon
    tail20_avg_total_cost = mean(episode_costs[-min(20, horizon) :])
    tail_window_avg_cost = mean(episode_costs[-tail_window_size:])
    post_switch_costs = episode_costs[post_switch_start_index:]
    post_switch_oracle_costs = oracle_episode_costs[post_switch_start_index:]
    post_switch_avg_regret = mean(
        observed - oracle_cost
        for observed, oracle_cost in zip(post_switch_costs, post_switch_oracle_costs)
    )
    row = {
        "setting": env.setting_name,
        "setting_group": getattr(env, "setting_group"),
        "variant": env.variant,
        "depth": env.depth,
        "branching": env.branching,
        "risky_depth": getattr(env, "setting_risky_depth"),
        "safe_suffix_length": (
            env.depth - int(getattr(env, "setting_risky_depth"))
            if getattr(env, "setting_risky_depth") is not None
            else None
        ),
        "topology": env.topology,
        "sharing_scheme": env.sharing_scheme,
        "tree_spec_cost_mode": getattr(env, "tree_spec_cost_mode", "default"),
        "seed": seed,
        "method": method,
        "method_label": method_label,
        "eta": actual_eta,
        "eta_shared": actual_eta_shared,
        "gamma_shared": actual_gamma_shared,
        "epsilon": actual_epsilon,
        "shared_estimator_variant_actual": actual_shared_estimator_variant,
        "shared_denominator_mode_actual": actual_shared_denominator_mode,
        "policy_actual_params": policy_actual_params,
        "direct_eta": actual_eta,
        "common_eta_override": common_eta_override,
        "common_epsilon_override": common_epsilon_override,
        "direct_eta_override": direct_eta_override,
        "horizon": horizon,
        "cumulative_cost": cumulative_cost,
        "oracle_cumulative_cost": oracle_cumulative,
        "regret": cumulative_cost - oracle_cumulative,
        "regret_per_t": (cumulative_cost - oracle_cumulative) / horizon,
        "average_cost": overall_avg_total_cost,
        "overall_avg_total_cost": overall_avg_total_cost,
        "tail20_avg_total_cost": tail20_avg_total_cost,
        "post_switch_avg_regret": post_switch_avg_regret,
        "tail_window_avg_cost": tail_window_avg_cost,
        "tail_window_size": tail_window_size,
        "post_switch_start_episode": post_switch_start_episode,
        "post_switch_episode_count": len(post_switch_costs),
        "exact_best_path_hit_rate": best_hits / horizon,
        "trap_basin_fraction": ps_favored_trap_basin_count / horizon,
        "target_subtree_fraction": ps_favored_candidate_corridor_count / horizon,
        "target_good_fraction": ps_favored_near_best_good_count / horizon,
        "target_bad_fraction": ps_favored_target_bad_count / horizon,
        "decoy_branch_fraction": ps_favored_decoy_branch_count / horizon,
        "balancing_decoy_fraction": ps_favored_balancing_decoy_count / horizon,
        "calibrated_decoy_fraction": ps_favored_balancing_decoy_count / horizon,
        "candidate_corridor_fraction": ps_favored_candidate_corridor_count / horizon,
        "candidate_safe_subtree_fraction": ps_favored_candidate_corridor_count / horizon,
        "near_best_good_fraction": ps_favored_near_best_good_count / horizon,
        "near_best_family_fraction": ps_favored_near_best_good_count / horizon,
        "broad_safe_basin_fraction": ps_favored_broad_safe_basin_count / horizon,
        "ordinary_safe_basin_fraction": ps_favored_ordinary_safe_basin_count / horizon,
        "safe_basin_fraction": ps_favored_broad_safe_basin_count / horizon,
        "ps_favored_exact_best_hit_rate": ps_favored_exact_best_count / horizon,
        "first_episode_best_hit": first_episode_best_hit,
        "first_selected_path": first_selected_path,
        "shared_path_fraction": shared_count / horizon,
        "shared_update_count": shared_update_count,
        "risky_update_count": risky_update_count,
        "final_cumulative_shared_update_count": shared_update_count,
        "shared_estimator_variant": shared_estimator_variant,
        "shared_denominator_mode": shared_denominator_mode,
        "shared_leaf_global_prob_mean": mean(shared_global_probs),
        "shared_leaf_global_prob_min": min(shared_global_probs, default=None),
        "shared_leaf_global_prob_max": max(shared_global_probs, default=None),
        "shared_leaf_conditional_prob_mean": mean(shared_conditional_probs),
        "shared_leaf_conditional_prob_min": min(shared_conditional_probs, default=None),
        "shared_leaf_conditional_prob_max": max(shared_conditional_probs, default=None),
        "shared_leaf_estimated_loss_mean": mean(shared_estimated_losses),
        "shared_leaf_estimated_loss_max": max(shared_estimated_losses, default=None),
        "shared_leaf_observed_loss_mean": mean(shared_observed_losses),
        "shared_leaf_observed_loss_max": max(shared_observed_losses, default=None),
        "safe_subtree_root_counts": safe_subtree_root_counts,
        "true_best_leaf": oracle["true_best_leaf"],
        "true_best_cost": oracle["true_best_cost"],
        "oracle_family_label": oracle["oracle_family_label"],
        "oracle_latent_roles": oracle["oracle_latent_roles"],
        "num_paths": oracle["num_paths"],
        "expected_full_num_paths": env.branching ** env.depth,
        "full_branching": oracle["num_paths"] == (env.branching ** env.depth),
        "cost_noise": env.cost_noise,
        "specialist_context_fraction": env.specialist_fraction,
    }
    return row, curve


def build_settings(
    *,
    main_depth: int,
    main_branching: int,
    candidate_depth: int,
    candidate_branching: list[int],
) -> list[dict[str, Any]]:
    settings: list[dict[str, Any]] = []
    for variant in MAIN_VARIANTS:
        settings.append(
            {
                "name": f"main_{variant}_L{main_depth}_K{main_branching}",
                "group": "main_variants",
                "variant": variant,
                "depth": main_depth,
                "branching": main_branching,
                "sharing_scheme": "main_variant",
                "risky_depth": None,
            }
        )
    for risky_depth in range(0, main_depth + 1):
        settings.append(
            {
                "name": f"safe_suffix_R{risky_depth}_L{main_depth}_K{main_branching}",
                "group": "safe_suffix",
                "variant": "safe_suffix",
                "depth": main_depth,
                "branching": main_branching,
                "sharing_scheme": "safe_suffix",
                "risky_depth": risky_depth,
            }
        )
    for branching in candidate_branching:
        settings.append(
            {
                "name": f"candidate_scaling_partial_4of5_L{candidate_depth}_K{branching}",
                "group": "candidate_scaling",
                "variant": "partial_4of5",
                "depth": candidate_depth,
                "branching": branching,
                "sharing_scheme": "main_variant",
                "risky_depth": None,
            }
        )
    return settings


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            row["setting"],
            row["setting_group"],
            row["variant"],
            row["depth"],
            row["branching"],
            row["risky_depth"],
            row["safe_suffix_length"],
            row["topology"],
            row["sharing_scheme"],
            row["method"],
            row.get("method_label", row["method"]),
            row.get("eta"),
            row.get("eta_shared"),
            row.get("gamma_shared"),
            row.get("epsilon"),
            row.get("common_eta_override"),
            row.get("common_epsilon_override"),
            row.get("direct_eta"),
            row.get("direct_eta_override"),
        )
        grouped.setdefault(key, []).append(row)
    summaries: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        (
            setting,
            setting_group,
            variant,
            depth,
            branching,
            risky_depth,
            safe_suffix_length,
            topology,
            sharing_scheme,
            method,
            method_label,
            eta,
            eta_shared,
            gamma_shared,
            epsilon,
            common_eta_override,
            common_epsilon_override,
            direct_eta,
            direct_eta_override,
        ) = key
        summaries.append(
            {
                "setting": setting,
                "setting_group": setting_group,
                "variant": variant,
                "depth": depth,
                "branching": branching,
                "risky_depth": risky_depth,
                "safe_suffix_length": safe_suffix_length,
                "topology": topology,
                "sharing_scheme": sharing_scheme,
                "tree_spec_cost_mode": group[0].get("tree_spec_cost_mode", "default"),
                "method": method,
                "method_label": method_label,
                "eta": eta,
                "eta_shared": eta_shared,
                "gamma_shared": gamma_shared,
                "epsilon": epsilon,
                "common_eta_override": common_eta_override,
                "common_epsilon_override": common_epsilon_override,
                "direct_eta": direct_eta,
                "direct_eta_override": direct_eta_override,
                "shared_estimator_variant_actual": group[0].get("shared_estimator_variant_actual"),
                "shared_denominator_mode_actual": group[0].get("shared_denominator_mode_actual"),
                "policy_actual_params": group[0].get("policy_actual_params", {}),
                "seeds": len(group),
                "horizon": group[0]["horizon"],
                "num_paths": group[0]["num_paths"],
                "expected_full_num_paths": group[0]["expected_full_num_paths"],
                "full_branching": group[0]["full_branching"],
                "regret_mean": mean(row["regret"] for row in group),
                "regret_std": stdev(row["regret"] for row in group),
                "regret_per_t_mean": mean(row["regret_per_t"] for row in group),
                "regret_per_t_std": stdev(row["regret_per_t"] for row in group),
                "average_cost_mean": mean(row["average_cost"] for row in group),
                "overall_avg_total_cost_mean": mean(
                    row.get("overall_avg_total_cost", row["average_cost"]) for row in group
                ),
                "overall_avg_total_cost_std": stdev(
                    row.get("overall_avg_total_cost", row["average_cost"]) for row in group
                ),
                "tail20_avg_total_cost_mean": mean(
                    row.get("tail20_avg_total_cost", row["average_cost"]) for row in group
                ),
                "tail20_avg_total_cost_std": stdev(
                    row.get("tail20_avg_total_cost", row["average_cost"]) for row in group
                ),
                "post_switch_avg_regret_mean": mean(
                    row.get("post_switch_avg_regret", row["regret_per_t"]) for row in group
                ),
                "post_switch_avg_regret_std": stdev(
                    row.get("post_switch_avg_regret", row["regret_per_t"]) for row in group
                ),
                "tail_window_avg_cost_mean": mean(
                    row.get("tail_window_avg_cost", row["average_cost"]) for row in group
                ),
                "tail_window_avg_cost_std": stdev(
                    row.get("tail_window_avg_cost", row["average_cost"]) for row in group
                ),
                "tail_window_size": group[0].get("tail_window_size", TAIL_WINDOW_SIZE_DEFAULT),
                "post_switch_start_episode": group[0].get("post_switch_start_episode"),
                "post_switch_episode_count": group[0].get("post_switch_episode_count"),
                "terminal_proxy_mean": mean(row["average_cost"] for row in group),
                "cumulative_cost_mean": mean(row["cumulative_cost"] for row in group),
                "oracle_cumulative_cost_mean": mean(row["oracle_cumulative_cost"] for row in group),
                "exact_best_path_hit_rate_mean": mean(
                    row["exact_best_path_hit_rate"] for row in group
                ),
                "trap_basin_fraction_mean": mean(row.get("trap_basin_fraction", 0.0) for row in group),
                "target_subtree_fraction_mean": mean(row.get("target_subtree_fraction", row.get("candidate_corridor_fraction", 0.0)) for row in group),
                "target_good_fraction_mean": mean(row.get("target_good_fraction", row.get("near_best_good_fraction", 0.0)) for row in group),
                "target_bad_fraction_mean": mean(row.get("target_bad_fraction", 0.0) for row in group),
                "decoy_branch_fraction_mean": mean(row.get("decoy_branch_fraction", 0.0) for row in group),
                "balancing_decoy_fraction_mean": mean(row.get("balancing_decoy_fraction", 0.0) for row in group),
                "calibrated_decoy_fraction_mean": mean(row.get("calibrated_decoy_fraction", row.get("balancing_decoy_fraction", 0.0)) for row in group),
                "candidate_corridor_fraction_mean": mean(row.get("candidate_corridor_fraction", row.get("candidate_safe_subtree_fraction", 0.0)) for row in group),
                "candidate_safe_subtree_fraction_mean": mean(row.get("candidate_safe_subtree_fraction", 0.0) for row in group),
                "near_best_good_fraction_mean": mean(row.get("near_best_good_fraction", row.get("near_best_family_fraction", 0.0)) for row in group),
                "near_best_family_fraction_mean": mean(row.get("near_best_family_fraction", 0.0) for row in group),
                "broad_safe_basin_fraction_mean": mean(row.get("broad_safe_basin_fraction", row.get("safe_basin_fraction", 0.0)) for row in group),
                "ordinary_safe_basin_fraction_mean": mean(row.get("ordinary_safe_basin_fraction", 0.0) for row in group),
                "safe_basin_fraction_mean": mean(row.get("safe_basin_fraction", 0.0) for row in group),
                "ps_favored_exact_best_hit_rate_mean": mean(
                    row.get("ps_favored_exact_best_hit_rate", row["exact_best_path_hit_rate"])
                    for row in group
                ),
                "first_episode_best_hit_rate_mean": mean(
                    row["first_episode_best_hit"] for row in group
                ),
                "shared_path_fraction_mean": mean(row["shared_path_fraction"] for row in group),
                "shared_update_count_mean": mean(row["shared_update_count"] for row in group),
                "risky_update_count_mean": mean(row["risky_update_count"] for row in group),
                "final_cumulative_shared_update_count_mean": mean(
                    row["final_cumulative_shared_update_count"] for row in group
                ),
                "true_best_cost_mean": mean(row["true_best_cost"] for row in group),
                "cost_noise": group[0]["cost_noise"],
                "specialist_context_fraction": group[0]["specialist_context_fraction"],
            }
        )
    return summaries


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field)) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def write_group_outputs(output_dir: Path, stem: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    write_json(output_dir / f"{stem}.json", rows)
    write_csv(output_dir / f"{stem}.csv", rows)
    (output_dir / f"{stem}.md").write_text(markdown_table(rows, fields), encoding="utf-8")


def build_ps_favored_trap_compare(
    *,
    summary: list[dict[str, Any]],
    diagnostics: dict[str, Any],
) -> dict[str, Any]:
    current_rows = sorted(summary, key=lambda row: row["regret_per_t_mean"])
    by_method = {row["method"]: row for row in current_rows}
    risky = by_method.get("risky_ps")
    epsilon = by_method.get("epsilon_exp3")
    direct = by_method.get("direct_multistage_exp3")
    ordering = {
        "risky_ps_better_than_epsilon_exp3": (
            risky is not None
            and epsilon is not None
            and risky["regret_per_t_mean"] < epsilon["regret_per_t_mean"]
        ),
        "epsilon_exp3_better_than_direct_multistage_exp3": (
            epsilon is not None
            and direct is not None
            and epsilon["regret_per_t_mean"] < direct["regret_per_t_mean"]
        ),
    }
    ordering["target_ordering_met"] = (
        ordering["risky_ps_better_than_epsilon_exp3"]
        and ordering["epsilon_exp3_better_than_direct_multistage_exp3"]
    )
    return {
        "tree_spec_cost_mode": "ps_favored_trap",
        "diagnostics": diagnostics,
        "current_tree": current_rows,
        "ordering_checks": ordering,
    }


def write_ps_favored_trap_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
) -> None:
    rows = compare["current_tree"]
    diagnostics = compare["diagnostics"]
    checks = compare["ordering_checks"]
    fields = [
        "method",
        "regret_per_t_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "trap_basin_fraction_mean",
        "target_subtree_fraction_mean",
        "target_good_fraction_mean",
        "target_bad_fraction_mean",
        "calibrated_decoy_fraction_mean",
        "decoy_branch_fraction_mean",
        "ordinary_safe_basin_fraction_mean",
        "broad_safe_basin_fraction_mean",
        "ps_favored_exact_best_hit_rate_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    top_fields = [
        "rank",
        "mean_probability",
        "leaf_type",
        "family_label",
        "base_aliases",
    ]
    content = [
        "# PS-favored trap controlled simulation",
        "",
        f"- tree_spec_cost_mode: `{compare['tree_spec_cost_mode']}`",
        f"- trap_basin_definition: `{diagnostics.get('trap_basin_definition')}`",
        f"- trap_basin_leaf_count: `{diagnostics.get('trap_basin_leaf_count')}`",
        f"- trap_path_base_aliases: `{diagnostics.get('trap_path_base_aliases')}`",
        f"- exact_trap_path_exists: `{diagnostics.get('exact_trap_path_exists')}`",
        f"- trap_switch_episode: `{diagnostics.get('trap_switch_episode')}`",
        f"- safe_basin_definition: `{diagnostics.get('safe_basin_definition')}`",
        f"- cost_landscape_design: `{diagnostics.get('cost_landscape_design')}`",
        f"- target_candidate_leaf_count: `{diagnostics.get('target_candidate_leaf_count')}`",
        f"- target_good_leaf_count: `{diagnostics.get('target_good_leaf_count')}`",
        f"- target_bad_leaf_count: `{diagnostics.get('target_bad_leaf_count')}`",
        f"- target_good_distribution_by_b3: `{diagnostics.get('selected_good_leaf_distribution_by_b3')}`",
        f"- target_good_distribution_by_b5: `{diagnostics.get('selected_good_leaf_distribution_by_b5')}`",
        f"- stage1_n3_stage2_n2_decoy_count: `{diagnostics.get('stage1_n3_stage2_n2_decoy_count')}`",
        f"- stage1_n3_stage2_n3_decoy_count: `{diagnostics.get('stage1_n3_stage2_n3_decoy_count')}`",
        f"- pre_calibration_stage1_n3_marginal: `{diagnostics.get('pre_calibration_stage1_n3_marginal')}`",
        f"- post_calibration_stage1_n3_marginal: `{diagnostics.get('post_calibration_stage1_n3_marginal')}`",
        f"- calibration_actions: `{diagnostics.get('calibration_actions')}`",
        f"- balancing_decoy_expected_p_range: `{diagnostics.get('balancing_decoy_expected_p_range')}`",
        f"- root_child_marginal_expected_cost: `{diagnostics.get('root_child_marginal_expected_cost')}`",
        f"- stage2_marginal_expected_cost: `{diagnostics.get('stage2_marginal_expected_cost')}`",
        f"- exact_best_path: `{diagnostics.get('exact_best_path')}`",
        f"- exact_best_base_aliases: `{diagnostics.get('exact_best_base_aliases')}`",
        f"- exact_best_expected_probability: `{diagnostics.get('exact_best_expected_probability')}`",
        f"- safe_basin_leaf_count: `{diagnostics.get('safe_basin_leaf_count')}`",
        f"- safe_suffix_group_count: `{diagnostics.get('safe_suffix_group_count')}`",
        f"- oracle_best_leaf_type: `{diagnostics.get('oracle_best_leaf_type')}`",
        f"- oracle_best_is_shared: `{diagnostics.get('oracle_best_is_shared')}`",
        f"- target_ordering_met: `{checks.get('target_ordering_met')}`",
        "",
        "## Method Results",
        markdown_table(rows, fields),
        "## Ordering Checks",
        markdown_table([checks], list(checks.keys())),
        "## Top-10 Leaf Expected Probabilities",
        markdown_table(diagnostics.get("top10_leaf_expected_probabilities", []), top_fields),
        "## Top Safe Suffix Signatures",
        markdown_table(
            diagnostics.get("top_safe_suffix_signatures", []),
            ["signature", "leaf_count", "mean_probability"],
        ),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def load_reference_partial_4of5(reference_dir: Path) -> list[dict[str, Any]]:
    compare_path = reference_dir / "controlled_sim_compare.json"
    if not compare_path.exists():
        return []
    rows = json.loads(compare_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for method in REFERENCE_METHODS:
        match = next(
            (
                row for row in rows
                if row.get("setting") == "main_partial_4of5_L5_K5"
                and row.get("method") == method
            ),
            None,
        )
        if match is None:
            continue
        terminal_proxy = match.get("terminal_proxy_mean", match.get("average_cost_mean"))
        results.append(
            {
                "method": method,
                "regret_per_t_mean": match["regret_per_t_mean"],
                "terminal_proxy_mean": terminal_proxy,
                "shared_path_fraction_mean": match.get("shared_path_fraction_mean"),
                "num_paths": match.get("num_paths"),
                "horizon": match.get("horizon"),
                "seeds": match.get("seeds"),
                "source_setting": match.get("setting"),
            }
        )
    return sorted(results, key=lambda row: row["regret_per_t_mean"])


def load_old_unique_agents_reference(reference_dir: Path) -> list[dict[str, Any]]:
    compare_path = reference_dir / "unique_agents_compare.json"
    if compare_path.exists():
        payload = json.loads(compare_path.read_text(encoding="utf-8"))
        rows = payload.get("new_tree", [])
        return sorted(
            [
                {
                    "method": row["method"],
                    "regret_per_t_mean": row["regret_per_t_mean"],
                    "terminal_proxy_mean": row.get("average_cost_mean", row.get("terminal_proxy_mean")),
                    "shared_path_fraction_mean": row.get("shared_path_fraction_mean"),
                    "num_paths": row.get("num_paths"),
                    "horizon": row.get("horizon"),
                    "seeds": row.get("seeds"),
                    "source_setting": row.get("setting"),
                }
                for row in rows
            ],
            key=lambda row: row["regret_per_t_mean"],
        )
    compare_path = reference_dir / "controlled_sim_compare.json"
    if not compare_path.exists():
        return []
    rows = json.loads(compare_path.read_text(encoding="utf-8"))
    results: list[dict[str, Any]] = []
    for method in REFERENCE_METHODS:
        match = next((row for row in rows if row.get("method") == method), None)
        if match is None:
            continue
        results.append(
            {
                "method": method,
                "regret_per_t_mean": match["regret_per_t_mean"],
                "terminal_proxy_mean": match.get("terminal_proxy_mean", match.get("average_cost_mean")),
                "shared_path_fraction_mean": match.get("shared_path_fraction_mean"),
                "num_paths": match.get("num_paths"),
                "horizon": match.get("horizon"),
                "seeds": match.get("seeds"),
                "source_setting": match.get("setting"),
            }
        )
    return sorted(results, key=lambda row: row["regret_per_t_mean"])


def build_delta_rows(
    new_rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    reference_by_method = {row["method"]: row for row in reference_rows}
    rows: list[dict[str, Any]] = []
    for row in new_rows:
        reference = reference_by_method.get(row["method"])
        if reference is None:
            continue
        rows.append(
            {
                "method": row["method"],
                "current_regret_per_t_mean": row["regret_per_t_mean"],
                "reference_regret_per_t_mean": reference["regret_per_t_mean"],
                "delta_regret_per_t": row["regret_per_t_mean"] - reference["regret_per_t_mean"],
                "current_terminal_proxy_mean": row["average_cost_mean"],
                "reference_terminal_proxy_mean": reference["terminal_proxy_mean"],
                "delta_terminal_proxy": row["average_cost_mean"] - reference["terminal_proxy_mean"],
                "current_shared_path_fraction_mean": row["shared_path_fraction_mean"],
                "reference_shared_path_fraction_mean": reference["shared_path_fraction_mean"],
                "delta_shared_path_fraction": (
                    row["shared_path_fraction_mean"] - reference["shared_path_fraction_mean"]
                    if reference["shared_path_fraction_mean"] is not None
                    else None
                ),
            }
        )
    return rows


def build_unique_agents_compare(
    *,
    summary: list[dict[str, Any]],
    validation: dict[str, Any],
    old_reference_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    new_rows = sorted(summary, key=lambda row: row["regret_per_t_mean"])
    old_by_method = {row["method"]: row for row in old_reference_rows}
    delta_vs_old: list[dict[str, Any]] = []
    for row in new_rows:
        old = old_by_method.get(row["method"])
        if old is None:
            continue
        delta_vs_old.append(
            {
                "method": row["method"],
                "unique_regret_per_t_mean": row["regret_per_t_mean"],
                "old_regret_per_t_mean": old["regret_per_t_mean"],
                "delta_regret_per_t": row["regret_per_t_mean"] - old["regret_per_t_mean"],
                "unique_terminal_proxy_mean": row["average_cost_mean"],
                "old_terminal_proxy_mean": old["terminal_proxy_mean"],
                "delta_terminal_proxy": row["average_cost_mean"] - old["terminal_proxy_mean"],
                "unique_shared_path_fraction_mean": row["shared_path_fraction_mean"],
                "old_shared_path_fraction_mean": old["shared_path_fraction_mean"],
                "delta_shared_path_fraction": (
                    row["shared_path_fraction_mean"] - old["shared_path_fraction_mean"]
                    if old["shared_path_fraction_mean"] is not None
                    else None
                ),
            }
        )
    return {
        "tree_validation": validation,
        "new_tree": new_rows,
        "old_tree_reference_partial_4of5": old_reference_rows,
        "delta_vs_old_tree_partial_4of5": delta_vs_old,
    }


def write_unique_agents_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
) -> None:
    new_rows = compare["new_tree"]
    old_rows = compare["old_tree_reference_partial_4of5"]
    deltas = compare["delta_vs_old_tree_partial_4of5"]
    validation = compare["tree_validation"]

    new_by_method = {row["method"]: row for row in new_rows}
    old_by_method = {row["method"]: row for row in old_rows}
    risky_new = new_by_method.get("risky_ps")
    epsilon_new = new_by_method.get("epsilon_exp3")
    direct_new = new_by_method.get("direct_multistage_exp3")
    risky_old = old_by_method.get("risky_ps")
    epsilon_old = old_by_method.get("epsilon_exp3")
    direct_old = old_by_method.get("direct_multistage_exp3")

    answers: list[str] = []
    if risky_new and epsilon_new and risky_old and epsilon_old:
        new_gap_eps = risky_new["regret_per_t_mean"] - epsilon_new["regret_per_t_mean"]
        old_gap_eps = risky_old["regret_per_t_mean"] - epsilon_old["regret_per_t_mean"]
        relation = "smaller" if new_gap_eps < old_gap_eps else "not smaller"
        answers.append(
            f"- risky_ps vs epsilon_exp3 gap is {relation}: unique={new_gap_eps:.6f}, old={old_gap_eps:.6f}."
        )
    if risky_new and direct_new and risky_old and direct_old:
        new_gap_direct = risky_new["regret_per_t_mean"] - direct_new["regret_per_t_mean"]
        old_gap_direct = risky_old["regret_per_t_mean"] - direct_old["regret_per_t_mean"]
        relation = "smaller" if new_gap_direct < old_gap_direct else "not smaller"
        answers.append(
            f"- risky_ps vs direct_multistage_exp3 gap is {relation}: unique={new_gap_direct:.6f}, old={old_gap_direct:.6f}."
        )
    if (
        risky_new and epsilon_new and direct_new
        and risky_old and epsilon_old and direct_old
    ):
        new_avg_gap = (
            (risky_new["regret_per_t_mean"] - epsilon_new["regret_per_t_mean"])
            + (risky_new["regret_per_t_mean"] - direct_new["regret_per_t_mean"])
        ) / 2.0
        old_avg_gap = (
            (risky_old["regret_per_t_mean"] - epsilon_old["regret_per_t_mean"])
            + (risky_old["regret_per_t_mean"] - direct_old["regret_per_t_mean"])
        ) / 2.0
        answers.append(
            (
                "- Cross-prefix reuse removal "
                + ("does shrink" if new_avg_gap < old_avg_gap else "does not shrink")
                + f" the average risky_ps gap to epsilon/direct: unique={new_avg_gap:.6f}, old={old_avg_gap:.6f}."
            )
        )
        if new_avg_gap >= old_avg_gap:
            answers.append(
                "- On this ablation, the remaining weakness looks more like aggregation/update behavior than suffix-family reuse alone."
            )
    if new_by_method.get("naive_mixed") and old_by_method.get("naive_mixed"):
        naive_rank_new = next(idx for idx, row in enumerate(new_rows, start=1) if row["method"] == "naive_mixed")
        naive_rank_old = next(idx for idx, row in enumerate(old_rows, start=1) if row["method"] == "naive_mixed")
        random_rank_new = next(idx for idx, row in enumerate(new_rows, start=1) if row["method"] == "random_path")
        random_rank_old = next(idx for idx, row in enumerate(old_rows, start=1) if row["method"] == "random_path")
        answers.append(
            f"- naive_mixed rank changed from old={naive_rank_old} to unique={naive_rank_new}; random_path changed from old={random_rank_old} to unique={random_rank_new}."
        )
    answers.append(
        f"- The unique tree has num_paths={validation['num_paths']} versus old partial_4of5 num_paths={old_rows[0]['num_paths'] if old_rows else 'unknown'}."
    )
    answers.append(
        "- This is a real caveat: fewer paths make exploration easier, so any comparison mixes structure cleanup with a smaller path space. If risky_ps still does not close the gap under fewer paths, that weakens the cross-prefix-reuse hypothesis."
    )

    fields = [
        "method",
        "regret_per_t_mean",
        "average_cost_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    old_fields = [
        "method",
        "regret_per_t_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    delta_fields = [
        "method",
        "delta_regret_per_t",
        "delta_terminal_proxy",
        "delta_shared_path_fraction",
    ]
    content = [
        "# Unique-agent 4/5-share controlled simulation",
        "",
        "## Direct Answers",
        *answers,
        "",
        "## Unique-Agent Tree Results",
        markdown_table(new_rows, fields),
        "## Old Tree Reference: main_partial_4of5_L5_K5",
        markdown_table(old_rows, old_fields),
        "## Delta vs Old Tree",
        markdown_table(deltas, delta_fields),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def build_unique_agents_unbound_compare(
    *,
    summary: list[dict[str, Any]],
    validation: dict[str, Any],
    old_unique_rows: list[dict[str, Any]],
    old_theory_rows: list[dict[str, Any]],
    role_mode: str,
) -> dict[str, Any]:
    current_rows = sorted(summary, key=lambda row: row["regret_per_t_mean"])
    return {
        "tree_validation": validation,
        "tree_spec_role_mode": role_mode,
        "current_tree": current_rows,
        "old_unique_reference": old_unique_rows,
        "old_theory_aligned_reference_partial_4of5": old_theory_rows,
        "delta_vs_old_unique": build_delta_rows(current_rows, old_unique_rows),
        "delta_vs_old_theory_aligned": build_delta_rows(current_rows, old_theory_rows),
    }


def write_unique_agents_unbound_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
) -> None:
    current_rows = compare["current_tree"]
    old_unique_rows = compare["old_unique_reference"]
    old_theory_rows = compare["old_theory_aligned_reference_partial_4of5"]
    validation = compare["tree_validation"]
    role_mode = compare["tree_spec_role_mode"]

    current_by_method = {row["method"]: row for row in current_rows}
    old_unique_by_method = {row["method"]: row for row in old_unique_rows}
    old_theory_by_method = {row["method"]: row for row in old_theory_rows}
    ps_family_methods = [
        "risky_ps",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    ]
    ps_family_ranked = [
        row for row in current_rows if row["method"] in ps_family_methods
    ]

    answers: list[str] = []
    risky = current_by_method.get("risky_ps")
    epsilon = current_by_method.get("epsilon_exp3")
    direct = current_by_method.get("direct_multistage_exp3")
    old_unique_risky = old_unique_by_method.get("risky_ps")
    old_unique_epsilon = old_unique_by_method.get("epsilon_exp3")
    old_unique_direct = old_unique_by_method.get("direct_multistage_exp3")
    old_theory_risky = old_theory_by_method.get("risky_ps")
    old_theory_epsilon = old_theory_by_method.get("epsilon_exp3")
    old_theory_direct = old_theory_by_method.get("direct_multistage_exp3")

    if risky and epsilon:
        current_gap_eps = risky["regret_per_t_mean"] - epsilon["regret_per_t_mean"]
        line = f"- Current risky_ps vs epsilon_exp3 gap: {current_gap_eps:.6f}."
        if old_unique_risky and old_unique_epsilon:
            old_unique_gap_eps = (
                old_unique_risky["regret_per_t_mean"] - old_unique_epsilon["regret_per_t_mean"]
            )
            line += f" Old unique-agent gap: {old_unique_gap_eps:.6f}."
        if old_theory_risky and old_theory_epsilon:
            old_theory_gap_eps = (
                old_theory_risky["regret_per_t_mean"] - old_theory_epsilon["regret_per_t_mean"]
            )
            line += f" Old theory-aligned gap: {old_theory_gap_eps:.6f}."
        answers.append(line)
    if risky and direct:
        current_gap_direct = risky["regret_per_t_mean"] - direct["regret_per_t_mean"]
        line = f"- Current risky_ps vs direct_multistage_exp3 gap: {current_gap_direct:.6f}."
        if old_unique_risky and old_unique_direct:
            old_unique_gap_direct = (
                old_unique_risky["regret_per_t_mean"] - old_unique_direct["regret_per_t_mean"]
            )
            line += f" Old unique-agent gap: {old_unique_gap_direct:.6f}."
        if old_theory_risky and old_theory_direct:
            old_theory_gap_direct = (
                old_theory_risky["regret_per_t_mean"] - old_theory_direct["regret_per_t_mean"]
            )
            line += f" Old theory-aligned gap: {old_theory_gap_direct:.6f}."
        answers.append(line)
    if risky and epsilon and direct and old_theory_risky and old_theory_epsilon and old_theory_direct:
        current_avg_gap = (
            (risky["regret_per_t_mean"] - epsilon["regret_per_t_mean"])
            + (risky["regret_per_t_mean"] - direct["regret_per_t_mean"])
        ) / 2.0
        old_theory_avg_gap = (
            (old_theory_risky["regret_per_t_mean"] - old_theory_epsilon["regret_per_t_mean"])
            + (old_theory_risky["regret_per_t_mean"] - old_theory_direct["regret_per_t_mean"])
        ) / 2.0
        answers.append(
            "- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is "
            + ("smaller" if current_avg_gap < old_theory_avg_gap else "not smaller")
            + f": current={current_avg_gap:.6f}, old_theory={old_theory_avg_gap:.6f}."
        )
        if old_unique_risky and old_unique_epsilon and old_unique_direct:
            old_unique_avg_gap = (
                (old_unique_risky["regret_per_t_mean"] - old_unique_epsilon["regret_per_t_mean"])
                + (old_unique_risky["regret_per_t_mean"] - old_unique_direct["regret_per_t_mean"])
            ) / 2.0
            answers.append(
                "- Relative to the older bound unique-agent tree, the average risky_ps gap to epsilon/direct is "
                + ("smaller" if current_avg_gap < old_unique_avg_gap else "not smaller")
                + f": current={current_avg_gap:.6f}, old_unique={old_unique_avg_gap:.6f}."
            )
    if len(ps_family_ranked) >= 2:
        answers.append(
            "- PS-family ranking on this truly-unbound tree: "
            + ", ".join(
                f"{idx}. {row['method']} ({row['regret_per_t_mean']:.6f})"
                for idx, row in enumerate(ps_family_ranked, start=1)
            )
            + "."
        )
    naive = current_by_method.get("naive_mixed")
    if naive:
        naive_rank = next(idx for idx, row in enumerate(current_rows, start=1) if row["method"] == "naive_mixed")
        answers.append(
            f"- naive_mixed rank is {naive_rank} with regret/T={naive['regret_per_t_mean']:.6f}."
        )
    answers.extend(
        [
            "- In `--tree-spec` mode, the synthetic cost role is now resolved in this order: explicit `cost_role`/`synthetic_role`/`latent_role` from the spec, then `agent_id` when role mode is unbound, and only `base_alias` in explicit compatibility mode.",
            "- On this run the external-tree suffix family is keyed by the unique cost-role sequence plus gate pattern, so repeated `base_alias` strings no longer create cross-prefix latent-family reuse.",
            "- This is still a structure ablation, not an apples-to-apples replacement for the old theory-aligned tree. The new tree keeps full branching and `num_paths=3125`, so any comparison to the old compact DAG must be read with that caveat.",
        ]
    )
    if risky and epsilon and direct and risky["regret_per_t_mean"] > min(epsilon["regret_per_t_mean"], direct["regret_per_t_mean"]):
        answers.append(
            "- If the PS-family gap still does not shrink enough here, that points more toward aggregation/update behavior than toward tree reuse confounding alone."
        )

    current_fields = [
        "method",
        "regret_per_t_mean",
        "average_cost_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    reference_fields = [
        "method",
        "regret_per_t_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    delta_fields = [
        "method",
        "delta_regret_per_t",
        "delta_terminal_proxy",
        "delta_shared_path_fraction",
    ]
    content = [
        "# Unique-agent full-branching unbound controlled simulation",
        "",
        f"- tree_spec_role_mode: `{role_mode}`",
        f"- num_paths: `{validation.get('num_paths')}`",
        f"- duplicate_agent_count: `{validation.get('duplicate_agent_count')}`",
        f"- cross_prefix_duplicate_count: `{validation.get('cross_prefix_duplicate_count')}`",
        "",
        "## Current Tree",
        markdown_table(current_rows, current_fields),
        "## Delta vs Old Unique-Agent Reference",
        markdown_table(compare["delta_vs_old_unique"], delta_fields)
        if compare["delta_vs_old_unique"]
        else "_No old unique-agent reference rows found._",
        "## Delta vs Old Theory-Aligned Partial 4/5 Reference",
        markdown_table(compare["delta_vs_old_theory_aligned"], delta_fields)
        if compare["delta_vs_old_theory_aligned"]
        else "_No old theory-aligned reference rows found._",
        "## Direct Answers",
        *answers,
        "",
        "## Old Unique-Agent Reference",
        markdown_table(old_unique_rows, reference_fields)
        if old_unique_rows
        else "_No old unique-agent reference rows found._",
        "## Old Theory-Aligned Reference",
        markdown_table(old_theory_rows, reference_fields)
        if old_theory_rows
        else "_No old theory-aligned reference rows found._",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def build_same_topology_unique_agents_compare(
    *,
    summary: list[dict[str, Any]],
    validation: dict[str, Any],
    old_theory_rows: list[dict[str, Any]],
    old_unbound_rows: list[dict[str, Any]],
    role_mode: str,
) -> dict[str, Any]:
    current_rows = sorted(summary, key=lambda row: row["regret_per_t_mean"])
    return {
        "tree_validation": validation,
        "tree_spec_role_mode": role_mode,
        "current_tree": current_rows,
        "old_theory_aligned_reference_partial_4of5": old_theory_rows,
        "old_unbound_fullbranch_reference": old_unbound_rows,
        "delta_vs_old_theory_aligned": build_delta_rows(current_rows, old_theory_rows),
        "delta_vs_old_unbound_fullbranch": build_delta_rows(current_rows, old_unbound_rows),
    }


def build_prefix_dedup_compare(
    *,
    summary: list[dict[str, Any]],
    validation: dict[str, Any],
    old_theory_rows: list[dict[str, Any]],
    role_mode: str,
) -> dict[str, Any]:
    current_rows = sorted(summary, key=lambda row: row["regret_per_t_mean"])
    return {
        "tree_validation": validation,
        "tree_spec_role_mode": role_mode,
        "current_tree": current_rows,
        "old_theory_aligned_reference_partial_4of5": old_theory_rows,
        "delta_vs_old_theory_aligned": build_delta_rows(current_rows, old_theory_rows),
    }


def write_prefix_dedup_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
) -> None:
    current_rows = compare["current_tree"]
    old_theory_rows = compare["old_theory_aligned_reference_partial_4of5"]
    validation = compare["tree_validation"]
    role_mode = compare["tree_spec_role_mode"]

    current_by_method = {row["method"]: row for row in current_rows}
    old_theory_by_method = {row["method"]: row for row in old_theory_rows}
    ps_family_methods = [
        "risky_ps_old",
        "risky_ps",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    ]
    ps_family_ranked = [row for row in current_rows if row["method"] in ps_family_methods]

    answers: list[str] = []
    risky_old = current_by_method.get("risky_ps_old")
    risky = current_by_method.get("risky_ps")
    epsilon = current_by_method.get("epsilon_exp3")
    direct = current_by_method.get("direct_multistage_exp3")
    if risky_old and risky:
        answers.append(
            f"- risky_ps_old vs risky_ps gap: {risky_old['regret_per_t_mean'] - risky['regret_per_t_mean']:.6f} "
            f"(old={risky_old['regret_per_t_mean']:.6f}, new={risky['regret_per_t_mean']:.6f})."
        )
    if risky and epsilon:
        answers.append(
            f"- risky_ps vs epsilon_exp3 gap: {risky['regret_per_t_mean'] - epsilon['regret_per_t_mean']:.6f}."
        )
    if risky and direct:
        answers.append(
            f"- risky_ps vs direct_multistage_exp3 gap: {risky['regret_per_t_mean'] - direct['regret_per_t_mean']:.6f}."
        )
    if ps_family_ranked:
        answers.append(
            "- PS-family ranking on the prefix-dedup tree: "
            + ", ".join(
                f"{idx}. {row['method']} ({row['regret_per_t_mean']:.6f})"
                for idx, row in enumerate(ps_family_ranked, start=1)
            )
            + "."
        )
    if risky and epsilon and abs(risky["regret_per_t_mean"] - epsilon["regret_per_t_mean"]) < 1e-12:
        answers.append("- risky_ps is exactly tied with epsilon_exp3 on this prefix-dedup run.")
    if risky:
        old_theory_risky = old_theory_by_method.get("risky_ps")
        old_theory_epsilon = old_theory_by_method.get("epsilon_exp3")
        old_theory_direct = old_theory_by_method.get("direct_multistage_exp3")
        if old_theory_risky and old_theory_epsilon and old_theory_direct and epsilon and direct:
            current_avg_gap = (
                (risky["regret_per_t_mean"] - epsilon["regret_per_t_mean"])
                + (risky["regret_per_t_mean"] - direct["regret_per_t_mean"])
            ) / 2.0
            old_avg_gap = (
                (old_theory_risky["regret_per_t_mean"] - old_theory_epsilon["regret_per_t_mean"])
                + (old_theory_risky["regret_per_t_mean"] - old_theory_direct["regret_per_t_mean"])
            ) / 2.0
            answers.append(
                "- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is "
                + ("smaller" if current_avg_gap < old_avg_gap else "not smaller")
                + f": current={current_avg_gap:.6f}, old_theory={old_avg_gap:.6f}."
            )
    answers.extend(
        [
            "- This tree preserves the original shared_basin_strong 4/5 minimal DAG connectivity and all original g values.",
            "- The only structural change is parent-specific cloning of reused child aliases, so cross-prefix repeated agent identity is removed while local continuation patterns are unchanged.",
            f"- tree_spec_role_mode for this run: `{role_mode}`.",
        ]
    )
    if len(ps_family_ranked) >= 2 and max(row["regret_per_t_mean"] for row in ps_family_ranked) - min(row["regret_per_t_mean"] for row in ps_family_ranked) < 1e-12:
        answers.append(
            "- PS-family methods still do not separate here, which points more toward aggregation/update dynamics than toward repeated-agent reuse alone."
        )

    current_fields = [
        "method",
        "regret_per_t_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    reference_fields = [
        "method",
        "regret_per_t_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    delta_fields = [
        "method",
        "delta_regret_per_t",
        "delta_terminal_proxy",
        "delta_shared_path_fraction",
    ]
    content = [
        "# Prefix-dedup controlled simulation",
        "",
        f"- tree_spec_role_mode: `{role_mode}`",
        f"- depth: `{validation.get('depth')}`",
        f"- num_paths: `{validation.get('num_paths')}`",
        f"- total_agent_ids: `{validation.get('total_agent_ids')}`",
        f"- duplicate_agent_count: `{validation.get('duplicate_agent_count')}`",
        f"- cross_prefix_duplicate_count: `{validation.get('cross_prefix_duplicate_count')}`",
        "",
        "## Current Tree",
        markdown_table(current_rows, current_fields),
        "## Delta vs Old Theory-Aligned Partial 4/5 Reference",
        markdown_table(compare["delta_vs_old_theory_aligned"], delta_fields)
        if compare["delta_vs_old_theory_aligned"]
        else "_No old theory-aligned reference rows found._",
        "## Direct Answers",
        *answers,
        "",
        "## Old Theory-Aligned Reference",
        markdown_table(old_theory_rows, reference_fields)
        if old_theory_rows
        else "_No old theory-aligned reference rows found._",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_same_topology_unique_agents_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
) -> None:
    current_rows = compare["current_tree"]
    old_theory_rows = compare["old_theory_aligned_reference_partial_4of5"]
    old_unbound_rows = compare["old_unbound_fullbranch_reference"]
    validation = compare["tree_validation"]
    role_mode = compare["tree_spec_role_mode"]

    current_by_method = {row["method"]: row for row in current_rows}
    old_theory_by_method = {row["method"]: row for row in old_theory_rows}
    old_unbound_by_method = {row["method"]: row for row in old_unbound_rows}
    ps_family_methods = [
        "risky_ps",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    ]
    ps_family_ranked = [row for row in current_rows if row["method"] in ps_family_methods]

    risky = current_by_method.get("risky_ps")
    epsilon = current_by_method.get("epsilon_exp3")
    direct = current_by_method.get("direct_multistage_exp3")
    old_theory_risky = old_theory_by_method.get("risky_ps")
    old_theory_epsilon = old_theory_by_method.get("epsilon_exp3")
    old_theory_direct = old_theory_by_method.get("direct_multistage_exp3")
    old_unbound_risky = old_unbound_by_method.get("risky_ps")
    old_unbound_epsilon = old_unbound_by_method.get("epsilon_exp3")
    old_unbound_direct = old_unbound_by_method.get("direct_multistage_exp3")

    answers: list[str] = []
    if risky and epsilon:
        current_gap_eps = risky["regret_per_t_mean"] - epsilon["regret_per_t_mean"]
        line = f"- Current risky_ps vs epsilon_exp3 gap: {current_gap_eps:.6f}."
        if old_theory_risky and old_theory_epsilon:
            line += (
                " Old theory-aligned gap: "
                f"{old_theory_risky['regret_per_t_mean'] - old_theory_epsilon['regret_per_t_mean']:.6f}."
            )
        if old_unbound_risky and old_unbound_epsilon:
            line += (
                " Old unbound full-branch gap: "
                f"{old_unbound_risky['regret_per_t_mean'] - old_unbound_epsilon['regret_per_t_mean']:.6f}."
            )
        answers.append(line)
    if risky and direct:
        current_gap_direct = risky["regret_per_t_mean"] - direct["regret_per_t_mean"]
        line = f"- Current risky_ps vs direct_multistage_exp3 gap: {current_gap_direct:.6f}."
        if old_theory_risky and old_theory_direct:
            line += (
                " Old theory-aligned gap: "
                f"{old_theory_risky['regret_per_t_mean'] - old_theory_direct['regret_per_t_mean']:.6f}."
            )
        if old_unbound_risky and old_unbound_direct:
            line += (
                " Old unbound full-branch gap: "
                f"{old_unbound_risky['regret_per_t_mean'] - old_unbound_direct['regret_per_t_mean']:.6f}."
            )
        answers.append(line)
    if risky and epsilon and direct:
        current_avg_gap = (
            (risky["regret_per_t_mean"] - epsilon["regret_per_t_mean"])
            + (risky["regret_per_t_mean"] - direct["regret_per_t_mean"])
        ) / 2.0
        if old_theory_risky and old_theory_epsilon and old_theory_direct:
            old_theory_avg_gap = (
                (old_theory_risky["regret_per_t_mean"] - old_theory_epsilon["regret_per_t_mean"])
                + (old_theory_risky["regret_per_t_mean"] - old_theory_direct["regret_per_t_mean"])
            ) / 2.0
            answers.append(
                "- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is "
                + ("smaller" if current_avg_gap < old_theory_avg_gap else "not smaller")
                + f": current={current_avg_gap:.6f}, old_theory={old_theory_avg_gap:.6f}."
            )
        if old_unbound_risky and old_unbound_epsilon and old_unbound_direct:
            old_unbound_avg_gap = (
                (old_unbound_risky["regret_per_t_mean"] - old_unbound_epsilon["regret_per_t_mean"])
                + (old_unbound_risky["regret_per_t_mean"] - old_unbound_direct["regret_per_t_mean"])
            ) / 2.0
            answers.append(
                "- Relative to the older fully-unbound full-branch tree, the average risky_ps gap to epsilon/direct is "
                + ("smaller" if current_avg_gap < old_unbound_avg_gap else "not smaller")
                + f": current={current_avg_gap:.6f}, old_unbound={old_unbound_avg_gap:.6f}."
            )
    if ps_family_ranked:
        answers.append(
            "- PS-family ranking on the subtree-local tree: "
            + ", ".join(
                f"{idx}. {row['method']} ({row['regret_per_t_mean']:.6f})"
                for idx, row in enumerate(ps_family_ranked, start=1)
            )
            + "."
        )
    naive = current_by_method.get("naive_mixed")
    if naive:
        naive_rank = next(idx for idx, row in enumerate(current_rows, start=1) if row["method"] == "naive_mixed")
        answers.append(f"- naive_mixed rank is {naive_rank} with regret/T={naive['regret_per_t_mean']:.6f}.")
    answers.extend(
        [
            "- `subtree_local` cost semantics uses prefix-local node/edge identity, not `base_alias`, to generate additive ancestor-chain bias along the sampled path.",
            "- Leaves under the same safe subtree share ancestor bias terms because they traverse the same concrete ancestor nodes.",
            "- Different prefixes do not share those bias terms because every parent-child occurrence has a unique agent_id/cost_role, so horizontal cross-prefix reuse is removed.",
            "- This remains a structure ablation. It preserves the official 4/5 full-branch topology and path count, but changes the external synthetic cost semantics to subtree-local correlated cost.",
        ]
    )
    if risky and epsilon and direct and risky["regret_per_t_mean"] >= epsilon["regret_per_t_mean"]:
        answers.append(
            "- If PS-family methods still do not separate here, that is evidence that the remaining limitation is in aggregation/update dynamics rather than in repeated-agent or repeated-cost-family confounding."
        )

    current_fields = [
        "method",
        "regret_per_t_mean",
        "average_cost_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    reference_fields = [
        "method",
        "regret_per_t_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "num_paths",
        "horizon",
        "seeds",
    ]
    delta_fields = [
        "method",
        "delta_regret_per_t",
        "delta_terminal_proxy",
        "delta_shared_path_fraction",
    ]
    content = [
        "# Same-topology unique-agent subtree-local controlled simulation",
        "",
        f"- tree_spec_role_mode: `{role_mode}`",
        f"- num_paths: `{validation.get('num_paths')}`",
        f"- duplicate_agent_count: `{validation.get('duplicate_agent_count')}`",
        f"- cross_prefix_duplicate_count: `{validation.get('cross_prefix_duplicate_count')}`",
        "",
        "## Current Tree",
        markdown_table(current_rows, current_fields),
        "## Delta vs Old Theory-Aligned Partial 4/5 Reference",
        markdown_table(compare["delta_vs_old_theory_aligned"], delta_fields)
        if compare["delta_vs_old_theory_aligned"]
        else "_No old theory-aligned reference rows found._",
        "## Delta vs Old Unbound Full-Branch Reference",
        markdown_table(compare["delta_vs_old_unbound_fullbranch"], delta_fields)
        if compare["delta_vs_old_unbound_fullbranch"]
        else "_No old unbound full-branch reference rows found._",
        "## Direct Answers",
        *answers,
        "",
        "## Old Theory-Aligned Reference",
        markdown_table(old_theory_rows, reference_fields)
        if old_theory_rows
        else "_No old theory-aligned reference rows found._",
        "## Old Unbound Full-Branch Reference",
        markdown_table(old_unbound_rows, reference_fields)
        if old_unbound_rows
        else "_No old unbound full-branch reference rows found._",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def build_findings(
    *,
    summary: list[dict[str, Any]],
    permutation_example: dict[str, Any],
) -> dict[str, Any]:
    by_setting: dict[str, list[dict[str, Any]]] = {}
    for row in summary:
        by_setting.setdefault(row["setting"], []).append(row)

    def method_row(setting_prefix: str, method: str) -> dict[str, Any] | None:
        for setting, rows in by_setting.items():
            if setting.startswith(setting_prefix):
                for row in rows:
                    if row["method"] == method:
                        return row
        return None

    findings: dict[str, Any] = {
        "permutation_example_seed0": permutation_example,
    }

    main_num_paths: dict[str, Any] = {}
    main_topology_rows: list[dict[str, Any]] = []
    for prefix in [
        "main_all_share",
        "main_partial_4of5",
        "main_partial_2of5",
        "main_all_unshare",
    ]:
        rows = [
            row
            for setting, setting_rows in by_setting.items()
            if setting.startswith(prefix)
            for row in setting_rows
        ]
        ordered = sorted(rows, key=lambda row: row["regret_per_t_mean"])
        findings[prefix] = {
            "best_method": ordered[0]["method"] if ordered else None,
            "ranking": [
                {
                    "method": row["method"],
                    "regret_per_t_mean": row["regret_per_t_mean"],
                    "average_cost_mean": row["average_cost_mean"],
                    "first_episode_best_hit_rate_mean": row["first_episode_best_hit_rate_mean"],
                }
                for row in ordered
            ],
        }
        if ordered:
            main_num_paths[prefix] = {
                "num_paths": ordered[0]["num_paths"],
                "expected_full_num_paths": ordered[0]["expected_full_num_paths"],
                "full_branching": ordered[0]["full_branching"],
            }
            main_topology_rows.append(ordered[0])

    findings["main_variant_invariants"] = {
        "num_paths": main_num_paths,
        "same_depth": len({row["depth"] for row in main_topology_rows}) == 1,
        "same_branching": len({row["branching"] for row in main_topology_rows}) == 1,
        "same_topology": len({row["topology"] for row in main_topology_rows}) == 1,
        "only_share_structure_changes": len({row["sharing_scheme"] for row in main_topology_rows}) == 1,
    }

    template_visible_children: dict[str, str | None] = {}
    for stage_name, target_role in (
        ("stage3", SHARED_CORE_A),
        ("stage4", SHARED_CORE_B),
        ("stage5", SHARED_CORE_A),
    ):
        stage_rows = permutation_example.get("stages", {}).get(stage_name, [])
        match = next(
            (row.get("visible_child") for row in stage_rows if row.get("latent_role") == target_role),
            None,
        )
        template_visible_children[stage_name] = match

    naive_all_share = method_row("main_all_share", "naive_mixed")
    risky_all_share = method_row("main_all_share", "risky_ps")
    findings["naive_tiebreak_leak_check"] = {
        "naive_first_episode_best_hit_rate_mean": (
            naive_all_share["first_episode_best_hit_rate_mean"] if naive_all_share else None
        ),
        "risky_ps_first_episode_best_hit_rate_mean": (
            risky_all_share["first_episode_best_hit_rate_mean"] if risky_all_share else None
        ),
        "shared_template_visible_children": template_visible_children,
        "shared_template_all_zero": all(
            (visible_child or "").endswith("_00")
            for visible_child in template_visible_children.values()
        ),
    }

    suffix_rows = [
        row for row in summary
        if row["setting_group"] == "safe_suffix" and row["method"] == "risky_ps"
    ]
    safe_suffix_payload = sorted(
        [
            {
                "risky_depth": row["risky_depth"],
                "safe_suffix_length": row["safe_suffix_length"],
                "regret_per_t_mean": row["regret_per_t_mean"],
                "average_cost_mean": row["average_cost_mean"],
            }
            for row in suffix_rows
        ],
        key=lambda row: row["risky_depth"],
    )
    findings["safe_suffix_risky_ps"] = safe_suffix_payload
    findings["safe_suffix_trend_non_decreasing_in_risky_depth"] = all(
        safe_suffix_payload[idx]["regret_per_t_mean"] <= safe_suffix_payload[idx + 1]["regret_per_t_mean"] + 1e-12
        for idx in range(len(safe_suffix_payload) - 1)
    )
    relevant_window = [row for row in safe_suffix_payload if row["risky_depth"] <= 2]
    findings["safe_suffix_reusable_window_trend"] = all(
        relevant_window[idx]["regret_per_t_mean"] >= relevant_window[idx + 1]["regret_per_t_mean"] - 1e-12
        for idx in range(len(relevant_window) - 1)
    )

    scaling_payload: list[dict[str, Any]] = []
    for setting, rows in by_setting.items():
        if not setting.startswith("candidate_scaling_"):
            continue
        risky = next((row for row in rows if row["method"] == "risky_ps"), None)
        direct = next((row for row in rows if row["method"] == "direct_multistage_exp3"), None)
        epsilon = next((row for row in rows if row["method"] == "epsilon_exp3"), None)
        if risky and direct and epsilon:
            scaling_payload.append(
                {
                    "setting": setting,
                    "branching": risky["branching"],
                    "risky_ps_regret_per_t": risky["regret_per_t_mean"],
                    "direct_exp3_regret_per_t": direct["regret_per_t_mean"],
                    "epsilon_exp3_regret_per_t": epsilon["regret_per_t_mean"],
                    "gap_vs_direct": direct["regret_per_t_mean"] - risky["regret_per_t_mean"],
                    "gap_vs_epsilon": epsilon["regret_per_t_mean"] - risky["regret_per_t_mean"],
                }
            )
    findings["candidate_scaling"] = sorted(scaling_payload, key=lambda row: row["branching"])

    all_unshare = {
        method: method_row("main_all_unshare", method)
        for method in ("risky_ps", "direct_multistage_exp3", "epsilon_exp3", "random_path", "naive_mixed")
    }
    if all(all_unshare.values()):
        baseline_rows = [all_unshare["direct_multistage_exp3"], all_unshare["epsilon_exp3"], all_unshare["random_path"]]
        baseline_regrets = [row["regret_per_t_mean"] for row in baseline_rows if row is not None]
        risky_regret = all_unshare["risky_ps"]["regret_per_t_mean"]
        findings["all_unshare_bandit_like_check"] = {
            "risky_ps_regret_per_t": risky_regret,
            "baseline_regret_per_t_mean": mean(baseline_regrets),
            "max_abs_gap_vs_direct_epsilon_random": max(
                abs(risky_regret - row["regret_per_t_mean"])
                for row in baseline_rows
                if row is not None
            ),
        }
    else:
        findings["all_unshare_bandit_like_check"] = None

    return findings


def build_ix_run_specs(
    *,
    eta_shared_values: Sequence[float],
    gamma_shared_values: Sequence[float],
) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "method": method,
            "method_label": method,
            "policy_kwargs": {},
        }
        for method in REFERENCE_METHODS
    ]
    for eta_shared in eta_shared_values:
        for gamma_shared in gamma_shared_values:
            specs.append(
                {
                    "method": "risky_ps_ix",
                    "method_label": (
                        "risky_ps_ix"
                        f"_eta{format_float_for_label(eta_shared)}"
                        f"_gamma{format_float_for_label(gamma_shared)}"
                    ),
                    "policy_kwargs": {
                        "eta_shared": eta_shared,
                        "gamma_shared": gamma_shared,
                    },
                }
            )
    return specs


def build_denominator_ablation_run_specs(gamma_shared: float) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for method in DENOMINATOR_ABLATION_METHODS:
        policy_kwargs: dict[str, Any] = {}
        if method == "risky_ps_safe_conditional_ix":
            policy_kwargs["gamma_shared"] = gamma_shared
        specs.append(
            {
                "method": method,
                "method_label": method,
                "policy_kwargs": policy_kwargs,
            }
        )
    return specs


def build_direct_cost_ablation_run_specs() -> list[dict[str, Any]]:
    return [
        {
            "method": method,
            "method_label": method,
            "policy_kwargs": {},
        }
        for method in DIRECT_COST_ABLATION_METHODS
    ]


def build_ix_grid_compare(summary: list[dict[str, Any]]) -> dict[str, Any]:
    ix_rows = [row for row in summary if row["method"] == "risky_ps_ix"]
    eval_rows = [
        row for row in summary
        if row["setting_group"] in {"main_variants", "safe_suffix"}
    ]
    eval_ix_rows = [row for row in ix_rows if row["setting_group"] in {"main_variants", "safe_suffix"}]

    def setting_key(row: dict[str, Any]) -> str:
        return str(row["setting"])

    by_setting_method: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summary:
        by_setting_method[(setting_key(row), str(row["method"]))] = row

    best_ix_by_setting: list[dict[str, Any]] = []
    for setting in sorted({setting_key(row) for row in eval_ix_rows}):
        candidates = [row for row in eval_ix_rows if setting_key(row) == setting]
        if not candidates:
            continue
        best_ix = min(candidates, key=lambda row: row["regret_per_t_mean"])
        payload = {
            "setting": setting,
            "setting_group": best_ix["setting_group"],
            "variant": best_ix["variant"],
            "risky_depth": best_ix["risky_depth"],
            "safe_suffix_length": best_ix["safe_suffix_length"],
            "best_ix_method_label": best_ix["method_label"],
            "eta_shared": best_ix["eta_shared"],
            "gamma_shared": best_ix["gamma_shared"],
            "regret_per_t_mean": best_ix["regret_per_t_mean"],
            "regret_per_t_std": best_ix["regret_per_t_std"],
            "regret_mean": best_ix["regret_mean"],
            "terminal_proxy_mean": best_ix["terminal_proxy_mean"],
            "seeds": best_ix["seeds"],
            "horizon": best_ix["horizon"],
            "depth": best_ix["depth"],
            "branching": best_ix["branching"],
            "num_paths": best_ix["num_paths"],
        }
        for method in REFERENCE_METHODS:
            reference = by_setting_method.get((setting, method))
            if reference is None:
                continue
            safe_name = method.replace("direct_multistage_exp3", "direct_exp3")
            payload[f"delta_regret_per_t_vs_{safe_name}"] = (
                best_ix["regret_per_t_mean"] - reference["regret_per_t_mean"]
            )
            payload[f"delta_terminal_proxy_vs_{safe_name}"] = (
                best_ix["terminal_proxy_mean"] - reference["terminal_proxy_mean"]
            )
            payload[f"beats_{safe_name}"] = (
                best_ix["regret_per_t_mean"] < reference["regret_per_t_mean"]
            )
        best_ix_by_setting.append(payload)

    def aggregate_rows(rows: list[dict[str, Any]], *, scope: str) -> list[dict[str, Any]]:
        grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault((float(row["eta_shared"]), float(row["gamma_shared"])), []).append(row)
        aggregates: list[dict[str, Any]] = []
        for (eta_shared, gamma_shared), group in sorted(grouped.items()):
            aggregates.append(
                {
                    "scope": scope,
                    "method": "risky_ps_ix",
                    "eta_shared": eta_shared,
                    "gamma_shared": gamma_shared,
                    "settings": len(group),
                    "seeds": group[0]["seeds"],
                    "horizon": group[0]["horizon"],
                    "regret_per_t_mean": mean(row["regret_per_t_mean"] for row in group),
                    "regret_per_t_std_across_settings": stdev(
                        row["regret_per_t_mean"] for row in group
                    ),
                    "regret_mean": mean(row["regret_mean"] for row in group),
                    "terminal_proxy_mean": mean(row["terminal_proxy_mean"] for row in group),
                    "shared_path_fraction_mean": mean(
                        row["shared_path_fraction_mean"] for row in group
                    ),
                    "shared_update_count_mean": mean(
                        row["shared_update_count_mean"] for row in group
                    ),
                }
            )
        return sorted(aggregates, key=lambda row: row["regret_per_t_mean"])

    parameter_grid_overall = aggregate_rows(eval_ix_rows, scope="main_plus_safe_suffix")
    parameter_grid_main = aggregate_rows(
        [row for row in eval_ix_rows if row["setting_group"] == "main_variants"],
        scope="main_variants",
    )
    parameter_grid_safe_suffix = aggregate_rows(
        [row for row in eval_ix_rows if row["setting_group"] == "safe_suffix"],
        scope="safe_suffix",
    )

    baseline_overall: list[dict[str, Any]] = []
    for method in REFERENCE_METHODS:
        group = [row for row in eval_rows if row["method"] == method]
        if not group:
            continue
        baseline_overall.append(
            {
                "scope": "main_plus_safe_suffix",
                "method": method,
                "settings": len(group),
                "seeds": group[0]["seeds"],
                "horizon": group[0]["horizon"],
                "regret_per_t_mean": mean(row["regret_per_t_mean"] for row in group),
                "regret_per_t_std_across_settings": stdev(
                    row["regret_per_t_mean"] for row in group
                ),
                "regret_mean": mean(row["regret_mean"] for row in group),
                "terminal_proxy_mean": mean(row["terminal_proxy_mean"] for row in group),
                "shared_path_fraction_mean": mean(row["shared_path_fraction_mean"] for row in group),
                "shared_update_count_mean": mean(row["shared_update_count_mean"] for row in group),
            }
        )
    baseline_overall.sort(key=lambda row: row["regret_per_t_mean"])

    baseline_delta_overall: list[dict[str, Any]] = []
    if parameter_grid_overall:
        best = parameter_grid_overall[0]
        for baseline in baseline_overall:
            baseline_delta_overall.append(
                {
                    "compare_to": baseline["method"],
                    "best_ix_eta_shared": best["eta_shared"],
                    "best_ix_gamma_shared": best["gamma_shared"],
                    "delta_regret_per_t": (
                        best["regret_per_t_mean"] - baseline["regret_per_t_mean"]
                    ),
                    "delta_terminal_proxy": (
                        best["terminal_proxy_mean"] - baseline["terminal_proxy_mean"]
                    ),
                }
            )

    return {
        "parameter_grid_overall": parameter_grid_overall,
        "parameter_grid_main_variants": parameter_grid_main,
        "parameter_grid_safe_suffix": parameter_grid_safe_suffix,
        "baseline_overall": baseline_overall,
        "baseline_delta_overall": baseline_delta_overall,
        "best_ix_by_setting": best_ix_by_setting,
        "raw_summary": summary,
    }


def build_denominator_ablation_compare(summary: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [
        row for row in summary
        if row["setting_group"] in {"main_variants", "safe_suffix"}
    ]

    def aggregate(methods: Sequence[str], *, scope: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        aggregates: list[dict[str, Any]] = []
        for method in methods:
            group = [row for row in rows if row["method"] == method]
            if not group:
                continue
            aggregates.append(
                {
                    "scope": scope,
                    "method": method,
                    "settings": len(group),
                    "seeds": group[0]["seeds"],
                    "horizon": group[0]["horizon"],
                    "depth": group[0]["depth"],
                    "branching": group[0]["branching"],
                    "num_paths": group[0]["num_paths"],
                    "regret_per_t_mean": mean(row["regret_per_t_mean"] for row in group),
                    "regret_per_t_std_across_settings": stdev(
                        row["regret_per_t_mean"] for row in group
                    ),
                    "regret_mean": mean(row["regret_mean"] for row in group),
                    "terminal_proxy_mean": mean(row["terminal_proxy_mean"] for row in group),
                    "shared_path_fraction_mean": mean(
                        row["shared_path_fraction_mean"] for row in group
                    ),
                    "shared_update_count_mean": mean(
                        row["shared_update_count_mean"] for row in group
                    ),
                    "risky_update_count_mean": mean(
                        row["risky_update_count_mean"] for row in group
                    ),
                }
            )
        return sorted(aggregates, key=lambda row: row["regret_per_t_mean"])

    main_rows = [row for row in eval_rows if row["setting_group"] == "main_variants"]
    safe_rows = [row for row in eval_rows if row["setting_group"] == "safe_suffix"]
    denominator_overall = aggregate(DENOMINATOR_METHODS, scope="main_plus_safe_suffix", rows=eval_rows)
    denominator_main = aggregate(DENOMINATOR_METHODS, scope="main_variants", rows=main_rows)
    denominator_safe = aggregate(DENOMINATOR_METHODS, scope="safe_suffix", rows=safe_rows)
    baseline_overall = aggregate(REFERENCE_METHODS, scope="main_plus_safe_suffix", rows=eval_rows)

    by_setting_method = {
        (row["setting"], row["method"]): row
        for row in eval_rows
    }
    per_setting: list[dict[str, Any]] = []
    for setting in sorted({row["setting"] for row in eval_rows}):
        risky = by_setting_method.get((setting, "risky_ps"))
        safe = by_setting_method.get((setting, "risky_ps_safe_conditional"))
        safe_ix = by_setting_method.get((setting, "risky_ps_safe_conditional_ix"))
        global_ix = by_setting_method.get((setting, "risky_ps_ix"))
        epsilon = by_setting_method.get((setting, "epsilon_exp3"))
        direct = by_setting_method.get((setting, "direct_multistage_exp3"))
        if risky is None:
            continue
        payload = {
            "setting": setting,
            "setting_group": risky["setting_group"],
            "variant": risky["variant"],
            "risky_depth": risky["risky_depth"],
            "safe_suffix_length": risky["safe_suffix_length"],
            "global_leaf_regret_per_t": risky["regret_per_t_mean"],
            "global_leaf_ix_regret_per_t": (
                global_ix["regret_per_t_mean"] if global_ix else None
            ),
            "safe_conditional_regret_per_t": (
                safe["regret_per_t_mean"] if safe else None
            ),
            "safe_conditional_ix_regret_per_t": (
                safe_ix["regret_per_t_mean"] if safe_ix else None
            ),
            "epsilon_exp3_regret_per_t": (
                epsilon["regret_per_t_mean"] if epsilon else None
            ),
            "direct_exp3_regret_per_t": (
                direct["regret_per_t_mean"] if direct else None
            ),
        }
        if safe is not None:
            payload["delta_safe_conditional_vs_global_leaf"] = (
                safe["regret_per_t_mean"] - risky["regret_per_t_mean"]
            )
        if safe_ix is not None and safe is not None:
            payload["delta_safe_conditional_ix_vs_safe_conditional"] = (
                safe_ix["regret_per_t_mean"] - safe["regret_per_t_mean"]
            )
        if safe_ix is not None:
            if epsilon is not None:
                payload["delta_safe_conditional_ix_vs_epsilon_exp3"] = (
                    safe_ix["regret_per_t_mean"] - epsilon["regret_per_t_mean"]
                )
            if direct is not None:
                payload["delta_safe_conditional_ix_vs_direct_exp3"] = (
                    safe_ix["regret_per_t_mean"] - direct["regret_per_t_mean"]
                )
        per_setting.append(payload)

    gap_rows: list[dict[str, Any]] = []
    method_rows = {row["method"]: row for row in denominator_overall + baseline_overall}
    for method in DENOMINATOR_METHODS:
        row = method_rows.get(method)
        if row is None:
            continue
        for baseline in ("epsilon_exp3", "direct_multistage_exp3"):
            base = method_rows.get(baseline)
            if base is None:
                continue
            gap_rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "delta_regret_per_t": row["regret_per_t_mean"] - base["regret_per_t_mean"],
                    "delta_terminal_proxy": row["terminal_proxy_mean"] - base["terminal_proxy_mean"],
                }
            )

    return {
        "denominator_overall": denominator_overall,
        "denominator_main_variants": denominator_main,
        "denominator_safe_suffix": denominator_safe,
        "baseline_overall": baseline_overall,
        "per_setting": per_setting,
        "gap_vs_baselines": gap_rows,
        "raw_summary": summary,
    }


def build_direct_cost_ablation_compare(summary: list[dict[str, Any]]) -> dict[str, Any]:
    eval_rows = [
        row for row in summary
        if row["setting_group"] in {"main_variants", "safe_suffix"}
    ]

    def aggregate(methods: Sequence[str], *, scope: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        aggregates: list[dict[str, Any]] = []
        for method in methods:
            group = [row for row in rows if row["method"] == method]
            if not group:
                continue
            aggregates.append(
                {
                    "scope": scope,
                    "method": method,
                    "settings": len(group),
                    "seeds": group[0]["seeds"],
                    "horizon": group[0]["horizon"],
                    "depth": group[0]["depth"],
                    "branching": group[0]["branching"],
                    "num_paths": group[0]["num_paths"],
                    "regret_per_t_mean": mean(row["regret_per_t_mean"] for row in group),
                    "regret_per_t_std_across_settings": stdev(
                        row["regret_per_t_mean"] for row in group
                    ),
                    "regret_mean": mean(row["regret_mean"] for row in group),
                    "terminal_proxy_mean": mean(row["terminal_proxy_mean"] for row in group),
                    "shared_path_fraction_mean": mean(
                        row["shared_path_fraction_mean"] for row in group
                    ),
                    "shared_update_count_mean": mean(
                        row["shared_update_count_mean"] for row in group
                    ),
                    "risky_update_count_mean": mean(
                        row["risky_update_count_mean"] for row in group
                    ),
                }
            )
        return sorted(aggregates, key=lambda row: row["regret_per_t_mean"])

    main_rows = [row for row in eval_rows if row["setting_group"] == "main_variants"]
    safe_rows = [row for row in eval_rows if row["setting_group"] == "safe_suffix"]
    direct_cost_overall = aggregate(DIRECT_COST_METHODS, scope="main_plus_safe_suffix", rows=eval_rows)
    direct_cost_main = aggregate(DIRECT_COST_METHODS, scope="main_variants", rows=main_rows)
    direct_cost_safe = aggregate(DIRECT_COST_METHODS, scope="safe_suffix", rows=safe_rows)
    baseline_overall = aggregate(REFERENCE_METHODS, scope="main_plus_safe_suffix", rows=eval_rows)

    by_setting_method = {
        (row["setting"], row["method"]): row
        for row in eval_rows
    }
    per_setting: list[dict[str, Any]] = []
    for setting in sorted({row["setting"] for row in eval_rows}):
        risky = by_setting_method.get((setting, "risky_ps"))
        ix = by_setting_method.get((setting, "risky_ps_ix"))
        safe = by_setting_method.get((setting, "risky_ps_safe_conditional"))
        direct_cost = by_setting_method.get((setting, "risky_ps_direct_cost"))
        epsilon = by_setting_method.get((setting, "epsilon_exp3"))
        direct = by_setting_method.get((setting, "direct_multistage_exp3"))
        if risky is None or direct_cost is None:
            continue
        payload = {
            "setting": setting,
            "setting_group": risky["setting_group"],
            "variant": risky["variant"],
            "risky_depth": risky["risky_depth"],
            "safe_suffix_length": risky["safe_suffix_length"],
            "global_leaf_regret_per_t": risky["regret_per_t_mean"],
            "global_leaf_ix_regret_per_t": ix["regret_per_t_mean"] if ix else None,
            "safe_conditional_regret_per_t": safe["regret_per_t_mean"] if safe else None,
            "direct_cost_regret_per_t": direct_cost["regret_per_t_mean"],
            "epsilon_exp3_regret_per_t": epsilon["regret_per_t_mean"] if epsilon else None,
            "direct_exp3_regret_per_t": direct["regret_per_t_mean"] if direct else None,
            "delta_direct_cost_vs_global_leaf": (
                direct_cost["regret_per_t_mean"] - risky["regret_per_t_mean"]
            ),
        }
        if ix is not None:
            payload["delta_direct_cost_vs_ix"] = (
                direct_cost["regret_per_t_mean"] - ix["regret_per_t_mean"]
            )
        if safe is not None:
            payload["delta_direct_cost_vs_safe_conditional"] = (
                direct_cost["regret_per_t_mean"] - safe["regret_per_t_mean"]
            )
        if epsilon is not None:
            payload["delta_direct_cost_vs_epsilon_exp3"] = (
                direct_cost["regret_per_t_mean"] - epsilon["regret_per_t_mean"]
            )
        if direct is not None:
            payload["delta_direct_cost_vs_direct_exp3"] = (
                direct_cost["regret_per_t_mean"] - direct["regret_per_t_mean"]
            )
        per_setting.append(payload)

    gap_rows: list[dict[str, Any]] = []
    method_rows = {row["method"]: row for row in direct_cost_overall + baseline_overall}
    for method in DIRECT_COST_METHODS:
        row = method_rows.get(method)
        if row is None:
            continue
        for baseline in ("epsilon_exp3", "direct_multistage_exp3"):
            base = method_rows.get(baseline)
            if base is None:
                continue
            gap_rows.append(
                {
                    "method": method,
                    "baseline": baseline,
                    "delta_regret_per_t": row["regret_per_t_mean"] - base["regret_per_t_mean"],
                    "delta_terminal_proxy": row["terminal_proxy_mean"] - base["terminal_proxy_mean"],
                }
            )

    return {
        "direct_cost_overall": direct_cost_overall,
        "direct_cost_main_variants": direct_cost_main,
        "direct_cost_safe_suffix": direct_cost_safe,
        "baseline_overall": baseline_overall,
        "per_setting": per_setting,
        "gap_vs_baselines": gap_rows,
        "raw_summary": summary,
    }


def write_denominator_ablation_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
    candidate_scaling_included: bool,
) -> None:
    overall = compare["denominator_overall"]
    main = compare["denominator_main_variants"]
    safe = compare["denominator_safe_suffix"]
    baseline = compare["baseline_overall"]
    per_setting = compare["per_setting"]
    gaps = compare["gap_vs_baselines"]

    per_setting_by_name = {row["setting"]: row for row in per_setting}
    all_share = per_setting_by_name.get("main_all_share_L5_K5")
    all_unshare = per_setting_by_name.get("main_all_unshare_L5_K5")
    safe_cond_better = [
        row for row in per_setting
        if row.get("delta_safe_conditional_vs_global_leaf") is not None
        and row["delta_safe_conditional_vs_global_leaf"] < 0.0
    ]
    safe_ix_better = [
        row for row in per_setting
        if row.get("delta_safe_conditional_ix_vs_safe_conditional") is not None
        and row["delta_safe_conditional_ix_vs_safe_conditional"] < 0.0
    ]
    partial_or_suffix_better = [
        row for row in safe_cond_better
        if row["setting_group"] == "safe_suffix" or row["variant"] in {"partial_4of5", "partial_2of5"}
    ]

    direct_answers = [
        (
            f"- safe_conditional improves over global_leaf on "
            f"{len(safe_cond_better)}/{len(per_setting)} settings."
        ),
        (
            f"- safe_conditional_ix improves over safe_conditional on "
            f"{len(safe_ix_better)}/{len(per_setting)} settings."
        ),
        (
            f"- Improvements in partial/safe-suffix settings: "
            f"{len(partial_or_suffix_better)} settings."
        ),
        (
            "- all-share delta safe_conditional - global_leaf = "
            f"{all_share.get('delta_safe_conditional_vs_global_leaf') if all_share else None}."
        ),
        (
            "- all-unshare delta safe_conditional - global_leaf = "
            f"{all_unshare.get('delta_safe_conditional_vs_global_leaf') if all_unshare else None}."
        ),
        "- Candidate scaling was "
        + ("included." if candidate_scaling_included else "skipped for this cheap ablation."),
    ]

    best_denominator = overall[0] if overall else None
    if best_denominator:
        direct_answers.insert(
            0,
            f"- Best denominator method overall: {best_denominator['method']} "
            f"(mean regret/T={best_denominator['regret_per_t_mean']:.6f}).",
        )

    fields_overall = [
        "method",
        "settings",
        "regret_per_t_mean",
        "regret_per_t_std_across_settings",
        "regret_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "shared_update_count_mean",
    ]
    fields_setting = [
        "setting",
        "setting_group",
        "variant",
        "risky_depth",
        "global_leaf_regret_per_t",
        "global_leaf_ix_regret_per_t",
        "safe_conditional_regret_per_t",
        "safe_conditional_ix_regret_per_t",
        "delta_safe_conditional_vs_global_leaf",
        "delta_safe_conditional_ix_vs_safe_conditional",
    ]
    fields_gap = [
        "method",
        "baseline",
        "delta_regret_per_t",
        "delta_terminal_proxy",
    ]
    content = [
        "# Shared denominator ablation",
        "",
        "## Direct Answers",
        *direct_answers,
        "",
        "## Denominator Methods Overall",
        markdown_table(overall, fields_overall),
        "## Main Variants",
        markdown_table(main, fields_overall),
        "## Safe Suffix",
        markdown_table(safe, fields_overall),
        "## Reference Baselines",
        markdown_table(baseline, fields_overall),
        "## Per-Setting Denominator Comparison",
        markdown_table(per_setting, fields_setting),
        "## Gap vs EXP3 Baselines",
        markdown_table(gaps, fields_gap),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_direct_cost_ablation_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
    candidate_scaling_included: bool,
) -> None:
    overall = compare["direct_cost_overall"]
    main = compare["direct_cost_main_variants"]
    safe = compare["direct_cost_safe_suffix"]
    baseline = compare["baseline_overall"]
    per_setting = compare["per_setting"]
    gaps = compare["gap_vs_baselines"]

    per_setting_by_name = {row["setting"]: row for row in per_setting}
    all_unshare = per_setting_by_name.get("main_all_unshare_L5_K5")
    direct_cost_better_risky = [
        row for row in per_setting
        if row.get("delta_direct_cost_vs_global_leaf") is not None
        and row["delta_direct_cost_vs_global_leaf"] < 0.0
    ]
    direct_cost_better_ix = [
        row for row in per_setting
        if row.get("delta_direct_cost_vs_ix") is not None
        and row["delta_direct_cost_vs_ix"] < 0.0
    ]
    direct_cost_better_safe = [
        row for row in per_setting
        if row.get("delta_direct_cost_vs_safe_conditional") is not None
        and row["delta_direct_cost_vs_safe_conditional"] < 0.0
    ]
    partial_or_suffix_better = [
        row for row in direct_cost_better_risky
        if row["setting_group"] == "safe_suffix" or row["variant"] in {"partial_4of5", "partial_2of5"}
    ]

    best_method = overall[0] if overall else None
    direct_answers = [
        (
            f"- risky_ps_direct_cost improves over risky_ps on "
            f"{len(direct_cost_better_risky)}/{len(per_setting)} settings."
        ),
        (
            f"- risky_ps_direct_cost improves over risky_ps_ix on "
            f"{len(direct_cost_better_ix)}/{len(per_setting)} settings."
        ),
        (
            f"- risky_ps_direct_cost improves over risky_ps_safe_conditional on "
            f"{len(direct_cost_better_safe)}/{len(per_setting)} settings."
        ),
        (
            "- Improvements in partial/safe-suffix settings: "
            f"{len(partial_or_suffix_better)} settings."
        ),
        (
            "- all-unshare delta direct_cost - risky_ps = "
            f"{all_unshare.get('delta_direct_cost_vs_global_leaf') if all_unshare else None}."
        ),
        "- Candidate scaling was "
        + ("included." if candidate_scaling_included else "skipped for this cheap ablation."),
    ]
    if best_method:
        direct_answers.insert(
            0,
            f"- Best RiskyPS-family method overall: {best_method['method']} "
            f"(mean regret/T={best_method['regret_per_t_mean']:.6f}).",
        )
    if direct_cost_better_risky:
        direct_answers.append(
            "- Direct-cost improvements indicate the bottleneck is consistent with "
            "high-variance shared importance weighting in the affected settings."
        )
    else:
        direct_answers.append(
            "- Direct-cost did not improve over global-leaf RiskyPS in this run, "
            "so shared leaf importance-weight variance is not the main bottleneck here."
        )

    fields_overall = [
        "method",
        "settings",
        "regret_per_t_mean",
        "regret_per_t_std_across_settings",
        "regret_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "shared_update_count_mean",
    ]
    fields_setting = [
        "setting",
        "setting_group",
        "variant",
        "risky_depth",
        "global_leaf_regret_per_t",
        "global_leaf_ix_regret_per_t",
        "safe_conditional_regret_per_t",
        "direct_cost_regret_per_t",
        "delta_direct_cost_vs_global_leaf",
        "delta_direct_cost_vs_ix",
        "delta_direct_cost_vs_safe_conditional",
    ]
    fields_gap = [
        "method",
        "baseline",
        "delta_regret_per_t",
        "delta_terminal_proxy",
    ]
    content = [
        "# Direct-cost shared update ablation",
        "",
        "## Direct Answers",
        *direct_answers,
        "",
        "## RiskyPS-Family Methods Overall",
        markdown_table(overall, fields_overall),
        "## Main Variants",
        markdown_table(main, fields_overall),
        "## Safe Suffix",
        markdown_table(safe, fields_overall),
        "## Reference Baselines",
        markdown_table(baseline, fields_overall),
        "## Per-Setting Direct-Cost Comparison",
        markdown_table(per_setting, fields_setting),
        "## Gap vs EXP3 Baselines",
        markdown_table(gaps, fields_gap),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def write_ix_grid_summary_markdown(
    path: Path,
    *,
    compare: dict[str, Any],
    candidate_scaling_included: bool,
) -> None:
    overall = compare["parameter_grid_overall"]
    main = compare["parameter_grid_main_variants"]
    safe = compare["parameter_grid_safe_suffix"]
    baseline = compare["baseline_overall"]
    deltas = compare["baseline_delta_overall"]
    best_by_setting = compare["best_ix_by_setting"]
    best = overall[0] if overall else None

    beats_risky = [
        row.get("beats_risky_ps")
        for row in best_by_setting
        if "beats_risky_ps" in row
    ]
    beats_epsilon = [
        row.get("beats_epsilon_exp3")
        for row in best_by_setting
        if "beats_epsilon_exp3" in row
    ]
    beats_direct = [
        row.get("beats_direct_exp3")
        for row in best_by_setting
        if "beats_direct_exp3" in row
    ]

    def count_true(values: list[Any]) -> int:
        return sum(1 for value in values if bool(value))

    conclusion_lines: list[str] = []
    if best is None:
        conclusion_lines.append("- No risky_ps_ix rows were generated.")
    else:
        conclusion_lines.append(
            f"- Best overall IX grid point: eta_shared={best['eta_shared']}, "
            f"gamma_shared={best['gamma_shared']} "
            f"(mean regret/T={best['regret_per_t_mean']:.6f})."
        )
        if beats_risky:
            conclusion_lines.append(
                f"- IX beats risky_ps on {count_true(beats_risky)}/{len(beats_risky)} "
                "main/safe-suffix settings using the best IX point per setting."
            )
        if beats_epsilon:
            conclusion_lines.append(
                f"- IX beats epsilon_exp3 on {count_true(beats_epsilon)}/{len(beats_epsilon)} "
                "main/safe-suffix settings using the best IX point per setting."
            )
        if beats_direct:
            conclusion_lines.append(
                f"- IX beats direct_multistage_exp3 on {count_true(beats_direct)}/{len(beats_direct)} "
                "main/safe-suffix settings using the best IX point per setting."
            )
        gamma_values = sorted({row["gamma_shared"] for row in overall})
        if gamma_values and best["gamma_shared"] == gamma_values[0]:
            conclusion_lines.append(
                "- The best gamma is at the smallest tested value; if IX is weak, the grid suggests "
                "larger gamma slows shared learning rather than fixing the bottleneck."
            )
        elif gamma_values and best["gamma_shared"] == gamma_values[-1]:
            conclusion_lines.append(
                "- The best gamma is at the largest tested value; more clipping may be worth testing "
                "only if IX also improves over references."
            )
        else:
            conclusion_lines.append(
                "- The best gamma is interior to the tested range, suggesting a real bias/variance tradeoff."
            )
    conclusion_lines.append(
        "- Candidate scaling was "
        + ("included." if candidate_scaling_included else "skipped for this cheap IX grid.")
    )

    fields_param = [
        "eta_shared",
        "gamma_shared",
        "settings",
        "regret_per_t_mean",
        "regret_per_t_std_across_settings",
        "regret_mean",
        "terminal_proxy_mean",
    ]
    fields_baseline = [
        "method",
        "settings",
        "regret_per_t_mean",
        "regret_mean",
        "terminal_proxy_mean",
        "shared_path_fraction_mean",
        "shared_update_count_mean",
    ]
    fields_delta = [
        "compare_to",
        "best_ix_eta_shared",
        "best_ix_gamma_shared",
        "delta_regret_per_t",
        "delta_terminal_proxy",
    ]
    fields_setting = [
        "setting",
        "setting_group",
        "variant",
        "risky_depth",
        "eta_shared",
        "gamma_shared",
        "regret_per_t_mean",
        "delta_regret_per_t_vs_risky_ps",
        "delta_regret_per_t_vs_epsilon_exp3",
        "delta_regret_per_t_vs_direct_exp3",
    ]

    content = [
        "# risky_ps_ix controlled simulation grid",
        "",
        "## Direct Answers",
        *conclusion_lines,
        "",
        "## Best Overall Parameters",
        markdown_table(overall[:8], fields_param),
        "## Best Main Variant Parameters",
        markdown_table(main[:8], fields_param),
        "## Best Safe-Suffix Parameters",
        markdown_table(safe[:8], fields_param),
        "## Reference Baselines",
        markdown_table(baseline, fields_baseline),
        "## Best IX vs References",
        markdown_table(deltas, fields_delta),
        "## Best IX By Setting",
        markdown_table(best_by_setting, fields_setting),
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run theory-aligned BarrierShare controlled simulation without LLMs.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "barriershare_controlled_sim_v2_theory_aligned",
    )
    parser.add_argument("--depth", type=int, default=5)
    parser.add_argument("--branching", type=int, default=5)
    parser.add_argument("--candidate-depth", type=int, default=3)
    parser.add_argument("--candidate-branching", type=int, nargs="+", default=[5, 15, 25])
    parser.add_argument("--horizon", type=int, default=5000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cost-noise", type=float, default=0.02)
    parser.add_argument("--specialist-fraction", type=float, default=0.15)
    parser.add_argument("--methods", nargs="+")
    parser.add_argument(
        "--tree-spec",
        type=Path,
        help="Run a single spec-backed controlled-sim setting from an external tree spec JSON.",
    )
    parser.add_argument(
        "--old-reference-dir",
        type=Path,
        default=ROOT / "outputs" / "barriershare_controlled_sim_v2_theory_aligned",
        help="Reference controlled-sim directory used for unique-tree delta summaries.",
    )
    parser.add_argument(
        "--old-unique-reference-dir",
        type=Path,
        default=ROOT / "outputs" / "barriershare_controlled_sim_unique_agents_4of5_v1",
        help="Older unique-agent controlled-sim directory used for unbound tree delta summaries.",
    )
    parser.add_argument(
        "--old-unbound-reference-dir",
        type=Path,
        default=ROOT / "outputs" / "barriershare_controlled_sim_unique_agents_fullbranch_unbound_v1",
        help="Earlier full-branch unbound unique-agent controlled-sim directory used for subtree-local delta summaries.",
    )
    parser.add_argument(
        "--tree-spec-role-mode",
        choices=TREE_SPEC_ROLE_MODES,
        default="spec_or_agent_id",
        help=(
            "How --tree-spec constructs synthetic cost roles: prefer explicit spec field then "
            "agent_id (default), force agent_id, use subtree-local correlated cost, or force legacy base_alias mapping."
        ),
    )
    parser.add_argument(
        "--tree-spec-cost-mode",
        choices=TREE_SPEC_COST_MODES,
        default="default",
        help=(
            "Cost landscape for --tree-spec runs. The default preserves existing behavior; "
            "ps_favored_trap keeps the tree fixed and replaces only leaf costs with a "
            "Bernoulli safe-corridor/trap landscape."
        ),
    )
    parser.add_argument(
        "--direct-eta-override",
        type=float,
        default=None,
        help=(
            "Override eta only for direct_multistage_exp3. Defaults to the policy's "
            "built-in eta and does not affect other methods."
        ),
    )
    parser.add_argument(
        "--common-eta-override",
        type=float,
        default=None,
        help=(
            "Override the main eta for direct_multistage_exp3, epsilon_exp3, and all "
            "PS-family methods. Defaults preserve each policy's built-in eta."
        ),
    )
    parser.add_argument(
        "--common-epsilon-override",
        type=float,
        default=None,
        help=(
            "Override explicit exploration epsilon for epsilon_exp3 and all PS-family "
            "methods. direct_multistage_exp3, naive_mixed, and random_path are unaffected."
        ),
    )
    parser.add_argument("--ix-grid", action="store_true", help="Run the risky_ps_ix eta_shared/gamma_shared grid.")
    parser.add_argument(
        "--denominator-ablation",
        action="store_true",
        help="Run shared estimator denominator ablation variants.",
    )
    parser.add_argument(
        "--direct-cost-ablation",
        action="store_true",
        help="Run direct-observed-cost shared update ablation variants.",
    )
    parser.add_argument(
        "--ix-eta-shared-values",
        default=",".join(str(value) for value in DEFAULT_IX_ETA_SHARED_VALUES),
        help="Comma-separated eta_shared grid for --ix-grid.",
    )
    parser.add_argument(
        "--ix-gamma-shared-values",
        default=",".join(str(value) for value in DEFAULT_IX_GAMMA_SHARED_VALUES),
        help="Comma-separated gamma_shared grid for --ix-grid.",
    )
    parser.add_argument(
        "--include-candidate-scaling",
        action="store_true",
        help="Include candidate scaling settings in cheap ablation modes.",
    )
    parser.add_argument(
        "--safe-conditional-gamma-shared",
        type=float,
        default=0.0005,
        help="gamma_shared for risky_ps_safe_conditional_ix.",
    )
    args = parser.parse_args()

    enabled_ablation_modes = [
        name
        for name, enabled in (
            ("--ix-grid", args.ix_grid),
            ("--denominator-ablation", args.denominator_ablation),
            ("--direct-cost-ablation", args.direct_cost_ablation),
        )
        if enabled
    ]
    if len(enabled_ablation_modes) > 1:
        raise SystemExit(f"Use only one ablation mode at a time: {enabled_ablation_modes}.")
    if args.tree_spec and enabled_ablation_modes:
        raise SystemExit("--tree-spec cannot be combined with ablation grid modes.")
    if args.tree_spec and not args.tree_spec.exists():
        raise SystemExit(f"Missing tree spec: {args.tree_spec}")
    if args.common_eta_override is not None and args.direct_eta_override is not None:
        raise SystemExit("Use either --common-eta-override or --direct-eta-override, not both.")

    if args.methods is None:
        args.methods = list(REFERENCE_METHODS) if args.tree_spec else list(METHODS)

    invalid_methods = [method for method in args.methods if method not in METHODS]
    if invalid_methods:
        raise SystemExit(f"Unknown methods: {invalid_methods}; allowed={sorted(METHODS)}")

    eta_shared_values = parse_float_list(args.ix_eta_shared_values)
    gamma_shared_values = parse_float_list(args.ix_gamma_shared_values)
    if args.ix_grid and (not eta_shared_values or not gamma_shared_values):
        raise SystemExit("--ix-grid requires non-empty eta_shared and gamma_shared grids.")

    if args.tree_spec:
        settings = [
            {
                "name": args.tree_spec.stem,
                "group": "external_tree",
                "variant": args.tree_spec.stem,
                "depth": None,
                "branching": None,
                "sharing_scheme": "external_tree_spec",
                "risky_depth": None,
                "tree_spec": args.tree_spec,
            }
        ]
    else:
        settings = build_settings(
            main_depth=args.depth,
            main_branching=args.branching,
            candidate_depth=args.candidate_depth,
            candidate_branching=args.candidate_branching,
        )
        if (
            args.ix_grid
            or args.denominator_ablation
            or args.direct_cost_ablation
        ) and not args.include_candidate_scaling:
            settings = [setting for setting in settings if setting["group"] != "candidate_scaling"]
    if args.ix_grid:
        run_specs = build_ix_run_specs(
            eta_shared_values=eta_shared_values,
            gamma_shared_values=gamma_shared_values,
        )
    elif args.denominator_ablation:
        run_specs = build_denominator_ablation_run_specs(
            gamma_shared=args.safe_conditional_gamma_shared,
        )
    elif args.direct_cost_ablation:
        run_specs = build_direct_cost_ablation_run_specs()
    else:
        run_specs = [
            {
                "method": method,
                "method_label": method,
                "policy_kwargs": {},
            }
            for method in args.methods
        ]
    if args.common_eta_override is not None:
        for run_spec in run_specs:
            if run_spec["method"] in COMMON_ETA_METHODS:
                run_spec["policy_kwargs"] = dict(run_spec.get("policy_kwargs", {}))
                run_spec["policy_kwargs"]["eta"] = args.common_eta_override
                run_spec["common_eta_override"] = args.common_eta_override
    if args.common_epsilon_override is not None:
        for run_spec in run_specs:
            if run_spec["method"] in COMMON_EPSILON_METHODS:
                run_spec["policy_kwargs"] = dict(run_spec.get("policy_kwargs", {}))
                run_spec["policy_kwargs"]["epsilon"] = args.common_epsilon_override
                run_spec["common_epsilon_override"] = args.common_epsilon_override
    if args.direct_eta_override is not None:
        for run_spec in run_specs:
            if run_spec["method"] == "direct_multistage_exp3":
                run_spec["policy_kwargs"] = dict(run_spec.get("policy_kwargs", {}))
                run_spec["policy_kwargs"]["eta"] = args.direct_eta_override
                run_spec["direct_eta_override"] = args.direct_eta_override
    rows: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    ps_favored_trap_diagnostics: dict[str, Any] | None = None

    for setting in settings:
        for seed in args.seeds:
            if args.tree_spec:
                env = SpecBackedControlledTreeEnv(
                    spec_path=setting["tree_spec"],
                    seed=seed,
                    cost_noise=args.cost_noise,
                    specialist_fraction=args.specialist_fraction,
                    tree_spec_role_mode=args.tree_spec_role_mode,
                    tree_spec_cost_mode=args.tree_spec_cost_mode,
                )
            else:
                env = ControlledTreeEnv(
                    setting_name=setting["name"],
                    variant=setting["variant"],
                    depth=setting["depth"],
                    branching=setting["branching"],
                    seed=seed,
                    cost_noise=args.cost_noise,
                    specialist_fraction=args.specialist_fraction,
                    sharing_scheme=setting["sharing_scheme"],
                    risky_depth=setting["risky_depth"],
                )
            env.setting_group = setting["group"]
            env.setting_risky_depth = setting["risky_depth"]
            instances = build_instances(
                horizon=args.horizon,
                seed=seed,
                specialist_fraction=args.specialist_fraction,
            )
            oracle = env.oracle_reference(instances)
            if (
                args.tree_spec
                and args.tree_spec_cost_mode == "ps_favored_trap"
                and isinstance(env, SpecBackedControlledTreeEnv)
                and ps_favored_trap_diagnostics is None
            ):
                ps_favored_trap_diagnostics = env.ps_favored_trap_diagnostics(
                    instances=instances,
                    oracle=oracle,
                )
            for run_spec in run_specs:
                method = run_spec["method"]
                row, curve = run_one(
                    env=env,
                    instances=instances,
                    oracle=oracle,
                    method=method,
                    method_label=run_spec["method_label"],
                    policy_kwargs=run_spec["policy_kwargs"],
                    common_eta_override=run_spec.get("common_eta_override"),
                    common_epsilon_override=run_spec.get("common_epsilon_override"),
                    direct_eta_override=run_spec.get("direct_eta_override"),
                    seed=seed,
                    horizon=args.horizon,
                )
                rows.append(row)
                curves.extend(curve)
                print(
                    f"[controlled-sim] setting={setting['name']} seed={seed} "
                    f"method={row['method_label']} regret/t={row['regret_per_t']:.6f}",
                    flush=True,
                )

    summary = summarize(rows)
    if args.tree_spec:
        permutation_example_env = SpecBackedControlledTreeEnv(
            spec_path=args.tree_spec,
            seed=args.seeds[0],
            cost_noise=args.cost_noise,
            specialist_fraction=args.specialist_fraction,
            tree_spec_role_mode=args.tree_spec_role_mode,
            tree_spec_cost_mode=args.tree_spec_cost_mode,
        )
        findings = {
            "external_tree_spec": str(args.tree_spec),
            "tree_spec_role_mode": args.tree_spec_role_mode,
            "tree_spec_cost_mode": args.tree_spec_cost_mode,
            "permutation_example_seed0": permutation_example_env.describe_role_permutation(),
        }
        if ps_favored_trap_diagnostics is not None:
            findings["ps_favored_trap_diagnostics"] = ps_favored_trap_diagnostics
    else:
        permutation_example_env = ControlledTreeEnv(
            setting_name=f"main_all_share_L{args.depth}_K{args.branching}",
            variant="all_share",
            depth=args.depth,
            branching=args.branching,
            seed=args.seeds[0],
            cost_noise=args.cost_noise,
            specialist_fraction=args.specialist_fraction,
            sharing_scheme="main_variant",
        )
        findings = build_findings(
            summary=summary,
            permutation_example=permutation_example_env.describe_role_permutation(),
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized_settings = json.loads(json.dumps(settings, default=str))
    write_json(
        output_dir / "run_config.json",
        {
            "runner": "run_barriershare_controlled_sim.py",
            "llm_api_used": False,
            "depth": args.depth,
            "branching": args.branching,
            "candidate_depth": args.candidate_depth,
            "candidate_branching": args.candidate_branching,
            "horizon": args.horizon,
            "seeds": args.seeds,
            "cost_noise": args.cost_noise,
            "specialist_fraction": args.specialist_fraction,
            "methods": args.methods,
            "tree_spec": str(args.tree_spec) if args.tree_spec else None,
            "tree_spec_role_mode": args.tree_spec_role_mode,
            "tree_spec_cost_mode": args.tree_spec_cost_mode,
            "common_eta_override": args.common_eta_override,
            "common_epsilon_override": args.common_epsilon_override,
            "direct_eta_override": args.direct_eta_override,
            "old_reference_dir": str(args.old_reference_dir),
            "old_unique_reference_dir": str(args.old_unique_reference_dir),
            "old_unbound_reference_dir": str(args.old_unbound_reference_dir),
            "run_specs": run_specs,
            "ix_grid": args.ix_grid,
            "denominator_ablation": args.denominator_ablation,
            "direct_cost_ablation": args.direct_cost_ablation,
            "ix_eta_shared_values": eta_shared_values,
            "ix_gamma_shared_values": gamma_shared_values,
            "safe_conditional_gamma_shared": args.safe_conditional_gamma_shared,
            "candidate_scaling_included": (
                (
                    not args.ix_grid
                    and not args.denominator_ablation
                    and not args.direct_cost_ablation
                    and not args.tree_spec
                )
                or args.include_candidate_scaling
            ),
            "settings": serialized_settings,
            "simulation_design": {
                "main_variants_full_branching": not args.tree_spec,
                "safe_suffix_full_branching": not args.tree_spec,
                "candidate_scaling_full_branching": (
                    (
                        not args.ix_grid
                        and not args.denominator_ablation
                        and not args.direct_cost_ablation
                        and not args.tree_spec
                    )
                    or args.include_candidate_scaling
                ),
                "latent_role_permutation": not args.tree_spec,
                "cost_landscape": (
                    "external_tree_ps_favored_trap_bernoulli"
                    if args.tree_spec and args.tree_spec_cost_mode == "ps_favored_trap"
                    else (
                        "external_tree_base_alias_family"
                        if args.tree_spec and args.tree_spec_role_mode == "base_alias"
                        else (
                            "external_tree_subtree_local_correlated"
                            if args.tree_spec and args.tree_spec_role_mode == "subtree_local"
                            else "external_tree_unbound_unique_suffix"
                        )
                        if args.tree_spec
                        else "shared_suffix_family"
                    )
                ),
            },
        },
    )
    write_json(output_dir / "per_seed_results.json", rows)
    write_csv(output_dir / "per_seed_results.csv", rows)
    write_json(output_dir / "controlled_sim_compare.json", summary)
    write_csv(output_dir / "controlled_sim_compare.csv", summary)
    write_json(output_dir / "regret_curve.json", curves)
    write_json(output_dir / "findings.json", findings)

    main_rows = [row for row in summary if row["setting_group"] == "main_variants"]
    safe_rows = [row for row in summary if row["setting_group"] == "safe_suffix"]
    scaling_rows = [row for row in summary if row["setting_group"] == "candidate_scaling"]
    if not args.tree_spec:
        write_group_outputs(
            output_dir,
            "main_variant_compare",
            main_rows,
            [
                "setting",
                "method",
                "num_paths",
                "regret_per_t_mean",
                "average_cost_mean",
                "exact_best_path_hit_rate_mean",
                "first_episode_best_hit_rate_mean",
                "shared_path_fraction_mean",
            ],
        )
        write_group_outputs(
            output_dir,
            "safe_suffix_compare",
            safe_rows,
            [
                "setting",
                "method",
                "risky_depth",
                "safe_suffix_length",
                "num_paths",
                "regret_per_t_mean",
                "average_cost_mean",
                "shared_path_fraction_mean",
            ],
        )
        write_group_outputs(
            output_dir,
            "candidate_scaling_compare",
            scaling_rows,
            [
                "setting",
                "method",
                "depth",
                "branching",
                "num_paths",
                "regret_per_t_mean",
                "average_cost_mean",
                "shared_path_fraction_mean",
            ],
        )
    (output_dir / "controlled_sim_compare.md").write_text(
        markdown_table(
            summary,
            [
                "setting",
                "method_label",
                "eta_shared",
                "gamma_shared",
                "num_paths",
                "regret_per_t_mean",
                "regret_per_t_std",
                "average_cost_mean",
                "exact_best_path_hit_rate_mean",
                "shared_path_fraction_mean",
            ],
        ),
        encoding="utf-8",
    )
    if args.tree_spec:
        validation_path = args.tree_spec.parents[1] / (
            args.tree_spec.stem.replace("_minimal", "") + "_validation.json"
        )
        validation = (
            json.loads(validation_path.read_text(encoding="utf-8"))
            if validation_path.exists()
            else {}
        )
        unique_compare = build_unique_agents_compare(
            summary=summary,
            validation=validation,
            old_reference_rows=load_reference_partial_4of5(args.old_reference_dir),
        )
        write_json(output_dir / "unique_agents_compare.json", unique_compare)
        (output_dir / "unique_agents_compare.md").write_text(
            markdown_table(
                unique_compare["new_tree"],
                [
                    "method",
                    "regret_per_t_mean",
                    "average_cost_mean",
                    "shared_path_fraction_mean",
                    "num_paths",
                    "horizon",
                    "seeds",
                ],
            ),
            encoding="utf-8",
        )
        write_unique_agents_summary_markdown(
            output_dir / "unique_agents_summary.md",
            compare=unique_compare,
        )
        if args.tree_spec_role_mode != "base_alias":
            unbound_compare = build_unique_agents_unbound_compare(
                summary=summary,
                validation=validation,
                old_unique_rows=load_old_unique_agents_reference(args.old_unique_reference_dir),
                old_theory_rows=load_reference_partial_4of5(args.old_reference_dir),
                role_mode=args.tree_spec_role_mode,
            )
            write_json(output_dir / "unique_agents_unbound_compare.json", unbound_compare)
            (output_dir / "unique_agents_unbound_compare.md").write_text(
                markdown_table(
                    unbound_compare["current_tree"],
                    [
                        "method",
                        "regret_per_t_mean",
                        "average_cost_mean",
                        "shared_path_fraction_mean",
                        "num_paths",
                        "horizon",
                        "seeds",
                    ],
                ),
                encoding="utf-8",
            )
            write_unique_agents_unbound_summary_markdown(
                output_dir / "unique_agents_unbound_summary.md",
                compare=unbound_compare,
            )
        if args.tree_spec_role_mode == "subtree_local":
            same_topology_compare = build_same_topology_unique_agents_compare(
                summary=summary,
                validation=validation,
                old_theory_rows=load_reference_partial_4of5(args.old_reference_dir),
                old_unbound_rows=load_old_unique_agents_reference(args.old_unbound_reference_dir),
                role_mode=args.tree_spec_role_mode,
            )
            write_json(output_dir / "same_topology_unique_agents_compare.json", same_topology_compare)
            (output_dir / "same_topology_unique_agents_compare.md").write_text(
                markdown_table(
                    same_topology_compare["current_tree"],
                    [
                        "method",
                        "regret_per_t_mean",
                        "average_cost_mean",
                        "shared_path_fraction_mean",
                        "num_paths",
                        "horizon",
                        "seeds",
                    ],
                ),
                encoding="utf-8",
            )
            write_same_topology_unique_agents_summary_markdown(
                output_dir / "same_topology_unique_agents_summary.md",
                compare=same_topology_compare,
            )
        if "prefix_dedup" in args.tree_spec.stem:
            prefix_dedup_compare = build_prefix_dedup_compare(
                summary=summary,
                validation=validation,
                old_theory_rows=load_reference_partial_4of5(args.old_reference_dir),
                role_mode=args.tree_spec_role_mode,
            )
            write_json(output_dir / "prefix_dedup_compare.json", prefix_dedup_compare)
            (output_dir / "prefix_dedup_compare.md").write_text(
                markdown_table(
                    prefix_dedup_compare["current_tree"],
                    [
                        "method",
                        "regret_per_t_mean",
                        "terminal_proxy_mean",
                        "shared_path_fraction_mean",
                        "trap_basin_fraction_mean",
                        "target_subtree_fraction_mean",
                        "target_good_fraction_mean",
                        "target_bad_fraction_mean",
                        "calibrated_decoy_fraction_mean",
                        "decoy_branch_fraction_mean",
                        "ordinary_safe_basin_fraction_mean",
                        "broad_safe_basin_fraction_mean",
                        "ps_favored_exact_best_hit_rate_mean",
                        "num_paths",
                        "horizon",
                        "seeds",
                    ],
                ),
                encoding="utf-8",
            )
            write_prefix_dedup_summary_markdown(
                output_dir / "prefix_dedup_summary.md",
                compare=prefix_dedup_compare,
            )
        if args.tree_spec_cost_mode == "ps_favored_trap":
            if ps_favored_trap_diagnostics is None:
                raise RuntimeError("Missing ps_favored_trap diagnostics for cost-mode run.")
            ps_favored_compare = build_ps_favored_trap_compare(
                summary=summary,
                diagnostics=ps_favored_trap_diagnostics,
            )
            write_json(output_dir / "ps_favored_trap_compare.json", ps_favored_compare)
            (output_dir / "ps_favored_trap_compare.md").write_text(
                markdown_table(
                    ps_favored_compare["current_tree"],
                    [
                        "method",
                        "regret_per_t_mean",
                        "terminal_proxy_mean",
                        "shared_path_fraction_mean",
                        "trap_basin_fraction_mean",
                        "target_subtree_fraction_mean",
                        "target_good_fraction_mean",
                        "target_bad_fraction_mean",
                        "calibrated_decoy_fraction_mean",
                        "decoy_branch_fraction_mean",
                        "ordinary_safe_basin_fraction_mean",
                        "broad_safe_basin_fraction_mean",
                        "ps_favored_exact_best_hit_rate_mean",
                        "num_paths",
                        "horizon",
                        "seeds",
                    ],
                ),
                encoding="utf-8",
            )
            write_ps_favored_trap_summary_markdown(
                output_dir / "ps_favored_trap_summary.md",
                compare=ps_favored_compare,
            )
    if args.ix_grid:
        compare = build_ix_grid_compare(summary)
        write_json(output_dir / "ix_grid_raw.json", rows)
        write_json(output_dir / "ix_grid_compare.json", compare)
        write_csv(output_dir / "ix_grid_compare_summary.csv", compare["raw_summary"])
        write_ix_grid_summary_markdown(
            output_dir / "ix_grid_summary.md",
            compare=compare,
            candidate_scaling_included=args.include_candidate_scaling,
        )
    if args.denominator_ablation:
        denominator_compare = build_denominator_ablation_compare(summary)
        write_json(output_dir / "denominator_ablation_raw.json", rows)
        write_json(output_dir / "denominator_ablation_compare.json", denominator_compare)
        write_csv(
            output_dir / "denominator_ablation_compare_summary.csv",
            denominator_compare["raw_summary"],
        )
        write_denominator_ablation_summary_markdown(
            output_dir / "denominator_ablation_summary.md",
            compare=denominator_compare,
            candidate_scaling_included=args.include_candidate_scaling,
        )
    if args.direct_cost_ablation:
        direct_cost_compare = build_direct_cost_ablation_compare(summary)
        write_json(output_dir / "direct_cost_ablation_raw.json", rows)
        write_json(output_dir / "direct_cost_ablation_compare.json", direct_cost_compare)
        write_csv(
            output_dir / "direct_cost_ablation_compare_summary.csv",
            direct_cost_compare["raw_summary"],
        )
        write_direct_cost_ablation_summary_markdown(
            output_dir / "direct_cost_ablation_summary.md",
            compare=direct_cost_compare,
            candidate_scaling_included=args.include_candidate_scaling,
        )
    print(json.dumps({"output_dir": str(output_dir), "findings": findings}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
