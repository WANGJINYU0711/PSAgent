from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from statistics import mean
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_ROOT = REPO_ROOT / "envs"
BASELINES_ROOT = REPO_ROOT / "baselines"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from tree_family.generator import TreeFamilyGenerator  # noqa: E402
from tree_family.specs import CAPABILITY_NAMES, FamilySpec  # noqa: E402


DATASET_PATH = (
    REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities" / "tasks.json"
)
OUTPUT_PATH = REPO_ROOT / "analysis" / "shared_basin_strong_static_analysis.json"
STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
TASK_ID_RE = re.compile(r"^\[(?P<family>[^\]]+)\](?P<body>.*)\[PERSONA:(?P<persona>[^\]]+)\]$")
SPECIALIST_BLOCKERS = {
    "break_apn_mms_setting",
    "user_abroad_roaming_disabled_on",
    "user_abroad_roaming_enabled_off",
    "user_abroad_roaming_disabled_off",
    "bad_network_preference",
    "bad_wifi_calling",
}


def load_rows() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text())


def parse_task_id(task_id: str) -> dict[str, Any]:
    match = TASK_ID_RE.match(task_id)
    if not match:
        raise ValueError(f"Unexpected task id: {task_id}")
    body = match.group("body")
    blockers = [] if not body else body.split("|")
    return {"blockers": blockers, "persona": match.group("persona")}


def enumerate_paths(
    family_spec: FamilySpec,
    *,
    topology_aware: bool,
) -> list[tuple[str, ...]]:
    if topology_aware and family_spec.allowed_children:
        out: list[tuple[str, ...]] = []

        def dfs(prefix: tuple[str, ...]) -> None:
            if len(prefix) == len(STAGES):
                out.append(prefix)
                return
            child_ids = family_spec.allowed_children.get(prefix)
            if child_ids is None:
                stage_name = STAGES[len(prefix)]
                child_ids = family_spec.stage_agents[stage_name]
            for agent_id in child_ids:
                dfs(prefix + (agent_id,))

        dfs(())
        return out

    out: list[tuple[str, ...]] = [()]
    for stage_name in STAGES:
        out = [
            prefix + (agent_id,)
            for prefix in out
            for agent_id in family_spec.stage_agents[stage_name]
        ]
    return out


def stage_match(requirement: dict[str, float], skill: dict[str, float]) -> float:
    denom = max(1e-9, sum(requirement.values()))
    numer = sum(requirement[capability_name] * skill[capability_name] for capability_name in CAPABILITY_NAMES)
    return numer / denom


def path_match(row: dict[str, Any], path: tuple[str, ...], agent_map: dict[str, Any]) -> float:
    scores = []
    for stage_name, agent_id in zip(STAGES, path):
        requirement = row[stage_name]["capability_requirements"]
        skill = agent_map[agent_id].attribute_skill
        scores.append(stage_match(requirement, skill))
    return mean(scores)


def is_shared_path(path: tuple[str, ...], agent_map: dict[str, Any]) -> bool:
    return all(agent_map[agent_id].g == 0 for agent_id in path)


def is_special_task(row: dict[str, Any]) -> bool:
    blockers = set(parse_task_id(row["original_task_id"])["blockers"])
    return bool(blockers & SPECIALIST_BLOCKERS)


def build_safe_prefix_stats(
    paths: list[tuple[str, ...]],
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    descendant_types: dict[tuple[str, ...], set[str]] = {}
    subtree_nonleaf_all_share: dict[tuple[str, ...], bool] = {(): True}
    all_prefixes: set[tuple[str, ...]] = {()}

    for path in paths:
        leaf_type = "shared" if is_shared_path(path, agent_map) else "unshared"
        for depth in range(0, len(path) + 1):
            prefix = tuple(path[:depth])
            all_prefixes.add(prefix)
            descendant_types.setdefault(prefix, set()).add(leaf_type)
            if 0 < depth < len(path):
                gate_is_share = agent_map[prefix[-1]].g == 0
                subtree_nonleaf_all_share[prefix] = (
                    subtree_nonleaf_all_share.get(prefix, True) and gate_is_share
                )

    safe_prefixes: dict[tuple[str, ...], bool] = {}
    safe_prefix_depth_counts: dict[int, int] = {}
    internal_safe_prefix_depth_counts: dict[int, int] = {}
    propagation_edge_count_distribution: dict[int, int] = {}
    min_reachable_safe_depth_counts: dict[int, int] = {}

    for prefix in all_prefixes:
        safe = (
            bool(descendant_types.get(prefix))
            and descendant_types[prefix] == {"shared"}
            and subtree_nonleaf_all_share.get(prefix, True)
        )
        safe_prefixes[prefix] = safe
        if safe:
            depth = len(prefix)
            safe_prefix_depth_counts[depth] = safe_prefix_depth_counts.get(depth, 0) + 1
            if depth < len(STAGES):
                internal_safe_prefix_depth_counts[depth] = (
                    internal_safe_prefix_depth_counts.get(depth, 0) + 1
                )

    for path in paths:
        if not is_shared_path(path, agent_map):
            continue
        propagation_edge_count = 0
        min_reachable_depth: int | None = None
        for depth in range(len(path) - 1, -1, -1):
            prefix = tuple(path[:depth])
            if not safe_prefixes.get(prefix, False):
                break
            propagation_edge_count += 1
            min_reachable_depth = depth
        propagation_edge_count_distribution[propagation_edge_count] = (
            propagation_edge_count_distribution.get(propagation_edge_count, 0) + 1
        )
        if min_reachable_depth is not None:
            min_reachable_safe_depth_counts[min_reachable_depth] = (
                min_reachable_safe_depth_counts.get(min_reachable_depth, 0) + 1
            )

    mean_propagation_edges = 0.0
    if propagation_edge_count_distribution:
        total_shared = sum(propagation_edge_count_distribution.values())
        weighted = sum(
            edge_count * count
            for edge_count, count in propagation_edge_count_distribution.items()
        )
        mean_propagation_edges = weighted / total_shared

    return {
        "num_safe_prefixes": sum(1 for is_safe in safe_prefixes.values() if is_safe),
        "safe_prefix_depth_counts": {
            str(depth): count for depth, count in sorted(safe_prefix_depth_counts.items())
        },
        "internal_safe_prefix_depth_counts": {
            str(depth): count for depth, count in sorted(internal_safe_prefix_depth_counts.items())
        },
        "has_internal_safe_prefix_depths": {
            str(depth): bool(internal_safe_prefix_depth_counts.get(depth, 0))
            for depth in range(1, len(STAGES))
        },
        "shared_leaf_propagation_edge_count_distribution": {
            str(edge_count): count
            for edge_count, count in sorted(propagation_edge_count_distribution.items())
        },
        "shared_leaf_min_reachable_safe_prefix_depth_counts": {
            str(depth): count for depth, count in sorted(min_reachable_safe_depth_counts.items())
        },
        "mean_shared_leaf_propagation_edges": round(mean_propagation_edges, 4),
    }


def build_match_summary(
    rows: list[dict[str, Any]],
    family_spec: FamilySpec,
    agent_map: dict[str, Any],
    *,
    topology_aware: bool,
) -> dict[str, Any]:
    paths = enumerate_paths(family_spec, topology_aware=topology_aware)

    shared_path_count = sum(1 for path in paths if is_shared_path(path, agent_map))
    top1_shared = 0
    top1_unshared = 0
    top5_shared_majority = 0
    top5_shared_fraction_values: list[float] = []
    shared_best_gap_values: list[float] = []
    shared_best_wins = 0
    unshared_best_wins = 0
    unshared_win_task_ids: list[str] = []

    for row in rows:
        scored = []
        for path in paths:
            scored.append((path_match(row, path, agent_map), path))
        scored.sort(key=lambda item: item[0], reverse=True)

        top1_score, top1_path = scored[0]
        top5 = scored[:5]
        top5_shared_count = sum(1 for _, path in top5 if is_shared_path(path, agent_map))
        top5_shared_fraction_values.append(top5_shared_count / len(top5))
        if top5_shared_count >= 3:
            top5_shared_majority += 1

        best_shared_score = max(score for score, path in scored if is_shared_path(path, agent_map))
        best_unshared_score = max(score for score, path in scored if not is_shared_path(path, agent_map))
        shared_best_gap_values.append(best_shared_score - top1_score)

        if is_shared_path(top1_path, agent_map):
            top1_shared += 1
        else:
            top1_unshared += 1

        if best_shared_score >= best_unshared_score:
            shared_best_wins += 1
        else:
            unshared_best_wins += 1
            unshared_win_task_ids.append(row["original_task_id"])

    return {
        "path_count": len(paths),
        "shared_path_count": shared_path_count,
        "shared_path_fraction": round(shared_path_count / len(paths), 4),
        "top1_shared_count": top1_shared,
        "top1_shared_fraction": round(top1_shared / len(rows), 4),
        "top1_unshared_count": top1_unshared,
        "top1_unshared_fraction": round(top1_unshared / len(rows), 4),
        "top5_shared_majority_count": top5_shared_majority,
        "top5_shared_majority_fraction": round(top5_shared_majority / len(rows), 4),
        "mean_top5_shared_fraction": round(mean(top5_shared_fraction_values), 4),
        "mean_best_shared_gap_from_overall_best": round(mean(shared_best_gap_values), 6),
        "shared_best_wins": shared_best_wins,
        "unshared_best_wins": unshared_best_wins,
        "unshared_win_task_ids": unshared_win_task_ids,
    }


def analyze() -> dict[str, Any]:
    rows = load_rows()
    family_spec, agent_map = TreeFamilyGenerator().build_family("shared_basin_strong", seed=0)

    cartesian_paths = enumerate_paths(family_spec, topology_aware=False)
    topology_paths = enumerate_paths(family_spec, topology_aware=True)

    cartesian_summary = build_match_summary(
        rows,
        family_spec,
        agent_map,
        topology_aware=False,
    )
    topology_summary = build_match_summary(
        rows,
        family_spec,
        agent_map,
        topology_aware=True,
    )

    cartesian_safe = build_safe_prefix_stats(cartesian_paths, agent_map)
    topology_safe = build_safe_prefix_stats(topology_paths, agent_map)

    return {
        "dataset_path": str(DATASET_PATH),
        "family_kind": "shared_basin_strong",
        "seed": 0,
        "task_count": len(rows),
        "cartesian_reference": {
            **cartesian_summary,
            **cartesian_safe,
        },
        "topology_aware": {
            **topology_summary,
            **topology_safe,
        },
    }


def main() -> None:
    summary = analyze()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
