from __future__ import annotations

import json
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

from fixed_tree_env import (  # noqa: E402
    FixedTreeEnvironment,
    compute_first_private_barrier_depth,
    compute_shared_upload_edges,
    leaf_starts_shared_upload,
)
from risky_ps import RiskyPSPolicy  # noqa: E402
from oracle_eval import enumerate_family_paths  # noqa: E402
from tree_family.generator import TreeFamilyGenerator  # noqa: E402
from tree_family.specs import CAPABILITY_NAMES, FamilySpec  # noqa: E402


DATASET_PATH = (
    REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities" / "tasks.json"
)
OUTPUT_PATH = REPO_ROOT / "analysis" / "shared_basin_strong_static_analysis.json"
STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
PrefixKey = tuple[str, ...]
EdgeKey = tuple[PrefixKey, PrefixKey]


def load_rows() -> list[dict[str, Any]]:
    return json.loads(DATASET_PATH.read_text())


def stage_match(requirement: dict[str, float], skill: dict[str, float]) -> float:
    denom = max(1e-9, sum(requirement.values()))
    numer = sum(
        requirement[capability_name] * skill[capability_name]
        for capability_name in CAPABILITY_NAMES
    )
    return numer / denom


def path_match(row: dict[str, Any], path: tuple[str, ...], agent_map: dict[str, Any]) -> float:
    scores = []
    for stage_name, agent_id in zip(STAGES, path):
        requirement = row[stage_name]["capability_requirements"]
        skill = agent_map[agent_id].attribute_skill
        scores.append(stage_match(requirement, skill))
    return mean(scores)


def is_shared_leaf(path: tuple[str, ...], agent_map: dict[str, Any]) -> bool:
    return leaf_starts_shared_upload(path, agent_map)


def build_parent_children(paths: list[tuple[str, ...]]) -> dict[PrefixKey, list[PrefixKey]]:
    parent_children: dict[PrefixKey, set[PrefixKey]] = {}
    for path in paths:
        for depth in range(len(path)):
            prefix = tuple(path[:depth])
            child_prefix = tuple(path[: depth + 1])
            parent_children.setdefault(prefix, set()).add(child_prefix)
    return {
        prefix: sorted(child_prefixes)
        for prefix, child_prefixes in parent_children.items()
    }


def build_upload_summary(
    paths: list[tuple[str, ...]],
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    parent_children = build_parent_children(paths)
    shared_edge_leaf_counts: dict[EdgeKey, int] = {}
    shared_leaf_count = 0
    total_shared_upload_edges = 0

    for path in paths:
        if not is_shared_leaf(path, agent_map):
            continue
        shared_leaf_count += 1
        upload_edges = compute_shared_upload_edges(path, agent_map)
        total_shared_upload_edges += len(upload_edges)
        for edge in upload_edges:
            shared_edge_leaf_counts[edge] = shared_edge_leaf_counts.get(edge, 0) + 1

    upload_reachable_prefix_depth_counts: dict[int, int] = {}
    mixed_parent_depth_counts: dict[int, int] = {}

    for prefix, child_prefixes in parent_children.items():
        exposures = [
            shared_edge_leaf_counts.get((prefix, child_prefix), 0) > 0
            for child_prefix in child_prefixes
        ]
        if any(exposures):
            depth = len(prefix)
            upload_reachable_prefix_depth_counts[depth] = (
                upload_reachable_prefix_depth_counts.get(depth, 0) + 1
            )
        if any(exposures) and not all(exposures):
            depth = len(prefix)
            mixed_parent_depth_counts[depth] = mixed_parent_depth_counts.get(depth, 0) + 1

    mean_shared_upload_edges = 0.0
    if shared_leaf_count:
        mean_shared_upload_edges = total_shared_upload_edges / shared_leaf_count

    return {
        "upload_reachable_prefix_depth_counts": {
            str(depth): count
            for depth, count in sorted(upload_reachable_prefix_depth_counts.items())
        },
        "mixed_parent_depth_counts": {
            str(depth): count
            for depth, count in sorted(mixed_parent_depth_counts.items())
        },
        "mean_shared_upload_edges": round(mean_shared_upload_edges, 4),
    }


def build_match_summary(
    rows: list[dict[str, Any]],
    paths: list[tuple[str, ...]],
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    shared_leaf_count = sum(1 for path in paths if is_shared_leaf(path, agent_map))
    top1_shared = 0
    top1_unshared = 0
    top5_shared_majority = 0
    top5_shared_fraction_values: list[float] = []
    shared_best_wins = 0
    unshared_best_wins = 0
    unshared_win_task_ids: list[str] = []
    first_private_barrier_stage_counts: dict[str, int] = {}
    optimal_stage_node_shared_counts = {stage_name: 0 for stage_name in STAGES}

    for row in rows:
        scored = [(path_match(row, path, agent_map), path) for path in paths]
        scored.sort(key=lambda item: item[0], reverse=True)

        _, top1_path = scored[0]
        top5 = scored[:5]
        top5_shared_count = sum(1 for _, path in top5 if is_shared_leaf(path, agent_map))
        top5_shared_fraction_values.append(top5_shared_count / len(top5))
        if top5_shared_count > (len(top5) // 2):
            top5_shared_majority += 1

        shared_scores = [score for score, path in scored if is_shared_leaf(path, agent_map)]
        unshared_scores = [score for score, path in scored if not is_shared_leaf(path, agent_map)]
        best_shared_score = max(shared_scores) if shared_scores else None
        best_unshared_score = max(unshared_scores) if unshared_scores else None

        if is_shared_leaf(top1_path, agent_map):
            top1_shared += 1
        else:
            top1_unshared += 1

        first_barrier_depth = compute_first_private_barrier_depth(top1_path, agent_map)
        first_barrier_stage = "none"
        if first_barrier_depth is not None:
            first_barrier_stage = STAGES[first_barrier_depth - 1]
        first_private_barrier_stage_counts[first_barrier_stage] = (
            first_private_barrier_stage_counts.get(first_barrier_stage, 0) + 1
        )

        for stage_name, agent_id in zip(STAGES, top1_path):
            if int(getattr(agent_map[agent_id], "g", 1)) == 0:
                optimal_stage_node_shared_counts[stage_name] += 1

        if best_unshared_score is None or (
            best_shared_score is not None and best_shared_score >= best_unshared_score
        ):
            shared_best_wins += 1
        else:
            unshared_best_wins += 1
            unshared_win_task_ids.append(row["original_task_id"])

    task_count = len(rows)
    return {
        "path_count": len(paths),
        "shared_leaf_count": shared_leaf_count,
        "shared_leaf_fraction": round(shared_leaf_count / len(paths), 4),
        "top1_shared_count": top1_shared,
        "top1_shared_fraction": round(top1_shared / task_count, 4),
        "top1_unshared_count": top1_unshared,
        "top1_unshared_fraction": round(top1_unshared / task_count, 4),
        "top5_shared_majority_count": top5_shared_majority,
        "top5_shared_majority_fraction": round(top5_shared_majority / task_count, 4),
        "mean_top5_shared_fraction": round(mean(top5_shared_fraction_values), 4),
        "shared_best_wins": shared_best_wins,
        "unshared_best_wins": unshared_best_wins,
        "first_private_barrier_stage_distribution": {
            stage_name: {
                "count": first_private_barrier_stage_counts.get(stage_name, 0),
                "fraction": round(first_private_barrier_stage_counts.get(stage_name, 0) / task_count, 4),
            }
            for stage_name in ["none", *STAGES]
            if stage_name == "none" or first_private_barrier_stage_counts.get(stage_name, 0) > 0
        },
        **{
            f"optimal_stage_node_shared_rate_{stage_name}": round(
                optimal_stage_node_shared_counts[stage_name] / task_count,
                4,
            )
            for stage_name in STAGES
        },
        "unshared_win_task_ids": unshared_win_task_ids,
    }


def build_risky_prefix_summary() -> dict[str, Any]:
    env = FixedTreeEnvironment(agent_catalog=[], family_kind="shared_basin_strong", family_seed=0)
    policy = RiskyPSPolicy()
    policy.bind_env(env)

    safe_depth_counts: dict[int, int] = {}
    mixed_depth_counts: dict[int, int] = {}

    for prefix, is_safe in policy.safe_prefixes.items():
        depth = len(prefix)
        if is_safe:
            safe_depth_counts[depth] = safe_depth_counts.get(depth, 0) + 1
        if policy.mixed_prefixes.get(prefix, False):
            mixed_depth_counts[depth] = mixed_depth_counts.get(depth, 0) + 1

    return {
        "safe_depth_counts": {
            str(depth): count
            for depth, count in sorted(safe_depth_counts.items())
        },
        "mixed_depth_counts": {
            str(depth): count
            for depth, count in sorted(mixed_depth_counts.items())
        },
    }


def analyze() -> dict[str, Any]:
    rows = load_rows()
    family_spec, agent_map = TreeFamilyGenerator().build_family("shared_basin_strong", seed=0)
    paths = enumerate_family_paths(
        stages=list(family_spec.stages),
        stage_agents=family_spec.stage_agents,
        allowed_children=family_spec.allowed_children,
    )

    match_summary = build_match_summary(rows, paths, agent_map)
    upload_summary = build_upload_summary(paths, agent_map)
    risky_prefix_summary = build_risky_prefix_summary()

    return {
        "dataset_path": str(DATASET_PATH),
        "family_kind": "shared_basin_strong",
        "seed": 0,
        "task_count": len(rows),
        **{key: value for key, value in match_summary.items() if key != "unshared_win_task_ids"},
        **upload_summary,
        **risky_prefix_summary,
        "unshared_win_task_ids": list(match_summary["unshared_win_task_ids"]),
    }


def main() -> None:
    summary = analyze()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
