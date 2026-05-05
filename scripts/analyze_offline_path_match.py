from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_ROOT = REPO_ROOT / "envs"
BASELINES_ROOT = REPO_ROOT / "baselines"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))
if str(BASELINES_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINES_ROOT))

from adapters.telecom_mms_adapter import TelecomMMSTaskAdapter  # noqa: E402
from fixed_tree_env import (  # noqa: E402
    compute_first_private_barrier_depth,
    leaf_starts_shared_upload,
)
from oracle_eval import enumerate_family_paths  # noqa: E402
from tree_family.generator import TreeFamilyGenerator  # noqa: E402
from tree_family.specs import CAPABILITY_NAMES, FamilySpec  # noqa: E402


STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
HYBRID_PATH_CLASSES = {
    "hybrid_trap_to_target",
    "hybrid_general_to_target",
    "hybrid_with_barrier",
}
BUCKET_LABELS = ("trap_favoring", "target_favoring", "specialist", "other")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Static offline path_match ranking over all legal family paths. "
            "No LLM, no orchestration, no benchmark execution."
        )
    )
    parser.add_argument("--data", required=True, help="Path to tasks.json")
    parser.add_argument("--family-kind", required=True, help="Family kind to analyze")
    parser.add_argument(
        "--task-ids",
        nargs="+",
        required=True,
        help="One or more original_task_id values to analyze",
    )
    parser.add_argument("--seed", type=int, default=0, help="Family seed")
    parser.add_argument("--top-k", type=int, default=20, help="Top-k rows to export to CSV")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument(
        "--bucket-file",
        help="Optional schedule bucket json used for task-bucket summaries",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_bucket_membership(path: Path | None) -> dict[str, set[str]]:
    if path is None:
        return {
            "trap_favoring": set(),
            "target_favoring": set(),
            "specialist": set(),
        }

    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "trap_favoring": {str(task_id) for task_id in payload.get("trap_favoring_task_ids", [])},
        "target_favoring": {str(task_id) for task_id in payload.get("target_favoring_task_ids", [])},
        "specialist": {str(task_id) for task_id in payload.get("specialist_task_ids", [])},
    }


def bucket_label_for_task(task_id: str, bucket_membership: dict[str, set[str]]) -> str:
    if task_id in bucket_membership.get("specialist", set()):
        return "specialist"
    if task_id in bucket_membership.get("trap_favoring", set()):
        return "trap_favoring"
    if task_id in bucket_membership.get("target_favoring", set()):
        return "target_favoring"
    return "other"


def bucket_flags_for_task(task_id: str, bucket_membership: dict[str, set[str]]) -> dict[str, bool]:
    return {
        "is_trap_favoring": task_id in bucket_membership.get("trap_favoring", set()),
        "is_target_favoring": task_id in bucket_membership.get("target_favoring", set()),
        "is_specialist": task_id in bucket_membership.get("specialist", set()),
    }


def index_rows_by_task_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("original_task_id", row.get("instance_id", "unknown")))
        indexed[task_id] = row
    return indexed


def build_stage_requirements(
    row: dict[str, Any],
    adapter: TelecomMMSTaskAdapter,
) -> dict[str, dict[str, float]]:
    descriptor = adapter.build_task_descriptor(row)
    stage_requirements = descriptor.stage_capability_requirements or {
        stage_name: dict(descriptor.attribute_weights)
        for stage_name in STAGES
    }
    return {
        stage_name: {
            capability_name: float(stage_requirements.get(stage_name, {}).get(capability_name, 0.0))
            for capability_name in CAPABILITY_NAMES
        }
        for stage_name in STAGES
    }


def stage_match(requirement: dict[str, float], skill: dict[str, float]) -> float:
    normalized_requirement = {
        capability_name: float(requirement.get(capability_name, 0.0))
        for capability_name in CAPABILITY_NAMES
    }
    denom = sum(normalized_requirement.values())
    if denom <= 0.0:
        return 0.0
    numer = sum(
        normalized_requirement[capability_name] * float(skill.get(capability_name, 0.0))
        for capability_name in CAPABILITY_NAMES
    )
    return numer / denom


def path_base_alias(agent_id: str) -> str:
    return str(agent_id).split("__from__", 1)[0]


def leaf_type_for_path(path: tuple[str, ...], agent_map: dict[str, Any]) -> str:
    return "shared" if leaf_starts_shared_upload(path, agent_map) else "unshared"


def lane_kind(route_label: str, node_semantic: str) -> str:
    tokens = f"{route_label} {node_semantic}".lower()
    if "trap" in tokens:
        return "trap"
    if "target" in tokens or node_semantic in {"target_specialist", "edge_specialist"}:
        return "target_specialist"
    if (
        "general" in tokens
        or node_semantic in {"general_shared", "safe_core", "mixed_shared"}
        or route_label.startswith(("public_", "general_", "mixed_"))
    ):
        return "general"
    if "barrier" in tokens or "private" in tokens:
        return "barrier_private"
    return "other"


def classify_path_purity(
    lane_sequence: list[str],
    route_labels: list[str],
    node_semantics: list[str],
) -> str:
    del route_labels, node_semantics

    trap_count = sum(1 for lane in lane_sequence if lane == "trap")
    target_count = sum(1 for lane in lane_sequence if lane == "target_specialist")
    general_count = sum(1 for lane in lane_sequence if lane == "general")
    barrier_count = sum(1 for lane in lane_sequence if lane == "barrier_private")
    early_trap_count = sum(1 for lane in lane_sequence[:3] if lane == "trap")
    late_target_count = sum(1 for lane in lane_sequence[2:] if lane == "target_specialist")
    early_general_count = sum(1 for lane in lane_sequence[:3] if lane == "general")
    first_target_idx = next(
        (idx for idx, lane in enumerate(lane_sequence) if lane == "target_specialist"),
        None,
    )

    # Pure trap: trap-dominant across the path, with no target-specialist or barrier handoff.
    if target_count == 0 and barrier_count == 0 and trap_count >= 3 and early_trap_count >= 2:
        return "pure_trap"

    # Pure target: later stages are target-specialist-heavy without an early trap detour or barrier.
    if trap_count == 0 and barrier_count == 0 and target_count >= 3 and late_target_count >= 2:
        return "pure_target"

    # Pure general: the path stays on the shared general lane essentially end-to-end.
    if trap_count == 0 and target_count == 0 and barrier_count == 0 and general_count >= 4:
        return "pure_general"

    # Hybrid trap->target: early trap routing, then a later target-specialist handoff.
    if early_trap_count >= 2 and first_target_idx is not None and first_target_idx >= 2:
        return "hybrid_trap_to_target"

    # Hybrid general->target: starts on general/shared routing before later specializing.
    if trap_count == 0 and early_general_count >= 1 and first_target_idx is not None and target_count >= 1:
        return "hybrid_general_to_target"

    # Hybrid with barrier: any mixed lane family where a private barrier is part of the route.
    active_nonbarrier_lanes = {
        lane for lane in lane_sequence if lane not in {"barrier_private", "other"}
    }
    if barrier_count >= 1 and active_nonbarrier_lanes:
        return "hybrid_with_barrier"

    return "other"


def summarize_path_route(
    base_aliases: list[str],
    route_labels: list[str],
    node_semantics: list[str],
) -> str:
    parts: list[str] = []
    for stage_name, base_alias, route_label, node_semantic in zip(
        STAGES,
        base_aliases,
        route_labels,
        node_semantics,
    ):
        parts.append(f"{stage_name}:{base_alias}|{route_label}|{node_semantic}")
    return " -> ".join(parts)


def build_path_record(
    task_id: str,
    stage_requirements: dict[str, dict[str, float]],
    path: tuple[str, ...],
    rank: int,
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    stage_scores: dict[str, float] = {}
    family_agents = [agent_map[agent_id] for agent_id in path]
    for stage_name, agent in zip(STAGES, family_agents):
        stage_scores[stage_name] = stage_match(stage_requirements[stage_name], {})

    base_aliases = [path_base_alias(agent_id) for agent_id in path]
    route_labels = [str(getattr(agent, "route_label", "")) for agent in family_agents]
    node_semantics = [str(getattr(agent, "node_semantic", "")) for agent in family_agents]
    lane_sequence = [
        lane_kind(route_label, node_semantic)
        for route_label, node_semantic in zip(route_labels, node_semantics)
    ]
    path_class = classify_path_purity(lane_sequence, route_labels, node_semantics)
    first_private_barrier_depth = compute_first_private_barrier_depth(path, agent_map)
    path_match_score = mean(stage_scores.values())
    leaf_type = leaf_type_for_path(path, agent_map)

    return {
        "rank": rank,
        "task_id": task_id,
        "path_match": round(path_match_score, 6),
        "stage1_match": round(stage_scores["stage1"], 6),
        "stage2_match": round(stage_scores["stage2"], 6),
        "stage3_match": round(stage_scores["stage3"], 6),
        "stage4_match": round(stage_scores["stage4"], 6),
        "stage5_match": round(stage_scores["stage5"], 6),
        "path_agent_ids": list(path),
        "path_base_aliases": base_aliases,
        "path_route_labels": route_labels,
        "path_node_semantics": node_semantics,
        "path_lane_sequence": lane_sequence,
        "path_class": path_class,
        "path_route_summary": summarize_path_route(base_aliases, route_labels, node_semantics),
        "path_base_cost_sum": round(sum(float(agent.base_cost) for agent in family_agents), 6),
        "first_private_barrier_depth": first_private_barrier_depth,
        "first_private_barrier_stage": (
            STAGES[first_private_barrier_depth - 1]
            if first_private_barrier_depth is not None
            else None
        ),
        "leaf_type": leaf_type,
        "shared_vs_unshared": leaf_type,
    }


def sort_path_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sorted_records = sorted(
        records,
        key=lambda row: (
            -float(row["path_match"]),
            -float(row["stage5_match"]),
            -float(row["stage4_match"]),
            -float(row["stage3_match"]),
            -float(row["stage2_match"]),
            -float(row["stage1_match"]),
            tuple(row["path_agent_ids"]),
        ),
    )
    for rank, row in enumerate(sorted_records, start=1):
        row["rank"] = rank
    return sorted_records


def validate_paths(
    family_spec: FamilySpec,
    agent_map: dict[str, Any],
    paths: list[tuple[str, ...]],
) -> dict[str, Any]:
    stage_count = len(family_spec.stages)
    unique_paths = set(paths)
    if len(unique_paths) != len(paths):
        raise ValueError("Enumerated paths are not unique.")

    for path in paths:
        if len(path) != stage_count:
            raise ValueError(f"Incomplete path found: {path!r}")
        prefix: tuple[str, ...] = ()
        for depth, agent_id in enumerate(path):
            stage_name = family_spec.stages[depth]
            if agent_id not in family_spec.stage_agents[stage_name]:
                raise ValueError(
                    f"Agent {agent_id!r} in path {path!r} is not registered for stage {stage_name}."
                )
            allowed_children = (
                family_spec.allowed_children.get(prefix)
                if family_spec.allowed_children
                else None
            )
            candidate_children = (
                allowed_children
                if allowed_children is not None
                else family_spec.stage_agents[stage_name]
            )
            if agent_id not in candidate_children:
                raise ValueError(
                    f"Illegal transition at prefix {prefix!r}: child {agent_id!r} is not allowed."
                )
            prefix = prefix + (agent_id,)

    return {
        "path_count": len(paths),
        "stage_count": stage_count,
        "unique_path_count": len(unique_paths),
        "family_validation": "passed",
        "agent_count": len(agent_map),
    }


def round_or_none(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def mean_or_none(values: Iterable[float | None]) -> float | None:
    numeric = [float(value) for value in values if value is not None]
    if not numeric:
        return None
    return mean(numeric)


def top_capabilities(values: dict[str, float], limit: int = 3) -> list[list[Any]]:
    ranked = sorted(values.items(), key=lambda item: (-float(item[1]), item[0]))
    return [[name, round(float(score), 6)] for name, score in ranked[:limit] if float(score) > 0.0]


def build_stage_agent_summary(
    stage_requirements: dict[str, dict[str, float]],
    family_spec: FamilySpec,
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for stage_name in STAGES:
        scored_agents: list[dict[str, Any]] = []
        for agent_id in family_spec.stage_agents[stage_name]:
            agent = agent_map[agent_id]
            score = stage_match(stage_requirements[stage_name], {})
            scored_agents.append(
                {
                    "agent_id": agent_id,
                    "match": round(score, 6),
                    "base_alias": path_base_alias(agent_id),
                    "route_label": str(getattr(agent, "route_label", "")),
                    "node_semantic": str(getattr(agent, "node_semantic", "")),
                    "base_cost": round(float(agent.base_cost), 6),
                    "top_skills": [],
                }
            )
        scored_agents.sort(key=lambda row: (-float(row["match"]), row["agent_id"]))
        summary[stage_name] = {
            "requirement_top_caps": top_capabilities(stage_requirements[stage_name]),
            "best_match_agent": scored_agents[0],
            "worst_match_agent": sorted(
                scored_agents,
                key=lambda row: (float(row["match"]), row["agent_id"]),
            )[0],
        }
    return summary


def counter_to_dict(counter: Counter[str]) -> dict[str, int]:
    return {key: int(counter[key]) for key in sorted(counter)}


def build_lane_summary(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    path_count = 0
    stage_counts: Counter[str] = Counter()
    path_presence_counts: Counter[str] = Counter()
    path_class_counts: Counter[str] = Counter()
    for record in records:
        path_count += 1
        sequence = [str(item) for item in record.get("path_lane_sequence", [])]
        local_presence = set(sequence)
        stage_counts.update(sequence)
        path_class_counts[str(record.get("path_class", "other"))] += 1
        for lane in local_presence:
            path_presence_counts[lane] += 1
    return {
        "path_count": path_count,
        "stage_counts": counter_to_dict(stage_counts),
        "path_presence_counts": counter_to_dict(path_presence_counts),
        "path_class_counts": counter_to_dict(path_class_counts),
    }


def serialize_representative_path(path_row: dict[str, Any] | None) -> dict[str, Any] | None:
    if path_row is None:
        return None
    return {
        "rank": int(path_row["rank"]),
        "path_match": round(float(path_row["path_match"]), 6),
        "path_class": str(path_row["path_class"]),
        "path_lane_sequence": list(path_row["path_lane_sequence"]),
        "path_route_summary": str(path_row["path_route_summary"]),
        "leaf_type": str(path_row["leaf_type"]),
        "first_private_barrier_depth": path_row["first_private_barrier_depth"],
    }


def first_record_by_class(
    rankings: list[dict[str, Any]],
    valid_classes: set[str],
) -> dict[str, Any] | None:
    for row in rankings:
        if str(row.get("path_class")) in valid_classes:
            return row
    return None


def mean_gap_from_top1(rankings: list[dict[str, Any]], end_rank: int) -> float | None:
    if len(rankings) < 2:
        return None
    competitor_rows = rankings[1 : min(end_rank, len(rankings))]
    if not competitor_rows:
        return None
    top1_score = float(rankings[0]["path_match"])
    return top1_score - mean(float(row["path_match"]) for row in competitor_rows)


def build_margin_summary(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    top1_score = float(rankings[0]["path_match"])
    best_pure_trap = first_record_by_class(rankings, {"pure_trap"})
    best_pure_target = first_record_by_class(rankings, {"pure_target"})
    best_pure_general = first_record_by_class(rankings, {"pure_general"})

    return {
        "top1_minus_top2": round_or_none(
            top1_score - float(rankings[1]["path_match"]) if len(rankings) >= 2 else None
        ),
        # Competitive margin against the next-best band, excluding rank 1 itself.
        "top1_minus_top5_mean": round_or_none(mean_gap_from_top1(rankings, 5)),
        "top1_minus_top10_mean": round_or_none(mean_gap_from_top1(rankings, 10)),
        "top1_minus_best_pure_trap": round_or_none(
            top1_score - float(best_pure_trap["path_match"]) if best_pure_trap else None
        ),
        "top1_minus_best_pure_target": round_or_none(
            top1_score - float(best_pure_target["path_match"]) if best_pure_target else None
        ),
        "top1_minus_best_pure_general": round_or_none(
            top1_score - float(best_pure_general["path_match"]) if best_pure_general else None
        ),
        "best_pure_trap_rank": int(best_pure_trap["rank"]) if best_pure_trap else None,
        "best_pure_target_rank": int(best_pure_target["rank"]) if best_pure_target else None,
        "best_pure_general_rank": int(best_pure_general["rank"]) if best_pure_general else None,
    }


def build_best_of_class_summary(rankings: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "best_overall_path": serialize_representative_path(rankings[0]),
        "best_pure_trap_path": serialize_representative_path(
            first_record_by_class(rankings, {"pure_trap"})
        ),
        "best_pure_target_path": serialize_representative_path(
            first_record_by_class(rankings, {"pure_target"})
        ),
        "best_pure_general_path": serialize_representative_path(
            first_record_by_class(rankings, {"pure_general"})
        ),
        "best_hybrid_path": serialize_representative_path(
            first_record_by_class(rankings, HYBRID_PATH_CLASSES)
        ),
    }


def build_task_summary(
    rankings: list[dict[str, Any]],
    stage_requirements: dict[str, dict[str, float]],
    family_spec: FamilySpec,
    agent_map: dict[str, Any],
    top_k: int,
    bucket_label: str,
    bucket_flags: dict[str, bool],
) -> dict[str, Any]:
    top1 = rankings[0]
    top5 = rankings[: min(5, len(rankings))]
    topk = rankings[: min(top_k, len(rankings))]
    top5_scores = [float(row["path_match"]) for row in top5]
    topk_leaf_counts = Counter(str(row["leaf_type"]) for row in topk)
    path_class_counts = Counter(str(row["path_class"]) for row in rankings)
    margin_summary = build_margin_summary(rankings)
    best_of_class_summary = build_best_of_class_summary(rankings)

    return {
        "bucket_label": bucket_label,
        "bucket_flags": bucket_flags,
        "path_count": len(rankings),
        "top1_path_match": round(float(top1["path_match"]), 6),
        "top1_path_class": str(top1["path_class"]),
        "top1_is_hybrid": str(top1["path_class"]) in HYBRID_PATH_CLASSES,
        "path_class_distribution": counter_to_dict(path_class_counts),
        "top5_path_match_range": {
            "max": round(max(top5_scores), 6),
            "min": round(min(top5_scores), 6),
        },
        "top1_path": {
            "rank": int(top1["rank"]),
            "path_match": round(float(top1["path_match"]), 6),
            "path_class": str(top1["path_class"]),
            "path_agent_ids": list(top1["path_agent_ids"]),
            "path_base_aliases": list(top1["path_base_aliases"]),
            "path_route_labels": list(top1["path_route_labels"]),
            "path_node_semantics": list(top1["path_node_semantics"]),
            "path_lane_sequence": list(top1["path_lane_sequence"]),
            "path_route_summary": str(top1["path_route_summary"]),
            "leaf_type": str(top1["leaf_type"]),
            "first_private_barrier_depth": top1["first_private_barrier_depth"],
        },
        "top1_lane_distribution": build_lane_summary([top1]),
        "top5_lane_distribution": build_lane_summary(top5),
        "topk_shared_vs_unshared_distribution": counter_to_dict(topk_leaf_counts),
        "stage_agent_match_summary": build_stage_agent_summary(
            stage_requirements,
            family_spec,
            agent_map,
        ),
        **margin_summary,
        "margin_summary": margin_summary,
        "best_of_class_summary": best_of_class_summary,
        **best_of_class_summary,
    }


def build_bucket_comparison_summary(
    task_results: list[dict[str, Any]],
    bucket_membership: dict[str, set[str]],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for task_result in task_results:
        grouped[str(task_result["bucket_label"])].append(task_result)

    buckets: dict[str, Any] = {}
    for bucket_label in BUCKET_LABELS:
        rows = grouped.get(bucket_label, [])
        top1_path_class_counter = Counter(
            str(row["summary"]["top1_path_class"]) for row in rows
        )
        buckets[bucket_label] = {
            "task_count": len(rows),
            "task_ids": [str(row["task_id"]) for row in rows],
            "mean_top1_path_match": round_or_none(
                mean_or_none(float(row["summary"]["top1_path_match"]) for row in rows)
            ),
            "mean_top1_minus_top2": round_or_none(
                mean_or_none(row["summary"].get("top1_minus_top2") for row in rows)
            ),
            "top1_path_class_distribution": counter_to_dict(top1_path_class_counter),
            "top1_path_class_by_task": {
                str(row["task_id"]): str(row["summary"]["top1_path_class"]) for row in rows
            },
            "hybrid_top1_count": sum(1 for row in rows if bool(row["summary"]["top1_is_hybrid"])),
            "best_overall_vs_best_pure_target_gap_mean": round_or_none(
                mean_or_none(row["summary"].get("top1_minus_best_pure_target") for row in rows)
            ),
            "best_overall_vs_best_pure_trap_gap_mean": round_or_none(
                mean_or_none(row["summary"].get("top1_minus_best_pure_trap") for row in rows)
            ),
            "best_overall_vs_best_pure_general_gap_mean": round_or_none(
                mean_or_none(row["summary"].get("top1_minus_best_pure_general") for row in rows)
            ),
        }

    return {
        "bucket_selection_source": {
            "trap_favoring_count": len(bucket_membership.get("trap_favoring", set())),
            "target_favoring_count": len(bucket_membership.get("target_favoring", set())),
            "specialist_count": len(bucket_membership.get("specialist", set())),
        },
        "buckets": buckets,
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_topk_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "task_id",
        "path_match",
        "stage1_match",
        "stage2_match",
        "stage3_match",
        "stage4_match",
        "stage5_match",
        "path_class",
        "path_agent_ids",
        "path_base_aliases",
        "path_route_labels",
        "path_node_semantics",
        "path_lane_sequence",
        "path_base_cost_sum",
        "first_private_barrier_depth",
        "first_private_barrier_stage",
        "leaf_type",
        "shared_vs_unshared",
        "path_route_summary",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{name: row.get(name) for name in fieldnames},
                    "path_agent_ids": json.dumps(row["path_agent_ids"], ensure_ascii=False),
                    "path_base_aliases": json.dumps(row["path_base_aliases"], ensure_ascii=False),
                    "path_route_labels": json.dumps(row["path_route_labels"], ensure_ascii=False),
                    "path_node_semantics": json.dumps(row["path_node_semantics"], ensure_ascii=False),
                    "path_lane_sequence": json.dumps(row["path_lane_sequence"], ensure_ascii=False),
                }
            )


def analyze_task(
    row: dict[str, Any],
    *,
    family_spec: FamilySpec,
    agent_map: dict[str, Any],
    paths: list[tuple[str, ...]],
    adapter: TelecomMMSTaskAdapter,
    top_k: int,
    bucket_membership: dict[str, set[str]],
) -> dict[str, Any]:
    task_id = str(row.get("original_task_id", row.get("instance_id", "unknown")))
    stage_requirements = build_stage_requirements(row, adapter)
    unsorted_records = [
        build_path_record(
            task_id=task_id,
            stage_requirements=stage_requirements,
            path=path,
            rank=0,
            agent_map=agent_map,
        )
        for path in paths
    ]
    rankings = sort_path_records(unsorted_records)
    bucket_label = bucket_label_for_task(task_id, bucket_membership)
    bucket_flags = bucket_flags_for_task(task_id, bucket_membership)
    summary = build_task_summary(
        rankings,
        stage_requirements,
        family_spec,
        agent_map,
        top_k,
        bucket_label,
        bucket_flags,
    )
    return {
        "task_id": task_id,
        "bucket_label": bucket_label,
        "bucket_flags": bucket_flags,
        "stage_capability_requirements": stage_requirements,
        "summary": summary,
        "rankings": rankings,
    }


def main() -> None:
    args = parse_args()
    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bucket_path = Path(args.bucket_file).resolve() if args.bucket_file else None

    rows = load_rows(data_path)
    indexed_rows = index_rows_by_task_id(rows)
    missing_task_ids = [task_id for task_id in args.task_ids if task_id not in indexed_rows]
    if missing_task_ids:
        raise SystemExit(f"Missing task ids in dataset: {missing_task_ids}")

    generator = TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(args.family_kind, seed=args.seed)
    validation_errors = generator.validate_family(family_spec, agent_map)
    if validation_errors:
        raise SystemExit(f"Family validation failed: {validation_errors}")

    paths = enumerate_family_paths(
        stages=list(family_spec.stages),
        stage_agents=family_spec.stage_agents,
        allowed_children=family_spec.allowed_children,
    )
    if not paths:
        raise SystemExit("No legal complete paths were enumerated.")
    path_validation = validate_paths(family_spec, agent_map, paths)

    bucket_membership = load_bucket_membership(bucket_path)
    adapter = TelecomMMSTaskAdapter()
    task_summaries: list[dict[str, Any]] = []
    task_results: list[dict[str, Any]] = []

    for task_id in args.task_ids:
        result = analyze_task(
            indexed_rows[task_id],
            family_spec=family_spec,
            agent_map=agent_map,
            paths=paths,
            adapter=adapter,
            top_k=args.top_k,
            bucket_membership=bucket_membership,
        )
        task_results.append(result)
        task_summaries.append(
            {
                "task_id": result["task_id"],
                "bucket_label": result["bucket_label"],
                "summary": result["summary"],
            }
        )
        task_output_path = output_dir / f"task_{task_id}_path_rankings.json"
        write_json(
            task_output_path,
            {
                "task_id": result["task_id"],
                "bucket_label": result["bucket_label"],
                "bucket_flags": result["bucket_flags"],
                "family_kind": args.family_kind,
                "seed": args.seed,
                "path_count": len(paths),
                "stage_capability_requirements": result["stage_capability_requirements"],
                "summary": result["summary"],
                "margin_summary": result["summary"]["margin_summary"],
                "best_of_class_summary": result["summary"]["best_of_class_summary"],
                "rankings": result["rankings"],
            },
        )
        write_topk_csv(
            output_dir / f"task_{task_id}_topk.csv",
            result["rankings"][: min(args.top_k, len(result["rankings"]))],
        )

    bucket_comparison_summary = build_bucket_comparison_summary(task_results, bucket_membership)
    write_json(output_dir / "bucket_comparison_summary.json", bucket_comparison_summary)

    write_json(
        output_dir / "summary.json",
        {
            "script": str(Path(__file__).resolve()),
            "analysis_mode": "static_offline_path_match",
            "data": str(data_path),
            "bucket_file": str(bucket_path) if bucket_path else None,
            "family_kind": args.family_kind,
            "seed": args.seed,
            "top_k": args.top_k,
            "task_ids": list(args.task_ids),
            "path_validation": path_validation,
            "task_summaries": task_summaries,
            "bucket_comparison_summary_path": str(output_dir / "bucket_comparison_summary.json"),
        },
    )


if __name__ == "__main__":
    main()
