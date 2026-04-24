from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT,
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
    ROOT / "baselines",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from adapters.telecom_mms_adapter import TelecomMMSTaskAdapter  # noqa: E402
from fixed_tree_env import (  # noqa: E402
    FixedTreeEnvironment,
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
DEFAULT_BUCKET_FILE = (
    ROOT / "analysis" / "shared_basin_prefix_dedup_profile_switch_schedule_buckets.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a small fixed-path llm_bench diagnostic sweep over representative paths."
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--family-kind", required=True)
    parser.add_argument("--task-ids", nargs="*")
    parser.add_argument("--path-mode", choices=["representative", "offline_topk"], default="representative")
    parser.add_argument("--top-k-offline", type=int, default=20)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--max-paths-per-task", type=int, default=5)
    parser.add_argument("--bucket-file", type=Path, default=DEFAULT_BUCKET_FILE)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("Dataset must be a JSON list.")
    return payload


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def index_rows_by_task_id(rows: Iterable[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for row in rows:
        task_id = str(row.get("original_task_id", row.get("instance_id", "unknown")))
        indexed[task_id] = row
    return indexed


def load_bucket_membership(path: Path) -> dict[str, list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "trap_favoring": sorted(str(task_id) for task_id in payload.get("trap_favoring_task_ids", [])),
        "target_favoring": sorted(str(task_id) for task_id in payload.get("target_favoring_task_ids", [])),
        "specialist": sorted(str(task_id) for task_id in payload.get("specialist_task_ids", [])),
    }


def bucket_label_for_task(task_id: str, membership: dict[str, list[str]]) -> str:
    if task_id in set(membership.get("specialist", [])):
        return "specialist"
    if task_id in set(membership.get("trap_favoring", [])):
        return "trap_favoring"
    if task_id in set(membership.get("target_favoring", [])):
        return "target_favoring"
    return "other"


def select_demo_task_ids(membership: dict[str, list[str]]) -> list[str]:
    specialist = membership.get("specialist", [])
    specialist_set = set(specialist)
    traps = membership.get("trap_favoring", [])[:2]
    targets = [task_id for task_id in membership.get("target_favoring", []) if task_id not in specialist_set]
    chosen = traps + targets[:1] + specialist[:1]
    deduped: list[str] = []
    seen: set[str] = set()
    for task_id in chosen:
        if task_id not in seen:
            deduped.append(task_id)
            seen.add(task_id)
    return deduped


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
    denom = sum(float(requirement.get(capability_name, 0.0)) for capability_name in CAPABILITY_NAMES)
    if denom <= 0.0:
        return 0.0
    numer = sum(
        float(requirement.get(capability_name, 0.0)) * float(skill.get(capability_name, 0.0))
        for capability_name in CAPABILITY_NAMES
    )
    return numer / denom


def path_base_alias(agent_id: str) -> str:
    return str(agent_id).split("__from__", 1)[0]


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


def classify_path_purity(lane_sequence: list[str]) -> str:
    trap_count = sum(1 for lane in lane_sequence if lane == "trap")
    target_count = sum(1 for lane in lane_sequence if lane == "target_specialist")
    general_count = sum(1 for lane in lane_sequence if lane == "general")
    barrier_count = sum(1 for lane in lane_sequence if lane == "barrier_private")
    early_trap_count = sum(1 for lane in lane_sequence[:3] if lane == "trap")
    early_general_count = sum(1 for lane in lane_sequence[:3] if lane == "general")
    late_target_count = sum(1 for lane in lane_sequence[2:] if lane == "target_specialist")
    first_target_idx = next(
        (idx for idx, lane in enumerate(lane_sequence) if lane == "target_specialist"),
        None,
    )
    if target_count == 0 and barrier_count == 0 and trap_count >= 3 and early_trap_count >= 2:
        return "pure_trap"
    if trap_count == 0 and barrier_count == 0 and target_count >= 3 and late_target_count >= 2:
        return "pure_target"
    if trap_count == 0 and target_count == 0 and barrier_count == 0 and general_count >= 4:
        return "pure_general"
    if early_trap_count >= 2 and first_target_idx is not None and first_target_idx >= 2:
        return "hybrid_trap_to_target"
    if trap_count == 0 and early_general_count >= 1 and first_target_idx is not None and target_count >= 1:
        return "hybrid_general_to_target"
    active_nonbarrier_lanes = {
        lane for lane in lane_sequence if lane not in {"barrier_private", "other"}
    }
    if barrier_count >= 1 and active_nonbarrier_lanes:
        return "hybrid_with_barrier"
    return "other"


def summarize_path_route(base_aliases: list[str], route_labels: list[str], node_semantics: list[str]) -> str:
    return " -> ".join(
        f"{stage_name}:{base_alias}|{route_label}|{node_semantic}"
        for stage_name, base_alias, route_label, node_semantic in zip(
            STAGES, base_aliases, route_labels, node_semantics
        )
    )


def build_offline_path_record(
    *,
    task_id: str,
    path: tuple[str, ...],
    stage_requirements: dict[str, dict[str, float]],
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    family_agents = [agent_map[agent_id] for agent_id in path]
    stage_scores = {
        stage_name: stage_match(stage_requirements[stage_name], agent.attribute_skill)
        for stage_name, agent in zip(STAGES, family_agents)
    }
    base_aliases = [path_base_alias(agent_id) for agent_id in path]
    route_labels = [str(getattr(agent, "route_label", "")) for agent in family_agents]
    node_semantics = [str(getattr(agent, "node_semantic", "")) for agent in family_agents]
    lane_sequence = [
        lane_kind(route_label, node_semantic)
        for route_label, node_semantic in zip(route_labels, node_semantics)
    ]
    return {
        "task_id": task_id,
        "path_agent_ids": list(path),
        "path_match": round(statistics.fmean(stage_scores.values()), 6),
        "stage_matches": {stage_name: round(score, 6) for stage_name, score in stage_scores.items()},
        "path_class": classify_path_purity(lane_sequence),
        "path_lane_sequence": lane_sequence,
        "path_base_aliases": base_aliases,
        "path_route_labels": route_labels,
        "path_node_semantics": node_semantics,
        "path_route_summary": summarize_path_route(base_aliases, route_labels, node_semantics),
        "path_base_cost_sum": round(sum(float(agent.base_cost) for agent in family_agents), 6),
        "first_private_barrier_depth": compute_first_private_barrier_depth(path, agent_map),
        "leaf_type": "shared" if leaf_starts_shared_upload(path, agent_map) else "unshared",
    }


def sort_offline_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = sorted(
        records,
        key=lambda row: (
            -float(row["path_match"]),
            tuple(row["path_agent_ids"]),
        ),
    )
    for idx, row in enumerate(rows, start=1):
        row["rank"] = idx
    return rows


def first_record_by_class(rankings: list[dict[str, Any]], valid_classes: set[str]) -> dict[str, Any] | None:
    for row in rankings:
        if str(row.get("path_class")) in valid_classes:
            return row
    return None


def dedupe_rankings(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    seen: set[tuple[str, ...]] = set()
    for row in rows:
        key = tuple(str(agent_id) for agent_id in row["path_agent_ids"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def select_representative_paths(
    rankings: list[dict[str, Any]],
    *,
    path_mode: str,
    top_k_offline: int,
    max_paths_per_task: int,
) -> list[dict[str, Any]]:
    if path_mode == "offline_topk":
        return rankings[: max_paths_per_task]

    candidates: list[dict[str, Any]] = [
        rankings[0],
        first_record_by_class(rankings, {"pure_trap"}),
        first_record_by_class(rankings, {"pure_target"}),
        first_record_by_class(rankings, {"pure_general"}),
        first_record_by_class(rankings, HYBRID_PATH_CLASSES),
    ]
    deduped = dedupe_rankings([row for row in candidates if row is not None])
    if len(deduped) < max_paths_per_task:
        deduped = dedupe_rankings(deduped + rankings[:top_k_offline])
    return deduped[: max_paths_per_task]


def flatten_stage_resource_summary(stage_trace: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for stage_name in STAGES:
        row = stage_trace.get(stage_name, {})
        summary.append(
            {
                "stage_name": stage_name,
                "llm_call_count_stage": int(row.get("llm_call_count_stage", 0) or 0),
                "prompt_tokens_total_stage": float(row.get("prompt_tokens_total_stage", 0.0) or 0.0),
                "completion_tokens_total_stage": float(
                    row.get("completion_tokens_total_stage", 0.0) or 0.0
                ),
                "tool_call_count_stage": len(row.get("executed_tool_calls", []) or []),
            }
        )
    return summary


def flatten_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": record["task_id"],
        "bucket_label": record["bucket_label"],
        "path_rank_offline": record["path_rank_offline"],
        "path_class": record["path_class"],
        "path_lane_sequence": json.dumps(record["path_lane_sequence"], ensure_ascii=False),
        "path_agent_ids": json.dumps(record["path_agent_ids"], ensure_ascii=False),
        "offline_path_match": record["offline_path_match"],
        "raw_terminal_penalty": record["raw_terminal_penalty"],
        "raw_path_cost_component": record["raw_path_cost_component"],
        "raw_reasoning_cost_component": record["raw_reasoning_cost_component"],
        "raw_total_cost": record["raw_total_cost"],
        "prompt_tokens_total": record["prompt_tokens_total"],
        "completion_tokens_total": record["completion_tokens_total"],
        "api_cost_total_usd_raw": record["api_cost_total_usd_raw"],
        "tool_call_count": record["tool_call_count"],
        "final_action": record["final_action"],
        "selected_blocker_ids": json.dumps(record["selected_blocker_ids"], ensure_ascii=False),
        "deferred_blocker_ids": json.dumps(record["deferred_blocker_ids"], ensure_ascii=False),
        "exact_match": record["exact_match"],
    }


def range_summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": round(min(values), 6),
        "max": round(max(values), 6),
        "spread": round(max(values) - min(values), 6),
    }


def pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(ys) < 2 or len(xs) != len(ys):
        return None
    x_mean = statistics.fmean(xs)
    y_mean = statistics.fmean(ys)
    x_var = sum((x - x_mean) ** 2 for x in xs)
    y_var = sum((y - y_mean) ** 2 for y in ys)
    if x_var <= 0.0 or y_var <= 0.0:
        return None
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    return round(cov / ((x_var ** 0.5) * (y_var ** 0.5)), 6)


def build_task_summary(task_id: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    terminal_best = min(rows, key=lambda row: (float(row["raw_terminal_penalty"]), float(row["raw_total_cost"])))
    token_top = max(rows, key=lambda row: (float(row["prompt_tokens_total"] + row["completion_tokens_total"]), row["path_rank_offline"]))
    offline_top = max(rows, key=lambda row: (float(row["offline_path_match"]), -int(row["path_rank_offline"])))
    return {
        "task_id": task_id,
        "bucket_label": rows[0]["bucket_label"] if rows else "other",
        "path_count": len(rows),
        "raw_terminal_penalty_range": range_summary([float(row["raw_terminal_penalty"]) for row in rows]),
        "raw_reasoning_cost_component_range": range_summary(
            [float(row["raw_reasoning_cost_component"]) for row in rows]
        ),
        "prompt_tokens_total_range": range_summary([float(row["prompt_tokens_total"]) for row in rows]),
        "completion_tokens_total_range": range_summary(
            [float(row["completion_tokens_total"]) for row in rows]
        ),
        "tool_call_count_range": range_summary([float(row["tool_call_count"]) for row in rows]),
        "unique_final_actions": sorted({str(row["final_action"]) for row in rows}),
        "unique_final_action_count": len({str(row["final_action"]) for row in rows}),
        "top_terminal_path": {
            "path_rank_offline": terminal_best["path_rank_offline"],
            "path_class": terminal_best["path_class"],
            "raw_terminal_penalty": terminal_best["raw_terminal_penalty"],
            "offline_path_match": terminal_best["offline_path_match"],
        },
        "top_token_path": {
            "path_rank_offline": token_top["path_rank_offline"],
            "path_class": token_top["path_class"],
            "tokens_total": round(
                float(token_top["prompt_tokens_total"]) + float(token_top["completion_tokens_total"]),
                6,
            ),
            "offline_path_match": token_top["offline_path_match"],
        },
        "top_offline_match_path": {
            "path_rank_offline": offline_top["path_rank_offline"],
            "path_class": offline_top["path_class"],
            "offline_path_match": offline_top["offline_path_match"],
            "raw_terminal_penalty": offline_top["raw_terminal_penalty"],
        },
        "top_terminal_matches_top_offline": terminal_best["path_agent_ids"] == offline_top["path_agent_ids"],
        "top_token_matches_top_offline": token_top["path_agent_ids"] == offline_top["path_agent_ids"],
        "offline_match_vs_terminal_corr": pearson_correlation(
            [float(row["offline_path_match"]) for row in rows],
            [-float(row["raw_terminal_penalty"]) for row in rows],
        ),
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.data.resolve())
    indexed_rows = index_rows_by_task_id(rows)
    bucket_membership = load_bucket_membership(args.bucket_file.resolve())

    task_ids = list(args.task_ids or [])
    if not task_ids:
        task_ids = select_demo_task_ids(bucket_membership)
    missing = [task_id for task_id in task_ids if task_id not in indexed_rows]
    if missing:
        raise SystemExit(f"Missing task ids in dataset: {missing}")
    print(f"[diagnostic] selected_task_ids={task_ids}", flush=True)

    generator = TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(args.family_kind, seed=args.seed)
    validation_errors = generator.validate_family(family_spec, agent_map)
    if validation_errors:
        raise SystemExit(f"Family validation failed: {validation_errors}")

    all_paths = enumerate_family_paths(
        stages=list(family_spec.stages),
        stage_agents=family_spec.stage_agents,
        allowed_children=family_spec.allowed_children,
    )
    adapter = TelecomMMSTaskAdapter()
    env = FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=args.family_kind,
        family_seed=args.seed,
        executor_name="llm_bench",
    )
    if args.model and getattr(env.family_executor, "model", None) is not None:
        env.family_executor.model = args.model

    records: list[dict[str, Any]] = []
    task_path_selection: dict[str, list[dict[str, Any]]] = {}
    task_summaries: list[dict[str, Any]] = []

    for task_id in task_ids:
        row = indexed_rows[task_id]
        print(f"[task] {task_id} bucket={bucket_label_for_task(task_id, bucket_membership)}", flush=True)
        stage_requirements = build_stage_requirements(row, adapter)
        offline_rankings = sort_offline_records(
            [
                build_offline_path_record(
                    task_id=task_id,
                    path=path,
                    stage_requirements=stage_requirements,
                    agent_map=agent_map,
                )
                for path in all_paths
            ]
        )
        selected_paths = select_representative_paths(
            offline_rankings,
            path_mode=args.path_mode,
            top_k_offline=args.top_k_offline,
            max_paths_per_task=args.max_paths_per_task,
        )
        task_path_selection[task_id] = [
            {
                "path_rank_offline": int(path_row["rank"]),
                "path_class": path_row["path_class"],
                "path_lane_sequence": list(path_row["path_lane_sequence"]),
                "path_agent_ids": list(path_row["path_agent_ids"]),
                "offline_path_match": path_row["path_match"],
                "path_route_summary": path_row["path_route_summary"],
            }
            for path_row in selected_paths
        ]

        for path_row in selected_paths:
            print(
                "[path] "
                f"task_id={task_id} "
                f"offline_rank={path_row['rank']} "
                f"path_class={path_row['path_class']} "
                f"path_match={path_row['path_match']}",
                flush=True,
            )
            env.reset(row)
            result = env.run_path(list(path_row["path_agent_ids"]))
            episode_log = result.episode_log or {}
            stage_trace = {
                str(stage_row["stage_name"]): stage_row
                for stage_row in episode_log.get("stage_trace", [])
            }
            stage5_output = (
                result.stage_outputs.get("stage5", {}).get("output", {})
                if isinstance(result.stage_outputs.get("stage5"), dict)
                else {}
            )
            tool_call_count = sum(
                len(stage_row.get("executed_tool_calls", []) or [])
                for stage_row in stage_trace.values()
            )
            record = {
                "task_id": task_id,
                "instance_id": str(row.get("instance_id")),
                "bucket_label": bucket_label_for_task(task_id, bucket_membership),
                "path_rank_offline": int(path_row["rank"]),
                "path_class": str(path_row["path_class"]),
                "path_lane_sequence": list(path_row["path_lane_sequence"]),
                "path_agent_ids": list(path_row["path_agent_ids"]),
                "path_route_summary": str(path_row["path_route_summary"]),
                "offline_path_match": float(path_row["path_match"]),
                "raw_terminal_penalty": float(result.raw_terminal_penalty),
                "raw_path_cost_component": float(result.raw_path_cost_component),
                "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
                "raw_total_cost": float(result.raw_total_cost),
                "prompt_tokens_total": float(result.prompt_tokens_total),
                "completion_tokens_total": float(result.completion_tokens_total),
                "total_tokens_total": float(result.total_tokens_total),
                "api_cost_total_usd_raw": float(result.api_cost_total_usd_raw),
                "tool_call_count": int(tool_call_count),
                "final_action": result.final_action,
                "oracle_action": result.oracle_action,
                "selected_blocker_ids": list(stage5_output.get("selected_blocker_ids", [])),
                "deferred_blocker_ids": list(stage5_output.get("deferred_blocker_ids", [])),
                "exact_match": bool(result.success),
                "stage_resource_summary": flatten_stage_resource_summary(stage_trace),
                "first_private_barrier_stage": episode_log.get("first_private_barrier_stage"),
            }
            records.append(record)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row["task_id"])].append(row)
    for task_id in task_ids:
        task_summaries.append(build_task_summary(task_id, grouped[task_id]))

    overall_summary = {
        "task_count": len(task_ids),
        "record_count": len(records),
        "mean_terminal_penalty_spread": round(
            statistics.fmean(
                summary["raw_terminal_penalty_range"]["spread"]
                for summary in task_summaries
                if summary["raw_terminal_penalty_range"] is not None
            ),
            6,
        )
        if task_summaries
        else 0.0,
        "mean_reasoning_cost_spread": round(
            statistics.fmean(
                summary["raw_reasoning_cost_component_range"]["spread"]
                for summary in task_summaries
                if summary["raw_reasoning_cost_component_range"] is not None
            ),
            6,
        )
        if task_summaries
        else 0.0,
        "mean_prompt_token_spread": round(
            statistics.fmean(
                summary["prompt_tokens_total_range"]["spread"]
                for summary in task_summaries
                if summary["prompt_tokens_total_range"] is not None
            ),
            6,
        )
        if task_summaries
        else 0.0,
        "tasks_with_action_diversity": sum(
            1 for summary in task_summaries if int(summary["unique_final_action_count"]) > 1
        ),
        "tasks_with_terminal_penalty_spread_gt_zero": sum(
            1
            for summary in task_summaries
            if summary["raw_terminal_penalty_range"] is not None
            and float(summary["raw_terminal_penalty_range"]["spread"]) > 0.0
        ),
        "tasks_with_reasoning_spread_gt_zero": sum(
            1
            for summary in task_summaries
            if summary["raw_reasoning_cost_component_range"] is not None
            and float(summary["raw_reasoning_cost_component_range"]["spread"]) > 0.0
        ),
        "tasks_with_prompt_token_spread_gt_zero": sum(
            1
            for summary in task_summaries
            if summary["prompt_tokens_total_range"] is not None
            and float(summary["prompt_tokens_total_range"]["spread"]) > 0.0
        ),
        "offline_match_vs_terminal_corr_mean": (
            round(
                statistics.fmean(
                    float(summary["offline_match_vs_terminal_corr"])
                    for summary in task_summaries
                    if summary["offline_match_vs_terminal_corr"] is not None
                ),
                6,
            )
            if any(summary["offline_match_vs_terminal_corr"] is not None for summary in task_summaries)
            else None
        ),
    }

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "records.json", records)
    write_csv(output_dir / "records.csv", [flatten_record_for_csv(row) for row in records])
    write_json(
        output_dir / "summary.json",
        {
            "script": str(Path(__file__).resolve()),
            "data": str(args.data.resolve()),
            "family_kind": args.family_kind,
            "seed": args.seed,
            "model": args.model or os.environ.get("PSAGENT_LLM_BENCH_MODEL"),
            "bucket_file": str(args.bucket_file.resolve()),
            "path_mode": args.path_mode,
            "top_k_offline": args.top_k_offline,
            "max_paths_per_task": args.max_paths_per_task,
            "selected_task_ids": task_ids,
            "task_path_selection": task_path_selection,
            "task_summaries": task_summaries,
            "overall_summary": overall_summary,
        },
    )


if __name__ == "__main__":
    main()
