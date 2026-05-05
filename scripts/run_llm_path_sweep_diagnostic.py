from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from copy import deepcopy
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
from tree_family.specs import FamilySpec  # noqa: E402


STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
HYBRID_PATH_CLASSES = {
    "hybrid_trap_to_target",
    "hybrid_general_to_target",
    "hybrid_with_barrier",
}
DEFAULT_BUCKET_FILE = (
    ROOT / "analysis" / "shared_basin_prefix_dedup_profile_switch_schedule_buckets.json"
)
FAST_TOKEN_BUDGET_PER_STAGE = 1200
FAST_TOKEN_PENALTY_BLOCK_SIZE = 200
FAST_TOKEN_OVER_BUDGET_PENALTY_PER_BLOCK = 0.25


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
    parser.add_argument(
        "--parallelism",
        type=int,
        default=max(1, int(os.environ.get("PSAGENT_LLM_DIAG_PARALLELISM", "1") or 1)),
    )
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


def json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def extract_contract_self_check(
    normalized_output: dict[str, Any],
    stage_trace: dict[str, Any],
) -> dict[str, Any] | None:
    """Return report-only self-check even if a legacy normalized output dropped it."""

    direct = normalized_output.get("contract_self_check")
    if isinstance(direct, dict):
        return json_ready(direct)

    raw_output = stage_trace.get("raw_output")
    if isinstance(raw_output, dict) and isinstance(raw_output.get("contract_self_check"), dict):
        return json_ready(raw_output["contract_self_check"])

    for message in reversed(stage_trace.get("llm_raw_output", []) or []):
        if not isinstance(message, dict):
            continue
        content = message.get("content")
        if not isinstance(content, str) or "contract_self_check" not in content:
            continue
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict) and isinstance(parsed.get("contract_self_check"), dict):
            return json_ready(parsed["contract_self_check"])
    return None


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
) -> dict[str, Any]:
    descriptor = adapter.build_task_descriptor(row)
    deliberation_requirements = descriptor.stage_deliberation_requirements or {
        stage_name: ("deep" if float(descriptor.stage_difficulty.get(stage_name, 0.0)) >= 0.42 else "fast")
        for stage_name in STAGES
    }
    return {
        "deliberation": {
            stage_name: (
                "deep"
                if str(deliberation_requirements.get(stage_name, "fast")).strip().lower() == "deep"
                else "fast"
            )
            for stage_name in STAGES
        }
    }


def stage_match(
    *,
    requirement_bundle: dict[str, Any],
    agent: Any,
    stage_name: str,
) -> float:
    reasoning_requirement = str(
        requirement_bundle["deliberation"].get(stage_name, "fast")
    ).strip().lower()
    reasoning_mode = str(getattr(agent, "deliberation_mode", "deep")).strip().lower()
    reasoning_score = 1.0 if reasoning_requirement == reasoning_mode else 0.0

    return reasoning_score


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
    stage_requirements: dict[str, Any],
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    family_agents = [agent_map[agent_id] for agent_id in path]
    stage_scores = {
        stage_name: stage_match(
            requirement_bundle=stage_requirements,
            agent=agent,
            stage_name=stage_name,
        )
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
                "stage_id": row.get("stage_id", stage_name),
                "stage_name": row.get("stage_name", stage_name),
                "model": row.get("model"),
                "agent_id": row.get("agent_id"),
                "agent_deliberation_mode": row.get("agent_deliberation_mode"),
                "stage_requirement": row.get("stage_requirement"),
                "base_round_budget": int(row.get("base_round_budget", 0) or 0),
                "max_rounds_allowed": int(row.get("max_rounds_allowed", 0) or 0),
                "llm_call_count_stage": int(row.get("llm_call_count_stage", 0) or 0),
                "llm_call_count_over_base_budget": int(
                    row.get("llm_call_count_over_base_budget", 0) or 0
                ),
                "valid_json_first_try": bool(row.get("valid_json_first_try", False)),
                "json_retry_count": int(row.get("json_retry_count", 0) or 0),
                "diagnostic_fallback_used": bool(row.get("diagnostic_fallback_used", False)),
                "verification_fallback_used": bool(row.get("verification_fallback_used", False)),
                "fallback_used": bool(row.get("fallback_used", False)),
                "replay_tool_call_count": int(row.get("replay_tool_call_count", 0) or 0),
                "prompt_tokens_stage": float(row.get("prompt_tokens_stage", 0.0) or 0.0),
                "completion_tokens_stage": float(
                    row.get("completion_tokens_stage", 0.0) or 0.0
                ),
                "total_tokens_stage": float(row.get("total_tokens_stage", 0.0) or 0.0),
                "prompt_tokens_total_stage": float(row.get("prompt_tokens_total_stage", 0.0) or 0.0),
                "completion_tokens_total_stage": float(
                    row.get("completion_tokens_total_stage", 0.0) or 0.0
                ),
                "total_tokens_total_stage": float(row.get("total_tokens_total_stage", 0.0) or 0.0),
                "token_usage_available": bool(row.get("token_usage_available", False)),
                "estimated_total_tokens_stage": float(
                    row.get("estimated_total_tokens_stage", 0.0) or 0.0
                ),
                "token_budget_stage": float(row.get("token_budget_stage", 0.0) or 0.0),
                "token_over_budget_stage": float(row.get("token_over_budget_stage", 0.0) or 0.0),
                "token_over_budget_units": int(row.get("token_over_budget_units", 0) or 0),
                "token_over_budget_penalty": float(
                    row.get("token_over_budget_penalty", 0.0) or 0.0
                ),
                "is_fast_agent": bool(row.get("is_fast_agent", False)),
                "is_deep_agent": bool(row.get("is_deep_agent", False)),
                "tool_call_count_stage": len(row.get("executed_tool_calls", []) or []),
            }
        )
    return summary


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def aggregate_path_resource_summary(stage_rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = {
        "llm_call_count_total": 0,
        "prompt_tokens_total": 0.0,
        "completion_tokens_total": 0.0,
        "total_tokens_total": 0.0,
        "json_retry_count_total": 0,
        "fallback_count_total": 0,
        "replay_tool_call_count_total": 0,
        "token_over_budget_penalty_total": 0.0,
        "fast_llm_call_count_total": 0,
        "fast_prompt_tokens_total": 0.0,
        "fast_completion_tokens_total": 0.0,
        "fast_total_tokens_total": 0.0,
        "fast_json_retry_count_total": 0,
        "fast_fallback_count_total": 0,
        "fast_replay_tool_call_count_total": 0,
        "fast_token_over_budget_total": 0.0,
        "fast_token_over_budget_penalty_total": 0.0,
        "fast_llm_call_count_over_base_budget_total": 0,
    }
    for row in stage_rows:
        totals["llm_call_count_total"] += int(row.get("llm_call_count_stage", 0) or 0)
        totals["prompt_tokens_total"] += float(row.get("prompt_tokens_stage", 0.0) or 0.0)
        totals["completion_tokens_total"] += float(
            row.get("completion_tokens_stage", 0.0) or 0.0
        )
        totals["total_tokens_total"] += float(row.get("total_tokens_stage", 0.0) or 0.0)
        totals["json_retry_count_total"] += int(row.get("json_retry_count", 0) or 0)
        totals["fallback_count_total"] += int(bool(row.get("fallback_used", False)))
        totals["replay_tool_call_count_total"] += int(row.get("replay_tool_call_count", 0) or 0)
        totals["token_over_budget_penalty_total"] += float(
            row.get("token_over_budget_penalty", 0.0) or 0.0
        )
        if bool(row.get("is_fast_agent", False)):
            totals["fast_llm_call_count_total"] += int(row.get("llm_call_count_stage", 0) or 0)
            totals["fast_prompt_tokens_total"] += float(row.get("prompt_tokens_stage", 0.0) or 0.0)
            totals["fast_completion_tokens_total"] += float(
                row.get("completion_tokens_stage", 0.0) or 0.0
            )
            totals["fast_total_tokens_total"] += float(row.get("total_tokens_stage", 0.0) or 0.0)
            totals["fast_json_retry_count_total"] += int(row.get("json_retry_count", 0) or 0)
            totals["fast_fallback_count_total"] += int(bool(row.get("fallback_used", False)))
            totals["fast_replay_tool_call_count_total"] += int(
                row.get("replay_tool_call_count", 0) or 0
            )
            totals["fast_token_over_budget_total"] += float(
                row.get("token_over_budget_stage", 0.0) or 0.0
            )
            totals["fast_token_over_budget_penalty_total"] += float(
                row.get("token_over_budget_penalty", 0.0) or 0.0
            )
            totals["fast_llm_call_count_over_base_budget_total"] += int(
                row.get("llm_call_count_over_base_budget", 0) or 0
            )
    return totals


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
        "raw_terminal_penalty_exec_clean_v4": record.get("raw_terminal_penalty_exec_clean_v4"),
        "terminal_adjustment_reasons": json.dumps(
            record.get("terminal_adjustment_reasons", []), ensure_ascii=False
        ),
        "terminal_clear_success_proxy": record.get("terminal_clear_success_proxy"),
        "terminal_auxiliary_success_proxy": record.get("terminal_auxiliary_success_proxy"),
        "raw_reasoning_cost_component": record["raw_reasoning_cost_component"],
        "raw_total_cost": record["raw_total_cost"],
        "raw_total_cost_with_token_penalty": record["raw_total_cost_with_token_penalty"],
        "prompt_tokens_total": record["prompt_tokens_total"],
        "completion_tokens_total": record["completion_tokens_total"],
        "total_tokens_total": record["total_tokens_total"],
        "llm_call_count_total": record["llm_call_count_total"],
        "json_retry_count_total": record["json_retry_count_total"],
        "fallback_count_total": record["fallback_count_total"],
        "replay_tool_call_count_total": record["replay_tool_call_count_total"],
        "token_over_budget_penalty_total": record["token_over_budget_penalty_total"],
        "fast_total_tokens_total": record["fast_total_tokens_total"],
        "fast_json_retry_count_total": record["fast_json_retry_count_total"],
        "fast_fallback_count_total": record["fast_fallback_count_total"],
        "fast_token_over_budget_penalty_total": record["fast_token_over_budget_penalty_total"],
        "api_cost_total_usd_raw": record["api_cost_total_usd_raw"],
        "tool_call_count": record["tool_call_count"],
        "final_action": record["final_action"],
        "stage4_repairability": record.get("stage4_repairability"),
        "stage4_transfer_reason": record.get("stage4_transfer_reason"),
        "stage4_contract_prompt_version": record.get("stage4_contract_prompt_version"),
        "stage4_contract_self_check": json.dumps(
            record.get("stage4_contract_self_check"), ensure_ascii=False
        ),
        "stage4_should_repair_true_count": record.get("stage4_should_repair_true_count"),
        "stage4_llm_call_count": record.get("stage4_llm_call_count"),
        "stage4_json_retry_count": record.get("stage4_json_retry_count"),
        "stage4_fallback_used": record.get("stage4_fallback_used"),
        "stage4_normalizer_changed_output": record.get("stage4_normalizer_changed_output"),
        "stage4_completion_pass_applied": record.get("stage4_completion_pass_applied"),
        "stage4_completion_prerequisite_pass_applied": record.get(
            "stage4_completion_prerequisite_pass_applied"
        ),
        "stage4_completion_added_prerequisite_blockers_count": len(
            record.get("stage4_completion_added_prerequisite_blockers", []) or []
        ),
        "stage4_completion_added_downstream_blockers_count": len(
            record.get("stage4_completion_added_downstream_blockers", []) or []
        ),
        "stage4_completion_added_blockers_count": len(
            record.get("stage4_completion_added_blockers", []) or []
        ),
        "stage5_raw_action_hint": record.get("stage5_raw_action_hint"),
        "stage5_contract_prompt_version": record.get("stage5_contract_prompt_version"),
        "stage5_contract_self_check": json.dumps(
            record.get("stage5_contract_self_check"), ensure_ascii=False
        ),
        "stage5_replay_tool_names": json.dumps(
            record.get("stage5_replay_tool_names", []), ensure_ascii=False
        ),
        "stage5_executed_tool_names": json.dumps(
            record.get("stage5_executed_tool_names", []), ensure_ascii=False
        ),
        "stage4_executed_tool_names": json.dumps(
            record.get("stage4_executed_tool_names", []), ensure_ascii=False
        ),
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
    token_top = max(
        rows,
        key=lambda row: (float(row["total_tokens_total"]), row["path_rank_offline"]),
    )
    offline_top = max(rows, key=lambda row: (float(row["offline_path_match"]), -int(row["path_rank_offline"])))
    return {
        "task_id": task_id,
        "bucket_label": rows[0]["bucket_label"] if rows else "other",
        "path_count": len(rows),
        "raw_terminal_penalty_range": range_summary([float(row["raw_terminal_penalty"]) for row in rows]),
        "raw_total_cost_range": range_summary([float(row["raw_total_cost"]) for row in rows]),
        "raw_total_cost_with_token_penalty_range": range_summary(
            [float(row["raw_total_cost_with_token_penalty"]) for row in rows]
        ),
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
            "tokens_total": round(float(token_top["total_tokens_total"]), 6),
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


def collect_stage_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record in records:
        for row in record.get("stage_resource_summary", []) or []:
            if isinstance(row, dict):
                out.append(row)
    return out


def build_path_class_summary(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        grouped[str(row.get("path_class", "other"))].append(row)

    out: dict[str, dict[str, Any]] = {}
    for path_class in ["pure_general", "pure_target", "pure_trap", "hybrid_with_barrier"]:
        rows = grouped.get(path_class, [])
        out[path_class] = {
            "n": len(rows),
            "exact_match_mean": _mean([1.0 if row.get("exact_match") else 0.0 for row in rows]),
            "raw_terminal_penalty_mean": _mean(
                [float(row.get("raw_terminal_penalty", 0.0)) for row in rows]
            ),
            "raw_total_cost_mean": _mean([float(row.get("raw_total_cost", 0.0)) for row in rows]),
            "raw_total_cost_with_token_penalty_mean": _mean(
                [float(row.get("raw_total_cost_with_token_penalty", 0.0)) for row in rows]
            ),
            "fast_total_tokens_mean": _mean(
                [float(row.get("fast_total_tokens_total", 0.0)) for row in rows]
            ),
            "fast_json_retry_count_mean": _mean(
                [float(row.get("fast_json_retry_count_total", 0.0)) for row in rows]
            ),
            "fast_fallback_count_mean": _mean(
                [float(row.get("fast_fallback_count_total", 0.0)) for row in rows]
            ),
            "fast_token_over_budget_penalty_mean": _mean(
                [float(row.get("fast_token_over_budget_penalty_total", 0.0)) for row in rows]
            ),
            "stage4_completion_added_blockers_mean": _mean(
                [
                    float(len(row.get("stage4_completion_added_blockers", []) or []))
                    for row in rows
                ]
            ),
            "stage4_completion_added_prerequisite_blockers_mean": _mean(
                [
                    float(
                        len(row.get("stage4_completion_added_prerequisite_blockers", []) or [])
                    )
                    for row in rows
                ]
            ),
            "stage4_completion_prerequisite_pass_applied_rate": _mean(
                [
                    1.0
                    if row.get("stage4_completion_prerequisite_pass_applied")
                    else 0.0
                    for row in rows
                ]
            ),
            "stage4_completion_pass_applied_rate": _mean(
                [1.0 if row.get("stage4_completion_pass_applied") else 0.0 for row in rows]
            ),
            "llm_call_count_total_mean": _mean(
                [float(row.get("llm_call_count_total", 0.0)) for row in rows]
            ),
        }
    return out


def build_extended_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    stage_rows = collect_stage_rows(records)
    fast_stage_rows = [row for row in stage_rows if bool(row.get("is_fast_agent", False))]
    fast_deep_stage_rows = [
        row
        for row in fast_stage_rows
        if str(row.get("stage_requirement", "")).strip().lower() == "deep"
    ]
    stage4_records = records
    return {
        "avg_llm_call_count_total": _mean(
            [float(row.get("llm_call_count_total", 0.0)) for row in records]
        ),
        "avg_total_tokens_total": _mean(
            [float(row.get("total_tokens_total", 0.0)) for row in records]
        ),
        "avg_json_retry_count_total": _mean(
            [float(row.get("json_retry_count_total", 0.0)) for row in records]
        ),
        "avg_fallback_count_total": _mean(
            [float(row.get("fallback_count_total", 0.0)) for row in records]
        ),
        "avg_replay_tool_call_count_total": _mean(
            [float(row.get("replay_tool_call_count_total", 0.0)) for row in records]
        ),
        "avg_token_over_budget_penalty_total": _mean(
            [float(row.get("token_over_budget_penalty_total", 0.0)) for row in records]
        ),
        "avg_raw_total_cost_with_token_penalty": _mean(
            [float(row.get("raw_total_cost_with_token_penalty", 0.0)) for row in records]
        ),
        "avg_fast_llm_call_count_total": _mean(
            [float(row.get("fast_llm_call_count_total", 0.0)) for row in records]
        ),
        "avg_fast_total_tokens_total": _mean(
            [float(row.get("fast_total_tokens_total", 0.0)) for row in records]
        ),
        "avg_fast_json_retry_count_total": _mean(
            [float(row.get("fast_json_retry_count_total", 0.0)) for row in records]
        ),
        "avg_fast_fallback_count_total": _mean(
            [float(row.get("fast_fallback_count_total", 0.0)) for row in records]
        ),
        "avg_fast_replay_tool_call_count_total": _mean(
            [float(row.get("fast_replay_tool_call_count_total", 0.0)) for row in records]
        ),
        "avg_fast_token_over_budget_total": _mean(
            [float(row.get("fast_token_over_budget_total", 0.0)) for row in records]
        ),
        "avg_fast_token_over_budget_penalty_total": _mean(
            [float(row.get("fast_token_over_budget_penalty_total", 0.0)) for row in records]
        ),
        "fast_token_over_budget_rate": _mean(
            [1.0 if float(row.get("token_over_budget_stage", 0.0) or 0.0) > 0.0 else 0.0 for row in fast_stage_rows]
        ),
        "fast_valid_json_first_try_rate": _mean(
            [1.0 if row.get("valid_json_first_try") else 0.0 for row in fast_stage_rows]
        ),
        "fast_on_deep_stage_retry_rate": _mean(
            [1.0 if int(row.get("json_retry_count", 0) or 0) > 0 else 0.0 for row in fast_deep_stage_rows]
        ),
        "stage4_valid_json_first_try_rate": _mean(
            [1.0 if row.get("stage4_valid_json_first_try") else 0.0 for row in stage4_records]
        ),
        "stage4_json_retry_count_mean": _mean(
            [float(row.get("stage4_json_retry_count", 0.0)) for row in stage4_records]
        ),
        "stage4_fallback_used_rate": _mean(
            [1.0 if row.get("stage4_fallback_used") else 0.0 for row in stage4_records]
        ),
        "stage4_completion_pass_applied_rate": _mean(
            [1.0 if row.get("stage4_completion_pass_applied") else 0.0 for row in stage4_records]
        ),
        "stage4_completion_prerequisite_pass_applied_rate": _mean(
            [
                1.0
                if row.get("stage4_completion_prerequisite_pass_applied")
                else 0.0
                for row in stage4_records
            ]
        ),
        "stage4_completion_added_prerequisite_blockers_mean": _mean(
            [
                float(len(row.get("stage4_completion_added_prerequisite_blockers", []) or []))
                for row in stage4_records
            ]
        ),
        "stage4_completion_added_blockers_mean": _mean(
            [
                float(len(row.get("stage4_completion_added_blockers", []) or []))
                for row in stage4_records
            ]
        ),
        "stage4_completion_blocked_by_hard_transfer_guard_count": int(
            sum(
                len(row.get("stage4_completion_blocked_by_hard_transfer_guard", []) or [])
                for row in stage4_records
            )
        ),
        "stage4_normalizer_changed_output_rate": _mean(
            [1.0 if row.get("stage4_normalizer_changed_output") else 0.0 for row in stage4_records]
        ),
        "token_usage_unavailable_stage_count": int(
            sum(1 for row in stage_rows if not bool(row.get("token_usage_available", False)))
        ),
        "fast_token_usage_unavailable_stage_count": int(
            sum(1 for row in fast_stage_rows if not bool(row.get("token_usage_available", False)))
        ),
        "path_class_summary": build_path_class_summary(records),
    }


def run_selected_path_job(job: dict[str, Any]) -> dict[str, Any]:
    env = FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=job["family_kind"],
        family_seed=int(job["seed"]),
        executor_name="llm_bench",
    )
    if job.get("model") and getattr(env.family_executor, "model", None) is not None:
        env.family_executor.model = job["model"]

    row = job["row"]
    path_row = job["path_row"]
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
    stage3_output = (
        result.stage_outputs.get("stage3", {}).get("output", {})
        if isinstance(result.stage_outputs.get("stage3"), dict)
        else {}
    )
    stage4_output = (
        result.stage_outputs.get("stage4", {}).get("output", {})
        if isinstance(result.stage_outputs.get("stage4"), dict)
        else {}
    )
    stage4_stage_trace = stage_trace.get("stage4", {})
    stage5_stage_trace = stage_trace.get("stage5", {})
    stage_resource_summary = flatten_stage_resource_summary(stage_trace)
    path_resource_summary = aggregate_path_resource_summary(stage_resource_summary)
    stage4_per_blocker = []
    for blocker_row in stage4_output.get("per_blocker", []) if isinstance(stage4_output, dict) else []:
        if not isinstance(blocker_row, dict):
            continue
        stage4_per_blocker.append(
            {
                "blocker_id": blocker_row.get("blocker_id"),
                "should_repair": blocker_row.get("should_repair"),
                "repairability": blocker_row.get("repairability"),
                "evidence": json_ready(blocker_row.get("evidence")),
                "repair_order": blocker_row.get("repair_order"),
                "execution_attempted": blocker_row.get("execution_attempted"),
                "execution_succeeded": blocker_row.get("execution_succeeded"),
                "executed_step_count": blocker_row.get("executed_step_count"),
            }
        )
    tool_call_count = sum(
        len(stage_row.get("executed_tool_calls", []) or [])
        for stage_row in stage_trace.values()
    )
    raw_total_cost_base = float(result.raw_total_cost)
    raw_total_cost_with_token_penalty = (
        raw_total_cost_base + float(path_resource_summary["fast_token_over_budget_penalty_total"])
    )
    record = {
        "task_id": job["task_id"],
        "instance_id": str(row.get("instance_id")),
        "bucket_label": job["bucket_label"],
        "path_rank_offline": int(path_row["rank"]),
        "path_class": str(path_row["path_class"]),
        "path_lane_sequence": list(path_row["path_lane_sequence"]),
        "path_agent_ids": list(path_row["path_agent_ids"]),
        "path_route_summary": str(path_row["path_route_summary"]),
        "offline_path_match": float(path_row["path_match"]),
        "raw_terminal_penalty": float(result.raw_terminal_penalty),
        "raw_terminal_penalty_exec_clean_v4": episode_log.get(
            "raw_terminal_penalty_exec_clean_v4"
        ),
        "terminal_adjustment_enabled": bool(
            episode_log.get("terminal_adjustment_enabled", False)
        ),
        "terminal_adjustment_floor": episode_log.get("terminal_adjustment_floor"),
        "terminal_adjustment_reasons": list(
            episode_log.get("terminal_adjustment_reasons", []) or []
        ),
        "terminal_clear_success_proxy": bool(
            episode_log.get("clear_success_proxy", bool(result.success))
        ),
        "terminal_auxiliary_success_proxy": bool(
            episode_log.get("auxiliary_success_proxy", True)
        ),
        "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
        "raw_total_cost": raw_total_cost_base,
        "raw_total_cost_with_token_penalty": raw_total_cost_with_token_penalty,
        "prompt_tokens_total": float(result.prompt_tokens_total),
        "completion_tokens_total": float(result.completion_tokens_total),
        "total_tokens_total": float(result.total_tokens_total),
        "api_cost_total_usd_raw": float(result.api_cost_total_usd_raw),
        "tool_call_count": int(tool_call_count),
        "final_action": result.final_action,
        "oracle_action": result.oracle_action,
        "selected_blocker_ids": list(stage5_output.get("selected_blocker_ids", [])),
        "deferred_blocker_ids": list(stage5_output.get("deferred_blocker_ids", [])),
        "stage3_output": json_ready(stage3_output),
        "stage4_output": json_ready(stage4_output),
        "stage5_output": json_ready(stage5_output),
        "stage4_repairability": stage4_output.get("repairability"),
        "stage4_transfer_reason": stage4_output.get("transfer_reason"),
        "stage4_contract_prompt_version": stage4_output.get("stage4_contract_prompt_version"),
        "stage4_contract_self_check": json_ready(stage4_output.get("stage4_contract_self_check")),
        "stage4_per_blocker": stage4_per_blocker,
        "stage4_should_repair_true_count": sum(
            1 for blocker_row in stage4_per_blocker if blocker_row.get("should_repair") is True
        ),
        "stage4_llm_raw_output": json_ready(stage4_stage_trace.get("llm_raw_output", [])),
        "stage4_raw_json_extracted": json_ready(stage4_output.get("stage4_raw_json_extracted")),
        "stage4_raw_action_hint": stage4_output.get("stage4_raw_action_hint"),
        "stage4_prompt_summary": stage4_stage_trace.get("prompt_summary"),
        "stage4_llm_call_count": int(stage4_stage_trace.get("llm_call_count_stage", 0) or 0),
        "stage4_max_rounds_allowed": int(
            stage4_stage_trace.get("max_rounds_allowed", 0) or 0
        ),
        "stage4_base_round_budget": int(
            stage4_stage_trace.get("base_round_budget", 0) or 0
        ),
        "stage4_valid_json_first_try": bool(
            stage4_stage_trace.get("valid_json_first_try", False)
        ),
        "stage4_json_retry_count": int(stage4_stage_trace.get("json_retry_count", 0) or 0),
        "stage4_fallback_used": bool(stage4_stage_trace.get("fallback_used", False)),
        "stage4_normalizer_changed_output": bool(
            stage4_output.get("stage4_normalizer_changed_output", False)
        ),
        "stage4_selected_before_normalization": list(
            stage4_output.get("stage4_selected_before_normalization", []) or []
        ),
        "stage4_deferred_before_normalization": list(
            stage4_output.get("stage4_deferred_before_normalization", []) or []
        ),
        "stage4_selected_after_normalization": list(
            stage4_output.get("stage4_selected_after_normalization", []) or []
        ),
        "stage4_deferred_after_normalization": list(
            stage4_output.get("stage4_deferred_after_normalization", []) or []
        ),
        "stage4_completion_pass_applied": bool(
            stage4_output.get("stage4_completion_pass_applied", False)
        ),
        "stage4_completion_prerequisite_pass_applied": bool(
            stage4_output.get("stage4_completion_prerequisite_pass_applied", False)
        ),
        "stage4_completion_added_prerequisite_blockers": list(
            stage4_output.get("stage4_completion_added_prerequisite_blockers", []) or []
        ),
        "stage4_completion_added_downstream_blockers": list(
            stage4_output.get("stage4_completion_added_downstream_blockers", []) or []
        ),
        "stage4_completion_added_blockers": list(
            stage4_output.get("stage4_completion_added_blockers", []) or []
        ),
        "stage4_completion_blocked_by_hard_transfer_guard": list(
            stage4_output.get("stage4_completion_blocked_by_hard_transfer_guard", []) or []
        ),
        "stage5_raw_action_hint": stage5_stage_trace.get("raw_output", {}).get("final_action")
        if isinstance(stage5_stage_trace.get("raw_output"), dict)
        else None,
        "stage5_contract_prompt_version": stage5_output.get("stage5_contract_prompt_version"),
        "stage5_contract_self_check": extract_contract_self_check(
            stage5_output,
            stage5_stage_trace,
        ),
        "stage5_llm_raw_output": json_ready(stage5_stage_trace.get("llm_raw_output", [])),
        "stage5_replay_tool_names": [
            call.get("name") for call in stage5_stage_trace.get("replay_tool_calls", []) or []
        ],
        "stage5_executed_tool_names": [
            call.get("name") for call in stage5_stage_trace.get("executed_tool_calls", []) or []
        ],
        "stage4_executed_tool_names": [
            call.get("name") for call in stage4_stage_trace.get("executed_tool_calls", []) or []
        ],
        "exact_match": bool(result.success),
        "stage_resource_summary": stage_resource_summary,
        "first_private_barrier_stage": episode_log.get("first_private_barrier_stage"),
        **path_resource_summary,
    }
    return {"job_index": int(job["job_index"]), "record": record}


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
    records: list[dict[str, Any]] = []
    task_path_selection: dict[str, list[dict[str, Any]]] = {}
    task_summaries: list[dict[str, Any]] = []
    jobs: list[dict[str, Any]] = []
    job_index = 0

    for task_id in task_ids:
        row = indexed_rows[task_id]
        bucket_label = bucket_label_for_task(task_id, bucket_membership)
        print(f"[task] {task_id} bucket={bucket_label}", flush=True)
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
            jobs.append(
                {
                    "job_index": job_index,
                    "task_id": task_id,
                    "bucket_label": bucket_label,
                    "row": row,
                    "path_row": path_row,
                    "family_kind": args.family_kind,
                    "seed": args.seed,
                    "model": args.model or os.environ.get("PSAGENT_LLM_BENCH_MODEL"),
                }
            )
            job_index += 1

    if args.parallelism <= 1:
        records = [run_selected_path_job(job)["record"] for job in jobs]
    else:
        completed: list[dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            for result in executor.map(run_selected_path_job, jobs):
                completed.append(result)
        completed.sort(key=lambda row: int(row["job_index"]))
        records = [row["record"] for row in completed]

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
    extended_summary = build_extended_summary(records)

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
            "parallelism": args.parallelism,
            "bucket_file": str(args.bucket_file.resolve()),
            "path_mode": args.path_mode,
            "top_k_offline": args.top_k_offline,
            "max_paths_per_task": args.max_paths_per_task,
            "selected_task_ids": task_ids,
            "task_path_selection": task_path_selection,
            "task_summaries": task_summaries,
            "overall_summary": overall_summary,
            "extended_summary": extended_summary,
        },
    )


if __name__ == "__main__":
    main()
