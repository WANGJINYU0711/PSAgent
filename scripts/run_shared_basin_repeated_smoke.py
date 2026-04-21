"""Run repeated smoke on shared_basin_strong with method-level incremental persistence.

Scope:
- family_kind = shared_basin_strong
- executor_name = llm_bench
- model = gpt-4o-mini
- smoke10 repeated for a fixed horizon
- repeated-smoke baselines, each run as one stateful T=100 sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


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

from fixed_tree_env import FixedTreeEnvironment  # noqa: E402
from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from oracle_eval import find_best_stationary_path  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402


SMOKE10_INDICES = list(range(10))
DATASET_DEFAULT = (
    ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time" / "tasks.json"
)
SPECIALIST_ANALYSIS_PATH = ROOT / "analysis" / "shared_basin_strong_static_analysis.json"
MODEL_REQUIRED = "gpt-4o-mini"
FAMILY_KIND = "shared_basin_strong"
SEED = 0
EXECUTOR_NAME = "llm_bench"


POLICY_REGISTRY = {
    "risky_ps": RiskyPSPolicy,
    "direct_multistage_exp3": DirectMultiStageExp3Policy,
    "epsilon_exp3": EpsilonExp3Policy,
    "naive_mixed": NaiveMixedPolicy,
    "random_path": RandomPathPolicy,
}

DEFAULT_METHODS = [
    "risky_ps",
    "naive_mixed",
    "direct_multistage_exp3",
    "epsilon_exp3",
    "random_path",
]


def validate_methods(methods: list[str]) -> None:
    invalid = [method for method in methods if method not in POLICY_REGISTRY]
    if invalid:
        raise SystemExit(
            f"Repeated smoke only supports these baselines: {sorted(POLICY_REGISTRY)}. "
            f"Unsupported methods: {invalid}"
        )


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


def write_json(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_text_atomic(path, payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def load_instances(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def load_specialist_task_ids() -> set[str]:
    data = load_json(SPECIALIST_ANALYSIS_PATH)
    return set(data.get("unshared_win_task_ids", []))


def build_env(*, executor_name: str) -> FixedTreeEnvironment:
    return FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=FAMILY_KIND,
        family_seed=SEED,
        executor_name=executor_name,
    )


def build_repeated_selection(
    instances: list[dict[str, Any]],
    *,
    indices: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        for position_in_cycle, dataset_index in enumerate(indices):
            repeated.append(
                {
                    "repeat_index": repeat_index,
                    "position_in_cycle": position_in_cycle,
                    "dataset_index": dataset_index,
                    "instance": instances[dataset_index],
                }
            )
    return repeated


def serialize_schedule(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_index, row in enumerate(selected):
        instance = row["instance"]
        rows.append(
            {
                "episode_index": episode_index,
                "repeat_index": row["repeat_index"],
                "position_in_cycle": row["position_in_cycle"],
                "dataset_index": row["dataset_index"],
                "instance_id": instance["instance_id"],
                "original_task_id": instance["original_task_id"],
            }
        )
    return rows


def materialize_schedule(
    instances: list[dict[str, Any]],
    schedule_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in schedule_rows:
        dataset_index = int(row["dataset_index"])
        instance = instances[dataset_index]
        if instance["instance_id"] != row["instance_id"]:
            raise ValueError(
                f"Schedule/dataset mismatch at episode {row['episode_index']}: "
                f"expected instance_id={row['instance_id']}, got {instance['instance_id']}"
            )
        selected.append(
            {
                "episode_index": int(row["episode_index"]),
                "repeat_index": int(row["repeat_index"]),
                "position_in_cycle": int(row["position_in_cycle"]),
                "dataset_index": dataset_index,
                "instance": instance,
            }
        )
    return selected


def compute_stationary_oracle(selected: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_env = build_env(executor_name="simulated")
    oracle_path, oracle_summary_raw = find_best_stationary_path(
        [row["instance"] for row in selected],
        oracle_env,
    )
    oracle_summary = {
        "path": list(oracle_path),
        "episode_total_costs": list(oracle_summary_raw["episode_total_costs"]),
        "episode_terminal_costs": list(oracle_summary_raw["episode_terminal_costs"]),
        "episode_raw_total_costs": list(oracle_summary_raw["episode_raw_total_costs"]),
        "episode_normalized_total_costs": list(oracle_summary_raw["episode_normalized_total_costs"]),
        "raw_cumulative_total_cost": float(oracle_summary_raw["raw_cumulative_total_cost"]),
        "raw_mean_total_cost": float(oracle_summary_raw["raw_mean_total_cost"]),
        "normalized_cumulative_total_cost": float(oracle_summary_raw["normalized_cumulative_total_cost"]),
        "normalized_mean_total_cost": float(oracle_summary_raw["normalized_mean_total_cost"]),
        "cost_scale_version": str(oracle_summary_raw["cost_scale_version"]),
        "cumulative_total_cost": float(oracle_summary_raw["cumulative_total_cost"]),
        "mean_total_cost": float(oracle_summary_raw["mean_total_cost"]),
    }
    return oracle_summary


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean_present(values: list[Any]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric)


def distribution_with_fraction(labels: list[Any]) -> dict[str, dict[str, Any]]:
    if not labels:
        return {}
    counter = Counter("none" if label in {None, ""} else str(label) for label in labels)
    total = sum(counter.values())
    return {
        key: {
            "count": count,
            "fraction": (count / total) if total else 0.0,
        }
        for key, count in sorted(counter.items())
    }


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = max(len(vector) for vector in vectors)
    result: list[float] = []
    for idx in range(width):
        result.append(
            mean([
                float(vector[idx])
                for vector in vectors
                if idx < len(vector)
            ])
        )
    return result


def flatten_episode(
    *,
    episode_index: int,
    row: dict[str, Any],
    result: Any,
    method: str,
    oracle_summary: dict[str, Any],
    selection_info: dict[str, Any],
    update_info: dict[str, Any],
    specialist_task_ids: set[str],
) -> dict[str, Any]:
    instance = row["instance"]
    log = result.episode_log or {}
    stage_trace = {
        stage_row["stage_name"]: stage_row for stage_row in log.get("stage_trace", [])
    }
    stage_sources = {name: stage_row.get("source") for name, stage_row in stage_trace.items()}
    llm_stage_names = [name for name, source in stage_sources.items() if source == "llm_bench"]
    llm_call_count = int(
        log.get(
            "llm_call_count",
            sum(
                int(
                    stage_trace[name].get(
                        "llm_call_count_stage",
                        len(stage_trace[name].get("llm_raw_output", [])),
                    )
                    or 0
                )
                for name in llm_stage_names
            ),
        )
        or 0
    )
    tool_calls_made = sum(
        len(stage_row.get("executed_tool_calls", [])) for stage_row in stage_trace.values()
    )
    mutating_tool_calls_made = len(stage_trace.get("stage4", {}).get("executed_tool_calls", []))
    assistant_side_mutating_tool_calls_made = sum(
        1
        for call in stage_trace.get("stage4", {}).get("executed_tool_calls", [])
        if call.get("requestor") == "assistant"
    )
    stage5_trace = stage_trace.get("stage5", {})
    leaf_type = result.leaf_type
    shared_path = leaf_type == "shared"
    specialist_task = instance["original_task_id"] in specialist_task_ids
    shared_updates = update_info.get("shared_safe_suffix_edges_updated", []) or []
    risky_updates = update_info.get("risky_edges_updated", []) or []
    stage_prompt_tokens = [
        float(stage_trace.get(stage_name, {}).get("prompt_tokens_total_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_completion_tokens = [
        float(
            stage_trace.get(stage_name, {}).get("completion_tokens_total_stage", 0.0) or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_total_tokens = [
        float(stage_trace.get(stage_name, {}).get("total_tokens_total_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_api_cost_usd = [
        float(stage_trace.get(stage_name, {}).get("api_cost_total_usd_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_generation_time_seconds = [
        float(
            stage_trace.get(stage_name, {}).get("generation_time_total_seconds_stage", 0.0)
            or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_llm_round_trip_seconds = [
        float(
            stage_trace.get(stage_name, {}).get("llm_round_trip_total_seconds_stage", 0.0)
            or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_wall_clock_seconds = [
        float(stage_trace.get(stage_name, {}).get("stage_wall_clock_seconds", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_tool_wall_clock_seconds = [
        float(
            stage_trace.get(stage_name, {}).get("tool_wall_clock_total_seconds_stage", 0.0)
            or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    return {
        "method": method,
        "episode_index": episode_index,
        "repeat_index": row["repeat_index"],
        "position_in_cycle": row["position_in_cycle"],
        "dataset_index": row["dataset_index"],
        "instance_id": instance["instance_id"],
        "original_task_id": instance["original_task_id"],
        "is_specialist_task": specialist_task,
        "selected_path": list(result.selected_path),
        "leaf_type": leaf_type,
        "selected_shared_path": shared_path,
        "selected_unshared_path": not shared_path,
        "oracle_action": result.oracle_action,
        "final_action": result.final_action,
        "exact_match": bool(result.success),
        "subset_mismatch": bool(log.get("subset_mismatch", False)),
        "terminal_penalty": float(result.terminal_cost),
        "raw_outcome_penalty": float(log.get("raw_outcome_penalty", 0.0) or 0.0),
        "raw_policy_penalty": float(log.get("raw_policy_penalty", 0.0) or 0.0),
        "raw_terminal_penalty": float(result.raw_terminal_penalty),
        "total_cost": float(result.total_cost),
        "raw_total_cost": float(result.raw_total_cost),
        "raw_total_cost_api": (
            float(log.get("raw_total_cost_api"))
            if log.get("raw_total_cost_api") is not None
            else None
        ),
        "raw_total_cost_token": (
            float(log.get("raw_total_cost_token"))
            if log.get("raw_total_cost_token") is not None
            else None
        ),
        "raw_path_cost_component": float(result.raw_path_cost_component),
        "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
        "raw_reasoning_cost_component_api": (
            float(log.get("raw_reasoning_cost_component_api"))
            if log.get("raw_reasoning_cost_component_api") is not None
            else None
        ),
        "raw_reasoning_cost_component_token": (
            float(log.get("raw_reasoning_cost_component_token"))
            if log.get("raw_reasoning_cost_component_token") is not None
            else None
        ),
        "reasoning_cost": float(result.reasoning_cost),
        "reasoning_cost_mode_default": log.get("reasoning_cost_mode_default"),
        "policy_eval_source": log.get("policy_eval_source"),
        "policy_eval_scope": log.get("policy_eval_scope"),
        "terminal_cost_upper_bound": log.get("terminal_cost_upper_bound"),
        "path_cost_upper_bound": log.get("path_cost_upper_bound"),
        "reasoning_cost_upper_bound": log.get("reasoning_cost_upper_bound"),
        "total_cost_upper_bound": log.get("total_cost_upper_bound"),
        "cost_scale_version": str(result.cost_scale_version),
        "stage_sources": stage_sources,
        "llm_stage_names": llm_stage_names,
        "llm_call_count": llm_call_count,
        "prompt_tokens_total": float(log.get("prompt_tokens_total", 0.0) or 0.0),
        "completion_tokens_total": float(log.get("completion_tokens_total", 0.0) or 0.0),
        "total_tokens_total": float(log.get("total_tokens_total", 0.0) or 0.0),
        "api_cost_total_usd_raw": float(log.get("api_cost_total_usd_raw", 0.0) or 0.0),
        "generation_time_total_seconds": float(
            log.get("generation_time_total_seconds", 0.0) or 0.0
        ),
        "llm_round_trip_total_seconds": float(
            log.get("llm_round_trip_total_seconds", 0.0) or 0.0
        ),
        "tool_wall_clock_total_seconds": float(
            log.get("tool_wall_clock_total_seconds", 0.0) or 0.0
        ),
        "episode_wall_clock_seconds": float(
            log.get("episode_wall_clock_seconds", 0.0) or 0.0
        ),
        "stage_prompt_tokens": stage_prompt_tokens,
        "stage_completion_tokens": stage_completion_tokens,
        "stage_total_tokens": stage_total_tokens,
        "stage_api_cost_usd": stage_api_cost_usd,
        "stage_generation_time_seconds": stage_generation_time_seconds,
        "stage_llm_round_trip_seconds": stage_llm_round_trip_seconds,
        "stage_tool_wall_clock_seconds": stage_tool_wall_clock_seconds,
        "stage_wall_clock_seconds": stage_wall_clock_seconds,
        "tool_calls_made": tool_calls_made,
        "mutating_tool_calls_made": mutating_tool_calls_made,
        "assistant_side_mutating_tool_calls_made": assistant_side_mutating_tool_calls_made,
        "stage5_replay_tool_names": [c.get("name") for c in stage5_trace.get("replay_tool_calls", [])],
        "stage5_executed_tool_names": [c.get("name") for c in stage5_trace.get("executed_tool_calls", [])],
        "policy_action_violation": bool(log.get("policy_action_violation", False)),
        "policy_communication_violation": bool(
            log.get("policy_communication_violation", False)
        ),
        "policy_nl_assertions_total": int(log.get("policy_nl_assertions_total", 0) or 0),
        "policy_nl_assertions_failed": int(
            log.get("policy_nl_assertions_failed", 0) or 0
        ),
        "policy_violation_count": int(log.get("policy_violation_count", 0) or 0),
        "first_private_barrier_stage": log.get("first_private_barrier_stage"),
        "barrier_stop_depth": log.get("barrier_stop_depth"),
        "candidate_count_per_stage": list(log.get("candidate_count_per_stage", []) or []),
        "legal_child_count_per_stage": list(
            log.get("legal_child_count_per_stage", []) or []
        ),
        "selection_path_prob": selection_info.get("path_prob"),
        "shared_branch_triggered": bool(update_info.get("shared_leaf_updated", False)),
        "unshared_branch_triggered": str(update_info.get("leaf_type")) == "unshared",
        "shared_update_count": len(shared_updates),
        "unshared_edge_update_count": len(risky_updates),
        "risky_edge_update_edges": risky_updates,
        "selection_info": selection_info,
        "update_info": update_info,
    }


def add_cumulative_fields(episodes: list[dict[str, Any]]) -> None:
    shared_count = 0
    unshared_count = 0
    shared_branch_count = 0
    unshared_branch_count = 0
    shared_update_count = 0
    unshared_edge_update_count = 0
    window: list[bool] = []
    for idx, row in enumerate(episodes, start=1):
        is_shared = bool(row["selected_shared_path"])
        shared_count += int(is_shared)
        unshared_count += int(not is_shared)
        shared_branch_count += int(bool(row["shared_branch_triggered"]))
        unshared_branch_count += int(bool(row["unshared_branch_triggered"]))
        shared_update_count += int(row["shared_update_count"])
        unshared_edge_update_count += int(row["unshared_edge_update_count"])
        window.append(is_shared)
        if len(window) > 10:
            window.pop(0)
        row["cumulative_shared_path_ratio"] = shared_count / idx
        row["cumulative_unshared_path_ratio"] = unshared_count / idx
        row["rolling_shared_path_ratio_last10"] = sum(window) / len(window)
        row["rolling_unshared_path_ratio_last10"] = 1.0 - row["rolling_shared_path_ratio_last10"]
        row["cumulative_shared_branch_count"] = shared_branch_count
        row["cumulative_unshared_branch_count"] = unshared_branch_count
        row["cumulative_shared_update_count"] = shared_update_count
        row["cumulative_unshared_edge_update_count"] = unshared_edge_update_count


def build_summary(
    *,
    method: str,
    dataset: str,
    repeats: int,
    model: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_source_summary: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        for stage_name, source in episode["stage_sources"].items():
            stage_source_summary[stage_name][str(source)] += 1
    policy_nl_total = sum(ep["policy_nl_assertions_total"] for ep in episodes)
    policy_nl_failed = sum(ep["policy_nl_assertions_failed"] for ep in episodes)
    return {
        "test_name": f"{method}_smoke10x{repeats}_{FAMILY_KIND}_full_llm",
        "dataset": dataset,
        "dataset_indices": SMOKE10_INDICES,
        "repeats": repeats,
        "episodes": len(episodes),
        "method": method,
        "mechanism": "algorithm_direct",
        "executor_name": EXECUTOR_NAME,
        "family_kind": FAMILY_KIND,
        "seed": SEED,
        "model": model,
        "stationary_oracle_path": oracle_summary["path"],
        "exact_match_mean": mean([float(ep["exact_match"]) for ep in episodes]),
        "terminal_penalty_mean": mean([ep["terminal_penalty"] for ep in episodes]),
        "raw_outcome_penalty_mean": mean([ep["raw_outcome_penalty"] for ep in episodes]),
        "raw_policy_penalty_mean": mean([ep["raw_policy_penalty"] for ep in episodes]),
        "raw_terminal_penalty_mean": mean([ep["raw_terminal_penalty"] for ep in episodes]),
        "total_cost_mean": mean([ep["total_cost"] for ep in episodes]),
        "raw_total_cost_mean": mean([ep["raw_total_cost"] for ep in episodes]),
        "raw_total_cost_api_mean": mean_present([ep["raw_total_cost_api"] for ep in episodes]),
        "raw_total_cost_token_mean": mean_present(
            [ep["raw_total_cost_token"] for ep in episodes]
        ),
        "reasoning_cost_mean": mean([ep["reasoning_cost"] for ep in episodes]),
        "raw_reasoning_cost_component_mean": mean([ep["raw_reasoning_cost_component"] for ep in episodes]),
        "raw_reasoning_cost_component_api_mean": mean_present(
            [ep["raw_reasoning_cost_component_api"] for ep in episodes]
        ),
        "raw_reasoning_cost_component_token_mean": mean_present(
            [ep["raw_reasoning_cost_component_token"] for ep in episodes]
        ),
        "raw_path_cost_component_mean": mean([ep["raw_path_cost_component"] for ep in episodes]),
        "algorithm_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
        "raw_algorithm_cumulative_total_cost": sum(ep["raw_total_cost"] for ep in episodes),
        "oracle_stationary_total_cost": oracle_summary["cumulative_total_cost"],
        "raw_oracle_stationary_total_cost": oracle_summary["raw_cumulative_total_cost"],
        "raw_outcome_penalty_cumulative": sum(ep["raw_outcome_penalty"] for ep in episodes),
        "raw_policy_penalty_cumulative": sum(ep["raw_policy_penalty"] for ep in episodes),
        "raw_terminal_penalty_cumulative": sum(ep["raw_terminal_penalty"] for ep in episodes),
        "raw_path_cost_component_cumulative": sum(ep["raw_path_cost_component"] for ep in episodes),
        "raw_reasoning_cost_component_cumulative": sum(ep["raw_reasoning_cost_component"] for ep in episodes),
        "mean_llm_call_count": mean([ep["llm_call_count"] for ep in episodes]),
        "mean_prompt_tokens": mean([ep["prompt_tokens_total"] for ep in episodes]),
        "mean_completion_tokens": mean(
            [ep["completion_tokens_total"] for ep in episodes]
        ),
        "mean_total_tokens": mean([ep["total_tokens_total"] for ep in episodes]),
        "cumulative_total_tokens": sum(ep["total_tokens_total"] for ep in episodes),
        "mean_api_cost_usd_raw": mean([ep["api_cost_total_usd_raw"] for ep in episodes]),
        "cumulative_api_cost_usd_raw": sum(
            ep["api_cost_total_usd_raw"] for ep in episodes
        ),
        "mean_generation_time_seconds": mean(
            [ep["generation_time_total_seconds"] for ep in episodes]
        ),
        "p50_generation_time_seconds": percentile(
            [ep["generation_time_total_seconds"] for ep in episodes],
            0.5,
        ),
        "p90_generation_time_seconds": percentile(
            [ep["generation_time_total_seconds"] for ep in episodes],
            0.9,
        ),
        "mean_llm_round_trip_seconds": mean(
            [ep["llm_round_trip_total_seconds"] for ep in episodes]
        ),
        "mean_episode_wall_clock_seconds": mean(
            [ep["episode_wall_clock_seconds"] for ep in episodes]
        ),
        "p50_episode_wall_clock_seconds": percentile(
            [ep["episode_wall_clock_seconds"] for ep in episodes],
            0.5,
        ),
        "p90_episode_wall_clock_seconds": percentile(
            [ep["episode_wall_clock_seconds"] for ep in episodes],
            0.9,
        ),
        "mean_tool_wall_clock_seconds": mean(
            [ep["tool_wall_clock_total_seconds"] for ep in episodes]
        ),
        "policy_action_violation_rate": mean(
            [float(ep["policy_action_violation"]) for ep in episodes]
        ),
        "policy_communication_violation_rate": mean(
            [float(ep["policy_communication_violation"]) for ep in episodes]
        ),
        "policy_nl_assertion_failure_rate": (
            policy_nl_failed / policy_nl_total if policy_nl_total else 0.0
        ),
        "mean_policy_violation_count": mean(
            [ep["policy_violation_count"] for ep in episodes]
        ),
        "subset_mismatch_count": sum(1 for ep in episodes if ep["subset_mismatch"]),
        "episodes_with_stage5_verification_tools": sum(1 for ep in episodes if ep["stage5_executed_tool_names"]),
        "shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in episodes]),
        "unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in episodes]),
        "mean_barrier_stop_depth": mean_present(
            [ep["barrier_stop_depth"] for ep in episodes]
        ),
        "first_private_barrier_stage_distribution": distribution_with_fraction(
            [ep["first_private_barrier_stage"] for ep in episodes]
        ),
        "mean_candidate_count_per_stage": mean_vector(
            [ep["candidate_count_per_stage"] for ep in episodes]
        ),
        "mean_legal_child_count_per_stage": mean_vector(
            [ep["legal_child_count_per_stage"] for ep in episodes]
        ),
        "specialist_task_count": sum(1 for ep in episodes if ep["is_specialist_task"]),
        "specialist_task_unshared_fraction": mean(
            [float(ep["selected_unshared_path"]) for ep in episodes if ep["is_specialist_task"]]
        ),
        "stage_source_summary": {k: dict(v) for k, v in stage_source_summary.items()},
        "reasoning_cost_mode_default": next(
            (ep["reasoning_cost_mode_default"] for ep in episodes if ep.get("reasoning_cost_mode_default")),
            None,
        ),
    }


def build_specialist_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    specialist = [ep for ep in episodes if ep["is_specialist_task"]]
    return {
        "specialist_episode_count": len(specialist),
        "specialist_shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in specialist]),
        "specialist_unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in specialist]),
        "specialist_exact_match_mean": mean([float(ep["exact_match"]) for ep in specialist]),
        "specialist_total_cost_mean": mean([ep["total_cost"] for ep in specialist]),
        "specialist_raw_outcome_penalty_mean": mean(
            [ep["raw_outcome_penalty"] for ep in specialist]
        ),
        "specialist_raw_policy_penalty_mean": mean(
            [ep["raw_policy_penalty"] for ep in specialist]
        ),
        "specialist_raw_terminal_penalty_mean": mean([ep["raw_terminal_penalty"] for ep in specialist]),
        "specialist_raw_path_cost_component_mean": mean([ep["raw_path_cost_component"] for ep in specialist]),
        "specialist_raw_reasoning_cost_component_mean": mean([ep["raw_reasoning_cost_component"] for ep in specialist]),
        "specialist_raw_reasoning_cost_component_api_mean": mean_present(
            [ep["raw_reasoning_cost_component_api"] for ep in specialist]
        ),
        "specialist_raw_reasoning_cost_component_token_mean": mean_present(
            [ep["raw_reasoning_cost_component_token"] for ep in specialist]
        ),
        "specialist_task_ids": sorted({ep["original_task_id"] for ep in specialist}),
    }


def build_partial_summary(
    *,
    method: str,
    dataset: str,
    repeats: int,
    model: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    total_episodes: int,
    status: str = "running",
) -> dict[str, Any]:
    summary = build_summary(
        method=method,
        dataset=dataset,
        repeats=repeats,
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
    )
    summary.update(
        {
            "scheduled_episodes": total_episodes,
            "completed_episodes": len(episodes),
            "status": status,
            "completed_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
            "completed_raw_terminal_penalty": sum(ep["raw_terminal_penalty"] for ep in episodes),
            "completed_raw_policy_penalty": sum(ep["raw_policy_penalty"] for ep in episodes),
            "completed_total_tokens": sum(ep["total_tokens_total"] for ep in episodes),
            "completed_api_cost_usd_raw": sum(
                ep["api_cost_total_usd_raw"] for ep in episodes
            ),
        }
    )
    return summary


def build_risky_dynamics_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": ep["episode_index"],
            "repeat_index": ep["repeat_index"],
            "position_in_cycle": ep["position_in_cycle"],
            "dataset_index": ep["dataset_index"],
            "original_task_id": ep["original_task_id"],
            "is_specialist_task": ep["is_specialist_task"],
            "selected_shared_path": ep["selected_shared_path"],
            "selected_unshared_path": ep["selected_unshared_path"],
            "cumulative_shared_path_ratio": ep["cumulative_shared_path_ratio"],
            "rolling_shared_path_ratio_last10": ep["rolling_shared_path_ratio_last10"],
            "shared_branch_triggered": ep["shared_branch_triggered"],
            "unshared_branch_triggered": ep["unshared_branch_triggered"],
            "shared_update_count": ep["shared_update_count"],
            "cumulative_shared_update_count": ep["cumulative_shared_update_count"],
            "unshared_edge_update_count": ep["unshared_edge_update_count"],
            "cumulative_unshared_edge_update_count": ep["cumulative_unshared_edge_update_count"],
            "selected_path": ep["selected_path"],
            "selected_shared_path_nodes": ep["selected_path"] if ep["selected_shared_path"] else [],
            "selected_unshared_path_nodes": ep["selected_path"] if ep["selected_unshared_path"] else [],
            "raw_terminal_penalty": ep["raw_terminal_penalty"],
            "total_cost": ep["total_cost"],
        }
        for ep in episodes
    ]


def summarize_window(episodes: list[dict[str, Any]], *, label: str, start: int, end: int) -> dict[str, Any]:
    window = episodes[start:end]
    return {
        "label": label,
        "start_episode_index": start,
        "end_episode_index_exclusive": end,
        "episode_count": len(window),
        "shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in window]),
        "unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in window]),
        "mean_shared_update_count_per_episode": mean([ep["shared_update_count"] for ep in window]),
        "mean_raw_terminal_penalty": mean([ep["raw_terminal_penalty"] for ep in window]),
        "mean_total_cost": mean([ep["total_cost"] for ep in window]),
    }


def build_risky_dynamics_payload(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": build_risky_dynamics_rows(episodes),
        "window_summaries": {
            "first20": summarize_window(episodes, label="first20", start=0, end=20),
            "middle20": summarize_window(episodes, label="middle20", start=40, end=60),
            "last20": summarize_window(episodes, label="last20", start=80, end=100),
        },
    }


def build_compare_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "method": summary["method"],
            "total_cost_mean": summary["total_cost_mean"],
            "raw_total_cost_mean": summary["raw_total_cost_mean"],
            "raw_total_cost_api_mean": summary["raw_total_cost_api_mean"],
            "raw_total_cost_token_mean": summary["raw_total_cost_token_mean"],
            "raw_outcome_penalty_mean": summary["raw_outcome_penalty_mean"],
            "raw_policy_penalty_mean": summary["raw_policy_penalty_mean"],
            "raw_terminal_penalty_mean": summary["raw_terminal_penalty_mean"],
            "raw_path_cost_component_mean": summary["raw_path_cost_component_mean"],
            "raw_reasoning_cost_component_mean": summary["raw_reasoning_cost_component_mean"],
            "raw_reasoning_cost_component_api_mean": summary[
                "raw_reasoning_cost_component_api_mean"
            ],
            "raw_reasoning_cost_component_token_mean": summary[
                "raw_reasoning_cost_component_token_mean"
            ],
            "exact_match_mean": summary["exact_match_mean"],
            "mean_llm_call_count": summary["mean_llm_call_count"],
            "mean_total_tokens": summary["mean_total_tokens"],
            "mean_api_cost_usd_raw": summary["mean_api_cost_usd_raw"],
            "mean_generation_time_seconds": summary["mean_generation_time_seconds"],
            "mean_episode_wall_clock_seconds": summary[
                "mean_episode_wall_clock_seconds"
            ],
        }
        for summary in summaries
    ]
    return sorted(rows, key=lambda row: (row["total_cost_mean"], row["method"]))


def compare_rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(f"{row[field]}" for field in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def build_specialist_hit_analysis(
    *,
    merged_episodes_by_method: dict[str, list[dict[str, Any]]],
    specialist_task_ids: set[str],
    schedule_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    specialist_schedule_rows = [
        row for row in schedule_rows if row["original_task_id"] in specialist_task_ids
    ]
    specialist_episode_count = len(specialist_schedule_rows)
    specialist_task_hit_ids = sorted({row["original_task_id"] for row in specialist_schedule_rows})
    payload: dict[str, Any] = {
        "specialist_episode_count": specialist_episode_count,
        "specialist_task_hit_ids": specialist_task_hit_ids,
        "schedule_episode_indices": [row["episode_index"] for row in specialist_schedule_rows],
    }
    if specialist_episode_count == 0:
        payload["methods"] = {}
        return payload

    method_payload: dict[str, Any] = {}
    for method, episodes in merged_episodes_by_method.items():
        specialist_eps = [ep for ep in episodes if ep["is_specialist_task"]]
        method_payload[method] = {
            "specialist_episode_count": len(specialist_eps),
            "specialist_unshared_path_fraction": mean(
                [float(ep["selected_unshared_path"]) for ep in specialist_eps]
            ),
            "specialist_shared_path_fraction": mean(
                [float(ep["selected_shared_path"]) for ep in specialist_eps]
            ),
            "specialist_total_cost_mean": mean([ep["total_cost"] for ep in specialist_eps]),
            "specialist_raw_terminal_penalty_mean": mean(
                [ep["raw_terminal_penalty"] for ep in specialist_eps]
            ),
        }
    payload["methods"] = method_payload
    return payload


def ensure_model_env(required: bool = True) -> str:
    model_name = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "")
    if required and model_name != MODEL_REQUIRED:
        raise SystemExit(f"PSAGENT_LLM_BENCH_MODEL must be {MODEL_REQUIRED!r}; got {model_name!r}")
    return model_name


def load_run_context(run_dir: Path) -> dict[str, Any]:
    run_config = load_json(run_dir / "run_config.json")
    schedule_rows = load_json(run_dir / "schedule.json")
    oracle_summary = load_json(run_dir / "stationary_oracle_summary.json")
    instances = load_instances(Path(run_config["dataset"]))
    selected = materialize_schedule(instances, schedule_rows)
    specialist_task_ids = load_specialist_task_ids()
    return {
        "run_config": run_config,
        "schedule_rows": schedule_rows,
        "oracle_summary": oracle_summary,
        "selected": selected,
        "specialist_task_ids": specialist_task_ids,
    }


def initialize_run(
    *,
    data_path: Path,
    output_dir: Path,
    repeats: int,
    methods: list[str],
    model_name: str,
) -> Path:
    validate_methods(methods)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    schedule_path = output_dir / "schedule.json"
    oracle_path = output_dir / "stationary_oracle_summary.json"
    specialist_path = output_dir / "specialist_task_ids.json"

    if run_config_path.exists() and schedule_path.exists() and oracle_path.exists():
        return output_dir

    instances = load_instances(data_path)
    selected = build_repeated_selection(instances, indices=SMOKE10_INDICES, repeats=repeats)
    oracle_summary = compute_stationary_oracle(selected)
    schedule_rows = serialize_schedule(selected)
    specialist_task_ids = sorted(load_specialist_task_ids())

    write_json(
        run_config_path,
        {
            "created_at": datetime.now().isoformat(),
            "dataset": str(data_path),
            "dataset_indices": SMOKE10_INDICES,
            "repeats": repeats,
            "horizon": len(selected),
            "family_kind": FAMILY_KIND,
            "executor_name": EXECUTOR_NAME,
            "model": model_name,
            "seed": SEED,
            "methods": methods,
            "parallelism": "method_only",
        },
    )
    write_json(schedule_path, schedule_rows)
    write_json(oracle_path, oracle_summary)
    write_json(specialist_path, specialist_task_ids)
    return output_dir


def build_progress_payload(
    *,
    method: str,
    completed_count: int,
    total_episodes: int,
    model: str,
    status: str,
) -> dict[str, Any]:
    last_completed = completed_count - 1 if completed_count else None
    return {
        "method": method,
        "scheduled_episodes": total_episodes,
        "completed_episodes": completed_count,
        "last_completed_episode_index": last_completed,
        "status": status,
        "model": model,
        "updated_at": datetime.now().isoformat(),
    }


def persist_method_state(
    *,
    method_dir: Path,
    method: str,
    episodes: list[dict[str, Any]],
    policy: Any | None,
    total_episodes: int,
    model: str,
    dataset: str,
    repeats: int,
    oracle_summary: dict[str, Any],
) -> None:
    add_cumulative_fields(episodes)
    checkpoint_payload = {
        "method": method,
        "completed_count": len(episodes),
        "episodes": episodes,
        "model": model,
        "policy": policy,
    }
    write_bytes_atomic(method_dir / "checkpoint.pkl", pickle.dumps(checkpoint_payload))
    write_jsonl(method_dir / "episodes.partial.jsonl", episodes)
    write_json(
        method_dir / "progress.json",
        build_progress_payload(
            method=method,
            completed_count=len(episodes),
            total_episodes=total_episodes,
            model=model,
            status="complete" if len(episodes) == total_episodes else "running",
        ),
    )
    partial_summary = build_partial_summary(
        method=method,
        dataset=dataset,
        repeats=repeats,
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
        total_episodes=total_episodes,
        status="complete" if len(episodes) == total_episodes else "running",
    )
    write_json(method_dir / "summary_partial.json", partial_summary)
    if len(episodes) == total_episodes:
        write_json(method_dir / "episodes.json", episodes)
        write_json(method_dir / "summary.json", partial_summary)
        write_json(method_dir / "summary_with_oracle.json", partial_summary)


def load_method_checkpoint(method_dir: Path) -> dict[str, Any] | None:
    checkpoint_path = method_dir / "checkpoint.pkl"
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def run_policy_method(
    *,
    run_dir: Path,
    method: str,
) -> None:
    ensure_model_env(required=True)
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    selected = context["selected"]
    oracle_summary = context["oracle_summary"]
    specialist_task_ids = context["specialist_task_ids"]
    total_episodes = len(selected)
    method_dir = run_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(executor_name=EXECUTOR_NAME)
    checkpoint = load_method_checkpoint(method_dir)
    if checkpoint is not None:
        policy = checkpoint["policy"]
        episodes = list(checkpoint["episodes"])
        model = checkpoint.get("model", getattr(env.family_executor, "model", MODEL_REQUIRED))
    else:
        episodes = []
        model = getattr(env.family_executor, "model", MODEL_REQUIRED)
        policy = POLICY_REGISTRY[method](seed=SEED)
        policy.bind_env(env)
        policy.reset()

    completed_count = len(episodes)
    if completed_count >= total_episodes:
        persist_method_state(
            method_dir=method_dir,
            method=method,
            episodes=episodes,
            policy=policy,
            total_episodes=total_episodes,
            model=model,
            dataset=run_config["dataset"],
            repeats=int(run_config["repeats"]),
            oracle_summary=oracle_summary,
        )
        return

    for local_offset in range(completed_count, total_episodes):
        row = selected[local_offset]
        episode_index = int(row["episode_index"])
        print(
            f"[run] method={method} episode={episode_index + 1}/{len(selected)} "
            f"repeat={row['repeat_index'] + 1} pos={row['position_in_cycle']} dataset_index={row['dataset_index']}",
            flush=True,
        )
        path = policy.select_path(row["instance"], env)
        selection_info = policy.get_last_selection_info() if hasattr(policy, "get_last_selection_info") else {}
        env.reset(row["instance"])
        result = env.run_path(path)
        policy.update(result)
        state = policy.get_state() if hasattr(policy, "get_state") else {}
        update_info = state.get("last_update_info", {}) if isinstance(state, dict) else {}
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                row=row,
                result=result,
                method=method,
                oracle_summary=oracle_summary,
                selection_info=selection_info if isinstance(selection_info, dict) else {},
                update_info=update_info if isinstance(update_info, dict) else {},
                specialist_task_ids=specialist_task_ids,
            )
        )
        persist_method_state(
            method_dir=method_dir,
            method=method,
            episodes=episodes,
            policy=policy,
            total_episodes=total_episodes,
            model=model,
            dataset=run_config["dataset"],
            repeats=int(run_config["repeats"]),
            oracle_summary=oracle_summary,
        )


def merge_method_results(run_dir: Path, method: str) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    oracle_summary = context["oracle_summary"]
    specialist_task_ids = context["specialist_task_ids"]
    total_episodes = int(run_config["horizon"])
    method_dir = run_dir / method
    model = run_config["model"]
    progress = load_json(method_dir / "progress.json")
    if progress["completed_episodes"] != total_episodes:
        raise RuntimeError(f"Method {method} is incomplete: {progress}")
    merged_episodes = load_json(method_dir / "episodes.json")
    model = progress.get("model", model)

    expected_indices = list(range(total_episodes))
    actual_indices = [int(row["episode_index"]) for row in merged_episodes]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"Merged episode indices mismatch for {method}. "
            f"expected={expected_indices[:3]}...{expected_indices[-3:]}, "
            f"actual={actual_indices[:3]}...{actual_indices[-3:]}"
        )
    add_cumulative_fields(merged_episodes)
    summary = build_summary(
        method=method,
        dataset=run_config["dataset"],
        repeats=int(run_config["repeats"]),
        model=model,
        oracle_summary=oracle_summary,
        episodes=merged_episodes,
    )
    specialist_summary = build_specialist_summary(merged_episodes)

    merged_dir = method_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    write_json(method_dir / "episodes.json", merged_episodes)
    write_json(method_dir / "summary.json", summary)
    write_json(method_dir / "summary_with_oracle.json", summary)
    write_json(method_dir / "specialist_summary.json", specialist_summary)
    write_json(merged_dir / "episodes.json", merged_episodes)
    write_json(merged_dir / "summary.json", summary)
    write_json(merged_dir / "summary_with_oracle.json", summary)
    write_json(merged_dir / "specialist_summary.json", specialist_summary)
    write_text_atomic(
        method_dir / "smoke_summary.md",
        json.dumps({"summary": summary, "specialist_summary": specialist_summary}, ensure_ascii=False, indent=2),
    )

    if method == "risky_ps":
        dynamics_payload = build_risky_dynamics_payload(merged_episodes)
        write_json(run_dir / "risky_ps_shared_unshared_dynamics.json", dynamics_payload)
        write_csv(run_dir / "risky_ps_shared_unshared_dynamics.csv", dynamics_payload["episodes"])

    return {
        "summary": summary,
        "specialist_summary": specialist_summary,
        "episodes": merged_episodes,
        "specialist_task_ids": specialist_task_ids,
    }


def merge_all_results(run_dir: Path) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    specialist_task_ids = context["specialist_task_ids"]
    summaries: list[dict[str, Any]] = []
    merged_episodes_by_method: dict[str, list[dict[str, Any]]] = {}

    for method in run_config["methods"]:
        method_summary = load_json(run_dir / method / "summary_with_oracle.json")
        summaries.append(method_summary)
        merged_episodes_by_method[method] = load_json(run_dir / method / "episodes.json")

    compare_rows = build_compare_rows(summaries)
    write_json(run_dir / "repeated_smoke_compare.json", compare_rows)
    write_csv(run_dir / "repeated_smoke_compare.csv", compare_rows)
    write_text_atomic(run_dir / "repeated_smoke_compare.md", compare_rows_to_markdown(compare_rows))

    specialist_payload = build_specialist_hit_analysis(
        merged_episodes_by_method=merged_episodes_by_method,
        specialist_task_ids=specialist_task_ids,
        schedule_rows=context["schedule_rows"],
    )
    write_json(run_dir / "specialist_unshared_hit_analysis.json", specialist_payload)
    return {
        "compare_rows": compare_rows,
        "specialist_analysis": specialist_payload,
        "merged_episodes_by_method": merged_episodes_by_method,
    }


def orchestrate_run(
    *,
    data_path: Path,
    output_dir: Path,
    repeats: int,
    methods: list[str],
) -> Path:
    model_name = ensure_model_env(required=True)
    validate_methods(methods)
    run_dir = initialize_run(
        data_path=data_path,
        output_dir=output_dir,
        repeats=repeats,
        methods=methods,
        model_name=model_name,
    )
    script_path = Path(__file__).resolve()
    launched: list[tuple[str, subprocess.Popen[Any], Any]] = []

    for method in methods:
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        log_path = method_dir / "runner.log"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"[launch] {datetime.now().isoformat()} method={method}\n")
        log_handle.flush()
        cmd = [
            sys.executable,
            str(script_path),
            "run-method",
            "--run-dir",
            str(run_dir),
            "--method",
            method,
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        launched.append((method, process, log_handle))

    failures: list[dict[str, Any]] = []
    for method, process, log_handle in launched:
        return_code = process.wait()
        log_handle.write(
            f"[exit] {datetime.now().isoformat()} method={method} return_code={return_code}\n"
        )
        log_handle.close()
        if return_code != 0:
            failures.append({"method": method, "return_code": return_code})

    if failures:
        write_json(run_dir / "orchestrator_failures.json", failures)
        raise SystemExit(f"One or more method runs failed: {failures}")

    for method in methods:
        merge_method_results(run_dir, method)
    merge_all_results(run_dir)
    return run_dir


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated shared-basin smoke with method-level persistence.")
    subparsers = parser.add_subparsers(dest="command")

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--data", type=Path, default=DATASET_DEFAULT)
    common_run.add_argument("--output-dir", type=Path, required=True)
    common_run.add_argument("--repeats", type=int, default=10)
    common_run.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))

    setup_parser = subparsers.add_parser("setup", parents=[common_run])
    setup_parser.set_defaults(command="setup")

    orchestrate_parser = subparsers.add_parser("orchestrate", parents=[common_run])
    orchestrate_parser.set_defaults(command="orchestrate")

    method_parser = subparsers.add_parser("run-method")
    method_parser.add_argument("--run-dir", type=Path, required=True)
    method_parser.add_argument("--method", type=str, required=True)
    method_parser.set_defaults(command="run-method")

    merge_method_parser = subparsers.add_parser("merge-method")
    merge_method_parser.add_argument("--run-dir", type=Path, required=True)
    merge_method_parser.add_argument("--method", type=str, required=True)
    merge_method_parser.set_defaults(command="merge-method")

    merge_all_parser = subparsers.add_parser("merge-all")
    merge_all_parser.add_argument("--run-dir", type=Path, required=True)
    merge_all_parser.set_defaults(command="merge-all")
    return parser


def main() -> None:
    parser = build_cli()
    argv = sys.argv[1:]
    known_commands = {"setup", "orchestrate", "run-method", "merge-method", "merge-all"}
    if not argv or argv[0] not in known_commands:
        argv = ["orchestrate", *argv]
    args = parser.parse_args(argv)

    if args.command == "setup":
        model_name = ensure_model_env(required=True)
        validate_methods(args.methods)
        run_dir = initialize_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            methods=args.methods,
            model_name=model_name,
        )
        print(str(run_dir))
        return

    if args.command == "orchestrate":
        validate_methods(args.methods)
        run_dir = orchestrate_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            methods=args.methods,
        )
        print(str(run_dir))
        return

    if args.command == "run-method":
        if args.method not in POLICY_REGISTRY:
            raise SystemExit(f"Unknown method for run-method: {args.method}")
        run_policy_method(run_dir=args.run_dir, method=args.method)
        print(str(args.run_dir / args.method))
        return

    if args.command == "merge-method":
        payload = merge_method_results(args.run_dir, args.method)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "total_cost_mean": payload["summary"]["total_cost_mean"],
                    "shared_path_fraction": payload["summary"]["shared_path_fraction"],
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "merge-all":
        payload = merge_all_results(args.run_dir)
        print(json.dumps(payload["compare_rows"], ensure_ascii=False))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
