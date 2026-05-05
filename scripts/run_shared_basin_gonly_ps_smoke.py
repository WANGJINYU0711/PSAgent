"""Run Risky-PS repeated smoke for shared-basin g-only variants.

This runner is intentionally separate from ``run_shared_basin_repeated_smoke.py``
so the existing ``shared_basin_strong`` workflow and resume layout stay
untouched. It is code-only infrastructure; invoking ``run`` or ``orchestrate``
will execute the LLM benchmark.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import statistics
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
from oracle_eval import find_best_stationary_path  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402


SMOKE10_INDICES = list(range(10))
DATASET_DEFAULT = (
    ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time" / "tasks.json"
)
BASE_FAMILY_KIND = "shared_basin_strong"
VARIANT_TYPE = "g_only"
FAMILY_KIND_CHOICES = [
    "shared_basin_strong_2of5_gonly",
    "shared_basin_strong_all_share_gonly",
    "shared_basin_strong_all_unshare_gonly",
]
FAMILY_KIND_DEFAULT = "shared_basin_strong_2of5_gonly"
METHOD = "risky_ps"
MODEL_REQUIRED = "gpt-4o-mini"
SEED = 0
EXECUTOR_NAME_DEFAULT = "llm_bench"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def mean_present(values: list[Any]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric)


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


def load_instances(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def ensure_model_env(required: bool = True) -> str:
    model_name = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "")
    if required and model_name != MODEL_REQUIRED:
        raise SystemExit(f"PSAGENT_LLM_BENCH_MODEL must be {MODEL_REQUIRED!r}; got {model_name!r}")
    return model_name


def build_env(*, family_kind: str, executor_name: str) -> FixedTreeEnvironment:
    return FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=family_kind,
        family_seed=SEED,
        executor_name=executor_name,
    )


def build_repeated_selection(
    instances: list[dict[str, Any]],
    *,
    indices: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        for position_in_cycle, dataset_index in enumerate(indices):
            selected.append(
                {
                    "episode_index": len(selected),
                    "repeat_index": repeat_index,
                    "position_in_cycle": position_in_cycle,
                    "dataset_index": dataset_index,
                    "instance": instances[dataset_index],
                }
            )
    return selected


def serialize_schedule(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in selected:
        instance = row["instance"]
        rows.append(
            {
                "episode_index": row["episode_index"],
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


def compute_stationary_oracle(selected: list[dict[str, Any]], *, family_kind: str) -> dict[str, Any]:
    oracle_env = build_env(family_kind=family_kind, executor_name="simulated")
    oracle_path, raw = find_best_stationary_path([row["instance"] for row in selected], oracle_env)
    return {
        "path": list(oracle_path),
        "episode_total_costs": list(raw["episode_total_costs"]),
        "episode_raw_total_costs": list(raw["episode_raw_total_costs"]),
        "episode_normalized_total_costs": list(raw["episode_normalized_total_costs"]),
        "cumulative_total_cost": float(raw["cumulative_total_cost"]),
        "raw_cumulative_total_cost": float(raw["raw_cumulative_total_cost"]),
        "normalized_cumulative_total_cost": float(raw["normalized_cumulative_total_cost"]),
        "mean_total_cost": float(raw["mean_total_cost"]),
        "raw_mean_total_cost": float(raw["raw_mean_total_cost"]),
        "normalized_mean_total_cost": float(raw["normalized_mean_total_cost"]),
        "cost_scale_version": str(raw["cost_scale_version"]),
    }


def load_run_context(run_dir: Path) -> dict[str, Any]:
    run_config = load_json(run_dir / "run_config.json")
    schedule_rows = load_json(run_dir / "schedule.json")
    oracle_summary = load_json(run_dir / "stationary_oracle_summary.json")
    instances = load_instances(Path(run_config["dataset"]))
    return {
        "run_config": run_config,
        "schedule_rows": schedule_rows,
        "oracle_summary": oracle_summary,
        "selected": materialize_schedule(instances, schedule_rows),
    }


def flatten_episode(
    *,
    row: dict[str, Any],
    result: Any,
    selection_info: dict[str, Any],
    update_info: dict[str, Any],
) -> dict[str, Any]:
    instance = row["instance"]
    episode_index = int(row["episode_index"])
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
    shared_updates = update_info.get("shared_safe_suffix_edges_updated", []) or []
    risky_updates = update_info.get("risky_edges_updated", []) or []
    stage_prompt_tokens = [
        float(stage_trace.get(stage_name, {}).get("prompt_tokens_total_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_completion_tokens = [
        float(stage_trace.get(stage_name, {}).get("completion_tokens_total_stage", 0.0) or 0.0)
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
        float(stage_trace.get(stage_name, {}).get("generation_time_total_seconds_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_llm_round_trip_seconds = [
        float(stage_trace.get(stage_name, {}).get("llm_round_trip_total_seconds_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_wall_clock_seconds = [
        float(stage_trace.get(stage_name, {}).get("stage_wall_clock_seconds", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_tool_wall_clock_seconds = [
        float(stage_trace.get(stage_name, {}).get("tool_wall_clock_total_seconds_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    raw_total_cost_api = (
        float(log.get("raw_total_cost_api"))
        if log.get("raw_total_cost_api") is not None
        else None
    )
    raw_total_cost_token = (
        float(log.get("raw_total_cost_token"))
        if log.get("raw_total_cost_token") is not None
        else None
    )
    raw_reasoning_cost_component_api = (
        float(log.get("raw_reasoning_cost_component_api"))
        if log.get("raw_reasoning_cost_component_api") is not None
        else None
    )
    raw_reasoning_cost_component_token = (
        float(log.get("raw_reasoning_cost_component_token"))
        if log.get("raw_reasoning_cost_component_token") is not None
        else None
    )
    stage5_trace = stage_trace.get("stage5", {})
    leaf_type = str(result.leaf_type)
    shared_path = leaf_type == "shared"
    return {
        "method": METHOD,
        "episode_index": episode_index,
        "repeat_index": row["repeat_index"],
        "position_in_cycle": row["position_in_cycle"],
        "dataset_index": row["dataset_index"],
        "instance_id": instance["instance_id"],
        "original_task_id": instance["original_task_id"],
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
        "raw_total_cost_api": raw_total_cost_api,
        "raw_total_cost_token": raw_total_cost_token,
        "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
        "raw_reasoning_cost_component_api": raw_reasoning_cost_component_api,
        "raw_reasoning_cost_component_token": raw_reasoning_cost_component_token,
        "reasoning_cost": float(result.reasoning_cost),
        "reasoning_cost_mode_default": log.get("reasoning_cost_mode_default"),
        "policy_eval_source": log.get("policy_eval_source"),
        "policy_eval_scope": log.get("policy_eval_scope"),
        "terminal_cost_upper_bound": log.get("terminal_cost_upper_bound"),
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
    run_config: dict[str, Any],
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    status: str,
) -> dict[str, Any]:
    stage_source_summary: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        for stage_name, source in episode["stage_sources"].items():
            stage_source_summary[stage_name][str(source)] += 1
    policy_nl_total = sum(ep["policy_nl_assertions_total"] for ep in episodes)
    policy_nl_failed = sum(ep["policy_nl_assertions_failed"] for ep in episodes)
    shared_update_total = sum(ep["shared_update_count"] for ep in episodes)
    unshared_edge_update_total = sum(ep["unshared_edge_update_count"] for ep in episodes)
    return {
        "test_name": f"{METHOD}_smoke10x{run_config['repeats']}_{run_config['family_kind']}_full_llm",
        "dataset": run_config["dataset"],
        "dataset_indices": run_config["dataset_indices"],
        "repeats": run_config["repeats"],
        "horizon": run_config["horizon"],
        "scheduled_episodes": run_config["horizon"],
        "completed_episodes": len(episodes),
        "status": status,
        "method": METHOD,
        "mechanism": "algorithm_direct",
        "executor_name": run_config["executor_name"],
        "family_kind": run_config["family_kind"],
        "base_family_kind": BASE_FAMILY_KIND,
        "variant_type": VARIANT_TYPE,
        "seed": run_config["seed"],
        "model": run_config["model"],
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
        "algorithm_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
        "raw_algorithm_cumulative_total_cost": sum(ep["raw_total_cost"] for ep in episodes),
        "oracle_stationary_total_cost": oracle_summary["cumulative_total_cost"],
        "raw_oracle_stationary_total_cost": oracle_summary["raw_cumulative_total_cost"],
        "raw_outcome_penalty_cumulative": sum(ep["raw_outcome_penalty"] for ep in episodes),
        "raw_policy_penalty_cumulative": sum(ep["raw_policy_penalty"] for ep in episodes),
        "raw_terminal_penalty_cumulative": sum(ep["raw_terminal_penalty"] for ep in episodes),
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
        "episodes_with_stage5_verification_tools": sum(
            1 for ep in episodes if ep["stage5_executed_tool_names"]
        ),
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
        "shared_update_count_total": shared_update_total,
        "unshared_edge_update_count_total": unshared_edge_update_total,
        "shared_update_episode_fraction": mean(
            [float(ep["shared_update_count"] > 0) for ep in episodes]
        ),
        "shared_branch_triggered_fraction": mean(
            [float(ep["shared_branch_triggered"]) for ep in episodes]
        ),
        "unshared_branch_triggered_fraction": mean(
            [float(ep["unshared_branch_triggered"]) for ep in episodes]
        ),
        "stage_source_summary": {key: dict(value) for key, value in stage_source_summary.items()},
        "reasoning_cost_mode_default": next(
            (
                ep["reasoning_cost_mode_default"]
                for ep in episodes
                if ep.get("reasoning_cost_mode_default")
            ),
            None,
        ),
        "cost_scale_version": episodes[0]["cost_scale_version"] if episodes else oracle_summary["cost_scale_version"],
        "updated_at": datetime.now().isoformat(),
    }


def persist_state(
    *,
    run_dir: Path,
    episodes: list[dict[str, Any]],
    policy: RiskyPSPolicy,
    run_config: dict[str, Any],
    oracle_summary: dict[str, Any],
    status: str,
) -> None:
    add_cumulative_fields(episodes)
    write_bytes_atomic(
        run_dir / "checkpoint.pkl",
        pickle.dumps(
            {
                "method": METHOD,
                "completed_count": len(episodes),
                "episodes": episodes,
                "policy": policy,
                "model": run_config["model"],
            }
        ),
    )
    write_jsonl(run_dir / "episodes.partial.jsonl", episodes)
    write_json(
        run_dir / "progress.json",
        {
            "method": METHOD,
            "scheduled_episodes": run_config["horizon"],
            "completed_episodes": len(episodes),
            "last_completed_episode_index": len(episodes) - 1 if episodes else None,
            "status": status,
            "model": run_config["model"],
            "updated_at": datetime.now().isoformat(),
        },
    )
    summary = build_summary(
        run_config=run_config,
        oracle_summary=oracle_summary,
        episodes=episodes,
        status=status,
    )
    write_json(run_dir / "summary_partial.json", summary)
    if status == "complete":
        write_json(run_dir / "episodes.json", episodes)
        write_json(run_dir / "summary.json", summary)
        write_json(run_dir / "summary_with_oracle.json", summary)


def initialize_run(
    *,
    data_path: Path,
    output_dir: Path,
    repeats: int,
    family_kind: str,
    executor_name: str,
    model_name: str,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    schedule_path = output_dir / "schedule.json"
    oracle_path = output_dir / "stationary_oracle_summary.json"
    if run_config_path.exists() and schedule_path.exists() and oracle_path.exists():
        return output_dir

    instances = load_instances(data_path)
    selected = build_repeated_selection(instances, indices=SMOKE10_INDICES, repeats=repeats)
    oracle_summary = compute_stationary_oracle(selected, family_kind=family_kind)
    schedule_rows = serialize_schedule(selected)
    run_config = {
        "created_at": datetime.now().isoformat(),
        "dataset": str(data_path),
        "dataset_indices": SMOKE10_INDICES,
        "repeats": repeats,
        "horizon": len(selected),
        "family_kind": family_kind,
        "base_family_kind": BASE_FAMILY_KIND,
        "variant_type": VARIANT_TYPE,
        "method": METHOD,
        "mechanism": "algorithm_direct",
        "executor_name": executor_name,
        "model": model_name,
        "seed": SEED,
    }
    write_json(run_config_path, run_config)
    write_json(schedule_path, schedule_rows)
    write_json(oracle_path, oracle_summary)
    return output_dir


def load_checkpoint(run_dir: Path) -> dict[str, Any] | None:
    checkpoint_path = run_dir / "checkpoint.pkl"
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def run_policy(run_dir: Path) -> None:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    selected = context["selected"]
    oracle_summary = context["oracle_summary"]
    env = build_env(
        family_kind=run_config["family_kind"],
        executor_name=run_config["executor_name"],
    )
    checkpoint = load_checkpoint(run_dir)
    if checkpoint is not None:
        policy = checkpoint["policy"]
        episodes = list(checkpoint["episodes"])
    else:
        policy = RiskyPSPolicy(seed=SEED)
        policy.bind_env(env)
        policy.reset()
        episodes = []

    for offset in range(len(episodes), len(selected)):
        row = selected[offset]
        path = policy.select_path(row["instance"], env)
        selection_info = policy.get_last_selection_info()
        env.reset(row["instance"])
        result = env.run_path(path)
        policy.update(result)
        state = policy.get_state()
        update_info = state.get("last_update_info", {}) if isinstance(state, dict) else {}
        episodes.append(
            flatten_episode(
                row=row,
                result=result,
                selection_info=selection_info,
                update_info=update_info if isinstance(update_info, dict) else {},
            )
        )
        persist_state(
            run_dir=run_dir,
            episodes=episodes,
            policy=policy,
            run_config=run_config,
            oracle_summary=oracle_summary,
            status="complete" if len(episodes) == len(selected) else "running",
        )


def merge_run(run_dir: Path) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    oracle_summary = context["oracle_summary"]
    checkpoint = load_checkpoint(run_dir)
    episodes = list(checkpoint["episodes"]) if checkpoint is not None else []
    if len(episodes) != int(run_config["horizon"]):
        raise RuntimeError(
            f"Run is incomplete: completed={len(episodes)} horizon={run_config['horizon']}"
        )
    add_cumulative_fields(episodes)
    summary = build_summary(
        run_config=run_config,
        oracle_summary=oracle_summary,
        episodes=episodes,
        status="complete",
    )
    write_json(run_dir / "episodes.json", episodes)
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "summary_with_oracle.json", summary)
    return summary


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Risky-PS smoke on shared-basin g-only variants.")
    subparsers = parser.add_subparsers(dest="command")

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--data", type=Path, default=DATASET_DEFAULT)
    common.add_argument("--output-dir", type=Path, required=True)
    common.add_argument("--repeats", type=int, default=10)
    common.add_argument(
        "--family-kind",
        choices=FAMILY_KIND_CHOICES,
        default=FAMILY_KIND_DEFAULT,
    )
    common.add_argument("--executor-name", choices=["llm_bench"], default=EXECUTOR_NAME_DEFAULT)

    setup_parser = subparsers.add_parser("setup", parents=[common])
    setup_parser.set_defaults(command="setup")

    orchestrate_parser = subparsers.add_parser("orchestrate", parents=[common])
    orchestrate_parser.set_defaults(command="orchestrate")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--run-dir", type=Path, required=True)
    run_parser.set_defaults(command="run")

    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--run-dir", type=Path, required=True)
    merge_parser.set_defaults(command="merge")
    return parser


def main() -> None:
    parser = build_cli()
    argv = sys.argv[1:]
    known_commands = {"setup", "orchestrate", "run", "merge"}
    if not argv or argv[0] not in known_commands:
        argv = ["orchestrate", *argv]
    args = parser.parse_args(argv)

    if args.command == "setup":
        model_name = ensure_model_env(required=True)
        run_dir = initialize_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            family_kind=args.family_kind,
            executor_name=args.executor_name,
            model_name=model_name,
        )
        print(str(run_dir))
        return

    if args.command == "orchestrate":
        model_name = ensure_model_env(required=True)
        run_dir = initialize_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            family_kind=args.family_kind,
            executor_name=args.executor_name,
            model_name=model_name,
        )
        run_policy(run_dir)
        summary = merge_run(run_dir)
        print(json.dumps(summary, ensure_ascii=False))
        return

    if args.command == "run":
        ensure_model_env(required=True)
        run_policy(args.run_dir)
        print(str(args.run_dir))
        return

    if args.command == "merge":
        summary = merge_run(args.run_dir)
        print(json.dumps(summary, ensure_ascii=False))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
