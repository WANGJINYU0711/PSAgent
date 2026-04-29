#!/usr/bin/env python3
"""Build an execution-calibrated clean telecom MMS profile-switch dataset.

Starting from the local-clean dataset, this runs one real llm_bench family path
per task. The chosen path must use a real family agent whose deliberation_mode
matches the task's fast/deep requirement at every stage. Tasks are retained only
when the resulting terminal signal is clean enough for smoke-test use.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any


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

from fixed_tree_env import FixedTreeEnvironment, leaf_starts_shared_upload  # noqa: E402
from oracle_eval import enumerate_family_paths  # noqa: E402
from tree_family.generator import TreeFamilyGenerator  # noqa: E402


STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5")
DEFAULT_FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch"
DEFAULT_SOURCE_DATA = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_clean_v1/tasks.json"
)
DEFAULT_SOURCE_BUCKETS = (
    ROOT / "analysis/shared_basin_prefix_dedup_profile_switch_local_clean_v1_schedule_buckets.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v1"
)
DEFAULT_OUTPUT_BUCKETS = (
    ROOT / "analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v1_schedule_buckets.json"
)
DEFAULT_REPORT_JSON = (
    ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v1_execution_report.json"
)
DEFAULT_REPORT_CSV = (
    ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v1_execution_report.csv"
)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def stage_modes(task: dict[str, Any]) -> list[str]:
    summary = (task.get("metadata") or {}).get("deliberation_requirement_summary") or {}
    return [
        "deep" if str(summary.get(stage, "fast")).strip().lower() == "deep" else "fast"
        for stage in STAGES
    ]


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


def path_profile(path: tuple[str, ...], agent_map: dict[str, Any]) -> dict[str, Any]:
    agents = [agent_map[agent_id] for agent_id in path]
    route_labels = [str(getattr(agent, "route_label", "")) for agent in agents]
    node_semantics = [str(getattr(agent, "node_semantic", "")) for agent in agents]
    modes = [
        "deep" if str(getattr(agent, "deliberation_mode", "fast")).strip().lower() == "deep" else "fast"
        for agent in agents
    ]
    lanes = [
        lane_kind(route_label, node_semantic)
        for route_label, node_semantic in zip(route_labels, node_semantics)
    ]
    return {
        "path_agent_ids": list(path),
        "agent_modes": modes,
        "route_labels": route_labels,
        "node_semantics": node_semantics,
        "lane_sequence": lanes,
        "base_cost_sum": sum(float(getattr(agent, "base_cost", 0.0) or 0.0) for agent in agents),
        "leaf_type": "shared" if leaf_starts_shared_upload(path, agent_map) else "unshared",
    }


def route_preference_score(required_modes: list[str], profile: dict[str, Any]) -> tuple[float, float, str]:
    lanes = list(profile["lane_sequence"])
    target_count = lanes.count("target_specialist")
    trap_count = lanes.count("trap")
    general_count = lanes.count("general")
    barrier_count = lanes.count("barrier_private")
    deep_count = required_modes.count("deep")
    if deep_count >= 3:
        route_score = (3.0 * target_count) + general_count - (2.0 * trap_count) - (4.0 * barrier_count)
    elif deep_count == 0:
        route_score = (2.0 * trap_count) + general_count - target_count - (4.0 * barrier_count)
    else:
        route_score = general_count + target_count - trap_count - (4.0 * barrier_count)
    shared_bonus = 0.25 if profile["leaf_type"] == "shared" else 0.0
    return (route_score + shared_bonus, -float(profile["base_cost_sum"]), "/".join(profile["path_agent_ids"]))


def choose_mode_matched_path(
    *,
    task: dict[str, Any],
    all_paths: list[tuple[str, ...]],
    agent_map: dict[str, Any],
) -> dict[str, Any] | None:
    required = stage_modes(task)
    candidates: list[dict[str, Any]] = []
    for path in all_paths:
        profile = path_profile(path, agent_map)
        if profile["agent_modes"] == required:
            candidates.append(profile)
    if not candidates:
        return None
    candidates.sort(key=lambda profile: route_preference_score(required, profile), reverse=True)
    chosen = dict(candidates[0])
    chosen["required_modes"] = required
    chosen["mode_match_exact"] = True
    chosen["mode_matched_candidate_count"] = len(candidates)
    chosen["route_preference_score"] = route_preference_score(required, chosen)[0]
    return chosen


def selected_and_deferred_from_stage5(result: Any) -> tuple[list[str], list[str]]:
    stage5 = result.stage_outputs.get("stage5", {}) if isinstance(result.stage_outputs, dict) else {}
    output = stage5.get("output", {}) if isinstance(stage5, dict) else {}
    return (
        [str(value) for value in output.get("selected_blocker_ids", [])],
        [str(value) for value in output.get("deferred_blocker_ids", [])],
    )


def run_task_job(job: dict[str, Any]) -> dict[str, Any]:
    env = FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=job["family_kind"],
        family_seed=int(job["seed"]),
        executor_name="llm_bench",
    )
    model = job.get("model")
    if model and getattr(env.family_executor, "model", None) is not None:
        env.family_executor.model = str(model)

    task = job["task"]
    path_profile_payload = job["path_profile"]
    env.reset(task)
    result = env.run_path(list(path_profile_payload["path_agent_ids"]))
    log = result.episode_log or {}
    selected_ids, deferred_ids = selected_and_deferred_from_stage5(result)
    subset_clean = not bool(log.get("subset_mismatch", False))
    exact = bool(result.success)
    policy_clean = int(log.get("policy_violation_count", 0) or 0) == 0
    clear_success = bool(exact and subset_clean)
    auxiliary_success = bool(policy_clean)
    strict_clean = bool(clear_success and auxiliary_success)
    metadata = task.get("metadata") or {}
    stage_trace = list(log.get("stage_trace", []) or [])
    stage_llm_calls = [
        int(row.get("llm_call_count_stage", 0) or 0)
        for row in stage_trace
        if isinstance(row, dict)
    ]
    stage_json_retries = [
        int(row.get("json_retry_count", 0) or 0)
        for row in stage_trace
        if isinstance(row, dict)
    ]
    return {
        "job_index": int(job["job_index"]),
        "original_task_id": str(task.get("original_task_id")),
        "instance_id": str(task.get("instance_id")),
        "expected_terminal_action": metadata.get("expected_terminal_action"),
        "repairability": metadata.get("repairability"),
        "persona_level": metadata.get("persona_level"),
        "num_blockers": metadata.get("num_blockers"),
        "required_modes": "/".join(path_profile_payload["required_modes"]),
        "agent_modes": "/".join(path_profile_payload["agent_modes"]),
        "mode_match_exact": bool(path_profile_payload["mode_match_exact"]),
        "mode_matched_candidate_count": int(path_profile_payload["mode_matched_candidate_count"]),
        "path_agent_ids": json.dumps(path_profile_payload["path_agent_ids"], ensure_ascii=False),
        "route_labels": " > ".join(path_profile_payload["route_labels"]),
        "lane_sequence": "/".join(path_profile_payload["lane_sequence"]),
        "leaf_type": path_profile_payload["leaf_type"],
        "route_preference_score": float(path_profile_payload["route_preference_score"]),
        "path_base_cost_sum": float(path_profile_payload["base_cost_sum"]),
        "oracle_action": result.oracle_action,
        "final_action": result.final_action,
        "terminal_cost": float(result.raw_terminal_penalty),
        "raw_terminal_penalty": float(result.raw_terminal_penalty),
        "raw_outcome_penalty": float(result.raw_outcome_penalty),
        "raw_policy_penalty": float(result.raw_policy_penalty),
        "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
        "raw_path_cost_component": float(result.raw_path_cost_component),
        "raw_total_cost": float(result.raw_total_cost),
        "exact_match": exact,
        "subset_clean": subset_clean,
        "subset_mismatch": not subset_clean,
        "policy_clean": policy_clean,
        "policy_violation_count": int(log.get("policy_violation_count", 0) or 0),
        "clear_success": clear_success,
        "auxiliary_success": auxiliary_success,
        "strict_clean": strict_clean,
        "selected_blocker_ids": json.dumps(selected_ids, ensure_ascii=False),
        "deferred_blocker_ids": json.dumps(deferred_ids, ensure_ascii=False),
        "prompt_tokens_total": float(result.prompt_tokens_total),
        "completion_tokens_total": float(result.completion_tokens_total),
        "total_tokens_total": float(result.total_tokens_total),
        "llm_call_count": sum(stage_llm_calls),
        "json_retry_count": sum(stage_json_retries),
        "stage_llm_calls": json.dumps(stage_llm_calls, ensure_ascii=False),
        "stage_json_retries": json.dumps(stage_json_retries, ensure_ascii=False),
        "policy_action_violation": bool(log.get("policy_action_violation", False)),
        "policy_communication_violation": bool(log.get("policy_communication_violation", False)),
        "stage5_executed_tool_names": json.dumps(
            [
                call.get("name")
                for row in stage_trace
                if isinstance(row, dict) and row.get("stage_name") == "stage5"
                for call in (row.get("executed_tool_calls", []) or [])
                if isinstance(call, dict)
            ],
            ensure_ascii=False,
        ),
    }


def pass_thresholds(row: dict[str, Any], *, terminal_threshold: float) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if float(row["terminal_cost"]) > terminal_threshold:
        reasons.append(f"terminal_cost_gt_{terminal_threshold:g}")
    if not bool(row["clear_success"]):
        reasons.append("clear_success_false")
    if str(row.get("expected_terminal_action")) in {"repair_all", "repair_subset"} and not bool(
        row["subset_clean"]
    ):
        reasons.append("repair_task_subset_clean_false")
    return (not reasons), reasons


def filter_bucket_ids(ids: list[str], kept_ids: set[str]) -> list[str]:
    return [str(task_id) for task_id in ids if str(task_id) in kept_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--source-buckets", type=Path, default=DEFAULT_SOURCE_BUCKETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-buckets", type=Path, default=DEFAULT_OUTPUT_BUCKETS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--family-kind", default=DEFAULT_FAMILY_KIND)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4o-mini"))
    parser.add_argument("--parallelism", type=int, default=max(1, int(os.environ.get("PSAGENT_EXEC_CLEAN_PARALLELISM", "3") or 3)))
    parser.add_argument("--terminal-threshold", type=float, default=2.0)
    parser.add_argument("--task-ids", nargs="*")
    args = parser.parse_args()

    tasks = load_json(args.source_data)
    if args.task_ids:
        task_id_set = {str(task_id) for task_id in args.task_ids}
        tasks = [task for task in tasks if str(task.get("original_task_id")) in task_id_set]
    task_by_id = {str(task.get("original_task_id")): task for task in tasks}

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

    jobs: list[dict[str, Any]] = []
    immediate_rows: list[dict[str, Any]] = []
    for task in tasks:
        task_id = str(task.get("original_task_id"))
        chosen = choose_mode_matched_path(task=task, all_paths=all_paths, agent_map=agent_map)
        if chosen is None:
            immediate_rows.append(
                {
                    "original_task_id": task_id,
                    "decision": "delete",
                    "delete_reasons": "no_real_family_path_with_exact_stage_mode_match",
                    "expected_terminal_action": (task.get("metadata") or {}).get("expected_terminal_action"),
                    "repairability": (task.get("metadata") or {}).get("repairability"),
                    "required_modes": "/".join(stage_modes(task)),
                    "agent_modes": "",
                    "mode_match_exact": False,
                    "terminal_cost": "",
                    "clear_success": False,
                    "auxiliary_success": False,
                    "exact_match": False,
                    "subset_clean": False,
                    "strict_clean": False,
                }
            )
            continue
        jobs.append(
            {
                "job_index": len(jobs),
                "task": task,
                "path_profile": chosen,
                "family_kind": args.family_kind,
                "seed": args.seed,
                "model": args.model,
            }
        )
        print(
            f"[select] {len(jobs)}/{len(tasks)} task={task_id} "
            f"required={'/'.join(chosen['required_modes'])} "
            f"lanes={'/'.join(chosen['lane_sequence'])}",
            flush=True,
        )

    completed: list[dict[str, Any]] = []
    if args.parallelism <= 1:
        for job in jobs:
            completed.append(run_task_job(job))
            print(f"[done] {len(completed)}/{len(jobs)} task={completed[-1]['original_task_id']}", flush=True)
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            futures = [executor.submit(run_task_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                completed.append(row)
                print(
                    f"[done] {len(completed)}/{len(jobs)} "
                    f"task={row['original_task_id']} terminal={row['terminal_cost']} "
                    f"clear={row['clear_success']} aux={row['auxiliary_success']}",
                    flush=True,
                )
        completed.sort(key=lambda row: int(row["job_index"]))

    report_rows: list[dict[str, Any]] = []
    kept_tasks: list[dict[str, Any]] = []
    kept_ids: set[str] = set()
    for row in [*immediate_rows, *completed]:
        if row.get("decision") == "delete" and row.get("delete_reasons"):
            report_rows.append(row)
            continue
        keep, reasons = pass_thresholds(row, terminal_threshold=args.terminal_threshold)
        task_id = str(row["original_task_id"])
        row["decision"] = "keep" if keep else "delete"
        row["delete_reasons"] = ";".join(reasons)
        row["keep_basis"] = (
            f"mode_match_exact=1;terminal_cost<={args.terminal_threshold:g};"
            "clear_success=1;repair_task_subset_clean=1"
            if keep
            else ""
        )
        report_rows.append(row)
        if keep:
            kept_tasks.append(task_by_id[task_id])
            kept_ids.add(task_id)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.json"
    manifest_path = args.output_dir / "manifest.json"
    raw_jsonl_path = args.output_dir / "execution_calibration_records.jsonl"
    write_json(tasks_path, kept_tasks)
    write_jsonl(raw_jsonl_path, report_rows)
    write_csv(args.report_csv, report_rows)

    source_buckets = load_json(args.source_buckets)
    trap_full = filter_bucket_ids(source_buckets.get("trap_favoring_task_ids", []), kept_ids)
    target_full = filter_bucket_ids(source_buckets.get("target_favoring_task_ids", []), kept_ids)
    bucket_size = min(len(trap_full), len(target_full))
    trap_ids = trap_full[:bucket_size]
    target_ids = target_full[:bucket_size]
    specialist_ids = filter_bucket_ids(source_buckets.get("specialist_task_ids", []), set(target_ids))
    bucket_payload = {
        "schema_version": "profile_switch_local_exec_clean_v1",
        "source_dataset": str(args.source_data),
        "source_buckets": str(args.source_buckets),
        "clean_dataset": str(tasks_path),
        "execution_report_csv": str(args.report_csv),
        "selection_criteria": {
            "path": "one real family path per task with exact per-stage fast/deep mode match",
            "terminal_threshold": args.terminal_threshold,
            "hard_keep_rules": [
                f"terminal_cost <= {args.terminal_threshold:g}",
                "clear_success_proxy == 1",
                "for repair_all/repair_subset tasks, subset_clean == 1",
            ],
            "recorded_not_hard_gated": [
                "auxiliary_success_proxy",
                "strict_clean",
                "raw_policy_penalty",
            ],
            "bucket_equalization": "trap and target buckets filtered to kept ids, then trimmed to equal size preserving source order",
        },
        "trap_favoring_task_ids": trap_ids,
        "target_favoring_task_ids": target_ids,
        "specialist_task_ids": specialist_ids,
        "coverage_summary": {
            "source_task_count": len(tasks),
            "executed_task_count": len(completed),
            "clean_task_count": len(kept_tasks),
            "deleted_task_count": len(tasks) - len(kept_tasks),
            "decision_counts": dict(Counter(row.get("decision") for row in report_rows)),
            "delete_reason_counts": dict(Counter(row.get("delete_reasons") for row in report_rows if row.get("decision") == "delete")),
            "clean_expected_terminal_action_counts": dict(
                Counter((task.get("metadata") or {}).get("expected_terminal_action") for task in kept_tasks)
            ),
            "trap_bucket_size_before_equalization": len(trap_full),
            "target_bucket_size_before_equalization": len(target_full),
            "trap_bucket_size": len(trap_ids),
            "target_bucket_size": len(target_ids),
            "specialist_bucket_size": len(specialist_ids),
        },
    }
    write_json(args.output_buckets, bucket_payload)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": args.output_dir.name,
        "tasks": str(tasks_path),
        "schedule_buckets": str(args.output_buckets),
        "source_dataset": str(args.source_data),
        "source_buckets": str(args.source_buckets),
        "execution_report_csv": str(args.report_csv),
        "execution_report_json": str(args.report_json),
        "execution_records_jsonl": str(raw_jsonl_path),
        "family_kind": args.family_kind,
        "model": args.model,
        "parallelism": args.parallelism,
        "terminal_threshold": args.terminal_threshold,
        "coverage_summary": bucket_payload["coverage_summary"],
    }
    write_json(manifest_path, manifest)
    write_json(
        args.report_json,
        {
            "manifest": manifest,
            "bucket_payload": bucket_payload,
            "task_rows": report_rows,
        },
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
