#!/usr/bin/env python3
"""Build a 100-task execution-calibrated clean MMS profile-switch dataset.

This derives extra candidates directly from the raw tau-bench telecom MMS task
pool, converts them to the same fixed-tree/capability/time/profile-switch schema
used by the current profile-switch datasets, then runs real llm_bench execution
on exact per-stage fast/deep matched family paths. Tasks are retained only when
their execution signal is clean enough for smoke-test use.
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

from oracle_eval import enumerate_family_paths  # noqa: E402
from scripts.build_shared_basin_profile_switch_assets import (  # noqa: E402
    ROAMING_BLOCKERS,
    SHALLOW_TRAP_BLOCKERS,
    SPECIALIST_STAGE_PROFILE,
    TARGET_STAGE_PROFILE,
    TRAP_STAGE_PROFILE,
    apply_stage_profile,
)
from scripts.build_telecom_mms_capability_benchmark import add_capability_requirements  # noqa: E402
from scripts.build_telecom_mms_capability_time_benchmark import add_deliberation_requirements  # noqa: E402
from scripts.build_telecom_mms_fixed_tree import (  # noqa: E402
    TELECOM_DB_PATH,
    TELECOM_SPLITS_PATH,
    TELECOM_TASKS_PATH,
    build_dataset_with_stats,
    build_reference_maps,
    load_json as load_source_json,
    load_telecom_reference_db,
    parse_task_id,
)
from scripts.build_telecom_mms_profile_switch_llm_exec_clean_dataset import (  # noqa: E402
    choose_mode_matched_path,
    pass_thresholds,
    run_task_job,
    stage_modes,
)
from tree_family.generator import TreeFamilyGenerator  # noqa: E402


STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5")
DEFAULT_FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch"
DATASET_NAME = "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100"
SCHEMA_VERSION = "profile_switch_local_exec_clean_v2_100"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "derived" / DATASET_NAME
DEFAULT_OUTPUT_BUCKETS = ROOT / "analysis" / f"shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_schedule_buckets.json"
DEFAULT_REPORT_JSON = ROOT / "analysis" / "telecom_mms_profile_switch_local_exec_clean_v2_100_execution_report.json"
DEFAULT_REPORT_CSV = ROOT / "analysis" / "telecom_mms_profile_switch_local_exec_clean_v2_100_execution_report.csv"
DEFAULT_CANDIDATE_CSV = ROOT / "analysis" / "telecom_mms_profile_switch_local_exec_clean_v2_100_candidate_prefilter.csv"
PRIOR_CLEAN_TASKS = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v1/tasks.json"
)
PRIOR_REPORT_CSV = ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v1_execution_report.csv"

PERMISSION_BLOCKERS = {
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
}
PROFILE_SWITCH_PROFILE_VERSION = "profile_switch_v1"


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


def build_source_subsplit_lookup(split_map: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for task_id in split_map.get("train", []):
        lookup[str(task_id)] = "train"
    for task_id in split_map.get("test", []):
        lookup[str(task_id)] = "test"
    for task_id in split_map.get("base", []):
        lookup.setdefault(str(task_id), "base")
    return lookup


def parse_blockers(task_id: str) -> list[str]:
    parsed = parse_task_id(task_id)
    return list(parsed["blockers"])


def blocker_family_tags(blockers: list[str]) -> list[str]:
    blocker_set = set(blockers)
    tags: list[str] = []
    if "break_apn_mms_setting" in blocker_set:
        tags.append("apn")
    if blocker_set & PERMISSION_BLOCKERS:
        tags.append("permission")
    if "bad_wifi_calling" in blocker_set:
        tags.append("wifi_calling")
    if "bad_network_preference" in blocker_set:
        tags.append("network_preference")
    if "unseat_sim_card" in blocker_set:
        tags.append("sim")
    if "data_mode_off" in blocker_set:
        tags.append("data_mode")
    if "data_usage_exceeded" in blocker_set:
        tags.append("data_usage_exceeded")
    if "user_abroad_roaming_disabled_on" in blocker_set:
        tags.append("roaming_disabled_on")
    if "user_abroad_roaming_enabled_off" in blocker_set:
        tags.append("roaming_enabled_off")
    if "user_abroad_roaming_disabled_off" in blocker_set:
        tags.append("hybrid_roaming_disabled_off")
    return tags


def is_trap_candidate_from_blockers(blockers: list[str]) -> bool:
    blocker_set = set(blockers)
    return (
        1 <= len(blockers) <= 3
        and "break_apn_mms_setting" not in blocker_set
        and "unseat_sim_card" not in blocker_set
        and not (blocker_set & ROAMING_BLOCKERS)
        and blocker_set <= SHALLOW_TRAP_BLOCKERS
    )


def is_target_candidate_from_blockers(blockers: list[str]) -> bool:
    blocker_set = set(blockers)
    return (
        len(blockers) >= 4
        and "break_apn_mms_setting" in blocker_set
        and ("unseat_sim_card" in blocker_set or bool(blocker_set & ROAMING_BLOCKERS))
        and "user_abroad_roaming_disabled_off" not in blocker_set
    )


def is_specialist_candidate_from_blockers(blockers: list[str]) -> bool:
    blocker_set = set(blockers)
    return (
        is_target_candidate_from_blockers(blockers)
        and len(blockers) >= 8
        and ("unseat_sim_card" in blocker_set or bool(blocker_set & ROAMING_BLOCKERS))
    )


def profile_switch_row(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row["original_task_id"])
    blockers = parse_blockers(task_id)
    if is_specialist_candidate_from_blockers(blockers):
        updated = apply_stage_profile(row, "target_specialist_post_switch", SPECIALIST_STAGE_PROFILE)
        updated["metadata"]["profile_switch_bucket"] = "specialist_target_favoring"
    elif is_target_candidate_from_blockers(blockers):
        updated = apply_stage_profile(row, "target_post_switch", TARGET_STAGE_PROFILE)
        updated["metadata"]["profile_switch_bucket"] = "target_favoring"
    elif is_trap_candidate_from_blockers(blockers):
        updated = apply_stage_profile(row, "trap_pre_switch", TRAP_STAGE_PROFILE)
        updated["metadata"]["profile_switch_bucket"] = "trap_favoring"
    else:
        updated = deepcopy(row)
        metadata = deepcopy(updated.get("metadata", {}))
        metadata["profile_switch_version"] = PROFILE_SWITCH_PROFILE_VERSION
        metadata["profile_switch_profile"] = "unchanged_source_profile"
        metadata["profile_switch_bucket"] = "neutral_other"
        metadata["profile_switch_blockers"] = blockers
        updated["metadata"] = metadata
    return updated


def terminal_action_for_raw(raw_task: dict[str, Any]) -> str:
    from envs.telecom_mms_specs import first_pass_terminal_decision

    return str(first_pass_terminal_decision(parse_blockers(raw_task["id"]))["final_action"])


def candidate_static_record(raw_task: dict[str, Any], prior_keep_ids: set[str], prior_delete_ids: set[str]) -> dict[str, Any]:
    task_id = str(raw_task["id"])
    parsed = parse_task_id(task_id)
    blockers = list(parsed["blockers"])
    blocker_set = set(blockers)
    terminal = terminal_action_for_raw(raw_task)
    families = blocker_family_tags(blockers)
    trap = is_trap_candidate_from_blockers(blockers)
    target = is_target_candidate_from_blockers(blockers)
    specialist = is_specialist_candidate_from_blockers(blockers)
    contains_hybrid = "user_abroad_roaming_disabled_off" in blocker_set
    contains_deferred = bool({"data_usage_exceeded", "user_abroad_roaming_disabled_on"} & blocker_set)
    contains_apn = "break_apn_mms_setting" in blocker_set
    contains_sim_or_roaming = "unseat_sim_card" in blocker_set or bool(blocker_set & ROAMING_BLOCKERS)
    static_reasons: list[str] = []
    if terminal == "transfer":
        static_reasons.append("static_exclude_transfer_oracle")
    if contains_hybrid:
        static_reasons.append("static_exclude_hybrid_roaming_disabled_off")

    static_keep = False
    if not static_reasons:
        if terminal == "repair_all":
            static_keep = (
                len(blockers) <= 6
                or trap
                or target
                or (contains_apn and len(blockers) <= 7)
            )
        elif terminal == "repair_subset":
            static_keep = (
                len(blockers) <= 5
                or target
                or (contains_deferred and contains_apn and contains_sim_or_roaming and len(blockers) <= 8)
            )
    if not static_keep and not static_reasons:
        static_reasons.append("static_exclude_low_priority_nontransfer_shape")

    score = 1000.0
    if static_keep:
        score = 0.0
        if task_id in prior_keep_ids:
            score -= 500.0
        if trap:
            score -= 80.0
        if target:
            score -= 60.0
        if specialist:
            score -= 10.0
        if terminal == "repair_all":
            score -= 40.0
        if terminal == "repair_subset":
            score -= 20.0
        if "bad_wifi_calling" in blocker_set:
            score -= 12.0
        if "break_app_storage_permission" in blocker_set:
            score -= 8.0
        if "break_app_sms_permission" in blocker_set:
            score -= 6.0
        if "break_app_both_permissions" in blocker_set:
            score += 8.0
        if "data_usage_exceeded" in blocker_set:
            score += 4.0
        score += len(blockers) * (2.0 if terminal == "repair_all" else 3.0)
        if parsed["persona"] == "Hard":
            score += 3.0
        if task_id in prior_delete_ids:
            score += 250.0
    return {
        "original_task_id": task_id,
        "static_decision": "candidate" if static_keep else "static_delete",
        "static_reasons": ";".join(static_reasons),
        "priority_score": round(score, 3),
        "expected_terminal_action": terminal,
        "persona_level": parsed["persona"],
        "num_blockers": len(blockers),
        "blocker_family_tags": "|".join(families),
        "profile_switch_bucket": (
            "specialist_target_favoring"
            if specialist
            else "target_favoring"
            if target
            else "trap_favoring"
            if trap
            else "neutral_other"
        ),
        "prior_calibration": "keep" if task_id in prior_keep_ids else "delete" if task_id in prior_delete_ids else "",
    }


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def prior_keep_report_rows(prior_report_csv: Path) -> list[dict[str, Any]]:
    rows = []
    for row in read_csv_rows(prior_report_csv):
        if row.get("decision") != "keep":
            continue
        copied = dict(row)
        copied["decision"] = "keep"
        copied["delete_reasons"] = ""
        copied["keep_basis"] = copied.get("keep_basis") or "prior_v1_execution_calibration_keep"
        copied["calibration_source"] = "prior_local_exec_clean_v1_reused"
        rows.append(copied)
    return rows


def build_candidate_rows(raw_tasks: list[dict[str, Any]], source_subsplit_lookup: dict[str, str]) -> list[dict[str, Any]]:
    reference_maps = build_reference_maps(load_telecom_reference_db())
    rows, skipped = build_dataset_with_stats(
        tasks=raw_tasks,
        source_split="all_mms_issue",
        source_subsplit_lookup=source_subsplit_lookup,
        subset_version=SCHEMA_VERSION,
        smoke_task_ids=None,
        reference_maps=reference_maps,
    )
    if skipped:
        print(f"[derive] skipped fixed-tree rows: {len(skipped)}", flush=True)
    enriched: list[dict[str, Any]] = []
    for row in rows:
        enriched.append(profile_switch_row(add_deliberation_requirements(add_capability_requirements(row))))
    return enriched


def run_jobs_in_batch(
    jobs: list[dict[str, Any]],
    *,
    parallelism: int,
) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    if parallelism <= 1:
        for job in jobs:
            completed.append(run_task_job(job))
            row = completed[-1]
            print(
                f"[done] {len(completed)}/{len(jobs)} task={row['original_task_id']} "
                f"terminal={row['terminal_cost']} clear={row['clear_success']} aux={row['auxiliary_success']}",
                flush=True,
            )
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
            futures = [executor.submit(run_task_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                completed.append(row)
                print(
                    f"[done] {len(completed)}/{len(jobs)} task={row['original_task_id']} "
                    f"terminal={row['terminal_cost']} clear={row['clear_success']} aux={row['auxiliary_success']}",
                    flush=True,
                )
    return completed


def compact_report_row(row: dict[str, Any], *, calibration_source: str = "new_llm_execution") -> dict[str, Any]:
    copied = dict(row)
    copied["calibration_source"] = calibration_source
    return copied


def summarize_bool(rows: list[dict[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(boolish(row.get(key)) for row in rows) / len(rows)


def build_bucket_payload(kept_tasks: list[dict[str, Any]], report_rows: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    kept_ids = {str(task["original_task_id"]) for task in kept_tasks}
    trap_ids = [
        str(task["original_task_id"])
        for task in kept_tasks
        if (task.get("metadata") or {}).get("profile_switch_bucket") == "trap_favoring"
    ]
    target_full = [
        str(task["original_task_id"])
        for task in kept_tasks
        if (task.get("metadata") or {}).get("profile_switch_bucket") in {"target_favoring", "specialist_target_favoring"}
    ]
    target_ids = target_full
    if trap_ids and target_full:
        bucket_size = min(len(trap_ids), len(target_full))
        trap_ids = trap_ids[:bucket_size]
        target_ids = target_full[:bucket_size]
    specialist_ids = [
        task_id
        for task_id in target_ids
        if task_id in kept_ids
        and any(
            str(task["original_task_id"]) == task_id
            and (task.get("metadata") or {}).get("profile_switch_bucket") == "specialist_target_favoring"
            for task in kept_tasks
        )
    ]
    decision_counts = Counter(row.get("decision") for row in report_rows)
    delete_reason_counts = Counter(
        row.get("delete_reasons", "") for row in report_rows if row.get("decision") == "delete"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "clean_dataset": str(args.output_dir / "tasks.json"),
        "execution_report_csv": str(args.report_csv),
        "candidate_prefilter_csv": str(args.candidate_csv),
        "selection_criteria": {
            "source_pool": "raw tau-bench telecom MMS tasks",
            "static_prefilter": [
                "exclude expected_terminal_action == transfer",
                "exclude hybrid user_abroad_roaming_disabled_off transfer-like tasks",
                "prioritize repair_all <= 6 blockers, trap-profile tasks, and target-profile local tasks",
                "prioritize repair_subset local/non-hybrid tasks with <=5 blockers or target-profile APN+SIM/roaming shapes",
            ],
            "execution_path": "one real family path per task with exact per-stage fast/deep mode match",
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
        },
        "trap_favoring_task_ids": trap_ids,
        "target_favoring_task_ids": target_ids,
        "specialist_task_ids": specialist_ids,
        "coverage_summary": {
            "clean_task_count": len(kept_tasks),
            "decision_counts": dict(decision_counts),
            "delete_reason_counts": dict(delete_reason_counts),
            "clean_expected_terminal_action_counts": dict(
                Counter((task.get("metadata") or {}).get("expected_terminal_action") for task in kept_tasks)
            ),
            "clean_requirement_counts": dict(Counter("/".join(stage_modes(task)) for task in kept_tasks)),
            "trap_bucket_size": len(trap_ids),
            "target_bucket_size": len(target_ids),
            "specialist_bucket_size": len(specialist_ids),
            "all_target_profile_count": len(target_full),
            "all_trap_profile_count": len(
                [
                    task
                    for task in kept_tasks
                    if (task.get("metadata") or {}).get("profile_switch_bucket") == "trap_favoring"
                ]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-buckets", type=Path, default=DEFAULT_OUTPUT_BUCKETS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--prior-clean-tasks", type=Path, default=PRIOR_CLEAN_TASKS)
    parser.add_argument("--prior-report-csv", type=Path, default=PRIOR_REPORT_CSV)
    parser.add_argument("--target-clean-count", type=int, default=100)
    parser.add_argument("--max-new-executions", type=int, default=int(os.environ.get("PSAGENT_EXEC_CLEAN_MAX_NEW", "650") or 650))
    parser.add_argument("--batch-size", type=int, default=int(os.environ.get("PSAGENT_EXEC_CLEAN_BATCH_SIZE", "72") or 72))
    parser.add_argument("--parallelism", type=int, default=max(1, int(os.environ.get("PSAGENT_EXEC_CLEAN_PARALLELISM", "6") or 6)))
    parser.add_argument("--terminal-threshold", type=float, default=2.0)
    parser.add_argument("--family-kind", default=DEFAULT_FAMILY_KIND)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4o-mini"))
    args = parser.parse_args()

    raw_all = load_source_json(TELECOM_TASKS_PATH)
    split_map = load_source_json(TELECOM_SPLITS_PATH)
    source_subsplit_lookup = build_source_subsplit_lookup(split_map)
    raw_mms = [
        task
        for task in raw_all
        if parse_task_id(str(task["id"]))["family"] == "mms_issue"
    ]

    prior_tasks = load_json(args.prior_clean_tasks) if args.prior_clean_tasks.exists() else []
    prior_keep_ids = {str(task["original_task_id"]) for task in prior_tasks}
    prior_rows = prior_keep_report_rows(args.prior_report_csv)
    prior_delete_ids = {
        str(row.get("original_task_id"))
        for row in read_csv_rows(args.prior_report_csv)
        if row.get("decision") == "delete"
    }

    static_records = [
        candidate_static_record(task, prior_keep_ids, prior_delete_ids) for task in raw_mms
    ]
    write_csv(args.candidate_csv, static_records)

    candidate_records = [
        row
        for row in static_records
        if row["static_decision"] == "candidate" and row["original_task_id"] not in prior_keep_ids
    ]
    candidate_records.sort(
        key=lambda row: (
            float(row["priority_score"]),
            int(row["num_blockers"]),
            row["expected_terminal_action"],
            row["original_task_id"],
        )
    )
    candidate_records = candidate_records[: args.max_new_executions]
    raw_by_id = {str(task["id"]): task for task in raw_mms}
    candidate_raw = [raw_by_id[row["original_task_id"]] for row in candidate_records]
    candidate_tasks = build_candidate_rows(candidate_raw, source_subsplit_lookup)
    candidate_by_id = {str(task["original_task_id"]): task for task in candidate_tasks}

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

    kept_tasks: list[dict[str, Any]] = list(prior_tasks)
    kept_ids = {str(task["original_task_id"]) for task in kept_tasks}
    report_rows: list[dict[str, Any]] = [compact_report_row(row, calibration_source="prior_local_exec_clean_v1_reused") for row in prior_rows]
    executed_new = 0
    cursor = 0
    print(
        f"[start] prior_kept={len(kept_tasks)} target={args.target_clean_count} "
        f"candidate_pool={len(candidate_records)} parallelism={args.parallelism}",
        flush=True,
    )

    while len(kept_tasks) < args.target_clean_count and cursor < len(candidate_records):
        batch_records = candidate_records[cursor : cursor + args.batch_size]
        cursor += len(batch_records)
        jobs: list[dict[str, Any]] = []
        for record in batch_records:
            task_id = str(record["original_task_id"])
            task = candidate_by_id[task_id]
            chosen = choose_mode_matched_path(task=task, all_paths=all_paths, agent_map=agent_map)
            if chosen is None:
                report_rows.append(
                    {
                        **record,
                        "decision": "delete",
                        "delete_reasons": "no_real_family_path_with_exact_stage_mode_match",
                        "required_modes": "/".join(stage_modes(task)),
                        "mode_match_exact": False,
                        "terminal_cost": "",
                        "clear_success": False,
                        "auxiliary_success": False,
                        "exact_match": False,
                        "subset_clean": False,
                        "strict_clean": False,
                        "calibration_source": "new_llm_execution",
                    }
                )
                continue
            jobs.append(
                {
                    "job_index": executed_new + len(jobs),
                    "task": task,
                    "path_profile": chosen,
                    "family_kind": args.family_kind,
                    "seed": args.seed,
                    "model": args.model,
                }
            )
            print(
                f"[select] task={task_id} required={'/'.join(chosen['required_modes'])} "
                f"lanes={'/'.join(chosen['lane_sequence'])}",
                flush=True,
            )
        completed = run_jobs_in_batch(jobs, parallelism=args.parallelism)
        executed_new += len(completed)
        for row in completed:
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
            row["calibration_source"] = "new_llm_execution"
            report_rows.append(row)
            if keep and task_id not in kept_ids and len(kept_tasks) < args.target_clean_count:
                kept_tasks.append(candidate_by_id[task_id])
                kept_ids.add(task_id)
        print(
            f"[batch] executed_new={executed_new} kept_total={len(kept_tasks)} "
            f"cursor={cursor}/{len(candidate_records)}",
            flush=True,
        )

    if len(kept_tasks) > args.target_clean_count:
        kept_tasks = kept_tasks[: args.target_clean_count]
        kept_ids = {str(task["original_task_id"]) for task in kept_tasks}
    for row in report_rows:
        if row.get("decision") == "keep" and str(row.get("original_task_id")) not in kept_ids:
            row["decision"] = "not_selected_after_target_full"
            row["delete_reasons"] = ""
            row["keep_basis"] = "passed_execution_calibration_but_target_count_already_reached"

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.json"
    manifest_path = args.output_dir / "manifest.json"
    records_path = args.output_dir / "execution_calibration_records.jsonl"
    write_json(tasks_path, kept_tasks)
    write_jsonl(records_path, report_rows)
    write_csv(args.report_csv, report_rows)

    bucket_payload = build_bucket_payload(kept_tasks, report_rows, args)
    write_json(args.output_buckets, bucket_payload)

    final_report_rows = [
        row for row in report_rows if str(row.get("original_task_id")) in kept_ids and row.get("decision") == "keep"
    ]
    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": args.output_dir.name,
        "schema_version": SCHEMA_VERSION,
        "task_count": len(kept_tasks),
        "target_clean_count": args.target_clean_count,
        "target_reached": len(kept_tasks) >= args.target_clean_count,
        "tasks": str(tasks_path),
        "schedule_buckets": str(args.output_buckets),
        "source_dataset": str(TELECOM_TASKS_PATH),
        "source_split_tasks": str(TELECOM_SPLITS_PATH),
        "source_db": str(TELECOM_DB_PATH),
        "prior_seed_dataset": str(args.prior_clean_tasks),
        "execution_report_csv": str(args.report_csv),
        "execution_report_json": str(args.report_json),
        "candidate_prefilter_csv": str(args.candidate_csv),
        "execution_records_jsonl": str(records_path),
        "family_kind": args.family_kind,
        "model": args.model,
        "parallelism": args.parallelism,
        "terminal_threshold": args.terminal_threshold,
        "raw_mms_task_count": len(raw_mms),
        "static_candidate_count": len([row for row in static_records if row["static_decision"] == "candidate"]),
        "max_new_executions": args.max_new_executions,
        "new_executed_count": executed_new,
        "prior_reused_count": len(prior_tasks),
        "coverage_summary": bucket_payload["coverage_summary"],
        "kept_execution_metric_rates": {
            "exact_match": round(summarize_bool(final_report_rows, "exact_match"), 4),
            "subset_clean": round(summarize_bool(final_report_rows, "subset_clean"), 4),
            "clear_success": round(summarize_bool(final_report_rows, "clear_success"), 4),
            "auxiliary_success": round(summarize_bool(final_report_rows, "auxiliary_success"), 4),
            "strict_clean": round(summarize_bool(final_report_rows, "strict_clean"), 4),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        args.report_json,
        {
            "manifest": manifest,
            "bucket_payload": bucket_payload,
            "task_rows": report_rows,
            "static_prefilter_rows": static_records,
        },
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
