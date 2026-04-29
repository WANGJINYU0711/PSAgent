#!/usr/bin/env python3
"""Build a local-repair calibrated profile-switch dataset and buckets.

This script does not overwrite the original 100-task profile-switch dataset.
It removes hard-transfer oracle tasks from the dataset-level pool, then builds
equal-size trap/target buckets suitable for the trap-switch repeated smoke.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_DATA = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch/tasks.json"
)
DEFAULT_SOURCE_BUCKETS = (
    ROOT / "analysis/shared_basin_prefix_dedup_profile_switch_schedule_buckets.json"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_clean_v1"
)
DEFAULT_OUTPUT_BUCKETS = (
    ROOT / "analysis/shared_basin_prefix_dedup_profile_switch_local_clean_v1_schedule_buckets.json"
)
DEFAULT_ANALYSIS_JSON = (
    ROOT / "analysis/telecom_mms_profile_switch_local_clean_v1_cleaning_report.json"
)
DEFAULT_ANALYSIS_CSV = (
    ROOT / "analysis/telecom_mms_profile_switch_local_clean_v1_cleaning_report.csv"
)
STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def stage_modes(task: dict[str, Any]) -> list[str]:
    summary = (task.get("metadata") or {}).get("deliberation_requirement_summary") or {}
    return [str(summary.get(stage, "")).strip().lower() for stage in STAGES]


def oracle_stage5_partition_valid(task: dict[str, Any]) -> bool:
    stage5 = task.get("stage5") or {}
    oracle = stage5.get("oracle_output") or {}
    input_rows = ((stage5.get("input") or {}).get("per_blocker") or [])
    input_ids = {str(row.get("blocker_id")) for row in input_rows if row.get("blocker_id")}
    selected = {str(value) for value in oracle.get("selected_blocker_ids", [])}
    deferred = {str(value) for value in oracle.get("deferred_blocker_ids", [])}
    if selected & deferred:
        return False
    return selected | deferred == input_ids


def task_keep_reason(task: dict[str, Any]) -> tuple[bool, str]:
    metadata = task.get("metadata") or {}
    expected = str(metadata.get("expected_terminal_action", ""))
    repairability = str(metadata.get("repairability", ""))
    if expected == "transfer" or repairability == "transfer_required":
        return False, "delete_hard_transfer_oracle"
    if bool(metadata.get("contains_hybrid_action", False)):
        return False, "delete_hybrid_action"
    if not oracle_stage5_partition_valid(task):
        return False, "delete_invalid_stage5_oracle_partition"
    modes = stage_modes(task)
    if len(modes) != len(STAGES) or any(mode not in {"fast", "deep"} for mode in modes):
        return False, "delete_missing_or_invalid_deliberation_requirements"
    return True, "keep_local_repair_or_partial_repair"


def is_fast_trap_candidate(task: dict[str, Any]) -> bool:
    metadata = task.get("metadata") or {}
    return (
        str(metadata.get("expected_terminal_action")) == "repair_all"
        and str(metadata.get("repairability")) == "repairable"
        and not bool(metadata.get("contains_hybrid_action", False))
        and not bool(metadata.get("contains_assistant_side_action", False))
        and stage_modes(task) == ["fast"] * len(STAGES)
    )


def is_deep_target_candidate(task: dict[str, Any]) -> bool:
    metadata = task.get("metadata") or {}
    modes = stage_modes(task)
    return (
        str(metadata.get("expected_terminal_action")) in {"repair_all", "repair_subset"}
        and str(metadata.get("repairability")) in {"repairable", "partially_repairable"}
        and not bool(metadata.get("contains_hybrid_action", False))
        and modes[0] == "fast"
        and modes[1:] == ["deep", "deep", "deep", "deep"]
    )


def task_row(task: dict[str, Any], decision: str, reason: str, bucket_role: str = "") -> dict[str, Any]:
    metadata = task.get("metadata") or {}
    modes = "/".join(stage_modes(task))
    return {
        "original_task_id": task.get("original_task_id"),
        "decision": decision,
        "reason": reason,
        "bucket_role": bucket_role,
        "expected_terminal_action": metadata.get("expected_terminal_action"),
        "repairability": metadata.get("repairability"),
        "contains_hybrid_action": metadata.get("contains_hybrid_action"),
        "contains_assistant_side_action": metadata.get("contains_assistant_side_action"),
        "contains_user_side_action": metadata.get("contains_user_side_action"),
        "persona_level": metadata.get("persona_level"),
        "num_blockers": metadata.get("num_blockers"),
        "required_modes": modes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-data", type=Path, default=DEFAULT_SOURCE_DATA)
    parser.add_argument("--source-buckets", type=Path, default=DEFAULT_SOURCE_BUCKETS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-buckets", type=Path, default=DEFAULT_OUTPUT_BUCKETS)
    parser.add_argument("--analysis-json", type=Path, default=DEFAULT_ANALYSIS_JSON)
    parser.add_argument("--analysis-csv", type=Path, default=DEFAULT_ANALYSIS_CSV)
    parser.add_argument("--bucket-size", type=int, default=10)
    args = parser.parse_args()

    tasks = load_json(args.source_data)
    buckets = load_json(args.source_buckets)
    by_id = {str(task["original_task_id"]): task for task in tasks}

    kept: list[dict[str, Any]] = []
    report_rows: list[dict[str, Any]] = []
    decision_by_id: dict[str, tuple[bool, str]] = {}
    for task in tasks:
        keep, reason = task_keep_reason(task)
        decision_by_id[str(task["original_task_id"])] = (keep, reason)
        if keep:
            kept.append(task)
        report_rows.append(task_row(task, "keep" if keep else "delete", reason))

    kept_ids = {str(task["original_task_id"]) for task in kept}

    original_trap = [str(value) for value in buckets.get("trap_favoring_task_ids", [])]
    original_target = [str(value) for value in buckets.get("target_favoring_task_ids", [])]
    original_specialist = [str(value) for value in buckets.get("specialist_task_ids", [])]

    trap_candidates = [
        task_id
        for task_id in original_trap
        if task_id in kept_ids and is_fast_trap_candidate(by_id[task_id])
    ]
    target_candidates = [
        task_id
        for task_id in original_target
        if task_id in kept_ids and is_deep_target_candidate(by_id[task_id])
    ]
    if len(trap_candidates) < args.bucket_size:
        raise SystemExit(
            f"Not enough clean trap candidates: {len(trap_candidates)} < {args.bucket_size}"
        )
    if len(target_candidates) < args.bucket_size:
        raise SystemExit(
            f"Not enough clean target candidates: {len(target_candidates)} < {args.bucket_size}"
        )

    trap_ids = trap_candidates[: args.bucket_size]
    target_ids = target_candidates[: args.bucket_size]
    specialist_ids = [
        task_id
        for task_id in original_specialist
        if task_id in set(target_ids) and is_deep_target_candidate(by_id[task_id])
    ]

    bucket_role_by_id: dict[str, str] = {}
    for task_id in trap_ids:
        bucket_role_by_id[task_id] = "trap_favoring_clean_fast_repair_all"
    for task_id in target_ids:
        bucket_role_by_id[task_id] = "target_favoring_clean_deep_local"
    for task_id in specialist_ids:
        bucket_role_by_id[task_id] = (
            bucket_role_by_id.get(task_id, "") + ";specialist_clean_deep_local"
        ).strip(";")
    for row in report_rows:
        task_id = str(row["original_task_id"])
        if task_id in bucket_role_by_id:
            row["bucket_role"] = bucket_role_by_id[task_id]

    tasks_out = args.output_dir / "tasks.json"
    manifest_out = args.output_dir / "manifest.json"
    write_json(tasks_out, kept)

    kept_counter = Counter((task.get("metadata") or {}).get("expected_terminal_action") for task in kept)
    deleted_counter = Counter(
        reason for keep, reason in decision_by_id.values() if not keep
    )
    bucket_payload = {
        "schema_version": "profile_switch_local_clean_v1",
        "source_dataset": str(args.source_data),
        "source_buckets": str(args.source_buckets),
        "clean_dataset": str(tasks_out),
        "selection_criteria": {
            "dataset_level": [
                "delete expected_terminal_action=transfer",
                "delete repairability=transfer_required",
                "delete contains_hybrid_action=true",
                "delete invalid Stage 5 oracle selected/deferred partition",
                "require complete fast/deep deliberation_requirement_summary",
            ],
            "trap_favoring": [
                "kept in original trap bucket",
                "repairability=repairable",
                "expected_terminal_action=repair_all",
                "contains_assistant_side_action=false",
                "required modes are fast/fast/fast/fast/fast",
            ],
            "target_favoring": [
                "kept in original target bucket after deleting transfer cases",
                "expected_terminal_action in repair_all/repair_subset",
                "required modes are fast/deep/deep/deep/deep",
            ],
            "bucket_size": args.bucket_size,
            "tie_break": "preserve source bucket order",
        },
        "trap_favoring_task_ids": trap_ids,
        "target_favoring_task_ids": target_ids,
        "specialist_task_ids": specialist_ids,
        "coverage_summary": {
            "source_task_count": len(tasks),
            "clean_task_count": len(kept),
            "deleted_task_count": len(tasks) - len(kept),
            "deleted_reason_counts": dict(sorted(deleted_counter.items())),
            "clean_expected_terminal_action_counts": dict(sorted(kept_counter.items())),
            "trap_bucket_size": len(trap_ids),
            "target_bucket_size": len(target_ids),
            "specialist_bucket_size": len(specialist_ids),
            "target_transfer_count": 0,
        },
        "bucket_labels": {
            "trap_favoring": "clean_fast_path_pre_switch",
            "target_favoring": "clean_deep_local_post_switch",
            "specialist": "clean_deep_local_specialist_subset",
        },
    }
    write_json(args.output_buckets, bucket_payload)

    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": args.output_dir.name,
        "tasks": str(tasks_out),
        "schedule_buckets": str(args.output_buckets),
        "cleaning_report_json": str(args.analysis_json),
        "cleaning_report_csv": str(args.analysis_csv),
        "source_dataset": str(args.source_data),
        "source_buckets": str(args.source_buckets),
        "coverage_summary": bucket_payload["coverage_summary"],
    }
    write_json(manifest_out, manifest)
    write_json(
        args.analysis_json,
        {
            "manifest": manifest,
            "bucket_payload": bucket_payload,
            "task_rows": report_rows,
        },
    )
    write_csv(args.analysis_csv, report_rows)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
