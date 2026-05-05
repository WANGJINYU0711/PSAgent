#!/usr/bin/env python3
"""Build local_exec_clean_v3_100 by replacing v2 neutral tasks with clean trap tasks.

The v3 dataset keeps all non-neutral v2 tasks, removes the eight
unchanged_source_profile controls, and fills the 100-task set with newly
execution-calibrated trap_pre_switch tasks. Candidate calibration uses the
current telecom llm_bench executor with stage1-3 on gpt-4o-mini and stage4-5 on
gpt-4.1-mini by default.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from collections import Counter
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
from scripts.build_telecom_mms_fixed_tree import (  # noqa: E402
    TELECOM_DB_PATH,
    TELECOM_SPLITS_PATH,
    TELECOM_TASKS_PATH,
    load_json as load_source_json,
    parse_task_id,
)
from scripts.build_shared_basin_profile_switch_assets import (  # noqa: E402
    TRAP_STAGE_PROFILE,
    apply_stage_profile,
)
from scripts.build_telecom_mms_profile_switch_exec_clean_100_dataset import (  # noqa: E402
    blocker_family_tags,
    build_candidate_rows,
    build_source_subsplit_lookup,
    candidate_static_record,
    is_target_candidate_from_blockers,
    is_trap_candidate_from_blockers,
    pass_thresholds,
    read_csv_rows,
    stage_modes,
    terminal_action_for_raw,
    write_csv,
    write_json,
    write_jsonl,
)
from scripts.build_telecom_mms_profile_switch_llm_exec_clean_dataset import (  # noqa: E402
    choose_mode_matched_path,
    run_task_job,
)
from tree_family.generator import TreeFamilyGenerator  # noqa: E402


DATASET_NAME = "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v3_100"
SCHEMA_VERSION = "profile_switch_local_exec_clean_v3_100"
DEFAULT_FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch"
DEFAULT_V2_TASKS = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json"
)
DEFAULT_V2_REPORT_CSV = (
    ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v2_100_execution_report.csv"
)
DEFAULT_OUTPUT_DIR = ROOT / "data/derived" / DATASET_NAME
DEFAULT_OUTPUT_BUCKETS = (
    ROOT / "analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v3_100_schedule_buckets.json"
)
DEFAULT_REPORT_JSON = ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v3_100_execution_report.json"
DEFAULT_REPORT_CSV = ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v3_100_execution_report.csv"
DEFAULT_CANDIDATE_CSV = ROOT / "analysis/telecom_mms_profile_switch_local_exec_clean_v3_100_candidate_prefilter.csv"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def boolish(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def task_profile(task: dict[str, Any]) -> str:
    return str((task.get("metadata") or {}).get("profile_switch_profile", ""))


def task_bucket(task: dict[str, Any]) -> str:
    return str((task.get("metadata") or {}).get("profile_switch_bucket", ""))


def candidate_diversity_key(task_id: str) -> tuple[str, int, str, str]:
    parsed = parse_task_id(task_id)
    blockers = list(parsed["blockers"])
    tags = tuple(blocker_family_tags(blockers))
    return (
        "|".join(tags),
        len(blockers),
        str(parsed.get("persona")),
        terminal_action_for_raw({"id": task_id}),
    )


def candidate_priority(row: dict[str, Any]) -> tuple[float, int, str, str]:
    return (
        float(row.get("priority_score", 0.0) or 0.0),
        int(row.get("num_blockers", 0) or 0),
        str(row.get("expected_terminal_action", "")),
        str(row.get("original_task_id", "")),
    )


def build_addition_candidate_records(
    *,
    raw_mms: list[dict[str, Any]],
    exclude_ids: set[str],
    prior_delete_ids: set[str],
    addition_profile: str,
) -> list[dict[str, Any]]:
    if addition_profile not in {"trap", "trap_broad", "target"}:
        raise ValueError(f"unsupported addition_profile: {addition_profile}")
    expected_bucket = "target_favoring" if addition_profile == "target" else "trap_favoring"
    records: list[dict[str, Any]] = []
    for raw_task in raw_mms:
        task_id = str(raw_task["id"])
        blockers = list(parse_task_id(task_id)["blockers"])
        if task_id in exclude_ids:
            continue
        if addition_profile == "trap" and not is_trap_candidate_from_blockers(blockers):
            continue
        if addition_profile == "trap_broad" and "user_abroad_roaming_disabled_off" in set(blockers):
            continue
        if addition_profile == "target" and not is_target_candidate_from_blockers(blockers):
            continue
        if terminal_action_for_raw(raw_task) == "transfer":
            continue
        if addition_profile == "trap_broad":
            parsed = parse_task_id(task_id)
            terminal = terminal_action_for_raw(raw_task)
            families = blocker_family_tags(blockers)
            contains_wifi = "bad_wifi_calling" in set(blockers)
            contains_apn = "break_apn_mms_setting" in set(blockers)
            contains_sim = "unseat_sim_card" in set(blockers)
            contains_deferred = bool({"data_usage_exceeded", "user_abroad_roaming_disabled_on"} & set(blockers))
            score = 0.0
            if terminal == "repair_all":
                score -= 40.0
            else:
                score += 20.0
            if contains_wifi:
                score -= 35.0
            if contains_apn:
                score -= 10.0
            if contains_sim:
                score -= 8.0
            if contains_deferred:
                score += 18.0
            if parsed["persona"] == "Hard":
                score += 3.0
            score += len(blockers) * (3.0 if terminal == "repair_all" else 5.0)
            if task_id in prior_delete_ids:
                score += 50.0
            record = {
                "original_task_id": task_id,
                "static_decision": "candidate",
                "static_reasons": "",
                "priority_score": round(score, 3),
                "expected_terminal_action": terminal,
                "persona_level": parsed["persona"],
                "num_blockers": len(blockers),
                "blocker_family_tags": "|".join(families),
                "profile_switch_bucket": expected_bucket,
                "prior_calibration": "delete" if task_id in prior_delete_ids else "",
                "forced_profile_note": "broad_nontransfer_nonhybrid_forced_trap_profile",
            }
        else:
            record = candidate_static_record(
                raw_task,
                prior_keep_ids=exclude_ids,
                prior_delete_ids=prior_delete_ids,
            )
        if record["static_decision"] == "candidate" and record["profile_switch_bucket"] == expected_bucket:
            records.append(record)
    records.sort(key=candidate_priority)
    return records


def select_diverse_additions(
    rows: list[dict[str, Any]],
    *,
    target_count: int,
    repair_subset_target: int,
    addition_bucket: str,
) -> list[dict[str, Any]]:
    passing = [
        row
        for row in rows
        if row.get("decision") == "keep"
        and row.get("profile_switch_bucket") == addition_bucket
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    tag_counts: Counter[str] = Counter()
    blocker_count_counts: Counter[int] = Counter()
    terminal_counts: Counter[str] = Counter()

    def score(row: dict[str, Any]) -> tuple[float, str]:
        task_id = str(row["original_task_id"])
        blockers = list(parse_task_id(task_id)["blockers"])
        tags = blocker_family_tags(blockers)
        terminal = str(row.get("expected_terminal_action") or terminal_action_for_raw({"id": task_id}))
        novelty = 0.0
        for tag in tags:
            novelty += 3.0 / (1.0 + tag_counts[tag])
        novelty += 2.0 / (1.0 + blocker_count_counts[len(blockers)])
        if terminal == "repair_subset" and terminal_counts["repair_subset"] < repair_subset_target:
            novelty += 4.0
        if terminal == "repair_all":
            novelty += 1.0
        if len(blockers) == 1:
            novelty += 0.5
        terminal_penalty = float(row.get("terminal_cost", 99.0) or 99.0)
        token_penalty = float(row.get("total_tokens_total", 0.0) or 0.0) / 1000000.0
        return (novelty - terminal_penalty - token_penalty, task_id)

    while len(selected) < target_count:
        candidates = [row for row in passing if str(row["original_task_id"]) not in selected_ids]
        if not candidates:
            break
        candidates.sort(key=score, reverse=True)
        chosen = candidates[0]
        task_id = str(chosen["original_task_id"])
        selected.append(chosen)
        selected_ids.add(task_id)
        blockers = list(parse_task_id(task_id)["blockers"])
        for tag in blocker_family_tags(blockers):
            tag_counts[tag] += 1
        blocker_count_counts[len(blockers)] += 1
        terminal_counts[str(chosen.get("expected_terminal_action"))] += 1
    return selected


def run_jobs(jobs: list[dict[str, Any]], *, parallelism: int) -> list[dict[str, Any]]:
    completed: list[dict[str, Any]] = []
    if parallelism <= 1:
        for job in jobs:
            row = run_task_job(job)
            completed.append(row)
            print(
                f"[done] {len(completed)}/{len(jobs)} task={row['original_task_id']} "
                f"terminal={row['terminal_cost']} clear={row['clear_success']} "
                f"subset={row['subset_clean']}",
                flush=True,
            )
        return completed

    with concurrent.futures.ThreadPoolExecutor(max_workers=parallelism) as executor:
        futures = [executor.submit(run_task_job, job) for job in jobs]
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            completed.append(row)
            print(
                f"[done] {len(completed)}/{len(jobs)} task={row['original_task_id']} "
                f"terminal={row['terminal_cost']} clear={row['clear_success']} "
                f"subset={row['subset_clean']}",
                flush=True,
            )
    completed.sort(key=lambda row: int(row["job_index"]))
    return completed


def annotate_model_config(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    copied = dict(row)
    copied["stage123_model"] = args.stage123_model
    copied["stage45_model"] = args.stage45_model
    copied["stage_models_intended"] = json.dumps(
        [
            args.stage123_model,
            args.stage123_model,
            args.stage123_model,
            args.stage45_model,
            args.stage45_model,
        ],
        ensure_ascii=False,
    )
    copied["stage45_contract_prompt_v1_1b"] = boolish(
        os.environ.get("PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B", "")
    )
    copied["terminal_v4_enabled"] = boolish(os.environ.get("PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4", ""))
    copied["reasoning_weight_calibration_v3_enabled"] = boolish(
        os.environ.get("PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3", "")
    )
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v2-tasks", type=Path, default=DEFAULT_V2_TASKS)
    parser.add_argument("--v2-report-csv", type=Path, default=DEFAULT_V2_REPORT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-buckets", type=Path, default=DEFAULT_OUTPUT_BUCKETS)
    parser.add_argument("--report-json", type=Path, default=DEFAULT_REPORT_JSON)
    parser.add_argument("--report-csv", type=Path, default=DEFAULT_REPORT_CSV)
    parser.add_argument("--candidate-csv", type=Path, default=DEFAULT_CANDIDATE_CSV)
    parser.add_argument("--family-kind", default=DEFAULT_FAMILY_KIND)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-count", type=int, default=100)
    parser.add_argument("--target-additions", type=int, default=8)
    parser.add_argument("--addition-profile", choices=("trap", "trap_broad", "target"), default="trap")
    parser.add_argument("--candidate-task-ids", nargs="*")
    parser.add_argument("--repair-subset-target", type=int, default=2)
    parser.add_argument("--max-candidates", type=int, default=80)
    parser.add_argument("--parallelism", type=int, default=max(1, int(os.environ.get("PSAGENT_EXEC_CLEAN_PARALLELISM", "4") or 4)))
    parser.add_argument("--terminal-threshold", type=float, default=2.0)
    parser.add_argument("--stage123-model", default=os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4o-mini"))
    parser.add_argument("--stage45-model", default=os.environ.get("PSAGENT_TELECOM_STAGE45_MODEL", "gpt-4.1-mini"))
    parser.add_argument("--enable-cconfig-env", action="store_true")
    args = parser.parse_args()

    os.environ["PSAGENT_LLM_BENCH_MODEL"] = args.stage123_model
    os.environ["PSAGENT_TELECOM_STAGE45_MODEL"] = args.stage45_model
    if args.enable_cconfig_env:
        os.environ.setdefault("PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B", "1")
        os.environ.setdefault("PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4", "1")
        os.environ.setdefault("PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3", "1")

    v2_tasks = load_json(args.v2_tasks)
    seed_tasks = [task for task in v2_tasks if task_profile(task) != "unchanged_source_profile"]
    removed_neutral = [task for task in v2_tasks if task_profile(task) == "unchanged_source_profile"]
    if len(seed_tasks) + args.target_additions != args.target_count:
        raise SystemExit(
            f"Expected seed_tasks + target_additions == target_count, got "
            f"{len(seed_tasks)} + {args.target_additions} != {args.target_count}"
        )
    seed_ids = {str(task["original_task_id"]) for task in seed_tasks}
    all_v2_ids = {str(task["original_task_id"]) for task in v2_tasks}
    prior_delete_ids = {
        str(row.get("original_task_id"))
        for row in read_csv_rows(args.v2_report_csv)
        if row.get("decision") == "delete"
    }

    raw_all = load_source_json(TELECOM_TASKS_PATH)
    split_map = load_source_json(TELECOM_SPLITS_PATH)
    source_subsplit_lookup = build_source_subsplit_lookup(split_map)
    raw_mms = [
        task
        for task in raw_all
        if parse_task_id(str(task["id"]))["family"] == "mms_issue"
    ]

    addition_bucket = "target_favoring" if args.addition_profile == "target" else "trap_favoring"
    addition_records = build_addition_candidate_records(
        raw_mms=raw_mms,
        exclude_ids=all_v2_ids,
        prior_delete_ids=prior_delete_ids,
        addition_profile=args.addition_profile,
    )
    if args.candidate_task_ids:
        requested_ids = {str(task_id) for task_id in args.candidate_task_ids}
        addition_records = [
            record
            for record in addition_records
            if str(record["original_task_id"]) in requested_ids
        ]
        missing_requested = requested_ids - {
            str(record["original_task_id"]) for record in addition_records
        }
        if missing_requested:
            raise SystemExit(
                "Requested candidate task ids were not eligible "
                f"{args.addition_profile} additions: "
                + ", ".join(sorted(missing_requested)[:10])
            )
    addition_records = addition_records[: args.max_candidates]
    write_csv(args.candidate_csv, addition_records)
    raw_by_id = {str(task["id"]): task for task in raw_mms}
    candidate_raw = [raw_by_id[str(row["original_task_id"])] for row in addition_records]
    candidate_tasks = build_candidate_rows(candidate_raw, source_subsplit_lookup)
    if args.addition_profile == "trap_broad":
        candidate_tasks = [
            apply_stage_profile(task, "trap_pre_switch", TRAP_STAGE_PROFILE)
            for task in candidate_tasks
        ]
        for task in candidate_tasks:
            metadata = dict(task.get("metadata") or {})
            metadata["profile_switch_bucket"] = "trap_favoring"
            metadata["forced_profile_note"] = "broad_nontransfer_nonhybrid_forced_trap_profile"
            task["metadata"] = metadata
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

    jobs: list[dict[str, Any]] = []
    immediate_rows: list[dict[str, Any]] = []
    for record in addition_records:
        task_id = str(record["original_task_id"])
        task = candidate_by_id[task_id]
        chosen = choose_mode_matched_path(task=task, all_paths=all_paths, agent_map=agent_map)
        if chosen is None:
            immediate_rows.append(
                annotate_model_config(
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
                        "calibration_source": "new_llm_execution_stage123_4omini_stage45_41mini",
                    },
                    args,
                )
            )
            continue
        jobs.append(
            {
                "job_index": len(jobs),
                "task": task,
                "path_profile": chosen,
                "family_kind": args.family_kind,
                "seed": args.seed,
                "model": args.stage123_model,
            }
        )
        print(
            f"[select] {len(jobs)}/{len(addition_records)} task={task_id} "
            f"terminal={record['expected_terminal_action']} "
            f"required={'/'.join(chosen['required_modes'])} lanes={'/'.join(chosen['lane_sequence'])}",
            flush=True,
        )

    print(
        f"[start] seed_non_neutral={len(seed_tasks)} removed_neutral={len(removed_neutral)} "
        f"{args.addition_profile}_candidates={len(addition_records)} executable_jobs={len(jobs)} "
        f"models=stage1-3:{args.stage123_model},stage4-5:{args.stage45_model} "
        f"parallelism={args.parallelism}",
        flush=True,
    )
    completed = [annotate_model_config(row, args) for row in run_jobs(jobs, parallelism=args.parallelism)]

    report_rows: list[dict[str, Any]] = [*immediate_rows]
    candidate_meta_by_id = {str(row["original_task_id"]): row for row in addition_records}
    for row in completed:
        task_id = str(row["original_task_id"])
        keep, reasons = pass_thresholds(row, terminal_threshold=args.terminal_threshold)
        row.update(
            {
                "profile_switch_bucket": addition_bucket,
                "blocker_family_tags": candidate_meta_by_id[task_id].get("blocker_family_tags", ""),
                "priority_score": candidate_meta_by_id[task_id].get("priority_score", ""),
                "decision": "keep" if keep else "delete",
                "delete_reasons": ";".join(reasons),
                "keep_basis": (
                    f"mode_match_exact=1;terminal_cost<={args.terminal_threshold:g};"
                    "clear_success=1;repair_task_subset_clean=1"
                    if keep
                    else ""
                ),
                "calibration_source": "new_llm_execution_stage123_4omini_stage45_41mini",
            }
        )
        report_rows.append(row)

    selected_rows = select_diverse_additions(
        report_rows,
        target_count=args.target_additions,
        repair_subset_target=args.repair_subset_target,
        addition_bucket=addition_bucket,
    )
    selected_ids = {str(row["original_task_id"]) for row in selected_rows}
    if len(selected_rows) < args.target_additions:
        raise SystemExit(
            f"Only found {len(selected_rows)} passing {args.addition_profile} additions; "
            f"need {args.target_additions}."
        )

    for row in report_rows:
        if row.get("decision") == "keep" and str(row.get("original_task_id")) not in selected_ids:
            row["decision"] = "not_selected_after_target_full"
            row["keep_basis"] = "passed_execution_calibration_but_not_selected_by_v3_diversity"

    additions = [candidate_by_id[task_id] for task_id in selected_ids]
    additions.sort(
        key=lambda task: (
            len(parse_task_id(str(task["original_task_id"]))["blockers"]),
            str(task["original_task_id"]),
        )
    )
    kept_tasks = [*seed_tasks, *additions]
    kept_ids = {str(task["original_task_id"]) for task in kept_tasks}
    if len(kept_tasks) != args.target_count:
        raise SystemExit(f"Built {len(kept_tasks)} tasks, expected {args.target_count}")
    if len(kept_ids) != len(kept_tasks):
        raise SystemExit("Duplicate original_task_id in v3 kept tasks")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tasks_path = args.output_dir / "tasks.json"
    manifest_path = args.output_dir / "manifest.json"
    records_path = args.output_dir / "execution_calibration_records.jsonl"
    write_json(tasks_path, kept_tasks)
    write_jsonl(records_path, report_rows)
    write_csv(args.report_csv, report_rows)

    trap_ids = [
        str(task["original_task_id"])
        for task in kept_tasks
        if task_bucket(task) == "trap_favoring"
    ]
    target_ids = [
        str(task["original_task_id"])
        for task in kept_tasks
        if task_bucket(task) in {"target_favoring", "specialist_target_favoring"}
    ]
    specialist_ids = [
        str(task["original_task_id"])
        for task in kept_tasks
        if task_bucket(task) == "specialist_target_favoring"
    ]
    decision_counts = Counter(row.get("decision") for row in report_rows)
    delete_reason_counts = Counter(
        row.get("delete_reasons", "") for row in report_rows if row.get("decision") == "delete"
    )
    bucket_payload = {
        "schema_version": SCHEMA_VERSION,
        "clean_dataset": str(tasks_path),
        "execution_report_csv": str(args.report_csv),
        "candidate_prefilter_csv": str(args.candidate_csv),
        "source_v2_dataset": str(args.v2_tasks),
        "source_v2_report_csv": str(args.v2_report_csv),
        "runner_compatibility_note": (
            "Full v3 buckets intentionally expose all trap and all target ids and are "
            "not equalized for the legacy trap_switch runner."
        ),
        "selection_criteria": {
            "seed": "all v2 tasks except unchanged_source_profile",
            "removed": "all v2 unchanged_source_profile neutral/control tasks",
            "new_candidate_pool": (
                f"raw tau-bench telecom MMS {addition_bucket} candidates not already in v2"
            ),
            "static_prefilter": [
                "expected_terminal_action != transfer",
                "exclude hybrid/nonlocal transfer-like blockers",
                f"profile_switch_bucket == {addition_bucket}",
            ],
            "execution_path": "one real family path per task with exact per-stage fast/deep mode match",
            "llm_models": {
                "stage1": args.stage123_model,
                "stage2": args.stage123_model,
                "stage3": args.stage123_model,
                "stage4": args.stage45_model,
                "stage5": args.stage45_model,
            },
            "hard_keep_rules": [
                f"terminal_cost <= {args.terminal_threshold:g}",
                "clear_success_proxy == 1",
                "for repair_all/repair_subset tasks, subset_clean == 1",
            ],
            "selection_after_pass": [
                f"prefer diverse {args.addition_profile} blocker families and blocker counts",
                f"prefer about {args.repair_subset_target} clean repair_subset additions when available",
                "avoid reintroducing neutral/control profiles",
            ],
        },
        "trap_favoring_task_ids": trap_ids,
        "target_favoring_task_ids": target_ids,
        "specialist_task_ids": specialist_ids,
        "removed_neutral_task_ids": [str(task["original_task_id"]) for task in removed_neutral],
        "selected_new_addition_task_ids": [str(row["original_task_id"]) for row in selected_rows],
        "selected_new_addition_profile": args.addition_profile,
        "coverage_summary": {
            "clean_task_count": len(kept_tasks),
            "seed_non_neutral_count": len(seed_tasks),
            "removed_neutral_count": len(removed_neutral),
            "new_addition_profile": args.addition_profile,
            "new_addition_selected_count": len(selected_rows),
            "new_addition_executed_count": len(completed),
            "new_addition_static_candidate_count": len(addition_records),
            "decision_counts_for_new_addition_calibration": dict(decision_counts),
            "delete_reason_counts_for_new_addition_calibration": dict(delete_reason_counts),
            "clean_profile_counts": dict(Counter(task_profile(task) for task in kept_tasks)),
            "clean_bucket_counts": dict(Counter(task_bucket(task) for task in kept_tasks)),
            "clean_expected_terminal_action_counts": dict(
                Counter((task.get("metadata") or {}).get("expected_terminal_action") for task in kept_tasks)
            ),
            "clean_requirement_counts": dict(Counter("/".join(stage_modes(task)) for task in kept_tasks)),
            "all_trap_profile_count": len(trap_ids),
            "all_target_profile_count": len(target_ids),
            "specialist_bucket_size": len(specialist_ids),
        },
    }
    write_json(args.output_buckets, bucket_payload)

    selected_final_rows = [
        row for row in report_rows if str(row.get("original_task_id")) in selected_ids
    ]
    manifest = {
        "created_at": datetime.now().isoformat(),
        "dataset_name": args.output_dir.name,
        "schema_version": SCHEMA_VERSION,
        "task_count": len(kept_tasks),
        "target_count": args.target_count,
        "tasks": str(tasks_path),
        "schedule_buckets": str(args.output_buckets),
        "source_dataset": str(TELECOM_TASKS_PATH),
        "source_split_tasks": str(TELECOM_SPLITS_PATH),
        "source_db": str(TELECOM_DB_PATH),
        "source_v2_dataset": str(args.v2_tasks),
        "execution_report_csv": str(args.report_csv),
        "execution_report_json": str(args.report_json),
        "candidate_prefilter_csv": str(args.candidate_csv),
        "execution_records_jsonl": str(records_path),
        "family_kind": args.family_kind,
        "seed": args.seed,
        "stage123_model": args.stage123_model,
        "stage45_model": args.stage45_model,
        "addition_profile": args.addition_profile,
        "parallelism": args.parallelism,
        "terminal_threshold": args.terminal_threshold,
        "enable_cconfig_env": bool(args.enable_cconfig_env),
        "coverage_summary": bucket_payload["coverage_summary"],
        "selected_new_addition_metric_rates": {
            "exact_match": round(sum(boolish(row.get("exact_match")) for row in selected_final_rows) / len(selected_final_rows), 4),
            "subset_clean": round(sum(boolish(row.get("subset_clean")) for row in selected_final_rows) / len(selected_final_rows), 4),
            "clear_success": round(sum(boolish(row.get("clear_success")) for row in selected_final_rows) / len(selected_final_rows), 4),
            "auxiliary_success": round(sum(boolish(row.get("auxiliary_success")) for row in selected_final_rows) / len(selected_final_rows), 4),
            "strict_clean": round(sum(boolish(row.get("strict_clean")) for row in selected_final_rows) / len(selected_final_rows), 4),
        },
    }
    write_json(manifest_path, manifest)
    write_json(
        args.report_json,
        {
            "manifest": manifest,
            "bucket_payload": bucket_payload,
            "new_addition_calibration_rows": report_rows,
            "selected_new_addition_rows": selected_final_rows,
        },
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
