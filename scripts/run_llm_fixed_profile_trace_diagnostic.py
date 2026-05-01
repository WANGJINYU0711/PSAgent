from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from run_llm_path_sweep_diagnostic import (  # noqa: E402
    STAGES,
    TelecomMMSTaskAdapter,
    TreeFamilyGenerator,
    aggregate_path_resource_summary,
    build_offline_path_record,
    build_stage_requirements,
    flatten_record_for_csv,
    json_ready,
    load_bucket_membership,
    load_rows,
    run_selected_path_job,
    sort_offline_records,
    write_csv,
    write_json,
)
from oracle_eval import enumerate_family_paths  # noqa: E402


DEFAULT_DATA = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json"
)
DEFAULT_BUCKET_FILE = (
    ROOT / "analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json"
)
DEFAULT_ALIGNED = ROOT / "tmp/llm_v8_seed1_old_vs_probfloor_targeted_diagnostic/aligned_old_vs_probfloor.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run fixed exact-mode/profile LLM trace diagnostics with persisted Stage 4/5 raw outputs. "
            "This is diagnostic-only and does not update PS or executor behavior."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--bucket-file", type=Path, default=DEFAULT_BUCKET_FILE)
    parser.add_argument("--family-kind", default="shared_basin_strong_prefix_dedup_profile_switch")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model", default=os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4o-mini"))
    parser.add_argument("--dataset-indices", type=int, nargs="*", default=[2, 10, 13, 16])
    parser.add_argument("--patterns", nargs="*", default=["fdddd", "ffddd", "ddddd"])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallelism", type=int, default=1)
    parser.add_argument(
        "--aligned-old-vs-probfloor",
        type=Path,
        default=DEFAULT_ALIGNED,
        help="Aligned diagnostic CSV used to harvest observed high-terminal exact-match paths.",
    )
    parser.add_argument(
        "--include-observed-high-terminal",
        action="store_true",
        help="Also rerun concrete fdddd-on-fdddd high-terminal paths from old/probfloor aligned CSV.",
    )
    parser.add_argument(
        "--high-terminal-threshold",
        type=float,
        default=10.0,
    )
    parser.add_argument("--max-observed-paths", type=int, default=0)
    return parser.parse_args()


def mode_pattern_for_path(path: list[str], agent_map: dict[str, Any]) -> str:
    chars = []
    for agent_id in path:
        mode = str(getattr(agent_map[agent_id], "deliberation_mode", "deep")).lower()
        chars.append("d" if mode == "deep" else "f")
    return "".join(chars)


def required_pattern(stage_requirements: dict[str, Any]) -> str:
    chars = []
    for stage_name in STAGES:
        mode = str(stage_requirements["deliberation"].get(stage_name, "fast")).lower()
        chars.append("d" if mode == "deep" else "f")
    return "".join(chars)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def load_focus_rows(data_path: Path, dataset_indices: list[int]) -> list[dict[str, Any]]:
    rows = load_rows(data_path.resolve())
    out = []
    for idx in dataset_indices:
        row = dict(rows[idx])
        row["_dataset_index"] = idx
        out.append(row)
    return out


def compact_task_id(row: dict[str, Any]) -> str:
    return str(row.get("original_task_id", row.get("instance_id", "unknown")))


def choose_canonical_pattern_paths(
    *,
    focus_rows: list[dict[str, Any]],
    family_kind: str,
    seed: int,
    patterns: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any], Any]:
    generator = TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(family_kind, seed=seed)
    validation_errors = generator.validate_family(family_spec, agent_map)
    if validation_errors:
        raise SystemExit(f"Family validation failed: {validation_errors}")
    all_paths = enumerate_family_paths(
        stages=list(family_spec.stages),
        stage_agents=family_spec.stage_agents,
        allowed_children=family_spec.allowed_children,
    )
    adapter = TelecomMMSTaskAdapter()
    path_rows: list[dict[str, Any]] = []
    for row in focus_rows:
        task_id = compact_task_id(row)
        stage_requirements = build_stage_requirements(row, adapter)
        rankings = sort_offline_records(
            [
                {
                    **build_offline_path_record(
                        task_id=task_id,
                        path=path,
                        stage_requirements=stage_requirements,
                        agent_map=agent_map,
                        weakening_level=0,
                    ),
                    "actual_pattern": mode_pattern_for_path(list(path), agent_map),
                    "required_pattern": required_pattern(stage_requirements),
                }
                for path in all_paths
            ]
        )
        for pattern in patterns:
            matches = [path_row for path_row in rankings if path_row["actual_pattern"] == pattern]
            if not matches:
                raise SystemExit(f"No path found for task={task_id} pattern={pattern}")
            chosen = dict(matches[0])
            chosen["diagnostic_group"] = "canonical_pattern"
            chosen["diagnostic_pattern"] = pattern
            chosen["source_episode"] = None
            chosen["source_run"] = None
            path_rows.append(chosen)
    return path_rows, agent_map, family_spec


def observed_high_terminal_paths(
    *,
    aligned_path: Path,
    focus_rows_by_index: dict[int, dict[str, Any]],
    agent_map: dict[str, Any],
    threshold: float,
    max_observed_paths: int = 0,
) -> list[dict[str, Any]]:
    rows = read_csv_dicts(aligned_path)
    out: list[dict[str, Any]] = []
    seen: set[tuple[int, tuple[str, ...], str]] = set()
    for row in rows:
        try:
            dataset_index = int(row["dataset_index"])
        except (KeyError, ValueError):
            continue
        if dataset_index not in focus_rows_by_index:
            continue
        for prefix in ("old", "pf"):
            if row.get(f"{prefix}_pattern") != "fdddd on fdddd":
                continue
            terminal = float(row.get(f"{prefix}_terminal") or 0.0)
            if terminal < threshold:
                continue
            path = [part.strip() for part in row.get(f"{prefix}_path", "").split(">") if part.strip()]
            if not path:
                continue
            key = (dataset_index, tuple(path), prefix)
            if key in seen:
                continue
            seen.add(key)
            if any(agent_id not in agent_map for agent_id in path):
                continue
            out.append(
                {
                    "task_id": compact_task_id(focus_rows_by_index[dataset_index]),
                    "path_agent_ids": path,
                    "path_match": 1.0,
                    "rank": 9000 + len(out),
                    "path_class": "observed_high_terminal_exact_match",
                    "path_lane_sequence": [],
                    "path_route_labels": [],
                    "path_node_semantics": [],
                    "path_route_summary": "",
                    "path_base_cost_sum": None,
                    "first_private_barrier_depth": None,
                    "leaf_type": None,
                    "actual_pattern": "fdddd",
                    "required_pattern": "fdddd",
                    "diagnostic_group": "observed_high_terminal_exact_match",
                    "diagnostic_pattern": "fdddd",
                    "source_episode": row.get("episode_index"),
                    "source_run": prefix,
                    "source_terminal": terminal,
                    "source_final": row.get(f"{prefix}_final"),
                    "source_cause": row.get(f"{prefix}_cause"),
                }
            )
            if 0 < int(max_observed_paths) <= len(out):
                return out
    return out


def task_oracle(row: dict[str, Any]) -> dict[str, Any]:
    ec = row.get("source_task", {}).get("evaluation_criteria", {})
    selected = list(ec.get("selected_blocker_ids", []) or [])
    deferred = list(ec.get("deferred_blocker_ids", []) or [])
    oracle_tools = [a.get("name") for a in ec.get("actions", []) if a.get("name")]
    return {
        "expected_terminal_action": ec.get("expected_terminal_action")
        or row.get("metadata", {}).get("expected_terminal_action"),
        "selected_blocker_ids_oracle": selected,
        "deferred_blocker_ids_oracle": deferred,
        "oracle_tools": oracle_tools,
    }


def replay_tools(record: dict[str, Any]) -> list[str]:
    return [str(name) for name in record.get("stage5_replay_tool_names", []) or []]


def enrich_record(record: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    row = job["row"]
    oracle = task_oracle(row)
    selected = set(record.get("selected_blocker_ids", []) or [])
    deferred = set(record.get("deferred_blocker_ids", []) or [])
    oracle_selected = set(oracle["selected_blocker_ids_oracle"])
    oracle_deferred = set(oracle["deferred_blocker_ids_oracle"])
    stage4_output = record.get("stage4_output", {}) if isinstance(record.get("stage4_output"), dict) else {}
    stage5_output = record.get("stage5_output", {}) if isinstance(record.get("stage5_output"), dict) else {}
    replay = replay_tools(record)
    oracle_tools = list(oracle["oracle_tools"])
    record.update(
        {
            "dataset_index": job["dataset_index"],
            "repeat_index": job["repeat_index"],
            "diagnostic_group": job["diagnostic_group"],
            "diagnostic_pattern": job["diagnostic_pattern"],
            "actual_pattern": job["actual_pattern"],
            "required_pattern": job["required_pattern"],
            "source_run": job.get("source_run"),
            "source_episode": job.get("source_episode"),
            "source_terminal": job.get("source_terminal"),
            "source_final": job.get("source_final"),
            "source_cause": job.get("source_cause"),
            "expected_terminal_action": oracle["expected_terminal_action"],
            "oracle_selected_blocker_ids": oracle["selected_blocker_ids_oracle"],
            "oracle_deferred_blocker_ids": oracle["deferred_blocker_ids_oracle"],
            "oracle_tools": oracle["oracle_tools"],
            "stage5_replay_tool_names": replay,
            "stage5_executed_tool_names": list(record.get("stage5_executed_tool_names", []) or []),
            "stage4_executed_tool_names": list(record.get("stage4_executed_tool_names", []) or []),
            "oracle_tools_missing_from_stage5_replay": sorted(set(oracle_tools) - set(replay)),
            "stage5_replay_tools_extra_vs_oracle": sorted(set(replay) - set(oracle_tools)),
            "selected_missing_vs_oracle": sorted(oracle_selected - selected),
            "selected_extra_vs_oracle": sorted(selected - oracle_selected - oracle_deferred),
            "deferred_missing_vs_oracle": sorted(oracle_deferred - deferred),
            "deferred_extra_vs_oracle": sorted(deferred - oracle_deferred - oracle_selected),
            "stage4_raw_per_blocker_count": len(stage4_output.get("per_blocker", []) or [])
            if isinstance(stage4_output, dict)
            else 0,
            "stage5_final_action_reason": stage5_output.get("final_action_reason")
            or stage5_output.get("transfer_reason")
            if isinstance(stage5_output, dict)
            else None,
            "terminal_adjustment_reasons": list(record.get("terminal_adjustment_reasons", []) or []),
        }
    )
    return record


def flatten_trace_record_for_csv(record: dict[str, Any]) -> dict[str, Any]:
    base = flatten_record_for_csv(record)
    extra = {
        "dataset_index": record.get("dataset_index"),
        "repeat_index": record.get("repeat_index"),
        "diagnostic_group": record.get("diagnostic_group"),
        "diagnostic_pattern": record.get("diagnostic_pattern"),
        "actual_pattern": record.get("actual_pattern"),
        "required_pattern": record.get("required_pattern"),
        "source_run": record.get("source_run"),
        "source_episode": record.get("source_episode"),
        "source_terminal": record.get("source_terminal"),
        "source_final": record.get("source_final"),
        "source_cause": record.get("source_cause"),
        "oracle_action": record.get("oracle_action"),
        "expected_terminal_action": record.get("expected_terminal_action"),
        "selected_missing_vs_oracle": json.dumps(record.get("selected_missing_vs_oracle", []), ensure_ascii=False),
        "selected_extra_vs_oracle": json.dumps(record.get("selected_extra_vs_oracle", []), ensure_ascii=False),
        "deferred_missing_vs_oracle": json.dumps(record.get("deferred_missing_vs_oracle", []), ensure_ascii=False),
        "deferred_extra_vs_oracle": json.dumps(record.get("deferred_extra_vs_oracle", []), ensure_ascii=False),
        "stage4_selected_before_normalization": json.dumps(
            record.get("stage4_selected_before_normalization", []), ensure_ascii=False
        ),
        "stage4_selected_after_normalization": json.dumps(
            record.get("stage4_selected_after_normalization", []), ensure_ascii=False
        ),
        "stage4_deferred_before_normalization": json.dumps(
            record.get("stage4_deferred_before_normalization", []), ensure_ascii=False
        ),
        "stage4_deferred_after_normalization": json.dumps(
            record.get("stage4_deferred_after_normalization", []), ensure_ascii=False
        ),
        "stage4_completion_added_blockers": json.dumps(
            record.get("stage4_completion_added_blockers", []), ensure_ascii=False
        ),
        "stage4_completion_added_prerequisite_blockers": json.dumps(
            record.get("stage4_completion_added_prerequisite_blockers", []), ensure_ascii=False
        ),
        "stage4_completion_added_downstream_blockers": json.dumps(
            record.get("stage4_completion_added_downstream_blockers", []), ensure_ascii=False
        ),
        "stage5_final_action_reason": record.get("stage5_final_action_reason"),
        "terminal_adjustment_reasons": json.dumps(
            record.get("terminal_adjustment_reasons", []), ensure_ascii=False
        ),
        "stage5_replay_tool_names": json.dumps(
            record.get("stage5_replay_tool_names", []), ensure_ascii=False
        ),
        "stage5_executed_tool_names": json.dumps(
            record.get("stage5_executed_tool_names", []), ensure_ascii=False
        ),
        "oracle_tools_missing_from_stage5_replay": json.dumps(
            record.get("oracle_tools_missing_from_stage5_replay", []), ensure_ascii=False
        ),
        "stage5_replay_tools_extra_vs_oracle": json.dumps(
            record.get("stage5_replay_tools_extra_vs_oracle", []), ensure_ascii=False
        ),
    }
    return {**extra, **base}


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return float(statistics.fmean(values))


def build_report(output_dir: Path, records: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[Any, Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[
            (
                record.get("dataset_index"),
                record.get("diagnostic_group"),
                record.get("diagnostic_pattern"),
            )
        ].append(record)

    summary_rows = []
    for (dataset_index, group, pattern), rows in sorted(grouped.items()):
        summary_rows.append(
            {
                "dataset_index": dataset_index,
                "group": group,
                "pattern": pattern,
                "n": len(rows),
                "terminal_mean": mean([float(r["raw_terminal_penalty"]) for r in rows]),
                "terminal_values": [float(r["raw_terminal_penalty"]) for r in rows],
                "final_counts": dict(Counter(str(r.get("final_action")) for r in rows)),
                "selected_missing_counts": dict(
                    Counter("|".join(r.get("selected_missing_vs_oracle", [])) for r in rows)
                ),
                "deferred_missing_counts": dict(
                    Counter("|".join(r.get("deferred_missing_vs_oracle", [])) for r in rows)
                ),
                "completion_pass_rate": mean(
                    [1.0 if r.get("stage4_completion_pass_applied") else 0.0 for r in rows]
                ),
                "normalizer_changed_rate": mean(
                    [1.0 if r.get("stage4_normalizer_changed_output") else 0.0 for r in rows]
                ),
            }
        )
    write_json(output_dir / "summary_by_dataset_pattern.json", summary_rows)

    lines = [
        "# Fixed Profile Trace Diagnostic",
        "",
        "This run persists Stage 4/5 raw/normalized trace fields in `records.json`.",
        "",
        "| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |",
        "|---:|---|---|---:|---:|---|---|---|---|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset_index"]),
                    str(row["group"]),
                    str(row["pattern"]),
                    str(row["n"]),
                    "" if row["terminal_mean"] is None else f"{row['terminal_mean']:.3f}",
                    json.dumps(row["terminal_values"], ensure_ascii=False),
                    json.dumps(row["final_counts"], ensure_ascii=False),
                    json.dumps(row["selected_missing_counts"], ensure_ascii=False),
                    json.dumps(row["deferred_missing_counts"], ensure_ascii=False),
                    "" if row["completion_pass_rate"] is None else f"{row['completion_pass_rate']:.3f}",
                    "" if row["normalizer_changed_rate"] is None else f"{row['normalizer_changed_rate']:.3f}",
                ]
            )
            + " |"
        )
    output_dir.joinpath("report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    focus_rows = load_focus_rows(args.data.resolve(), args.dataset_indices)
    focus_by_index = {int(row["_dataset_index"]): row for row in focus_rows}
    bucket_membership = load_bucket_membership(args.bucket_file.resolve())
    canonical_paths, agent_map, _ = choose_canonical_pattern_paths(
        focus_rows=focus_rows,
        family_kind=args.family_kind,
        seed=args.seed,
        patterns=args.patterns,
    )
    observed_paths = (
        observed_high_terminal_paths(
            aligned_path=args.aligned_old_vs_probfloor.resolve(),
            focus_rows_by_index=focus_by_index,
            agent_map=agent_map,
            threshold=args.high_terminal_threshold,
            max_observed_paths=args.max_observed_paths,
        )
        if args.include_observed_high_terminal
        else []
    )
    selected_paths = canonical_paths + observed_paths
    write_json(output_dir / "selected_paths.json", json_ready(selected_paths))

    jobs: list[dict[str, Any]] = []
    job_index = 0
    for path_row in selected_paths:
        task_id = str(path_row["task_id"])
        row = next(row for row in focus_rows if compact_task_id(row) == task_id)
        for repeat_index in range(args.repeats):
            jobs.append(
                {
                    "job_index": job_index,
                    "task_id": task_id,
                    "bucket_label": "focus",
                    "row": row,
                    "path_row": path_row,
                    "family_kind": args.family_kind,
                    "seed": args.seed,
                    "model": args.model,
                    "dataset_index": int(row["_dataset_index"]),
                    "repeat_index": repeat_index,
                    "diagnostic_group": path_row["diagnostic_group"],
                    "diagnostic_pattern": path_row["diagnostic_pattern"],
                    "actual_pattern": path_row["actual_pattern"],
                    "required_pattern": path_row["required_pattern"],
                    "source_run": path_row.get("source_run"),
                    "source_episode": path_row.get("source_episode"),
                    "source_terminal": path_row.get("source_terminal"),
                    "source_final": path_row.get("source_final"),
                    "source_cause": path_row.get("source_cause"),
                }
            )
            job_index += 1

    write_json(
        output_dir / "run_config.json",
        {
            "script": str(Path(__file__).resolve()),
            "data": str(args.data.resolve()),
            "bucket_file": str(args.bucket_file.resolve()),
            "family_kind": args.family_kind,
            "seed": args.seed,
            "model": args.model,
            "dataset_indices": args.dataset_indices,
            "patterns": args.patterns,
            "repeats": args.repeats,
            "include_observed_high_terminal": args.include_observed_high_terminal,
            "high_terminal_threshold": args.high_terminal_threshold,
            "max_observed_paths": args.max_observed_paths,
            "parallelism": args.parallelism,
            "job_count": len(jobs),
        },
    )

    partial_path = output_dir / "records.partial.jsonl"
    if partial_path.exists():
        partial_path.unlink()

    def persist_partial(record: dict[str, Any]) -> None:
        with partial_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(json_ready(record), ensure_ascii=False) + "\n")

    if args.parallelism <= 1:
        records = []
        for job in jobs:
            record = enrich_record(run_selected_path_job(job)["record"], job)
            records.append(record)
            persist_partial(record)
            print(
                f"[done] {len(records)}/{len(jobs)} "
                f"dataset={record.get('dataset_index')} group={record.get('diagnostic_group')} "
                f"pattern={record.get('diagnostic_pattern')} terminal={record.get('raw_terminal_penalty')} "
                f"final={record.get('final_action')}",
                flush=True,
            )
    else:
        completed: list[tuple[int, dict[str, Any]]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallelism) as executor:
            future_to_job = {executor.submit(run_selected_path_job, job): job for job in jobs}
            for future in concurrent.futures.as_completed(future_to_job):
                job = future_to_job[future]
                result = future.result()
                record = enrich_record(result["record"], job)
                completed.append((int(result["job_index"]), record))
                persist_partial(record)
                print(
                    f"[done] {len(completed)}/{len(jobs)} "
                    f"dataset={record.get('dataset_index')} group={record.get('diagnostic_group')} "
                    f"pattern={record.get('diagnostic_pattern')} terminal={record.get('raw_terminal_penalty')} "
                    f"final={record.get('final_action')}",
                    flush=True,
                )
        completed.sort(key=lambda item: item[0])
        records = [record for _, record in completed]

    write_json(output_dir / "records.json", json_ready(records))
    write_csv(output_dir / "records.csv", [flatten_trace_record_for_csv(row) for row in records])
    build_report(output_dir, records)


if __name__ == "__main__":
    main()
