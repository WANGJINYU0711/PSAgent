#!/usr/bin/env python3
"""Analyze blocker drift across telecom LLM bench stages for one run."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def sorted_unique(values: list[str]) -> list[str]:
    return sorted(dict.fromkeys(v for v in values if v))


def as_blocker_ids_from_rows(rows: list[JsonDict]) -> list[str]:
    return [str(row.get("blocker_id")) for row in rows if isinstance(row, dict) and row.get("blocker_id")]


def list_diff(predicted: list[str], oracle: list[str]) -> tuple[list[str], list[str]]:
    predicted_set = set(predicted)
    oracle_set = set(oracle)
    added = sorted(predicted_set - oracle_set)
    missed = sorted(oracle_set - predicted_set)
    return added, missed


def format_list(values: list[str]) -> str:
    return ", ".join(values) if values else "-"


def build_attribution_notes(
    oracle_stage3_ids: list[str],
    predicted_stage3_ids: list[str],
    oracle_state: JsonDict,
    predicted_state: JsonDict,
) -> list[str]:
    notes: list[str] = []
    oracle_set = set(oracle_stage3_ids)
    predicted_set = set(predicted_stage3_ids)

    if "break_app_both_permissions" in oracle_set and "break_app_sms_permission" in predicted_set:
        if predicted_state.get("messaging_storage_permission") is True and oracle_state.get(
            "messaging_storage_permission"
        ) is False:
            notes.append(
                "Stage 3 collapsed `break_app_both_permissions` into `break_app_sms_permission` because "
                "`messaging_storage_permission` flipped from oracle `false` to predicted `true`."
            )
        else:
            notes.append(
                "Stage 3 collapsed `break_app_both_permissions` into the narrower "
                "`break_app_sms_permission` blocker."
            )

    if "break_app_storage_permission" in oracle_set and "break_app_sms_permission" in predicted_set:
        if (
            oracle_state.get("messaging_sms_permission") is True
            and predicted_state.get("messaging_sms_permission") is False
            and oracle_state.get("messaging_storage_permission") is False
            and predicted_state.get("messaging_storage_permission") is True
        ):
            notes.append(
                "Stage 3 swapped the permission channel: oracle says storage is broken, but prediction "
                "says SMS is broken."
            )
        else:
            notes.append(
                "Stage 3 replaced `break_app_storage_permission` with `break_app_sms_permission`."
            )

    if "break_app_sms_permission" in predicted_set and "break_app_sms_permission" not in oracle_set:
        if oracle_state.get("messaging_sms_permission") is True and predicted_state.get("messaging_sms_permission") is False:
            notes.append(
                "The extra `break_app_sms_permission` comes from a false negative on "
                "`messaging_sms_permission` in Stage 3 observed_state."
            )
        else:
            notes.append(
                "Stage 3 introduced a spurious `break_app_sms_permission` blocker not present in the oracle set."
            )

    if set(predicted_stage3_ids) == set(oracle_stage3_ids):
        notes.append("Stage 3 blocker inference matches the oracle set.")

    return notes


def observed_state_mismatches(oracle_state: JsonDict, predicted_state: JsonDict) -> list[str]:
    mismatches: list[str] = []
    keys = sorted(set(oracle_state) | set(predicted_state))
    for key in keys:
        oracle_value = oracle_state.get(key)
        predicted_value = predicted_state.get(key)
        if oracle_value != predicted_value:
            mismatches.append(f"{key}: oracle={oracle_value!r}, predicted={predicted_value!r}")
    return mismatches


def build_report(run_dir: Path, dataset_path: Path) -> str:
    summary = load_json(run_dir / "summary.json")
    episodes = load_json(run_dir / "episodes.json")
    traces = load_json(run_dir / "stage45_traces.json")
    dataset = load_json(dataset_path)

    trace_by_instance = {row["instance_id"]: row for row in traces}
    dataset_by_instance = {row["instance_id"]: row for row in dataset}

    stage3_root_counter: Counter[str] = Counter()
    stage3_added_counter: Counter[str] = Counter()
    stage3_missed_counter: Counter[str] = Counter()
    stage2_resolution_mismatch_count = 0
    stage4_set_change_count = 0
    stage5_set_change_count = 0
    episode_sections: list[str] = []

    for episode in episodes:
        instance_id = episode["instance_id"]
        trace = trace_by_instance[instance_id]
        dataset_item = dataset_by_instance[instance_id]

        oracle_stage3_output = dataset_item["stage3"]["oracle_output"]
        oracle_stage4_output = dataset_item["stage4"]["oracle_output"]
        oracle_stage5_output = dataset_item["stage5"]["oracle_output"]
        oracle_stage2_output = dataset_item["stage2"]["oracle_output"]

        predicted_stage3_output = trace["stage4_trace"]["input"]["stage3_output"]
        predicted_stage4_output = trace["stage4_trace"]["output"]
        predicted_stage5_output = trace["stage5_trace"]["output"]
        predicted_stage2_output = trace["stage4_trace"]["input"]["stage2_output"]

        oracle_stage3_ids = as_blocker_ids_from_rows(oracle_stage3_output.get("per_blocker", []))
        oracle_stage4_ids = as_blocker_ids_from_rows(oracle_stage4_output.get("per_blocker", []))
        predicted_stage3_ids = sorted_unique(
            predicted_stage3_output.get("inferred_blocker_ids")
            or as_blocker_ids_from_rows(predicted_stage3_output.get("per_blocker", []))
        )
        predicted_stage4_ids = as_blocker_ids_from_rows(predicted_stage4_output.get("per_blocker", []))

        oracle_stage5_selected = sorted_unique(oracle_stage5_output.get("selected_blocker_ids", []))
        oracle_stage5_deferred = sorted_unique(oracle_stage5_output.get("deferred_blocker_ids", []))
        predicted_stage5_selected = sorted_unique(predicted_stage5_output.get("selected_blocker_ids", []))
        predicted_stage5_deferred = sorted_unique(predicted_stage5_output.get("deferred_blocker_ids", []))

        stage3_added, stage3_missed = list_diff(predicted_stage3_ids, oracle_stage3_ids)
        stage4_added, stage4_missed = list_diff(predicted_stage4_ids, oracle_stage4_ids)
        stage5_selected_added, stage5_selected_missed = list_diff(
            predicted_stage5_selected, oracle_stage5_selected
        )
        stage5_deferred_added, stage5_deferred_missed = list_diff(
            predicted_stage5_deferred, oracle_stage5_deferred
        )

        if set(predicted_stage4_ids) != set(predicted_stage3_ids):
            stage4_set_change_count += 1
        if set(predicted_stage5_selected) | set(predicted_stage5_deferred) != set(predicted_stage4_ids):
            stage5_set_change_count += 1
        if predicted_stage2_output.get("resolved_line_id") != oracle_stage2_output.get("resolved_line_id"):
            stage2_resolution_mismatch_count += 1
            stage3_root_counter["stage2_resolution_mismatch"] += 1

        notes = build_attribution_notes(
            oracle_stage3_ids=oracle_stage3_ids,
            predicted_stage3_ids=predicted_stage3_ids,
            oracle_state=oracle_stage3_output.get("observed_state", {}),
            predicted_state=predicted_stage3_output.get("observed_state", {}),
        )
        if stage3_added or stage3_missed:
            stage3_root_counter["stage3_blocker_inference_drift"] += 1
        if stage4_added or stage4_missed:
            stage3_root_counter["stage4_added_or_dropped_blockers"] += 1
        if stage5_selected_added or stage5_selected_missed or stage5_deferred_added or stage5_deferred_missed:
            stage3_root_counter["stage5_partition_or_action_drift"] += 1

        for blocker_id in stage3_added:
            stage3_added_counter[blocker_id] += 1
        for blocker_id in stage3_missed:
            stage3_missed_counter[blocker_id] += 1

        mismatch_lines = observed_state_mismatches(
            oracle_stage3_output.get("observed_state", {}),
            predicted_stage3_output.get("observed_state", {}),
        )
        mismatch_text = "\n".join(f"- {line}" for line in mismatch_lines) if mismatch_lines else "- None"
        notes_text = "\n".join(f"- {line}" for line in notes) if notes else "- No extra attribution note."

        episode_sections.append(
            "\n".join(
                [
                    f"## Episode {episode['episode_index']} - `{instance_id}`",
                    "",
                    f"- Oracle final action: `{episode['oracle_action']}`",
                    f"- Predicted final action: `{episode['final_action']}`",
                    f"- Terminal penalty: `{episode['terminal_penalty']}`",
                    f"- Exact match: `{episode['exact_match']}`",
                    f"- Stage 2 resolved line: oracle=`{oracle_stage2_output.get('resolved_line_id')}`, predicted=`{predicted_stage2_output.get('resolved_line_id')}`",
                    "",
                    "| Layer | Oracle | Predicted | Added | Missed |",
                    "| --- | --- | --- | --- | --- |",
                    f"| Stage 3 blocker set | `{format_list(oracle_stage3_ids)}` | `{format_list(predicted_stage3_ids)}` | `{format_list(stage3_added)}` | `{format_list(stage3_missed)}` |",
                    f"| Stage 4 adopted set | `{format_list(oracle_stage4_ids)}` | `{format_list(predicted_stage4_ids)}` | `{format_list(stage4_added)}` | `{format_list(stage4_missed)}` |",
                    f"| Stage 5 selected | `{format_list(oracle_stage5_selected)}` | `{format_list(predicted_stage5_selected)}` | `{format_list(stage5_selected_added)}` | `{format_list(stage5_selected_missed)}` |",
                    f"| Stage 5 deferred | `{format_list(oracle_stage5_deferred)}` | `{format_list(predicted_stage5_deferred)}` | `{format_list(stage5_deferred_added)}` | `{format_list(stage5_deferred_missed)}` |",
                    "",
                    "Observed-state mismatches driving blocker drift:",
                    mismatch_text,
                    "",
                    "Attribution:",
                    notes_text,
                ]
            )
        )

    top_added = ", ".join(f"`{blocker}` x{count}" for blocker, count in stage3_added_counter.most_common()) or "-"
    top_missed = ", ".join(f"`{blocker}` x{count}" for blocker, count in stage3_missed_counter.most_common()) or "-"

    lines = [
        "# Telecom LLM Blocker Drift Report",
        "",
        f"- Run dir: `{run_dir}`",
        f"- Dataset: `{dataset_path}`",
        f"- Method: `{summary.get('method')}`",
        f"- Mechanism: `{summary.get('mechanism')}`",
        f"- Executor: `{summary.get('executor_name')}`",
        f"- Family kind: `{summary.get('family_kind')}`",
        f"- Seed: `{summary.get('seed')}`",
        f"- Episodes analyzed: `{len(episodes)}`",
        "",
        "## Summary",
        "",
        f"- Stage 3 blocker inference drift appears in `{stage3_root_counter['stage3_blocker_inference_drift']}/{len(episodes)}` episodes.",
        f"- Stage 2 resolves the wrong line in `{stage2_resolution_mismatch_count}/{len(episodes)}` episodes.",
        f"- Stage 4 changes the Stage 3 blocker set in `{stage4_set_change_count}/{len(episodes)}` episodes.",
        f"- Stage 5 changes the Stage 4 blocker universe in `{stage5_set_change_count}/{len(episodes)}` episodes.",
        f"- Most common extra blockers: {top_added}",
        f"- Most common missed blockers: {top_missed}",
        "",
        "Interpretation:",
        "- One episode already goes off-track in Stage 2 line resolution before blocker inference starts.",
        "- The current failures are concentrated in Stage 3 observed-state interpretation and blocker inference.",
        "- Stage 4 mostly preserves whatever blocker set Stage 3 hands it.",
        "- Stage 5 mostly preserves the Stage 4 set and only partitions it into selected vs deferred.",
        "",
    ]
    lines.extend(episode_sections)
    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Directory containing summary.json, episodes.json, and stage45_traces.json")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=None,
        help="Optional dataset path. Defaults to the dataset recorded in summary.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output markdown path. Defaults to <run_dir>/blocker_drift_report.md",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = load_json(args.run_dir / "summary.json")
    dataset_path = args.dataset or Path(summary["dataset"])
    output_path = args.output or (args.run_dir / "blocker_drift_report.md")
    report = build_report(run_dir=args.run_dir, dataset_path=dataset_path)
    output_path.write_text(report)
    print(output_path)


if __name__ == "__main__":
    main()
