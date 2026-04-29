"""Analyze full-tree post-switch probability-freeze LLM runs."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4", "stage5"]


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def mode_majority(modes: list[str]) -> str:
    fast = sum(1 for mode in modes if mode == "fast")
    deep = sum(1 for mode in modes if mode == "deep")
    if deep > fast:
        return "mostly_deep"
    if fast > deep:
        return "mostly_fast"
    return "mixed"


def distribution(values: list[Any]) -> dict[str, int]:
    return dict(sorted(Counter(str(value) for value in values).items()))


def normalize_modes(values: Any) -> list[str]:
    if isinstance(values, dict):
        return [str(values.get(stage, "unknown")) for stage in STAGE_NAMES]
    if isinstance(values, list):
        cleaned = [str(value) for value in values[: len(STAGE_NAMES)]]
        return cleaned + ["unknown"] * (len(STAGE_NAMES) - len(cleaned))
    return ["unknown"] * len(STAGE_NAMES)


def required_modes_for_instance(instance: dict[str, Any]) -> list[str]:
    metadata = instance.get("metadata", {})
    return normalize_modes(metadata.get("deliberation_requirement_summary", {}))


def mismatch_bucket(actual_modes: list[str], required_modes: list[str]) -> str:
    fast_on_deep = any(a == "fast" and r == "deep" for a, r in zip(actual_modes, required_modes))
    deep_on_fast = any(a == "deep" and r == "fast" for a, r in zip(actual_modes, required_modes))
    if fast_on_deep and deep_on_fast:
        return "both_mismatch_types"
    if fast_on_deep:
        return "fast_on_deep_required"
    if deep_on_fast:
        return "deep_on_fast_required"
    return "all_stage_modes_match"


def success_status(ep: dict[str, Any], instance: dict[str, Any]) -> dict[str, Any]:
    clean_success = (
        bool(ep.get("exact_match"))
        and not bool(ep.get("subset_mismatch"))
        and int(ep.get("policy_violation_count", 0) or 0) == 0
    )
    assistant_required = bool(
        instance.get("metadata", {}).get("contains_assistant_side_action", False)
    )
    assistant_tool_calls = int(ep.get("assistant_side_mutating_tool_calls_made", 0) or 0)
    final_action = str(ep.get("final_action"))
    if not assistant_required:
        assistant_status = "not_required"
        assistant_success = None
    elif clean_success and final_action == "transfer":
        assistant_status = "clean_transfer_no_assistant_tool"
        assistant_success = True
    elif clean_success and assistant_tool_calls > 0 and final_action in {"repair_all", "repair_subset"}:
        assistant_status = "assisted_repair_success"
        assistant_success = True
    elif assistant_tool_calls > 0:
        assistant_status = "assistant_tool_called_but_not_clean"
        assistant_success = False
    else:
        assistant_status = "assistant_required_but_no_tool_or_not_clean"
        assistant_success = False
    return {
        "clear_execution_success": clean_success,
        "assistant_side_required": assistant_required,
        "assistant_side_tool_calls": assistant_tool_calls,
        "assistant_execution_status": assistant_status,
        "assistant_execution_success": assistant_success,
    }


def summarize_rows(rows: list[dict[str, Any]], keys: list[str]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(key) for key in keys)].append(row)
    summaries: list[dict[str, Any]] = []
    for group_key, group_rows in sorted(grouped.items(), key=lambda item: tuple(str(x) for x in item[0])):
        assistant_applicable = [
            row for row in group_rows if row.get("assistant_execution_success") is not None
        ]
        summary = {key: value for key, value in zip(keys, group_key)}
        summary.update(
            {
                "episode_count": len(group_rows),
                "terminal_cost_avg": mean([float(row["terminal_cost"]) for row in group_rows]),
                "reasoning_cost_avg": mean([float(row["reasoning_cost"]) for row in group_rows]),
                "total_cost_avg": mean([float(row["total_cost"]) for row in group_rows]),
                "exact_match_rate": mean(
                    [float(bool(row.get("exact_match", False))) for row in group_rows]
                ),
                "clear_execution_success_rate": mean(
                    [float(bool(row["clear_execution_success"])) for row in group_rows]
                ),
                "assistant_required_count": len(assistant_applicable),
                "assistant_execution_success_rate_applicable": mean(
                    [
                        float(bool(row["assistant_execution_success"]))
                        for row in assistant_applicable
                    ]
                ),
                "actual_majority_distribution": json.dumps(
                    distribution(
                        [
                            row.get("actual_majority", row.get("actual_mode", "unknown"))
                            for row in group_rows
                        ]
                    ),
                    ensure_ascii=False,
                ),
                "required_majority_distribution": json.dumps(
                    distribution(
                        [
                            row.get("required_majority", row.get("required_mode", "unknown"))
                            for row in group_rows
                        ]
                    ),
                    ensure_ascii=False,
                ),
            }
        )
        summaries.append(summary)
    return summaries


def markdown_table(rows: list[dict[str, Any]], columns: list[str], *, max_rows: int | None = None) -> str:
    if max_rows is not None:
        rows = rows[:max_rows]
    if not rows:
        return "_empty_\n"
    def fmt(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        text = str(value)
        return text.replace("|", "\\|")
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join([header, divider, *body]) + "\n"


def build_analysis(run_dir: Path, output_dir: Path) -> None:
    run_config = load_json(run_dir / "run_config.json")
    schedule = load_json(run_dir / "schedule.json")
    dataset = load_json(Path(run_config["dataset"]))
    switch_episode = int((run_config.get("schedule_metadata") or {}).get("switch_episode", 0))
    methods = list(run_config["methods"])

    by_dataset_index = {idx: instance for idx, instance in enumerate(dataset)}
    schedule_by_episode = {int(row["episode_index"]): row for row in schedule}
    episode_rows: list[dict[str, Any]] = []
    stage_mode_rows: list[dict[str, Any]] = []
    snapshot_summary_rows: list[dict[str, Any]] = []

    for method in methods:
        episodes_path = run_dir / method / "episodes.json"
        if not episodes_path.exists():
            raise FileNotFoundError(f"Missing completed episodes for {method}: {episodes_path}")
        episodes = load_json(episodes_path)
        snapshot_rows = read_csv(run_dir / method / "post_switch_tree_probability_snapshot.csv")
        if not snapshot_rows:
            snapshot_rows = read_csv(run_dir / method / "post_switch_probability_snapshot.csv")
        depths = sorted({int(row["prefix_depth"]) for row in snapshot_rows}) if snapshot_rows else []
        parent_prefixes = {row["prefix"] for row in snapshot_rows}
        snapshot_summary_rows.append(
            {
                "method": method,
                "snapshot_rows": len(snapshot_rows),
                "parent_prefix_count": len(parent_prefixes),
                "prefix_depths": ",".join(str(depth) for depth in depths),
                "freeze_mode": (
                    snapshot_rows[0].get("freeze_mode")
                    if snapshot_rows
                    else run_config.get("post_switch_probability_freeze_mode")
                ),
            }
        )

        for ep in episodes:
            episode_index = int(ep["episode_index"])
            schedule_row = schedule_by_episode[episode_index]
            instance = by_dataset_index[int(ep["dataset_index"])]
            metadata = instance.get("metadata", {})
            required_modes = required_modes_for_instance(instance)
            actual_modes = normalize_modes(ep.get("family_deliberation_modes"))
            required_majority = mode_majority(required_modes)
            actual_majority = mode_majority(actual_modes)
            bucket = mismatch_bucket(actual_modes, required_modes)
            success = success_status(ep, instance)
            row = {
                "method": method,
                "episode_index": episode_index,
                "episode_1based": episode_index + 1,
                "phase": "pre_switch" if episode_index < switch_episode else "post_switch",
                "schedule_phase": ep.get("schedule_phase"),
                "repeat_index": ep.get("repeat_index"),
                "position_in_cycle": ep.get("position_in_cycle"),
                "dataset_index": ep.get("dataset_index"),
                "original_task_id": ep.get("original_task_id"),
                "expected_terminal_action": metadata.get("expected_terminal_action"),
                "oracle_action": ep.get("oracle_action"),
                "final_action": ep.get("final_action"),
                "terminal_cost": float(ep.get("raw_terminal_penalty", 0.0) or 0.0),
                "reasoning_cost": float(ep.get("raw_reasoning_cost_component", 0.0) or 0.0),
                "total_cost": float(ep.get("raw_total_cost", 0.0) or 0.0),
                "exact_match": bool(ep.get("exact_match")),
                "subset_mismatch": bool(ep.get("subset_mismatch")),
                "policy_violation_count": int(ep.get("policy_violation_count", 0) or 0),
                "actual_modes": "/".join(actual_modes),
                "required_modes": "/".join(required_modes),
                "actual_fast_count": sum(1 for mode in actual_modes if mode == "fast"),
                "actual_deep_count": sum(1 for mode in actual_modes if mode == "deep"),
                "required_fast_count": sum(1 for mode in required_modes if mode == "fast"),
                "required_deep_count": sum(1 for mode in required_modes if mode == "deep"),
                "actual_majority": actual_majority,
                "required_majority": required_majority,
                "majority_pair": f"{actual_majority}_path__{required_majority}_task",
                "mismatch_bucket": bucket,
                "post_switch_probability_freeze_active": bool(
                    ep.get("post_switch_probability_freeze_active", False)
                ),
                "post_switch_probability_freeze_mode": ep.get("post_switch_probability_freeze_mode"),
                "root_child_id": ep.get("root_child_id"),
                "root_selection_mode": ep.get("root_selection_mode"),
                "leaf_type": ep.get("leaf_type"),
                "selected_shared_path": bool(ep.get("selected_shared_path")),
                "selected_unshared_path": bool(ep.get("selected_unshared_path")),
                **success,
            }
            episode_rows.append(row)
            for stage_idx, (actual, required) in enumerate(zip(actual_modes, required_modes), start=1):
                stage_mode_rows.append(
                    {
                        "method": method,
                        "episode_index": episode_index,
                        "phase": row["phase"],
                        "stage": f"stage{stage_idx}",
                        "actual_mode": actual,
                        "required_mode": required,
                        "mode_pair": f"{actual}_path__{required}_task",
                        "terminal_cost": row["terminal_cost"],
                        "reasoning_cost": row["reasoning_cost"],
                        "total_cost": row["total_cost"],
                        "clear_execution_success": row["clear_execution_success"],
                    }
                )

    method_phase = summarize_rows(episode_rows, ["method", "phase"])
    method_total = summarize_rows(episode_rows, ["method"])
    mismatch_summary = summarize_rows(episode_rows, ["method", "mismatch_bucket"])
    majority_pair_summary = summarize_rows(episode_rows, ["method", "majority_pair"])
    stage_mode_summary = summarize_rows(stage_mode_rows, ["method", "stage", "mode_pair"])
    terminal_action_summary = summarize_rows(episode_rows, ["method", "phase", "expected_terminal_action"])
    assistant_status_summary = summarize_rows(episode_rows, ["method", "assistant_execution_status"])

    write_csv(output_dir / "episode_cost_success_mode_analysis.csv", episode_rows)
    write_json(output_dir / "episode_cost_success_mode_analysis.json", episode_rows)
    write_csv(
        output_dir / "episode_compact_view.csv",
        [
            {
                key: row[key]
                for key in [
                    "method",
                    "episode_1based",
                    "phase",
                    "expected_terminal_action",
                    "terminal_cost",
                    "reasoning_cost",
                    "total_cost",
                    "clear_execution_success",
                    "assistant_execution_status",
                    "actual_modes",
                    "required_modes",
                    "majority_pair",
                    "mismatch_bucket",
                    "final_action",
                ]
            }
            for row in episode_rows
        ],
    )
    write_csv(output_dir / "summary_cost_success_mode.csv", [*method_total, *method_phase])
    write_json(
        output_dir / "summary_cost_success_mode.json",
        {"total": method_total, "phase": method_phase},
    )
    write_csv(output_dir / "snapshot_summary.csv", snapshot_summary_rows)
    write_csv(output_dir / "mismatch_bucket_cost_summary.csv", mismatch_summary)
    write_csv(output_dir / "majority_pair_cost_summary.csv", majority_pair_summary)
    write_csv(output_dir / "stage_mode_pair_cost_summary.csv", stage_mode_summary)
    write_csv(output_dir / "terminal_action_cost_summary.csv", terminal_action_summary)
    write_csv(output_dir / "assistant_execution_status_summary.csv", assistant_status_summary)

    report = []
    report.append(f"# Full-tree Fixed Probability LLM Sanity Report\n")
    report.append(f"- Run dir: `{run_dir}`")
    report.append(f"- Experiment name: `{run_dir.name}`")
    report.append(f"- Freeze mode: `{run_config.get('post_switch_probability_freeze_mode')}`")
    report.append(f"- Model: `{run_config.get('model')}`")
    report.append(f"- Methods: `{', '.join(methods)}`")
    report.append(f"- Repeats/horizon: `{run_config.get('repeats')}` / `{run_config.get('horizon')}`")
    report.append(f"- Switch episode index: `{switch_episode}` (1-based episode `{switch_episode + 1}`)")
    report.append(f"- Schedule: 25 pre-switch episodes from the trap bucket, 75 post-switch episodes from the target bucket for the 10x10 run.\n")

    report.append("## Tree-freeze validation\n")
    report.append(markdown_table(snapshot_summary_rows, ["method", "snapshot_rows", "parent_prefix_count", "prefix_depths", "freeze_mode"]))

    total_sorted = sorted(method_total, key=lambda row: float(row["total_cost_avg"]))
    report.append("## Total cost by method\n")
    report.append(markdown_table(total_sorted, [
        "method",
        "episode_count",
        "terminal_cost_avg",
        "reasoning_cost_avg",
        "total_cost_avg",
        "exact_match_rate",
        "clear_execution_success_rate",
        "assistant_required_count",
        "assistant_execution_success_rate_applicable",
    ]))

    phase_sorted = sorted(method_phase, key=lambda row: (str(row["method"]), str(row["phase"])))
    report.append("## Pre/post switch cost\n")
    report.append(markdown_table(phase_sorted, [
        "method",
        "phase",
        "episode_count",
        "terminal_cost_avg",
        "reasoning_cost_avg",
        "total_cost_avg",
        "clear_execution_success_rate",
        "assistant_required_count",
        "assistant_execution_success_rate_applicable",
        "actual_majority_distribution",
        "required_majority_distribution",
    ]))

    report.append("## Path majority vs task requirement\n")
    report.append(markdown_table(sorted(majority_pair_summary, key=lambda row: (str(row["method"]), str(row["majority_pair"]))), [
        "method",
        "majority_pair",
        "episode_count",
        "terminal_cost_avg",
        "reasoning_cost_avg",
        "total_cost_avg",
        "clear_execution_success_rate",
    ]))

    report.append("## Stage mismatch buckets\n")
    report.append(markdown_table(sorted(mismatch_summary, key=lambda row: (str(row["method"]), str(row["mismatch_bucket"]))), [
        "method",
        "mismatch_bucket",
        "episode_count",
        "terminal_cost_avg",
        "reasoning_cost_avg",
        "total_cost_avg",
        "clear_execution_success_rate",
    ]))

    report.append("## Terminal action split\n")
    report.append(markdown_table(sorted(terminal_action_summary, key=lambda row: (str(row["method"]), str(row["phase"]), str(row["expected_terminal_action"]))), [
        "method",
        "phase",
        "expected_terminal_action",
        "episode_count",
        "terminal_cost_avg",
        "reasoning_cost_avg",
        "total_cost_avg",
        "clear_execution_success_rate",
    ]))

    report.append("## Assistant execution status\n")
    report.append(markdown_table(sorted(assistant_status_summary, key=lambda row: (str(row["method"]), str(row["assistant_execution_status"]))), [
        "method",
        "assistant_execution_status",
        "episode_count",
        "terminal_cost_avg",
        "reasoning_cost_avg",
        "total_cost_avg",
        "clear_execution_success_rate",
    ]))

    report.append("## Output files\n")
    report.append("- `episode_cost_success_mode_analysis.csv`: full per-episode table")
    report.append("- `episode_compact_view.csv`: compact per-episode table")
    report.append("- `summary_cost_success_mode.csv`: total and pre/post summaries")
    report.append("- `majority_pair_cost_summary.csv`: mostly-fast/deep path vs mostly-fast/deep task table")
    report.append("- `mismatch_bucket_cost_summary.csv`: fast-on-deep/deep-on-fast mismatch table")
    report.append("- `stage_mode_pair_cost_summary.csv`: stage-level mode-pair table")
    report.append("- `terminal_action_cost_summary.csv`: repair_all/repair_subset/transfer split")
    report.append("- `assistant_execution_status_summary.csv`: assistant-side status split\n")

    (output_dir / "report.md").write_text("\n".join(report), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    output_dir = args.output_dir or (args.run_dir / "fulltree_freeze_analysis")
    build_analysis(args.run_dir, output_dir)
    print(output_dir)


if __name__ == "__main__":
    main()
