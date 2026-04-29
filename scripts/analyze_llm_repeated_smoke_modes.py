#!/usr/bin/env python3
"""Build mode/cost diagnostics for repeated LLM smoke outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any


STAGES = ("stage1", "stage2", "stage3", "stage4", "stage5")


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


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


def fmt(value: Any, digits: int = 3) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if value is None:
        return ""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(number):
        return ""
    return f"{number:.{digits}f}"


def md_table(rows: list[dict[str, Any]], columns: list[tuple[str, str]], digits: int = 3) -> list[str]:
    if not rows:
        return ["_No rows._"]
    out = [
        "| " + " | ".join(label for _, label in columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(fmt(row.get(key), digits) for key, _ in columns) + " |")
    return out


def task_requirements_by_id(data_path: Path) -> dict[str, list[str]]:
    rows = load_json(data_path)
    out: dict[str, list[str]] = {}
    for row in rows:
        metadata = row.get("metadata") or {}
        summary = metadata.get("deliberation_requirement_summary") or {}
        out[str(row.get("original_task_id"))] = [
            str(summary.get(stage, "fast")).strip().lower() for stage in STAGES
        ]
    return out


def classify_mismatch(required: list[str], actual: list[str]) -> tuple[str, int, int, int]:
    fast_on_deep = sum(1 for req, act in zip(required, actual) if req == "deep" and act == "fast")
    deep_on_fast = sum(1 for req, act in zip(required, actual) if req == "fast" and act == "deep")
    match = sum(1 for req, act in zip(required, actual) if req == act)
    if fast_on_deep and deep_on_fast:
        bucket = "both_mismatch_types"
    elif fast_on_deep:
        bucket = "fast_on_deep_required"
    elif deep_on_fast:
        bucket = "deep_on_fast_required"
    else:
        bucket = "all_stage_modes_match"
    return bucket, fast_on_deep, deep_on_fast, match


def majority_pair(required: list[str], actual: list[str]) -> str:
    actual_majority = "mostly_fast" if actual.count("fast") > actual.count("deep") else "mostly_deep"
    required_majority = (
        "mostly_fast_required"
        if required.count("fast") > required.count("deep")
        else "mostly_deep_required"
    )
    return f"{actual_majority}_vs_{required_majority}"


def clean_float(row: dict[str, Any], key: str) -> float:
    try:
        return float(row.get(key, 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def summarize(rows: list[dict[str, Any]], group_keys: list[str]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(key) for key in group_keys)].append(row)
    out: list[dict[str, Any]] = []
    for key_tuple, group in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        payload = {key: value for key, value in zip(group_keys, key_tuple)}
        payload.update(
            {
                "n": len(group),
                "terminal": mean(clean_float(row, "raw_terminal_penalty") for row in group),
                "reasoning": mean(clean_float(row, "raw_reasoning_cost_component") for row in group),
                "path": mean(clean_float(row, "raw_path_cost_component") for row in group),
                "total": mean(clean_float(row, "raw_total_cost") for row in group),
                "exact": mean(float(row["exact_match"]) for row in group),
                "clear_success_proxy": mean(float(row["clear_success_proxy"]) for row in group),
                "auxiliary_success_proxy": mean(float(row["auxiliary_success_proxy"]) for row in group),
                "strict_clean": mean(float(row["strict_clean_success"]) for row in group),
                "policy_clean": mean(float(row["policy_clean"]) for row in group),
                "subset_clean": mean(float(row["subset_clean"]) for row in group),
                "fast_on_deep": mean(clean_float(row, "fast_on_deep_count") for row in group),
                "deep_on_fast": mean(clean_float(row, "deep_on_fast_count") for row in group),
                "match_count": mean(clean_float(row, "match_count") for row in group),
                "llm_calls": mean(clean_float(row, "llm_call_count") for row in group),
            }
        )
        out.append(payload)
    return out


def build_stage_pair_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expanded: list[dict[str, Any]] = []
    for row in rows:
        required = str(row["required_modes"]).split("/")
        actual = str(row["actual_modes"]).split("/")
        for index, (req, act) in enumerate(zip(required, actual), start=1):
            item = dict(row)
            item["stage"] = f"stage{index}"
            item["required"] = req
            item["actual"] = act
            expanded.append(item)
            all_item = dict(row)
            all_item["stage"] = "ALL"
            all_item["required"] = req
            all_item["actual"] = act
            expanded.append(all_item)
    groups: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in expanded:
        groups[
            (
                str(row["method"]),
                str(row["stage"]),
                str(row["required"]),
                str(row["actual"]),
            )
        ].append(row)
    out: list[dict[str, Any]] = []
    for (method, stage, required, actual), group in sorted(groups.items()):
        episode_keys = {(row["method"], row["episode_index"]) for row in group}
        out.append(
            {
                "method": method,
                "stage": stage,
                "required": required,
                "actual": actual,
                "n_stage_observations": len(group),
                "episode_n": len(episode_keys),
                "terminal": mean(clean_float(row, "raw_terminal_penalty") for row in group),
                "reasoning": mean(clean_float(row, "raw_reasoning_cost_component") for row in group),
                "total": mean(clean_float(row, "raw_total_cost") for row in group),
                "exact": mean(float(row["exact_match"]) for row in group),
                "strict_clean": mean(float(row["strict_clean_success"]) for row in group),
                "clear_success_proxy": mean(float(row["clear_success_proxy"]) for row in group),
                "auxiliary_success_proxy": mean(float(row["auxiliary_success_proxy"]) for row in group),
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--title", default="LLM Repeated Smoke Mode Diagnostic")
    args = parser.parse_args()

    run_config = load_json(args.run_dir / "run_config.json")
    requirements = task_requirements_by_id(Path(run_config["dataset"]))

    episode_rows: list[dict[str, Any]] = []
    for method in run_config["methods"]:
        method_episodes = load_json(args.run_dir / method / "episodes.json")
        for episode in method_episodes:
            task_id = str(episode.get("original_task_id"))
            required = requirements.get(task_id, ["fast"] * len(STAGES))
            actual = [
                "deep" if str(value).strip().lower() == "deep" else "fast"
                for value in episode.get("family_deliberation_modes", [])
            ]
            if len(actual) != len(STAGES):
                actual = ["fast"] * len(STAGES)
            bucket, fast_on_deep, deep_on_fast, match = classify_mismatch(required, actual)
            exact = bool(episode.get("exact_match", False))
            subset_clean = not bool(episode.get("subset_mismatch", False))
            policy_clean = int(episode.get("policy_violation_count", 0) or 0) == 0
            clear_success = exact and subset_clean
            auxiliary_success = policy_clean
            strict_clean = clear_success and auxiliary_success
            route_labels = list(episode.get("family_route_labels", []) or [])
            row = {
                "method": method,
                "episode_index": int(episode.get("episode_index", 0)),
                "repeat_index": int(episode.get("repeat_index", 0)),
                "dataset_index": int(episode.get("dataset_index", 0)),
                "schedule_phase": episode.get("schedule_phase"),
                "task_bucket": episode.get("task_bucket"),
                "is_specialist_task": bool(episode.get("is_specialist_task", False)),
                "oracle_action": episode.get("oracle_action"),
                "final_action": episode.get("final_action"),
                "exact_match": exact,
                "policy_clean": policy_clean,
                "subset_clean": subset_clean,
                "clear_success_proxy": clear_success,
                "auxiliary_success_proxy": auxiliary_success,
                "strict_clean_success": strict_clean,
                "policy_violation_count": int(episode.get("policy_violation_count", 0) or 0),
                "subset_mismatch": bool(episode.get("subset_mismatch", False)),
                "raw_terminal_penalty": clean_float(episode, "raw_terminal_penalty"),
                "raw_outcome_penalty": clean_float(episode, "raw_outcome_penalty"),
                "raw_policy_penalty": clean_float(episode, "raw_policy_penalty"),
                "raw_reasoning_cost_component": clean_float(episode, "raw_reasoning_cost_component"),
                "raw_path_cost_component": clean_float(episode, "raw_path_cost_component"),
                "raw_total_cost": clean_float(episode, "raw_total_cost"),
                "llm_call_count": int(episode.get("llm_call_count", 0) or 0),
                "prompt_tokens_total": clean_float(episode, "prompt_tokens_total"),
                "completion_tokens_total": clean_float(episode, "completion_tokens_total"),
                "total_tokens_total": clean_float(episode, "total_tokens_total"),
                "actual_modes": "/".join(actual),
                "required_modes": "/".join(required),
                "match_count": match,
                "fast_on_deep_count": fast_on_deep,
                "deep_on_fast_count": deep_on_fast,
                "actual_fast_count": actual.count("fast"),
                "actual_deep_count": actual.count("deep"),
                "required_fast_count": required.count("fast"),
                "required_deep_count": required.count("deep"),
                "mismatch_bucket": bucket,
                "majority_pair": majority_pair(required, actual),
                "route_labels": " > ".join(route_labels),
                "route_bucket": episode.get("family_behavior_archetype") or "unknown",
                "selected_shared_path": bool(episode.get("selected_shared_path", False)),
                "selected_unshared_path": bool(episode.get("selected_unshared_path", False)),
                "original_task_id": task_id,
            }
            episode_rows.append(row)

    summary_rows = summarize(episode_rows, ["method"])
    split_rows = []
    for split_name, predicate in [
        ("all", lambda row: True),
        ("pre", lambda row: row["schedule_phase"] == "trap_pre_switch"),
        ("post", lambda row: row["schedule_phase"] == "target_post_switch"),
        ("post_local_nontransfer", lambda row: row["schedule_phase"] == "target_post_switch" and row["oracle_action"] != "transfer"),
    ]:
        for row in summarize([row for row in episode_rows if predicate(row)], ["method"]):
            row = {"split": split_name, **row}
            split_rows.append(row)
    bucket_rows = summarize(episode_rows, ["method", "mismatch_bucket"])
    majority_rows = summarize(episode_rows, ["method", "majority_pair"])
    phase_majority_rows = summarize(episode_rows, ["method", "schedule_phase", "majority_pair"])
    stage_rows = build_stage_pair_rows(episode_rows)

    write_csv(args.run_dir / "episode_mode_cost_analysis.csv", episode_rows)
    write_csv(args.run_dir / "summary_mode_cost.csv", split_rows)
    write_json(args.run_dir / "summary_mode_cost.json", split_rows)
    write_csv(args.run_dir / "bucket_mode_cost.csv", bucket_rows)
    write_json(args.run_dir / "bucket_mode_cost.json", bucket_rows)
    write_csv(args.run_dir / "majority_mode_cost.csv", majority_rows)
    write_json(args.run_dir / "majority_mode_cost.json", majority_rows)
    write_csv(args.run_dir / "phase_majority_mode_cost.csv", phase_majority_rows)
    write_json(args.run_dir / "phase_majority_mode_cost.json", phase_majority_rows)
    write_csv(args.run_dir / "stage_mode_pair_cost.csv", stage_rows)
    write_json(args.run_dir / "stage_mode_pair_cost.json", stage_rows)

    compare_rows = sorted(summary_rows, key=lambda row: row["total"])
    report: list[str] = [
        f"# {args.title}",
        "",
        f"Date: {datetime.now().date().isoformat()}",
        "",
        f"Experiment name: `{args.run_dir.name}`",
        "",
        f"Output directory: `{args.run_dir}`",
        "",
        "## Setting",
        "",
    ]
    setting_rows = [
        {"field": "model", "value": run_config.get("model")},
        {"field": "executor", "value": run_config.get("executor_name")},
        {"field": "family", "value": run_config.get("family_kind")},
        {"field": "schedule", "value": run_config.get("schedule_mode")},
        {"field": "d / switch denominator", "value": run_config.get("switch_denominator")},
        {"field": "eta", "value": run_config.get("common_eta_override")},
        {"field": "epsilon", "value": run_config.get("common_epsilon_override")},
        {"field": "repeats", "value": run_config.get("repeats")},
        {"field": "horizon per method", "value": run_config.get("horizon")},
        {"field": "switch episode", "value": (run_config.get("schedule_metadata") or {}).get("switch_episode")},
        {"field": "methods", "value": ", ".join(run_config.get("methods", []))},
        {"field": "dataset", "value": run_config.get("dataset")},
        {"field": "schedule buckets", "value": run_config.get("schedule_buckets")},
    ]
    report.extend(md_table(setting_rows, [("field", "field"), ("value", "value")]))
    report.extend(
        [
            "",
            "## Main Cost And Success Summary",
            "",
            "Definitions: `clear_success_proxy = exact_match && subset_clean`; "
            "`auxiliary_success_proxy = policy_violation_count == 0`; "
            "`strict_clean = clear_success_proxy && auxiliary_success_proxy`. "
            "The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.",
            "",
        ]
    )
    report.extend(
        md_table(
            split_rows,
            [
                ("method", "method"),
                ("split", "split"),
                ("n", "n"),
                ("terminal", "terminal"),
                ("reasoning", "reasoning"),
                ("path", "path"),
                ("total", "total"),
                ("exact", "exact"),
                ("clear_success_proxy", "clear"),
                ("auxiliary_success_proxy", "aux"),
                ("strict_clean", "strict"),
                ("fast_on_deep", "fast-on-deep"),
                ("deep_on_fast", "deep-on-fast"),
            ],
        )
    )
    report.extend(["", "## Ranking By Raw Total Cost", ""])
    ranking_rows = [
        {"rank": index + 1, **row}
        for index, row in enumerate(compare_rows)
    ]
    report.extend(
        md_table(
            ranking_rows,
            [
                ("rank", "rank"),
                ("method", "method"),
                ("total", "raw total"),
                ("terminal", "terminal"),
                ("reasoning", "reasoning"),
                ("path", "path"),
                ("exact", "exact"),
                ("strict_clean", "strict"),
            ],
        )
    )
    report.extend(["", "## Mode-Mismatch Bucket Summary", ""])
    report.extend(
        md_table(
            bucket_rows,
            [
                ("method", "method"),
                ("mismatch_bucket", "bucket"),
                ("n", "n"),
                ("terminal", "terminal"),
                ("reasoning", "reasoning"),
                ("total", "total"),
                ("clear_success_proxy", "clear"),
                ("auxiliary_success_proxy", "aux"),
                ("strict_clean", "strict"),
                ("fast_on_deep", "avg fast-on-deep"),
                ("deep_on_fast", "avg deep-on-fast"),
            ],
        )
    )
    report.extend(["", "## Majority Fast/Deep Pair Summary", ""])
    report.extend(
        md_table(
            majority_rows,
            [
                ("method", "method"),
                ("majority_pair", "majority pair"),
                ("n", "n"),
                ("terminal", "terminal"),
                ("reasoning", "reasoning"),
                ("total", "total"),
                ("clear_success_proxy", "clear"),
                ("auxiliary_success_proxy", "aux"),
                ("strict_clean", "strict"),
            ],
        )
    )
    report.extend(["", "## Phase + Majority Pair Summary", ""])
    report.extend(
        md_table(
            phase_majority_rows,
            [
                ("method", "method"),
                ("schedule_phase", "phase"),
                ("majority_pair", "majority pair"),
                ("n", "n"),
                ("terminal", "terminal"),
                ("reasoning", "reasoning"),
                ("total", "total"),
                ("strict_clean", "strict"),
            ],
        )
    )
    report.extend(["", "## Stage-Level Required/Actual Mode Pair Summary", ""])
    all_stage_rows = [row for row in stage_rows if row["stage"] == "ALL"]
    report.extend(
        md_table(
            all_stage_rows,
            [
                ("method", "method"),
                ("required", "required"),
                ("actual", "actual"),
                ("n_stage_observations", "n stage obs"),
                ("episode_n", "episode n"),
                ("terminal", "terminal"),
                ("reasoning", "reasoning"),
                ("total", "total"),
                ("clear_success_proxy", "clear"),
                ("auxiliary_success_proxy", "aux"),
                ("strict_clean", "strict"),
            ],
        )
    )
    report.extend(["", "## Episode-Level Table", ""])
    report.append("Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.")
    report.append("")
    compact = episode_rows[:80]
    report.extend(
        md_table(
            compact,
            [
                ("method", "method"),
                ("episode_index", "ep"),
                ("schedule_phase", "phase"),
                ("oracle_action", "oracle"),
                ("final_action", "final"),
                ("raw_terminal_penalty", "terminal"),
                ("raw_reasoning_cost_component", "reasoning"),
                ("raw_total_cost", "total"),
                ("clear_success_proxy", "clear"),
                ("auxiliary_success_proxy", "aux"),
                ("strict_clean_success", "strict"),
                ("required_modes", "required"),
                ("actual_modes", "actual"),
                ("mismatch_bucket", "mismatch"),
            ],
        )
    )
    report.extend(["", "## Schedule Composition", ""])
    composition = Counter((row["schedule_phase"], row["oracle_action"]) for row in episode_rows if row["method"] == run_config["methods"][0])
    comp_rows = [
        {"phase": phase, "oracle_action": action, "n": count}
        for (phase, action), count in sorted(composition.items())
    ]
    report.extend(md_table(comp_rows, [("phase", "phase"), ("oracle_action", "oracle"), ("n", "n")]))
    write_text(args.run_dir / "report.md", "\n".join(report) + "\n")
    print(args.run_dir / "report.md")


if __name__ == "__main__":
    main()
