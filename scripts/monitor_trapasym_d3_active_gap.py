#!/usr/bin/env python
"""Monitor trap-asym LLM run for trap lock-in and post-switch PS loss."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    rows.sort(key=lambda row: int(row.get("episode_index", -1)))
    return rows


def read_summary(run_dir: Path, method: str) -> dict[str, Any]:
    for name in ("summary_partial.json", "summary.json"):
        path = run_dir / method / name
        if path.exists():
            try:
                data = load_json(path)
                if isinstance(data, dict):
                    return data
            except Exception as exc:
                return {"status": "read_error", "error": str(exc)}
    return {"status": "missing", "completed_episodes": 0}


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def float_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = row.get(key)
        if value is not None:
            values.append(float(value))
    return values


def last5_trap_summary(
    *,
    run_dir: Path,
    methods: list[str],
    switch_episode: int,
) -> dict[str, Any]:
    start = max(0, switch_episode - 5)
    result: dict[str, Any] = {}
    for method in methods:
        rows = load_jsonl(run_dir / method / "episodes.partial.jsonl")
        last5 = [
            row
            for row in rows
            if start <= int(row.get("episode_index", -1)) < switch_episode
        ]
        result[method] = {
            "count": len(last5),
            "episode_indices": [int(row["episode_index"]) for row in last5],
            "root_trap_subtree_prob_mean": mean(
                float_values(last5, "root_trap_subtree_prob")
            ),
            "stage4_trap_child_prob_mean": mean(
                float_values(last5, "stage4_trap_child_prob")
            ),
            "all_fast_trap_route_prob_mean": mean(
                float_values(last5, "all_fast_trap_route_prob")
            ),
            "selected_trap_like_rate": mean(
                [1.0 if row.get("family_trap_like_path") else 0.0 for row in last5]
            ),
            "selected_all_fast_trap_rate": mean(
                [
                    1.0
                    if row.get("family_trap_like_path")
                    and row.get("family_deliberation_modes") == ["fast"] * 5
                    else 0.0
                    for row in last5
                ]
            ),
        }
    return result


def post_switch_means(run_dir: Path, method: str, switch_episode: int) -> dict[str, float]:
    rows = [
        row
        for row in load_jsonl(run_dir / method / "episodes.partial.jsonl")
        if int(row.get("episode_index", -1)) >= switch_episode
    ]
    return {
        "post_episode_count": float(len(rows)),
        "post_total_cost_mean": mean(float_values(rows, "total_cost")) or 0.0,
        "post_raw_total_cost_mean": mean(float_values(rows, "raw_total_cost")) or 0.0,
        "post_raw_terminal_penalty_mean": mean(float_values(rows, "raw_terminal_penalty"))
        or 0.0,
    }


def write_report(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload['experiment_name']} monitor report",
        "",
        f"- decision: `{payload['decision']}`",
        f"- reason: {payload['reason']}",
        f"- switch_episode: `{payload['switch_episode']}`",
        "",
        "## Last-5 Pre-Switch Trap Probabilities",
        "",
        "| method | n | root trap | stage4 trap | all-fast trap route | selected trap-like | selected all-fast |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method, row in payload.get("last5_trap_summary", {}).items():
        def fmt(value: Any) -> str:
            return "NA" if value is None else f"{float(value):.3f}"

        lines.append(
            "| "
            + " | ".join(
                [
                    method,
                    str(row.get("count", 0)),
                    fmt(row.get("root_trap_subtree_prob_mean")),
                    fmt(row.get("stage4_trap_child_prob_mean")),
                    fmt(row.get("all_fast_trap_route_prob_mean")),
                    fmt(row.get("selected_trap_like_rate")),
                    fmt(row.get("selected_all_fast_trap_rate")),
                ]
            )
            + " |"
        )
    if payload.get("post_switch_means"):
        lines.extend(
            [
                "",
                "## Post-Switch Means",
                "",
                "| method | post n | total mean | raw mean | terminal mean |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for method, row in payload["post_switch_means"].items():
            lines.append(
                f"| {method} | {int(row['post_episode_count'])} | "
                f"{row['post_total_cost_mean']:.4f} | "
                f"{row['post_raw_total_cost_mean']:.3f} | "
                f"{row['post_raw_terminal_penalty_mean']:.3f} |"
            )
    path.write_text("\n".join(lines) + "\n")


def stop_tmux(session: str | None) -> None:
    if session:
        subprocess.run(["tmux", "kill-session", "-t", session], check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--ps-method", default="risky_ps")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["direct_multistage_exp3", "epsilon_exp3"],
    )
    parser.add_argument("--all-methods", nargs="+")
    parser.add_argument("--trap-root-threshold", type=float, default=0.5)
    parser.add_argument("--trap-selected-threshold", type=float, default=0.6)
    parser.add_argument("--episode-threshold", type=int, default=75)
    parser.add_argument("--post-total-gap-threshold", type=float, default=0.02)
    parser.add_argument("--post-raw-gap-threshold", type=float, default=0.75)
    parser.add_argument("--tmux-method-session")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_config = load_json(args.run_dir / "run_config.json")
    switch_episode = int(run_config["schedule_metadata"]["switch_episode"])
    methods = args.all_methods or list(run_config["methods"])
    watched_methods = [args.ps_method, *args.baseline_methods]
    log_path = args.run_dir / "trap_d3_active_gap_monitor.log"
    decision_path = args.run_dir / "trap_d3_active_gap_decision.json"
    report_path = args.run_dir / "trap_d3_active_gap_report.md"
    checked_trap = False

    while True:
        summaries = {method: read_summary(args.run_dir, method) for method in methods}
        completed = {
            method: int(summaries[method].get("completed_episodes", 0) or 0)
            for method in methods
        }
        status = {
            "ts": time.time(),
            "completed": completed,
            "summary_status": {
                method: summaries[method].get("status") for method in methods
            },
        }
        with log_path.open("a") as handle:
            handle.write(json.dumps(status, sort_keys=True) + "\n")

        if not checked_trap and all(
            completed.get(method, 0) >= switch_episode for method in methods
        ):
            checked_trap = True
            trap_summary = last5_trap_summary(
                run_dir=args.run_dir,
                methods=methods,
                switch_episode=switch_episode,
            )
            no_method_locked = all(
                (row.get("root_trap_subtree_prob_mean") or 0.0)
                < args.trap_root_threshold
                and (row.get("selected_trap_like_rate") or 0.0)
                < args.trap_selected_threshold
                for row in trap_summary.values()
            )
            payload = {
                **status,
                "experiment_name": args.experiment_name,
                "switch_episode": switch_episode,
                "decision": "early_stop" if no_method_locked else "continue",
                "reason": (
                    "no algorithm reached trap lock-in over the last five pre-switch episodes"
                    if no_method_locked
                    else "at least one algorithm shows enough trap lock-in to continue"
                ),
                "last5_trap_summary": trap_summary,
                "thresholds": {
                    "trap_root_threshold": args.trap_root_threshold,
                    "trap_selected_threshold": args.trap_selected_threshold,
                },
            }
            decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            write_report(report_path, payload)
            if no_method_locked:
                stop_tmux(args.tmux_method_session)
                break

        if all(completed.get(method, 0) >= args.episode_threshold for method in watched_methods):
            post = {
                method: post_switch_means(args.run_dir, method, switch_episode)
                for method in watched_methods
            }
            best_baseline_method, best_baseline = min(
                ((method, post[method]) for method in args.baseline_methods),
                key=lambda item: item[1]["post_total_cost_mean"],
            )
            ps = post[args.ps_method]
            total_gap = (
                ps["post_total_cost_mean"] - best_baseline["post_total_cost_mean"]
            )
            raw_gap = ps["post_raw_total_cost_mean"] - best_baseline[
                "post_raw_total_cost_mean"
            ]
            early_stop = (
                total_gap >= args.post_total_gap_threshold
                or raw_gap >= args.post_raw_gap_threshold
            )
            payload = {
                **status,
                "experiment_name": args.experiment_name,
                "switch_episode": switch_episode,
                "decision": "early_stop" if early_stop else "continue",
                "reason": (
                    "PS is clearly behind the best EXP3 baseline at the episode threshold"
                    if early_stop
                    else "PS is not clearly behind at the episode threshold"
                ),
                "last5_trap_summary": last5_trap_summary(
                    run_dir=args.run_dir,
                    methods=methods,
                    switch_episode=switch_episode,
                ),
                "post_switch_means": post,
                "ps_method": args.ps_method,
                "best_baseline_method": best_baseline_method,
                "post_total_gap_vs_best_baseline": total_gap,
                "post_raw_gap_vs_best_baseline": raw_gap,
                "thresholds": {
                    "episode_threshold": args.episode_threshold,
                    "post_total_gap_threshold": args.post_total_gap_threshold,
                    "post_raw_gap_threshold": args.post_raw_gap_threshold,
                },
            }
            decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            write_report(report_path, payload)
            if early_stop:
                stop_tmux(args.tmux_method_session)
            break

        if all(
            summaries[method].get("status") in {"completed", "failed", "error"}
            for method in methods
        ):
            payload = {
                **status,
                "experiment_name": args.experiment_name,
                "switch_episode": switch_episode,
                "decision": "completed",
                "reason": "all methods finished before an early-stop condition fired",
                "last5_trap_summary": last5_trap_summary(
                    run_dir=args.run_dir,
                    methods=methods,
                    switch_episode=switch_episode,
                ),
                "post_switch_means": {
                    method: post_switch_means(args.run_dir, method, switch_episode)
                    for method in watched_methods
                },
            }
            decision_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
            write_report(report_path, payload)
            break

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
