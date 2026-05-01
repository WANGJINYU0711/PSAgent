#!/usr/bin/env python
"""Monitor partial repeated-smoke summaries and stop a tmux run if needed."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--ps-method", default="risky_ps")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["direct_multistage_exp3", "epsilon_exp3"],
    )
    parser.add_argument("--min-episodes", type=int, default=75)
    parser.add_argument("--raw-gap-threshold", type=float, default=0.5)
    parser.add_argument("--tmux-session")
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def read_partial(run_dir: Path, method: str) -> dict[str, Any]:
    for name in ("summary_partial.json", "summary.json"):
        path = run_dir / method / name
        if path.exists():
            try:
                return load_json(path)
            except Exception as exc:
                return {"status": "read_error", "error": str(exc)}
    return {"status": "missing", "completed_episodes": 0}


def main() -> None:
    args = parse_args()
    methods = [args.ps_method, *args.baseline_methods]
    log_path = args.run_dir / "early_stop_monitor.log"
    decision_path = args.run_dir / f"early_stop_decision_at{args.min_episodes}.json"

    while True:
        summaries = {method: read_partial(args.run_dir, method) for method in methods}
        status_row = {
            method: {
                "completed": summaries[method].get("completed_episodes", 0),
                "raw_mean": summaries[method].get("raw_total_cost_mean"),
                "status": summaries[method].get("status"),
            }
            for method in methods
        }
        with log_path.open("a") as f:
            f.write(json.dumps({"ts": time.time(), "status": status_row}) + "\n")

        completed = {
            method: int(summaries[method].get("completed_episodes", 0) or 0)
            for method in methods
        }
        if all(count >= args.min_episodes for count in completed.values()):
            ps_raw = summaries[args.ps_method].get("raw_total_cost_mean")
            baseline_raw = {
                method: summaries[method].get("raw_total_cost_mean")
                for method in args.baseline_methods
            }
            if ps_raw is not None and all(v is not None for v in baseline_raw.values()):
                best_baseline_method, best_baseline_raw = min(
                    baseline_raw.items(), key=lambda item: float(item[1])
                )
                gap = float(ps_raw) - float(best_baseline_raw)
                early_stop = gap >= args.raw_gap_threshold
                decision_path.write_text(
                    json.dumps(
                        {
                            "min_episodes": args.min_episodes,
                            "raw_gap_threshold": args.raw_gap_threshold,
                            "completed": completed,
                            "ps_method": args.ps_method,
                            "ps_raw_mean": ps_raw,
                            "baseline_raw_means": baseline_raw,
                            "best_baseline_method": best_baseline_method,
                            "best_baseline_raw_mean": best_baseline_raw,
                            "gap_vs_best_baseline": gap,
                            "early_stop": early_stop,
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
                if early_stop and args.tmux_session:
                    subprocess.run(
                        ["tmux", "kill-session", "-t", args.tmux_session],
                        check=False,
                    )
                break

        if all(summaries[m].get("status") in {"completed", "failed", "error"} for m in methods):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
