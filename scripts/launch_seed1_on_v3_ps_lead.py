#!/usr/bin/env python
"""Launch the seed1 trap_asym-v3 run if seed0 shows a mid-run PS lead."""

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


def read_summary(run_dir: Path, method: str) -> dict[str, Any]:
    for name in ("summary_partial.json", "summary.json"):
        path = run_dir / method / name
        if path.exists():
            try:
                return load_json(path)
            except Exception as exc:
                return {"status": "read_error", "error": str(exc), "completed_episodes": 0}
    return {"status": "missing", "completed_episodes": 0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed0-run-dir", type=Path, required=True)
    parser.add_argument("--seed1-run-dir", type=Path, required=True)
    parser.add_argument("--launcher", type=Path, required=True)
    parser.add_argument("--ps-method", default="risky_ps")
    parser.add_argument(
        "--baseline-methods",
        nargs="+",
        default=["direct_multistage_exp3", "epsilon_exp3"],
    )
    parser.add_argument("--min-episodes", type=int, default=45)
    parser.add_argument("--max-episodes", type=int, default=55)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    log_path = args.seed0_run_dir / "seed1_trigger_monitor.log"
    decision_path = args.seed0_run_dir / "seed1_trigger_decision.json"
    methods = [args.ps_method, *args.baseline_methods]

    while True:
        summaries = {method: read_summary(args.seed0_run_dir, method) for method in methods}
        completed = {
            method: int(summaries[method].get("completed_episodes", 0) or 0)
            for method in methods
        }
        raw_means = {
            method: summaries[method].get("raw_total_cost_mean")
            for method in methods
        }
        status = {
            "ts": time.time(),
            "completed": completed,
            "raw_means": raw_means,
        }
        with log_path.open("a") as f:
            f.write(json.dumps(status, sort_keys=True) + "\n")

        if args.seed1_run_dir.exists():
            decision = {
                **status,
                "launch_seed1": False,
                "reason": "seed1_run_dir_already_exists",
            }
            decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
            break

        min_completed = min(completed.values()) if completed else 0
        if min_completed >= args.min_episodes:
            ps_raw = raw_means.get(args.ps_method)
            baseline_raw = {
                method: raw_means.get(method)
                for method in args.baseline_methods
            }
            if ps_raw is not None and all(value is not None for value in baseline_raw.values()):
                best_method, best_raw = min(baseline_raw.items(), key=lambda item: float(item[1]))
                lead = float(best_raw) - float(ps_raw)
                launch = lead > 0.0
                decision = {
                    **status,
                    "best_baseline_method": best_method,
                    "best_baseline_raw_mean": best_raw,
                    "ps_raw_mean": ps_raw,
                    "ps_lead_vs_best_baseline": lead,
                    "launch_seed1": launch,
                    "min_episodes": args.min_episodes,
                    "max_episodes": args.max_episodes,
                }
                decision_path.write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
                if launch:
                    subprocess.run([str(args.launcher), "1"], check=True)
                    break
                if min_completed >= args.max_episodes:
                    break

        if all(summaries[m].get("status") in {"completed", "failed", "error"} for m in methods):
            break
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
