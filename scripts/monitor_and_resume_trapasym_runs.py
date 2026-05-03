#!/usr/bin/env python
"""Monitor trap-asym runs and resume stalled methods from checkpoint.pkl.

This helper watches each method directory under one or more run roots.
If a method's `progress.json` stops updating for longer than the configured
staleness window while it is still `running`, the script relaunches the
method using the existing `run_shared_basin_repeated_smoke.py run-method`
entrypoint. The runner resumes from `checkpoint.pkl` automatically.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_shared_basin_repeated_smoke.py"


@dataclass
class MethodState:
    run_dir: Path
    method: str
    status: str = "missing"
    completed_episodes: int = 0
    last_completed_episode_index: int = -1
    updated_at: str | None = None
    progress_path: Path | None = None
    restart_count: int = 0
    last_restart_at: float | None = None
    proc: subprocess.Popen[Any] | None = field(default=None, repr=False)


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, dict):
        return {}
    return data


def parse_iso(ts: str | None) -> float | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            return dt.timestamp()
        return dt.timestamp()
    except Exception:
        return None


def read_state(run_dir: Path, method: str) -> MethodState:
    progress_path = run_dir / method / "progress.json"
    state = MethodState(run_dir=run_dir, method=method, progress_path=progress_path)
    if not progress_path.exists():
        return state
    try:
        payload = load_json(progress_path)
    except Exception:
        state.status = "read_error"
        return state
    state.status = str(payload.get("status", "unknown"))
    state.completed_episodes = int(payload.get("completed_episodes", 0) or 0)
    state.last_completed_episode_index = int(
        payload.get("last_completed_episode_index", -1) or -1
    )
    state.updated_at = payload.get("updated_at")
    return state


def merge_runtime_state(current: MethodState, previous: MethodState) -> MethodState:
    current.restart_count = previous.restart_count
    current.last_restart_at = previous.last_restart_at
    current.proc = previous.proc
    return current


def stale_seconds(state: MethodState) -> float | None:
    if state.progress_path and state.progress_path.exists():
        return time.time() - state.progress_path.stat().st_mtime
    ts = parse_iso(state.updated_at)
    if ts is None:
        return None
    return time.time() - ts


def launch_method(run_dir: Path, method: str, log_dir: Path | None = None) -> subprocess.Popen[Any]:
    log_dir = log_dir or (run_dir / method)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "resume_monitor.log"
    handle = log_path.open("a", encoding="utf-8")
    handle.write(f"[launch] {datetime.now().isoformat()} method={method}\n")
    handle.flush()
    cmd = [
        sys.executable,
        str(RUNNER),
        "run-method",
        "--run-dir",
        str(run_dir),
        "--method",
        method,
    ]
    return subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=handle,
        stderr=subprocess.STDOUT,
        env=os.environ.copy(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--methods", nargs="+", action="append")
    parser.add_argument("--poll-seconds", type=float, default=120.0)
    parser.add_argument("--stale-seconds", type=float, default=1800.0)
    parser.add_argument("--max-restarts", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.methods and len(args.methods) not in {1, len(args.run_dir)}:
        raise SystemExit("If provided, --methods must be passed once per --run-dir.")

    methods_by_run: dict[Path, list[str]] = {}
    for idx, run_dir in enumerate(args.run_dir):
        if args.methods:
            methods_by_run[run_dir] = args.methods[idx]
        else:
            cfg = load_json(run_dir / "run_config.json")
            methods_by_run[run_dir] = list(cfg.get("methods", []))

    run_log = ROOT / "tmp" / "trapasym_resume_monitor.log"
    run_log.parent.mkdir(parents=True, exist_ok=True)

    states: dict[tuple[Path, str], MethodState] = {}
    for run_dir, methods in methods_by_run.items():
        for method in methods:
            states[(run_dir, method)] = read_state(run_dir, method)

    while True:
        any_action = False
        snapshot: list[dict[str, Any]] = []
        for (run_dir, method), state in states.items():
            current = merge_runtime_state(read_state(run_dir, method), state)
            states[(run_dir, method)] = current
            stale = stale_seconds(current)
            snapshot.append(
                {
                    "run_dir": str(run_dir),
                    "method": method,
                    "status": current.status,
                    "completed_episodes": current.completed_episodes,
                    "last_completed_episode_index": current.last_completed_episode_index,
                    "updated_at": current.updated_at,
                    "stale_seconds": stale,
                    "restart_count": current.restart_count,
                }
            )

            if current.status in {"completed", "failed", "error"}:
                continue
            if stale is None or stale < args.stale_seconds:
                continue
            if current.restart_count >= args.max_restarts:
                continue

            proc = launch_method(run_dir, method)
            current.proc = proc
            current.restart_count += 1
            current.last_restart_at = time.time()
            any_action = True
            with run_log.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "ts": time.time(),
                            "action": "restart",
                            "run_dir": str(run_dir),
                            "method": method,
                            "completed_episodes": current.completed_episodes,
                            "last_completed_episode_index": current.last_completed_episode_index,
                            "updated_at": current.updated_at,
                            "stale_seconds": stale,
                            "restart_count": current.restart_count,
                            "pid": proc.pid,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

        with run_log.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps({"ts": time.time(), "snapshot": snapshot}, sort_keys=True)
                + "\n"
            )
        if not any_action:
            time.sleep(args.poll_seconds)
        else:
            time.sleep(10.0)


if __name__ == "__main__":
    main()
