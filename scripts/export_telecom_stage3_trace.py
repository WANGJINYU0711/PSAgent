"""Export compact Stage 3 traces from telecom run outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    raise ValueError(f"Unsupported input file: {path}")


def _stage3_export_row(episode: dict[str, Any]) -> dict[str, Any] | None:
    stage_trace = episode.get("stage_trace", [])
    if not isinstance(stage_trace, list):
        return None
    stage3 = next(
        (
            row
            for row in stage_trace
            if isinstance(row, dict) and row.get("stage_name") == "stage3"
        ),
        None,
    )
    if stage3 is None:
        return None

    stage2_input = stage3.get("input", {}).get("stage2_output", {})
    stage3_output = stage3.get("output", {})
    return {
        "instance_id": episode.get("instance_id"),
        "resolved_line_id": stage2_input.get("resolved_line_id"),
        "target_phone_number": stage2_input.get("target_phone_number"),
        "executed_tool_calls": stage3.get("executed_tool_calls", []),
        "tool_results": stage3.get("tool_results", []),
        "tool_errors": stage3.get("tool_errors", []),
        "observed_state": stage3_output.get("observed_state"),
        "inferred_blocker_ids": stage3_output.get("inferred_blocker_ids"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Export telecom Stage 3 traces from run outputs.")
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to an episode log file such as episode_logs.jsonl or episodes.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write the exported Stage 3 trace JSON.",
    )
    args = parser.parse_args()

    episodes = _load_rows(args.input)
    exports = [row for episode in episodes if (row := _stage3_export_row(episode)) is not None]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(exports, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
