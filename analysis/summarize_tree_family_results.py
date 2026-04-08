"""Summarize neutral / moderate / strong Group A experiment results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize tree family experiment results.")
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("/home/ubuntu/data/PSAgent/outputs/tree_family_groupA/results.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/home/ubuntu/data/PSAgent/outputs/tree_family_groupA/summary.json"),
    )
    args = parser.parse_args()

    with args.results.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["family_kind"], row["method"])].append(row)

    summary: list[dict] = []
    metrics = [
        "mean_total_cost",
        "mean_terminal_penalty",
        "mean_path_agent_cost",
        "exact_match_rate",
        "subset_mismatch_rate",
        "mean_false_cancel_count",
        "mean_missed_cancel_count",
        "mean_false_refuse_count",
        "mean_missed_refuse_count",
        "shared_leaf_ratio",
        "unshared_leaf_ratio",
    ]
    for (family_kind, method), items in sorted(grouped.items()):
        row = {"family_kind": family_kind, "method": method, "num_runs": len(items)}
        for metric in metrics:
            row[metric] = sum(item[metric] for item in items) / len(items)
        summary.append(row)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
