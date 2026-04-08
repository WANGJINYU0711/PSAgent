"""Build and summarize TaskDescriptor objects for the airline derived dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "envs", ROOT / "envs" / "adapters", ROOT / "envs" / "tree_family"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from airline_adapter import AirlineTaskAdapter  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Build TaskDescriptor summaries for airline data.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "airline_task_descriptor_summary.json",
    )
    args = parser.parse_args()

    with args.data.open("r", encoding="utf-8") as handle:
        instances = json.load(handle)

    adapter = AirlineTaskAdapter()
    descriptors = [adapter.build_task_descriptor(instance) for instance in instances]

    attr_totals: dict[int, float] = {}
    stage_totals: dict[str, float] = {}
    for desc in descriptors:
        for attr_id, weight in desc.attribute_weights.items():
            attr_totals[attr_id] = attr_totals.get(attr_id, 0.0) + weight
        for stage, diff in desc.stage_difficulty.items():
            stage_totals[stage] = stage_totals.get(stage, 0.0) + diff

    summary = {
        "num_tasks": len(descriptors),
        "mean_attribute_weights": {
            str(attr_id): attr_totals[attr_id] / len(descriptors) for attr_id in sorted(attr_totals)
        },
        "mean_stage_difficulty": {
            stage: stage_totals[stage] / len(descriptors) for stage in sorted(stage_totals)
        },
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
