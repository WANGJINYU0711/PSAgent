"""Validate and summarize a generated tree family."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "envs",):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tree_family.generator import TreeFamilyGenerator  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Check a generated tree family.")
    parser.add_argument("--family-kind", required=True, choices=["neutral", "moderate", "strong"])
    parser.add_argument("--family-seed", type=int, default=0)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = parser.parse_args()

    generator = TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(args.family_kind, seed=args.family_seed)
    errors = generator.validate_family(family_spec, agent_map)
    description = generator.describe_family(family_spec, agent_map)

    result = {
        "family_spec": {
            "family_name": family_spec.family_name,
            "stages": family_spec.stages,
            "stage_agents": family_spec.stage_agents,
        },
        "validation_errors": errors,
        "description": description,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
