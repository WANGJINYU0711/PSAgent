from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "envs",
    ROOT / "envs" / "tree_family",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from tree_family.generator import TreeFamilyGenerator  # noqa: E402


DEFAULT_SPEC_PATH = (
    ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json"
)


def _base_alias(agent_id: str) -> str:
    return agent_id.split("__from__", 1)[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check the shared_basin_strong_prefix_dedup family topology."
    )
    parser.add_argument(
        "--family-kind",
        default="shared_basin_strong_prefix_dedup",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC_PATH)
    args = parser.parse_args()

    generator = TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(args.family_kind, seed=args.seed)
    errors = generator.validate_family(family_spec, agent_map)
    if errors:
        raise SystemExit("Family validation failed: " + "; ".join(errors))

    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    spec_stage_counts = {
        stage_name: len(spec.get("nodes", {}).get(stage_name, []))
        for stage_name in family_spec.stages
    }
    family_stage_counts = {
        stage_name: len(family_spec.stage_agents.get(stage_name, []))
        for stage_name in family_spec.stages
    }
    g_counts = {
        stage_name: Counter(agent_map[agent_id].g for agent_id in family_spec.stage_agents[stage_name])
        for stage_name in family_spec.stages
    }
    spec_g_counts = {
        stage_name: Counter(
            int(node.get("g", 0)) for node in spec.get("nodes", {}).get(stage_name, [])
        )
        for stage_name in family_spec.stages
    }

    child_to_parent_prefixes: dict[str, list[tuple[str, ...]]] = defaultdict(list)
    for prefix, child_ids in (family_spec.allowed_children or {}).items():
        for child_id in child_ids:
            child_to_parent_prefixes[child_id].append(prefix)
    cross_prefix_reused_children = {
        child_id: parents
        for child_id, parents in child_to_parent_prefixes.items()
        if len(parents) > 1
    }

    base_alias_parent_fanout: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    for child_id, parents in child_to_parent_prefixes.items():
        for parent_prefix in parents:
            base_alias_parent_fanout[_base_alias(child_id)].add(parent_prefix)

    output = {
        "family_kind": args.family_kind,
        "family_name": family_spec.family_name,
        "stage_count": len(family_spec.stages),
        "stages": list(family_spec.stages),
        "num_agents_total": len(agent_map),
        "num_agents_per_stage": family_stage_counts,
        "spec_stage_counts": spec_stage_counts,
        "allowed_children_prefix_count": len(family_spec.allowed_children or {}),
        "root_child_count": len((family_spec.allowed_children or {}).get((), [])),
        "g_counts_per_stage": {stage: dict(counter) for stage, counter in g_counts.items()},
        "spec_g_counts_per_stage": {
            stage: dict(counter) for stage, counter in spec_g_counts.items()
        },
        "cross_prefix_reused_child_count": len(cross_prefix_reused_children),
        "cross_prefix_reused_children": sorted(cross_prefix_reused_children)[:10],
        "base_alias_parent_fanout_examples": {
            key: len(value)
        for key, value in sorted(base_alias_parent_fanout.items())[:10]
        },
        "summary": generator.describe_family(family_spec, agent_map),
    }
    if family_stage_counts != spec_stage_counts:
        raise SystemExit(
            "Stage counts differ from topology spec: "
            + json.dumps(
                {
                    "family_stage_counts": family_stage_counts,
                    "spec_stage_counts": spec_stage_counts,
                },
                ensure_ascii=False,
            )
        )
    if any(dict(g_counts[stage]) != dict(spec_g_counts[stage]) for stage in family_spec.stages):
        raise SystemExit(
            "Per-stage g counts differ from topology spec: "
            + json.dumps(
                {
                    "family_g_counts": {
                        stage: dict(counter) for stage, counter in g_counts.items()
                    },
                    "spec_g_counts": {
                        stage: dict(counter) for stage, counter in spec_g_counts.items()
                    },
                },
                ensure_ascii=False,
            )
        )
    if cross_prefix_reused_children:
        raise SystemExit(
            "Prefix-dedup family still reuses child agent IDs across multiple parents: "
            + ", ".join(sorted(cross_prefix_reused_children)[:10])
        )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
