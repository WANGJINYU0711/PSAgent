"""Write a same-topology unique-agent 4/5-share full-branch tree.

This tree preserves the official full-branching 4/5-share topology exactly:
depth 5, branching 5 at every internal node, and 3125 total paths. The only
structural change is agent identity de-duplication: each parent-child
occurrence gets its own agent_id and cost_role, so there is no cross-prefix
agent reuse.
"""

from __future__ import annotations

import json
from pathlib import Path

from render_unique_agents_fullbranch_tree import (
    ROOT,
    build_spec,
    render_svg,
    validate_spec,
    write_json,
)


SPEC_PATH = ROOT / "analysis" / "tree_specs" / "unique_agents_same_topology_4of5_L5_K5.json"
SVG_PATH = ROOT / "analysis" / "unique_agents_same_topology_4of5_L5_K5_tree.svg"
VALIDATION_PATH = ROOT / "analysis" / "unique_agents_same_topology_4of5_L5_K5_validation.json"


def transform_spec() -> dict:
    spec = build_spec()
    spec["tree_name"] = "unique_agents_same_topology_4of5_L5_K5"
    for stage in spec["stages"]:
        for node in spec["nodes"][stage]:
            node["cost_role"] = node["agent_id"]
            node["synthetic_cost_scope"] = "subtree_local_unique_node"
    spec["metadata"] = {
        "purpose": "same-topology 4of5 tree with unique agents and subtree-local correlated cost",
        "construction": "topology-preserving agent de-duplication from official 4/5 full-branch tree",
        "agent_reuse_policy": "unique_per_parent_no_cross_prefix_reuse",
        "cross_prefix_reuse_rate": 0.0,
        "preserve_topology": True,
        "preserve_gate_profile": True,
        "preserve_num_paths": True,
        "compatible_with": ["risky_ps", "barriershare_controlled_sim"],
        "share_ratio_definition": (
            "every internal node keeps five direct children with four g=0 shareable "
            "interfaces and one g=1 barrier/unshare interface"
        ),
        "design_notes": [
            "This tree preserves the official full-branching L=5, K=5 topology exactly.",
            "Each parent-child occurrence has a unique agent_id and an explicit cost_role equal to that agent_id.",
            "base_alias is retained only for provenance and compatibility.",
            "Cross-prefix agent reuse is removed without changing path count or gate profile.",
        ],
    }
    spec["metadata"].update(validate_spec(spec, include_metadata=False))
    return spec


def main() -> None:
    spec = transform_spec()
    validation = validate_spec(spec)
    write_json(SPEC_PATH, spec)
    write_json(VALIDATION_PATH, validation)
    svg = render_svg(spec, validation)
    svg = svg.replace(
        "Unique-Agent Full-Branching 4/5-Share Tree",
        "Same-Topology Unique-Agent 4/5-Share Tree",
    ).replace(
        "The tree keeps the original difficulty of a full 5-ary depth-5 search space; the only removed signal is cross-prefix agent identity reuse.",
        "The tree preserves the official full-branching 4/5-share topology exactly; the only structural change is removing cross-prefix agent reuse.",
    )
    SVG_PATH.write_text(svg, encoding="utf-8")
    print(
        json.dumps(
            {
                "spec_path": str(SPEC_PATH.relative_to(ROOT)),
                "svg_path": str(SVG_PATH.relative_to(ROOT)),
                "validation_path": str(VALIDATION_PATH.relative_to(ROOT)),
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
