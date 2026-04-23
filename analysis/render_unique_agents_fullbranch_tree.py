"""Generate a full-branching unique-agent 4/5-share tree and SVG.

This tree keeps the main controlled-simulation difficulty at ``L=5, K=5`` so
the path space stays at ``5^5 = 3125``. The only structural change relative to
the original shared-basin family is agent identity reuse: every parent-child
occurrence gets its own unique agent id, so there is no cross-prefix suffix
family reuse.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC_PATH = ROOT / "analysis" / "tree_specs" / "unique_agents_fullbranch_4of5_L5_K5.json"
SVG_PATH = ROOT / "analysis" / "unique_agents_fullbranch_4of5_L5_K5_tree.svg"
VALIDATION_PATH = ROOT / "analysis" / "unique_agents_fullbranch_4of5_L5_K5_validation.json"


DEPTH = 5
BRANCHING = 5
STAGES = [f"stage{i}" for i in range(1, DEPTH + 1)]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_paths(edges: dict[str, list[str]], depth: int) -> int:
    total = 0
    stack = [("ROOT", 0)]
    while stack:
        alias, current_depth = stack.pop()
        if current_depth == depth:
            total += 1
            continue
        for child in edges.get(alias, []):
            stack.append((child, current_depth + 1))
    return total


def build_spec() -> dict[str, Any]:
    nodes: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    edges: dict[str, list[str]] = {"ROOT": []}
    frontier: list[dict[str, str]] = [{"alias": "ROOT", "serial": "root"}]
    next_serial = 1

    for stage_idx, stage in enumerate(STAGES, start=1):
        next_frontier: list[dict[str, str]] = []
        for parent in frontier:
            parent_alias = parent["alias"]
            parent_serial = parent["serial"]
            edges[parent_alias] = []
            for child_idx in range(1, BRANCHING + 1):
                node_serial = f"n{next_serial:04d}"
                next_serial += 1
                alias = f"s{stage_idx}_{parent_serial}_c{child_idx:02d}_agent"
                base_alias = f"stage{stage_idx}_n{child_idx}"
                node = {
                    "alias": alias,
                    "agent_id": alias,
                    "stage": stage,
                    "g": 0 if child_idx <= 4 else 1,
                    "base_alias": base_alias,
                    "parent_alias": parent_alias,
                    "parent_serial": parent_serial,
                    "node_serial": node_serial,
                    "local_child_index": child_idx,
                    "agent_reuse_scope": "unique_to_parent_prefix",
                    "upload_semantics": (
                        "share_upload_allowed_or_continue"
                        if child_idx <= 4
                        else "barrier_or_unshared"
                    ),
                }
                nodes[stage].append(node)
                edges[parent_alias].append(alias)
                edges[alias] = []
                next_frontier.append({"alias": alias, "serial": node_serial})
        frontier = next_frontier

    spec = {
        "tree_name": "unique_agents_fullbranch_4of5_L5_K5",
        "depth": DEPTH,
        "stages": STAGES,
        "metadata": {
            "purpose": "fair unique-agent harder tree for ancestor-chain RiskyPS",
            "construction": "full-branching prefix-expanded unique-agent 4/5-share tree",
            "agent_reuse_policy": "unique_per_parent_no_cross_prefix_reuse",
            "cross_prefix_reuse_rate": 0.0,
            "no_cross_prefix_suffix_family": True,
            "compatible_with": ["risky_ps", "barriershare_controlled_sim"],
            "share_ratio_definition": (
                "every internal node has five direct children with four g=0 shareable "
                "interfaces and one g=1 barrier/unshare interface"
            ),
            "design_notes": [
                "This is a full Cartesian tree with L=5 and K=5 at every internal node.",
                "Every parent-child occurrence gets its own unique agent_id.",
                "base_alias is retained only as latent-role provenance for existing controlled-sim loaders.",
                "No child alias or agent id is reused across prefixes.",
            ],
        },
        "nodes": nodes,
        "edges_by_node_alias": edges,
    }
    spec["metadata"].update(validate_spec(spec, include_metadata=False))
    return spec


def validate_spec(spec: dict[str, Any], *, include_metadata: bool = True) -> dict[str, Any]:
    stages = list(spec["stages"])
    nodes_by_stage = spec["nodes"]
    edges = spec["edges_by_node_alias"]
    all_nodes = [node for stage in stages for node in nodes_by_stage.get(stage, [])]
    aliases = [node["alias"] for node in all_nodes]
    agent_ids = [node.get("agent_id", node["alias"]) for node in all_nodes]
    alias_counts = Counter(aliases)
    agent_counts = Counter(agent_ids)

    parent_by_child: dict[str, list[str]] = defaultdict(list)
    child_counts: list[int] = []
    internal_nodes = ["ROOT"] + [node["alias"] for stage in stages[:-1] for node in nodes_by_stage[stage]]
    for parent, children in edges.items():
        if parent in internal_nodes:
            child_counts.append(len(children))
        for child in children:
            parent_by_child[child].append(parent)

    cross_prefix_duplicates = {
        child: parents
        for child, parents in parent_by_child.items()
        if len(parents) > 1
    }
    duplicate_agent_ids = {
        agent_id: count
        for agent_id, count in agent_counts.items()
        if count > 1
    }
    g_by_stage = {
        stage: dict(Counter(int(node["g"]) for node in nodes_by_stage.get(stage, [])))
        for stage in stages
    }
    share_fraction_by_stage = {
        stage: g_counts.get(0, 0) / max(1, g_counts.get(0, 0) + g_counts.get(1, 0))
        for stage, g_counts in g_by_stage.items()
    }
    base_alias_counts_by_stage = {
        stage: dict(Counter(str(node["base_alias"]) for node in nodes_by_stage.get(stage, [])))
        for stage in stages
    }

    validation = {
        "depth": int(spec["depth"]),
        "stages": stages,
        "root_branching": len(edges.get("ROOT", [])),
        "min_branching": min(child_counts) if child_counts else 0,
        "max_branching": max(child_counts) if child_counts else 0,
        "mean_branching": statistics.fmean(child_counts) if child_counts else 0.0,
        "num_paths": count_paths(edges, int(spec["depth"])),
        "total_agent_ids": len(agent_ids),
        "total_nodes": len(all_nodes),
        "duplicate_agent_count": len(duplicate_agent_ids),
        "duplicate_agent_ids": duplicate_agent_ids,
        "cross_prefix_duplicate_count": len(cross_prefix_duplicates),
        "cross_prefix_duplicate_agent_ids": sorted(cross_prefix_duplicates),
        "cross_prefix_reuse_rate": (
            len(cross_prefix_duplicates) / max(1, len(agent_ids))
        ),
        "alias_duplicate_count": sum(1 for count in alias_counts.values() if count > 1),
        "g_counts_by_stage": g_by_stage,
        "expanded_share_fraction_by_stage": share_fraction_by_stage,
        "base_alias_counts_by_stage": base_alias_counts_by_stage,
        "share_barrier_summary": {
            "g0_total": sum(int(node["g"]) == 0 for node in all_nodes),
            "g1_total": sum(int(node["g"]) == 1 for node in all_nodes),
            "share_ratio_per_parent": "4/5",
            "barrier_node_count": sum(int(node["g"]) == 1 for node in all_nodes),
            "barrier_node_sample": [
                node["alias"]
                for node in all_nodes
                if int(node["g"]) == 1
            ][:12],
        },
    }
    if include_metadata:
        validation["metadata"] = spec.get("metadata", {})
    return validation


def sample_children(spec: dict[str, Any], parent_alias: str) -> list[dict[str, Any]]:
    node_by_alias = {
        node["alias"]: node
        for stage in spec["stages"]
        for node in spec["nodes"][stage]
    }
    return [node_by_alias[alias] for alias in spec["edges_by_node_alias"].get(parent_alias, [])]


def render_svg(spec: dict[str, Any], validation: dict[str, Any]) -> str:
    width = 1880
    height = 1160
    title_y = 42

    stage_counts = {stage: len(spec["nodes"][stage]) for stage in spec["stages"]}
    stage_boxes_x = [80, 360, 640, 920, 1200, 1480]
    stage_box_y = 170
    stage_box_w = 180
    stage_box_h = 120

    sample_root_children = sample_children(spec, "ROOT")
    sample_stage2_children = sample_children(spec, spec["edges_by_node_alias"]["ROOT"][0])
    sample_stage2_children_other = sample_children(spec, spec["edges_by_node_alias"]["ROOT"][1])
    sample_stage3_children = sample_children(spec, sample_stage2_children[0]["alias"])
    sample_stage3_children_other = sample_children(spec, sample_stage2_children[1]["alias"])

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 26px; font-weight: 800; }",
        ".subtitle { font-size: 13px; fill: #475569; }",
        ".panel-title { font-size: 16px; font-weight: 800; }",
        ".stage-title { font-size: 15px; font-weight: 800; }",
        ".body { font-size: 12px; }",
        ".small { font-size: 11px; fill: #475569; }",
        ".tiny { font-size: 10px; fill: #475569; }",
        ".edge { stroke: #94a3b8; stroke-width: 2.4; fill: none; opacity: 0.85; }",
        ".edge-label { font-size: 12px; fill: #334155; font-weight: 700; }",
        ".node-box { rx: 12; stroke-width: 1.4; }",
        ".sample-node { rx: 7; stroke-width: 1.0; }",
        ".green { fill: #eaf7ef; stroke: #2f855a; }",
        ".red { fill: #fff1f2; stroke: #c53030; }",
        ".neutral { fill: #ffffff; stroke: #cbd5e1; }",
        ".legend-box { rx: 10; fill: #ffffff; stroke: #cbd5e1; }",
        "</style>",
        f'<text class="title" x="54" y="{title_y}">Unique-Agent Full-Branching 4/5-Share Tree</text>',
        (
            f'<text class="subtitle" x="54" y="{title_y + 24}">'
            f'L=5, K=5, {validation["num_paths"]} paths, {validation["total_agent_ids"]} unique agent IDs, '
            f'cross-prefix reuse={validation["cross_prefix_duplicate_count"]}.'
            "</text>"
        ),
        (
            f'<text class="subtitle" x="54" y="{title_y + 44}">'
            "The tree keeps the original difficulty of a full 5-ary depth-5 search space; "
            "the only removed signal is cross-prefix agent identity reuse."
            "</text>"
        ),
    ]

    lines.extend(
        [
            '<rect class="legend-box" x="1410" y="26" width="390" height="88" />',
            '<rect class="sample-node green" x="1430" y="48" width="22" height="16" />',
            '<text class="small" x="1462" y="60">g=0 share/upload-capable child interface</text>',
            '<rect class="sample-node red" x="1430" y="76" width="22" height="16" />',
            '<text class="small" x="1462" y="88">g=1 barrier / unshared child interface</text>',
        ]
    )

    stage_labels = [
        ("ROOT", 1, 5, "5 unique children"),
        ("stage1", 5, 25, "5 children per parent"),
        ("stage2", 25, 125, "5 children per parent"),
        ("stage3", 125, 625, "5 children per parent"),
        ("stage4", 625, 3125, "5 children per parent"),
        ("stage5", 3125, 0, "leaf stage"),
    ]

    for idx, (label, count, next_count, note) in enumerate(stage_labels):
        x = stage_boxes_x[idx]
        y = stage_box_y
        lines.append(f'<rect class="node-box neutral" x="{x}" y="{y}" width="{stage_box_w}" height="{stage_box_h}" />')
        lines.append(f'<text class="stage-title" x="{x + 18}" y="{y + 26}">{escape(label)}</text>')
        lines.append(f'<text class="body" x="{x + 18}" y="{y + 52}">nodes: {count}</text>')
        if label != "ROOT":
            g_counts = validation["g_counts_by_stage"][label]
            lines.append(
                f'<text class="body" x="{x + 18}" y="{y + 72}">g=0: {g_counts.get("0", g_counts.get(0, 0))} | g=1: {g_counts.get("1", g_counts.get(1, 0))}</text>'
            )
        else:
            lines.append(f'<text class="body" x="{x + 18}" y="{y + 72}">branching: 5</text>')
        lines.append(f'<text class="small" x="{x + 18}" y="{y + 96}">{escape(note)}</text>')
        if next_count:
            x1 = x + stage_box_w
            y1 = y + stage_box_h / 2
            x2 = stage_boxes_x[idx + 1]
            y2 = stage_box_y + stage_box_h / 2
            lines.append(f'<path class="edge" d="M {x1},{y1} C {x1 + 40},{y1} {x2 - 40},{y2} {x2},{y2}" />')
            lines.append(
                f'<text class="edge-label" x="{(x1 + x2) / 2 - 16}" y="{y1 - 10}">×5</text>'
            )

    lines.extend(
        [
            '<rect class="legend-box" x="54" y="340" width="840" height="330" />',
            '<text class="panel-title" x="78" y="372">Parent-Local Unique Identity Examples</text>',
            '<text class="small" x="78" y="394">The same local child slot under different parents produces different agent IDs. No cross-prefix suffix-family reuse remains.</text>',
        ]
    )

    def draw_sample_group(
        title: str,
        parent_label: str,
        children: list[dict[str, Any]],
        x: int,
        y: int,
    ) -> None:
        lines.append(f'<text class="body" x="{x}" y="{y}">{escape(title)}</text>')
        lines.append(f'<text class="tiny" x="{x}" y="{y + 16}">parent: {escape(parent_label)}</text>')
        for idx, child in enumerate(children):
            box_y = y + 30 + idx * 34
            color_class = "green" if int(child["g"]) == 0 else "red"
            lines.append(f'<rect class="sample-node {color_class}" x="{x}" y="{box_y}" width="345" height="24" />')
            lines.append(
                f'<text class="tiny" x="{x + 8}" y="{box_y + 15}">{escape(child["alias"])} | base={escape(child["base_alias"])} | g={int(child["g"])}</text>'
            )

    draw_sample_group("Stage 1 children of ROOT", "ROOT", sample_root_children, 84, 430)
    draw_sample_group(
        "Stage 2 children of the first stage-1 parent",
        sample_root_children[0]["alias"],
        sample_stage2_children,
        432,
        430,
    )

    lines.extend(
        [
            '<rect class="legend-box" x="930" y="340" width="870" height="330" />',
            '<text class="panel-title" x="954" y="372">Same Local Slot, Different Parent, Different Agent</text>',
            '<text class="small" x="954" y="394">These pairs would have been shared-family aliases in a compact DAG. Here they stay prefix-specific.</text>',
        ]
    )

    pair_examples = [
        (sample_stage2_children[0], sample_stage2_children_other[0]),
        (sample_stage2_children[4], sample_stage2_children_other[4]),
        (sample_stage3_children[0], sample_stage3_children_other[0]),
        (sample_stage3_children[4], sample_stage3_children_other[4]),
    ]

    for idx, (left_node, right_node) in enumerate(pair_examples):
        row_y = 430 + idx * 62
        left_class = "green" if int(left_node["g"]) == 0 else "red"
        right_class = "green" if int(right_node["g"]) == 0 else "red"
        lines.append(f'<rect class="sample-node {left_class}" x="956" y="{row_y}" width="362" height="24" />')
        lines.append(f'<rect class="sample-node {right_class}" x="1388" y="{row_y}" width="362" height="24" />')
        lines.append(
            f'<text class="tiny" x="964" y="{row_y + 15}">{escape(left_node["alias"])} | parent={escape(left_node["parent_serial"])}</text>'
        )
        lines.append(
            f'<text class="tiny" x="1396" y="{row_y + 15}">{escape(right_node["alias"])} | parent={escape(right_node["parent_serial"])}</text>'
        )
        lines.append(f'<path class="edge" d="M 1318,{row_y + 12} C 1340,{row_y + 12} 1360,{row_y + 12} 1388,{row_y + 12}" />')
        lines.append(
            f'<text class="tiny" x="1216" y="{row_y + 43}">same local slot/base_alias, different parent, different agent_id</text>'
        )

    lines.extend(
        [
            '<rect class="legend-box" x="54" y="708" width="840" height="376" />',
            '<text class="panel-title" x="78" y="740">Validation Summary</text>',
            f'<text class="body" x="78" y="774">depth = {validation["depth"]}</text>',
            f'<text class="body" x="78" y="800">root branching = {validation["root_branching"]}</text>',
            f'<text class="body" x="78" y="826">min / max / mean branching = {validation["min_branching"]} / {validation["max_branching"]} / {validation["mean_branching"]:.1f}</text>',
            f'<text class="body" x="78" y="852">num paths = {validation["num_paths"]}</text>',
            f'<text class="body" x="78" y="878">total agent IDs = {validation["total_agent_ids"]}</text>',
            f'<text class="body" x="78" y="904">duplicate agent count = {validation["duplicate_agent_count"]}</text>',
            f'<text class="body" x="78" y="930">cross-prefix duplicate count = {validation["cross_prefix_duplicate_count"]}</text>',
            f'<text class="body" x="78" y="956">cross-prefix reuse rate = {validation["cross_prefix_reuse_rate"]:.1f}</text>',
            '<text class="small" x="78" y="990">The tree keeps full 5-ary difficulty while removing only identity reuse.</text>',
        ]
    )

    lines.extend(
        [
            '<rect class="legend-box" x="930" y="708" width="870" height="376" />',
            '<text class="panel-title" x="954" y="740">Per-Stage Share / Barrier Profile</text>',
            '<text class="small" x="954" y="762">Every stage preserves the same 4/5 shareable profile under full branching.</text>',
        ]
    )

    table_x = 954
    table_y = 792
    row_h = 44
    col_x = [table_x, table_x + 120, table_x + 260, table_x + 420]
    headers = ["Stage", "g=0", "g=1", "Share Fraction"]
    for idx, header in enumerate(headers):
        lines.append(f'<text class="body" x="{col_x[idx]}" y="{table_y}">{header}</text>')
    for row_idx, stage in enumerate(spec["stages"], start=1):
        y = table_y + row_idx * row_h
        g_counts = validation["g_counts_by_stage"][stage]
        g0 = g_counts.get("0", g_counts.get(0, 0))
        g1 = g_counts.get("1", g_counts.get(1, 0))
        share_fraction = validation["expanded_share_fraction_by_stage"][stage]
        lines.append(f'<text class="body" x="{col_x[0]}" y="{y}">{escape(stage)}</text>')
        lines.append(f'<text class="body" x="{col_x[1]}" y="{y}">{g0}</text>')
        lines.append(f'<text class="body" x="{col_x[2]}" y="{y}">{g1}</text>')
        lines.append(f'<text class="body" x="{col_x[3]}" y="{y}">{share_fraction:.1%}</text>')
        bar_x = col_x[3] + 138
        bar_y = y - 12
        lines.append(f'<rect class="sample-node neutral" x="{bar_x}" y="{bar_y}" width="180" height="16" />')
        lines.append(f'<rect class="sample-node green" x="{bar_x}" y="{bar_y}" width="{180 * share_fraction:.1f}" height="16" />')

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    spec = build_spec()
    validation = validate_spec(spec)
    write_json(SPEC_PATH, spec)
    write_json(VALIDATION_PATH, validation)
    SVG_PATH.write_text(render_svg(spec, validation), encoding="utf-8")
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
