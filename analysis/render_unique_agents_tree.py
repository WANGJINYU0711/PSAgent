"""Render a prefix-expanded low-reuse 4/5-share tree spec.

The existing ``shared_basin_strong_4of5_minimal`` spec is a compact DAG: stage
aliases are intentionally reused across multiple parents. This script keeps the
same depth, stages, gate labels, and legal continuation topology, but expands
every parent-child occurrence into a unique agent id. The resulting tree better
matches ancestor-chain BarrierShare/RiskyPS updates because reusable signal can
only flow through the sampled ancestor chain, not through repeated global
suffix-agent identities.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE_SPEC_PATH = ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_minimal.json"
SPEC_PATH = ROOT / "analysis" / "tree_specs" / "unique_agents_4of5_minimal.json"
SVG_PATH = ROOT / "analysis" / "unique_agents_4of5_tree.svg"
VALIDATION_PATH = ROOT / "analysis" / "unique_agents_4of5_validation.json"


STAGE_X_GAP = 300.0
NODE_W = 252.0
NODE_H = 20.0
NODE_Y_GAP = 27.0
TOP_MARGIN = 128.0
LEFT_MARGIN = 42.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def short_base(base_alias: str) -> str:
    return base_alias.replace("stage", "s").replace("_", "")


def count_paths(edges: dict[str, list[str]], depth: int) -> int:
    count = 0
    stack = [("ROOT", 0)]
    while stack:
        node, node_depth = stack.pop()
        if node_depth == depth:
            count += 1
            continue
        for child in edges.get(node, []):
            stack.append((child, node_depth + 1))
    return count


def expand_unique_agent_spec(base_spec: dict[str, Any]) -> dict[str, Any]:
    stages = list(base_spec["stages"])
    base_edges = {
        str(parent): list(children)
        for parent, children in base_spec["edges_by_node_alias"].items()
    }
    base_node_by_alias = {
        str(node["alias"]): dict(node)
        for stage_nodes in base_spec["nodes"].values()
        for node in stage_nodes
    }

    expanded_nodes: dict[str, list[dict[str, Any]]] = {stage: [] for stage in stages}
    expanded_edges: dict[str, list[str]] = {"ROOT": []}
    base_alias_by_expanded_alias: dict[str, str] = {"ROOT": "ROOT"}
    node_serial_by_alias: dict[str, str] = {"ROOT": "root"}

    frontier = ["ROOT"]
    next_serial = 1
    for stage_idx, stage in enumerate(stages, start=1):
        next_frontier: list[str] = []
        for parent_alias in frontier:
            parent_base_alias = base_alias_by_expanded_alias[parent_alias]
            parent_serial = node_serial_by_alias[parent_alias]
            base_children = base_edges.get(parent_base_alias, [])
            expanded_edges[parent_alias] = []
            for local_idx, base_child_alias in enumerate(base_children, start=1):
                base_child = base_node_by_alias[base_child_alias]
                node_serial = f"n{next_serial:04d}"
                next_serial += 1
                alias = (
                    f"s{stage_idx}_{parent_serial}_c{local_idx:02d}_"
                    f"{short_base(base_child_alias)}_agent"
                )
                node = {
                    "alias": alias,
                    "agent_id": alias,
                    "stage": stage,
                    "g": int(base_child["g"]),
                    "base_alias": base_child_alias,
                    "parent_alias": parent_alias,
                    "parent_serial": parent_serial,
                    "node_serial": node_serial,
                    "local_child_index": local_idx,
                    "agent_reuse_scope": "unique_to_parent_prefix",
                    "upload_semantics": (
                        "share_upload_allowed_or_continue" if int(base_child["g"]) == 0 else "barrier_or_unshared"
                    ),
                }
                expanded_nodes[stage].append(node)
                expanded_edges[parent_alias].append(alias)
                expanded_edges[alias] = []
                base_alias_by_expanded_alias[alias] = base_child_alias
                node_serial_by_alias[alias] = node_serial
                next_frontier.append(alias)
        frontier = next_frontier

    base_profile_gate_summary = {}
    for stage, nodes in base_spec["nodes"].items():
        gates = Counter(int(node["g"]) for node in nodes)
        base_profile_gate_summary[stage] = {
            "g0_base_profiles": gates.get(0, 0),
            "g1_base_profiles": gates.get(1, 0),
            "base_profile_share_ratio": (
                gates.get(0, 0) / max(1, gates.get(0, 0) + gates.get(1, 0))
            ),
        }

    spec: dict[str, Any] = {
        "tree_name": "unique_agents_4of5_minimal",
        "depth": int(base_spec["depth"]),
        "stages": stages,
        "metadata": {
            "purpose": "align with ancestor-chain BarrierShare / RiskyPS",
            "source_spec": str(BASE_SPEC_PATH.relative_to(ROOT)),
            "construction": "prefix-expanded copy of the source 4/5-share DAG",
            "no_cross_prefix_suffix_family": True,
            "agent_reuse_policy": "unique_per_parent_low_reuse",
            "cross_prefix_reuse_rate": 0.0,
            "compatible_with": ["risky_ps", "barriershare_controlled_sim"],
            "share_ratio_definition": (
                "source stage profiles keep four g=0 share-capable profiles and one g=1 barrier/unshare profile"
            ),
            "base_profile_gate_summary": base_profile_gate_summary,
            "design_notes": [
                "Every expanded node alias is a unique agent_id.",
                "A base_alias is retained only as provenance for visualization and validation.",
                "No child alias is attached to more than one parent.",
                "The legal continuation pattern and gate values are inherited from the source minimal 4/5 spec.",
            ],
        },
        "nodes": expanded_nodes,
        "edges_by_node_alias": expanded_edges,
    }
    spec["metadata"].update(validate_spec(spec, include_metadata=False))
    return spec


def validate_spec(spec: dict[str, Any], *, include_metadata: bool = True) -> dict[str, Any]:
    stages = list(spec["stages"])
    depth = int(spec["depth"])
    nodes_by_stage = spec["nodes"]
    edges = spec["edges_by_node_alias"]
    all_nodes = [
        node
        for stage in stages
        for node in nodes_by_stage.get(stage, [])
    ]
    aliases = [node["alias"] for node in all_nodes]
    agent_ids = [node.get("agent_id", node["alias"]) for node in all_nodes]
    alias_counts = Counter(aliases)
    agent_counts = Counter(agent_ids)

    parent_by_child: dict[str, list[str]] = defaultdict(list)
    child_counts = []
    for parent, children in edges.items():
        if children:
            child_counts.append(len(children))
        for child in children:
            parent_by_child[child].append(parent)
    cross_prefix_duplicate_children = {
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
    expanded_share_fraction_by_stage = {
        stage: (
            g_counts.get(0, 0) / max(1, g_counts.get(0, 0) + g_counts.get(1, 0))
        )
        for stage, g_counts in g_by_stage.items()
    }
    base_alias_counts_by_stage = {
        stage: dict(Counter(str(node.get("base_alias")) for node in nodes_by_stage.get(stage, [])))
        for stage in stages
    }
    barrier_nodes = [
        node["alias"]
        for node in all_nodes
        if int(node["g"]) == 1
    ]
    validation = {
        "depth": depth,
        "stages": stages,
        "root_branching": len(edges.get("ROOT", [])),
        "min_branching": min(child_counts) if child_counts else 0,
        "max_branching": max(child_counts) if child_counts else 0,
        "mean_branching": statistics.fmean(child_counts) if child_counts else 0.0,
        "num_paths": count_paths(edges, depth),
        "total_agent_ids": len(agent_ids),
        "total_nodes": len(all_nodes),
        "duplicate_agent_count": len(duplicate_agent_ids),
        "duplicate_agent_ids": duplicate_agent_ids,
        "cross_prefix_duplicate_count": len(cross_prefix_duplicate_children),
        "cross_prefix_duplicate_agent_ids": sorted(cross_prefix_duplicate_children),
        "cross_prefix_reuse_rate": (
            len(cross_prefix_duplicate_children) / max(1, len(agent_ids))
        ),
        "alias_duplicate_count": sum(1 for count in alias_counts.values() if count > 1),
        "g_counts_by_stage": g_by_stage,
        "expanded_node_share_fraction_by_stage": expanded_share_fraction_by_stage,
        "base_alias_counts_by_stage": base_alias_counts_by_stage,
        "share_barrier_summary": {
            "g0_total": sum(int(node["g"]) == 0 for node in all_nodes),
            "g1_total": sum(int(node["g"]) == 1 for node in all_nodes),
            "base_profile_share_ratio": "4/5 in each source stage",
            "barrier_node_count": len(barrier_nodes),
            "barrier_node_sample": barrier_nodes[:12],
        },
    }
    if include_metadata:
        validation["metadata"] = spec.get("metadata", {})
    return validation


def layout_nodes(spec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {"ROOT": (LEFT_MARGIN, TOP_MARGIN)}
    max_stage_count = max(len(spec["nodes"].get(stage, [])) for stage in spec["stages"])
    for stage_idx, stage in enumerate(spec["stages"], start=1):
        nodes = spec["nodes"][stage]
        x = LEFT_MARGIN + stage_idx * STAGE_X_GAP
        column_height = (len(nodes) - 1) * NODE_Y_GAP
        y0 = TOP_MARGIN + max(0.0, (max_stage_count - len(nodes)) * NODE_Y_GAP / 2.0)
        for row_idx, node in enumerate(nodes):
            positions[node["alias"]] = (x, y0 + row_idx * NODE_Y_GAP)
    return positions


def render_svg(spec: dict[str, Any], validation: dict[str, Any]) -> str:
    stages = list(spec["stages"])
    max_count = max(len(spec["nodes"].get(stage, [])) for stage in stages)
    width = LEFT_MARGIN + (len(stages) + 1) * STAGE_X_GAP + NODE_W + 110
    height = TOP_MARGIN + max_count * NODE_Y_GAP + 130
    positions = layout_nodes(spec)
    node_by_alias = {
        node["alias"]: node
        for stage in stages
        for node in spec["nodes"][stage]
    }

    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.1f}" height="{height:.1f}" viewBox="0 0 {width:.1f} {height:.1f}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #172033; }",
        ".title { font-size: 22px; font-weight: 800; }",
        ".subtitle { font-size: 12px; fill: #475569; }",
        ".stage-title { font-size: 15px; font-weight: 700; }",
        ".node { stroke-width: 1.1; rx: 5; }",
        ".node-label { font-size: 7px; font-weight: 700; }",
        ".node-meta { font-size: 6px; fill: #475569; }",
        ".edge { stroke: #64748b; stroke-width: 0.55; fill: none; opacity: 0.18; }",
        ".root { fill: #f8fafc; stroke: #111827; stroke-width: 1.5; rx: 8; }",
        "</style>",
        '<text class="title" x="42" y="34">Unique-Agent 4/5-Share Tree</text>',
        (
            '<text class="subtitle" x="42" y="54">'
            f'Prefix-expanded from shared_basin_strong_4of5_minimal: '
            f'{validation["num_paths"]} paths, {validation["total_agent_ids"]} unique agent IDs, '
            f'cross-prefix reuse={validation["cross_prefix_duplicate_count"]}.'
            "</text>"
        ),
        (
            '<text class="subtitle" x="42" y="72">'
            'Green nodes: g=0 share/upload-capable. Red nodes: g=1 barrier or unshared leaf. '
            'Base aliases are provenance only; agent IDs are unique per parent prefix.'
            "</text>"
        ),
    ]

    root_x, root_y = positions["ROOT"]
    lines.append(f'<rect class="root" x="{root_x:.1f}" y="{root_y:.1f}" width="88" height="30" />')
    lines.append(f'<text class="stage-title" x="{root_x + 23:.1f}" y="{root_y + 20:.1f}">ROOT</text>')
    for stage_idx, stage in enumerate(stages, start=1):
        x = LEFT_MARGIN + stage_idx * STAGE_X_GAP
        lines.append(
            f'<text class="stage-title" x="{x:.1f}" y="102">{escape(stage)}</text>'
        )

    for parent, children in spec["edges_by_node_alias"].items():
        if parent not in positions:
            continue
        px, py = positions[parent]
        sx = px + (88 if parent == "ROOT" else NODE_W)
        sy = py + (15 if parent == "ROOT" else NODE_H / 2)
        for child in children:
            if child not in positions:
                continue
            cx, cy = positions[child]
            tx = cx
            ty = cy + NODE_H / 2
            lines.append(
                f'<path class="edge" d="M {sx:.1f},{sy:.1f} C {sx + 36:.1f},{sy:.1f} {tx - 36:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" />'
            )

    for stage in stages:
        for node in spec["nodes"][stage]:
            x, y = positions[node["alias"]]
            g = int(node["g"])
            fill = "#eaf7ef" if g == 0 else "#fff1f2"
            stroke = "#2f855a" if g == 0 else "#c53030"
            lines.append(
                f'<rect class="node" x="{x:.1f}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" fill="{fill}" stroke="{stroke}" />'
            )
            lines.append(
                f'<text class="node-label" x="{x + 4:.1f}" y="{y + 8:.1f}">{escape(node["alias"])}</text>'
            )
            lines.append(
                f'<text class="node-meta" x="{x + 4:.1f}" y="{y + 17:.1f}">g={g} | base={escape(str(node["base_alias"]))} | parent={escape(str(node["parent_serial"]))}</text>'
            )

    legend_x = width - 390
    legend_y = 26
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="330" height="74" rx="10" fill="#ffffff" stroke="#cbd5e1" />',
            f'<rect x="{legend_x + 12}" y="{legend_y + 16}" width="18" height="14" rx="4" fill="#eaf7ef" stroke="#2f855a" />',
            f'<text class="subtitle" x="{legend_x + 38}" y="{legend_y + 27}">g=0 share/upload-capable interface</text>',
            f'<rect x="{legend_x + 12}" y="{legend_y + 42}" width="18" height="14" rx="4" fill="#fff1f2" stroke="#c53030" />',
            f'<text class="subtitle" x="{legend_x + 38}" y="{legend_y + 53}">g=1 barrier / unshared leaf start block</text>',
        ]
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    base_spec = read_json(BASE_SPEC_PATH)
    spec = expand_unique_agent_spec(base_spec)
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
