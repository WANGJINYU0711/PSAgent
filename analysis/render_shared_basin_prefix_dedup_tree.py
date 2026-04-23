"""Render a prefix-specific de-dup expansion of shared_basin_strong_4of5_minimal.

The source spec is a compact DAG: the same child alias can be referenced by
multiple different parents. This script keeps the same continuation pattern and
the same gate value ``g`` for every base alias, but clones every reused child
per parent prefix. The result is a prefix-expanded tree whose local workflow
continuations still match the original DAG exactly.
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
SPEC_PATH = ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json"
SVG_PATH = ROOT / "analysis" / "shared_basin_strong_4of5_prefix_dedup_tree.svg"
VALIDATION_PATH = ROOT / "analysis" / "shared_basin_strong_4of5_prefix_dedup_validation.json"


STAGE_X_GAP = 300.0
NODE_W = 300.0
NODE_H = 22.0
NODE_Y_GAP = 29.0
TOP_MARGIN = 128.0
LEFT_MARGIN = 42.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_paths(edges: dict[str, list[str]], depth: int) -> int:
    count = 0
    stack = [("ROOT", 0)]
    while stack:
        node_alias, node_depth = stack.pop()
        if node_depth == depth:
            count += 1
            continue
        for child_alias in edges.get(node_alias, []):
            stack.append((child_alias, node_depth + 1))
    return count


def enumerate_paths(
    edges: dict[str, list[str]],
    depth: int,
    *,
    base_alias_by_expanded_alias: dict[str, str] | None = None,
) -> list[list[str]]:
    paths: list[list[str]] = []
    stack: list[tuple[str, int, list[str]]] = [("ROOT", 0, [])]
    while stack:
        node_alias, node_depth, prefix = stack.pop()
        if node_depth == depth:
            paths.append(prefix)
            continue
        for child_alias in reversed(edges.get(node_alias, [])):
            child_base = (
                base_alias_by_expanded_alias[child_alias]
                if base_alias_by_expanded_alias is not None
                else child_alias
            )
            stack.append((child_alias, node_depth + 1, prefix + [child_base]))
    return paths


def source_parent_sets(base_edges: dict[str, list[str]]) -> dict[str, list[str]]:
    parents_by_child: dict[str, list[str]] = defaultdict(list)
    for parent_alias, child_aliases in base_edges.items():
        for child_alias in child_aliases:
            parents_by_child[child_alias].append(parent_alias)
    return {
        child_alias: sorted(parent_aliases)
        for child_alias, parent_aliases in parents_by_child.items()
        if len(parent_aliases) > 1
    }


def expand_prefix_dedup_spec(base_spec: dict[str, Any]) -> tuple[dict[str, Any], dict[str, str], dict[str, int]]:
    stages = list(base_spec["stages"])
    base_edges = {
        str(parent_alias): list(child_aliases)
        for parent_alias, child_aliases in base_spec["edges_by_node_alias"].items()
    }
    base_node_by_alias = {
        str(node["alias"]): dict(node)
        for stage_nodes in base_spec["nodes"].values()
        for node in stage_nodes
    }

    expanded_nodes: dict[str, list[dict[str, Any]]] = {stage: [] for stage in stages}
    expanded_edges: dict[str, list[str]] = {"ROOT": []}
    base_alias_by_expanded_alias: dict[str, str] = {"ROOT": "ROOT"}
    serial_by_expanded_alias: dict[str, str] = {"ROOT": "root"}
    clone_count_by_base_alias: dict[str, int] = Counter()

    frontier = ["ROOT"]
    next_serial = 1
    for stage_idx, stage in enumerate(stages, start=1):
        next_frontier: list[str] = []
        for parent_alias in frontier:
            parent_base_alias = base_alias_by_expanded_alias[parent_alias]
            parent_serial = serial_by_expanded_alias[parent_alias]
            base_child_aliases = base_edges.get(parent_base_alias, [])
            expanded_edges[parent_alias] = []
            for local_idx, base_child_alias in enumerate(base_child_aliases, start=1):
                base_child = base_node_by_alias[base_child_alias]
                node_serial = f"n{next_serial:04d}"
                next_serial += 1
                alias = f"{base_child_alias}__from__{parent_serial}__c{local_idx:02d}"
                display_alias = f"{base_child_alias} <- {parent_serial}"
                node = {
                    "alias": alias,
                    "display_alias": display_alias,
                    "agent_id": alias,
                    "stage": stage,
                    "g": int(base_child["g"]),
                    "base_alias": base_child_alias,
                    "source_alias": base_child_alias,
                    "parent_alias": parent_alias,
                    "parent_base_alias": parent_base_alias,
                    "parent_serial": parent_serial,
                    "node_serial": node_serial,
                    "local_child_index": local_idx,
                    "clone_scope": "parent_specific",
                }
                expanded_nodes[stage].append(node)
                expanded_edges[parent_alias].append(alias)
                expanded_edges[alias] = []
                base_alias_by_expanded_alias[alias] = base_child_alias
                serial_by_expanded_alias[alias] = node_serial
                clone_count_by_base_alias[base_child_alias] += 1
                next_frontier.append(alias)
        frontier = next_frontier

    spec: dict[str, Any] = {
        "tree_name": "shared_basin_strong_4of5_prefix_dedup",
        "depth": int(base_spec["depth"]),
        "stages": stages,
        "metadata": {
            "purpose": "prefix-specific de-dup of shared_basin_strong_4of5_minimal",
            "source_spec": str(BASE_SPEC_PATH.relative_to(ROOT)),
            "preserve_connectivity_pattern": True,
            "preserve_g": True,
            "agent_reuse_policy": "parent_specific_clone",
            "cross_prefix_agent_reuse_removed": True,
            "compatible_with": ["risky_ps", "barriershare_controlled_sim"],
            "llm_runner_requires_adapter": True,
            "not_directly_compatible_with_current_shared_basin_llm_runner": True,
        },
        "nodes": expanded_nodes,
        "edges_by_node_alias": expanded_edges,
    }
    return spec, base_alias_by_expanded_alias, dict(clone_count_by_base_alias)


def validate_spec(
    spec: dict[str, Any],
    base_spec: dict[str, Any],
    base_alias_by_expanded_alias: dict[str, str],
    clone_count_by_base_alias: dict[str, int],
) -> dict[str, Any]:
    stages = list(spec["stages"])
    depth = int(spec["depth"])
    nodes_by_stage = spec["nodes"]
    edges = spec["edges_by_node_alias"]
    base_edges = {
        str(parent_alias): list(child_aliases)
        for parent_alias, child_aliases in base_spec["edges_by_node_alias"].items()
    }
    base_nodes = {
        str(node["alias"]): dict(node)
        for stage_nodes in base_spec["nodes"].values()
        for node in stage_nodes
    }
    all_nodes = [node for stage in stages for node in nodes_by_stage.get(stage, [])]
    aliases = [str(node["alias"]) for node in all_nodes]
    agent_ids = [str(node.get("agent_id", node["alias"])) for node in all_nodes]
    alias_counts = Counter(aliases)
    agent_counts = Counter(agent_ids)

    parent_by_child: dict[str, list[str]] = defaultdict(list)
    child_counts = []
    for parent_alias, child_aliases in edges.items():
        if child_aliases:
            child_counts.append(len(child_aliases))
        for child_alias in child_aliases:
            parent_by_child[child_alias].append(parent_alias)
    cross_prefix_duplicate_children = {
        child_alias: sorted(parent_aliases)
        for child_alias, parent_aliases in parent_by_child.items()
        if len(parent_aliases) > 1
    }
    duplicate_agent_ids = {
        agent_id: count
        for agent_id, count in agent_counts.items()
        if count > 1
    }
    per_stage_node_counts = {
        stage: len(nodes_by_stage.get(stage, []))
        for stage in stages
    }
    g_counts_by_stage = {
        stage: dict(Counter(int(node["g"]) for node in nodes_by_stage.get(stage, [])))
        for stage in stages
    }

    clone_g_consistency_errors: list[dict[str, Any]] = []
    connectivity_pattern_errors: list[dict[str, Any]] = []
    for node in all_nodes:
        alias = str(node["alias"])
        base_alias = str(node["base_alias"])
        if int(node["g"]) != int(base_nodes[base_alias]["g"]):
            clone_g_consistency_errors.append(
                {
                    "alias": alias,
                    "base_alias": base_alias,
                    "clone_g": int(node["g"]),
                    "base_g": int(base_nodes[base_alias]["g"]),
                }
            )
        clone_child_bases = [base_alias_by_expanded_alias[child] for child in edges.get(alias, [])]
        expected_child_bases = list(base_edges.get(base_alias, []))
        if clone_child_bases != expected_child_bases:
            connectivity_pattern_errors.append(
                {
                    "alias": alias,
                    "base_alias": base_alias,
                    "expected_child_bases": expected_child_bases,
                    "actual_child_bases": clone_child_bases,
                }
            )

    base_paths_original = sorted(tuple(path) for path in enumerate_paths(base_edges, depth))
    base_paths_expanded = sorted(tuple(path) for path in enumerate_paths(edges, depth, base_alias_by_expanded_alias=base_alias_by_expanded_alias))
    path_semantics_preserved = base_paths_original == base_paths_expanded

    reused_aliases = source_parent_sets(base_edges)
    validation = {
        "depth": depth,
        "num_paths": count_paths(edges, depth),
        "source_num_paths": count_paths(base_edges, depth),
        "total_agent_ids": len(agent_ids),
        "duplicate_agent_count": len(duplicate_agent_ids),
        "duplicate_agent_ids": duplicate_agent_ids,
        "cross_prefix_duplicate_count": len(cross_prefix_duplicate_children),
        "cross_prefix_duplicate_agent_ids": sorted(cross_prefix_duplicate_children),
        "cross_prefix_reuse_rate": len(cross_prefix_duplicate_children) / max(1, len(agent_ids)),
        "alias_duplicate_count": sum(1 for count in alias_counts.values() if count > 1),
        "per_stage_node_counts": per_stage_node_counts,
        "g_counts_by_stage": g_counts_by_stage,
        "root_branching": len(edges.get("ROOT", [])),
        "min_branching": min(child_counts) if child_counts else 0,
        "max_branching": max(child_counts) if child_counts else 0,
        "mean_branching": statistics.fmean(child_counts) if child_counts else 0.0,
        "original_connectivity_pattern_preserved": not connectivity_pattern_errors,
        "connectivity_pattern_error_count": len(connectivity_pattern_errors),
        "connectivity_pattern_error_sample": connectivity_pattern_errors[:10],
        "clone_g_consistent_with_base_alias": not clone_g_consistency_errors,
        "clone_g_consistency_error_count": len(clone_g_consistency_errors),
        "clone_g_consistency_error_sample": clone_g_consistency_errors[:10],
        "path_semantics_preserved": path_semantics_preserved,
        "source_reused_aliases": reused_aliases,
        "clone_count_by_base_alias": clone_count_by_base_alias,
        "cloned_aliases": {
            base_alias: clone_count
            for base_alias, clone_count in clone_count_by_base_alias.items()
            if clone_count > 1
        },
        "metadata": spec.get("metadata", {}),
    }
    return validation


def layout_nodes(spec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    positions: dict[str, tuple[float, float]] = {"ROOT": (LEFT_MARGIN, TOP_MARGIN)}
    max_stage_count = max(len(spec["nodes"].get(stage, [])) for stage in spec["stages"])
    for stage_idx, stage in enumerate(spec["stages"], start=1):
        nodes = spec["nodes"][stage]
        x = LEFT_MARGIN + stage_idx * STAGE_X_GAP
        y0 = TOP_MARGIN + max(0.0, (max_stage_count - len(nodes)) * NODE_Y_GAP / 2.0)
        for row_idx, node in enumerate(nodes):
            positions[node["alias"]] = (x, y0 + row_idx * NODE_Y_GAP)
    return positions


def render_svg(spec: dict[str, Any], validation: dict[str, Any]) -> str:
    stages = list(spec["stages"])
    max_count = max(len(spec["nodes"].get(stage, [])) for stage in stages)
    width = LEFT_MARGIN + (len(stages) + 1) * STAGE_X_GAP + NODE_W + 140
    height = TOP_MARGIN + max_count * NODE_Y_GAP + 130
    positions = layout_nodes(spec)

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
        '<text class="title" x="42" y="34">Shared Basin 4/5 Prefix-Dedup Tree</text>',
        (
            '<text class="subtitle" x="42" y="54">'
            f'Parent-specific clone expansion of shared_basin_strong_4of5_minimal: '
            f'{validation["num_paths"]} paths, {validation["total_agent_ids"]} nodes, '
            f'cross-prefix reuse={validation["cross_prefix_duplicate_count"]}.'
            "</text>"
        ),
        (
            '<text class="subtitle" x="42" y="72">'
            'Green nodes: g=0 share/upload-capable. Red nodes: g=1 barrier/unshared. '
            'display_alias shows the cloned node; base_alias records the original DAG alias.'
            "</text>"
        ),
    ]

    root_x, root_y = positions["ROOT"]
    lines.append(f'<rect class="root" x="{root_x:.1f}" y="{root_y:.1f}" width="88" height="30" />')
    lines.append(f'<text class="stage-title" x="{root_x + 23:.1f}" y="{root_y + 20:.1f}">ROOT</text>')
    for stage_idx, stage in enumerate(stages, start=1):
        x = LEFT_MARGIN + stage_idx * STAGE_X_GAP
        lines.append(f'<text class="stage-title" x="{x:.1f}" y="102">{escape(stage)}</text>')

    for parent_alias, child_aliases in spec["edges_by_node_alias"].items():
        if parent_alias not in positions:
            continue
        px, py = positions[parent_alias]
        sx = px + (88 if parent_alias == "ROOT" else NODE_W)
        sy = py + (15 if parent_alias == "ROOT" else NODE_H / 2)
        for child_alias in child_aliases:
            if child_alias not in positions:
                continue
            cx, cy = positions[child_alias]
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
                f'<text class="node-label" x="{x + 4:.1f}" y="{y + 8:.1f}">{escape(str(node["display_alias"]))}</text>'
            )
            lines.append(
                f'<text class="node-meta" x="{x + 4:.1f}" y="{y + 17:.1f}">base={escape(str(node["base_alias"]))} | g={g} | parent={escape(str(node["parent_alias"]))}</text>'
            )

    legend_x = width - 420
    legend_y = 26
    lines.extend(
        [
            f'<rect x="{legend_x}" y="{legend_y}" width="360" height="74" rx="10" fill="#ffffff" stroke="#cbd5e1" />',
            f'<rect x="{legend_x + 12}" y="{legend_y + 16}" width="18" height="14" rx="4" fill="#eaf7ef" stroke="#2f855a" />',
            f'<text class="subtitle" x="{legend_x + 38}" y="{legend_y + 27}">g=0 share/upload-capable interface</text>',
            f'<rect x="{legend_x + 12}" y="{legend_y + 42}" width="18" height="14" rx="4" fill="#fff1f2" stroke="#c53030" />',
            f'<text class="subtitle" x="{legend_x + 38}" y="{legend_y + 53}">g=1 barrier / unshared interface</text>',
        ]
    )
    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    base_spec = read_json(BASE_SPEC_PATH)
    spec, base_alias_by_expanded_alias, clone_count_by_base_alias = expand_prefix_dedup_spec(base_spec)
    validation = validate_spec(spec, base_spec, base_alias_by_expanded_alias, clone_count_by_base_alias)
    write_json(SPEC_PATH, spec)
    write_json(VALIDATION_PATH, validation)
    SVG_PATH.write_text(render_svg(spec, validation), encoding="utf-8")
    print(
        json.dumps(
            {
                "spec_path": str(SPEC_PATH.relative_to(ROOT)),
                "validation_path": str(VALIDATION_PATH.relative_to(ROOT)),
                "svg_path": str(SVG_PATH.relative_to(ROOT)),
                "validation": validation,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
