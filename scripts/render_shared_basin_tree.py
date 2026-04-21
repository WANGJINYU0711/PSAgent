"""Render the current shared_basin_strong workflow tree as an SVG diagram.

This script reads the current family definition from ``TreeFamilyGenerator``
and emits:

- an SVG diagram with root + stage nodes + legal continuation edges
- a Markdown legend mapping aliases (A1, B3, ...) to full agent ids/lanes
"""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "envs") not in sys.path:
    sys.path.insert(0, str(ROOT / "envs"))
if str(ROOT / "baselines") not in sys.path:
    sys.path.insert(0, str(ROOT / "baselines"))

from tree_family.generator import TreeFamilyGenerator
from fixed_tree_env import FixedTreeEnvironment
from risky_ps import RiskyPSPolicy


OUTPUT_DIR = Path("analysis")
SVG_PATH = OUTPUT_DIR / "shared_basin_strong_tree.svg"
LEGEND_PATH = OUTPUT_DIR / "shared_basin_strong_tree_legend.md"

STAGE_LETTERS = {
    "stage1": "A",
    "stage2": "B",
    "stage3": "C",
    "stage4": "D",
    "stage5": "E",
}

LANE_COLORS = {
    "public": ("#eaf7ef", "#2f855a"),
    "mixed": ("#eef4ff", "#3157c8"),
    "private": ("#fff4e6", "#b86b00"),
    "barrier": ("#fff1f2", "#c53030"),
}

EDGE_PALETTE = [
    "#2563eb",
    "#dc2626",
    "#059669",
    "#d97706",
    "#7c3aed",
    "#0891b2",
    "#e11d48",
    "#4f46e5",
    "#65a30d",
    "#c2410c",
]


@dataclass
class NodeView:
    alias: str
    agent_id: str
    role: str
    stage_name: str
    route_label: str
    g: int
    x: float
    y: float
    safe_prefix_count: int = 0
    mixed_prefix_count: int = 0


def lane_kind(route_label: str) -> str:
    if route_label.startswith("public_"):
        return "public"
    if route_label.startswith("mixed_"):
        return "mixed"
    if route_label.startswith("private_barrier_"):
        return "barrier"
    if route_label.startswith("private_"):
        return "private"
    return "mixed"


def shorten_role(agent_id: str) -> str:
    role = agent_id
    if role.startswith("stage"):
        role = role.split("_", 1)[1]
    if "_g0_" in role:
        role = role.split("_g0_", 1)[0]
    elif "_g1_" in role:
        role = role.split("_g1_", 1)[0]
    return role


def edge_color(parent_alias: str) -> str:
    if parent_alias == "R":
        return "#475569"

    stage_letter = parent_alias[0]
    try:
        local_idx = int(parent_alias[1:]) - 1
    except ValueError:
        local_idx = 0
    stage_offset = {"A": 0, "B": 2, "C": 4, "D": 6, "E": 8}.get(stage_letter, 0)
    return EDGE_PALETTE[(stage_offset + local_idx) % len(EDGE_PALETTE)]


def build_safe_prefix_counts() -> tuple[dict[str, int], dict[str, int]]:
    env = FixedTreeEnvironment(
        agent_catalog=[],
        family_kind="shared_basin_strong",
        family_seed=0,
        executor_name="simulated",
    )
    policy = RiskyPSPolicy(seed=0)
    policy.bind_env(env)

    safe_counts: dict[str, int] = {}
    mixed_counts: dict[str, int] = {}
    for prefix, is_safe in policy.safe_prefixes.items():
        if not prefix:
            continue
        agent_id = prefix[-1]
        if is_safe:
            safe_counts[agent_id] = safe_counts.get(agent_id, 0) + 1
    for prefix, is_mixed in policy.mixed_prefixes.items():
        if not prefix:
            continue
        agent_id = prefix[-1]
        if is_mixed:
            mixed_counts[agent_id] = mixed_counts.get(agent_id, 0) + 1
    return safe_counts, mixed_counts


def build_views() -> tuple[list[NodeView], dict[str, list[str]], list[str]]:
    family_spec, agent_map = TreeFamilyGenerator().build_family("shared_basin_strong", seed=0)
    stages = family_spec.stages
    safe_counts, mixed_counts = build_safe_prefix_counts()

    x_origin = 220.0
    x_gap = 260.0
    y_origin = 100.0
    y_gap = 120.0

    views: list[NodeView] = []
    alias_by_id: dict[str, str] = {}
    for stage_idx, stage_name in enumerate(stages):
        letter = STAGE_LETTERS[stage_name]
        agent_ids = family_spec.stage_agents[stage_name]
        for idx, agent_id in enumerate(agent_ids, start=1):
            spec = agent_map[agent_id]
            alias = f"{letter}{idx}"
            alias_by_id[agent_id] = alias
            views.append(
                NodeView(
                    alias=alias,
                    agent_id=agent_id,
                    role=shorten_role(agent_id),
                    stage_name=stage_name,
                    route_label=spec.route_label,
                    g=spec.g,
                    x=x_origin + stage_idx * x_gap,
                    y=y_origin + (idx - 1) * y_gap,
                    safe_prefix_count=safe_counts.get(agent_id, 0),
                    mixed_prefix_count=mixed_counts.get(agent_id, 0),
                )
            )

    adjacency: dict[str, list[str]] = {"R": []}
    for agent_id in family_spec.stage_agents["stage1"]:
        adjacency["R"].append(alias_by_id[agent_id])

    allowed_children = family_spec.allowed_children or {}
    for prefix, child_ids in allowed_children.items():
        if not prefix:
            continue
        parent_alias = alias_by_id[prefix[-1]]
        adjacency[parent_alias] = [alias_by_id[child_id] for child_id in child_ids]

    return views, adjacency, stages


def svg_header(width: float, height: float) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".stage-title { font-size: 18px; font-weight: 700; }",
        ".node-alias { font-size: 18px; font-weight: 700; }",
        ".node-role { font-size: 11px; }",
        ".node-meta { font-size: 10px; fill: #4b5563; }",
        ".node-badge { font-size: 10px; font-weight: 700; fill: #7c2d12; }",
        ".edge { stroke-width: 2.1; fill: none; opacity: 0.9; }",
        ".root-box { fill: #f9fafb; stroke: #111827; stroke-width: 1.8; rx: 12; }",
        "</style>",
    ]


def render_svg(views: list[NodeView], adjacency: dict[str, list[str]], stages: list[str]) -> str:
    node_width = 188.0
    node_height = 64.0
    width = 1840.0
    height = 760.0
    root_x = 45.0
    root_y = 315.0
    root_w = 110.0
    root_h = 56.0

    view_by_alias = {view.alias: view for view in views}
    stage_x = {STAGE_LETTERS[s]: 220.0 + i * 260.0 for i, s in enumerate(stages)}

    lines = svg_header(width, height)

    for stage_name in stages:
        letter = STAGE_LETTERS[stage_name]
        lines.append(
            f'<text class="stage-title" x="{stage_x[letter] + 10:.1f}" y="42">'
            f'{escape(letter)} / {escape(stage_name)}</text>'
        )

    lines.append(
        f'<rect class="root-box" x="{root_x}" y="{root_y}" width="{root_w}" height="{root_h}" />'
    )
    lines.append(f'<text class="node-alias" x="{root_x + 38}" y="{root_y + 24}">R</text>')
    lines.append(f'<text class="node-role" x="{root_x + 23}" y="{root_y + 44}">human / root</text>')

    def node_anchor(alias: str) -> tuple[float, float]:
        view = view_by_alias[alias]
        return view.x, view.y

    for child_alias in adjacency.get("R", []):
        cx, cy = node_anchor(child_alias)
        sx = root_x + root_w
        sy = root_y + root_h / 2
        tx = cx
        ty = cy + node_height / 2
        c1x = sx + 36
        c2x = tx - 36
        stroke = edge_color("R")
        lines.append(
            f'<path class="edge" stroke="{stroke}" d="M {sx:.1f},{sy:.1f} C {c1x:.1f},{sy:.1f} '
            f'{c2x:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" />'
        )

    for parent_alias, child_aliases in adjacency.items():
        if parent_alias == "R":
            continue
        parent = view_by_alias[parent_alias]
        sx = parent.x + node_width
        sy = parent.y + node_height / 2
        stroke = edge_color(parent_alias)
        for child_alias in child_aliases:
            child = view_by_alias[child_alias]
            tx = child.x
            ty = child.y + node_height / 2
            c1x = sx + 34
            c2x = tx - 34
            lines.append(
                f'<path class="edge" stroke="{stroke}" d="M {sx:.1f},{sy:.1f} C {c1x:.1f},{sy:.1f} '
                f'{c2x:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" />'
            )

    for view in views:
        fill, stroke = LANE_COLORS[lane_kind(view.route_label)]
        stroke_width = 3.6 if view.safe_prefix_count > 0 else 1.8
        border = "#d4a017" if view.safe_prefix_count > 0 else stroke
        lines.append(
            f'<rect x="{view.x}" y="{view.y}" width="{node_width}" height="{node_height}" '
            f'rx="12" fill="{fill}" stroke="{border}" stroke-width="{stroke_width}" />'
        )
        lines.append(
            f'<text class="node-alias" x="{view.x + 10}" y="{view.y + 21}">'
            f'{escape(view.alias)}</text>'
        )
        lines.append(
            f'<text class="node-meta" x="{view.x + 50}" y="{view.y + 21}">'
            f'g={view.g} | {escape(view.route_label)}</text>'
        )
        lines.append(
            f'<text class="node-role" x="{view.x + 10}" y="{view.y + 40}">'
            f'{escape(view.role)}</text>'
        )
        lines.append(
            f'<text class="node-meta" x="{view.x + 10}" y="{view.y + 56}">'
            f'{escape(view.agent_id)}</text>'
        )
        if view.safe_prefix_count > 0:
            badge_x = view.x + node_width - 54
            badge_y = view.y + 8
            lines.append(
                f'<rect x="{badge_x}" y="{badge_y}" width="42" height="18" rx="8" '
                'fill="#fef3c7" stroke="#d97706" stroke-width="1.2" />'
            )
            lines.append(
                f'<text class="node-badge" x="{badge_x + 8}" y="{badge_y + 13}">'
                f'FS×{view.safe_prefix_count}</text>'
            )

    legend_x = 1505.0
    legend_y = 48.0
    legend = [
        ("public", "pure / public lane"),
        ("mixed", "mixed or handoff lane"),
        ("private", "private edge lane"),
        ("barrier", "g=1 barrier lane"),
    ]
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="300" height="168" rx="12" '
        'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" />'
    )
    lines.append(
        f'<text class="stage-title" x="{legend_x + 12}" y="{legend_y + 24}">Legend</text>'
    )
    for idx, (kind, label) in enumerate(legend):
        fill, stroke = LANE_COLORS[kind]
        y = legend_y + 42 + idx * 28
        lines.append(
            f'<rect x="{legend_x + 12}" y="{y}" width="18" height="18" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
        )
        lines.append(
            f'<text class="node-role" x="{legend_x + 40}" y="{y + 13}">{escape(label)}</text>'
        )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 154}">'
        "Edges = legal continuations; color = parent node</text>"
    )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 170}">'
        "Gold border / FS badge = safe-prefix node</text>"
    )

    lines.append("</svg>")
    return "\n".join(lines)


def write_legend(views: Iterable[NodeView], adjacency: dict[str, list[str]]) -> str:
    lines = [
        "# shared_basin_strong tree legend",
        "",
        "| Alias | Stage | g | Route lane | Role | Safe count | Mixed count | Next |",
        "|---|---|---:|---|---|---:|---:|---|",
    ]
    alias_order = {"R": -1}
    for view in views:
        alias_order[view.alias] = len(alias_order)
        next_aliases = ",".join(adjacency.get(view.alias, []))
        lines.append(
            f"| {view.alias} | {view.stage_name} | {view.g} | "
            f"{view.route_label} | {view.role} | {view.safe_prefix_count} | "
            f"{view.mixed_prefix_count} | {next_aliases or '-'} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    views, adjacency, stages = build_views()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(render_svg(views, adjacency, stages))
    LEGEND_PATH.write_text(write_legend(views, adjacency))
    print(f"wrote {SVG_PATH}")
    print(f"wrote {LEGEND_PATH}")


if __name__ == "__main__":
    main()
