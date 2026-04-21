"""Render a 2-of-5 g-only variant of the current shared_basin_strong tree.

This diagram intentionally reuses the current 4/5 ``shared_basin_strong`` node
roles, route labels, legal continuation topology, costs/capability profiles, and
deliberation-mode semantics. The only visualized intervention is the per-stage
``g`` assignment: the first two aliases in each stage are g=0 and the remaining
three aliases are g=1.
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

from tree_family.generator import TreeFamilyGenerator


OUTPUT_DIR = Path("analysis")
SVG_PATH = OUTPUT_DIR / "shared_basin_2of5_tree.svg"
LEGEND_PATH = OUTPUT_DIR / "shared_basin_2of5_tree_legend.md"

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


@dataclass(frozen=True)
class NodeView:
    alias: str
    agent_id: str
    base_agent_id_4of5: str
    role: str
    stage_name: str
    route_label: str
    g_2of5: int
    g_4of5: int
    deliberation_mode: str
    base_cost: float
    x: float
    y: float


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


def role_from_agent_id(agent_id: str) -> str:
    role = agent_id.split("_", 1)[1]
    if "_g0_" in role:
        return role.split("_g0_", 1)[0]
    if "_g1_" in role:
        return role.split("_g1_", 1)[0]
    return role


def display_agent_id(stage_name: str, role: str, g: int, alias: str) -> str:
    idx = int(alias[1:]) - 1
    return f"{stage_name}_{role}_g{g}_{idx}"


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


def g_2of5_for_alias(alias: str) -> int:
    return 0 if int(alias[1:]) <= 2 else 1


def build_views_and_adjacency() -> tuple[list[NodeView], dict[str, list[str]], list[str]]:
    family_spec, agent_map = TreeFamilyGenerator().build_family("shared_basin_strong", seed=0)
    stages = family_spec.stages

    x_origin = 220.0
    x_gap = 260.0
    y_origin = 100.0
    y_gap = 120.0

    alias_by_id: dict[str, str] = {}
    views: list[NodeView] = []
    for stage_idx, stage_name in enumerate(stages):
        letter = STAGE_LETTERS[stage_name]
        for idx, base_agent_id in enumerate(family_spec.stage_agents[stage_name], start=1):
            base_spec = agent_map[base_agent_id]
            alias = f"{letter}{idx}"
            role = role_from_agent_id(base_agent_id)
            g_2of5 = g_2of5_for_alias(alias)
            alias_by_id[base_agent_id] = alias
            views.append(
                NodeView(
                    alias=alias,
                    agent_id=display_agent_id(stage_name, role, g_2of5, alias),
                    base_agent_id_4of5=base_agent_id,
                    role=role,
                    stage_name=stage_name,
                    route_label=str(base_spec.route_label),
                    g_2of5=g_2of5,
                    g_4of5=int(base_spec.g),
                    deliberation_mode=str(base_spec.deliberation_mode),
                    base_cost=float(base_spec.base_cost),
                    x=x_origin + stage_idx * x_gap,
                    y=y_origin + (idx - 1) * y_gap,
                )
            )

    adjacency: dict[str, list[str]] = {"R": []}
    for base_agent_id in family_spec.stage_agents["stage1"]:
        adjacency["R"].append(alias_by_id[base_agent_id])

    allowed_children = family_spec.allowed_children or {}
    for prefix, child_ids in allowed_children.items():
        if not prefix:
            continue
        parent_alias = alias_by_id[prefix[-1]]
        adjacency[parent_alias] = [alias_by_id[child_id] for child_id in child_ids]

    return views, adjacency, stages


def is_full_share_selection_node(
    alias: str,
    adjacency: dict[str, list[str]],
    g_by_alias: dict[str, int],
) -> bool:
    child_aliases = adjacency.get(alias, [])
    if not child_aliases:
        return False
    stack = list(child_aliases)
    while stack:
        child_alias = stack.pop()
        if g_by_alias[child_alias] != 0:
            return False
        stack.extend(adjacency.get(child_alias, []))
    return True


def is_share_leaf(alias: str, adjacency: dict[str, list[str]], g_by_alias: dict[str, int]) -> bool:
    return not adjacency.get(alias, []) and g_by_alias[alias] == 0


def has_gold_border(alias: str, adjacency: dict[str, list[str]], g_by_alias: dict[str, int]) -> bool:
    return is_full_share_selection_node(alias, adjacency, g_by_alias) or is_share_leaf(
        alias,
        adjacency,
        g_by_alias,
    )


def svg_header(width: float, height: float) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: Arial, sans-serif; fill: #1f2937; }",
        ".title { font-size: 22px; font-weight: 800; }",
        ".subtitle { font-size: 12px; fill: #475569; }",
        ".stage-title { font-size: 18px; font-weight: 700; }",
        ".node-alias { font-size: 18px; font-weight: 700; }",
        ".node-role { font-size: 11px; }",
        ".node-meta { font-size: 10px; fill: #4b5563; }",
        ".node-note { font-size: 10px; font-weight: 700; fill: #7c2d12; }",
        ".edge { stroke-width: 2.1; fill: none; opacity: 0.88; }",
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
    g_by_alias = {view.alias: view.g_2of5 for view in views}
    stage_x = {STAGE_LETTERS[s]: 220.0 + i * 260.0 for i, s in enumerate(stages)}

    lines = svg_header(width, height)
    lines.append('<text class="title" x="45" y="34">2-of-5 Share Tree: g-only Variant</text>')
    lines.append(
        '<text class="subtitle" x="45" y="54">'
        "Same nodes, routes, continuations, base costs, capability profiles, and "
        "deliberation modes as shared_basin_strong; only g is changed to 2/5."
        "</text>"
    )

    for stage_name in stages:
        letter = STAGE_LETTERS[stage_name]
        lines.append(
            f'<text class="stage-title" x="{stage_x[letter] + 10:.1f}" y="82">'
            f'{escape(letter)} / {escape(stage_name)}</text>'
        )

    lines.append(
        f'<rect class="root-box" x="{root_x}" y="{root_y}" width="{root_w}" height="{root_h}" />'
    )
    lines.append(f'<text class="node-alias" x="{root_x + 38}" y="{root_y + 24}">R</text>')
    lines.append(f'<text class="node-role" x="{root_x + 23}" y="{root_y + 44}">human / root</text>')

    def anchor(alias: str) -> tuple[float, float]:
        view = view_by_alias[alias]
        return view.x, view.y

    for child_alias in adjacency.get("R", []):
        cx, cy = anchor(child_alias)
        sx = root_x + root_w
        sy = root_y + root_h / 2
        tx = cx
        ty = cy + node_height / 2
        c1x = sx + 36
        c2x = tx - 36
        lines.append(
            f'<path class="edge" stroke="{edge_color("R")}" '
            f'd="M {sx:.1f},{sy:.1f} C {c1x:.1f},{sy:.1f} '
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
                f'<path class="edge" stroke="{stroke}" '
                f'd="M {sx:.1f},{sy:.1f} C {c1x:.1f},{sy:.1f} '
                f'{c2x:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" />'
            )

    for view in views:
        fill, stroke = LANE_COLORS[lane_kind(view.route_label)]
        gold_border = has_gold_border(view.alias, adjacency, g_by_alias)
        border = "#d4a017" if gold_border else stroke
        stroke_width = 3.0 if gold_border else 1.8
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
            f'g={view.g_2of5} | {escape(view.route_label)}</text>'
        )
        lines.append(
            f'<text class="node-role" x="{view.x + 10}" y="{view.y + 40}">'
            f'{escape(view.role)}</text>'
        )
        lines.append(
            f'<text class="node-meta" x="{view.x + 10}" y="{view.y + 56}">'
            f'delib={view.deliberation_mode} | base_cost={view.base_cost:.3f}</text>'
        )
        if gold_border:
            badge_x = view.x + node_width - 54
            badge_y = view.y + 8
            badge_text = (
                "FS"
                if is_full_share_selection_node(view.alias, adjacency, g_by_alias)
                else "SL"
            )
            lines.append(
                f'<rect x="{badge_x}" y="{badge_y}" width="42" height="18" rx="8" '
                'fill="#fef3c7" stroke="#d97706" stroke-width="1.2" />'
            )
            lines.append(
                f'<text class="node-note" x="{badge_x + 13}" y="{badge_y + 13}">'
                f"{badge_text}</text>"
            )

    legend_x = 1505.0
    legend_y = 48.0
    legend = [
        ("public", "inherited public route lane"),
        ("mixed", "inherited mixed route lane"),
        ("private", "inherited private route lane"),
        ("barrier", "inherited barrier route lane"),
    ]
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="300" height="194" rx="12" '
        'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" />'
    )
    lines.append(
        f'<text class="stage-title" x="{legend_x + 12}" y="{legend_y + 24}">Legend</text>'
    )
    for idx, (kind, label) in enumerate(legend):
        fill, stroke = LANE_COLORS[kind]
        y = legend_y + 42 + idx * 27
        lines.append(
            f'<rect x="{legend_x + 12}" y="{y}" width="18" height="18" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
        )
        lines.append(
            f'<text class="node-role" x="{legend_x + 40}" y="{y + 13}">{escape(label)}</text>'
        )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 158}">'
        "Edges = inherited legal continuations from 4/5 tree</text>"
    )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 175}">'
        "Gold: FS = descendant child-subtree all g=0; SL = g=0 share leaf</text>"
    )

    lines.append("</svg>")
    return "\n".join(lines)


def write_legend(views: Iterable[NodeView], adjacency: dict[str, list[str]]) -> str:
    views_list = list(views)
    g_by_alias = {view.alias: view.g_2of5 for view in views_list}
    lines = [
        "# shared_basin_2of5 tree legend",
        "",
        "This is a g-only 2-of-5 variant of the current `shared_basin_strong` "
        "4/5 tree. It reuses the same node roles, route labels, legal "
        "continuations, base costs, capability-profile semantics, and "
        "deliberation modes. The only intended difference is the displayed "
        "`g_2of5`: aliases 1-2 in each stage are `g=0`, aliases 3-5 are `g=1`.",
        "",
        "Gold border marks full-share selection points (`FS`: the whole descendant "
        "child-subtree is `g=0`) and share leaves (`SL`: `g=0` leaf).",
        "",
        "| Alias | Stage | g_2of5 | g_4of5 | FS subtree | Share leaf | Gold | Route lane | Role | Delib | Base cost | Next | 4/5 base agent id |",
        "|---|---|---:|---:|---|---|---|---|---|---|---:|---|---|",
    ]
    for view in views_list:
        next_aliases = ",".join(adjacency.get(view.alias, []))
        full_share_selection = (
            "yes" if is_full_share_selection_node(view.alias, adjacency, g_by_alias) else "no"
        )
        share_leaf = "yes" if is_share_leaf(view.alias, adjacency, g_by_alias) else "no"
        gold_border = "yes" if has_gold_border(view.alias, adjacency, g_by_alias) else "no"
        lines.append(
            f"| {view.alias} | {view.stage_name} | {view.g_2of5} | {view.g_4of5} | "
            f"{full_share_selection} | {share_leaf} | {gold_border} | "
            f"{view.route_label} | {view.role} | {view.deliberation_mode} | "
            f"{view.base_cost:.3f} | {next_aliases or '-'} | {view.base_agent_id_4of5} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    views, adjacency, stages = build_views_and_adjacency()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_PATH.write_text(render_svg(views, adjacency, stages), encoding="utf-8")
    LEGEND_PATH.write_text(write_legend(views, adjacency), encoding="utf-8")
    print(f"wrote {SVG_PATH}")
    print(f"wrote {LEGEND_PATH}")


if __name__ == "__main__":
    main()
