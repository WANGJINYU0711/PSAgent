"""Render 15-agent and 25-agent 4/5-share variants of shared_basin_strong.

These diagrams are profile-clone expansions of the current 5-agent
``shared_basin_strong`` tree:

- 15-agent version: each original stage profile is cloned 3 times.
- 25-agent version: each original stage profile is cloned 5 times.

The original 4/5 share ratio is preserved because the base tree has four g=0
profiles and one g=1 profile per stage. Route labels, legal continuation
patterns, base costs, capability profile semantics, and deliberation modes are
inherited from the base profile being cloned.
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
    base_alias: str
    clone_index: int
    agent_id: str
    base_agent_id: str
    role: str
    stage_name: str
    route_label: str
    g: int
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


def edge_color(base_alias: str) -> str:
    if base_alias == "R":
        return "#475569"

    stage_letter = base_alias[0]
    try:
        local_idx = int(base_alias[1:]) - 1
    except ValueError:
        local_idx = 0
    stage_offset = {"A": 0, "B": 2, "C": 4, "D": 6, "E": 8}.get(stage_letter, 0)
    return EDGE_PALETTE[(stage_offset + local_idx) % len(EDGE_PALETTE)]


def build_base_tree() -> tuple[list[str], dict[str, str], dict[str, list[str]], dict[str, object]]:
    family_spec, agent_map = TreeFamilyGenerator().build_family("shared_basin_strong", seed=0)

    alias_by_id: dict[str, str] = {}
    base_id_by_alias: dict[str, str] = {}
    for stage_name in family_spec.stages:
        letter = STAGE_LETTERS[stage_name]
        for idx, agent_id in enumerate(family_spec.stage_agents[stage_name], start=1):
            alias = f"{letter}{idx}"
            alias_by_id[agent_id] = alias
            base_id_by_alias[alias] = agent_id

    base_adjacency: dict[str, list[str]] = {"R": []}
    for agent_id in family_spec.stage_agents["stage1"]:
        base_adjacency["R"].append(alias_by_id[agent_id])
    for prefix, child_ids in (family_spec.allowed_children or {}).items():
        if not prefix:
            continue
        parent_alias = alias_by_id[prefix[-1]]
        base_adjacency[parent_alias] = [alias_by_id[child_id] for child_id in child_ids]

    return family_spec.stages, base_id_by_alias, base_adjacency, agent_map


def build_scaled_views(
    *,
    clones_per_profile: int,
) -> tuple[list[NodeView], dict[str, list[str]], list[str]]:
    stages, base_id_by_alias, base_adjacency, agent_map = build_base_tree()
    total_per_stage = 5 * clones_per_profile

    x_origin = 220.0
    x_gap = 290.0
    y_origin = 112.0
    y_gap = 68.0

    views: list[NodeView] = []
    aliases_by_base_alias: dict[str, list[str]] = {}

    for stage_idx, stage_name in enumerate(stages):
        letter = STAGE_LETTERS[stage_name]
        ordinal = 1
        for base_profile_idx in range(1, 6):
            base_alias = f"{letter}{base_profile_idx}"
            base_agent_id = base_id_by_alias[base_alias]
            base_spec = agent_map[base_agent_id]
            role = role_from_agent_id(base_agent_id)
            for clone_index in range(1, clones_per_profile + 1):
                alias = f"{letter}{ordinal:02d}"
                ordinal += 1
                aliases_by_base_alias.setdefault(base_alias, []).append(alias)
                views.append(
                    NodeView(
                        alias=alias,
                        base_alias=base_alias,
                        clone_index=clone_index,
                        agent_id=(
                            f"{stage_name}_{role}_clone{clone_index:02d}"
                            f"_g{base_spec.g}_{ordinal - 2}"
                        ),
                        base_agent_id=base_agent_id,
                        role=role,
                        stage_name=stage_name,
                        route_label=str(base_spec.route_label),
                        g=int(base_spec.g),
                        deliberation_mode=str(base_spec.deliberation_mode),
                        base_cost=float(base_spec.base_cost),
                        x=x_origin + stage_idx * x_gap,
                        y=y_origin + (ordinal - 2) * y_gap,
                    )
                )

    adjacency: dict[str, list[str]] = {"R": [view.alias for view in views if view.stage_name == "stage1"]}
    view_by_alias = {view.alias: view for view in views}
    for view in views:
        if view.stage_name == stages[-1]:
            adjacency[view.alias] = []
            continue
        child_base_aliases = base_adjacency.get(view.base_alias, [])
        children: list[str] = []
        for child_base_alias in child_base_aliases:
            children.extend(aliases_by_base_alias.get(child_base_alias, []))
        adjacency[view.alias] = children

    # Preserve deterministic ordering and drop any accidental aliases not created.
    valid_aliases = set(view_by_alias)
    adjacency = {
        parent: [child for child in children if child in valid_aliases]
        for parent, children in adjacency.items()
    }
    if len(adjacency["R"]) != total_per_stage:
        raise RuntimeError(f"Expected {total_per_stage} root children, got {len(adjacency['R'])}.")
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
        ".node-alias { font-size: 16px; font-weight: 700; }",
        ".node-role { font-size: 10px; }",
        ".node-meta { font-size: 9px; fill: #4b5563; }",
        ".node-note { font-size: 9px; font-weight: 700; fill: #7c2d12; }",
        ".edge { stroke-width: 1.15; fill: none; opacity: 0.22; }",
        ".root-box { fill: #f9fafb; stroke: #111827; stroke-width: 1.8; rx: 12; }",
        "</style>",
    ]


def render_svg(
    *,
    clones_per_profile: int,
    views: list[NodeView],
    adjacency: dict[str, list[str]],
    stages: list[str],
) -> str:
    node_width = 218.0
    node_height = 50.0
    width = 2150.0
    height = max(760.0, max(view.y for view in views) + node_height + 92.0)
    root_x = 45.0
    root_y = height / 2.0 - 28.0
    root_w = 110.0
    root_h = 56.0

    view_by_alias = {view.alias: view for view in views}
    g_by_alias = {view.alias: view.g for view in views}
    stage_x = {STAGE_LETTERS[s]: 220.0 + i * 290.0 for i, s in enumerate(stages)}
    total_per_stage = 5 * clones_per_profile

    lines = svg_header(width, height)
    lines.append(
        f'<text class="title" x="45" y="34">4/5-Share Scaled Tree: '
        f'{total_per_stage} Candidates Per Stage</text>'
    )
    lines.append(
        '<text class="subtitle" x="45" y="54">'
        f"Each original profile is cloned {clones_per_profile}x. "
        f"Per stage: {4 * clones_per_profile} g=0 share nodes + "
        f"{clones_per_profile} g=1 unshare nodes. Route/topology/distribution inherited from 4/5."
        "</text>"
    )

    for stage_name in stages:
        letter = STAGE_LETTERS[stage_name]
        lines.append(
            f'<text class="stage-title" x="{stage_x[letter] + 10:.1f}" y="91">'
            f'{escape(letter)} / {escape(stage_name)}</text>'
        )

    lines.append(
        f'<rect class="root-box" x="{root_x}" y="{root_y}" width="{root_w}" height="{root_h}" />'
    )
    lines.append(f'<text class="node-alias" x="{root_x + 40}" y="{root_y + 24}">R</text>')
    lines.append(f'<text class="node-role" x="{root_x + 23}" y="{root_y + 44}">human / root</text>')

    for child_alias in adjacency.get("R", []):
        child = view_by_alias[child_alias]
        sx = root_x + root_w
        sy = root_y + root_h / 2
        tx = child.x
        ty = child.y + node_height / 2
        lines.append(
            f'<path class="edge" stroke="{edge_color("R")}" '
            f'd="M {sx:.1f},{sy:.1f} C {sx + 36:.1f},{sy:.1f} '
            f'{tx - 36:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" />'
        )

    for parent_alias, child_aliases in adjacency.items():
        if parent_alias == "R":
            continue
        parent = view_by_alias[parent_alias]
        sx = parent.x + node_width
        sy = parent.y + node_height / 2
        stroke = edge_color(parent.base_alias)
        for child_alias in child_aliases:
            child = view_by_alias[child_alias]
            tx = child.x
            ty = child.y + node_height / 2
            lines.append(
                f'<path class="edge" stroke="{stroke}" '
                f'd="M {sx:.1f},{sy:.1f} C {sx + 34:.1f},{sy:.1f} '
                f'{tx - 34:.1f},{ty:.1f} {tx:.1f},{ty:.1f}" />'
            )

    for view in views:
        fill, stroke = LANE_COLORS[lane_kind(view.route_label)]
        gold_border = has_gold_border(view.alias, adjacency, g_by_alias)
        border = "#d4a017" if gold_border else stroke
        stroke_width = 2.7 if gold_border else 1.5
        lines.append(
            f'<rect x="{view.x}" y="{view.y}" width="{node_width}" height="{node_height}" '
            f'rx="10" fill="{fill}" stroke="{border}" stroke-width="{stroke_width}" />'
        )
        lines.append(
            f'<text class="node-alias" x="{view.x + 9}" y="{view.y + 18}">'
            f'{escape(view.alias)}</text>'
        )
        lines.append(
            f'<text class="node-meta" x="{view.x + 52}" y="{view.y + 18}">'
            f'g={view.g} | base={view.base_alias} | {escape(view.route_label)}</text>'
        )
        lines.append(
            f'<text class="node-role" x="{view.x + 9}" y="{view.y + 34}">'
            f'{escape(view.role)}</text>'
        )
        lines.append(
            f'<text class="node-meta" x="{view.x + 9}" y="{view.y + 46}">'
            f'clone={view.clone_index} | delib={view.deliberation_mode} | cost={view.base_cost:.3f}</text>'
        )
        if gold_border:
            badge_x = view.x + node_width - 42
            badge_y = view.y + 6
            badge_text = (
                "FS"
                if is_full_share_selection_node(view.alias, adjacency, g_by_alias)
                else "SL"
            )
            lines.append(
                f'<rect x="{badge_x}" y="{badge_y}" width="32" height="16" rx="7" '
                'fill="#fef3c7" stroke="#d97706" stroke-width="1.1" />'
            )
            lines.append(
                f'<text class="node-note" x="{badge_x + 8}" y="{badge_y + 12}">'
                f"{badge_text}</text>"
            )

    legend_x = 1685.0
    legend_y = 48.0
    legend = [
        ("public", "inherited public route lane"),
        ("mixed", "inherited mixed route lane"),
        ("private", "inherited private route lane"),
        ("barrier", "inherited barrier route lane"),
    ]
    lines.append(
        f'<rect x="{legend_x}" y="{legend_y}" width="420" height="218" rx="12" '
        'fill="#ffffff" stroke="#cbd5e1" stroke-width="1.2" />'
    )
    lines.append(
        f'<text class="stage-title" x="{legend_x + 12}" y="{legend_y + 24}">Legend</text>'
    )
    for idx, (kind, label) in enumerate(legend):
        fill, stroke = LANE_COLORS[kind]
        y = legend_y + 43 + idx * 28
        lines.append(
            f'<rect x="{legend_x + 12}" y="{y}" width="18" height="18" rx="4" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="1.5" />'
        )
        lines.append(
            f'<text class="node-role" x="{legend_x + 40}" y="{y + 13}">{escape(label)}</text>'
        )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 162}">'
        "Gold: FS = descendant child-subtree all g=0; SL = g=0 share leaf</text>"
    )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 179}">'
        "Edges = route-label continuations inherited from original 4/5 tree</text>"
    )
    lines.append(
        f'<text class="node-meta" x="{legend_x + 12}" y="{legend_y + 196}">'
        f"Profile clone count = {clones_per_profile}; share ratio = 4/5 at every stage</text>"
    )

    lines.append("</svg>")
    return "\n".join(lines)


def write_legend(
    *,
    clones_per_profile: int,
    views: Iterable[NodeView],
    adjacency: dict[str, list[str]],
) -> str:
    views_list = list(views)
    g_by_alias = {view.alias: view.g for view in views_list}
    total_per_stage = 5 * clones_per_profile
    lines = [
        f"# shared_basin 4/5 scaled tree legend ({total_per_stage} candidates per stage)",
        "",
        "This is a profile-clone expansion of the current `shared_basin_strong` "
        "4/5 tree. It keeps the original route labels, legal continuation "
        "patterns, base costs, capability-profile semantics, and deliberation "
        "modes. The only structural change is candidate count per stage.",
        "",
        f"Each of the five original profiles is cloned `{clones_per_profile}` times. "
        f"Each stage therefore has `{4 * clones_per_profile}` `g=0` share nodes "
        f"and `{clones_per_profile}` `g=1` unshare nodes.",
        "",
        "| Alias | Stage | g | Base alias | Clone | FS subtree | Share leaf | Gold | Route lane | Role | Delib | Base cost | Next count | Base agent id |",
        "|---|---|---:|---|---:|---|---|---|---|---|---|---:|---:|---|",
    ]
    for view in views_list:
        full_share_selection = (
            "yes" if is_full_share_selection_node(view.alias, adjacency, g_by_alias) else "no"
        )
        share_leaf = "yes" if is_share_leaf(view.alias, adjacency, g_by_alias) else "no"
        gold_border = "yes" if has_gold_border(view.alias, adjacency, g_by_alias) else "no"
        lines.append(
            f"| {view.alias} | {view.stage_name} | {view.g} | {view.base_alias} | "
            f"{view.clone_index} | {full_share_selection} | {share_leaf} | {gold_border} | "
            f"{view.route_label} | {view.role} | {view.deliberation_mode} | "
            f"{view.base_cost:.3f} | {len(adjacency.get(view.alias, []))} | "
            f"{view.base_agent_id} |"
        )
    return "\n".join(lines) + "\n"


def render_variant(clones_per_profile: int) -> tuple[Path, Path]:
    total_per_stage = 5 * clones_per_profile
    views, adjacency, stages = build_scaled_views(clones_per_profile=clones_per_profile)
    svg_path = OUTPUT_DIR / f"shared_basin_4of5_{total_per_stage}agents_tree.svg"
    legend_path = OUTPUT_DIR / f"shared_basin_4of5_{total_per_stage}agents_tree_legend.md"
    svg_path.write_text(
        render_svg(
            clones_per_profile=clones_per_profile,
            views=views,
            adjacency=adjacency,
            stages=stages,
        ),
        encoding="utf-8",
    )
    legend_path.write_text(
        write_legend(
            clones_per_profile=clones_per_profile,
            views=views,
            adjacency=adjacency,
        ),
        encoding="utf-8",
    )
    return svg_path, legend_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for clones_per_profile in (3, 5):
        svg_path, legend_path = render_variant(clones_per_profile)
        print(f"wrote {svg_path}")
        print(f"wrote {legend_path}")


if __name__ == "__main__":
    main()
