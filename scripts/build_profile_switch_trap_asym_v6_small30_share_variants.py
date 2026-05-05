from __future__ import annotations

import json
from collections import Counter
from copy import deepcopy
from html import escape
from pathlib import Path
from typing import Any

from build_profile_switch_trap_asym_v6_small30_4of5_tree import (
    ROOT,
    TARGET_SPEC as SOURCE_SPEC,
    descendants_inclusive,
    enumerate_paths,
    validate as validate_source_topology,
    write_json,
)


OUTPUT_DIR = ROOT / "analysis" / "tree_specs"

SOURCE_VARIANT_KEY = "4of5"
BASE_PREFIX = "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v6_small30"
FAMILY_PREFIX = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v6_small30"

VARIANTS: dict[str, dict[str, Any]] = {
    "4of5": {
        "tree_name": f"{BASE_PREFIX}_4of5",
        "family_kind": f"{FAMILY_PREFIX}_4of5",
        "profile_switch_variant": "trap_asym_v6_small30_4of5",
        "g_layout_policy": "small30_exact_internal_and_leaf_4of5_full_stage1_share_basin_v1",
    },
    "2of5": {
        "tree_name": f"{BASE_PREFIX}_2of5",
        "family_kind": f"{FAMILY_PREFIX}_2of5",
        "profile_switch_variant": "trap_asym_v6_small30_2of5",
        "g_layout_policy": "small30_exact_internal_and_leaf_2of5_three_full_stage2_subtrees_v1",
        "full_share_stage2_roots": (
            "s2_n2_general_core_B",
            "s2_n3_general_roaming_C",
            "s2_n4_target_router_D",
        ),
    },
    "all_share": {
        "tree_name": f"{BASE_PREFIX}_all_share",
        "family_kind": f"{FAMILY_PREFIX}_all_share",
        "profile_switch_variant": "trap_asym_v6_small30_all_share",
        "g_layout_policy": "small30_all_share_gonly_v1",
        "g": 0,
    },
    "all_unshare": {
        "tree_name": f"{BASE_PREFIX}_all_unshare",
        "family_kind": f"{FAMILY_PREFIX}_all_unshare",
        "profile_switch_variant": "trap_asym_v6_small30_all_unshare",
        "g_layout_policy": "small30_all_unshare_gonly_v1",
        "g": 1,
    },
}

STAGE_X = {
    "ROOT": 45.0,
    "stage1": 180.0,
    "stage2": 360.0,
    "stage3": 560.0,
    "stage4": 790.0,
    "stage5": 1040.0,
}
NODE_W = 158.0
NODE_H = 34.0
LEAF_GAP = 52.0
TOP_Y = 112.0


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def all_nodes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for stage in spec["stages"] for node in spec["nodes"].get(stage, [])]


def alias_by_conceptual(source: dict[str, Any]) -> dict[str, str]:
    return {
        str(node.get("conceptual_alias")): str(node["alias"])
        for node in all_nodes(source)
    }


def variant_g_values(source: dict[str, Any], variant_key: str) -> dict[str, int]:
    nodes = all_nodes(source)
    if variant_key in {"all_share", "all_unshare"}:
        g = int(VARIANTS[variant_key]["g"])
        return {str(node["alias"]): g for node in nodes}
    if variant_key == "4of5":
        return {str(node["alias"]): int(node["g"]) for node in nodes}
    if variant_key == "2of5":
        aliases = alias_by_conceptual(source)
        shared_aliases: set[str] = set()
        for conceptual_alias in VARIANTS[variant_key]["full_share_stage2_roots"]:
            shared_aliases.update(
                descendants_inclusive(
                    aliases[conceptual_alias],
                    source["edges_by_node_alias"],
                )
            )
        return {
            str(node["alias"]): 0 if str(node["alias"]) in shared_aliases else 1
            for node in nodes
        }
    raise KeyError(f"Unsupported v6 share variant: {variant_key}")


def build_variant(source: dict[str, Any], variant_key: str) -> dict[str, Any]:
    if variant_key == SOURCE_VARIANT_KEY:
        return deepcopy(source)

    variant = VARIANTS[variant_key]
    spec = deepcopy(source)
    spec["tree_name"] = str(variant["tree_name"])
    g_by_alias = variant_g_values(source, variant_key)
    for node in all_nodes(spec):
        node["g"] = int(g_by_alias[str(node["alias"])])

    metadata = dict(spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    conceptual_design = dict(metadata.get("conceptual_design", {}) or {})
    internal_nodes = [node for node in all_nodes(spec) if node["stage"] != "stage5"]
    leaf_nodes = [node for node in all_nodes(spec) if node["stage"] == "stage5"]
    internal_share_count = sum(1 for node in internal_nodes if int(node["g"]) == 0)
    leaf_share_count = sum(1 for node in leaf_nodes if int(node["g"]) == 0)
    conceptual_design.update(
        {
            "leaf_share_fraction": f"{leaf_share_count}/{len(leaf_nodes)}",
            "internal_share_fraction": f"{internal_share_count}/{len(internal_nodes)}",
        }
    )
    if variant_key == "2of5":
        conceptual_design.update(
            {
                "stage2_full_share_roots": list(variant["full_share_stage2_roots"]),
                "stage2_full_share_root_count": len(variant["full_share_stage2_roots"]),
            }
        )

    metadata.update(
        {
            "source_tree_name": source.get("tree_name"),
            "source_tree_path": str(SOURCE_SPEC.relative_to(ROOT)),
            "profile_switch_variant": variant["profile_switch_variant"],
            "share_variant": variant_key,
            "share_variant_family_kind": variant["family_kind"],
            "compatible_with": sorted(compatible_with),
            "preserve_g": False,
            "g_layout_policy": variant["g_layout_policy"],
            "conceptual_design": conceptual_design,
            "purpose": (
                "G-only share control for the v6 small30 profile-switch tree. "
                "Topology, base aliases, conceptual aliases, profile modes, route "
                "semantics, and fast/deep layout are unchanged from the v6 4/5 tree."
            ),
            "notes": [
                "Only expanded-node g values and top-level identifying metadata are changed.",
                "Every g=0 internal node is a full-share subtree root: all descendants are g=0.",
                "Use these variants to isolate PS shared-update behavior from v6 execution topology.",
            ],
        }
    )
    spec["metadata"] = metadata
    return spec


def assert_node_attributes_unchanged(*, source: dict[str, Any], spec: dict[str, Any]) -> bool:
    source_by_alias = {str(node["alias"]): node for node in all_nodes(source)}
    for node in all_nodes(spec):
        source_node = source_by_alias.get(str(node["alias"]))
        if source_node is None:
            return False
        for key, value in node.items():
            if key == "g":
                continue
            if source_node.get(key) != value:
                return False
        for key in source_node:
            if key != "g" and key not in node:
                return False
    return True


def share_subtrees_are_closed(spec: dict[str, Any]) -> tuple[bool, list[str]]:
    edges = spec["edges_by_node_alias"]
    nodes = all_nodes(spec)
    node_by_alias = {str(node["alias"]): node for node in nodes}
    invalid: list[str] = []
    for node in nodes:
        alias = str(node["alias"])
        if int(node["g"]) != 0:
            continue
        descendants = descendants_inclusive(alias, edges)
        if any(int(node_by_alias[item]["g"]) != 0 for item in descendants):
            invalid.append(alias)
    return not invalid, invalid


def validate_variant(
    spec: dict[str, Any],
    *,
    source: dict[str, Any],
    variant_key: str,
) -> dict[str, Any]:
    topology_validation = validate_source_topology(spec)
    expected_g_by_alias = variant_g_values(source, variant_key)
    nodes = all_nodes(spec)
    internal_nodes = [node for node in nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in nodes if node["stage"] == "stage5"]
    share_closed, invalid_share_roots = share_subtrees_are_closed(spec)
    path_records = topology_validation["paths"]
    share_path_count = sum(1 for record in path_records if all(int(g) == 0 for g in record["g"]))
    any_share_path_count = sum(1 for record in path_records if any(int(g) == 0 for g in record["g"]))

    validation = dict(topology_validation)
    validation["family_kind"] = str(VARIANTS[variant_key]["family_kind"])
    validation["share_variant"] = variant_key
    validation["g_counts_by_stage"] = {
        stage: dict(Counter(int(node["g"]) for node in spec["nodes"][stage]))
        for stage in spec["stages"]
    }
    validation["internal_share_count"] = sum(1 for node in internal_nodes if int(node["g"]) == 0)
    validation["internal_private_count"] = sum(1 for node in internal_nodes if int(node["g"]) == 1)
    validation["internal_share_fraction"] = validation["internal_share_count"] / max(1, len(internal_nodes))
    validation["leaf_share_count"] = sum(1 for node in leaf_nodes if int(node["g"]) == 0)
    validation["leaf_private_count"] = sum(1 for node in leaf_nodes if int(node["g"]) == 1)
    validation["leaf_share_fraction"] = validation["leaf_share_count"] / max(1, len(leaf_nodes))
    validation["share_node_count"] = sum(1 for node in nodes if int(node["g"]) == 0)
    validation["unshare_node_count"] = sum(1 for node in nodes if int(node["g"]) == 1)
    validation["share_node_fraction"] = validation["share_node_count"] / max(1, len(nodes))
    validation["share_path_count"] = share_path_count
    validation["any_share_path_count"] = any_share_path_count
    validation["invalid_share_subtree_roots"] = invalid_share_roots
    validation["expected_g_by_alias"] = expected_g_by_alias

    expected_checks = {
        "num_paths_is_30": validation["num_paths"] == 30,
        "all_leaves_at_stage5_depth": validation["leaf_depths"] == [5],
        "topology_edges_unchanged": spec["edges_by_node_alias"] == source["edges_by_node_alias"],
        "topology_stages_unchanged": spec["stages"] == source["stages"],
        "node_attributes_except_g_unchanged": assert_node_attributes_unchanged(
            source=source,
            spec=spec,
        ),
        "only_expected_g_values_changed": all(
            int(node["g"]) == expected_g_by_alias[str(node["alias"])] for node in nodes
        ),
        "share_subtrees_are_closed": share_closed,
        "stage4_has_fast_and_deep": set(validation["profile_mode_counts_by_stage"]["stage4"]) == {
            "fast",
            "deep",
        },
        "stage5_has_fast_and_deep": set(validation["profile_mode_counts_by_stage"]["stage5"]) == {
            "fast",
            "deep",
        },
    }
    if variant_key == "4of5":
        expected_checks.update(
            {
                "internal_share_count_is_24_of_30": validation["internal_share_count"] == 24
                and len(internal_nodes) == 30,
                "leaf_share_count_is_24_of_30": validation["leaf_share_count"] == 24
                and len(leaf_nodes) == 30,
                "full_share_path_count_is_24": share_path_count == 24,
            }
        )
    elif variant_key == "2of5":
        expected_checks.update(
            {
                "internal_share_count_is_12_of_30": validation["internal_share_count"] == 12
                and len(internal_nodes) == 30,
                "leaf_share_count_is_12_of_30": validation["leaf_share_count"] == 12
                and len(leaf_nodes) == 30,
                "stage2_share_suffix_leaf_count_is_12": validation["leaf_share_count"] == 12,
            }
        )
    elif variant_key == "all_share":
        expected_checks.update(
            {
                "all_nodes_are_share": validation["share_node_count"] == len(nodes),
                "full_share_path_count_is_30": share_path_count == 30,
            }
        )
    elif variant_key == "all_unshare":
        expected_checks.update(
            {
                "all_nodes_are_unshare": validation["unshare_node_count"] == len(nodes),
                "full_share_path_count_is_0": share_path_count == 0,
                "any_share_path_count_is_0": any_share_path_count == 0,
            }
        )
    validation["expected_checks"] = expected_checks
    validation["validation_errors"] = [name for name, ok in expected_checks.items() if not ok]
    return validation


def compute_positions(spec: dict[str, Any]) -> dict[str, tuple[float, float]]:
    edges = spec["edges_by_node_alias"]
    paths = enumerate_paths(edges)
    positions: dict[str, tuple[float, float]] = {}
    leaf_y = {path[-1]: TOP_Y + idx * LEAF_GAP for idx, path in enumerate(paths)}
    node_by_alias = {str(node["alias"]): node for node in all_nodes(spec)}

    def y_for(alias: str) -> float:
        if alias in leaf_y:
            return leaf_y[alias]
        children = edges.get(alias, [])
        if not children:
            return TOP_Y
        return sum(y_for(child) for child in children) / len(children)

    positions["ROOT"] = (STAGE_X["ROOT"], y_for("ROOT"))
    for alias, node in node_by_alias.items():
        positions[alias] = (STAGE_X[str(node["stage"])], y_for(alias))
    return positions


def mode_sequence_for_path(path: list[str], node_by_alias: dict[str, dict[str, Any]]) -> str:
    return "".join(
        "F" if str(node_by_alias[alias].get("profile_mode")) == "fast" else "D"
        for alias in path
    )


def render_svg(spec: dict[str, Any], validation: dict[str, Any], variant_key: str) -> str:
    nodes = all_nodes(spec)
    node_by_alias = {str(node["alias"]): node for node in nodes}
    edges = spec["edges_by_node_alias"]
    positions = compute_positions(spec)
    paths = enumerate_paths(edges)
    leaf_index_by_alias = {path[-1]: idx for idx, path in enumerate(paths)}
    leaf_path_by_alias = {path[-1]: path for path in paths}
    width = 1240.0
    height = max(320.0, TOP_Y + (len(paths) - 1) * LEAF_GAP + 74.0)
    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
            f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}">'
        ),
        "<style>",
        "  .bg { fill: #fbfbf8; }",
        "  .title { font: 700 22px Arial, sans-serif; fill: #1f2933; }",
        "  .subtitle { font: 13px Arial, sans-serif; fill: #52616b; }",
        "  .stage { font: 700 12px Arial, sans-serif; fill: #52616b; text-anchor: middle; }",
        "  .edge { fill: none; stroke: #a7b0ba; stroke-width: 1.4; }",
        "  .node { stroke: #24313f; stroke-width: 1.1; rx: 7; ry: 7; }",
        "  .g0.fast { fill: #d7f0ec; }",
        "  .g0.deep { fill: #dff3e6; }",
        "  .g1.fast { fill: #ffe0dc; }",
        "  .g1.deep { fill: #eadcff; }",
        "  .root { fill: #e8eef7; }",
        "  .node-title { font: 700 10.3px Arial, sans-serif; fill: #1f2933; }",
        "  .node-meta { font: 10px Arial, sans-serif; fill: #3d4a57; }",
        "  .leaf { font: 700 9.5px Arial, sans-serif; fill: #111827; text-anchor: middle; }",
        "  .seq { font: 9.5px Arial, sans-serif; fill: #52616b; text-anchor: middle; }",
        "  .legend { font: 12px Arial, sans-serif; fill: #374151; }",
        "</style>",
        f'<rect class="bg" width="{width:.0f}" height="{height:.0f}" />',
        (
            f'<text class="title" x="34" y="34">trap_asym_v6_small30_{escape(variant_key)} '
            "mixed fast/deep tree</text>"
        ),
        (
            f'<text class="subtitle" x="34" y="56">30 leaves; '
            f"share leaves={validation['leaf_share_count']}/30; "
            f"share internal={validation['internal_share_count']}/30; "
            f"share-closed={str(not validation['invalid_share_subtree_roots']).lower()}; "
            f"unique FD sequences={validation['mode_sequence_unique_count']}</text>"
        ),
    ]
    legend = (
        ("#dff3e6", "g=0 deep"),
        ("#d7f0ec", "g=0 fast"),
        ("#ffe0dc", "g=1 fast"),
        ("#eadcff", "g=1 deep"),
    )
    for idx, (fill, label) in enumerate(legend):
        x = 34 + idx * 98
        lines.append(
            f'<rect x="{x}" y="68" width="14" height="14" rx="3" fill="{fill}" stroke="#24313f" />'
        )
        lines.append(f'<text class="legend" x="{x + 20}" y="80">{escape(label)}</text>')
    lines.append('<text class="legend" x="450" y="80">Stage5 nodes show FD sequence above the node label</text>')

    for stage, x in STAGE_X.items():
        if stage != "ROOT":
            lines.append(f'<text class="stage" x="{x:.0f}" y="94">{escape(stage)}</text>')

    root_x, root_y_center = positions["ROOT"]
    root_y = root_y_center - NODE_H / 2
    lines.append(
        f'<rect class="node root" x="{root_x}" y="{root_y:.1f}" width="52" height="{NODE_H}" />'
    )
    lines.append(f'<text class="node-title" x="{root_x + 13}" y="{root_y + 21:.1f}">ROOT</text>')

    for parent, children in edges.items():
        if parent != "ROOT" and parent not in node_by_alias:
            continue
        px, py = positions[parent]
        sx = px + (52.0 if parent == "ROOT" else NODE_W)
        for child in children:
            cx, cy = positions[child]
            lines.append(
                f'<path class="edge" d="M {sx:.1f} {py:.1f} C {sx + 18:.1f} {py:.1f}, '
                f'{cx - 18:.1f} {cy:.1f}, {cx:.1f} {cy:.1f}" />'
            )

    for node in nodes:
        alias = str(node["alias"])
        x, y_center = positions[alias]
        y = y_center - NODE_H / 2
        g = int(node["g"])
        mode = str(node.get("profile_mode", "deep"))
        css = f"g{g} {'fast' if mode == 'fast' else 'deep'}"
        label = str(node.get("conceptual_alias", alias))
        short_label = label[:23] + "..." if len(label) > 26 else label
        lines.append(
            f'<rect class="node {css}" x="{x}" y="{y:.1f}" width="{NODE_W}" height="{NODE_H}" />'
        )
        lines.append(
            f'<text class="node-title" x="{x + 7:.1f}" y="{y + 13:.1f}">{escape(short_label)}</text>'
        )
        lines.append(
            f'<text class="node-meta" x="{x + 7:.1f}" y="{y + 27:.1f}">'
            f"g={g} | {escape(mode)} | {escape(str(node.get('base_alias')))}</text>"
        )
        if node["stage"] == "stage5":
            path = leaf_path_by_alias[alias]
            seq = mode_sequence_for_path(path, node_by_alias)
            lines.append(f'<text class="seq" x="{x + NODE_W / 2:.1f}" y="{y - 5:.1f}">{escape(seq)}</text>')
            lines.append(
                f'<text class="leaf" x="{x + NODE_W + 18:.1f}" y="{y + 21:.1f}">'
                f"L{leaf_index_by_alias[alias]:02d}</text>"
            )

    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def output_paths(tree_name: str) -> tuple[Path, Path, Path]:
    return (
        OUTPUT_DIR / f"{tree_name}.json",
        OUTPUT_DIR / f"{tree_name}_validation.json",
        OUTPUT_DIR / f"{tree_name}_tree.svg",
    )


def main() -> None:
    source = read_json(SOURCE_SPEC)
    outputs: dict[str, Any] = {}
    for variant_key, variant in VARIANTS.items():
        spec = build_variant(source, variant_key)
        validation = validate_variant(spec, source=source, variant_key=variant_key)
        spec_path, validation_path, svg_path = output_paths(str(variant["tree_name"]))
        if variant_key != SOURCE_VARIANT_KEY:
            write_json(spec_path, spec)
            write_json(validation_path, validation)
        svg_path.write_text(render_svg(spec, validation, variant_key), encoding="utf-8")
        outputs[variant_key] = {
            "spec": str(spec_path.relative_to(ROOT)),
            "validation": str(validation_path.relative_to(ROOT)),
            "svg": str(svg_path.relative_to(ROOT)),
            "family_kind": variant["family_kind"],
            "g_counts_by_stage": validation["g_counts_by_stage"],
            "internal_share_count": validation["internal_share_count"],
            "leaf_share_count": validation["leaf_share_count"],
            "share_path_count": validation["share_path_count"],
            "validation_errors": validation["validation_errors"],
        }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
