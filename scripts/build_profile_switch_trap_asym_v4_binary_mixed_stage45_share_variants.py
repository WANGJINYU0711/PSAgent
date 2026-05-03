from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from build_profile_switch_trap_asym_v4_binary_mixed_stage45_tree import (
    ROOT,
    TARGET_SPEC as SOURCE_SPEC,
    validate,
    write_json,
)


OUTPUT_DIR = ROOT / "analysis" / "tree_specs"

VARIANTS = {
    "4of5": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_4of5"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_4of5"
        ),
        "profile_switch_variant": "trap_asym_v4_binary_mixed_stage45_4of5",
        "g_layout_policy": (
            "v4_binary_mixed_stage45_4of5_gonly_stage1_deep_basin_plus_fast_spine_v1"
        ),
    },
    "2of5": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_2of5"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_2of5"
        ),
        "profile_switch_variant": "trap_asym_v4_binary_mixed_stage45_2of5",
        "g_layout_policy": (
            "v4_binary_mixed_stage45_2of5_gonly_stage2_deep_subtree_v1"
        ),
    },
    "all_share": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_all_share"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_all_share"
        ),
        "g": 0,
        "profile_switch_variant": "trap_asym_v4_binary_mixed_stage45_all_share",
    },
    "all_unshare": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_all_unshare"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_"
            "trap_asym_v4_binary_mixed_stage45_all_unshare"
        ),
        "g": 1,
        "profile_switch_variant": "trap_asym_v4_binary_mixed_stage45_all_unshare",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def all_nodes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for stage in spec["stages"] for node in spec["nodes"].get(stage, [])]


def descendants_inclusive(root_alias: str, edges: dict[str, list[str]]) -> set[str]:
    descendants: set[str] = set()

    def rec(alias: str) -> None:
        descendants.add(alias)
        for child in edges.get(alias, []):
            rec(child)

    rec(root_alias)
    return descendants


def variant_g_values(source: dict[str, Any], variant_key: str) -> dict[str, int]:
    nodes = all_nodes(source)
    alias_by_conceptual = {
        str(node.get("conceptual_alias")): str(node["alias"])
        for node in nodes
    }
    if variant_key == "all_share":
        return {str(node["alias"]): 0 for node in nodes}
    if variant_key == "all_unshare":
        return {str(node["alias"]): 1 for node in nodes}
    if variant_key == "2of5":
        stage2_deep_root = alias_by_conceptual["s2_n1_deep_shared_A_mixed"]
        shared_aliases = descendants_inclusive(stage2_deep_root, source["edges_by_node_alias"])
        return {
            str(node["alias"]): 0 if str(node["alias"]) in shared_aliases else 1
            for node in nodes
        }
    if variant_key == "4of5":
        stage1_deep_root = alias_by_conceptual["s1_n1_deep_shared_basin_root"]
        shared_aliases = descendants_inclusive(stage1_deep_root, source["edges_by_node_alias"])
        shared_aliases.update(
            alias_by_conceptual[name]
            for name in (
                "s1_n0_fast_trap_root",
                "s2_n0_fast_trap_contract",
                "s3_n0_fast_trap_A",
                "s4_n0_fast_stage4_A",
            )
        )
        return {
            str(node["alias"]): 0 if str(node["alias"]) in shared_aliases else 1
            for node in nodes
        }
    raise KeyError(f"Unsupported v4 share variant: {variant_key}")


def build_variant(source: dict[str, Any], variant_key: str) -> dict[str, Any]:
    variant = VARIANTS[variant_key]
    spec = deepcopy(source)
    spec["tree_name"] = str(variant["tree_name"])
    g_by_alias = variant_g_values(source, variant_key)
    for node in all_nodes(spec):
        node["g"] = int(g_by_alias[str(node["alias"])])

    metadata = dict(spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    metadata.update(
        {
            "source_tree_name": source.get("tree_name"),
            "source_tree_path": str(SOURCE_SPEC.relative_to(ROOT)),
            "profile_switch_variant": variant["profile_switch_variant"],
            "share_variant": variant_key,
            "share_variant_family_kind": variant["family_kind"],
            "compatible_with": sorted(compatible_with),
            "preserve_g": False,
            "g_layout_policy": variant.get(
                "g_layout_policy",
                f"v4_binary_mixed_stage45_{variant_key}_gonly_v1",
            ),
            "purpose": (
                "G-only share control for v4 binary mixed stage45 tree. Topology, "
                "base aliases, conceptual aliases, profile modes, and route semantics "
                "are unchanged from the v4 base tree."
            ),
            "notes": [
                "Only expanded-node g values and top-level identifying metadata are changed.",
                "Use this to isolate PS shared-update behavior from the compact v4 execution topology.",
            ],
        }
    )
    spec["metadata"] = metadata
    return spec


def validate_variant(
    spec: dict[str, Any],
    *,
    source: dict[str, Any],
    variant_key: str,
) -> dict[str, Any]:
    validation = validate(spec)
    expected_g_by_alias = variant_g_values(source, variant_key)
    all_spec_nodes = all_nodes(spec)
    share_count = sum(1 for node in all_spec_nodes if int(node["g"]) == 0)
    unshare_count = sum(1 for node in all_spec_nodes if int(node["g"]) == 1)
    expected_share_count = sum(1 for value in expected_g_by_alias.values() if value == 0)
    expected_unshare_count = sum(1 for value in expected_g_by_alias.values() if value == 1)
    expected_checks = {
        "num_paths_is_8": validation["num_paths"] == 8,
        "all_leaves_at_stage5_depth": validation["leaf_depths"] == [5],
        "root_is_binary": validation["root_branching"] == 2,
        "max_branching_at_most_2": validation["max_branching"] <= 2,
        "only_expected_g_values_changed": all(
            int(node["g"]) == expected_g_by_alias[str(node["alias"])]
            for node in all_spec_nodes
        ),
        "share_count_matches_variant": share_count == expected_share_count,
        "unshare_count_matches_variant": unshare_count == expected_unshare_count,
    }
    validation["expected_checks"] = expected_checks
    validation["validation_errors"] = [
        name for name, ok in expected_checks.items() if not ok
    ]
    validation["share_variant"] = variant_key
    validation["share_node_count"] = share_count
    validation["unshare_node_count"] = unshare_count
    validation["share_node_fraction"] = share_count / max(1, len(all_spec_nodes))
    validation["unshare_node_fraction"] = unshare_count / max(1, len(all_spec_nodes))
    validation["expected_g_by_alias"] = expected_g_by_alias
    return validation


def main() -> None:
    source = read_json(SOURCE_SPEC)
    outputs: dict[str, Any] = {}
    for variant_key, variant in VARIANTS.items():
        spec = build_variant(source, variant_key)
        tree_name = str(variant["tree_name"])
        spec_path = OUTPUT_DIR / f"{tree_name}.json"
        validation_path = OUTPUT_DIR / f"{tree_name}_validation.json"
        validation = validate_variant(
            spec,
            source=source,
            variant_key=variant_key,
        )
        validation["family_kind"] = variant["family_kind"]
        write_json(spec_path, spec)
        write_json(validation_path, validation)
        outputs[variant_key] = {
            "spec": str(spec_path.relative_to(ROOT)),
            "validation": str(validation_path.relative_to(ROOT)),
            "family_kind": variant["family_kind"],
            "g_counts_by_stage": validation["g_counts_by_stage"],
            "share_node_count": validation["share_node_count"],
            "unshare_node_count": validation["unshare_node_count"],
            "validation_errors": validation["validation_errors"],
        }
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
