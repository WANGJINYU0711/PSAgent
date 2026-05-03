from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

from build_profile_switch_trap_asym_v2_neutral_tree import (
    classify_base_path,
    classify_prefix,
    descendants_inclusive,
    enumerate_alias_paths,
    path_base_aliases,
    weighted_sample_without_replacement,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json"
)
OUTPUT_DIR = ROOT / "analysis" / "tree_specs"

G_LAYOUT_SEED = 20260501
STAGE1_ANCHOR_BASE_ALIAS = "stage1_n1"

VARIANTS = {
    "all_share": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_"
            "efficient_anchor_all_share"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_"
            "efficient_anchor_all_share"
        ),
        "target_share_numerator": 1,
        "target_share_denominator": 1,
        "g_layout_policy": "v3_same_topology_all_share_gonly_v1",
    },
    "2of5": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_"
            "efficient_anchor_2of5"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_"
            "efficient_anchor_2of5"
        ),
        "target_share_numerator": 2,
        "target_share_denominator": 5,
        "g_layout_policy": "v3_efficient_anchor_lower_share_2of5_v1",
    },
    "all_unshare": {
        "tree_name": (
            "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_"
            "efficient_anchor_all_unshare"
        ),
        "family_kind": (
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_"
            "efficient_anchor_all_unshare"
        ),
        "target_share_numerator": 0,
        "target_share_denominator": 1,
        "g_layout_policy": "v3_same_topology_all_unshare_gonly_v1",
    },
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def is_late_trap_base_path(base_path: list[str]) -> bool:
    return "stage4_n4" in base_path or "stage5_n4" in base_path


def all_nodes(spec: dict[str, Any]) -> list[dict[str, Any]]:
    return [node for stage in spec["stages"] for node in spec["nodes"].get(stage, [])]


def node_maps(spec: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    node_by_alias = {str(node["alias"]): node for node in all_nodes(spec)}
    parent_by_alias = {
        child: parent
        for parent, children in spec["edges_by_node_alias"].items()
        for child in children
    }
    return node_by_alias, parent_by_alias


def build_variant(source: dict[str, Any], variant_key: str) -> dict[str, Any]:
    variant = VARIANTS[variant_key]
    spec = deepcopy(source)
    spec["tree_name"] = str(variant["tree_name"])

    if variant_key == "all_share":
        for node in all_nodes(spec):
            node["g"] = 0
        g_layout_metadata = {
            "share_assignment": "all expanded nodes are g=0",
            "target_share_numerator": 1,
            "target_share_denominator": 1,
        }
    elif variant_key == "all_unshare":
        for node in all_nodes(spec):
            node["g"] = 1
        g_layout_metadata = {
            "share_assignment": "all expanded nodes are g=1",
            "target_share_numerator": 0,
            "target_share_denominator": 1,
        }
    elif variant_key == "2of5":
        g_layout_metadata = apply_2of5_layout(spec)
    else:
        raise ValueError(f"Unknown variant_key={variant_key!r}")

    metadata = dict(spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    metadata.update(
        {
            "source_tree_name": source.get("tree_name"),
            "source_tree_path": str(SOURCE_SPEC.relative_to(ROOT)),
            "profile_switch_variant": f"trap_asym_v3_efficient_anchor_{variant_key}",
            "profile_switch_version": metadata.get("profile_switch_version", "profile_switch_v1"),
            "share_variant": variant_key,
            "share_variant_family_kind": variant["family_kind"],
            "compatible_with": sorted(compatible_with),
            "not_directly_compatible_with_current_shared_basin_llm_runner": False,
            "preserve_g": False,
            "g_layout_policy": variant["g_layout_policy"],
            "g_layout_seed": G_LAYOUT_SEED,
            "g_layout_metadata": g_layout_metadata,
            "base_cost_policy": "unchanged_from_profile_switch_preset",
            "purpose": (
                "G-only share-ratio control for the paper-facing v3 efficient-anchor "
                "profile-switch topology."
            ),
            "notes": [
                "Topology, expanded aliases, legal continuations, base aliases, profile capabilities, deliberation modes, and base-cost policy are unchanged from the v3 4/5 main tree.",
                "Only expanded-node g values are changed, so this isolates PS shared-update structure from execution topology.",
            ],
        }
    )
    spec["metadata"] = metadata
    return spec


def apply_2of5_layout(spec: dict[str, Any]) -> dict[str, Any]:
    rng = random.Random(G_LAYOUT_SEED)
    nodes = all_nodes(spec)
    edges = spec["edges_by_node_alias"]
    node_by_alias, parent_by_alias = node_maps(spec)

    for node in nodes:
        node["g"] = 1

    internal_nodes = [node for node in nodes if str(node["stage"]) != "stage5"]
    leaf_nodes = [node for node in nodes if str(node["stage"]) == "stage5"]
    internal_share_target = len(internal_nodes) * 2 // 5
    leaf_share_target = len(leaf_nodes) * 2 // 5

    anchor_roots = [
        str(node["alias"])
        for node in spec["nodes"].get("stage1", [])
        if str(node["base_alias"]) == STAGE1_ANCHOR_BASE_ALIAS
    ]
    if len(anchor_roots) != 1:
        raise ValueError(f"Expected one {STAGE1_ANCHOR_BASE_ALIAS} root, found {anchor_roots}.")
    protected_anchor_aliases = descendants_inclusive(anchor_roots[0], edges, node_by_alias)
    for alias in protected_anchor_aliases:
        node_by_alias[alias]["g"] = 0

    protected_internal_count = sum(
        1 for alias in protected_anchor_aliases if str(node_by_alias[alias]["stage"]) != "stage5"
    )
    protected_leaf_count = sum(
        1 for alias in protected_anchor_aliases if str(node_by_alias[alias]["stage"]) == "stage5"
    )
    extra_internal_needed = internal_share_target - protected_internal_count
    extra_leaf_needed = leaf_share_target - protected_leaf_count
    if extra_internal_needed < 0 or extra_leaf_needed < 0:
        raise ValueError(
            "Stage1 anchor is larger than the requested 2/5 share budget: "
            f"internal_extra={extra_internal_needed} leaf_extra={extra_leaf_needed}."
        )

    paths = enumerate_alias_paths(edges)
    base_path_by_leaf = {
        path[-1]: [str(node_by_alias[item]["base_alias"]) for item in path]
        for path in paths
    }
    leaf_archetype_by_alias = {
        leaf: classify_base_path(base_path)
        for leaf, base_path in base_path_by_leaf.items()
    }

    internal_candidates = [
        node
        for node in internal_nodes
        if str(node["alias"]) not in protected_anchor_aliases
        and str(node["stage"]) != "stage1"
    ]
    sampled_internal = weighted_sample_without_replacement(
        internal_candidates,
        k=extra_internal_needed,
        weight_fn=lambda node: internal_share_weight(node, node_by_alias, parent_by_alias),
        rng=rng,
    )
    extra_internal_aliases = {str(node["alias"]) for node in sampled_internal}
    for alias in extra_internal_aliases:
        node_by_alias[alias]["g"] = 0

    leaf_candidates = [
        node
        for node in leaf_nodes
        if str(node["alias"]) not in protected_anchor_aliases
    ]
    sampled_leaves = weighted_sample_without_replacement(
        leaf_candidates,
        k=extra_leaf_needed,
        weight_fn=lambda node: leaf_share_weight(
            base_path_by_leaf[str(node["alias"])],
            leaf_archetype_by_alias[str(node["alias"])],
        ),
        rng=rng,
    )
    extra_leaf_aliases = {str(node["alias"]) for node in sampled_leaves}
    for alias in extra_leaf_aliases:
        node_by_alias[alias]["g"] = 0

    return {
        "seed": G_LAYOUT_SEED,
        "target_share_numerator": 2,
        "target_share_denominator": 5,
        "stage1_anchor_base_alias": STAGE1_ANCHOR_BASE_ALIAS,
        "protected_stage1_anchor_root_alias": anchor_roots[0],
        "protected_stage1_anchor_internal_count": protected_internal_count,
        "protected_stage1_anchor_leaf_count": protected_leaf_count,
        "internal_share_target": internal_share_target,
        "leaf_share_target": leaf_share_target,
        "extra_internal_share_count": len(extra_internal_aliases),
        "extra_leaf_share_count": len(extra_leaf_aliases),
        "extra_share_sampling_policy": {
            "internal": "seeded weighted sample outside stage1 anchor; stage1 non-anchor roots stay private",
            "leaf": "seeded weighted sample outside stage1 anchor",
            "preference": "favor clean target/shared prefixes and downweight trap/decoy leakage",
        },
    }


def internal_share_weight(
    node: dict[str, Any],
    node_by_alias: dict[str, dict[str, Any]],
    parent_by_alias: dict[str, str],
) -> float:
    base_path = path_base_aliases(str(node["alias"]), node_by_alias, parent_by_alias)
    prefix_class = classify_prefix(base_path)
    weight = {
        "target_shared": 3.0,
        "mixed": 1.0,
        "decoy": 0.55,
        "trap_root": 0.20,
    }.get(prefix_class, 1.0)
    if is_late_trap_base_path(base_path):
        weight *= 0.25
    return max(weight, 0.01)


def leaf_share_weight(base_path: list[str], archetype: str) -> float:
    weight = {
        "target_shared": 4.0,
        "decoy": 0.55,
        "trap_root": 0.15,
    }.get(archetype, 1.0)
    if is_late_trap_base_path(base_path):
        weight *= 0.20
    return max(weight, 0.01)


def validate(spec: dict[str, Any], family_kind: str, variant_key: str) -> dict[str, Any]:
    stages = list(spec["stages"])
    edges = spec["edges_by_node_alias"]
    nodes = all_nodes(spec)
    node_by_alias, _ = node_maps(spec)
    paths = enumerate_alias_paths(edges)
    base_paths = [[str(node_by_alias[alias]["base_alias"]) for alias in path] for path in paths]
    internal_nodes = [node for node in nodes if str(node["stage"]) != "stage5"]
    leaf_nodes = [node for node in nodes if str(node["stage"]) == "stage5"]
    child_counts = [len(children) for children in edges.values() if children]
    leaf_archetype_by_alias = {
        path[-1]: classify_base_path([str(node_by_alias[item]["base_alias"]) for item in path])
        for path in paths
    }

    stage1_full_share_roots = full_share_subtree_roots(spec, "stage1", node_by_alias)
    stage2_full_share_roots = full_share_subtree_roots(spec, "stage2", node_by_alias)
    target_shared_paths = [path for path in base_paths if classify_base_path(path) == "target_shared"]
    trap_root_paths = [path for path in base_paths if classify_base_path(path) == "trap_root"]
    decoy_paths = [path for path in base_paths if classify_base_path(path) == "decoy"]
    late_trap_leaf_aliases = {
        path[-1]
        for path, base_path in zip(paths, base_paths)
        if is_late_trap_base_path(base_path)
    }

    internal_share_count = sum(1 for node in internal_nodes if int(node["g"]) == 0)
    leaf_share_count = sum(1 for node in leaf_nodes if int(node["g"]) == 0)
    target_share_num = int(VARIANTS[variant_key]["target_share_numerator"])
    target_share_den = int(VARIANTS[variant_key]["target_share_denominator"])
    expected_internal_share = len(internal_nodes) * target_share_num // target_share_den
    expected_leaf_share = len(leaf_nodes) * target_share_num // target_share_den

    validation = {
        "tree_name": spec["tree_name"],
        "family_kind": family_kind,
        "share_variant": variant_key,
        "depth": int(spec["depth"]),
        "num_paths": len(paths),
        "total_agent_ids": len(nodes),
        "per_stage_node_counts": {stage: len(spec["nodes"].get(stage, [])) for stage in stages},
        "g_counts_by_stage": {
            stage: dict(Counter(int(node["g"]) for node in spec["nodes"].get(stage, [])))
            for stage in stages
        },
        "internal_node_count": len(internal_nodes),
        "internal_share_count": internal_share_count,
        "internal_private_count": len(internal_nodes) - internal_share_count,
        "internal_share_fraction": internal_share_count / max(1, len(internal_nodes)),
        "leaf_node_count": len(leaf_nodes),
        "leaf_share_count": leaf_share_count,
        "leaf_private_count": len(leaf_nodes) - leaf_share_count,
        "leaf_share_fraction": leaf_share_count / max(1, len(leaf_nodes)),
        "expected_internal_share_count": expected_internal_share,
        "expected_leaf_share_count": expected_leaf_share,
        "stage1_full_share_subtree_roots": stage1_full_share_roots,
        "stage2_full_share_subtree_roots": stage2_full_share_roots,
        "late_trap_leaf_count": len(late_trap_leaf_aliases),
        "late_trap_shared_leaf_count": sum(
            1 for alias in late_trap_leaf_aliases if int(node_by_alias[alias]["g"]) == 0
        ),
        "shared_leaf_counts_by_archetype": dict(
            sorted(
                Counter(
                    leaf_archetype_by_alias[str(node["alias"])]
                    for node in leaf_nodes
                    if int(node["g"]) == 0
                ).items()
            )
        ),
        "private_leaf_counts_by_archetype": dict(
            sorted(
                Counter(
                    leaf_archetype_by_alias[str(node["alias"])]
                    for node in leaf_nodes
                    if int(node["g"]) == 1
                ).items()
            )
        ),
        "target_shared_path_count": len(target_shared_paths),
        "target_shared_path_fraction": len(target_shared_paths) / max(1, len(base_paths)),
        "trap_root_path_count": len(trap_root_paths),
        "trap_root_path_fraction": len(trap_root_paths) / max(1, len(base_paths)),
        "decoy_path_count": len(decoy_paths),
        "decoy_path_fraction": len(decoy_paths) / max(1, len(base_paths)),
        "root_branching": len(edges.get("ROOT", [])),
        "min_branching": min(child_counts) if child_counts else 0,
        "max_branching": max(child_counts) if child_counts else 0,
        "mean_branching": statistics.fmean(child_counts) if child_counts else 0.0,
        "metadata": spec.get("metadata", {}),
    }
    expected_checks = {
        "topology_matches_v3_num_paths_250": validation["num_paths"] == 250,
        "topology_matches_v3_total_agent_ids_380": validation["total_agent_ids"] == 380,
        "internal_share_count_matches_variant": internal_share_count == expected_internal_share,
        "leaf_share_count_matches_variant": leaf_share_count == expected_leaf_share,
    }
    if variant_key == "2of5":
        expected_checks["stage1_anchor_is_full_share"] = bool(stage1_full_share_roots) and any(
            item["base_alias"] == STAGE1_ANCHOR_BASE_ALIAS for item in stage1_full_share_roots
        )
    validation["expected_checks"] = expected_checks
    validation["validation_errors"] = [
        name for name, passed in expected_checks.items() if not passed
    ]
    return validation


def full_share_subtree_roots(
    spec: dict[str, Any],
    stage: str,
    node_by_alias: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    roots = []
    edges = spec["edges_by_node_alias"]
    for node in spec["nodes"].get(stage, []):
        alias = str(node["alias"])
        descendants = descendants_inclusive(alias, edges, node_by_alias)
        if int(node["g"]) == 0 and all(int(node_by_alias[item]["g"]) == 0 for item in descendants):
            roots.append(
                {
                    "alias": alias,
                    "base_alias": str(node["base_alias"]),
                    "descendant_internal_count": sum(
                        1 for item in descendants if str(node_by_alias[item]["stage"]) != "stage5"
                    ),
                    "descendant_leaf_count": sum(
                        1 for item in descendants if str(node_by_alias[item]["stage"]) == "stage5"
                    ),
                }
            )
    return roots


def main() -> None:
    source = read_json(SOURCE_SPEC)
    summaries = {}
    for variant_key, variant in VARIANTS.items():
        spec = build_variant(source, variant_key)
        family_kind = str(variant["family_kind"])
        validation = validate(spec, family_kind, variant_key)
        spec_path = OUTPUT_DIR / f"{spec['tree_name']}.json"
        validation_path = OUTPUT_DIR / f"{spec['tree_name']}_validation.json"
        write_json(spec_path, spec)
        write_json(validation_path, validation)
        summaries[variant_key] = {
            "tree_name": spec["tree_name"],
            "family_kind": family_kind,
            "spec": str(spec_path.relative_to(ROOT)),
            "validation": str(validation_path.relative_to(ROOT)),
            "internal_share_fraction": validation["internal_share_fraction"],
            "leaf_share_fraction": validation["leaf_share_fraction"],
            "g_counts_by_stage": validation["g_counts_by_stage"],
            "validation_errors": validation["validation_errors"],
        }
    print(json.dumps(summaries, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
