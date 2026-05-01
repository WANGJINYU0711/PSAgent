from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from build_profile_switch_trap_asym_tree import BASE_EDGES, ROOT, SOURCE_SPEC, read_json, write_json
from build_profile_switch_trap_asym_v2_neutral_tree import (
    classify_base_path,
    classify_prefix,
    descendants_inclusive,
    enumerate_alias_paths,
    path_base_aliases,
    weighted_sample_without_replacement,
)


TARGET_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json"
)
VALIDATION_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5_validation.json"
)

TREE_NAME = "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"
FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"

G_LAYOUT_SEED = 20260501
LEAF_SHARE_NUMERATOR = 4
LEAF_SHARE_DENOMINATOR = 5
INTERNAL_SHARE_NUMERATOR = 4
INTERNAL_SHARE_DENOMINATOR = 5
STAGE1_ANCHOR_BASE_ALIAS = "stage1_n1"
ROOT_PRIVATE_BASE_ALIAS = "stage1_n2"
TARGET_PRIVATE_LEAF_TARGET = 25


def expand_spec(source_spec: dict[str, Any]) -> dict[str, Any]:
    stages = list(source_spec["stages"])
    base_node_by_alias: dict[str, dict[str, Any]] = {}
    for stage_nodes in source_spec["nodes"].values():
        for node in stage_nodes:
            base_alias = str(node["base_alias"])
            base_node_by_alias.setdefault(
                base_alias,
                {
                    "stage": str(node["stage"]),
                    "g": int(node["g"]),
                    "base_alias": base_alias,
                },
            )

    expanded_nodes: dict[str, list[dict[str, Any]]] = {stage: [] for stage in stages}
    expanded_edges: dict[str, list[str]] = {"ROOT": []}
    base_alias_by_expanded_alias: dict[str, str] = {"ROOT": "ROOT"}
    serial_by_expanded_alias: dict[str, str] = {"ROOT": "root"}

    frontier = ["ROOT"]
    next_serial = 1
    for stage in stages:
        next_frontier: list[str] = []
        for parent_alias in frontier:
            parent_base_alias = base_alias_by_expanded_alias[parent_alias]
            parent_serial = serial_by_expanded_alias[parent_alias]
            expanded_edges[parent_alias] = []
            for local_idx, base_child_alias in enumerate(BASE_EDGES.get(parent_base_alias, []), start=1):
                base_node = base_node_by_alias[base_child_alias]
                if str(base_node["stage"]) != stage:
                    raise ValueError(
                        f"Base child {base_child_alias} belongs to {base_node['stage']}, expected {stage}."
                    )
                node_serial = f"n{next_serial:04d}"
                next_serial += 1
                alias = f"{base_child_alias}__from__{parent_serial}__c{local_idx:02d}"
                node = {
                    "alias": alias,
                    "display_alias": f"{base_child_alias} <- {parent_serial}",
                    "agent_id": alias,
                    "stage": stage,
                    "g": int(base_node["g"]),
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
                next_frontier.append(alias)
        frontier = next_frontier

    g_layout_metadata = apply_efficient_anchor_4of5_g_layout(expanded_nodes, expanded_edges)

    metadata = dict(source_spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    metadata.update(
        {
            "source_tree_name": source_spec.get("tree_name"),
            "profile_switch_variant": "trap_asym_v3_efficient_anchor_4of5",
            "profile_switch_version": metadata.get("profile_switch_version", "profile_switch_v1"),
            "compatible_with": sorted(compatible_with),
            "not_directly_compatible_with_current_shared_basin_llm_runner": False,
            "preserve_g": False,
            "g_layout_policy": "stage1_whole_subtree_anchor_closure_aware_4of5_v1",
            "g_layout_seed": G_LAYOUT_SEED,
            "g_layout_metadata": g_layout_metadata,
            "base_cost_policy": "unchanged_from_profile_switch_preset",
            "purpose": (
                "Paper-facing efficient-anchor 4/5-share trap-asymmetric profile-switch topology: "
                "same trap_asym path topology, with a single stage1_n1 whole-subtree shared workflow anchor "
                "and closure-aware private placement outside that anchor."
            ),
            "notes": [
                "Topology only; profile capabilities, deliberation modes, and base_cost ranges are unchanged.",
                "Exactly 4/5 of expanded internal nodes and exactly 4/5 of expanded leaf nodes are g=0.",
                "The concrete stage1_n1 subtree is forced all-share, so Risky-PS can use shared mass inside that workflow lane.",
                "Exactly one root child is private: stage1_n2.",
                "Target leaves are not fully clean: 25 target_shared leaves are private.",
                "Non-anchor late-trap leaves are privatized first. Because stage1_n1 itself contains 13 late-trap leaves, global late_trap_shared_rate <= 30% is impossible under this stage1-anchor constraint; the generated layout attains the minimum feasible global late-trap share rate.",
            ],
        }
    )
    return {
        "tree_name": TREE_NAME,
        "depth": 5,
        "stages": stages,
        "metadata": metadata,
        "nodes": expanded_nodes,
        "edges_by_node_alias": expanded_edges,
    }


def apply_efficient_anchor_4of5_g_layout(
    expanded_nodes: dict[str, list[dict[str, Any]]],
    expanded_edges: dict[str, list[str]],
) -> dict[str, Any]:
    rng = random.Random(G_LAYOUT_SEED)
    all_nodes = [node for stage_nodes in expanded_nodes.values() for node in stage_nodes]
    node_by_alias = {str(node["alias"]): node for node in all_nodes}
    parent_by_alias = {
        child: parent
        for parent, children in expanded_edges.items()
        for child in children
    }
    paths = enumerate_alias_paths(expanded_edges)
    base_path_by_leaf = {
        path[-1]: [str(node_by_alias[item]["base_alias"]) for item in path]
        for path in paths
    }
    leaf_archetype_by_alias = {
        leaf: classify_base_path(base_path)
        for leaf, base_path in base_path_by_leaf.items()
    }
    late_trap_leaf_aliases = {
        leaf
        for leaf, base_path in base_path_by_leaf.items()
        if is_late_trap_base_path(base_path)
    }

    for node in all_nodes:
        node["g"] = 0

    stage1_anchor_roots = [
        str(node["alias"])
        for node in expanded_nodes["stage1"]
        if str(node["base_alias"]) == STAGE1_ANCHOR_BASE_ALIAS
    ]
    if len(stage1_anchor_roots) != 1:
        raise ValueError(f"Expected one {STAGE1_ANCHOR_BASE_ALIAS} root, found {stage1_anchor_roots}.")
    protected_stage1_anchor_aliases = descendants_inclusive(
        stage1_anchor_roots[0],
        expanded_edges,
        node_by_alias,
    )

    root_private_aliases = [
        str(node["alias"])
        for node in expanded_nodes["stage1"]
        if str(node["base_alias"]) == ROOT_PRIVATE_BASE_ALIAS
    ]
    if len(root_private_aliases) != 1:
        raise ValueError(f"Expected one {ROOT_PRIVATE_BASE_ALIAS} root, found {root_private_aliases}.")
    root_private_alias = root_private_aliases[0]

    internal_nodes = [node for node in all_nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in all_nodes if node["stage"] == "stage5"]
    internal_private_target = len(internal_nodes) - (
        len(internal_nodes) * INTERNAL_SHARE_NUMERATOR // INTERNAL_SHARE_DENOMINATOR
    )
    leaf_private_target = len(leaf_nodes) - (
        len(leaf_nodes) * LEAF_SHARE_NUMERATOR // LEAF_SHARE_DENOMINATOR
    )

    protected_leaf_aliases = {
        str(node["alias"])
        for node in leaf_nodes
        if str(node["alias"]) in protected_stage1_anchor_aliases
    }
    leaf_by_alias = {str(node["alias"]): node for node in leaf_nodes}
    target_leaf_candidates = [
        leaf_by_alias[leaf]
        for leaf, archetype in leaf_archetype_by_alias.items()
        if archetype == "target_shared" and leaf not in protected_leaf_aliases
    ]
    private_target_leaves = weighted_sample_without_replacement(
        target_leaf_candidates,
        k=TARGET_PRIVATE_LEAF_TARGET,
        weight_fn=lambda node: target_private_leaf_weight(
            base_path_by_leaf[str(node["alias"])],
        ),
        rng=rng,
    )
    private_leaf_aliases = {str(node["alias"]) for node in private_target_leaves}

    non_anchor_late_leaf_aliases = sorted(late_trap_leaf_aliases - protected_leaf_aliases)
    remaining_leaf_private = leaf_private_target - len(private_leaf_aliases)
    for leaf in non_anchor_late_leaf_aliases:
        if remaining_leaf_private <= 0:
            break
        private_leaf_aliases.add(leaf)
        remaining_leaf_private -= 1

    if remaining_leaf_private > 0:
        residual_candidates = [
            node
            for node in leaf_nodes
            if str(node["alias"]) not in protected_leaf_aliases
            and str(node["alias"]) not in private_leaf_aliases
        ]
        sampled_residual = weighted_sample_without_replacement(
            residual_candidates,
            k=remaining_leaf_private,
            weight_fn=lambda node: residual_leaf_private_weight(
                base_path_by_leaf[str(node["alias"])],
                leaf_archetype_by_alias[str(node["alias"])],
            ),
            rng=rng,
        )
        private_leaf_aliases.update(str(node["alias"]) for node in sampled_residual)

    if len(private_leaf_aliases) != leaf_private_target:
        raise ValueError(
            f"Expected {leaf_private_target} private leaves, got {len(private_leaf_aliases)}."
        )

    for alias in private_leaf_aliases:
        node_by_alias[alias]["g"] = 1

    forced_private_internal = {root_private_alias}
    blocked_internal_aliases = protected_stage1_anchor_aliases | {
        str(node["alias"]) for node in expanded_nodes["stage1"]
    }
    internal_candidates = [
        node
        for node in internal_nodes
        if str(node["alias"]) not in blocked_internal_aliases
    ]
    sampled_internal = weighted_sample_without_replacement(
        internal_candidates,
        k=internal_private_target - len(forced_private_internal),
        weight_fn=lambda node: internal_private_weight(
            node,
            node_by_alias,
            parent_by_alias,
            private_leaf_aliases,
            expanded_edges,
        ),
        rng=rng,
    )
    private_internal_aliases = forced_private_internal | {str(node["alias"]) for node in sampled_internal}
    for alias in private_internal_aliases:
        node_by_alias[alias]["g"] = 1

    return {
        "seed": G_LAYOUT_SEED,
        "stage1_anchor_base_alias": STAGE1_ANCHOR_BASE_ALIAS,
        "protected_stage1_anchor_root_alias": stage1_anchor_roots[0],
        "protected_stage1_anchor_alias_count": len(protected_stage1_anchor_aliases),
        "protected_stage1_anchor_leaf_count": len(protected_leaf_aliases),
        "root_private_alias": root_private_alias,
        "root_private_base_alias": node_by_alias[root_private_alias]["base_alias"],
        "internal_private_target": internal_private_target,
        "leaf_private_target": leaf_private_target,
        "target_private_leaf_target": TARGET_PRIVATE_LEAF_TARGET,
        "late_trap_leaf_count": len(late_trap_leaf_aliases),
        "forced_stage1_anchor_late_trap_share_count": len(
            late_trap_leaf_aliases & protected_leaf_aliases
        ),
        "minimum_feasible_late_trap_shared_rate": (
            len(late_trap_leaf_aliases & protected_leaf_aliases)
            / max(1, len(late_trap_leaf_aliases))
        ),
        "sampling_policy": {
            "leaf": "target-private quota first, then non-anchor late-trap leaves, then weighted residual",
            "internal": "root barrier first, then weighted toward prefixes covering private leaves / late trap leakage",
        },
    }


def is_late_trap_base_path(base_path: list[str]) -> bool:
    return "stage4_n4" in base_path or "stage5_n4" in base_path


def target_private_leaf_weight(base_path: list[str]) -> float:
    if base_path[0] == STAGE1_ANCHOR_BASE_ALIAS:
        return 0.0
    # Keep exact target endings a little cleaner than broad target/general endings.
    if base_path[3] == "stage4_n3" and base_path[4] in {"stage5_n1", "stage5_n2", "stage5_n3"}:
        return 0.65
    return 1.0


def residual_leaf_private_weight(base_path: list[str], archetype: str) -> float:
    if base_path[0] == STAGE1_ANCHOR_BASE_ALIAS:
        return 0.0
    if is_late_trap_base_path(base_path):
        return 3.0
    if archetype == "decoy":
        return 1.0
    if archetype == "trap_root":
        return 0.8
    return 0.35


def internal_private_weight(
    node: dict[str, Any],
    node_by_alias: dict[str, dict[str, Any]],
    parent_by_alias: dict[str, str],
    private_leaf_aliases: set[str],
    expanded_edges: dict[str, list[str]],
) -> float:
    alias = str(node["alias"])
    base_path = path_base_aliases(alias, node_by_alias, parent_by_alias)
    prefix_class = classify_prefix(base_path)
    descendants = descendants_inclusive(alias, expanded_edges, node_by_alias)
    private_leaf_descendants = sum(1 for item in descendants if item in private_leaf_aliases)
    descendant_leaf_base_paths = [
        path_base_aliases(item, node_by_alias, parent_by_alias)
        for item in descendants
        if node_by_alias[item]["stage"] == "stage5"
    ]
    late_descendants = sum(1 for path in descendant_leaf_base_paths if is_late_trap_base_path(path))
    weight = 0.35 + private_leaf_descendants + 0.35 * late_descendants
    if prefix_class == "trap_root":
        weight += 1.25
    elif prefix_class == "decoy":
        weight += 1.0
    elif prefix_class == "target_shared":
        weight += 0.25
    return max(weight, 0.01)


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    stages = list(spec["stages"])
    edges = spec["edges_by_node_alias"]
    nodes = [node for stage in stages for node in spec["nodes"].get(stage, [])]
    node_by_alias = {str(node["alias"]): node for node in nodes}
    paths = enumerate_alias_paths(edges)
    base_paths = [[str(node_by_alias[alias]["base_alias"]) for alias in path] for path in paths]
    internal_nodes = [node for node in nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in nodes if node["stage"] == "stage5"]
    child_counts = [len(children) for children in edges.values() if children]
    leaf_archetype_by_alias = {
        path[-1]: classify_base_path([str(node_by_alias[item]["base_alias"]) for item in path])
        for path in paths
    }

    mode_by_base = {
        "stage1_n1": "f",
        "stage1_n2": "d",
        "stage1_n3": "d",
        "stage1_n4": "f",
        "stage1_n5": "f",
        "stage2_n1": "d",
        "stage2_n2": "f",
        "stage2_n3": "f",
        "stage2_n4": "d",
        "stage2_n5": "d",
        "stage3_n1": "d",
        "stage3_n2": "d",
        "stage3_n3": "d",
        "stage3_n4": "f",
        "stage3_n5": "d",
        "stage4_n1": "d",
        "stage4_n2": "d",
        "stage4_n3": "d",
        "stage4_n4": "f",
        "stage4_n5": "d",
        "stage5_n1": "d",
        "stage5_n2": "d",
        "stage5_n3": "d",
        "stage5_n4": "f",
        "stage5_n5": "d",
    }
    mode_patterns = Counter("".join(mode_by_base[base] for base in path) for path in base_paths)
    root_counts = Counter(path[0] for path in base_paths)
    stage2_root_counts = Counter(path[1] for path in base_paths)
    target_shared_paths = [path for path in base_paths if classify_base_path(path) == "target_shared"]
    trap_root_paths = [path for path in base_paths if classify_base_path(path) == "trap_root"]
    decoy_paths = [path for path in base_paths if classify_base_path(path) == "decoy"]

    stage1_full_share_roots = []
    for node in spec["nodes"].get("stage1", []):
        alias = str(node["alias"])
        descendants = descendants_inclusive(alias, edges, node_by_alias)
        if int(node["g"]) == 0 and all(int(node_by_alias[item]["g"]) == 0 for item in descendants):
            stage1_full_share_roots.append(
                {
                    "alias": alias,
                    "base_alias": str(node["base_alias"]),
                    "descendant_internal_count": sum(
                        1 for item in descendants if node_by_alias[item]["stage"] != "stage5"
                    ),
                    "descendant_leaf_count": sum(
                        1 for item in descendants if node_by_alias[item]["stage"] == "stage5"
                    ),
                }
            )

    stage2_full_share_roots = []
    for node in spec["nodes"].get("stage2", []):
        alias = str(node["alias"])
        descendants = descendants_inclusive(alias, edges, node_by_alias)
        if int(node["g"]) == 0 and all(int(node_by_alias[item]["g"]) == 0 for item in descendants):
            stage2_full_share_roots.append(
                {
                    "alias": alias,
                    "base_alias": str(node["base_alias"]),
                    "parent_base_alias": str(node["parent_base_alias"]),
                    "descendant_internal_count": sum(
                        1 for item in descendants if node_by_alias[item]["stage"] != "stage5"
                    ),
                    "descendant_leaf_count": sum(
                        1 for item in descendants if node_by_alias[item]["stage"] == "stage5"
                    ),
                }
            )

    late_trap_leaf_aliases = {
        path[-1]
        for path, base_path in zip(paths, base_paths)
        if is_late_trap_base_path(base_path)
    }
    late_trap_shared_leaf_count = sum(
        1 for alias in late_trap_leaf_aliases if int(node_by_alias[alias]["g"]) == 0
    )
    target_private_leaf_count = sum(
        1
        for node in leaf_nodes
        if int(node["g"]) == 1
        and leaf_archetype_by_alias[str(node["alias"])] == "target_shared"
    )
    private_leaf_archetypes = Counter(
        leaf_archetype_by_alias[str(node["alias"])]
        for node in leaf_nodes
        if int(node["g"]) == 1
    )
    root_private_nodes = [
        {
            "alias": str(node["alias"]),
            "base_alias": str(node["base_alias"]),
        }
        for node in spec["nodes"].get("stage1", [])
        if int(node["g"]) == 1
    ]

    validation = {
        "tree_name": spec["tree_name"],
        "family_kind": FAMILY_KIND,
        "depth": int(spec["depth"]),
        "num_paths": len(paths),
        "total_agent_ids": len(nodes),
        "per_stage_node_counts": {stage: len(spec["nodes"].get(stage, [])) for stage in stages},
        "g_counts_by_stage": {
            stage: dict(Counter(int(node["g"]) for node in spec["nodes"].get(stage, [])))
            for stage in stages
        },
        "internal_node_count": len(internal_nodes),
        "internal_share_count": sum(1 for node in internal_nodes if int(node["g"]) == 0),
        "internal_private_count": sum(1 for node in internal_nodes if int(node["g"]) == 1),
        "internal_share_fraction": (
            sum(1 for node in internal_nodes if int(node["g"]) == 0) / max(1, len(internal_nodes))
        ),
        "leaf_node_count": len(leaf_nodes),
        "leaf_share_count": sum(1 for node in leaf_nodes if int(node["g"]) == 0),
        "leaf_private_count": sum(1 for node in leaf_nodes if int(node["g"]) == 1),
        "leaf_share_fraction": (
            sum(1 for node in leaf_nodes if int(node["g"]) == 0) / max(1, len(leaf_nodes))
        ),
        "root_private_nodes": root_private_nodes,
        "stage1_full_share_subtree_roots": stage1_full_share_roots,
        "stage2_full_share_subtree_roots": stage2_full_share_roots,
        "target_private_leaf_count": target_private_leaf_count,
        "target_private_leaf_fraction_of_target": target_private_leaf_count / max(1, len(target_shared_paths)),
        "late_trap_leaf_count": len(late_trap_leaf_aliases),
        "late_trap_shared_leaf_count": late_trap_shared_leaf_count,
        "late_trap_private_leaf_count": len(late_trap_leaf_aliases) - late_trap_shared_leaf_count,
        "late_trap_shared_rate": late_trap_shared_leaf_count / max(1, len(late_trap_leaf_aliases)),
        "late_trap_shared_rate_requested_max": 0.30,
        "late_trap_shared_rate_constraint_feasible": False,
        "late_trap_shared_rate_feasibility_note": (
            "With stage1_n1 whole-subtree all-share and unchanged trap_asym topology, "
            "13 of 37 late-trap leaves are forced share, so the global rate cannot be below 0.351351."
        ),
        "private_internal_counts_by_base_alias": dict(
            sorted(Counter(str(node["base_alias"]) for node in internal_nodes if int(node["g"]) == 1).items())
        ),
        "private_leaf_counts_by_base_alias": dict(
            sorted(Counter(str(node["base_alias"]) for node in leaf_nodes if int(node["g"]) == 1).items())
        ),
        "private_leaf_counts_by_archetype": dict(sorted(private_leaf_archetypes.items())),
        "root_branching": len(edges.get("ROOT", [])),
        "min_branching": min(child_counts) if child_counts else 0,
        "max_branching": max(child_counts) if child_counts else 0,
        "mean_branching": statistics.fmean(child_counts) if child_counts else 0.0,
        "root_base_path_counts": dict(sorted(root_counts.items())),
        "stage2_base_path_counts": dict(sorted(stage2_root_counts.items())),
        "mode_pattern_counts": dict(sorted(mode_patterns.items())),
        "target_shared_path_count": len(target_shared_paths),
        "target_shared_path_fraction": len(target_shared_paths) / max(1, len(base_paths)),
        "trap_root_path_count": len(trap_root_paths),
        "trap_root_path_fraction": len(trap_root_paths) / max(1, len(base_paths)),
        "decoy_path_count": len(decoy_paths),
        "decoy_path_fraction": len(decoy_paths) / max(1, len(base_paths)),
        "excluded_base_aliases": ["stage1_n5", "stage2_n5", "stage3_n5", "stage4_n5", "stage5_n5"],
        "base_edge_rules": BASE_EDGES,
        "metadata": spec.get("metadata", {}),
    }
    expected_checks = {
        "internal_share_count_is_104": validation["internal_share_count"] == 104,
        "leaf_share_count_is_200": validation["leaf_share_count"] == 200,
        "one_root_private_barrier": len(root_private_nodes) == 1,
        "root_private_is_stage1_n2": root_private_nodes
        == [{"alias": "stage1_n2__from__root__c02", "base_alias": "stage1_n2"}],
        "exactly_one_stage1_full_share_root": len(stage1_full_share_roots) == 1,
        "stage1_full_share_root_is_stage1_n1": bool(stage1_full_share_roots)
        and stage1_full_share_roots[0]["base_alias"] == STAGE1_ANCHOR_BASE_ALIAS,
        "target_private_leaf_count_in_25_35": 25 <= target_private_leaf_count <= 35,
    }
    validation["expected_checks"] = expected_checks
    validation["validation_errors"] = [
        name for name, passed in expected_checks.items() if not passed
    ]
    return validation


def main() -> None:
    source = read_json(SOURCE_SPEC)
    spec = expand_spec(source)
    validation = validate(spec)
    write_json(TARGET_SPEC, spec)
    write_json(VALIDATION_PATH, validation)
    print(
        json.dumps(
            {
                "target_spec": str(TARGET_SPEC.relative_to(ROOT)),
                "validation": str(VALIDATION_PATH.relative_to(ROOT)),
                "family_kind": FAMILY_KIND,
                "summary": {
                    "num_paths": validation["num_paths"],
                    "total_agent_ids": validation["total_agent_ids"],
                    "per_stage_node_counts": validation["per_stage_node_counts"],
                    "g_counts_by_stage": validation["g_counts_by_stage"],
                    "internal_share_fraction": validation["internal_share_fraction"],
                    "leaf_share_fraction": validation["leaf_share_fraction"],
                    "root_private_nodes": validation["root_private_nodes"],
                    "stage1_full_share_subtree_roots": validation["stage1_full_share_subtree_roots"],
                    "stage2_full_share_subtree_roots": validation["stage2_full_share_subtree_roots"],
                    "target_private_leaf_count": validation["target_private_leaf_count"],
                    "late_trap_shared_rate": validation["late_trap_shared_rate"],
                    "late_trap_shared_rate_feasibility_note": validation[
                        "late_trap_shared_rate_feasibility_note"
                    ],
                    "private_leaf_counts_by_archetype": validation["private_leaf_counts_by_archetype"],
                    "validation_errors": validation["validation_errors"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
