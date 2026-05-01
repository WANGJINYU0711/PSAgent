from __future__ import annotations

import json
import random
import statistics
from collections import Counter
from pathlib import Path
from typing import Any

from build_profile_switch_trap_asym_tree import BASE_EDGES, ROOT, SOURCE_SPEC, read_json, write_json


TARGET_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5.json"
)
VALIDATION_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5_validation.json"
)

TREE_NAME = "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5"
FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5"

G_LAYOUT_SEED = 20260501
LEAF_SHARE_NUMERATOR = 4
LEAF_SHARE_DENOMINATOR = 5
INTERNAL_SHARE_NUMERATOR = 4
INTERNAL_SHARE_DENOMINATOR = 5

ANCHOR_ROOT_BASE_ALIAS = "stage1_n1"
ANCHOR_STAGE2_BASE_ALIAS = "stage2_n1"
ROOT_PRIVATE_CANDIDATE_WEIGHTS = {
    "stage1_n2": 1.0,
    "stage1_n3": 1.0,
    "stage1_n4": 0.35,
}


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

    g_layout_metadata = apply_seeded_neutral_4of5_g_layout(expanded_nodes, expanded_edges)

    metadata = dict(source_spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    metadata.update(
        {
            "source_tree_name": source_spec.get("tree_name"),
            "profile_switch_variant": "trap_asym_v2_neutral_4of5",
            "profile_switch_version": metadata.get("profile_switch_version", "profile_switch_v1"),
            "compatible_with": sorted(compatible_with),
            "not_directly_compatible_with_current_shared_basin_llm_runner": False,
            "preserve_g": False,
            "g_layout_policy": "seeded_weighted_sampling_with_minimal_anchor_constraints_v1",
            "g_layout_seed": G_LAYOUT_SEED,
            "g_layout_metadata": g_layout_metadata,
            "base_cost_policy": "unchanged_from_profile_switch_preset",
            "purpose": (
                "Paper-facing neutral 4/5-share trap-asymmetric profile-switch topology: "
                "same path topology as trap_asym_v1, but g is assigned by seeded weighted sampling "
                "rather than by protecting the entire target basin."
            ),
            "notes": [
                "Topology only; profile capabilities, deliberation modes, and base_cost ranges are unchanged.",
                "Exactly 4/5 of expanded internal nodes and exactly 4/5 of expanded leaf nodes are g=0.",
                "One concrete stage2_n1 subtree under stage1_n1 is forced full-share to satisfy the stage2 shared-subtree requirement.",
                "One root child outside stage1_n1 is sampled as g=1 with trap-root downweighted, so root has a barrier without making the trap branch the default private mass sink.",
                "All remaining private nodes are selected by seeded weighted sampling over target, decoy, and trap candidates; target paths may receive private leaves.",
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


def apply_seeded_neutral_4of5_g_layout(
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
    leaf_archetype_by_alias = {
        path[-1]: classify_base_path([str(node_by_alias[item]["base_alias"]) for item in path])
        for path in paths
    }

    for node in all_nodes:
        node["g"] = 0

    protected_full_share_aliases = protected_anchor_subtree_aliases(
        expanded_nodes=expanded_nodes,
        expanded_edges=expanded_edges,
        node_by_alias=node_by_alias,
    )
    root_private_alias = sample_root_private_alias(
        root_nodes=expanded_nodes["stage1"],
        protected_anchor_root_base_alias=ANCHOR_ROOT_BASE_ALIAS,
        rng=rng,
    )

    internal_nodes = [node for node in all_nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in all_nodes if node["stage"] == "stage5"]
    internal_private_target = len(internal_nodes) - (
        len(internal_nodes) * INTERNAL_SHARE_NUMERATOR // INTERNAL_SHARE_DENOMINATOR
    )
    leaf_private_target = len(leaf_nodes) - (
        len(leaf_nodes) * LEAF_SHARE_NUMERATOR // LEAF_SHARE_DENOMINATOR
    )

    forced_private_internal = {root_private_alias}
    blocked_aliases = protected_full_share_aliases | {str(node["alias"]) for node in expanded_nodes["stage1"]}
    internal_candidates = [
        node
        for node in internal_nodes
        if str(node["alias"]) not in blocked_aliases
    ]
    sampled_internal = weighted_sample_without_replacement(
        internal_candidates,
        k=internal_private_target - len(forced_private_internal),
        weight_fn=lambda node: internal_private_weight(node, node_by_alias, parent_by_alias),
        rng=rng,
    )
    private_internal_aliases = forced_private_internal | {str(node["alias"]) for node in sampled_internal}
    for alias in private_internal_aliases:
        node_by_alias[alias]["g"] = 1

    leaf_candidates = [
        node
        for node in leaf_nodes
        if str(node["alias"]) not in protected_full_share_aliases
    ]
    sampled_leaves = weighted_sample_without_replacement(
        leaf_candidates,
        k=leaf_private_target,
        weight_fn=lambda node: leaf_private_weight(leaf_archetype_by_alias[str(node["alias"])]),
        rng=rng,
    )
    private_leaf_aliases = {str(node["alias"]) for node in sampled_leaves}
    for alias in private_leaf_aliases:
        node_by_alias[alias]["g"] = 1

    return {
        "seed": G_LAYOUT_SEED,
        "anchor_root_base_alias": ANCHOR_ROOT_BASE_ALIAS,
        "anchor_stage2_base_alias": ANCHOR_STAGE2_BASE_ALIAS,
        "protected_anchor_subtree_alias_count": len(protected_full_share_aliases),
        "root_private_alias": root_private_alias,
        "root_private_base_alias": node_by_alias[root_private_alias]["base_alias"],
        "internal_private_target": internal_private_target,
        "leaf_private_target": leaf_private_target,
        "sampling_weights": {
            "root_private_candidate_weights": ROOT_PRIVATE_CANDIDATE_WEIGHTS,
            "internal": {
                "target_shared": 0.85,
                "decoy": 1.00,
                "trap_root": 0.35,
                "mixed": 0.80,
            },
            "leaf": {
                "target_shared": 0.80,
                "decoy": 1.00,
                "trap_root": 0.30,
            },
        },
    }


def protected_anchor_subtree_aliases(
    *,
    expanded_nodes: dict[str, list[dict[str, Any]]],
    expanded_edges: dict[str, list[str]],
    node_by_alias: dict[str, dict[str, Any]],
) -> set[str]:
    anchor_stage1_aliases = {
        str(node["alias"])
        for node in expanded_nodes["stage1"]
        if str(node["base_alias"]) == ANCHOR_ROOT_BASE_ALIAS
    }
    anchor_stage2_roots = [
        str(node["alias"])
        for node in expanded_nodes["stage2"]
        if str(node["base_alias"]) == ANCHOR_STAGE2_BASE_ALIAS
        and str(node["parent_alias"]) in anchor_stage1_aliases
    ]
    if len(anchor_stage2_roots) != 1:
        raise ValueError(
            f"Expected exactly one {ANCHOR_STAGE2_BASE_ALIAS} child under {ANCHOR_ROOT_BASE_ALIAS}, "
            f"found {anchor_stage2_roots}."
        )
    return descendants_inclusive(anchor_stage2_roots[0], expanded_edges, node_by_alias)


def sample_root_private_alias(
    *,
    root_nodes: list[dict[str, Any]],
    protected_anchor_root_base_alias: str,
    rng: random.Random,
) -> str:
    candidates = [
        node
        for node in root_nodes
        if str(node["base_alias"]) != protected_anchor_root_base_alias
    ]
    weights = [
        ROOT_PRIVATE_CANDIDATE_WEIGHTS.get(str(node["base_alias"]), 1.0)
        for node in candidates
    ]
    return str(rng.choices(candidates, weights=weights, k=1)[0]["alias"])


def weighted_sample_without_replacement(
    candidates: list[dict[str, Any]],
    *,
    k: int,
    weight_fn: Any,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if k < 0:
        raise ValueError(f"Cannot sample negative k={k}.")
    if k > len(candidates):
        raise ValueError(f"Cannot sample k={k} from {len(candidates)} candidates.")
    keyed: list[tuple[float, dict[str, Any]]] = []
    for node in candidates:
        weight = float(weight_fn(node))
        if weight <= 0:
            continue
        # Efraimidis-Spirakis weighted sampling without replacement.
        key = rng.random() ** (1.0 / weight)
        keyed.append((key, node))
    if k > len(keyed):
        raise ValueError(f"Only {len(keyed)} positive-weight candidates for k={k}.")
    keyed.sort(key=lambda item: item[0], reverse=True)
    return [node for _, node in keyed[:k]]


def internal_private_weight(
    node: dict[str, Any],
    node_by_alias: dict[str, dict[str, Any]],
    parent_by_alias: dict[str, str],
) -> float:
    base_path = path_base_aliases(str(node["alias"]), node_by_alias, parent_by_alias)
    prefix_class = classify_prefix(base_path)
    if prefix_class == "trap_root":
        return 0.35
    if prefix_class == "decoy":
        return 1.00
    if prefix_class == "target_shared":
        return 0.85
    return 0.80


def leaf_private_weight(archetype: str) -> float:
    if archetype == "trap_root":
        return 0.30
    if archetype == "decoy":
        return 1.00
    if archetype == "target_shared":
        return 0.80
    return 0.80


def enumerate_alias_paths(edges: dict[str, list[str]]) -> list[list[str]]:
    paths: list[list[str]] = []

    def rec(parent: str, depth: int, prefix: list[str]) -> None:
        if depth == 5:
            paths.append(prefix)
            return
        for child in edges.get(parent, []):
            rec(child, depth + 1, prefix + [child])

    rec("ROOT", 0, [])
    return paths


def descendants_inclusive(
    root: str,
    edges: dict[str, list[str]],
    node_by_alias: dict[str, dict[str, Any]],
) -> set[str]:
    aliases: set[str] = set()

    def rec(alias: str) -> None:
        if alias in node_by_alias:
            aliases.add(alias)
        for child in edges.get(alias, []):
            rec(child)

    rec(root)
    return aliases


def path_base_aliases(
    alias: str,
    node_by_alias: dict[str, dict[str, Any]],
    parent_by_alias: dict[str, str],
) -> list[str]:
    aliases: list[str] = []
    cursor = alias
    while cursor != "ROOT":
        aliases.append(cursor)
        cursor = parent_by_alias[cursor]
    aliases.reverse()
    return [str(node_by_alias[item]["base_alias"]) for item in aliases]


def classify_prefix(base_path: list[str]) -> str:
    if not base_path:
        return "mixed"
    if base_path[0] == "stage1_n4":
        return "trap_root"
    if any(alias in {"stage2_n3", "stage3_n4", "stage4_n4", "stage5_n4"} for alias in base_path):
        return "decoy"
    if len(base_path) >= 2 and base_path[1] in {"stage2_n1", "stage2_n2", "stage2_n4"}:
        return "target_shared"
    return "mixed"


def classify_base_path(base_path: list[str]) -> str:
    if base_path[0] == "stage1_n4":
        return "trap_root"
    if (
        base_path[1] in {"stage2_n1", "stage2_n2", "stage2_n4"}
        and base_path[2] in {"stage3_n1", "stage3_n2", "stage3_n3"}
        and base_path[3] in {"stage4_n1", "stage4_n2", "stage4_n3"}
        and base_path[4] in {"stage5_n1", "stage5_n2", "stage5_n3"}
    ):
        return "target_shared"
    return "decoy"


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    stages = list(spec["stages"])
    edges = spec["edges_by_node_alias"]
    nodes = [node for stage in stages for node in spec["nodes"].get(stage, [])]
    node_by_alias = {str(node["alias"]): node for node in nodes}
    paths = enumerate_alias_paths(edges)
    base_paths = [
        [str(node_by_alias[alias]["base_alias"]) for alias in path]
        for path in paths
    ]
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

    return {
        "tree_name": spec["tree_name"],
        "family_kind": FAMILY_KIND,
        "depth": int(spec["depth"]),
        "num_paths": len(paths),
        "total_agent_ids": len(nodes),
        "per_stage_node_counts": {
            stage: len(spec["nodes"].get(stage, []))
            for stage in stages
        },
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
        "stage2_full_share_subtree_roots": stage2_full_share_roots,
        "excluded_base_aliases": ["stage1_n5", "stage2_n5", "stage3_n5", "stage4_n5", "stage5_n5"],
        "base_edge_rules": BASE_EDGES,
        "metadata": spec.get("metadata", {}),
    }


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
                    "private_leaf_counts_by_archetype": validation["private_leaf_counts_by_archetype"],
                    "stage2_full_share_subtree_roots": validation["stage2_full_share_subtree_roots"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
