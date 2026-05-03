from __future__ import annotations

import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch.json"
)
TARGET_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v1.json"
)
VALIDATION_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v1_validation.json"
)

TREE_NAME = "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v1"

# Compact base-alias continuation map. The prefix-expansion below clones every
# child per concrete parent prefix, matching the existing prefix-dedup layout.
BASE_EDGES: dict[str, list[str]] = {
    # Keep one narrow all-fast trap, two target-compatible entrances, and one
    # decoy/general entrance. Barrier n5 is intentionally absent from this local
    # PS-vs-exp3 tree; the 4/5 share ratio is assigned over the expanded nodes
    # below, without changing this topology.
    "ROOT": ["stage1_n1", "stage1_n2", "stage1_n3", "stage1_n4"],
    # Target-compatible entrances: broad g=0 basin with fast stage1 available
    # through n1, which matches the dominant fdddd target requirement.
    "stage1_n1": ["stage2_n1", "stage2_n2", "stage2_n4"],
    "stage1_n3": ["stage2_n1", "stage2_n2", "stage2_n4"],
    # Decoy/general entrance: can look target-adjacent but has trap leakage.
    "stage1_n2": ["stage2_n2", "stage2_n3", "stage2_n4"],
    # Narrow pre-switch bait.
    "stage1_n4": ["stage2_n3"],
    # Stage2 target/shared subtree roots. The g layout below protects n1 as a
    # full-share descendant subtree; n3 is the narrow trap/decoy leakage.
    "stage2_n1": ["stage3_n1", "stage3_n2", "stage3_n3"],
    "stage2_n2": ["stage3_n1", "stage3_n3", "stage3_n4"],
    "stage2_n3": ["stage3_n4"],
    "stage2_n4": ["stage3_n1", "stage3_n2", "stage3_n3", "stage3_n4"],
    # Stage3 keeps the target basin wide but pure; n4 injects fast trap leakage.
    "stage3_n1": ["stage4_n1", "stage4_n2", "stage4_n3"],
    "stage3_n2": ["stage4_n1", "stage4_n2", "stage4_n3"],
    "stage3_n3": ["stage4_n1", "stage4_n2", "stage4_n3", "stage4_n4"],
    "stage3_n4": ["stage4_n2", "stage4_n4"],
    # Stage4 target/shared endings remain wide; n4 is the all-fast terminal.
    "stage4_n1": ["stage5_n1", "stage5_n2", "stage5_n3"],
    "stage4_n2": ["stage5_n1", "stage5_n2", "stage5_n3"],
    "stage4_n3": ["stage5_n1", "stage5_n2", "stage5_n3", "stage5_n4"],
    "stage4_n4": ["stage5_n4"],
}

LEAF_SHARE_NUMERATOR = 4
LEAF_SHARE_DENOMINATOR = 5
INTERNAL_SHARE_NUMERATOR = 4
INTERNAL_SHARE_DENOMINATOR = 5
FULL_SHARE_STAGE2_BASE_ALIAS = "stage2_n1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stage_from_base_alias(base_alias: str) -> str:
    return base_alias.split("_n", 1)[0]


def enumerate_base_paths() -> list[list[str]]:
    paths: list[list[str]] = []

    def rec(parent: str, depth: int, prefix: list[str]) -> None:
        if depth == 5:
            paths.append(prefix)
            return
        for child in BASE_EDGES.get(parent, []):
            rec(child, depth + 1, prefix + [child])

    rec("ROOT", 0, [])
    return paths


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
            base_child_aliases = list(BASE_EDGES.get(parent_base_alias, []))
            expanded_edges[parent_alias] = []
            for local_idx, base_child_alias in enumerate(base_child_aliases, start=1):
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

    apply_4of5_g_layout(expanded_nodes, expanded_edges)

    metadata = dict(source_spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    metadata.update(
        {
            "source_tree_name": source_spec.get("tree_name"),
            "profile_switch_variant": "trap_asym_v1",
            "profile_switch_version": metadata.get("profile_switch_version", "profile_switch_v1"),
            "compatible_with": sorted(compatible_with),
            "not_directly_compatible_with_current_shared_basin_llm_runner": False,
            "preserve_g": False,
            "g_layout_policy": "trap_asym_4of5_expanded_node_layout_v1",
            "base_cost_policy": "unchanged_from_profile_switch_preset",
            "purpose": (
                "Trap-asymmetric profile-switch topology: narrow pre-switch fast trap, "
                "wide 4/5-share target basin, and medium decoy leakage for PS-vs-exp3 comparison."
            ),
            "notes": [
                "Topology only; profile capabilities, deliberation modes, and base_cost ranges are unchanged.",
                "Expanded-node g values are reassigned to make exactly 4/5 of leaves and 4/5 of internal nodes shareable.",
                "Barrier base aliases n5 are intentionally excluded from this local comparison tree; private g=1 mass is placed on trap/decoy clones instead.",
                "The narrow trap is stage1_n4/stage2_n3/stage3_n4/stage4_n4/stage5_n4 plus small leakage variants.",
                "All expanded stage2_n1 roots are protected as full-share subtree roots.",
                "Hard-transfer/barrier safety should be checked with a separate control variant, not this local PS-vs-exp3 tree.",
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


def apply_4of5_g_layout(
    expanded_nodes: dict[str, list[dict[str, Any]]],
    expanded_edges: dict[str, list[str]],
) -> None:
    """Assign g over expanded clones while preserving the trap-asym topology."""
    all_nodes = [node for stage_nodes in expanded_nodes.values() for node in stage_nodes]
    node_by_alias = {str(node["alias"]): node for node in all_nodes}
    parent_by_alias = {
        child: parent
        for parent, children in expanded_edges.items()
        for child in children
    }

    def descendants_inclusive(root: str) -> set[str]:
        aliases: set[str] = set()

        def rec(alias: str) -> None:
            aliases.add(alias)
            for child in expanded_edges.get(alias, []):
                rec(child)

        rec(root)
        return aliases

    protected_full_share_aliases: set[str] = set()
    for node in all_nodes:
        if node["stage"] == "stage2" and node["base_alias"] == FULL_SHARE_STAGE2_BASE_ALIAS:
            protected_full_share_aliases.update(descendants_inclusive(str(node["alias"])))

    internal_nodes = [node for node in all_nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in all_nodes if node["stage"] == "stage5"]
    internal_private_target = len(internal_nodes) - (
        len(internal_nodes) * INTERNAL_SHARE_NUMERATOR // INTERNAL_SHARE_DENOMINATOR
    )
    leaf_private_target = len(leaf_nodes) - (
        len(leaf_nodes) * LEAF_SHARE_NUMERATOR // LEAF_SHARE_DENOMINATOR
    )

    def path_base_aliases(alias: str) -> list[str]:
        aliases: list[str] = []
        cursor = alias
        while cursor != "ROOT":
            aliases.append(cursor)
            cursor = parent_by_alias[cursor]
        aliases.reverse()
        return [str(node_by_alias[item]["base_alias"]) for item in aliases]

    def is_target_shared_path(base_path: list[str]) -> bool:
        return (
            len(base_path) == 5
            and base_path[1] in {"stage2_n1", "stage2_n2", "stage2_n4"}
            and base_path[2] in {"stage3_n1", "stage3_n2", "stage3_n3"}
            and base_path[3] in {"stage4_n1", "stage4_n2", "stage4_n3"}
            and base_path[4] in {"stage5_n1", "stage5_n2", "stage5_n3"}
        )

    for node in all_nodes:
        node["g"] = 0

    # Private internal mass is concentrated on the narrow fast trap and on a
    # tiny number of trap-leakage bridge clones. This preserves a full-share
    # stage2_n1 target subtree while making the overall internal ratio exactly
    # 4/5 share.
    internal_candidates = [
        node
        for node in internal_nodes
        if str(node["alias"]) not in protected_full_share_aliases
        and node["base_alias"] in {"stage2_n3", "stage3_n4", "stage4_n4"}
    ]
    internal_candidates.extend(
        node
        for node in internal_nodes
        if str(node["alias"]) not in protected_full_share_aliases
        and node["base_alias"] == "stage4_n2"
        and node["parent_base_alias"] == "stage3_n4"
    )
    internal_candidates = sorted(
        {str(node["alias"]): node for node in internal_candidates}.values(),
        key=lambda node: (
            {"stage2": 0, "stage3": 1, "stage4": 2}.get(str(node["stage"]), 9),
            str(node["base_alias"]),
            str(node["node_serial"]),
        ),
    )
    if len(internal_candidates) < internal_private_target:
        raise ValueError(
            f"Need {internal_private_target} internal g=1 nodes, "
            f"only found {len(internal_candidates)} candidates."
        )
    for node in internal_candidates[:internal_private_target]:
        node["g"] = 1

    # Private leaves are selected from non-target trap/decoy terminal paths,
    # leaving the protected stage2_n1 subtree and all target-shared leaves as
    # shareable whenever possible.
    leaf_candidates = [
        node
        for node in leaf_nodes
        if str(node["alias"]) not in protected_full_share_aliases
        and not is_target_shared_path(path_base_aliases(str(node["alias"])))
    ]
    leaf_candidates = sorted(
        leaf_candidates,
        key=lambda node: (
            0 if node["base_alias"] == "stage5_n4" else 1,
            str(node["base_alias"]),
            str(node["node_serial"]),
        ),
    )
    if len(leaf_candidates) < leaf_private_target:
        raise ValueError(
            f"Need {leaf_private_target} leaf g=1 nodes, "
            f"only found {len(leaf_candidates)} candidates."
        )
    for node in leaf_candidates[:leaf_private_target]:
        node["g"] = 1


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    stages = list(spec["stages"])
    edges = spec["edges_by_node_alias"]
    nodes = [node for stage in stages for node in spec["nodes"].get(stage, [])]
    node_by_alias = {str(node["alias"]): node for node in nodes}
    parent_by_alias = {
        child: parent
        for parent, children in edges.items()
        for child in children
    }

    paths: list[list[str]] = []

    def rec(parent: str, depth: int, prefix: list[str]) -> None:
        if depth == 5:
            paths.append(prefix)
            return
        for child in edges.get(parent, []):
            rec(child, depth + 1, prefix + [child])

    rec("ROOT", 0, [])
    base_paths = [
        [str(node_by_alias[alias]["base_alias"]) for alias in path]
        for path in paths
    ]
    alias_paths = paths
    internal_nodes = [node for node in nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in nodes if node["stage"] == "stage5"]
    child_counts = [len(children) for children in edges.values() if children]
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
    trap_leaf_paths = [
        path
        for path in base_paths
        if path
        == ["stage1_n4", "stage2_n3", "stage3_n4", "stage4_n4", "stage5_n4"]
    ]
    trap_root_paths = [path for path in base_paths if path[0] == "stage1_n4"]
    target_shared_paths = [
        path
        for path in base_paths
        if path[1] in {"stage2_n1", "stage2_n2", "stage2_n4"}
        and path[2] in {"stage3_n1", "stage3_n2", "stage3_n3"}
        and path[3] in {"stage4_n1", "stage4_n2", "stage4_n3"}
        and path[4] in {"stage5_n1", "stage5_n2", "stage5_n3"}
    ]
    decoy_paths = [
        path
        for path in base_paths
        if path not in target_shared_paths and path[0] != "stage1_n4"
    ]

    def descendants_inclusive(root: str) -> set[str]:
        aliases: set[str] = set()

        def rec(alias: str) -> None:
            aliases.add(alias)
            for child in edges.get(alias, []):
                rec(child)

        rec(root)
        return aliases

    stage2_full_share_roots = []
    for node in spec["nodes"].get("stage2", []):
        alias = str(node["alias"])
        descendants = descendants_inclusive(alias)
        if int(node["g"]) == 0 and all(int(node_by_alias[item]["g"]) == 0 for item in descendants):
            stage2_full_share_roots.append(
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
    stage2_full_share_base_aliases = sorted(
        {
            root["base_alias"]
            for root in stage2_full_share_roots
            if all(
                any(
                    item["alias"] == str(node["alias"])
                    for item in stage2_full_share_roots
                )
                for node in spec["nodes"].get("stage2", [])
                if str(node["base_alias"]) == root["base_alias"]
            )
        }
    )
    target_shared_private_leaf_count = 0
    non_target_private_leaf_count = 0
    for alias_path, base_path in zip(alias_paths, base_paths, strict=True):
        leaf = node_by_alias[alias_path[-1]]
        is_target_shared = base_path in target_shared_paths
        if int(leaf["g"]) == 1 and is_target_shared:
            target_shared_private_leaf_count += 1
        if int(leaf["g"]) == 1 and not is_target_shared:
            non_target_private_leaf_count += 1

    return {
        "tree_name": spec["tree_name"],
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
        "private_internal_counts_by_base_alias": dict(
            sorted(Counter(str(node["base_alias"]) for node in internal_nodes if int(node["g"]) == 1).items())
        ),
        "private_leaf_counts_by_base_alias": dict(
            sorted(Counter(str(node["base_alias"]) for node in leaf_nodes if int(node["g"]) == 1).items())
        ),
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
        "exact_full_trap_path_count": len(trap_leaf_paths),
        "decoy_path_count": len(decoy_paths),
        "decoy_path_fraction": len(decoy_paths) / max(1, len(base_paths)),
        "all_nodes_g0": all(int(node["g"]) == 0 for node in nodes),
        "target_shared_private_leaf_count": target_shared_private_leaf_count,
        "non_target_private_leaf_count": non_target_private_leaf_count,
        "stage2_full_share_subtree_roots": stage2_full_share_roots,
        "stage2_full_share_base_aliases": stage2_full_share_base_aliases,
        "protected_full_share_stage2_base_alias": FULL_SHARE_STAGE2_BASE_ALIAS,
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
                "summary": {
                    "num_paths": validation["num_paths"],
                    "total_agent_ids": validation["total_agent_ids"],
                    "per_stage_node_counts": validation["per_stage_node_counts"],
                    "target_shared_path_count": validation["target_shared_path_count"],
                    "trap_root_path_count": validation["trap_root_path_count"],
                    "decoy_path_count": validation["decoy_path_count"],
                    "mode_pattern_counts": validation["mode_pattern_counts"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
