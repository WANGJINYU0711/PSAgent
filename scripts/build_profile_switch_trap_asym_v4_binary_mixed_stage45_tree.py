from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TARGET_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45.json"
)
VALIDATION_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_validation.json"
)

TREE_NAME = "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"

STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def node(
    *,
    alias: str,
    conceptual_alias: str,
    display_alias: str,
    stage: str,
    g: int,
    base_alias: str,
    parent_alias: str,
    parent_serial: str,
    node_serial: str,
    local_child_index: int,
    profile_mode: str,
    role_note: str,
) -> dict[str, Any]:
    return {
        "alias": alias,
        "display_alias": display_alias,
        "agent_id": alias,
        "stage": stage,
        "g": int(g),
        "base_alias": base_alias,
        "source_alias": base_alias,
        "conceptual_alias": conceptual_alias,
        "parent_alias": parent_alias,
        "parent_base_alias": "ROOT" if parent_alias == "ROOT" else parent_alias.split("__from__", 1)[0],
        "parent_serial": parent_serial,
        "node_serial": node_serial,
        "local_child_index": int(local_child_index),
        "clone_scope": "parent_specific",
        "profile_mode": profile_mode,
        "role_note": role_note,
    }


def enumerate_paths(edges: dict[str, list[str]]) -> list[list[str]]:
    paths: list[list[str]] = []

    def rec(alias: str, prefix: list[str]) -> None:
        children = edges.get(alias, [])
        if not children:
            paths.append(prefix)
            return
        for child in children:
            rec(child, prefix + [child])

    rec("ROOT", [])
    return paths


def descendants_inclusive(root: str, edges: dict[str, list[str]]) -> set[str]:
    out: set[str] = set()

    def rec(alias: str) -> None:
        out.add(alias)
        for child in edges.get(alias, []):
            rec(child)

    rec(root)
    return out


def build_spec() -> dict[str, Any]:
    nodes: dict[str, list[dict[str, Any]]] = {stage: [] for stage in STAGES}
    edges: dict[str, list[str]] = {"ROOT": []}

    serial = 1

    def add(
        *,
        conceptual_alias: str,
        stage: str,
        g: int,
        base_alias: str,
        parent_alias: str,
        parent_serial: str,
        local_child_index: int,
        profile_mode: str,
        role_note: str,
    ) -> str:
        nonlocal serial
        node_serial = f"n{serial:04d}"
        serial += 1
        alias = f"{base_alias}__from__{parent_serial}__c{local_child_index:02d}"
        item = node(
            alias=alias,
            conceptual_alias=conceptual_alias,
            display_alias=f"{conceptual_alias} ({base_alias}) <- {parent_serial}",
            stage=stage,
            g=g,
            base_alias=base_alias,
            parent_alias=parent_alias,
            parent_serial=parent_serial,
            node_serial=node_serial,
            local_child_index=local_child_index,
            profile_mode=profile_mode,
            role_note=role_note,
        )
        nodes[stage].append(item)
        edges.setdefault(parent_alias, []).append(alias)
        edges[alias] = []
        return alias

    s1_n0 = add(
        conceptual_alias="s1_n0_fast_trap_root",
        stage="stage1",
        g=1,
        base_alias="stage1_n4",
        parent_alias="ROOT",
        parent_serial="root",
        local_child_index=1,
        profile_mode="fast",
        role_note="non-share fast trap root, covers L0-L1",
    )
    s1_n1 = add(
        conceptual_alias="s1_n1_deep_shared_basin_root",
        stage="stage1",
        g=0,
        base_alias="stage1_n3",
        parent_alias="ROOT",
        parent_serial="root",
        local_child_index=2,
        profile_mode="deep",
        role_note="largest full-share basin root, covers L2-L7",
    )

    s2_n0 = add(
        conceptual_alias="s2_n0_fast_trap_contract",
        stage="stage2",
        g=1,
        base_alias="stage2_n3",
        parent_alias=s1_n0,
        parent_serial="n0001",
        local_child_index=1,
        profile_mode="fast",
        role_note="non-share fast trap contract",
    )
    s2_n1 = add(
        conceptual_alias="s2_n1_deep_shared_A_mixed",
        stage="stage2",
        g=0,
        base_alias="stage2_n1",
        parent_alias=s1_n1,
        parent_serial="n0002",
        local_child_index=1,
        profile_mode="deep",
        role_note="4-leaf mixed shared basin with one fast leaf decoy",
    )
    s2_n2 = add(
        conceptual_alias="s2_n2_fastish_shared_B_wrapper",
        stage="stage2",
        g=0,
        base_alias="stage2_n2",
        parent_alias=s1_n1,
        parent_serial="n0002",
        local_child_index=2,
        profile_mode="fast",
        role_note="fast-deliberation shared wrapper above deep stage4 target leaves",
    )

    s3_n0 = add(
        conceptual_alias="s3_n0_fast_trap_A",
        stage="stage3",
        g=1,
        base_alias="stage3_n4",
        parent_alias=s2_n0,
        parent_serial="n0003",
        local_child_index=1,
        profile_mode="fast",
        role_note="non-share fast trap A",
    )
    s3_n1 = add(
        conceptual_alias="s3_n1_mixed_trap_B_deep_hint",
        stage="stage3",
        g=1,
        base_alias="stage3_n1",
        parent_alias=s2_n0,
        parent_serial="n0003",
        local_child_index=2,
        profile_mode="deep",
        role_note="non-share deep hint inside fast trap side",
    )
    s3_n2 = add(
        conceptual_alias="s3_n2_deep_A1_shared",
        stage="stage3",
        g=0,
        base_alias="stage3_n1",
        parent_alias=s2_n1,
        parent_serial="n0004",
        local_child_index=1,
        profile_mode="deep",
        role_note="2-leaf deep shared A1 root",
    )
    s3_n3 = add(
        conceptual_alias="s3_n3_deep_A2_shared_with_decoy",
        stage="stage3",
        g=0,
        base_alias="stage3_n2",
        parent_alias=s2_n1,
        parent_serial="n0004",
        local_child_index=2,
        profile_mode="deep",
        role_note="2-leaf deep shared A2 root, one fast terminal decoy",
    )
    s3_n4 = add(
        conceptual_alias="s3_n4_fastish_B_shared_wrapper",
        stage="stage3",
        g=0,
        base_alias="stage3_n4",
        parent_alias=s2_n2,
        parent_serial="n0005",
        local_child_index=1,
        profile_mode="fast",
        role_note="fast-deliberation non-trap wrapper for L6-L7",
    )

    s4_n0 = add(
        conceptual_alias="s4_n0_fast_stage4_A",
        stage="stage4",
        g=1,
        base_alias="stage4_n4",
        parent_alias=s3_n0,
        parent_serial="n0006",
        local_child_index=1,
        profile_mode="fast",
        role_note="non-share fast stage4 trap A",
    )
    s4_n1 = add(
        conceptual_alias="s4_n1_mixed_stage4_B_deep_hint",
        stage="stage4",
        g=1,
        base_alias="stage4_n1",
        parent_alias=s3_n1,
        parent_serial="n0007",
        local_child_index=1,
        profile_mode="deep",
        role_note="non-share deep stage4 hint inside fast trap side",
    )
    s4_n2 = add(
        conceptual_alias="s4_n2_deep_stage4_A1",
        stage="stage4",
        g=0,
        base_alias="stage4_n1",
        parent_alias=s3_n2,
        parent_serial="n0008",
        local_child_index=1,
        profile_mode="deep",
        role_note="deep stage4 repair anchor for L2-L3",
    )
    s4_n3 = add(
        conceptual_alias="s4_n3_deep_stage4_A2",
        stage="stage4",
        g=0,
        base_alias="stage4_n1",
        parent_alias=s3_n3,
        parent_serial="n0009",
        local_child_index=1,
        profile_mode="deep",
        role_note="deep stage4 repair anchor for L4 plus L5 fast decoy",
    )
    s4_n4 = add(
        conceptual_alias="s4_n4_deep_stage4_B_repair_anchor",
        stage="stage4",
        g=0,
        base_alias="stage4_n1",
        parent_alias=s3_n4,
        parent_serial="n0010",
        local_child_index=1,
        profile_mode="deep",
        role_note="deep stage4 repair anchor below fastish wrapper, covers L6-L7",
    )

    leaf_defs = [
        (s4_n0, "n0011", 1, "s5_n0_L0_fast_trap_short", "stage5_n4", 1, "fast", "pre-switch fast trap leaf L0"),
        (s4_n1, "n0012", 1, "s5_n1_L1_fast_trap_stable", "stage5_n4", 1, "fast", "pre-switch fast trap leaf L1 with deep hints upstream"),
        (s4_n2, "n0013", 1, "s5_n2_L2_deep_target_A1a", "stage5_n1", 0, "deep", "post-switch deep target L2"),
        (s4_n2, "n0013", 2, "s5_n3_L3_deep_target_A1b", "stage5_n2", 0, "deep", "post-switch deep target L3"),
        (s4_n3, "n0014", 1, "s5_n4_L4_deep_target_A2a", "stage5_n1", 0, "deep", "post-switch deep target L4"),
        (s4_n3, "n0014", 2, "s5_n5_L5_fast_decoy_inside_deep_A", "stage5_n4", 0, "fast", "shared fast terminal decoy inside s2_n1"),
        (s4_n4, "n0015", 1, "s5_n6_L6_deep_target_Ba", "stage5_n1", 0, "deep", "post-switch deep target L6 under fastish wrapper"),
        (s4_n4, "n0015", 2, "s5_n7_L7_deep_target_Bb", "stage5_n2", 0, "deep", "post-switch deep target L7 under fastish wrapper"),
    ]
    for parent_alias, parent_serial, child_index, conceptual_alias, base_alias, g, mode, note in leaf_defs:
        add(
            conceptual_alias=conceptual_alias,
            stage="stage5",
            g=g,
            base_alias=base_alias,
            parent_alias=parent_alias,
            parent_serial=parent_serial,
            local_child_index=child_index,
            profile_mode=mode,
            role_note=note,
        )

    metadata = {
        "source_tree_name": "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5",
        "profile_switch_variant": "trap_asym_v4_binary_mixed_stage45",
        "profile_switch_version": "profile_switch_v1",
        "compatible_with": ["run_shared_basin_repeated_smoke_setup"],
        "not_directly_compatible_with_current_shared_basin_llm_runner": False,
        "preserve_g": False,
        "g_layout_policy": "binary_mixed_stage45_fixed_full_share_subtrees_v1",
        "base_cost_policy": "profile_switch_preset_with_v4_stage3_n4_fastish_nontrap_override",
        "purpose": (
            "Compact 8-leaf binary profile-switch tree for PS-vs-EXP mechanism tests: "
            "two non-share fast trap leaves and a six-leaf shared basin with mixed "
            "fast/deep profiles, deep stage4 anchors, and one shared fast terminal decoy."
        ),
        "conceptual_design": {
            "leaf_depth": 5,
            "root_depth_including_root": 6,
            "max_branching": 2,
            "fast_leaves": ["L0", "L1", "L5"],
            "deep_target_leaves": ["L2", "L3", "L4", "L6", "L7"],
            "largest_full_share_root": "s1_n1_deep_shared_basin_root",
            "stage2_full_share_roots": [
                "s2_n1_deep_shared_A_mixed",
                "s2_n2_fastish_shared_B_wrapper",
            ],
            "stage4_deep_share_roots": [
                "s4_n2_deep_stage4_A1",
                "s4_n3_deep_stage4_A2",
                "s4_n4_deep_stage4_B_repair_anchor",
            ],
        },
        "notes": [
            "Tree topology is intentionally compact and binary; it is not the large v3 prefix-expanded 250-leaf topology.",
            "The JSON schema matches existing prefix-dedup topology specs: stages, nodes by stage, base_alias, agent_id, g, and edges_by_node_alias.",
            "The conceptual s4_n4 node uses base_alias stage4_n1 so the LLM executor receives a deep target-stage repair profile.",
            "The new family preset overrides stage3_n4 to be fast deliberation but non-trap route, allowing L6/L7 to remain target-safe despite the fastish wrapper.",
            "Share flags are fixed by the conceptual design: the whole s1_n1 subtree is g=0, while the fast trap side is g=1.",
        ],
    }

    return {
        "tree_name": TREE_NAME,
        "depth": 5,
        "stages": STAGES,
        "metadata": metadata,
        "nodes": nodes,
        "edges_by_node_alias": edges,
    }


def validate(spec: dict[str, Any]) -> dict[str, Any]:
    edges = spec["edges_by_node_alias"]
    nodes = [node for stage in spec["stages"] for node in spec["nodes"][stage]]
    node_by_alias = {str(node["alias"]): node for node in nodes}
    paths = enumerate_paths(edges)
    full_share_roots: dict[str, list[dict[str, Any]]] = {}
    for stage in spec["stages"]:
        full_share_roots[stage] = []
        for node in spec["nodes"][stage]:
            alias = str(node["alias"])
            descendants = descendants_inclusive(alias, edges)
            if int(node["g"]) == 0 and all(int(node_by_alias[item]["g"]) == 0 for item in descendants):
                full_share_roots[stage].append(
                    {
                        "alias": alias,
                        "conceptual_alias": node.get("conceptual_alias"),
                        "base_alias": node.get("base_alias"),
                        "descendant_leaf_count": sum(
                            1 for item in descendants if node_by_alias[item]["stage"] == "stage5"
                        ),
                    }
                )

    child_counts = [len(children) for children in edges.values() if children]
    leaf_depths = [len(path) for path in paths]
    profile_mode_by_leaf = Counter(
        str(node_by_alias[path[-1]].get("profile_mode", "")) for path in paths
    )
    validation = {
        "tree_name": spec["tree_name"],
        "family_kind": FAMILY_KIND,
        "depth": spec["depth"],
        "num_paths": len(paths),
        "total_agent_ids": len(nodes),
        "per_stage_node_counts": {
            stage: len(spec["nodes"][stage]) for stage in spec["stages"]
        },
        "g_counts_by_stage": {
            stage: dict(Counter(int(node["g"]) for node in spec["nodes"][stage]))
            for stage in spec["stages"]
        },
        "leaf_depths": sorted(set(leaf_depths)),
        "root_branching": len(edges.get("ROOT", [])),
        "max_branching": max(child_counts) if child_counts else 0,
        "full_share_roots_by_stage": full_share_roots,
        "profile_mode_counts_by_stage": {
            stage: dict(Counter(str(node.get("profile_mode", "")) for node in spec["nodes"][stage]))
            for stage in spec["stages"]
        },
        "leaf_profile_mode_counts": dict(profile_mode_by_leaf),
        "paths": [
            {
                "aliases": path,
                "conceptual_aliases": [node_by_alias[item].get("conceptual_alias") for item in path],
                "base_aliases": [node_by_alias[item].get("base_alias") for item in path],
                "g": [int(node_by_alias[item]["g"]) for item in path],
                "profile_modes": [node_by_alias[item].get("profile_mode") for item in path],
            }
            for path in paths
        ],
        "metadata": spec.get("metadata", {}),
    }
    expected_checks = {
        "num_paths_is_8": len(paths) == 8,
        "all_leaves_at_stage5_depth": sorted(set(leaf_depths)) == [5],
        "root_is_binary": len(edges.get("ROOT", [])) == 2,
        "max_branching_at_most_2": (max(child_counts) if child_counts else 0) <= 2,
        "stage1_has_one_full_share_root": len(full_share_roots["stage1"]) == 1,
        "stage1_full_share_root_covers_6_leaves": bool(full_share_roots["stage1"])
        and full_share_roots["stage1"][0]["descendant_leaf_count"] == 6,
        "leaf_profile_modes_are_3_fast_5_deep": dict(profile_mode_by_leaf) == {"fast": 3, "deep": 5},
    }
    validation["expected_checks"] = expected_checks
    validation["validation_errors"] = [
        name for name, ok in expected_checks.items() if not ok
    ]
    return validation


def main() -> None:
    spec = build_spec()
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
                    "leaf_profile_mode_counts": validation["leaf_profile_mode_counts"],
                    "full_share_roots_by_stage": validation["full_share_roots_by_stage"],
                    "validation_errors": validation["validation_errors"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
