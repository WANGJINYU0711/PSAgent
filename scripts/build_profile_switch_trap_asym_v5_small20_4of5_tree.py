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
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5.json"
)
VALIDATION_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5_validation.json"
)

TREE_NAME = "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5"
FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5"

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

    s1_trap = add(
        conceptual_alias="s1_n0_fast_trap_root",
        stage="stage1",
        g=1,
        base_alias="stage1_n4",
        parent_alias="ROOT",
        parent_serial="root",
        local_child_index=1,
        profile_mode="fast",
        role_note="narrow private fast trap root, covers L00-L03",
    )
    s1_share = add(
        conceptual_alias="s1_n1_full_share_target_basin_root",
        stage="stage1",
        g=0,
        base_alias="stage1_n1",
        parent_alias="ROOT",
        parent_serial="root",
        local_child_index=2,
        profile_mode="fast",
        role_note="full-share target/shared basin root, covers L04-L19",
    )

    s2_trap = add(
        conceptual_alias="s2_n0_fast_trap_router",
        stage="stage2",
        g=1,
        base_alias="stage2_n3",
        parent_alias=s1_trap,
        parent_serial="n0001",
        local_child_index=1,
        profile_mode="fast",
        role_note="private fast trap router",
    )
    shared_stage2 = [
        add(
            conceptual_alias="s2_n1_target_router_A",
            stage="stage2",
            g=0,
            base_alias="stage2_n1",
            parent_alias=s1_share,
            parent_serial="n0002",
            local_child_index=1,
            profile_mode="deep",
            role_note="shared target router A",
        ),
        add(
            conceptual_alias="s2_n2_general_core_B",
            stage="stage2",
            g=0,
            base_alias="stage2_n2",
            parent_alias=s1_share,
            parent_serial="n0002",
            local_child_index=2,
            profile_mode="fast",
            role_note="shared general core B",
        ),
        add(
            conceptual_alias="s2_n3_general_roaming_C",
            stage="stage2",
            g=0,
            base_alias="stage2_n4",
            parent_alias=s1_share,
            parent_serial="n0002",
            local_child_index=3,
            profile_mode="deep",
            role_note="shared roaming/general lane C",
        ),
        add(
            conceptual_alias="s2_n4_target_router_D",
            stage="stage2",
            g=0,
            base_alias="stage2_n1",
            parent_alias=s1_share,
            parent_serial="n0002",
            local_child_index=4,
            profile_mode="deep",
            role_note="shared target router D",
        ),
    ]

    s3_trap = add(
        conceptual_alias="s3_n0_fast_trap_network",
        stage="stage3",
        g=1,
        base_alias="stage3_n4",
        parent_alias=s2_trap,
        parent_serial="n0003",
        local_child_index=1,
        profile_mode="fast",
        role_note="private fast trap network",
    )
    shared_stage3 = [
        add(
            conceptual_alias="s3_n1_target_apn_A1",
            stage="stage3",
            g=0,
            base_alias="stage3_n1",
            parent_alias=shared_stage2[0],
            parent_serial="n0004",
            local_child_index=1,
            profile_mode="deep",
            role_note="shared target APN A1",
        ),
        add(
            conceptual_alias="s3_n2_target_roaming_A2",
            stage="stage3",
            g=0,
            base_alias="stage3_n2",
            parent_alias=shared_stage2[0],
            parent_serial="n0004",
            local_child_index=2,
            profile_mode="deep",
            role_note="shared target roaming A2",
        ),
        add(
            conceptual_alias="s3_n3_general_network_B1",
            stage="stage3",
            g=0,
            base_alias="stage3_n3",
            parent_alias=shared_stage2[1],
            parent_serial="n0005",
            local_child_index=1,
            profile_mode="deep",
            role_note="shared general network B1",
        ),
        add(
            conceptual_alias="s3_n4_target_apn_B2",
            stage="stage3",
            g=0,
            base_alias="stage3_n1",
            parent_alias=shared_stage2[1],
            parent_serial="n0005",
            local_child_index=2,
            profile_mode="deep",
            role_note="shared target APN B2",
        ),
        add(
            conceptual_alias="s3_n5_target_roaming_C1",
            stage="stage3",
            g=0,
            base_alias="stage3_n2",
            parent_alias=shared_stage2[2],
            parent_serial="n0006",
            local_child_index=1,
            profile_mode="deep",
            role_note="shared target roaming C1",
        ),
        add(
            conceptual_alias="s3_n6_general_network_C2",
            stage="stage3",
            g=0,
            base_alias="stage3_n3",
            parent_alias=shared_stage2[2],
            parent_serial="n0006",
            local_child_index=2,
            profile_mode="deep",
            role_note="shared general network C2",
        ),
        add(
            conceptual_alias="s3_n7_target_apn_D1",
            stage="stage3",
            g=0,
            base_alias="stage3_n1",
            parent_alias=shared_stage2[3],
            parent_serial="n0007",
            local_child_index=1,
            profile_mode="deep",
            role_note="shared target APN D1",
        ),
    ]

    trap_stage4 = [
        add(
            conceptual_alias="s4_n0_fast_trap_execute_A",
            stage="stage4",
            g=1,
            base_alias="stage4_n4",
            parent_alias=s3_trap,
            parent_serial="n0008",
            local_child_index=1,
            profile_mode="fast",
            role_note="private fast stage4 trap A",
        ),
        add(
            conceptual_alias="s4_n1_fast_trap_execute_B",
            stage="stage4",
            g=1,
            base_alias="stage4_n4",
            parent_alias=s3_trap,
            parent_serial="n0008",
            local_child_index=2,
            profile_mode="fast",
            role_note="private fast stage4 trap B",
        ),
    ]

    shared_stage4_specs = [
        (shared_stage3[0], "n0009", 1, "s4_n2_target_repair_A1a", "stage4_n1"),
        (shared_stage3[0], "n0009", 2, "s4_n3_general_repair_A1b", "stage4_n2"),
        (shared_stage3[1], "n0010", 1, "s4_n4_target_repair_A2", "stage4_n1"),
        (shared_stage3[2], "n0011", 1, "s4_n5_general_repair_B1", "stage4_n2"),
        (shared_stage3[3], "n0012", 1, "s4_n6_general_verify_B2", "stage4_n3"),
        (shared_stage3[4], "n0013", 1, "s4_n7_target_repair_C1", "stage4_n1"),
        (shared_stage3[5], "n0014", 1, "s4_n8_general_repair_C2", "stage4_n2"),
        (shared_stage3[6], "n0015", 1, "s4_n9_target_repair_D1", "stage4_n1"),
    ]
    shared_stage4 = [
        add(
            conceptual_alias=conceptual_alias,
            stage="stage4",
            g=0,
            base_alias=base_alias,
            parent_alias=parent_alias,
            parent_serial=parent_serial,
            local_child_index=child_index,
            profile_mode="deep",
            role_note="shared deep target-safe stage4 repair/verify",
        )
        for parent_alias, parent_serial, child_index, conceptual_alias, base_alias in shared_stage4_specs
    ]

    for idx, parent_alias in enumerate(trap_stage4):
        parent_serial = f"n{16 + idx:04d}"
        for child_index in (1, 2):
            add(
                conceptual_alias=f"s5_n{idx * 2 + child_index - 1:02d}_L{idx * 2 + child_index - 1:02d}_fast_trap",
                stage="stage5",
                g=1,
                base_alias="stage5_n4",
                parent_alias=parent_alias,
                parent_serial=parent_serial,
                local_child_index=child_index,
                profile_mode="fast",
                role_note="private fast trap terminal leaf",
            )

    leaf_serial = 4
    for offset, parent_alias in enumerate(shared_stage4):
        parent_serial = f"n{18 + offset:04d}"
        for child_index, base_alias in enumerate(("stage5_n1", "stage5_n2"), start=1):
            add(
                conceptual_alias=f"s5_n{leaf_serial:02d}_L{leaf_serial:02d}_deep_target",
                stage="stage5",
                g=0,
                base_alias=base_alias,
                parent_alias=parent_alias,
                parent_serial=parent_serial,
                local_child_index=child_index,
                profile_mode="deep",
                role_note="shared deep target terminal leaf",
            )
            leaf_serial += 1

    metadata = {
        "source_tree_name": "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5",
        "profile_switch_variant": "trap_asym_v5_small20_4of5",
        "profile_switch_version": "profile_switch_v1",
        "compatible_with": ["run_shared_basin_repeated_smoke_setup"],
        "not_directly_compatible_with_current_shared_basin_llm_runner": False,
        "preserve_g": False,
        "g_layout_policy": "small20_exact_internal_and_leaf_4of5_full_stage1_share_basin_v1",
        "base_cost_policy": "unchanged_from_profile_switch_preset",
        "purpose": (
            "Compact 20-leaf 4/5-share profile-switch tree for 100-episode "
            "pre-switch exploration: preserve the v3 efficient-anchor constraints "
            "while making most leaves discoverable before the episode-100 switch."
        ),
        "conceptual_design": {
            "leaf_depth": 5,
            "root_depth_including_root": 6,
            "num_leaves": 20,
            "trap_leaves": ["L00", "L01", "L02", "L03"],
            "deep_target_leaves": [f"L{i:02d}" for i in range(4, 20)],
            "leaf_share_fraction": "16/20",
            "internal_share_fraction": "20/25",
            "stage1_full_share_root": "s1_n1_full_share_target_basin_root",
            "stage1_full_share_leaf_count": 16,
            "trap_root_leaf_count": 4,
        },
        "notes": [
            "The JSON schema matches existing v3/v4 prefix-dedup topology specs.",
            "Exactly 4/5 of leaves are shareable and exactly 4/5 of internal nodes are shareable.",
            "The stage1_n1 subtree is forced all-share from stage1 through all 16 descendant leaves.",
            "Trap paths remain narrow: one stage1 fast trap entrance with four private fast terminal leaves.",
            "No barrier base aliases n5 are included; hard-transfer safety should still be evaluated separately.",
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
    leaf_depths = [len(path) for path in paths]
    internal_nodes = [node for node in nodes if node["stage"] != "stage5"]
    leaf_nodes = [node for node in nodes if node["stage"] == "stage5"]
    child_counts = [len(children) for children in edges.values() if children]

    full_share_roots: dict[str, list[dict[str, Any]]] = {}
    for stage in spec["stages"]:
        full_share_roots[stage] = []
        for node_item in spec["nodes"][stage]:
            alias = str(node_item["alias"])
            descendants = descendants_inclusive(alias, edges)
            if int(node_item["g"]) == 0 and all(int(node_by_alias[item]["g"]) == 0 for item in descendants):
                full_share_roots[stage].append(
                    {
                        "alias": alias,
                        "conceptual_alias": node_item.get("conceptual_alias"),
                        "base_alias": node_item.get("base_alias"),
                        "descendant_internal_count": sum(
                            1 for item in descendants if node_by_alias[item]["stage"] != "stage5"
                        ),
                        "descendant_leaf_count": sum(
                            1 for item in descendants if node_by_alias[item]["stage"] == "stage5"
                        ),
                    }
                )

    path_records = []
    for path in paths:
        path_records.append(
            {
                "aliases": path,
                "conceptual_aliases": [node_by_alias[item].get("conceptual_alias") for item in path],
                "base_aliases": [node_by_alias[item].get("base_alias") for item in path],
                "g": [int(node_by_alias[item]["g"]) for item in path],
                "profile_modes": [node_by_alias[item].get("profile_mode") for item in path],
            }
        )

    trap_paths = [
        record
        for record in path_records
        if record["base_aliases"][0] == "stage1_n4"
        or "stage5_n4" in record["base_aliases"]
    ]
    target_safe_paths = [
        record
        for record in path_records
        if record["base_aliases"][0] in {"stage1_n1", "stage1_n3"}
        and record["base_aliases"][3] in {"stage4_n1", "stage4_n2", "stage4_n3"}
        and record["base_aliases"][4] in {"stage5_n1", "stage5_n2"}
    ]

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
        "leaf_depths": sorted(set(leaf_depths)),
        "root_branching": len(edges.get("ROOT", [])),
        "min_branching": min(child_counts) if child_counts else 0,
        "max_branching": max(child_counts) if child_counts else 0,
        "full_share_roots_by_stage": full_share_roots,
        "profile_mode_counts_by_stage": {
            stage: dict(Counter(str(node.get("profile_mode", "")) for node in spec["nodes"][stage]))
            for stage in spec["stages"]
        },
        "leaf_profile_mode_counts": dict(
            Counter(str(node.get("profile_mode", "")) for node in leaf_nodes)
        ),
        "trap_path_count": len(trap_paths),
        "trap_path_fraction": len(trap_paths) / max(1, len(paths)),
        "target_safe_path_count": len(target_safe_paths),
        "target_safe_path_fraction": len(target_safe_paths) / max(1, len(paths)),
        "paths": path_records,
        "metadata": spec.get("metadata", {}),
    }
    expected_checks = {
        "num_paths_is_20": len(paths) == 20,
        "all_leaves_at_stage5_depth": sorted(set(leaf_depths)) == [5],
        "internal_share_count_is_20_of_25": validation["internal_share_count"] == 20
        and validation["internal_node_count"] == 25,
        "leaf_share_count_is_16_of_20": validation["leaf_share_count"] == 16
        and validation["leaf_node_count"] == 20,
        "stage1_has_one_full_share_root": len(full_share_roots["stage1"]) == 1,
        "stage1_full_share_root_covers_16_leaves": bool(full_share_roots["stage1"])
        and full_share_roots["stage1"][0]["descendant_leaf_count"] == 16,
        "trap_path_count_is_4": len(trap_paths) == 4,
        "target_safe_path_count_is_16": len(target_safe_paths) == 16,
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
                    "internal_share_fraction": validation["internal_share_fraction"],
                    "leaf_share_fraction": validation["leaf_share_fraction"],
                    "trap_path_count": validation["trap_path_count"],
                    "target_safe_path_count": validation["target_safe_path_count"],
                    "stage1_full_share_roots": validation["full_share_roots_by_stage"]["stage1"],
                    "validation_errors": validation["validation_errors"],
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
