"""Preset configurations for neutral / moderate / strong tree families."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SHARED_BASIN_PREFIX_DEDUP_SPEC_PATH = (
    ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V1_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v1.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V2_NEUTRAL_4OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_4OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_ALL_SHARE_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_share.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_2OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_2of5.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_ALL_UNSHARE_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_unshare.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_4OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_4of5.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_2OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_2of5.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_ALL_SHARE_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_ALL_UNSHARE_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_unshare.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V5_SMALL20_4OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5.json"
)
SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V6_SMALL30_4OF5_SPEC_PATH = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v6_small30_4of5.json"
)


def build_neutral_family_spec() -> dict:
    return {
        "family_name": "neutral",
        "stages": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        "num_agents_per_stage": 3,
        "g1_per_stage": {"stage1": 1, "stage2": 1, "stage3": 1, "stage4": 1, "stage5": 1},
        "competence_per_stage": {"high": 1, "low": 2},
        "scope_per_stage": {"broad": 2, "narrow": 1},
        "stability_per_stage": {"stable": 2, "unstable": 1},
        "skill_ranges": {
            "broad": (0.55, 0.70),
            "narrow_focus": (0.70, 0.80),
            "narrow_other": (0.35, 0.50),
            "high_bonus": 0.10,
        },
        "cost_ranges": {
            "safe": (0.08, 0.12),
            "special": (0.12, 0.16),
        },
    }


def build_moderate_family_spec() -> dict:
    return {
        "family_name": "moderate",
        "stages": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        "num_agents_per_stage": 4,
        "g1_per_stage": {"stage1": 2, "stage2": 2, "stage3": 2, "stage4": 2, "stage5": 2},
        "competence_per_stage": {"high": 2, "low": 2},
        "scope_per_stage": {"broad": 2, "narrow": 2},
        "stability_per_stage": {"stable": 2, "unstable": 2},
        "skill_ranges": {
            "broad": (0.55, 0.75),
            "narrow_focus": (0.85, 0.95),
            "narrow_other": (0.25, 0.50),
            "high_bonus": 0.10,
        },
        "cost_ranges": {
            "safe": (0.08, 0.12),
            "special": (0.15, 0.22),
        },
    }


def build_strong_family_spec() -> dict:
    return {
        "family_name": "strong",
        "generation_mode": "legacy_attribute_ranges",
        "stages": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        "num_agents_per_stage": 4,
        "g1_per_stage": {"stage1": 1, "stage2": 2, "stage3": 2, "stage4": 3, "stage5": 3},
        "competence_per_stage": {"high": 2, "low": 2},
        "scope_per_stage": {"broad": 1, "narrow": 3},
        "stability_per_stage": {"stable": 2, "unstable": 2},
        "skill_ranges": {
            "broad": (0.55, 0.72),
            "narrow_focus": (0.90, 1.00),
            "narrow_other": (0.15, 0.45),
            "high_bonus": 0.10,
        },
        "cost_ranges": {
            "safe": (0.08, 0.14),
            "special": (0.18, 0.28),
        },
    }


def build_shared_basin_strong_family_spec() -> dict:
    return {
        "family_name": "shared_basin_strong",
        "generation_mode": "capability_shared_basin",
        "stages": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        "num_agents_per_stage": 5,
        "profile_fields": {
            "competence_level": "capability",
            "scope_level": "capability",
            "stability_level": "capability",
        },
        "cost_ranges": {
            "uniform": (0.12, 0.16),
        },
        "semantic_skill_ranges": {
            "safe_core": {
                "anchor": (0.88, 0.95),
                "support": (0.58, 0.72),
                "background": (0.18, 0.30),
                "focus_fallback": (0.52, 0.66),
            },
            "mixed_shared": {
                "anchor": (0.79, 0.88),
                "support": (0.40, 0.56),
                "background": (0.12, 0.24),
                "focus_fallback": (0.34, 0.48),
            },
            "private_edge": {
                "anchor": (0.89, 0.97),
                "support": (0.22, 0.36),
                "background": (0.05, 0.14),
                "focus_fallback": (0.12, 0.24),
            },
            "private_barrier": {
                "anchor": (0.86, 0.94),
                "support": (0.34, 0.50),
                "background": (0.08, 0.18),
                "focus_fallback": (0.28, 0.42),
            },
            "edge_specialist": {
                "anchor": (0.88, 0.95),
                "support": (0.18, 0.28),
                "background": (0.04, 0.12),
                "focus_fallback": (0.08, 0.18),
            },
        },
        "stage_profiles": {
            "stage1": [
                {
                    "role": "safe_core_user_grounding",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage1_intake",
                    "allowed_next_labels": ["public_stage2_core", "mixed_stage2_lane"],
                    "anchor_caps": ["user_grounding", "account_lookup"],
                    "support_caps": ["line_resolution", "verification"],
                },
                {
                    "role": "safe_core_lookup_line",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage1_intake",
                    "allowed_next_labels": ["public_stage2_core", "mixed_stage2_lane"],
                    "anchor_caps": ["account_lookup", "line_resolution"],
                    "support_caps": ["user_grounding"],
                },
                {
                    "role": "safe_core_context_verify",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage1_intake",
                    "allowed_next_labels": ["public_stage2_core", "mixed_stage2_lane"],
                    "anchor_caps": ["user_grounding", "verification"],
                    "support_caps": ["account_lookup", "line_resolution"],
                },
                {
                    "role": "mixed_shared_edge_intake",
                    "g": 0,
                    "node_semantic": "mixed_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "mixed_stage1_intake",
                    "allowed_next_labels": [
                        "public_stage2_core",
                        "mixed_stage2_lane",
                        "private_stage2_lane",
                        "private_barrier_stage2",
                    ],
                    "anchor_caps": ["user_grounding", "line_resolution"],
                    "anchor_boost": 0.06,
                    "support_caps": ["account_lookup", "verification"],
                },
                {
                    "role": "private_barrier_intake_gate",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "private_barrier_stage1",
                    "allowed_next_labels": [
                        "mixed_stage2_lane",
                        "private_stage2_lane",
                        "private_barrier_stage2",
                    ],
                    "anchor_caps": ["user_grounding", "account_lookup"],
                    "support_caps": ["line_resolution"],
                },
            ],
            "stage2": [
                {
                    "role": "safe_core_account_core",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage2_core",
                    "allowed_next_labels": [
                        "public_stage3_core",
                        "public_stage3_verify",
                    ],
                    "anchor_caps": ["account_lookup", "line_resolution"],
                    "anchor_boost": 0.04,
                    "support_caps": ["roaming_diagnosis", "verification"],
                },
                {
                    "role": "safe_core_line_core",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage2_core",
                    "allowed_next_labels": [
                        "public_stage3_edge",
                        "public_stage3_verify",
                        "mixed_stage3_lane",
                        "private_barrier_stage3",
                    ],
                    "anchor_caps": ["line_resolution"],
                    "anchor_boost": 0.01,
                    "support_caps": ["verification"],
                },
                {
                    "role": "mixed_shared_roaming_ready",
                    "g": 0,
                    "node_semantic": "mixed_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "mixed_stage2_lane",
                    "allowed_next_labels": [
                        "public_stage3_edge",
                        "mixed_stage3_lane",
                        "private_barrier_stage3",
                    ],
                    "anchor_caps": ["roaming_diagnosis", "account_lookup"],
                    "anchor_boost": 0.11,
                    "support_caps": ["line_resolution", "verification"],
                },
                {
                    "role": "private_edge_roaming_lane",
                    "g": 0,
                    "node_semantic": "private_edge",
                    "profile_kind": "shared_basin",
                    "route_label": "private_stage2_lane",
                    "allowed_next_labels": [
                        "public_stage3_edge",
                        "mixed_stage3_lane",
                        "private_barrier_stage3",
                    ],
                    "anchor_caps": ["roaming_diagnosis", "line_resolution"],
                    "anchor_boost": 0.06,
                    "support_caps": ["verification"],
                },
                {
                    "role": "private_barrier_roaming_gate",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "private_barrier_stage2",
                    "allowed_next_labels": [
                        "public_stage3_edge",
                        "mixed_stage3_lane",
                        "private_barrier_stage3",
                    ],
                    "anchor_caps": ["account_lookup", "roaming_diagnosis"],
                    "support_caps": ["line_resolution"],
                },
            ],
            "stage3": [
                {
                    "role": "safe_core_network_core",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage3_core",
                    "allowed_next_labels": ["public_stage4_core"],
                    "anchor_caps": ["network_diagnosis", "permission_diagnosis"],
                    "support_caps": ["verification", "repair_execution"],
                },
                {
                    "role": "safe_core_network_verify",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage3_verify",
                    "allowed_next_labels": ["public_stage4_core"],
                    "anchor_caps": ["network_diagnosis", "verification"],
                    "support_caps": ["permission_diagnosis", "apn_diagnosis"],
                },
                {
                    "role": "mixed_shared_edge_diagnosis",
                    "g": 0,
                    "node_semantic": "mixed_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage3_edge",
                    "allowed_next_labels": ["public_stage4_verify", "mixed_stage4_lane"],
                    "anchor_caps": ["apn_diagnosis", "verification"],
                    "anchor_boost": 0.07,
                    "support_caps": ["network_diagnosis", "permission_diagnosis"],
                },
                {
                    "role": "private_edge_config_lane",
                    "g": 0,
                    "node_semantic": "private_edge",
                    "profile_kind": "shared_basin",
                    "route_label": "mixed_stage3_lane",
                    "allowed_next_labels": [
                        "public_stage4_verify",
                        "mixed_stage4_lane",
                        "private_stage4_lane",
                        "private_barrier_stage4",
                    ],
                    "anchor_caps": ["network_diagnosis", "apn_diagnosis", "roaming_diagnosis"],
                    "anchor_boost": 0.12,
                    "support_caps": ["verification"],
                },
                {
                    "role": "private_barrier_config_gate",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "private_barrier_stage3",
                    "allowed_next_labels": [
                        "public_stage4_verify",
                        "mixed_stage4_lane",
                        "private_stage4_lane",
                        "private_barrier_stage4",
                    ],
                    "anchor_caps": ["network_diagnosis", "apn_diagnosis", "roaming_diagnosis"],
                    "anchor_boost": 0.09,
                    "support_caps": ["verification"],
                },
            ],
            "stage4": [
                {
                    "role": "safe_core_repair_core",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage4_core",
                    "allowed_next_labels": ["public_stage5_verify", "public_stage5_decision"],
                    "anchor_caps": ["repair_execution", "permission_diagnosis"],
                    "anchor_boost": 0.0,
                    "support_caps": ["verification", "terminal_decision"],
                },
                {
                    "role": "safe_core_repair_verify",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage4_verify",
                    "allowed_next_labels": ["public_stage5_verify", "public_stage5_decision"],
                    "anchor_caps": ["repair_execution", "verification"],
                    "anchor_boost": 0.03,
                    "support_caps": ["permission_diagnosis", "apn_diagnosis", "terminal_decision"],
                },
                {
                    "role": "mixed_shared_repair_escalation",
                    "g": 0,
                    "node_semantic": "mixed_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "mixed_stage4_lane",
                    "allowed_next_labels": [
                        "public_stage5_verify",
                        "public_stage5_decision",
                        "mixed_stage5_transfer",
                        "private_stage5_edge",
                    ],
                    "anchor_caps": ["repair_execution", "terminal_decision"],
                    "anchor_boost": 0.09,
                    "support_caps": ["verification"],
                },
                {
                    "role": "private_edge_repair_lane",
                    "g": 0,
                    "node_semantic": "private_edge",
                    "profile_kind": "shared_basin",
                    "route_label": "private_stage4_lane",
                    "allowed_next_labels": [
                        "public_stage5_verify",
                        "mixed_stage5_transfer",
                        "private_stage5_edge",
                        "private_stage5_leaf",
                    ],
                    "anchor_caps": ["repair_execution", "apn_diagnosis"],
                    "anchor_boost": 0.08,
                    "support_caps": ["verification", "terminal_decision", "roaming_diagnosis"],
                },
                {
                    "role": "private_barrier_edge_repair",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "private_barrier_stage4",
                    "allowed_next_labels": [
                        "public_stage5_decision",
                        "private_stage5_edge",
                        "private_stage5_leaf",
                    ],
                    "anchor_caps": ["repair_execution", "terminal_decision", "apn_diagnosis"],
                    "anchor_boost": 0.09,
                    "support_caps": ["verification", "terminal_decision", "roaming_diagnosis"],
                },
            ],
            "stage5": [
                {
                    "role": "safe_core_verify_core",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage5_verify",
                    "anchor_caps": ["verification", "repair_execution"],
                    "anchor_boost": 0.02,
                    "support_caps": ["terminal_decision"],
                },
                {
                    "role": "safe_core_decision_core",
                    "g": 0,
                    "node_semantic": "safe_core",
                    "profile_kind": "shared_basin",
                    "route_label": "public_stage5_decision",
                    "anchor_caps": ["terminal_decision", "verification"],
                    "anchor_boost": 0.02,
                    "support_caps": ["repair_execution"],
                },
                {
                    "role": "mixed_shared_transfer_ready",
                    "g": 0,
                    "node_semantic": "mixed_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "mixed_stage5_transfer",
                    "anchor_caps": ["repair_execution", "terminal_decision", "verification"],
                    "anchor_boost": 0.07,
                    "support_caps": ["verification"],
                },
                {
                    "role": "private_edge_resolution",
                    "g": 0,
                    "node_semantic": "private_edge",
                    "profile_kind": "shared_basin",
                    "route_label": "private_stage5_edge",
                    "anchor_caps": ["repair_execution", "verification"],
                    "anchor_boost": 0.07,
                    "support_caps": ["terminal_decision"],
                },
                {
                    "role": "private_leaf_transfer_edge",
                    "g": 1,
                    "node_semantic": "edge_specialist",
                    "profile_kind": "specialist",
                    "route_label": "private_stage5_leaf",
                    "anchor_caps": ["verification", "terminal_decision"],
                    "anchor_boost": 0.14,
                    "support_caps": ["repair_execution"],
                },
            ],
        },
    }


def build_shared_basin_strong_prefix_dedup_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_family_spec())
    config["family_name"] = "shared_basin_strong_prefix_dedup"
    config["generation_mode"] = "capability_shared_basin_prefix_dedup"
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_SPEC_PATH
    )
    config["source_family_name"] = "shared_basin_strong"
    return config


def build_shared_basin_strong_prefix_dedup_profile_switch_family_spec() -> dict:
    return {
        "family_name": "shared_basin_strong_prefix_dedup_profile_switch",
        "generation_mode": "capability_shared_basin_prefix_dedup",
        "stages": ["stage1", "stage2", "stage3", "stage4", "stage5"],
        "profile_fields": {
            "competence_level": "capability",
            "scope_level": "capability",
            "stability_level": "capability",
        },
        "cost_ranges": {
            "uniform": (0.12, 0.16),
        },
        "semantic_skill_ranges": {
            "general_shared": {
                "anchor": (0.76, 0.86),
                "support": (0.46, 0.60),
                "background": (0.16, 0.28),
                "focus_fallback": (0.34, 0.48),
            },
            "trap_lane": {
                "anchor": (0.86, 0.94),
                "support": (0.42, 0.56),
                "background": (0.05, 0.16),
                "focus_fallback": (0.18, 0.32),
            },
            "target_specialist": {
                "anchor": (0.90, 0.98),
                "support": (0.28, 0.42),
                "background": (0.03, 0.10),
                "focus_fallback": (0.10, 0.22),
            },
            "private_barrier": {
                "anchor": (0.86, 0.94),
                "support": (0.34, 0.50),
                "background": (0.08, 0.18),
                "focus_fallback": (0.28, 0.42),
            },
        },
        "stage_profiles": {
            "stage1": [
                {
                    "role": "general_shared_context_intake",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage1_intake",
                    "anchor_caps": ["user_grounding", "account_lookup"],
                    "support_caps": ["line_resolution", "verification"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.112, 0.132),
                },
                {
                    "role": "general_shared_grounded_verify",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage1_verify",
                    "anchor_caps": ["user_grounding", "verification"],
                    "support_caps": ["account_lookup", "line_resolution"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.116, 0.136),
                },
                {
                    "role": "target_specialist_handoff",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "target_stage1_handoff",
                    "anchor_caps": ["user_grounding", "account_lookup"],
                    "support_caps": ["verification"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.120, 0.145),
                },
                {
                    "role": "trap_lane_fast_intake",
                    "g": 0,
                    "node_semantic": "trap_lane",
                    "profile_kind": "shared_basin",
                    "route_label": "trap_stage1_intake",
                    "anchor_caps": ["user_grounding", "account_lookup", "line_resolution"],
                    "anchor_boost": 0.03,
                    "support_caps": ["permission_diagnosis"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.090, 0.110),
                },
                {
                    "role": "barrier_gate_intake",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "barrier_stage1_gate",
                    "anchor_caps": ["user_grounding", "account_lookup"],
                    "support_caps": ["verification"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.145, 0.170),
                },
            ],
            "stage2": [
                {
                    "role": "target_specialist_router",
                    "g": 0,
                    "node_semantic": "target_specialist",
                    "profile_kind": "specialist",
                    "route_label": "target_stage2_router",
                    "anchor_caps": ["line_resolution", "roaming_diagnosis"],
                    "support_caps": ["account_lookup", "verification"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.140, 0.160),
                },
                {
                    "role": "general_shared_core",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage2_core",
                    "anchor_caps": ["line_resolution", "account_lookup"],
                    "support_caps": ["roaming_diagnosis"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.120, 0.140),
                },
                {
                    "role": "trap_lane_router",
                    "g": 0,
                    "node_semantic": "trap_lane",
                    "profile_kind": "shared_basin",
                    "route_label": "trap_stage2_router",
                    "anchor_caps": ["line_resolution", "account_lookup"],
                    "support_caps": ["network_diagnosis"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.095, 0.115),
                },
                {
                    "role": "general_shared_roaming_lane",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage2_roaming",
                    "anchor_caps": ["roaming_diagnosis", "line_resolution"],
                    "support_caps": ["account_lookup", "verification"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.125, 0.145),
                },
                {
                    "role": "barrier_stage2_gate",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "barrier_stage2_gate",
                    "anchor_caps": ["roaming_diagnosis", "line_resolution"],
                    "support_caps": ["account_lookup"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.150, 0.180),
                },
            ],
            "stage3": [
                {
                    "role": "target_specialist_apn",
                    "g": 0,
                    "node_semantic": "target_specialist",
                    "profile_kind": "specialist",
                    "route_label": "target_stage3_apn",
                    "anchor_caps": ["apn_diagnosis", "roaming_diagnosis"],
                    "support_caps": ["network_diagnosis", "verification"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.145, 0.175),
                },
                {
                    "role": "target_specialist_roaming",
                    "g": 0,
                    "node_semantic": "target_specialist",
                    "profile_kind": "specialist",
                    "route_label": "target_stage3_roaming",
                    "anchor_caps": ["apn_diagnosis", "roaming_diagnosis"],
                    "support_caps": ["network_diagnosis", "verification"],
                    "anchor_boost": 0.02,
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.145, 0.175),
                },
                {
                    "role": "general_shared_network_core",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage3_network",
                    "anchor_caps": ["network_diagnosis", "permission_diagnosis"],
                    "support_caps": ["verification"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.125, 0.145),
                },
                {
                    "role": "trap_lane_network_fast",
                    "g": 0,
                    "node_semantic": "trap_lane",
                    "profile_kind": "shared_basin",
                    "route_label": "trap_stage3_network",
                    "anchor_caps": ["network_diagnosis", "permission_diagnosis"],
                    "support_caps": ["line_resolution"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.100, 0.120),
                },
                {
                    "role": "barrier_stage3_gate",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "barrier_stage3_gate",
                    "anchor_caps": ["network_diagnosis", "apn_diagnosis", "roaming_diagnosis"],
                    "support_caps": ["verification"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.155, 0.190),
                },
            ],
            "stage4": [
                {
                    "role": "target_specialist_repair",
                    "g": 0,
                    "node_semantic": "target_specialist",
                    "profile_kind": "specialist",
                    "route_label": "target_stage4_repair",
                    "anchor_caps": ["repair_execution", "apn_diagnosis"],
                    "anchor_boost": 0.03,
                    "support_caps": ["verification", "roaming_diagnosis", "terminal_decision"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.150, 0.180),
                },
                {
                    "role": "general_shared_repair",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage4_repair",
                    "anchor_caps": ["repair_execution", "network_diagnosis"],
                    "support_caps": ["verification", "terminal_decision"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.130, 0.150),
                },
                {
                    "role": "general_shared_verify_decide",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage4_verify",
                    "anchor_caps": ["repair_execution", "verification", "terminal_decision"],
                    "support_caps": ["permission_diagnosis", "apn_diagnosis"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.130, 0.150),
                },
                {
                    "role": "trap_lane_execute_fast",
                    "g": 0,
                    "node_semantic": "trap_lane",
                    "profile_kind": "shared_basin",
                    "route_label": "trap_stage4_execute",
                    "anchor_caps": ["network_diagnosis", "permission_diagnosis"],
                    "support_caps": ["terminal_decision", "line_resolution", "repair_execution"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.098, 0.118),
                },
                {
                    "role": "barrier_stage4_gate",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "barrier_stage4_gate",
                    "anchor_caps": ["repair_execution", "terminal_decision", "apn_diagnosis"],
                    "support_caps": ["verification", "roaming_diagnosis"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.160, 0.200),
                },
            ],
            "stage5": [
                {
                    "role": "target_specialist_verify",
                    "g": 0,
                    "node_semantic": "target_specialist",
                    "profile_kind": "specialist",
                    "route_label": "target_stage5_verify",
                    "anchor_caps": ["verification", "repair_execution"],
                    "support_caps": ["terminal_decision"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.145, 0.175),
                },
                {
                    "role": "target_specialist_decision",
                    "g": 0,
                    "node_semantic": "target_specialist",
                    "profile_kind": "specialist",
                    "route_label": "target_stage5_decision",
                    "anchor_caps": ["terminal_decision", "verification"],
                    "support_caps": ["repair_execution"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.145, 0.175),
                },
                {
                    "role": "general_shared_verify",
                    "g": 0,
                    "node_semantic": "general_shared",
                    "profile_kind": "shared_basin",
                    "route_label": "general_stage5_verify",
                    "anchor_caps": ["verification", "repair_execution"],
                    "support_caps": ["terminal_decision"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.125, 0.145),
                },
                {
                    "role": "trap_lane_terminal",
                    "g": 0,
                    "node_semantic": "trap_lane",
                    "profile_kind": "shared_basin",
                    "route_label": "trap_stage5_terminal",
                    "anchor_caps": ["terminal_decision"],
                    "support_caps": ["repair_execution", "verification"],
                    "deliberation_mode": "fast",
                    "base_cost_range": (0.098, 0.118),
                },
                {
                    "role": "barrier_stage5_transfer",
                    "g": 1,
                    "node_semantic": "private_barrier",
                    "profile_kind": "barrier",
                    "route_label": "barrier_stage5_transfer",
                    "anchor_caps": ["terminal_decision", "verification"],
                    "support_caps": ["repair_execution"],
                    "deliberation_mode": "deep",
                    "base_cost_range": (0.160, 0.200),
                },
            ],
        },
        "prefix_dedup_topology_spec_path": str(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_SPEC_PATH
        ),
        "source_family_name": "shared_basin_strong_prefix_dedup",
    }


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v1_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v1"
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V1_SPEC_PATH
    )
    config["source_family_name"] = "shared_basin_strong_prefix_dedup_profile_switch"
    return config


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5"
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V2_NEUTRAL_4OF5_SPEC_PATH
    )
    config["source_family_name"] = "shared_basin_strong_prefix_dedup_profile_switch"
    return config


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_4OF5_SPEC_PATH
    )
    config["source_family_name"] = "shared_basin_strong_prefix_dedup_profile_switch"
    return config


def _build_v3_efficient_anchor_gonly_family_spec(
    *,
    family_name: str,
    topology_spec_path: Path,
) -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = family_name
    config["prefix_dedup_topology_spec_path"] = str(topology_spec_path)
    config["source_family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"
    )
    return config


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_share_family_spec() -> dict:
    return _build_v3_efficient_anchor_gonly_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_share"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_ALL_SHARE_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_2of5_family_spec() -> dict:
    return _build_v3_efficient_anchor_gonly_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_2of5"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_2OF5_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_unshare_family_spec() -> dict:
    return _build_v3_efficient_anchor_gonly_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_all_unshare"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V3_EFFICIENT_ANCHOR_ALL_UNSHARE_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
    )
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_SPEC_PATH
    )
    config["source_family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"
    )

    # In the v4 compact binary tree, conceptual s3_n4 is a fastish wrapper inside
    # the target basin, not a terminal trap marker. Keep the fast deliberation
    # pressure, but make route/semantic compatible with target-safe path logic.
    stage3_n4 = config["stage_profiles"]["stage3"][3]
    stage3_n4.update(
        {
            "role": "general_shared_network_fast_wrapper",
            "node_semantic": "general_shared",
            "profile_kind": "shared_basin",
            "route_label": "general_stage3_network",
            "deliberation_mode": "fast",
            "base_cost_range": (0.105, 0.125),
        }
    )
    return config


def _build_v4_binary_mixed_stage45_share_variant_family_spec(
    *,
    family_name: str,
    topology_spec_path: Path,
) -> dict:
    config = build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_family_spec()
    config["family_name"] = family_name
    config["prefix_dedup_topology_spec_path"] = str(topology_spec_path)
    config["source_family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45"
    )
    return config


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share_family_spec() -> dict:
    return _build_v4_binary_mixed_stage45_share_variant_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_share"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_ALL_SHARE_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_4of5_family_spec() -> dict:
    return _build_v4_binary_mixed_stage45_share_variant_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_4of5"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_4OF5_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_2of5_family_spec() -> dict:
    return _build_v4_binary_mixed_stage45_share_variant_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_2of5"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_2OF5_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_unshare_family_spec() -> dict:
    return _build_v4_binary_mixed_stage45_share_variant_family_spec(
        family_name=(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_all_unshare"
        ),
        topology_spec_path=(
            SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V4_BINARY_MIXED_STAGE45_ALL_UNSHARE_SPEC_PATH
        ),
    )


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v5_small20_4of5"
    )
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V5_SMALL20_4OF5_SPEC_PATH
    )
    config["source_family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"
    )
    return config


def build_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v6_small30_4of5_family_spec() -> dict:
    config = deepcopy(build_shared_basin_strong_prefix_dedup_profile_switch_family_spec())
    config["family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v6_small30_4of5"
    )
    config["prefix_dedup_topology_spec_path"] = str(
        SHARED_BASIN_PREFIX_DEDUP_PROFILE_SWITCH_TRAP_ASYM_V6_SMALL30_4OF5_SPEC_PATH
    )
    config["source_family_name"] = (
        "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5"
    )
    return config


def _build_shared_basin_strong_gonly_variant(
    *,
    family_name: str,
    g_for_index: dict[int, int] | None = None,
    default_g: int | None = None,
) -> dict:
    config = deepcopy(build_shared_basin_strong_family_spec())
    config["family_name"] = family_name
    for stage_name in config["stages"]:
        for idx, profile in enumerate(config["stage_profiles"][stage_name]):
            if g_for_index is not None:
                profile["g"] = int(g_for_index[idx])
            elif default_g is not None:
                profile["g"] = int(default_g)
            else:
                raise ValueError("Either g_for_index or default_g must be provided.")
    return config


def build_shared_basin_strong_2of5_gonly_family_spec() -> dict:
    return _build_shared_basin_strong_gonly_variant(
        family_name="shared_basin_strong_2of5_gonly",
        g_for_index={0: 0, 1: 0, 2: 1, 3: 1, 4: 1},
    )


def build_shared_basin_strong_all_share_gonly_family_spec() -> dict:
    return _build_shared_basin_strong_gonly_variant(
        family_name="shared_basin_strong_all_share_gonly",
        default_g=0,
    )


def build_shared_basin_strong_all_unshare_gonly_family_spec() -> dict:
    return _build_shared_basin_strong_gonly_variant(
        family_name="shared_basin_strong_all_unshare_gonly",
        default_g=1,
    )
