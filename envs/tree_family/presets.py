"""Preset configurations for neutral / moderate / strong tree families."""

from __future__ import annotations


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
