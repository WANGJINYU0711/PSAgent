from __future__ import annotations

import argparse
import json
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = (
    ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time" / "tasks.json"
)
SOURCE_MANIFEST = (
    ROOT
    / "data"
    / "derived"
    / "telecom_mms_fixed_tree_base_v2_100_capabilities_time"
    / "manifest.json"
)
SOURCE_TREE_SPEC = (
    ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json"
)
TARGET_DATASET = (
    ROOT
    / "data"
    / "derived"
    / "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch"
    / "tasks.json"
)
TARGET_MANIFEST = TARGET_DATASET.with_name("manifest.json")
TARGET_TREE_SPEC = (
    ROOT
    / "analysis"
    / "tree_specs"
    / "shared_basin_strong_4of5_prefix_dedup_profile_switch.json"
)
TARGET_BUCKETS = (
    ROOT / "analysis" / "shared_basin_prefix_dedup_profile_switch_schedule_buckets.json"
)

CAPABILITY_NAMES = [
    "user_grounding",
    "account_lookup",
    "line_resolution",
    "network_diagnosis",
    "permission_diagnosis",
    "apn_diagnosis",
    "roaming_diagnosis",
    "repair_execution",
    "verification",
    "terminal_decision",
]
ROAMING_BLOCKERS = {
    "user_abroad_roaming_disabled_off",
    "user_abroad_roaming_disabled_on",
    "user_abroad_roaming_enabled_off",
}
SHALLOW_TRAP_BLOCKERS = {
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
    "data_mode_off",
    "data_usage_exceeded",
    "airplane_mode_on",
    "bad_network_preference",
    "bad_wifi_calling",
}
PROFILE_SWITCH_VERSION = "profile_switch_v1"

TRAP_PROFILE_DISPLAY_NAME = "fast_path_pre_switch"
TARGET_PROFILE_DISPLAY_NAME = "deep_path_post_switch"
SPECIALIST_PROFILE_DISPLAY_NAME = "verification_heavy_post_switch"

TRAP_PROFILE_NOTE = (
    "Pre-switch path: fast, shallow, and early-closure friendly. These tasks reward compact evidence chains and tolerate earlier stop decisions."
)
TARGET_PROFILE_NOTE = (
    "Post-switch path: deeper reasoning and repair-precondition completion matter more than route labels or archetypes. These tasks reward fuller evidence chains and cautious closure."
)
SPECIALIST_PROFILE_NOTE = (
    "Compatibility specialist subset: this now denotes the post-switch deep-validation/high-complexity subset, not an expert-agent archetype."
)

TRAP_STAGE_BEHAVIOR = {
    "stage1": "Fast grounding path: do the minimum identity and line grounding needed for downstream execution.",
    "stage2": "Fast resolution path: stabilize the customer and line quickly without speculative extra validation.",
    "stage3": "Shallow diagnosis path: validate only the highest-yield blocker families and leave long-tail branches unopened.",
    "stage4": "Fast repair path: act only on clear blocker evidence and avoid long precondition chains.",
    "stage5": "Fast closure path: use a compact verification floor and close quickly once a defensible local action is supported.",
}

TARGET_STAGE_BEHAVIOR = {
    "stage1": "Grounding remains compact, but the post-switch path starts preserving enough structure for later deep verification.",
    "stage2": "Deep path entry: establish a sturdier account and line evidence chain before diagnosis widens.",
    "stage3": "Deep diagnosis path: keep high-risk APN, roaming, SIM, and downstream blocker families open until decisive evidence resolves them.",
    "stage4": "Verification-heavy repair path: complete blocker-specific repair preconditions before mutation and prefer supported repair subsets over blanket escalation.",
    "stage5": "Cautious terminal path: verify decisive post-repair evidence, distinguish local evidence gaps from true external/manual blockers, and close only when the evidence chain supports the final action.",
}

SPECIALIST_STAGE_BEHAVIOR = {
    "stage1": "Compact grounding with slightly more caution before handing off into the high-complexity post-switch path.",
    "stage2": "High-complexity deep path: preserve the account and line evidence needed for later verification-heavy closure.",
    "stage3": "Verification-heavy diagnosis: keep multiple high-risk blocker families live until decisive evidence collapses them.",
    "stage4": "Deep-validation repair path: spend budget on completing repair preconditions and blocker-specific evidence chains before mutating the env.",
    "stage5": "Evidence-chain-required closure: verify the post-repair state thoroughly, prefer supported repair_all/repair_subset outcomes, and use transfer only for explicit non-local/manual blockers or verified local impossibility.",
}

PROFILE_METADATA = {
    "trap_pre_switch": {
        "display_name": TRAP_PROFILE_DISPLAY_NAME,
        "bucket_display": "fast_path_favoring",
        "profile_note": TRAP_PROFILE_NOTE,
        "bucket_note": "Fast/shallow/early-closure-friendly pre-switch bucket.",
        "stage_behavior": TRAP_STAGE_BEHAVIOR,
    },
    "target_post_switch": {
        "display_name": TARGET_PROFILE_DISPLAY_NAME,
        "bucket_display": "deep_path_favoring",
        "profile_note": TARGET_PROFILE_NOTE,
        "bucket_note": "Post-switch deep-path bucket where deeper reasoning, fuller verification, and cautious closure should pay off.",
        "stage_behavior": TARGET_STAGE_BEHAVIOR,
    },
    "target_specialist_post_switch": {
        "display_name": SPECIALIST_PROFILE_DISPLAY_NAME,
        "bucket_display": "deep_validation_subset",
        "profile_note": SPECIALIST_PROFILE_NOTE,
        "bucket_note": "Compatibility specialist bucket for the post-switch deep-validation/high-complexity subset.",
        "stage_behavior": SPECIALIST_STAGE_BEHAVIOR,
    },
    "unchanged_source_profile": {
        "display_name": "unchanged_source_profile",
        "bucket_display": "neutral_other",
        "profile_note": "No profile-switch override is applied to this task.",
        "bucket_note": "Neutral bucket outside the profile-switch selection.",
        "stage_behavior": {},
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def parse_blockers(task_id: str) -> list[str]:
    core = task_id
    if core.startswith("[mms_issue]"):
        core = core[len("[mms_issue]") :]
    core = core.split("[PERSONA:", 1)[0]
    return [] if not core else core.split("|")


def parse_persona(task_id: str) -> str:
    if "[PERSONA:" not in task_id:
        return "None"
    return task_id.split("[PERSONA:", 1)[1].rstrip("]") or "None"


def normalized_caps(values: dict[str, float]) -> dict[str, float]:
    result = {name: 0.0 for name in CAPABILITY_NAMES}
    for name, value in values.items():
        if name in result:
            result[name] = round(float(value), 3)
    return result


TRAP_STAGE_PROFILE = {
    "stage1": {
        "capability_requirements": normalized_caps(
            {
                "user_grounding": 0.90,
                "account_lookup": 0.74,
                "line_resolution": 0.28,
                "verification": 0.08,
            }
        ),
        "deliberation_requirement": "fast",
    },
    "stage2": {
        "capability_requirements": normalized_caps(
            {
                "account_lookup": 0.72,
                "line_resolution": 0.90,
                "roaming_diagnosis": 0.08,
            }
        ),
        "deliberation_requirement": "fast",
    },
    "stage3": {
        "capability_requirements": normalized_caps(
            {
                "network_diagnosis": 0.86,
                "permission_diagnosis": 0.76,
                "apn_diagnosis": 0.04,
                "roaming_diagnosis": 0.04,
                "verification": 0.04,
            }
        ),
        "deliberation_requirement": "fast",
    },
    "stage4": {
        "capability_requirements": normalized_caps(
            {
                "repair_execution": 0.34,
                "verification": 0.08,
                "terminal_decision": 0.18,
            }
        ),
        "deliberation_requirement": "fast",
    },
    "stage5": {
        "capability_requirements": normalized_caps(
            {
                "repair_execution": 0.18,
                "verification": 0.22,
                "terminal_decision": 0.48,
            }
        ),
        "deliberation_requirement": "fast",
    },
}

TARGET_STAGE_PROFILE = {
    "stage1": {
        "capability_requirements": normalized_caps(
            {
                "user_grounding": 0.46,
                "account_lookup": 0.46,
                "verification": 0.18,
            }
        ),
        "deliberation_requirement": "fast",
    },
    "stage2": {
        "capability_requirements": normalized_caps(
            {
                "account_lookup": 0.34,
                "line_resolution": 0.42,
                "roaming_diagnosis": 0.46,
                "verification": 0.28,
            }
        ),
        "deliberation_requirement": "deep",
    },
    "stage3": {
        "capability_requirements": normalized_caps(
            {
                "network_diagnosis": 0.46,
                "permission_diagnosis": 0.18,
                "apn_diagnosis": 0.68,
                "roaming_diagnosis": 0.62,
                "verification": 0.42,
                "terminal_decision": 0.08,
            }
        ),
        "deliberation_requirement": "deep",
    },
    "stage4": {
        "capability_requirements": normalized_caps(
            {
                "apn_diagnosis": 0.24,
                "roaming_diagnosis": 0.18,
                "repair_execution": 0.88,
                "verification": 0.78,
                "terminal_decision": 0.42,
            }
        ),
        "deliberation_requirement": "deep",
    },
    "stage5": {
        "capability_requirements": normalized_caps(
            {
                "repair_execution": 0.24,
                "verification": 0.96,
                "terminal_decision": 0.9,
            }
        ),
        "deliberation_requirement": "deep",
    },
}

SPECIALIST_STAGE_PROFILE = {
    "stage1": {
        "capability_requirements": normalized_caps(
            {
                "user_grounding": 0.42,
                "account_lookup": 0.42,
                "verification": 0.22,
            }
        ),
        "deliberation_requirement": "fast",
    },
    "stage2": {
        "capability_requirements": normalized_caps(
            {
                "account_lookup": 0.30,
                "line_resolution": 0.40,
                "roaming_diagnosis": 0.48,
                "verification": 0.34,
            }
        ),
        "deliberation_requirement": "deep",
    },
    "stage3": {
        "capability_requirements": normalized_caps(
            {
                "network_diagnosis": 0.36,
                "permission_diagnosis": 0.14,
                "apn_diagnosis": 0.64,
                "roaming_diagnosis": 0.58,
                "verification": 0.54,
                "terminal_decision": 0.12,
            }
        ),
        "deliberation_requirement": "deep",
    },
    "stage4": {
        "capability_requirements": normalized_caps(
            {
                "apn_diagnosis": 0.22,
                "roaming_diagnosis": 0.20,
                "repair_execution": 0.82,
                "verification": 0.92,
                "terminal_decision": 0.58,
            }
        ),
        "deliberation_requirement": "deep",
    },
    "stage5": {
        "capability_requirements": normalized_caps(
            {
                "repair_execution": 0.22,
                "verification": 0.98,
                "terminal_decision": 0.96,
            }
        ),
        "deliberation_requirement": "deep",
    },
}


def build_descriptor(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row["original_task_id"])
    blockers = parse_blockers(task_id)
    blocker_set = set(blockers)
    return {
        "task_id": task_id,
        "num_blockers": len(blockers),
        "blockers": blockers,
        "blocker_set": blocker_set,
        "persona": parse_persona(task_id),
        "has_apn": "break_apn_mms_setting" in blocker_set,
        "has_sim": "unseat_sim_card" in blocker_set,
        "roaming_blockers": sorted(blocker_set & ROAMING_BLOCKERS),
        "is_trap_candidate": (
            1 <= len(blockers) <= 3
            and "break_apn_mms_setting" not in blocker_set
            and "unseat_sim_card" not in blocker_set
            and not (blocker_set & ROAMING_BLOCKERS)
            and blocker_set <= SHALLOW_TRAP_BLOCKERS
        ),
        "is_target_candidate": (
            len(blockers) >= 4
            and "break_apn_mms_setting" in blocker_set
            and ("unseat_sim_card" in blocker_set or bool(blocker_set & ROAMING_BLOCKERS))
        ),
    }


def candidate_features(descriptor: dict[str, Any]) -> set[str]:
    features = {
        f"num_blockers:{descriptor['num_blockers']}",
        f"persona:{descriptor['persona']}",
        f"has_sim:{descriptor['has_sim']}",
        f"has_roaming:{bool(descriptor['roaming_blockers'])}",
    }
    for blocker in descriptor["blockers"]:
        features.add(f"blocker:{blocker}")
    for blocker in descriptor["roaming_blockers"]:
        features.add(f"roaming:{blocker}")
    return features


def select_target_bucket(candidates: list[dict[str, Any]], size: int) -> list[dict[str, Any]]:
    if len(candidates) < size:
        raise ValueError(f"Need at least {size} target candidates, got {len(candidates)}.")
    remaining = sorted(candidates, key=lambda row: row["task_id"])
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    while len(selected) < size:
        best = max(
            remaining,
            key=lambda row: (
                len(candidate_features(row) - covered),
                row["num_blockers"],
                int(row["has_sim"]),
                len(row["roaming_blockers"]),
                row["task_id"],
            ),
        )
        selected.append(best)
        covered |= candidate_features(best)
        remaining.remove(best)
    return sorted(selected, key=lambda row: row["task_id"])


def select_specialist_subset(target_bucket: list[dict[str, Any]]) -> list[dict[str, Any]]:
    specialist = [
        row
        for row in target_bucket
        if row["num_blockers"] >= 8 and (row["has_sim"] or row["roaming_blockers"])
    ]
    if len(specialist) >= len(target_bucket):
        specialist = [
            row
            for row in target_bucket
            if row["num_blockers"] >= 8 and row["has_sim"] and row["roaming_blockers"]
        ]
    if not specialist:
        specialist = [
            row
            for row in target_bucket
            if row["num_blockers"] >= 7 and row["has_sim"] and row["roaming_blockers"]
        ]
    if not specialist:
        raise ValueError("Failed to derive a non-empty specialist subset from target bucket.")
    if len(specialist) >= len(target_bucket):
        raise ValueError("Specialist subset must be a strict subset of selected target bucket.")
    return sorted(specialist, key=lambda row: row["task_id"])


def summarize_bucket(rows: list[dict[str, Any]]) -> dict[str, Any]:
    blocker_counter: Counter[str] = Counter()
    persona_counter: Counter[str] = Counter()
    blocker_count_counter: Counter[int] = Counter()
    roaming_counter: Counter[str] = Counter()
    sim_count = 0
    for row in rows:
        blocker_counter.update(row["blockers"])
        persona_counter[row["persona"]] += 1
        blocker_count_counter[row["num_blockers"]] += 1
        roaming_counter.update(row["roaming_blockers"])
        sim_count += int(row["has_sim"])
    return {
        "task_count": len(rows),
        "blocker_count_distribution": {str(key): blocker_count_counter[key] for key in sorted(blocker_count_counter)},
        "persona_distribution": dict(sorted(persona_counter.items())),
        "has_sim_count": sim_count,
        "has_roaming_count": sum(1 for row in rows if row["roaming_blockers"]),
        "blocker_frequency": dict(sorted(blocker_counter.items())),
        "roaming_blocker_frequency": dict(sorted(roaming_counter.items())),
    }


def apply_stage_profile(row: dict[str, Any], profile_name: str, profile: dict[str, dict[str, Any]]) -> dict[str, Any]:
    row_copy = deepcopy(row)
    metadata = row_copy.setdefault("metadata", {})
    profile_meta = PROFILE_METADATA[profile_name]
    metadata["profile_switch_version"] = PROFILE_SWITCH_VERSION
    metadata["profile_switch_profile"] = profile_name
    metadata["profile_switch_profile_display"] = profile_meta["display_name"]
    metadata["profile_switch_profile_note"] = profile_meta["profile_note"]
    metadata["profile_switch_bucket_display"] = profile_meta["bucket_display"]
    metadata["profile_switch_bucket_note"] = profile_meta["bucket_note"]
    metadata["profile_switch_blockers"] = parse_blockers(str(row_copy["original_task_id"]))
    metadata["deliberation_requirement_summary"] = {
        stage_name: stage_profile["deliberation_requirement"]
        for stage_name, stage_profile in profile.items()
    }
    metadata["profile_switch_stage_behavior"] = dict(profile_meta["stage_behavior"])
    for stage_name, stage_profile in profile.items():
        stage_payload = row_copy.get(stage_name, {})
        if not isinstance(stage_payload, dict):
            continue
        stage_payload["capability_requirements"] = dict(stage_profile["capability_requirements"])
        stage_payload["deliberation_requirement"] = stage_profile["deliberation_requirement"]
        stage_payload["stage_behavior_note"] = profile_meta["stage_behavior"][stage_name]
    return row_copy


def build_dataset(
    rows: list[dict[str, Any]],
    trap_ids: set[str],
    target_ids: set[str],
    specialist_ids: set[str],
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        task_id = str(row["original_task_id"])
        if task_id in specialist_ids:
            updated = apply_stage_profile(
                row,
                "target_specialist_post_switch",
                SPECIALIST_STAGE_PROFILE,
            )
            updated["metadata"]["profile_switch_bucket"] = "specialist_target_favoring"
        elif task_id in target_ids:
            updated = apply_stage_profile(
                row,
                "target_post_switch",
                TARGET_STAGE_PROFILE,
            )
            updated["metadata"]["profile_switch_bucket"] = "target_favoring"
        elif task_id in trap_ids:
            updated = apply_stage_profile(
                row,
                "trap_pre_switch",
                TRAP_STAGE_PROFILE,
            )
            updated["metadata"]["profile_switch_bucket"] = "trap_favoring"
        else:
            updated = deepcopy(row)
            metadata = updated.setdefault("metadata", {})
            metadata["profile_switch_version"] = PROFILE_SWITCH_VERSION
            metadata["profile_switch_profile"] = "unchanged_source_profile"
            metadata["profile_switch_profile_display"] = PROFILE_METADATA["unchanged_source_profile"]["display_name"]
            metadata["profile_switch_profile_note"] = PROFILE_METADATA["unchanged_source_profile"]["profile_note"]
            metadata["profile_switch_bucket"] = "neutral_other"
            metadata["profile_switch_bucket_display"] = PROFILE_METADATA["unchanged_source_profile"]["bucket_display"]
            metadata["profile_switch_bucket_note"] = PROFILE_METADATA["unchanged_source_profile"]["bucket_note"]
            metadata["profile_switch_blockers"] = parse_blockers(task_id)
        output.append(updated)
    return output


def build_manifest(
    source_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    trap_bucket: list[dict[str, Any]],
    target_bucket: list[dict[str, Any]],
    specialist_bucket: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "subset_name": "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch",
        "subset_version": PROFILE_SWITCH_VERSION,
        "family": "telecom_mms_recovery",
        "task_count": len(rows),
        "source_dataset": str(SOURCE_DATASET),
        "source_manifest": str(SOURCE_MANIFEST),
        "notes": [
            "Derived from telecom_mms_fixed_tree_base_v2_100_capabilities_time.",
            "Profile-switch is produced by task requirement profile shift only.",
            "No runtime path-label archetype shaping is used by the new family.",
            "Pre-switch tasks are fast/shallow/early-closure friendly; post-switch tasks require deeper reasoning, fuller verification, and more cautious closure.",
            "The compatibility specialist subset now denotes the post-switch deep-validation/high-complexity subset, not an expert-agent persona.",
            "Only task-intrinsic blocker rules decide trap/target/specialist buckets.",
        ],
        "selection_criteria": {
            "trap_favoring": {
                "display_name": TRAP_PROFILE_DISPLAY_NAME,
                "behavior_note": TRAP_PROFILE_NOTE,
                "num_blockers": "1-3",
                "must_exclude": [
                    "break_apn_mms_setting",
                    "unseat_sim_card",
                    *sorted(ROAMING_BLOCKERS),
                ],
                "allowed_blockers_subset": sorted(SHALLOW_TRAP_BLOCKERS),
            },
            "target_favoring": {
                "display_name": TARGET_PROFILE_DISPLAY_NAME,
                "behavior_note": TARGET_PROFILE_NOTE,
                "num_blockers": ">=4",
                "must_include": ["break_apn_mms_setting"],
                "must_also_include_one_of": ["unseat_sim_card", "any_roaming_blocker"],
            },
            "specialist": {
                "display_name": SPECIALIST_PROFILE_DISPLAY_NAME,
                "behavior_note": SPECIALIST_PROFILE_NOTE,
                "strict_subset_of": "target_favoring_task_ids",
                "selected_from_target_bucket": "num_blockers >= 8 and (has_sim or has_roaming)",
                "compatibility_semantics": "Compatibility label only: this subset marks the post-switch deep-validation/high-complexity cases.",
            },
        },
        "profile_summary": {
            "trap_profile": {
                "display_name": TRAP_PROFILE_DISPLAY_NAME,
                "behavior_note": TRAP_PROFILE_NOTE,
                "stages": {
                    stage: {
                        "deliberation_requirement": payload["deliberation_requirement"],
                        "capability_requirements": payload["capability_requirements"],
                        "stage_behavior_note": TRAP_STAGE_BEHAVIOR[stage],
                    }
                    for stage, payload in TRAP_STAGE_PROFILE.items()
                }
            },
            "target_profile": {
                "display_name": TARGET_PROFILE_DISPLAY_NAME,
                "behavior_note": TARGET_PROFILE_NOTE,
                "stages": {
                    stage: {
                        "deliberation_requirement": payload["deliberation_requirement"],
                        "capability_requirements": payload["capability_requirements"],
                        "stage_behavior_note": TARGET_STAGE_BEHAVIOR[stage],
                    }
                    for stage, payload in TARGET_STAGE_PROFILE.items()
                }
            },
            "specialist_profile": {
                "display_name": SPECIALIST_PROFILE_DISPLAY_NAME,
                "behavior_note": SPECIALIST_PROFILE_NOTE,
                "stages": {
                    stage: {
                        "deliberation_requirement": payload["deliberation_requirement"],
                        "capability_requirements": payload["capability_requirements"],
                        "stage_behavior_note": SPECIALIST_STAGE_BEHAVIOR[stage],
                    }
                    for stage, payload in SPECIALIST_STAGE_PROFILE.items()
                }
            },
        },
        "coverage_summary": {
            "trap_bucket": summarize_bucket(trap_bucket),
            "target_bucket": summarize_bucket(target_bucket),
            "specialist_bucket": summarize_bucket(specialist_bucket),
            "specialist_is_subset_of_target": all(
                row["task_id"] in {target["task_id"] for target in target_bucket}
                for row in specialist_bucket
            ),
        },
        "copied_manifest_metadata": source_manifest,
    }


def build_schedule_bucket_payload(
    trap_bucket: list[dict[str, Any]],
    target_bucket: list[dict[str, Any]],
    specialist_bucket: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "schema_version": PROFILE_SWITCH_VERSION,
        "source_dataset": str(TARGET_DATASET),
        "source_manifest": str(TARGET_MANIFEST),
        "selection_criteria": {
            "trap_favoring": "Pre-switch fast-path bucket: 1-3 blockers, excludes APN/SIM/roaming blockers, shallow blocker subset only.",
            "target_favoring": "Post-switch deep-path bucket: >=4 blockers, must include APN, and at least one SIM or roaming blocker.",
            "target_selection": "Greedy deterministic coverage over blocker count, persona, SIM/roaming, and blocker identities; target bucket size matches trap bucket size.",
            "specialist": "Compatibility label only: strict subset of the selected target bucket used as the post-switch deep-validation/high-complexity subset (num_blockers >= 8 and has_sim or has_roaming).",
        },
        "bucket_labels": {
            "trap_favoring": TRAP_PROFILE_DISPLAY_NAME,
            "target_favoring": TARGET_PROFILE_DISPLAY_NAME,
            "specialist": SPECIALIST_PROFILE_DISPLAY_NAME,
        },
        "bucket_notes": {
            "trap_favoring": TRAP_PROFILE_NOTE,
            "target_favoring": TARGET_PROFILE_NOTE,
            "specialist": SPECIALIST_PROFILE_NOTE,
        },
        "trap_favoring_task_ids": [row["task_id"] for row in trap_bucket],
        "target_favoring_task_ids": [row["task_id"] for row in target_bucket],
        "specialist_task_ids": [row["task_id"] for row in specialist_bucket],
        "coverage_summary": {
            "trap_bucket": summarize_bucket(trap_bucket),
            "target_bucket": summarize_bucket(target_bucket),
            "specialist_bucket": summarize_bucket(specialist_bucket),
            "specialist_is_subset_of_target": all(
                row["task_id"] in {target["task_id"] for target in target_bucket}
                for row in specialist_bucket
            ),
        },
    }


def build_tree_spec(source_spec: dict[str, Any]) -> dict[str, Any]:
    target_spec = deepcopy(source_spec)
    target_spec["tree_name"] = "shared_basin_strong_4of5_prefix_dedup_profile_switch"
    metadata = dict(target_spec.get("metadata", {}) or {})
    compatible_with = set(metadata.get("compatible_with", []) or [])
    compatible_with.add("run_shared_basin_repeated_smoke_setup")
    metadata.update(
        {
            "source_tree_name": source_spec.get("tree_name"),
            "profile_switch_version": PROFILE_SWITCH_VERSION,
            "compatible_with": sorted(compatible_with),
            "not_directly_compatible_with_current_shared_basin_llm_runner": False,
            "notes": [
                "Topology copied from shared_basin_strong_4of5_prefix_dedup.",
                "Profile-switch family changes agent capabilities, deliberation modes, and base costs only.",
                "Post-switch semantics are about deeper reasoning, verification burden, and closure caution rather than route-archetype shaping.",
                "No path-label runtime archetype shaping is encoded in this spec.",
            ],
        }
    )
    target_spec["metadata"] = metadata
    return target_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Build profile-switch prefix-dedup assets.")
    parser.add_argument("--source-dataset", type=Path, default=SOURCE_DATASET)
    parser.add_argument("--source-manifest", type=Path, default=SOURCE_MANIFEST)
    parser.add_argument("--source-tree-spec", type=Path, default=SOURCE_TREE_SPEC)
    parser.add_argument("--target-dataset", type=Path, default=TARGET_DATASET)
    parser.add_argument("--target-manifest", type=Path, default=TARGET_MANIFEST)
    parser.add_argument("--target-tree-spec", type=Path, default=TARGET_TREE_SPEC)
    parser.add_argument("--target-buckets", type=Path, default=TARGET_BUCKETS)
    args = parser.parse_args()

    source_rows = load_json(args.source_dataset)
    source_manifest = load_json(args.source_manifest)
    source_tree_spec = load_json(args.source_tree_spec)

    descriptors = [build_descriptor(row) for row in source_rows]
    trap_candidates = [row for row in descriptors if row["is_trap_candidate"]]
    target_candidates = [row for row in descriptors if row["is_target_candidate"]]
    trap_bucket = sorted(trap_candidates, key=lambda row: row["task_id"])
    target_bucket = select_target_bucket(target_candidates, len(trap_bucket))
    specialist_bucket = select_specialist_subset(target_bucket)

    trap_ids = {row["task_id"] for row in trap_bucket}
    target_ids = {row["task_id"] for row in target_bucket}
    specialist_ids = {row["task_id"] for row in specialist_bucket}

    dataset_rows = build_dataset(source_rows, trap_ids, target_ids, specialist_ids)
    manifest = build_manifest(
        source_manifest=source_manifest,
        rows=dataset_rows,
        trap_bucket=trap_bucket,
        target_bucket=target_bucket,
        specialist_bucket=specialist_bucket,
    )
    bucket_payload = build_schedule_bucket_payload(
        trap_bucket=trap_bucket,
        target_bucket=target_bucket,
        specialist_bucket=specialist_bucket,
    )
    tree_spec = build_tree_spec(source_tree_spec)

    write_json(args.target_dataset, dataset_rows)
    write_json(args.target_manifest, manifest)
    write_json(args.target_buckets, bucket_payload)
    write_json(args.target_tree_spec, tree_spec)

    print(
        json.dumps(
            {
                "target_dataset": str(args.target_dataset),
                "target_manifest": str(args.target_manifest),
                "target_buckets": str(args.target_buckets),
                "target_tree_spec": str(args.target_tree_spec),
                "trap_bucket_count": len(trap_bucket),
                "target_bucket_count": len(target_bucket),
                "specialist_bucket_count": len(specialist_bucket),
                "specialist_is_subset_of_target": specialist_ids < target_ids,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
