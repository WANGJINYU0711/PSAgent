from __future__ import annotations

import hashlib
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.telecom_mms_specs import CANONICAL_BLOCKER_SPECS, first_pass_terminal_decision  # noqa: E402
from scripts.build_shared_basin_profile_switch_assets import (  # noqa: E402
    CAPABILITY_NAMES,
    PROFILE_SWITCH_VERSION,
    build_dataset as apply_profile_switch_dataset,
)
from scripts.build_telecom_mms_capability_benchmark import add_capability_requirements  # noqa: E402
from scripts.build_telecom_mms_capability_time_benchmark import add_deliberation_requirements  # noqa: E402
from scripts.build_telecom_mms_fixed_tree import (  # noqa: E402
    TELECOM_DB_PATH,
    TELECOM_SPLITS_PATH,
    TELECOM_TASKS_PATH,
    build_dataset_with_stats,
    build_reference_maps,
    load_telecom_reference_db,
    parse_task_id,
)


OUTPUT_DIR = (
    REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_low_transfer"
)
OUTPUT_TASKS_PATH = OUTPUT_DIR / "tasks.json"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "manifest.json"
OUTPUT_BUCKETS_PATH = (
    REPO_ROOT / "analysis" / "shared_basin_prefix_dedup_profile_switch_low_transfer_schedule_buckets.json"
)
REFERENCE_DATASET_PATH = (
    REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch" / "tasks.json"
)

DATASET_NAME = "telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_low_transfer"
DATASET_VERSION = "profile_switch_low_transfer_v1"
TARGET_COUNTS = {"repair_all": 42, "repair_subset": 50, "transfer": 8}
TRAP_BUCKET_SIZE = 16
TARGET_REPAIR_SUBSET_BUCKET_SIZE = 10
TARGET_REPAIR_ALL_BUCKET_SIZE = 6
HARD_TRANSFER_CONTROL_COUNT = 8
SPECIALIST_MIN_BLOCKERS = 6

PERMISSION_BLOCKERS = {
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
}
TRAP_SHALLOW_BLOCKERS = {
    "airplane_mode_on",
    "bad_network_preference",
    "bad_wifi_calling",
    "break_app_both_permissions",
    "break_app_sms_permission",
    "break_app_storage_permission",
    "data_mode_off",
}
FOCUS_BLOCKER_FAMILIES = [
    "apn",
    "permission",
    "wifi_calling",
    "network_preference",
    "sim",
    "data_mode",
    "roaming_enabled_off",
    "data_usage_exceeded",
    "roaming_disabled_on",
]
PROFILE_SWITCH_CLASS_FAST = "fast_local_pre_switch"
PROFILE_SWITCH_CLASS_POST = "non_transfer_post_switch"
PROFILE_SWITCH_CLASS_HARD_TRANSFER = "hard_transfer_control"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_subsplit_lookup(split_map: dict[str, Any]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for task_id in split_map.get("train", []):
        lookup[task_id] = "train"
    for task_id in split_map.get("test", []):
        lookup[task_id] = "test"
    for task_id in split_map.get("base", []):
        lookup.setdefault(task_id, "base")
    return lookup


def blocker_count_bucket(blocker_count: int) -> str:
    return str(blocker_count) if blocker_count < 5 else "5+"


def blocker_family_tags(blockers: list[str]) -> list[str]:
    blocker_set = set(blockers)
    tags: list[str] = []
    if "break_apn_mms_setting" in blocker_set:
        tags.append("apn")
    if blocker_set & PERMISSION_BLOCKERS:
        tags.append("permission")
    if "bad_wifi_calling" in blocker_set:
        tags.append("wifi_calling")
    if "bad_network_preference" in blocker_set:
        tags.append("network_preference")
    if "unseat_sim_card" in blocker_set:
        tags.append("sim")
    if "data_mode_off" in blocker_set:
        tags.append("data_mode")
    if "user_abroad_roaming_enabled_off" in blocker_set:
        tags.append("roaming_enabled_off")
    if "data_usage_exceeded" in blocker_set:
        tags.append("data_usage_exceeded")
    if "user_abroad_roaming_disabled_on" in blocker_set:
        tags.append("roaming_disabled_on")
    if "user_abroad_roaming_disabled_off" in blocker_set:
        tags.append("hybrid_roaming_disabled_off")
    return tags


def build_candidate(raw_task: dict[str, Any], source_subsplit_lookup: dict[str, str]) -> dict[str, Any]:
    parsed = parse_task_id(raw_task["id"])
    blockers = list(parsed["blockers"])
    blocker_set = set(blockers)
    specs = [CANONICAL_BLOCKER_SPECS[blocker_id] for blocker_id in blockers]
    terminal = first_pass_terminal_decision(blockers)
    family_tags = blocker_family_tags(blockers)
    focus_core_tags = [tag for tag in family_tags if tag in FOCUS_BLOCKER_FAMILIES[:7]]
    assistant_deferred_tags = [
        tag for tag in family_tags if tag in {"data_usage_exceeded", "roaming_disabled_on"}
    ]
    return {
        "task_id": raw_task["id"],
        "raw_task": raw_task,
        "persona": parsed["persona"],
        "blockers": blockers,
        "blocker_set": blocker_set,
        "num_blockers": len(blockers),
        "blocker_count_bucket": blocker_count_bucket(len(blockers)),
        "source_subsplit": source_subsplit_lookup.get(raw_task["id"], "unknown"),
        "expected_terminal_action": terminal["final_action"],
        "repairability": terminal["repairability"],
        "contains_assistant_side_action": any(spec["assistant_side_required"] for spec in specs),
        "contains_hybrid_action": any(spec["hybrid_required"] for spec in specs),
        "contains_user_side_action": any(spec["user_side_required"] for spec in specs),
        "has_only_local_repairable_blockers": not any(
            spec["assistant_side_required"] or spec["hybrid_required"] for spec in specs
        ),
        "has_hybrid_required_blocker": any(spec["hybrid_required"] for spec in specs),
        "blocker_family_tags": family_tags,
        "focus_core_tags": focus_core_tags,
        "assistant_deferred_tags": assistant_deferred_tags,
        "permission_blockers": [blocker_id for blocker_id in blockers if blocker_id in PERMISSION_BLOCKERS],
        "has_apn": "apn" in family_tags,
        "has_permission": "permission" in family_tags,
        "has_wifi_calling": "wifi_calling" in family_tags,
        "has_network_preference": "network_preference" in family_tags,
        "has_sim": "sim" in family_tags,
        "has_data_mode": "data_mode" in family_tags,
        "has_roaming_enabled_off": "roaming_enabled_off" in family_tags,
        "has_data_usage_exceeded": "data_usage_exceeded" in family_tags,
        "has_roaming_disabled_on": "roaming_disabled_on" in family_tags,
        "has_hybrid_roaming_disabled_off": "hybrid_roaming_disabled_off" in family_tags,
    }


def values_for_group(candidate: dict[str, Any], group_name: str) -> list[str]:
    value = candidate[group_name]
    if isinstance(value, list):
        return value
    return [str(value)]


def choose_balanced_candidates(
    candidates: list[dict[str, Any]],
    *,
    count: int,
    groups: list[str],
    prefer_high_complexity: bool = False,
) -> list[dict[str, Any]]:
    if len(candidates) < count:
        raise ValueError(f"Need at least {count} candidates, got {len(candidates)}.")

    counters = {group_name: Counter() for group_name in groups}
    remaining = sorted(candidates, key=lambda row: row["task_id"])
    selected: list[dict[str, Any]] = []

    def priority(row: dict[str, Any]) -> tuple[Any, ...]:
        tuple_parts: list[Any] = []
        for group_name in groups:
            group_values = values_for_group(row, group_name)
            if group_values:
                counts = [counters[group_name][value] for value in group_values]
                tuple_parts.append(min(counts))
                tuple_parts.append(sum(counts))
            else:
                tuple_parts.extend([10**9, 10**9])
        if prefer_high_complexity:
            tuple_parts.extend(
                [
                    -len(row["focus_core_tags"]),
                    -row["num_blockers"],
                ]
            )
        else:
            tuple_parts.extend(
                [
                    row["num_blockers"],
                    -len(row["focus_core_tags"]),
                ]
            )
        tuple_parts.append(row["task_id"])
        return tuple(tuple_parts)

    while len(selected) < count:
        chosen = min(remaining, key=priority)
        selected.append(chosen)
        remaining.remove(chosen)
        for group_name in groups:
            for value in values_for_group(chosen, group_name):
                counters[group_name][value] += 1

    return selected


def is_trap_candidate(candidate: dict[str, Any]) -> bool:
    return (
        candidate["expected_terminal_action"] == "repair_all"
        and candidate["has_only_local_repairable_blockers"]
        and 1 <= candidate["num_blockers"] <= 3
        and "apn" not in candidate["blocker_family_tags"]
        and "sim" not in candidate["blocker_family_tags"]
        and "roaming_enabled_off" not in candidate["blocker_family_tags"]
        and candidate["blocker_set"] <= TRAP_SHALLOW_BLOCKERS
    )


def is_target_candidate(candidate: dict[str, Any]) -> bool:
    return (
        candidate["expected_terminal_action"] in {"repair_all", "repair_subset"}
        and not candidate["contains_hybrid_action"]
        and candidate["num_blockers"] >= 4
        and candidate["has_apn"]
        and len(candidate["focus_core_tags"]) >= 3
    )


def is_hard_transfer_control_candidate(candidate: dict[str, Any]) -> bool:
    non_roaming_core = [tag for tag in candidate["focus_core_tags"] if tag != "roaming_enabled_off"]
    return (
        candidate["expected_terminal_action"] == "transfer"
        and candidate["has_hybrid_required_blocker"]
        and candidate["contains_assistant_side_action"]
        and candidate["num_blockers"] >= 4
        and candidate["has_apn"]
        and len(non_roaming_core) >= 3
    )


def classify_profile_switch_terminal_class(
    candidate: dict[str, Any],
    hard_transfer_control_ids: set[str],
) -> str:
    task_id = candidate["task_id"]
    if task_id in hard_transfer_control_ids:
        return PROFILE_SWITCH_CLASS_HARD_TRANSFER
    if (
        candidate["expected_terminal_action"] != "transfer"
        and candidate["num_blockers"] <= 3
        and "apn" not in candidate["blocker_family_tags"]
        and "sim" not in candidate["blocker_family_tags"]
        and "roaming_enabled_off" not in candidate["blocker_family_tags"]
        and candidate["blocker_set"] <= (TRAP_SHALLOW_BLOCKERS | {"data_usage_exceeded"})
    ):
        return PROFILE_SWITCH_CLASS_FAST
    return PROFILE_SWITCH_CLASS_POST


def build_selected_candidate_set(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    trap_bucket = choose_balanced_candidates(
        [candidate for candidate in candidates if is_trap_candidate(candidate)],
        count=TRAP_BUCKET_SIZE,
        groups=["persona", "blocker_count_bucket", "source_subsplit", "focus_core_tags", "permission_blockers"],
    )
    target_subset_bucket = choose_balanced_candidates(
        [
            candidate
            for candidate in candidates
            if is_target_candidate(candidate) and candidate["expected_terminal_action"] == "repair_subset"
        ],
        count=TARGET_REPAIR_SUBSET_BUCKET_SIZE,
        groups=[
            "persona",
            "blocker_count_bucket",
            "source_subsplit",
            "assistant_deferred_tags",
            "focus_core_tags",
            "permission_blockers",
        ],
        prefer_high_complexity=True,
    )
    target_repair_all_bucket = choose_balanced_candidates(
        [
            candidate
            for candidate in candidates
            if is_target_candidate(candidate) and candidate["expected_terminal_action"] == "repair_all"
        ],
        count=TARGET_REPAIR_ALL_BUCKET_SIZE,
        groups=["persona", "blocker_count_bucket", "source_subsplit", "focus_core_tags", "permission_blockers"],
        prefer_high_complexity=True,
    )
    hard_transfer_controls = choose_balanced_candidates(
        [candidate for candidate in candidates if is_hard_transfer_control_candidate(candidate)],
        count=HARD_TRANSFER_CONTROL_COUNT,
        groups=["persona", "blocker_count_bucket", "source_subsplit", "focus_core_tags", "permission_blockers"],
        prefer_high_complexity=True,
    )

    selected_map = {
        candidate["task_id"]: candidate
        for candidate in [*trap_bucket, *target_subset_bucket, *target_repair_all_bucket, *hard_transfer_controls]
    }
    hard_transfer_control_ids = {candidate["task_id"] for candidate in hard_transfer_controls}

    repair_all_needed = TARGET_COUNTS["repair_all"] - sum(
        1
        for candidate in selected_map.values()
        if candidate["expected_terminal_action"] == "repair_all"
    )
    repair_subset_needed = TARGET_COUNTS["repair_subset"] - sum(
        1
        for candidate in selected_map.values()
        if candidate["expected_terminal_action"] == "repair_subset"
    )
    transfer_needed = TARGET_COUNTS["transfer"] - sum(
        1
        for candidate in selected_map.values()
        if candidate["expected_terminal_action"] == "transfer"
    )
    if transfer_needed != 0:
        raise ValueError(f"Seeded hard transfer controls should fully satisfy transfer quota, got {transfer_needed}.")

    repair_all_fill = choose_balanced_candidates(
        [
            candidate
            for candidate in candidates
            if candidate["task_id"] not in selected_map
            and candidate["expected_terminal_action"] == "repair_all"
            and candidate["has_only_local_repairable_blockers"]
        ],
        count=repair_all_needed,
        groups=["persona", "blocker_count_bucket", "source_subsplit", "focus_core_tags", "permission_blockers"],
    )
    repair_subset_fill = choose_balanced_candidates(
        [
            candidate
            for candidate in candidates
            if candidate["task_id"] not in selected_map
            and candidate["expected_terminal_action"] == "repair_subset"
            and not candidate["contains_hybrid_action"]
        ],
        count=repair_subset_needed,
        groups=[
            "persona",
            "blocker_count_bucket",
            "source_subsplit",
            "assistant_deferred_tags",
            "focus_core_tags",
            "permission_blockers",
        ],
        prefer_high_complexity=True,
    )

    all_selected = [
        *trap_bucket,
        *target_subset_bucket,
        *target_repair_all_bucket,
        *hard_transfer_controls,
        *repair_all_fill,
        *repair_subset_fill,
    ]
    if len({candidate["task_id"] for candidate in all_selected}) != 100:
        raise ValueError("Low-transfer selector produced duplicate or non-100 task set.")

    target_bucket = sorted(
        [*target_subset_bucket, *target_repair_all_bucket],
        key=lambda candidate: candidate["task_id"],
    )
    specialist_bucket = sorted(
        [
            candidate
            for candidate in target_bucket
            if candidate["num_blockers"] >= SPECIALIST_MIN_BLOCKERS
            and (
                candidate["has_sim"]
                or candidate["has_roaming_enabled_off"]
                or candidate["has_roaming_disabled_on"]
            )
        ],
        key=lambda candidate: candidate["task_id"],
    )
    if not specialist_bucket or len(specialist_bucket) >= len(target_bucket):
        raise ValueError("Specialist subset must be non-empty and a strict subset of target bucket.")

    return {
        "selected_candidates": sorted(all_selected, key=lambda candidate: candidate["task_id"]),
        "trap_bucket": sorted(trap_bucket, key=lambda candidate: candidate["task_id"]),
        "target_bucket": target_bucket,
        "specialist_bucket": specialist_bucket,
        "hard_transfer_controls": sorted(hard_transfer_controls, key=lambda candidate: candidate["task_id"]),
        "hard_transfer_control_ids": hard_transfer_control_ids,
    }


def enrich_profile_switch_metadata(
    row: dict[str, Any],
    *,
    candidate: dict[str, Any],
    trap_ids: set[str],
    target_ids: set[str],
    specialist_ids: set[str],
    hard_transfer_control_ids: set[str],
) -> dict[str, Any]:
    enriched = deepcopy(row)
    metadata = deepcopy(enriched.get("metadata", {}))
    terminal_class = classify_profile_switch_terminal_class(candidate, hard_transfer_control_ids)
    metadata.update(
        {
            "subset_version": DATASET_VERSION,
            "source_original_task_id": enriched["original_task_id"],
            "profile_switch_terminal_class": terminal_class,
            "is_hard_transfer_control": enriched["original_task_id"] in hard_transfer_control_ids,
            "is_non_transfer_post_switch": terminal_class == PROFILE_SWITCH_CLASS_POST,
            "has_hybrid_required_blocker": candidate["has_hybrid_required_blocker"],
            "has_only_local_repairable_blockers": candidate["has_only_local_repairable_blockers"],
            "blocker_family_tags": list(candidate["blocker_family_tags"]),
            "is_profile_switch_trap_bucket": enriched["original_task_id"] in trap_ids,
            "is_profile_switch_target_bucket": enriched["original_task_id"] in target_ids,
            "is_profile_switch_specialist_bucket": enriched["original_task_id"] in specialist_ids,
        }
    )
    enriched["metadata"] = metadata
    return enriched


def summarize_candidates(
    candidates: list[dict[str, Any]],
    hard_transfer_control_ids: set[str] | None = None,
) -> dict[str, Any]:
    terminal_counter: Counter[str] = Counter()
    repairability_counter: Counter[str] = Counter()
    persona_counter: Counter[str] = Counter()
    blocker_count_counter: Counter[int] = Counter()
    blocker_count_bucket_counter: Counter[str] = Counter()
    source_subsplit_counter: Counter[str] = Counter()
    family_counter: Counter[str] = Counter()
    assistant_deferred_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    local_only_count = 0
    hard_transfer_count = 0

    for candidate in candidates:
        terminal_counter[candidate["expected_terminal_action"]] += 1
        repairability_counter[candidate["repairability"]] += 1
        persona_counter[candidate["persona"]] += 1
        blocker_count_counter[candidate["num_blockers"]] += 1
        blocker_count_bucket_counter[candidate["blocker_count_bucket"]] += 1
        source_subsplit_counter[candidate["source_subsplit"]] += 1
        family_counter.update(candidate["blocker_family_tags"])
        assistant_deferred_counter.update(candidate["assistant_deferred_tags"])
        local_only_count += int(candidate["has_only_local_repairable_blockers"])
        hard_transfer_count += int(candidate["expected_terminal_action"] == "transfer")
        if hard_transfer_control_ids is not None:
            class_counter[
                classify_profile_switch_terminal_class(candidate, hard_transfer_control_ids)
            ] += 1

    summary = {
        "terminal_action_distribution": dict(sorted(terminal_counter.items())),
        "repairability_distribution": dict(sorted(repairability_counter.items())),
        "persona_distribution": dict(sorted(persona_counter.items())),
        "blocker_count_distribution": {
            str(key): blocker_count_counter[key] for key in sorted(blocker_count_counter)
        },
        "blocker_count_bucket_distribution": {
            key: blocker_count_bucket_counter[key]
            for key in ["1", "2", "3", "4", "5+"]
            if blocker_count_bucket_counter[key]
        },
        "source_subsplit_distribution": dict(sorted(source_subsplit_counter.items())),
        "blocker_family_coverage": dict(sorted(family_counter.items())),
        "assistant_deferred_blocker_coverage": dict(sorted(assistant_deferred_counter.items())),
        "local_only_count": local_only_count,
        "hard_transfer_count": hard_transfer_count,
        "task_count": len(candidates),
        "task_ids": [candidate["task_id"] for candidate in sorted(candidates, key=lambda row: row["task_id"])],
    }
    if hard_transfer_control_ids is not None:
        summary["profile_switch_terminal_class_distribution"] = dict(sorted(class_counter.items()))
    return summary


def summarize_terminal_classes(
    candidates: list[dict[str, Any]],
    hard_transfer_control_ids: set[str],
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for candidate in candidates:
        counter[classify_profile_switch_terminal_class(candidate, hard_transfer_control_ids)] += 1
    return dict(sorted(counter.items()))


def build_manifest(
    *,
    selected_candidates: list[dict[str, Any]],
    trap_bucket: list[dict[str, Any]],
    target_bucket: list[dict[str, Any]],
    specialist_bucket: list[dict[str, Any]],
    hard_transfer_controls: list[dict[str, Any]],
    hard_transfer_control_ids: set[str],
    validation_summary: dict[str, Any],
    source_pool_summary: dict[str, Any],
) -> dict[str, Any]:
    current_reference_rows = load_json(REFERENCE_DATASET_PATH)
    current_reference_counter = Counter(
        row["metadata"]["expected_terminal_action"] for row in current_reference_rows
    )
    transfer_before = current_reference_counter["transfer"]
    transfer_after = sum(
        1 for candidate in selected_candidates if candidate["expected_terminal_action"] == "transfer"
    )
    selected_summary = summarize_candidates(
        selected_candidates,
        hard_transfer_control_ids=hard_transfer_control_ids,
    )

    return {
        "subset_name": DATASET_NAME,
        "subset_version": DATASET_VERSION,
        "family": "telecom_mms_recovery",
        "task_count": len(selected_candidates),
        "builder_script": str(Path(__file__).relative_to(REPO_ROOT)),
        "derived_with_scripts": [
            "scripts/build_telecom_mms_fixed_tree.py",
            "scripts/build_telecom_mms_capability_benchmark.py",
            "scripts/build_telecom_mms_capability_time_benchmark.py",
            "scripts/build_shared_basin_profile_switch_assets.py",
            str(Path(__file__).relative_to(REPO_ROOT)),
        ],
        "source_dataset_path": str(TELECOM_TASKS_PATH),
        "source_dataset_sha256": sha256_file(TELECOM_TASKS_PATH),
        "source_split_tasks_path": str(TELECOM_SPLITS_PATH),
        "source_split_tasks_sha256": sha256_file(TELECOM_SPLITS_PATH),
        "source_db_path": str(TELECOM_DB_PATH),
        "source_db_sha256": sha256_file(TELECOM_DB_PATH),
        "source_pool_summary": source_pool_summary,
        "notes": [
            "Derived directly from the raw telecom tau2 task source rather than resampling the existing balanced 100-task profile-switch dataset.",
            "Selection intentionally suppresses oracle-transfer tasks except for a small hard transfer control slice.",
            "Low-transfer profile-switch keeps the existing fixed-tree schema, oracle labels, capability requirements, and deliberation requirements unchanged.",
            "Profile-switch stage overrides remain trap/target/specialist scoped so existing family semantics stay compatible.",
        ],
        "sampling_criteria": {
            "target_counts": TARGET_COUNTS,
            "transfer_upper_bound_if_source_limited": 15,
            "major_exclusions": [
                "Downsample metadata.expected_terminal_action == transfer to hard transfer control quota only.",
                "Downsample metadata.repairability == transfer_required to hard transfer control quota only.",
                "Downsample contains_hybrid_action == true and contains_assistant_side_action == true to hard transfer control quota only.",
            ],
            "hard_transfer_control_policy": {
                "count": HARD_TRANSFER_CONTROL_COUNT,
                "must_have": [
                    "expected_terminal_action == transfer",
                    "contains_hybrid_action == true",
                    "contains_assistant_side_action == true",
                    "num_blockers >= 4",
                    "break_apn_mms_setting present",
                ],
            },
            "non_transfer_priority_families": FOCUS_BLOCKER_FAMILIES,
            "trap_bucket_policy": {
                "count": TRAP_BUCKET_SIZE,
                "rules": [
                    "repair_all only",
                    "1-3 blockers",
                    "no APN, no SIM, no local roaming toggle",
                    "shallow/local blocker families only",
                ],
            },
            "target_bucket_policy": {
                "count": len(target_bucket),
                "mix": {
                    "repair_subset": TARGET_REPAIR_SUBSET_BUCKET_SIZE,
                    "repair_all": TARGET_REPAIR_ALL_BUCKET_SIZE,
                },
                "rules": [
                    "non-transfer only",
                    "num_blockers >= 4",
                    "APN present",
                    "at least three local core blocker families",
                ],
            },
        },
        "selection_summary": selected_summary,
        "bucket_summary": {
            "trap_bucket": summarize_candidates(trap_bucket, hard_transfer_control_ids=hard_transfer_control_ids),
            "target_bucket": summarize_candidates(target_bucket, hard_transfer_control_ids=hard_transfer_control_ids),
            "specialist_bucket": summarize_candidates(
                specialist_bucket,
                hard_transfer_control_ids=hard_transfer_control_ids,
            ),
            "hard_transfer_control_bucket": summarize_candidates(
                hard_transfer_controls,
                hard_transfer_control_ids=hard_transfer_control_ids,
            ),
            "specialist_is_subset_of_target": all(
                candidate["task_id"] in {row["task_id"] for row in target_bucket}
                for candidate in specialist_bucket
            ),
        },
        "comparison_to_existing_profile_switch_dataset": {
            "reference_dataset_path": str(REFERENCE_DATASET_PATH),
            "reference_terminal_action_distribution": dict(sorted(current_reference_counter.items())),
            "transfer_count_before": transfer_before,
            "transfer_count_after": transfer_after,
            "transfer_count_delta": transfer_after - transfer_before,
            "transfer_rate_before": round(transfer_before / len(current_reference_rows), 4),
            "transfer_rate_after": round(transfer_after / len(selected_candidates), 4),
            "transfer_rate_delta": round(
                (transfer_after / len(selected_candidates))
                - (transfer_before / len(current_reference_rows)),
                4,
            ),
        },
        "validation_summary": validation_summary,
        "task_ids": [candidate["task_id"] for candidate in selected_candidates],
        "trap_favoring_task_ids": [candidate["task_id"] for candidate in trap_bucket],
        "target_favoring_task_ids": [candidate["task_id"] for candidate in target_bucket],
        "specialist_task_ids": [candidate["task_id"] for candidate in specialist_bucket],
        "hard_transfer_control_task_ids": [candidate["task_id"] for candidate in hard_transfer_controls],
        "local_only_task_ids": [
            candidate["task_id"]
            for candidate in selected_candidates
            if candidate["has_only_local_repairable_blockers"]
        ],
    }


def build_schedule_bucket_payload(
    *,
    trap_bucket: list[dict[str, Any]],
    target_bucket: list[dict[str, Any]],
    specialist_bucket: list[dict[str, Any]],
    hard_transfer_controls: list[dict[str, Any]],
    selected_candidates: list[dict[str, Any]],
    hard_transfer_control_ids: set[str],
) -> dict[str, Any]:
    return {
        "schema_version": DATASET_VERSION,
        "source_dataset": str(OUTPUT_TASKS_PATH),
        "source_manifest": str(OUTPUT_MANIFEST_PATH),
        "selection_criteria": {
            "trap_favoring": "Fast/local pre-switch bucket: 1-3 blockers, local-only repair_all, excludes APN/SIM/local-roaming-toggle blockers.",
            "target_favoring": "Deep/local post-switch bucket: non-transfer tasks with >=4 blockers, APN present, and at least three local core blocker families.",
            "specialist": "Strict subset of target_favoring for verification-heavy deep local repair tasks (>=6 blockers and SIM or roaming involvement).",
            "hard_transfer_control": "Held-out hybrid/nonlocal transfer controls retained to verify deep policies do not over-repair impossible local cases.",
        },
        "bucket_labels": {
            "trap_favoring": "fast_path_pre_switch",
            "target_favoring": "deep_path_post_switch_low_transfer",
            "specialist": "verification_heavy_post_switch_low_transfer",
            "hard_transfer_control": "hard_transfer_control",
        },
        "bucket_notes": {
            "trap_favoring": "Shallow local tasks intended for the pre-switch fast path.",
            "target_favoring": "Local-repair-heavy post-switch tasks where deeper reasoning and cautious closure should help.",
            "specialist": "High-complexity target subset for verification-heavy post-switch analysis.",
            "hard_transfer_control": "Hybrid/manual blocker tasks intentionally preserved as negative controls.",
        },
        "trap_favoring_task_ids": [candidate["task_id"] for candidate in trap_bucket],
        "target_favoring_task_ids": [candidate["task_id"] for candidate in target_bucket],
        "specialist_task_ids": [candidate["task_id"] for candidate in specialist_bucket],
        "hard_transfer_control_task_ids": [candidate["task_id"] for candidate in hard_transfer_controls],
        "local_only_task_ids": [
            candidate["task_id"]
            for candidate in selected_candidates
            if candidate["has_only_local_repairable_blockers"]
        ],
        "coverage_summary": {
            "trap_bucket": summarize_candidates(trap_bucket, hard_transfer_control_ids=hard_transfer_control_ids),
            "target_bucket": summarize_candidates(target_bucket, hard_transfer_control_ids=hard_transfer_control_ids),
            "specialist_bucket": summarize_candidates(
                specialist_bucket,
                hard_transfer_control_ids=hard_transfer_control_ids,
            ),
            "hard_transfer_control_bucket": summarize_candidates(
                hard_transfer_controls,
                hard_transfer_control_ids=hard_transfer_control_ids,
            ),
        },
    }


def validate_dataset(
    rows: list[dict[str, Any]],
    *,
    selected_candidates: list[dict[str, Any]],
    hard_transfer_control_ids: set[str],
) -> dict[str, Any]:
    if len(rows) != 100:
        raise ValueError(f"Expected 100 rows, got {len(rows)}.")

    reference_rows = load_json(REFERENCE_DATASET_PATH)
    reference_top_level_keys = set(reference_rows[0].keys())
    top_level_key_mismatches = [
        row["original_task_id"]
        for row in rows
        if set(row.keys()) != reference_top_level_keys
    ]
    if top_level_key_mismatches:
        raise ValueError(
            "Schema mismatch against reference profile-switch dataset for tasks: "
            + ", ".join(top_level_key_mismatches[:5])
        )

    terminal_counter = Counter()
    hard_transfer_count = 0
    for row in rows:
        metadata = row.get("metadata", {})
        for required_key in [
            "expected_terminal_action",
            "profile_switch_terminal_class",
            "is_hard_transfer_control",
            "is_non_transfer_post_switch",
            "has_hybrid_required_blocker",
            "has_only_local_repairable_blockers",
            "source_original_task_id",
        ]:
            if required_key not in metadata:
                raise ValueError(f"Missing metadata key {required_key} on {row['original_task_id']}.")

        for stage_name in ["stage1", "stage2", "stage3", "stage4", "stage5"]:
            stage_payload = row.get(stage_name)
            if not isinstance(stage_payload, dict):
                raise ValueError(f"Stage payload {stage_name} missing on {row['original_task_id']}.")
            if "capability_requirements" not in stage_payload or "deliberation_requirement" not in stage_payload:
                raise ValueError(
                    f"Stage {stage_name} missing capability/deliberation data on {row['original_task_id']}."
                )
            capability_requirements = stage_payload["capability_requirements"]
            if sorted(capability_requirements.keys()) != sorted(CAPABILITY_NAMES):
                raise ValueError(
                    f"Stage {stage_name} capability keys mismatch on {row['original_task_id']}."
                )

        terminal_counter[metadata["expected_terminal_action"]] += 1
        hard_transfer_count += int(metadata["is_hard_transfer_control"])

    if terminal_counter != Counter(TARGET_COUNTS):
        raise ValueError(f"Terminal action distribution mismatch: {terminal_counter} != {TARGET_COUNTS}.")
    if hard_transfer_count > 10:
        raise ValueError(f"Hard transfer control count exceeds upper bound: {hard_transfer_count}.")
    if hard_transfer_count != TARGET_COUNTS["transfer"]:
        raise ValueError(
            "Every transfer row in the low-transfer dataset should be a hard transfer control."
        )

    selected_counter = Counter(
        candidate["expected_terminal_action"] for candidate in selected_candidates
    )
    if selected_counter != terminal_counter:
        raise ValueError(
            f"Selected candidate distribution {selected_counter} disagrees with built rows {terminal_counter}."
        )

    profile_class_distribution = summarize_terminal_classes(selected_candidates, hard_transfer_control_ids)
    return {
        "row_count": len(rows),
        "schema_compatible_with": str(REFERENCE_DATASET_PATH),
        "terminal_action_distribution": dict(sorted(terminal_counter.items())),
        "hard_transfer_control_count": hard_transfer_count,
        "profile_switch_terminal_class_distribution": profile_class_distribution,
        "all_stage_capability_requirements_complete": True,
        "all_stage_deliberation_requirements_complete": True,
        "metadata_fields_complete": True,
    }


def build_source_pool_summary(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    hard_transfer_candidate_ids = {
        candidate["task_id"]
        for candidate in candidates
        if is_hard_transfer_control_candidate(candidate)
    }
    return summarize_candidates(
        candidates,
        hard_transfer_control_ids=hard_transfer_candidate_ids,
    )


def main() -> None:
    raw_tasks = load_json(TELECOM_TASKS_PATH)
    split_map = load_json(TELECOM_SPLITS_PATH)
    source_subsplit_lookup = build_source_subsplit_lookup(split_map)
    reference_maps = build_reference_maps(load_telecom_reference_db())

    all_candidates = [
        build_candidate(raw_task, source_subsplit_lookup)
        for raw_task in raw_tasks
        if raw_task["id"].startswith("[mms_issue]")
    ]
    source_pool_summary = build_source_pool_summary(all_candidates)
    selection = build_selected_candidate_set(all_candidates)
    selected_candidates = selection["selected_candidates"]
    trap_bucket = selection["trap_bucket"]
    target_bucket = selection["target_bucket"]
    specialist_bucket = selection["specialist_bucket"]
    hard_transfer_controls = selection["hard_transfer_controls"]
    hard_transfer_control_ids = selection["hard_transfer_control_ids"]

    selected_tasks = [candidate["raw_task"] for candidate in selected_candidates]
    selected_rows, skipped = build_dataset_with_stats(
        tasks=selected_tasks,
        source_split="all_mms_issue",
        source_subsplit_lookup=source_subsplit_lookup,
        subset_version=DATASET_VERSION,
        smoke_task_ids=None,
        reference_maps=reference_maps,
    )
    if skipped:
        raise ValueError(f"Unexpected skipped tasks while building low-transfer dataset: {skipped[:3]}")

    base_row_map = {row["original_task_id"]: row for row in selected_rows}
    capability_rows = [add_capability_requirements(base_row_map[candidate["task_id"]]) for candidate in selected_candidates]
    time_rows = [add_deliberation_requirements(row) for row in capability_rows]

    trap_ids = {candidate["task_id"] for candidate in trap_bucket}
    target_ids = {candidate["task_id"] for candidate in target_bucket}
    specialist_ids = {candidate["task_id"] for candidate in specialist_bucket}
    profiled_rows = apply_profile_switch_dataset(
        rows=time_rows,
        trap_ids=trap_ids,
        target_ids=target_ids,
        specialist_ids=specialist_ids,
    )

    candidate_map = {candidate["task_id"]: candidate for candidate in selected_candidates}
    final_rows = [
        enrich_profile_switch_metadata(
            row,
            candidate=candidate_map[row["original_task_id"]],
            trap_ids=trap_ids,
            target_ids=target_ids,
            specialist_ids=specialist_ids,
            hard_transfer_control_ids=hard_transfer_control_ids,
        )
        for row in profiled_rows
    ]

    validation_summary = validate_dataset(
        final_rows,
        selected_candidates=selected_candidates,
        hard_transfer_control_ids=hard_transfer_control_ids,
    )
    manifest = build_manifest(
        selected_candidates=selected_candidates,
        trap_bucket=trap_bucket,
        target_bucket=target_bucket,
        specialist_bucket=specialist_bucket,
        hard_transfer_controls=hard_transfer_controls,
        hard_transfer_control_ids=hard_transfer_control_ids,
        validation_summary=validation_summary,
        source_pool_summary=source_pool_summary,
    )
    bucket_payload = build_schedule_bucket_payload(
        trap_bucket=trap_bucket,
        target_bucket=target_bucket,
        specialist_bucket=specialist_bucket,
        hard_transfer_controls=hard_transfer_controls,
        selected_candidates=selected_candidates,
        hard_transfer_control_ids=hard_transfer_control_ids,
    )

    write_json(OUTPUT_TASKS_PATH, final_rows)
    write_json(OUTPUT_MANIFEST_PATH, manifest)
    write_json(OUTPUT_BUCKETS_PATH, bucket_payload)

    print(
        json.dumps(
            {
                "output_tasks": str(OUTPUT_TASKS_PATH.relative_to(REPO_ROOT)),
                "output_manifest": str(OUTPUT_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "output_buckets": str(OUTPUT_BUCKETS_PATH.relative_to(REPO_ROOT)),
                "terminal_action_distribution": manifest["selection_summary"]["terminal_action_distribution"],
                "profile_switch_terminal_class_distribution": manifest["selection_summary"][
                    "profile_switch_terminal_class_distribution"
                ],
                "hard_transfer_control_count": manifest["selection_summary"]["hard_transfer_count"],
                "local_only_count": manifest["selection_summary"]["local_only_count"],
                "trap_bucket_count": len(trap_bucket),
                "target_bucket_count": len(target_bucket),
                "specialist_bucket_count": len(specialist_bucket),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
