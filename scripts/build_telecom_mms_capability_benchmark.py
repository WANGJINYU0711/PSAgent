from __future__ import annotations

import hashlib
import json
import re
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean, pstdev
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
ENVS_ROOT = REPO_ROOT / "envs"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(ENVS_ROOT) not in sys.path:
    sys.path.insert(0, str(ENVS_ROOT))

from tree_family.generator import TreeFamilyGenerator  # noqa: E402
from tree_family.specs import CAPABILITY_DESCRIPTIONS, CAPABILITY_NAMES  # noqa: E402


INPUT_TASKS_PATH = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100" / "tasks.json"
INPUT_MANIFEST_PATH = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100" / "manifest.json"
OUTPUT_DIR = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities"
OUTPUT_TASKS_PATH = OUTPUT_DIR / "tasks.json"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

TASK_ID_RE = re.compile(r"^\[(?P<family>[^\]]+)\](?P<body>.*)\[PERSONA:(?P<persona>[^\]]+)\]$")

PERMISSION_BLOCKERS = {
    "break_app_sms_permission",
    "break_app_storage_permission",
    "break_app_both_permissions",
}
APN_BLOCKERS = {"break_apn_mms_setting"}
ROAMING_BLOCKERS = {
    "user_abroad_roaming_disabled_on",
    "user_abroad_roaming_enabled_off",
    "user_abroad_roaming_disabled_off",
}
NETWORKISH_BLOCKERS = {
    "airplane_mode_on",
    "unseat_sim_card",
    "data_mode_off",
    "data_usage_exceeded",
    "bad_network_preference",
    "bad_wifi_calling",
}

STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
CAPABILITY_REQUIREMENT_VERSION = "telecom_stage_capabilities_v1"

STAGE_BASE_REQUIREMENTS: dict[str, dict[str, float]] = {
    "stage1": {
        "user_grounding": 0.65,
        "account_lookup": 0.35,
        "line_resolution": 0.30,
    },
    "stage2": {
        "account_lookup": 0.72,
        "line_resolution": 0.66,
        "roaming_diagnosis": 0.12,
    },
    "stage3": {
        "network_diagnosis": 0.42,
        "permission_diagnosis": 0.10,
        "apn_diagnosis": 0.10,
        "roaming_diagnosis": 0.10,
        "verification": 0.05,
    },
    "stage4": {
        "repair_execution": 0.72,
        "network_diagnosis": 0.10,
        "permission_diagnosis": 0.10,
        "apn_diagnosis": 0.10,
        "roaming_diagnosis": 0.10,
    },
    "stage5": {
        "verification": 0.60,
        "terminal_decision": 0.68,
        "repair_execution": 0.15,
    },
}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_task_id(task_id: str) -> dict[str, Any]:
    match = TASK_ID_RE.match(task_id)
    if not match:
        raise ValueError(f"Unexpected telecom task id format: {task_id}")
    body = match.group("body")
    blockers = [] if not body else body.split("|")
    return {
        "family": match.group("family"),
        "blockers": blockers,
        "persona": match.group("persona"),
    }


def new_stage_requirement_map() -> dict[str, dict[str, float]]:
    stage_map: dict[str, dict[str, float]] = {}
    for stage_name in STAGES:
        req = {capability_name: 0.0 for capability_name in CAPABILITY_NAMES}
        for capability_name, value in STAGE_BASE_REQUIREMENTS[stage_name].items():
            req[capability_name] = value
        stage_map[stage_name] = req
    return stage_map


def boost(stage_map: dict[str, dict[str, float]], stage_name: str, capability_name: str, delta: float) -> None:
    stage_map[stage_name][capability_name] += delta


def finalize_requirements(stage_map: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    finalized: dict[str, dict[str, float]] = {}
    for stage_name, req in stage_map.items():
        finalized[stage_name] = {
            capability_name: round(min(1.0, max(0.0, value)), 3)
            for capability_name, value in req.items()
        }
    return finalized


def build_stage_capability_requirements(row: dict[str, Any]) -> dict[str, dict[str, float]]:
    parsed = parse_task_id(row["original_task_id"])
    blocker_ids = parsed["blockers"]
    metadata = row["metadata"]
    stage_map = new_stage_requirement_map()

    for blocker_id in blocker_ids:
        if blocker_id in PERMISSION_BLOCKERS:
            boost(stage_map, "stage3", "permission_diagnosis", 0.42)
            boost(stage_map, "stage4", "repair_execution", 0.10)
            boost(stage_map, "stage5", "verification", 0.08)

        if blocker_id in APN_BLOCKERS:
            boost(stage_map, "stage3", "apn_diagnosis", 0.42)
            boost(stage_map, "stage4", "repair_execution", 0.10)
            boost(stage_map, "stage5", "verification", 0.08)

        if blocker_id in ROAMING_BLOCKERS:
            boost(stage_map, "stage2", "account_lookup", 0.12)
            boost(stage_map, "stage2", "line_resolution", 0.08)
            boost(stage_map, "stage3", "roaming_diagnosis", 0.42)
            boost(stage_map, "stage4", "repair_execution", 0.08)
            boost(stage_map, "stage5", "terminal_decision", 0.08)
            boost(stage_map, "stage5", "verification", 0.05)

        if blocker_id in NETWORKISH_BLOCKERS:
            boost(stage_map, "stage3", "network_diagnosis", 0.18)
            boost(stage_map, "stage4", "repair_execution", 0.05)
            boost(stage_map, "stage5", "verification", 0.04)

        if blocker_id == "data_usage_exceeded":
            boost(stage_map, "stage2", "account_lookup", 0.12)
            boost(stage_map, "stage5", "terminal_decision", 0.05)

    num_blockers = int(metadata.get("num_blockers", 0))
    if num_blockers >= 4:
        boost(stage_map, "stage3", "network_diagnosis", 0.08)
        boost(stage_map, "stage4", "repair_execution", 0.08)
        boost(stage_map, "stage5", "terminal_decision", 0.06)
    if num_blockers >= 6:
        boost(stage_map, "stage3", "verification", 0.06)
        boost(stage_map, "stage5", "verification", 0.08)

    if metadata.get("contains_assistant_side_action"):
        boost(stage_map, "stage2", "account_lookup", 0.10)
        boost(stage_map, "stage4", "repair_execution", 0.06)
        boost(stage_map, "stage5", "terminal_decision", 0.06)

    if metadata.get("contains_hybrid_action"):
        boost(stage_map, "stage2", "line_resolution", 0.06)
        boost(stage_map, "stage5", "terminal_decision", 0.14)
        boost(stage_map, "stage5", "verification", 0.04)

    if metadata.get("persona_level") == "Hard":
        boost(stage_map, "stage1", "user_grounding", 0.10)
        boost(stage_map, "stage3", "verification", 0.04)
        boost(stage_map, "stage5", "verification", 0.04)

    final_action = metadata.get("expected_terminal_action")
    if final_action == "repair_all":
        boost(stage_map, "stage4", "repair_execution", 0.06)
        boost(stage_map, "stage5", "verification", 0.08)
    elif final_action == "repair_subset":
        boost(stage_map, "stage5", "terminal_decision", 0.20)
        boost(stage_map, "stage5", "verification", 0.12)
    elif final_action == "transfer":
        boost(stage_map, "stage2", "account_lookup", 0.06)
        boost(stage_map, "stage5", "terminal_decision", 0.30)
        boost(stage_map, "stage5", "verification", 0.05)

    return finalize_requirements(stage_map)


def add_capability_requirements(row: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(row)
    stage_requirements = build_stage_capability_requirements(row)
    for stage_name in STAGES:
        enriched[stage_name]["capability_requirements"] = stage_requirements[stage_name]
    enriched["metadata"] = deepcopy(enriched["metadata"])
    enriched["metadata"]["capability_requirement_version"] = CAPABILITY_REQUIREMENT_VERSION
    return enriched


def stage_capability_averages(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for stage_name in STAGES:
        summary[stage_name] = {}
        for capability_name in CAPABILITY_NAMES:
            summary[stage_name][capability_name] = round(
                mean(row[stage_name]["capability_requirements"][capability_name] for row in rows),
                4,
            )
    return summary


def blocker_group_capability_effects(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    groups = {
        "permission_tasks": lambda blockers: any(blocker_id in PERMISSION_BLOCKERS for blocker_id in blockers),
        "apn_tasks": lambda blockers: any(blocker_id in APN_BLOCKERS for blocker_id in blockers),
        "roaming_tasks": lambda blockers: any(blocker_id in ROAMING_BLOCKERS for blocker_id in blockers),
        "networkish_tasks": lambda blockers: any(blocker_id in NETWORKISH_BLOCKERS for blocker_id in blockers),
    }
    summary: dict[str, dict[str, float]] = {}
    for group_name, predicate in groups.items():
        group_rows = [row for row in rows if predicate(parse_task_id(row["original_task_id"])["blockers"])]
        if not group_rows:
            continue
        summary[group_name] = {
            "task_count": len(group_rows),
            "stage3_network_diagnosis_mean": round(
                mean(row["stage3"]["capability_requirements"]["network_diagnosis"] for row in group_rows),
                4,
            ),
            "stage3_permission_diagnosis_mean": round(
                mean(row["stage3"]["capability_requirements"]["permission_diagnosis"] for row in group_rows),
                4,
            ),
            "stage3_apn_diagnosis_mean": round(
                mean(row["stage3"]["capability_requirements"]["apn_diagnosis"] for row in group_rows),
                4,
            ),
            "stage3_roaming_diagnosis_mean": round(
                mean(row["stage3"]["capability_requirements"]["roaming_diagnosis"] for row in group_rows),
                4,
            ),
            "stage5_verification_mean": round(
                mean(row["stage5"]["capability_requirements"]["verification"] for row in group_rows),
                4,
            ),
            "stage5_terminal_decision_mean": round(
                mean(row["stage5"]["capability_requirements"]["terminal_decision"] for row in group_rows),
                4,
            ),
        }
    return summary


def capability_presence_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    for stage_name in STAGES:
        summary[stage_name] = {
            capability_name: sum(
                1
                for row in rows
                if row[stage_name]["capability_requirements"][capability_name] > 0.05
            )
            for capability_name in CAPABILITY_NAMES
        }
    return summary


def skill_requirement_dispersion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    family_spec, agent_map = TreeFamilyGenerator().build_family("strong", seed=0)
    sample_rows = rows[: min(20, len(rows))]
    stage_dispersion: dict[str, dict[str, float]] = {}

    for stage_name in STAGES:
        per_task_stds: list[float] = []
        per_task_ranges: list[float] = []
        stage_agent_ids = family_spec.stage_agents[stage_name]
        for row in sample_rows:
            requirement = row[stage_name]["capability_requirements"]
            scores = []
            for agent_id in stage_agent_ids:
                score = 0.0
                scores.append(score)
            per_task_stds.append(pstdev(scores))
            per_task_ranges.append(max(scores) - min(scores))
        stage_dispersion[stage_name] = {
            "mean_score_std": round(mean(per_task_stds), 6),
            "mean_score_range": round(mean(per_task_ranges), 6),
        }

    return {
        "family_name": family_spec.family_name,
        "sample_task_count": len(sample_rows),
        "stage_dispersion": stage_dispersion,
    }


def verify_stage_requirements(rows: list[dict[str, Any]]) -> dict[str, Any]:
    missing = 0
    for row in rows:
        for stage_name in STAGES:
            if "capability_requirements" not in row[stage_name]:
                missing += 1
    return {
        "all_stage_requirements_present": missing == 0,
        "missing_stage_requirement_slots": missing,
    }


def build_manifest(
    source_manifest: dict[str, Any],
    rows: list[dict[str, Any]],
    *,
    source_hash_before: str,
    source_hash_after: str,
    source_manifest_hash_before: str,
    source_manifest_hash_after: str,
) -> dict[str, Any]:
    return {
        "subset_name": "telecom_mms_fixed_tree_base_v2_100_capabilities",
        "source_subset": source_manifest.get("subset_name"),
        "family": "telecom_mms_recovery",
        "task_count": len(rows),
        "source_tasks_path": str(INPUT_TASKS_PATH),
        "source_manifest_path": str(INPUT_MANIFEST_PATH),
        "source_tasks_sha256_before": source_hash_before,
        "source_tasks_sha256_after": source_hash_after,
        "source_manifest_sha256_before": source_manifest_hash_before,
        "source_manifest_sha256_after": source_manifest_hash_after,
        "capability_requirement_version": CAPABILITY_REQUIREMENT_VERSION,
        "capability_space": {
            capability_name: CAPABILITY_DESCRIPTIONS[capability_name]
            for capability_name in CAPABILITY_NAMES
        },
        "notes": [
            "This is a derived benchmark that adds stage-level capability_requirements.",
            "Original telecom_mms_fixed_tree_base_v2_100 files are left unchanged.",
            "Agent-side attribute profiles are disabled; tasks only receive stage-side requirements.",
        ],
        "coverage_summary": {
            "stage_capability_averages": stage_capability_averages(rows),
            "stage_capability_presence": capability_presence_summary(rows),
            "blocker_group_effects": blocker_group_capability_effects(rows),
            "skill_requirement_dispersion": skill_requirement_dispersion(rows),
            "stage_requirement_presence_check": verify_stage_requirements(rows),
        },
        "task_ids": [row["original_task_id"] for row in rows],
    }


def main() -> None:
    source_hash_before = sha256_file(INPUT_TASKS_PATH)
    source_manifest_hash_before = sha256_file(INPUT_MANIFEST_PATH)
    source_rows = load_json(INPUT_TASKS_PATH)
    source_manifest = load_json(INPUT_MANIFEST_PATH)
    derived_rows = [add_capability_requirements(row) for row in source_rows]
    source_hash_after = sha256_file(INPUT_TASKS_PATH)
    source_manifest_hash_after = sha256_file(INPUT_MANIFEST_PATH)

    manifest = build_manifest(
        source_manifest,
        derived_rows,
        source_hash_before=source_hash_before,
        source_hash_after=source_hash_after,
        source_manifest_hash_before=source_manifest_hash_before,
        source_manifest_hash_after=source_manifest_hash_after,
    )

    dump_json(OUTPUT_TASKS_PATH, derived_rows)
    dump_json(OUTPUT_MANIFEST_PATH, manifest)

    print(
        json.dumps(
            {
                "output_tasks_path": str(OUTPUT_TASKS_PATH),
                "output_manifest_path": str(OUTPUT_MANIFEST_PATH),
                "task_count": len(derived_rows),
                "source_hash_unchanged": source_hash_before == source_hash_after,
                "source_manifest_hash_unchanged": source_manifest_hash_before == source_manifest_hash_after,
                "capability_requirement_version": CAPABILITY_REQUIREMENT_VERSION,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
