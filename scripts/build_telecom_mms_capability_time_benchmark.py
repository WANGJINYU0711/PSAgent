from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path
from statistics import mean
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

INPUT_DIR = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities"
INPUT_TASKS_PATH = INPUT_DIR / "tasks.json"
INPUT_MANIFEST_PATH = INPUT_DIR / "manifest.json"
OUTPUT_DIR = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time"
OUTPUT_TASKS_PATH = OUTPUT_DIR / "tasks.json"
OUTPUT_MANIFEST_PATH = OUTPUT_DIR / "manifest.json"

STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
TIME_REQUIREMENT_VERSION = "telecom_stage_deliberation_v1"


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_deep_stage(
    *,
    stage_name: str,
    metadata: dict[str, Any],
    capability_requirements: dict[str, float],
) -> bool:
    top_cap = max(capability_requirements, key=capability_requirements.get)
    num_blockers = int(metadata.get("num_blockers", 0))
    terminal_action = str(metadata.get("expected_terminal_action", ""))
    hard_persona = metadata.get("persona_level") == "Hard"
    hybrid = bool(metadata.get("contains_hybrid_action"))
    assistant_side = bool(metadata.get("contains_assistant_side_action"))

    if stage_name == "stage1":
        return hard_persona or num_blockers >= 5
    if stage_name == "stage2":
        return hybrid or assistant_side or capability_requirements.get("roaming_diagnosis", 0.0) >= 0.35
    if stage_name == "stage3":
        return (
            num_blockers >= 3
            or top_cap in {"permission_diagnosis", "apn_diagnosis", "roaming_diagnosis"}
            or capability_requirements.get("verification", 0.0) >= 0.12
        )
    if stage_name == "stage4":
        return hybrid or terminal_action in {"repair_subset", "transfer"} or num_blockers >= 4
    if stage_name == "stage5":
        return terminal_action in {"repair_subset", "transfer"} or num_blockers >= 3 or hard_persona
    return False


def add_deliberation_requirements(row: dict[str, Any]) -> dict[str, Any]:
    enriched = deepcopy(row)
    metadata = deepcopy(enriched.get("metadata", {}))
    per_stage: dict[str, str] = {}
    for stage_name in STAGES:
        stage_payload = deepcopy(enriched[stage_name])
        cap_req = stage_payload.get("capability_requirements", {})
        requirement = "deep" if _is_deep_stage(
            stage_name=stage_name,
            metadata=metadata,
            capability_requirements=cap_req,
        ) else "fast"
        stage_payload["deliberation_requirement"] = requirement
        enriched[stage_name] = stage_payload
        per_stage[stage_name] = requirement

    metadata["deliberation_requirement_version"] = TIME_REQUIREMENT_VERSION
    metadata["deliberation_requirement_summary"] = per_stage
    enriched["metadata"] = metadata
    return enriched


def deliberation_distribution(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for stage_name in STAGES:
        fast = sum(1 for row in rows if row[stage_name]["deliberation_requirement"] == "fast")
        deep = sum(1 for row in rows if row[stage_name]["deliberation_requirement"] == "deep")
        out[stage_name] = {"fast": fast, "deep": deep}
    return out


def average_stage_difficulty(rows: list[dict[str, Any]]) -> dict[str, float]:
    out: dict[str, float] = {}
    for stage_name in STAGES:
        vals = []
        for row in rows:
            req = row[stage_name].get("capability_requirements", {})
            if req:
                avg = sum(req.values()) / max(1, len(req))
                top = max(req.values(), default=0.0)
                vals.append(0.15 + (0.35 * avg) + (0.25 * top))
        out[stage_name] = round(mean(vals), 4) if vals else 0.0
    return out


def build_manifest(
    source_manifest: dict[str, Any],
    *,
    rows: list[dict[str, Any]],
    source_hash_before: str,
    source_hash_after: str,
    source_manifest_hash_before: str,
    source_manifest_hash_after: str,
) -> dict[str, Any]:
    return {
        "subset_name": "telecom_mms_fixed_tree_base_v2_100_capabilities_time",
        "family": "telecom_mms_recovery",
        "source_tasks_path": str(INPUT_TASKS_PATH.relative_to(REPO_ROOT)),
        "row_count": len(rows),
        "capability_requirement_source": str(INPUT_TASKS_PATH.relative_to(REPO_ROOT)),
        "deliberation_requirement_version": TIME_REQUIREMENT_VERSION,
        "source_hash_before": source_hash_before,
        "source_hash_after": source_hash_after,
        "source_hash_unchanged": source_hash_before == source_hash_after,
        "source_manifest_hash_before": source_manifest_hash_before,
        "source_manifest_hash_after": source_manifest_hash_after,
        "source_manifest_hash_unchanged": source_manifest_hash_before == source_manifest_hash_after,
        "stage_deliberation_distribution": deliberation_distribution(rows),
        "average_stage_difficulty_proxy": average_stage_difficulty(rows),
        "copied_manifest_metadata": source_manifest,
    }


def main() -> None:
    source_hash_before = sha256_file(INPUT_TASKS_PATH)
    source_manifest_hash_before = sha256_file(INPUT_MANIFEST_PATH)

    rows = load_json(INPUT_TASKS_PATH)
    source_manifest = load_json(INPUT_MANIFEST_PATH)
    enriched_rows = [add_deliberation_requirements(row) for row in rows]

    dump_json(OUTPUT_TASKS_PATH, enriched_rows)

    source_hash_after = sha256_file(INPUT_TASKS_PATH)
    source_manifest_hash_after = sha256_file(INPUT_MANIFEST_PATH)
    manifest = build_manifest(
        source_manifest,
        rows=enriched_rows,
        source_hash_before=source_hash_before,
        source_hash_after=source_hash_after,
        source_manifest_hash_before=source_manifest_hash_before,
        source_manifest_hash_after=source_manifest_hash_after,
    )
    dump_json(OUTPUT_MANIFEST_PATH, manifest)

    print(
        json.dumps(
            {
                "output_tasks": str(OUTPUT_TASKS_PATH.relative_to(REPO_ROOT)),
                "output_manifest": str(OUTPUT_MANIFEST_PATH.relative_to(REPO_ROOT)),
                "row_count": len(enriched_rows),
                "stage_deliberation_distribution": manifest["stage_deliberation_distribution"],
                "source_hash_unchanged": manifest["source_hash_unchanged"],
                "source_manifest_hash_unchanged": manifest["source_manifest_hash_unchanged"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
