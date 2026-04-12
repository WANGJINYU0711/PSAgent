from __future__ import annotations

import json
import re
import sys
import tomllib
import warnings
from copy import deepcopy
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
TAU2_SRC = REPO_ROOT / "tau2-bench" / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from envs.telecom_mms_specs import (  # noqa: E402
    CANONICAL_BLOCKER_SPECS,
    first_pass_terminal_decision,
    get_blocker_spec,
    materialize_repair_steps,
)


TELECOM_TASKS_PATH = REPO_ROOT / "tau2-bench" / "data" / "tau2" / "domains" / "telecom" / "tasks.json"
TELECOM_SPLITS_PATH = REPO_ROOT / "tau2-bench" / "data" / "tau2" / "domains" / "telecom" / "split_tasks.json"
TELECOM_DB_PATH = REPO_ROOT / "tau2-bench" / "data" / "tau2" / "domains" / "telecom" / "db.toml"
OUTPUT_BASE_PATH = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base" / "tasks.json"
OUTPUT_SMOKE_PATH = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_smoke10" / "tasks.json"
OUTPUT_MANIFEST_PATH = REPO_ROOT / "data" / "derived" / "telecom_mms_fixed_tree_smoke10" / "manifest.json"

TASK_ID_RE = re.compile(r"^\[(?P<family>[^\]]+)\](?P<body>.*)\[PERSONA:(?P<persona>[^\]]+)\]$")
PHONE_RE = re.compile(r"(\d{3}-\d{3}-\d{4})")
NAME_RE = re.compile(r"You are ([A-Z][a-z]+ [A-Z][a-z]+)")
REFUEL_RE = re.compile(r"refuel ([0-9]+(?:\.[0-9]+)?) GB", re.IGNORECASE)

SMOKE10_TASK_IDS = [
    "[mms_issue]airplane_mode_on|break_app_both_permissions[PERSONA:Hard]",
    "[mms_issue]bad_network_preference|break_app_both_permissions[PERSONA:Easy]",
    "[mms_issue]bad_network_preference|user_abroad_roaming_disabled_off[PERSONA:None]",
    "[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]",
    "[mms_issue]bad_wifi_calling|data_mode_off|data_usage_exceeded[PERSONA:Easy]",
    "[mms_issue]break_apn_mms_setting|data_mode_off|user_abroad_roaming_disabled_on[PERSONA:Hard]",
    "[mms_issue]bad_wifi_calling|break_apn_mms_setting|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]",
    "[mms_issue]airplane_mode_on|bad_network_preference|break_apn_mms_setting|data_usage_exceeded[PERSONA:Hard]",
    "[mms_issue]bad_network_preference|bad_wifi_calling|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
    "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]",
]


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


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


def extract_user_context(raw_task: dict[str, Any]) -> dict[str, str]:
    instructions = raw_task["user_scenario"]["instructions"]
    return {
        "reason_for_call": instructions["reason_for_call"],
        "known_info": instructions["known_info"],
        "task_instructions": instructions["task_instructions"],
    }


def extract_name(known_info: str) -> str | None:
    match = NAME_RE.search(known_info)
    return match.group(1) if match else None


def extract_phone(known_info: str) -> str | None:
    match = PHONE_RE.search(known_info)
    return match.group(1) if match else None


def extract_refuel_gb(task_instructions: str, default: float = 2.0) -> float:
    match = REFUEL_RE.search(task_instructions)
    if not match:
        return default
    return float(match.group(1))


def build_stage1_oracle(user_context: dict[str, str]) -> dict[str, Any]:
    phone_number = extract_phone(user_context["known_info"])
    full_name = extract_name(user_context["known_info"])
    is_abroad = "abroad" in user_context["known_info"].lower()
    max_refuel_gb = extract_refuel_gb(user_context["task_instructions"], default=2.0)
    return {
        "domain": "telecom",
        "problem_family": "mms_issue",
        "customer_lookup": {
            "full_name": full_name,
            "phone_number": phone_number,
            "lookup_confidence": "high" if phone_number else "medium",
        },
        "line_selector": {
            "type": "phone_number",
            "value": phone_number,
        },
        "symptom_report": {
            "cannot_send_mms": True,
            "wants_resolution": True,
            "target_success_signal": "can_send_mms_true",
        },
        "context_flags": {
            "is_abroad_claimed": is_abroad,
            "refuel_allowed": "willing to refuel" in user_context["task_instructions"].lower(),
            "max_refuel_gb": max_refuel_gb,
            "plan_change_allowed": "do not want to change your mobile data plan"
            not in user_context["task_instructions"].lower(),
        },
        "conversation_risk_flags": [
            "mild_frustration_after_unsuccessful_attempt",
            "tool_grounding_required",
        ],
    }


def _model_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    raise TypeError(f"Unsupported telecom object type: {type(obj)!r}")


def load_telecom_reference_db() -> dict[str, Any]:
    return tomllib.loads(TELECOM_DB_PATH.read_text())


def build_reference_maps(db_payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        "customers": {row["customer_id"]: row for row in db_payload["customers"]},
        "lines": {row["line_id"]: row for row in db_payload["lines"]},
        "plans": {row["plan_id"]: row for row in db_payload["plans"]},
    }


def find_customer_by_phone(reference_maps: dict[str, dict[str, Any]], phone_number: str) -> dict[str, Any]:
    lines = reference_maps["lines"]
    for customer in reference_maps["customers"].values():
        if customer["phone_number"] == phone_number:
            return customer
        for line_id in customer["line_ids"]:
            line = lines[line_id]
            if line["phone_number"] == phone_number:
                return customer
    raise ValueError(f"No telecom customer found for phone number {phone_number}")


def compute_stage2_oracle(
    reference_maps: dict[str, dict[str, Any]],
    raw_task: dict[str, Any],
    stage1_oracle: dict[str, Any],
) -> dict[str, Any]:
    phone_number = stage1_oracle["line_selector"]["value"]
    customer = find_customer_by_phone(reference_maps, phone_number)
    candidate_line_ids = list(customer["line_ids"])
    resolved_line = None
    for line_id in candidate_line_ids:
        line = deepcopy(reference_maps["lines"][line_id])
        if line["phone_number"] == phone_number:
            resolved_line = line
            break
    if resolved_line is None:
        raise ValueError(f"Could not resolve line for phone number {phone_number}")
    for action in raw_task["initial_state"]["initialization_actions"]:
        if action["env_type"] == "assistant" and action["func_name"] == "set_data_usage":
            resolved_line["data_used_gb"] = action["arguments"]["data_used_gb"]
        elif action["env_type"] == "assistant" and action["func_name"] == "disable_roaming":
            resolved_line["roaming_enabled"] = False
        elif action["env_type"] == "assistant" and action["func_name"] == "enable_roaming":
            resolved_line["roaming_enabled"] = True
    plan = reference_maps["plans"][resolved_line["plan_id"]]
    return {
        "candidate_customers": [customer["customer_id"]],
        "resolved_customer_id": customer["customer_id"],
        "candidate_line_ids": candidate_line_ids,
        "resolved_line_id": resolved_line["line_id"],
        "target_phone_number": phone_number,
        "assistant_account_snapshot": {
            "line_status": str(resolved_line["status"]).lower(),
            "roaming_enabled_on_account": resolved_line["roaming_enabled"],
            "plan_id": resolved_line["plan_id"],
            "data_used_gb": resolved_line["data_used_gb"],
            "data_limit_gb": plan["data_limit_gb"],
        },
        "resolution_status": "resolved",
    }


def build_observed_state(
    parsed_task_id: dict[str, Any],
    stage2_oracle: dict[str, Any],
    user_context: dict[str, str],
) -> dict[str, Any]:
    blocker_ids = set(parsed_task_id["blockers"])
    data_usage_exceeded = "data_usage_exceeded" in blocker_ids
    service_blocked = bool({"airplane_mode_on", "unseat_sim_card"} & blocker_ids)
    data_blocked = bool(
        {
            "data_mode_off",
            "data_usage_exceeded",
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
        }
        & blocker_ids
    )
    mobile_data_working = not (service_blocked or data_blocked)
    internet_speed_desc = "No Connection"
    if mobile_data_working:
        internet_speed_desc = "Poor" if "bad_network_preference" in blocker_ids else "Excellent"

    roaming_enabled_on_account = stage2_oracle["assistant_account_snapshot"]["roaming_enabled_on_account"]
    roaming_enabled_on_device = False
    if "user_abroad_roaming_disabled_on" in blocker_ids:
        roaming_enabled_on_device = True
    elif "user_abroad_roaming_enabled_off" in blocker_ids:
        roaming_enabled_on_device = False
        roaming_enabled_on_account = True
    elif "user_abroad_roaming_disabled_off" in blocker_ids:
        roaming_enabled_on_device = False
        roaming_enabled_on_account = False
    elif "abroad" in user_context["known_info"].lower():
        roaming_enabled_on_device = True
    observed = {
        "can_send_mms": False,
        "service_status": "no_service" if service_blocked else "connected",
        "mobile_data_working": mobile_data_working,
        "internet_speed_desc": internet_speed_desc,
        "is_abroad": "abroad" in user_context["known_info"].lower(),
        "roaming_enabled_on_device": roaming_enabled_on_device,
        "roaming_enabled_on_account": roaming_enabled_on_account,
        "airplane_mode": "airplane_mode_on" in blocker_ids,
        "sim_status": "missing" if "unseat_sim_card" in blocker_ids else "active",
        "network_mode_preference": "2g_only" if "bad_network_preference" in blocker_ids else "4g_5g_preferred",
        "wifi_calling_enabled": "bad_wifi_calling" in blocker_ids,
        "apn_mms_ok": "break_apn_mms_setting" not in blocker_ids,
        "messaging_sms_permission": "break_app_sms_permission" not in blocker_ids
        and "break_app_both_permissions" not in blocker_ids,
        "messaging_storage_permission": "break_app_storage_permission" not in blocker_ids
        and "break_app_both_permissions" not in blocker_ids,
        "data_usage_exceeded": data_usage_exceeded,
    }
    return observed


def blocker_matches_observed_state(blocker_id: str, observed_state: dict[str, Any]) -> bool:
    checks = {
        "airplane_mode_on": observed_state["airplane_mode"] is True,
        "unseat_sim_card": observed_state["sim_status"] == "missing",
        "data_mode_off": observed_state["mobile_data_working"] is False,
        "data_usage_exceeded": observed_state["data_usage_exceeded"] is True,
        "user_abroad_roaming_disabled_on": (
            observed_state["is_abroad"] is True
            and observed_state["roaming_enabled_on_account"] is False
            and observed_state["roaming_enabled_on_device"] is True
        ),
        "user_abroad_roaming_enabled_off": (
            observed_state["is_abroad"] is True
            and observed_state["roaming_enabled_on_account"] is True
            and observed_state["roaming_enabled_on_device"] is False
        ),
        "user_abroad_roaming_disabled_off": (
            observed_state["is_abroad"] is True
            and observed_state["roaming_enabled_on_account"] is False
            and observed_state["roaming_enabled_on_device"] is False
        ),
        "bad_network_preference": observed_state["network_mode_preference"] == "2g_only",
        "bad_wifi_calling": observed_state["wifi_calling_enabled"] is True,
        "break_apn_mms_setting": observed_state["apn_mms_ok"] is False,
        "break_app_sms_permission": observed_state["messaging_sms_permission"] is False,
        "break_app_storage_permission": observed_state["messaging_storage_permission"] is False,
        "break_app_both_permissions": (
            observed_state["messaging_sms_permission"] is False
            and observed_state["messaging_storage_permission"] is False
        ),
    }
    return checks.get(blocker_id, True)


def diagnostic_evidence_for(blocker_id: str) -> list[str]:
    evidence_map = {
        "airplane_mode_on": ["check_network_status", "check_status_bar"],
        "unseat_sim_card": ["check_sim_status", "check_network_status"],
        "data_mode_off": ["check_network_status"],
        "data_usage_exceeded": ["get_details_by_id"],
        "user_abroad_roaming_disabled_on": ["check_network_status", "get_details_by_id"],
        "user_abroad_roaming_enabled_off": ["check_network_status", "get_details_by_id"],
        "user_abroad_roaming_disabled_off": ["check_network_status", "get_details_by_id"],
        "bad_network_preference": ["check_network_mode_preference"],
        "bad_wifi_calling": ["check_wifi_calling_status"],
        "break_apn_mms_setting": ["check_apn_settings"],
        "break_app_sms_permission": ["check_app_permissions"],
        "break_app_storage_permission": ["check_app_permissions"],
        "break_app_both_permissions": ["check_app_permissions"],
    }
    return evidence_map.get(blocker_id, [])


def compute_stage3_oracle(
    parsed_task_id: dict[str, Any],
    stage2_oracle: dict[str, Any],
    user_context: dict[str, str],
) -> dict[str, Any]:
    observed_state = build_observed_state(parsed_task_id, stage2_oracle, user_context)
    per_blocker = []
    for blocker_id in parsed_task_id["blockers"]:
        spec = get_blocker_spec(blocker_id)
        if not blocker_matches_observed_state(blocker_id, observed_state):
            warnings.warn(
                f"Observed state does not cleanly confirm blocker {blocker_id} in task {parsed_task_id}"
            )
        per_blocker.append(
            {
                "blocker_id": blocker_id,
                "blocker_layer": spec["blocker_layer"],
                "repair_owner": spec["repair_owner"],
                "repair_action_family": spec["repair_action_family"],
                "diagnostic_evidence": diagnostic_evidence_for(blocker_id),
                "verification_signal": spec["verification_signal"],
            }
        )
    return {"observed_state": observed_state, "per_blocker": per_blocker}


def build_stage4_oracle(
    blocker_ids: list[str],
    stage2_oracle: dict[str, Any],
) -> dict[str, Any]:
    decision = first_pass_terminal_decision(blocker_ids)
    present = set(blocker_ids)
    selected = set(decision["selected_blocker_ids"])
    deferred = set(decision["deferred_blocker_ids"])
    ordered = [
        *decision["selected_blocker_ids"],
        *[bid for bid in decision["deferred_blocker_ids"] if bid not in selected],
    ]
    rows = []
    for repair_order, blocker_id in enumerate(ordered, start=1):
        spec = get_blocker_spec(blocker_id)
        if blocker_id in selected:
            should_repair = True
            execute_decision = "repair"
            adjudication_label = f"repair_{spec['blocker_layer']}_blocker"
            refusal_code = None
        elif decision["final_action"] == "transfer":
            should_repair = False
            execute_decision = "transfer"
            adjudication_label = f"transfer_{spec['blocker_layer']}_blocker"
            refusal_code = decision["transfer_reason"]
        elif blocker_id in deferred:
            should_repair = False
            execute_decision = "defer"
            adjudication_label = f"defer_{spec['blocker_layer']}_blocker"
            refusal_code = "deferred_assistant_side_blocker_v1"
        else:
            should_repair = False
            execute_decision = "defer"
            adjudication_label = f"defer_{spec['blocker_layer']}_blocker"
            refusal_code = "deferred_unspecified_v1"
        rows.append(
            {
                "blocker_id": blocker_id,
                "should_repair": should_repair,
                "repair_order": repair_order,
                "canonical_repair_steps": materialize_repair_steps(
                    blocker_id=blocker_id,
                    resolved_customer_id=stage2_oracle["resolved_customer_id"],
                    resolved_line_id=stage2_oracle["resolved_line_id"],
                ),
                "oracle_execute_decision": execute_decision,
                "adjudication_label": adjudication_label,
                "refusal_code": refusal_code,
                "depends_on": [dep for dep in spec["depends_on"] if dep in present],
            }
        )
    return {
        "per_blocker": rows,
        "repairability": decision["repairability"],
        "transfer_reason": decision["transfer_reason"],
        "decision_policy_version": decision["decision_policy_version"],
    }


def build_stage5_oracle(stage4_oracle: dict[str, Any]) -> dict[str, Any]:
    selected_blocker_ids = [
        row["blocker_id"]
        for row in stage4_oracle["per_blocker"]
        if row["oracle_execute_decision"] == "repair"
    ]
    deferred_blocker_ids = [
        row["blocker_id"]
        for row in stage4_oracle["per_blocker"]
        if row["oracle_execute_decision"] != "repair"
    ]
    repairability = stage4_oracle["repairability"]
    if repairability == "transfer_required":
        final_action = "transfer"
        verification_plan = {
            "required_postchecks": [],
            "success_condition": "transfer_required",
        }
    elif deferred_blocker_ids:
        final_action = "repair_subset"
        verification_plan = {
            "required_postchecks": [],
            "success_condition": "partial_resolution_only",
        }
    else:
        final_action = "repair_all"
        verification_plan = {
            "required_postchecks": ["can_send_mms"],
            "success_condition": "can_send_mms_true",
        }
    return {
        "final_action": final_action,
        "selected_blocker_ids": selected_blocker_ids,
        "deferred_blocker_ids": deferred_blocker_ids,
        "response_mode": "telecom_structured_execution",
        "verification_plan": verification_plan,
        "transfer_reason": stage4_oracle.get("transfer_reason"),
        "cancelled_reservation_ids": selected_blocker_ids,
        "refused_reservation_ids": deferred_blocker_ids,
    }


def build_source_evaluation_criteria(
    raw_evaluation_criteria: dict[str, Any],
    stage4_oracle: dict[str, Any],
    stage5_oracle: dict[str, Any],
) -> dict[str, Any]:
    final_action = stage5_oracle["final_action"]
    selected_blocker_ids = set(stage5_oracle["selected_blocker_ids"])

    selected_actions: list[dict[str, Any]] = []
    action_counter = 0
    for row in stage4_oracle["per_blocker"]:
        if row["blocker_id"] not in selected_blocker_ids:
            continue
        for step in row["canonical_repair_steps"]:
            selected_actions.append(
                {
                    "action_id": f"{step['tool_name']}_{action_counter}",
                    "requestor": step["requestor"],
                    "name": step["tool_name"],
                    "arguments": deepcopy(step.get("arguments", {})),
                    "info": None,
                    "compare_args": None,
                }
            )
            action_counter += 1

    derived = {
        "actions": selected_actions,
        "env_assertions": [],
        "communicate_info": raw_evaluation_criteria.get("communicate_info"),
        "nl_assertions": raw_evaluation_criteria.get("nl_assertions"),
        "reward_basis": ["ENV_ASSERTION"] if final_action == "repair_all" else ["TERMINAL_DECISION"],
        "success_mode": stage5_oracle["verification_plan"]["success_condition"],
        "expected_terminal_action": final_action,
        "selected_blocker_ids": stage5_oracle["selected_blocker_ids"],
        "deferred_blocker_ids": stage5_oracle["deferred_blocker_ids"],
        "transfer_reason": stage5_oracle.get("transfer_reason"),
    }

    if final_action == "repair_all":
        derived["env_assertions"] = deepcopy(raw_evaluation_criteria.get("env_assertions", []))

    return derived


def build_metadata(
    parsed_task_id: dict[str, Any],
    source_split: str,
    source_subsplit: str,
    subset_version: str,
    stage4_oracle: dict[str, Any],
    stage5_oracle: dict[str, Any],
    smoke_candidate: bool,
) -> dict[str, Any]:
    blocker_ids = parsed_task_id["blockers"]
    specs = [CANONICAL_BLOCKER_SPECS[bid] for bid in blocker_ids]
    repair_owners = {spec["repair_owner"] for spec in specs}
    return {
        "domain": "telecom",
        "family_version": "telecom_mms_fixed_tree_v1",
        "subset_version": subset_version,
        "source_split": source_split,
        "source_subsplit": source_subsplit,
        "problem_family": "mms_issue",
        "persona_level": parsed_task_id["persona"],
        "num_blockers": len(blocker_ids),
        "blocker_layers_present": sorted({spec["blocker_layer"] for spec in specs}),
        "contains_assistant_side_action": any(
            spec["assistant_side_required"] for spec in specs
        ),
        "contains_user_side_action": any(spec["user_side_required"] for spec in specs),
        "contains_hybrid_action": any(spec["hybrid_required"] for spec in specs),
        "requires_roaming_account_check": any(
            blocker_id.startswith("user_abroad_roaming_") for blocker_id in blocker_ids
        ),
        "requires_data_refuel": "data_usage_exceeded" in blocker_ids,
        "expected_terminal_action": stage5_oracle["final_action"],
        "target_success_signal": "can_send_mms_true",
        "smoke_candidate": smoke_candidate,
        "builder_version": "v1",
        "stage2_oracle_mode": "static_precomputed_v1",
        "stage3_oracle_mode": "heuristic_precomputed_v1",
        "repairability": stage4_oracle["repairability"],
        "repair_owner_set": sorted(repair_owners),
        "decision_policy_version": stage4_oracle.get("decision_policy_version", "first_pass_v1"),
        "num_selected_blockers": len(stage5_oracle["selected_blocker_ids"]),
        "num_deferred_blockers": len(stage5_oracle["deferred_blocker_ids"]),
    }


def make_instance_id(task_id: str) -> str:
    normalized = task_id
    for src, dst in {
        "[": "",
        "]": "",
        "|": "__",
        ":": "_",
        " ": "_",
    }.items():
        normalized = normalized.replace(src, dst)
    return f"telecom_mms_task_{normalized}"


def build_one(
    raw_task: dict[str, Any],
    source_split: str,
    source_subsplit: str,
    subset_version: str,
    smoke_candidate: bool = False,
    reference_maps: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    parsed_task_id = parse_task_id(raw_task["id"])
    user_context = extract_user_context(raw_task)
    stage1_input = deepcopy(user_context)
    stage1_oracle = build_stage1_oracle(user_context)
    stage2_input = {
        "customer_lookup": stage1_oracle["customer_lookup"],
        "line_selector": stage1_oracle["line_selector"],
    }
    assert reference_maps is not None
    stage2_oracle = compute_stage2_oracle(reference_maps, raw_task, stage1_oracle)
    stage3_input = {
        "resolved_customer_id": stage2_oracle["resolved_customer_id"],
        "resolved_line_id": stage2_oracle["resolved_line_id"],
        "target_phone_number": stage2_oracle["target_phone_number"],
        "problem_family": "mms_issue",
    }
    stage3_oracle = compute_stage3_oracle(
        parsed_task_id=parsed_task_id,
        stage2_oracle=stage2_oracle,
        user_context=user_context,
    )
    stage4_input = {
        "observed_state": {
            k: stage3_oracle["observed_state"][k]
            for k in [
                "can_send_mms",
                "service_status",
                "mobile_data_working",
                "internet_speed_desc",
            ]
        },
        "per_blocker": [{"blocker_id": bid} for bid in parsed_task_id["blockers"]],
    }
    stage4_oracle = build_stage4_oracle(parsed_task_id["blockers"], stage2_oracle)
    stage5_input = {
        "repairability": stage4_oracle["repairability"],
        "per_blocker": [
            {
                "blocker_id": row["blocker_id"],
                "should_repair": row["should_repair"],
            }
            for row in stage4_oracle["per_blocker"]
        ],
    }
    stage5_oracle = build_stage5_oracle(stage4_oracle)
    source_evaluation_criteria = build_source_evaluation_criteria(
        raw_evaluation_criteria=raw_task["evaluation_criteria"],
        stage4_oracle=stage4_oracle,
        stage5_oracle=stage5_oracle,
    )
    metadata = build_metadata(
        parsed_task_id=parsed_task_id,
        source_split=source_split,
        source_subsplit=source_subsplit,
        subset_version=subset_version,
        stage4_oracle=stage4_oracle,
        stage5_oracle=stage5_oracle,
        smoke_candidate=smoke_candidate,
    )
    return {
        "instance_id": make_instance_id(raw_task["id"]),
        "family": "telecom_mms_recovery",
        "original_task_id": raw_task["id"],
        "source_task": {
            "description": raw_task["description"],
            "evaluation_criteria": source_evaluation_criteria,
        },
        "user_context": user_context,
        "stage1": {"input": stage1_input, "oracle_output": stage1_oracle},
        "stage2": {"input": stage2_input, "oracle_output": stage2_oracle},
        "stage3": {"input": stage3_input, "oracle_output": stage3_oracle},
        "stage4": {"input": stage4_input, "oracle_output": stage4_oracle},
        "stage5": {"input": stage5_input, "oracle_output": stage5_oracle},
        "metadata": metadata,
    }


def select_mms_tasks(all_tasks: list[dict[str, Any]], split_ids: list[str]) -> list[dict[str, Any]]:
    task_map = {task["id"]: task for task in all_tasks}
    selected = []
    for task_id in split_ids:
        parsed = parse_task_id(task_id)
        if parsed["family"] == "mms_issue":
            selected.append(task_map[task_id])
    return sorted(selected, key=lambda task: (len(parse_task_id(task["id"])["blockers"]), task["id"]))


def build_smoke10_manifest(base_tasks: list[dict[str, Any]]) -> dict[str, Any]:
    base_ids = {task["id"] for task in base_tasks}
    missing = [task_id for task_id in SMOKE10_TASK_IDS if task_id not in base_ids]
    if missing:
        raise ValueError(f"Smoke10 task ids missing from telecom base split: {missing}")
    return {
        "subset_name": "telecom_mms_fixed_tree_smoke10",
        "source_subset": "base",
        "family": "telecom_mms_recovery",
        "selection_criteria": {
            "only_from_base": True,
            "exclude_small": True,
            "cover_persona_levels": ["None", "Easy", "Hard"],
            "cover_blocker_counts": ["2", "3", "4", "6", "9"],
            "cover_layers": ["service", "data", "mms_app"],
            "cover_action_ownership": ["user", "assistant", "hybrid"],
        },
        "notes": [
            "Smoke10 is a fixed handpicked subset of base.",
            "Subset and transfer examples exist inside smoke10, but not every sanity-check example from base is included.",
            "In particular, [mms_issue]bad_network_preference|user_abroad_roaming_disabled_on[PERSONA:None] is a base-only subset sample, not a smoke10 sample.",
        ],
        "sanity_check_examples": {
            "subset_in_smoke10": "[mms_issue]break_app_storage_permission|data_usage_exceeded[PERSONA:Easy]",
            "transfer_in_smoke10": "[mms_issue]bad_network_preference|user_abroad_roaming_disabled_off[PERSONA:None]",
            "base_only_subset_example": "[mms_issue]bad_network_preference|user_abroad_roaming_disabled_on[PERSONA:None]",
        },
        "task_ids": list(SMOKE10_TASK_IDS),
    }


def build_dataset(
    tasks: list[dict[str, Any]],
    source_split: str,
    source_subsplit_lookup: dict[str, str],
    subset_version: str,
    smoke_task_ids: set[str] | None = None,
    reference_maps: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for raw_task in tasks:
        try:
            rows.append(
                build_one(
                    raw_task=raw_task,
                    source_split=source_split,
                    source_subsplit=source_subsplit_lookup.get(raw_task["id"], "unknown"),
                    subset_version=subset_version,
                    smoke_candidate=raw_task["id"] in (smoke_task_ids or set()),
                    reference_maps=reference_maps,
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.warn(f"Skipping task {raw_task['id']}: {exc}")
    return rows


def main() -> None:
    all_tasks = load_json(TELECOM_TASKS_PATH)
    split_map = load_json(TELECOM_SPLITS_PATH)
    reference_maps = build_reference_maps(load_telecom_reference_db())

    source_subsplit_lookup: dict[str, str] = {}
    for task_id in split_map["train"]:
        source_subsplit_lookup[task_id] = "train"
    for task_id in split_map["test"]:
        source_subsplit_lookup[task_id] = "test"

    base_tasks = select_mms_tasks(all_tasks, split_map["base"])
    manifest = build_smoke10_manifest(base_tasks)
    smoke_raw = [task for task in base_tasks if task["id"] in set(manifest["task_ids"])]

    base_rows = build_dataset(
        tasks=base_tasks,
        source_split="base",
        source_subsplit_lookup=source_subsplit_lookup,
        subset_version="base",
        smoke_task_ids=set(manifest["task_ids"]),
        reference_maps=reference_maps,
    )
    smoke_rows = build_dataset(
        tasks=smoke_raw,
        source_split="base",
        source_subsplit_lookup=source_subsplit_lookup,
        subset_version="smoke10",
        smoke_task_ids=set(manifest["task_ids"]),
        reference_maps=reference_maps,
    )

    dump_json(OUTPUT_BASE_PATH, base_rows)
    dump_json(OUTPUT_SMOKE_PATH, smoke_rows)
    dump_json(OUTPUT_MANIFEST_PATH, manifest)

    print(
        json.dumps(
            {
                "base_count": len(base_rows),
                "smoke_count": len(smoke_rows),
                "base_output": str(OUTPUT_BASE_PATH),
                "smoke_output": str(OUTPUT_SMOKE_PATH),
                "manifest_output": str(OUTPUT_MANIFEST_PATH),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
