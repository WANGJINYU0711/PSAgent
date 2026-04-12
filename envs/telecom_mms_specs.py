from __future__ import annotations

from copy import deepcopy
from typing import Any


CANONICAL_BLOCKER_SPECS: dict[str, dict[str, Any]] = {
    "airplane_mode_on": {
        "blocker_id": "airplane_mode_on",
        "blocker_layer": "service",
        "repair_owner": "user",
        "repair_action_family": "toggle_airplane_mode",
        "canonical_repair_steps": [
            {"requestor": "user", "tool_name": "toggle_airplane_mode", "arguments": {}}
        ],
        "verification_signal": "service_restored",
        "default_priority": 10,
        "depends_on": [],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Most upstream blocker; clear before any data or MMS-specific repair.",
    },
    "unseat_sim_card": {
        "blocker_id": "unseat_sim_card",
        "blocker_layer": "service",
        "repair_owner": "user",
        "repair_action_family": "reseat_sim_card",
        "canonical_repair_steps": [
            {"requestor": "user", "tool_name": "reseat_sim_card", "arguments": {}}
        ],
        "verification_signal": "service_restored",
        "default_priority": 20,
        "depends_on": ["airplane_mode_on"],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "If present with airplane mode, clear airplane mode first.",
    },
    "data_mode_off": {
        "blocker_id": "data_mode_off",
        "blocker_layer": "data",
        "repair_owner": "user",
        "repair_action_family": "toggle_data",
        "canonical_repair_steps": [
            {"requestor": "user", "tool_name": "toggle_data", "arguments": {}}
        ],
        "verification_signal": "mobile_data_available",
        "default_priority": 30,
        "depends_on": ["airplane_mode_on", "unseat_sim_card"],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Global mobile data switch; should be fixed before roaming or MMS app checks.",
    },
    "user_abroad_roaming_disabled_on": {
        "blocker_id": "user_abroad_roaming_disabled_on",
        "blocker_layer": "data",
        "repair_owner": "assistant",
        "repair_action_family": "enable_roaming",
        "canonical_repair_steps": [
            {
                "requestor": "assistant",
                "tool_name": "enable_roaming",
                "arguments": {
                    "customer_id": "<resolved_customer_id>",
                    "line_id": "<resolved_line_id>",
                },
            }
        ],
        "verification_signal": "mobile_data_available",
        "default_priority": 35,
        "depends_on": ["airplane_mode_on", "unseat_sim_card", "data_mode_off"],
        "assistant_side_required": True,
        "user_side_required": False,
        "hybrid_required": False,
        "can_be_deferred": True,
        "notes": "User abroad; account roaming disabled while device roaming already on.",
    },
    "user_abroad_roaming_enabled_off": {
        "blocker_id": "user_abroad_roaming_enabled_off",
        "blocker_layer": "data",
        "repair_owner": "user",
        "repair_action_family": "toggle_roaming",
        "canonical_repair_steps": [
            {"requestor": "user", "tool_name": "toggle_roaming", "arguments": {}}
        ],
        "verification_signal": "mobile_data_available",
        "default_priority": 36,
        "depends_on": ["airplane_mode_on", "unseat_sim_card", "data_mode_off"],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "User abroad; account roaming enabled while device roaming off.",
    },
    "user_abroad_roaming_disabled_off": {
        "blocker_id": "user_abroad_roaming_disabled_off",
        "blocker_layer": "data",
        "repair_owner": "hybrid",
        "repair_action_family": "enable_and_toggle_roaming",
        "canonical_repair_steps": [
            {
                "requestor": "assistant",
                "tool_name": "enable_roaming",
                "arguments": {
                    "customer_id": "<resolved_customer_id>",
                    "line_id": "<resolved_line_id>",
                },
            },
            {"requestor": "user", "tool_name": "toggle_roaming", "arguments": {}},
        ],
        "verification_signal": "mobile_data_available",
        "default_priority": 37,
        "depends_on": ["airplane_mode_on", "unseat_sim_card", "data_mode_off"],
        "assistant_side_required": True,
        "user_side_required": True,
        "hybrid_required": True,
        "can_be_deferred": False,
        "notes": "User abroad; both account-side and device-side roaming are off.",
    },
    "data_usage_exceeded": {
        "blocker_id": "data_usage_exceeded",
        "blocker_layer": "data",
        "repair_owner": "assistant",
        "repair_action_family": "refuel_data",
        "canonical_repair_steps": [
            {
                "requestor": "assistant",
                "tool_name": "refuel_data",
                "arguments": {
                    "customer_id": "<resolved_customer_id>",
                    "line_id": "<resolved_line_id>",
                    "gb_amount": 2.0,
                },
            }
        ],
        "verification_signal": "mobile_data_available",
        "default_priority": 40,
        "depends_on": ["airplane_mode_on", "unseat_sim_card", "data_mode_off"],
        "assistant_side_required": True,
        "user_side_required": False,
        "hybrid_required": False,
        "can_be_deferred": True,
        "notes": "V1 assumes user allows a 2.0 GB refuel when requested.",
    },
    "bad_network_preference": {
        "blocker_id": "bad_network_preference",
        "blocker_layer": "data",
        "repair_owner": "user",
        "repair_action_family": "set_network_mode_preference",
        "canonical_repair_steps": [
            {
                "requestor": "user",
                "tool_name": "set_network_mode_preference",
                "arguments": {"mode": "4g_5g_preferred"},
            }
        ],
        "verification_signal": "network_mode_sufficient",
        "default_priority": 45,
        "depends_on": ["airplane_mode_on", "unseat_sim_card", "data_mode_off"],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Bad network preference is set to 2g_only in the source tasks.",
    },
    "bad_wifi_calling": {
        "blocker_id": "bad_wifi_calling",
        "blocker_layer": "mms_app",
        "repair_owner": "user",
        "repair_action_family": "toggle_wifi_calling",
        "canonical_repair_steps": [
            {"requestor": "user", "tool_name": "toggle_wifi_calling", "arguments": {}}
        ],
        "verification_signal": "can_send_mms_true",
        "default_priority": 60,
        "depends_on": [
            "airplane_mode_on",
            "unseat_sim_card",
            "data_mode_off",
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
            "data_usage_exceeded",
            "bad_network_preference",
        ],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Carrier/device combination does not support MMS over Wi-Fi in these tasks.",
    },
    "break_apn_mms_setting": {
        "blocker_id": "break_apn_mms_setting",
        "blocker_layer": "mms_app",
        "repair_owner": "user",
        "repair_action_family": "reset_apn_then_reboot",
        "canonical_repair_steps": [
            {"requestor": "user", "tool_name": "reset_apn_settings", "arguments": {}},
            {"requestor": "user", "tool_name": "reboot_device", "arguments": {}},
        ],
        "verification_signal": "can_send_mms_true",
        "default_priority": 70,
        "depends_on": [
            "airplane_mode_on",
            "unseat_sim_card",
            "data_mode_off",
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
            "data_usage_exceeded",
            "bad_network_preference",
        ],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "The source task breaks MMSC URL and expects reset_apn_settings plus reboot_device.",
    },
    "break_app_sms_permission": {
        "blocker_id": "break_app_sms_permission",
        "blocker_layer": "mms_app",
        "repair_owner": "user",
        "repair_action_family": "grant_app_permission_sms",
        "canonical_repair_steps": [
            {
                "requestor": "user",
                "tool_name": "grant_app_permission",
                "arguments": {"app_name": "messaging", "permission": "sms"},
            }
        ],
        "verification_signal": "can_send_mms_true",
        "default_priority": 80,
        "depends_on": [
            "airplane_mode_on",
            "unseat_sim_card",
            "data_mode_off",
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
            "data_usage_exceeded",
            "bad_network_preference",
        ],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Single missing SMS permission on the messaging app.",
    },
    "break_app_storage_permission": {
        "blocker_id": "break_app_storage_permission",
        "blocker_layer": "mms_app",
        "repair_owner": "user",
        "repair_action_family": "grant_app_permission_storage",
        "canonical_repair_steps": [
            {
                "requestor": "user",
                "tool_name": "grant_app_permission",
                "arguments": {"app_name": "messaging", "permission": "storage"},
            }
        ],
        "verification_signal": "can_send_mms_true",
        "default_priority": 81,
        "depends_on": [
            "airplane_mode_on",
            "unseat_sim_card",
            "data_mode_off",
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
            "data_usage_exceeded",
            "bad_network_preference",
        ],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Single missing storage permission on the messaging app.",
    },
    "break_app_both_permissions": {
        "blocker_id": "break_app_both_permissions",
        "blocker_layer": "mms_app",
        "repair_owner": "user",
        "repair_action_family": "grant_app_permissions_both",
        "canonical_repair_steps": [
            {
                "requestor": "user",
                "tool_name": "grant_app_permission",
                "arguments": {"app_name": "messaging", "permission": "sms"},
            },
            {
                "requestor": "user",
                "tool_name": "grant_app_permission",
                "arguments": {"app_name": "messaging", "permission": "storage"},
            },
        ],
        "verification_signal": "can_send_mms_true",
        "default_priority": 82,
        "depends_on": [
            "airplane_mode_on",
            "unseat_sim_card",
            "data_mode_off",
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
            "data_usage_exceeded",
            "bad_network_preference",
        ],
        "assistant_side_required": False,
        "user_side_required": True,
        "hybrid_required": False,
        "can_be_deferred": False,
        "notes": "Superset app permission blocker; builder should not emit this together with the single-permission blockers.",
    },
}


def first_pass_terminal_decision(blocker_ids: list[str]) -> dict[str, Any]:
    ordered_blocker_ids = sort_blocker_ids(blocker_ids)
    specs = [CANONICAL_BLOCKER_SPECS[blocker_id] for blocker_id in ordered_blocker_ids]

    if any(spec["hybrid_required"] for spec in specs):
        return {
            "final_action": "transfer",
            "selected_blocker_ids": [],
            "deferred_blocker_ids": ordered_blocker_ids,
            "repairability": "transfer_required",
            "transfer_reason": "hybrid_blocker_requires_transfer_v1",
            "decision_policy_version": "first_pass_v1",
        }

    selected_blocker_ids: list[str] = []
    deferred_blocker_ids: list[str] = []
    for blocker_id, spec in zip(ordered_blocker_ids, specs):
        if spec["assistant_side_required"] and spec["can_be_deferred"]:
            deferred_blocker_ids.append(blocker_id)
        else:
            selected_blocker_ids.append(blocker_id)

    if deferred_blocker_ids:
        return {
            "final_action": "repair_subset" if selected_blocker_ids else "transfer",
            "selected_blocker_ids": selected_blocker_ids,
            "deferred_blocker_ids": deferred_blocker_ids,
            "repairability": "partially_repairable" if selected_blocker_ids else "transfer_required",
            "transfer_reason": None if selected_blocker_ids else "no_safe_auto_subset_v1",
            "decision_policy_version": "first_pass_v1",
        }

    return {
        "final_action": "repair_all",
        "selected_blocker_ids": selected_blocker_ids,
        "deferred_blocker_ids": deferred_blocker_ids,
        "repairability": "repairable",
        "transfer_reason": None,
        "decision_policy_version": "first_pass_v1",
    }


def get_blocker_spec(blocker_id: str) -> dict[str, Any]:
    return deepcopy(CANONICAL_BLOCKER_SPECS[blocker_id])


def materialize_repair_steps(
    blocker_id: str,
    resolved_customer_id: str,
    resolved_line_id: str,
) -> list[dict[str, Any]]:
    steps = deepcopy(CANONICAL_BLOCKER_SPECS[blocker_id]["canonical_repair_steps"])
    for step in steps:
        args = step.get("arguments", {})
        if args.get("customer_id") == "<resolved_customer_id>":
            args["customer_id"] = resolved_customer_id
        if args.get("line_id") == "<resolved_line_id>":
            args["line_id"] = resolved_line_id
    return steps


def blocker_diagnostic_evidence(blocker_id: str) -> list[str]:
    evidence_map = {
        "airplane_mode_on": ["check_network_status"],
        "unseat_sim_card": ["check_sim_status", "check_network_status"],
        "data_mode_off": ["check_network_status", "run_speed_test"],
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


def infer_blocker_ids_from_observed_state(observed_state: dict[str, Any]) -> list[str]:
    state = dict(observed_state or {})
    inferred: list[str] = []

    airplane_mode = state.get("airplane_mode") is True
    sim_status = state.get("sim_status")
    mobile_data_working = state.get("mobile_data_working")
    data_usage_exceeded = state.get("data_usage_exceeded") is True
    is_abroad = state.get("is_abroad") is True
    roaming_on_account = state.get("roaming_enabled_on_account") is True
    roaming_on_device = state.get("roaming_enabled_on_device") is True
    network_mode = state.get("network_mode_preference")
    wifi_calling_enabled = state.get("wifi_calling_enabled") is True
    apn_mms_ok = state.get("apn_mms_ok")
    sms_permission = state.get("messaging_sms_permission")
    storage_permission = state.get("messaging_storage_permission")

    if airplane_mode:
        inferred.append("airplane_mode_on")

    if sim_status == "missing":
        inferred.append("unseat_sim_card")

    if data_usage_exceeded:
        inferred.append("data_usage_exceeded")

    if is_abroad:
        if (not roaming_on_account) and roaming_on_device:
            inferred.append("user_abroad_roaming_disabled_on")
        elif roaming_on_account and (not roaming_on_device):
            inferred.append("user_abroad_roaming_enabled_off")
        elif (not roaming_on_account) and (not roaming_on_device):
            inferred.append("user_abroad_roaming_disabled_off")

    if network_mode == "2g_only":
        inferred.append("bad_network_preference")

    # This is a conservative proxy for the mobile-data master toggle:
    # data is not working, the failure is not already explained by airplane mode,
    # missing SIM, roaming mismatch, or data-cap exhaustion.
    roaming_blocker_present = any(
        blocker_id in inferred
        for blocker_id in (
            "user_abroad_roaming_disabled_on",
            "user_abroad_roaming_enabled_off",
            "user_abroad_roaming_disabled_off",
        )
    )
    if (
        mobile_data_working is False
        and not airplane_mode
        and sim_status != "missing"
        and not data_usage_exceeded
        and not roaming_blocker_present
    ):
        inferred.append("data_mode_off")

    if wifi_calling_enabled:
        inferred.append("bad_wifi_calling")

    if apn_mms_ok is False:
        inferred.append("break_apn_mms_setting")

    if sms_permission is False and storage_permission is False:
        inferred.append("break_app_both_permissions")
    else:
        if sms_permission is False:
            inferred.append("break_app_sms_permission")
        if storage_permission is False:
            inferred.append("break_app_storage_permission")

    return sort_blocker_ids(inferred)


def sort_blocker_ids(blocker_ids: list[str]) -> list[str]:
    unique_ids = list(dict.fromkeys(blocker_ids))
    return sorted(unique_ids, key=lambda blocker_id: (CANONICAL_BLOCKER_SPECS[blocker_id]["default_priority"], blocker_id))


def build_per_blocker_from_ids(blocker_ids: list[str]) -> list[dict[str, Any]]:
    rows = []
    for blocker_id in sort_blocker_ids(blocker_ids):
        spec = get_blocker_spec(blocker_id)
        rows.append(
            {
                "blocker_id": blocker_id,
                "blocker_layer": spec["blocker_layer"],
                "repair_owner": spec["repair_owner"],
                "repair_action_family": spec["repair_action_family"],
                "diagnostic_evidence": blocker_diagnostic_evidence(blocker_id),
                "verification_signal": spec["verification_signal"],
            }
        )
    return rows
