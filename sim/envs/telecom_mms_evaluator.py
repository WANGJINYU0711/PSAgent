from __future__ import annotations

from dataclasses import dataclass
from typing import Any


TELECOM_MMS_COST_SCALE_VERSION = "telecom_mms_cost_norm_v2_full_trajectory_dual_reasoning"


@dataclass(frozen=True)
class TelecomMMSCostSpec:
    final_action_mismatch_penalty: float = 1.0
    false_selected_penalty: float = 1.5
    missed_selected_penalty: float = 1.0
    false_deferred_penalty: float = 1.0
    missed_deferred_penalty: float = 1.0
    invalid_transfer_penalty: float = 1.5
    missed_transfer_penalty: float = 1.5
    policy_action_violation_penalty: float = 2.0
    policy_communication_violation_penalty: float = 1.0
    policy_nl_assertion_violation_penalty: float = 0.5
    path_agent_cost_weight: float = 0.1


DEFAULT_COST_SPEC = TelecomMMSCostSpec()


# Outcome normalization stays on the original telecom task-level cap; v2 extends it
# with explicit policy and reasoning envelopes instead of implicitly folding them into
# the old terminal/total constants.
TELECOM_MMS_OUTCOME_UPPER_BOUND_V2 = 25.0
TELECOM_MMS_POLICY_MAX_NL_ASSERTIONS_FAILED_V2 = 0
TELECOM_MMS_POLICY_UPPER_BOUND_V2 = (
    DEFAULT_COST_SPEC.policy_action_violation_penalty
    + DEFAULT_COST_SPEC.policy_communication_violation_penalty
    + (
        DEFAULT_COST_SPEC.policy_nl_assertion_violation_penalty
        * TELECOM_MMS_POLICY_MAX_NL_ASSERTIONS_FAILED_V2
    )
)
TELECOM_MMS_TERMINAL_UPPER_BOUND_V2 = (
    TELECOM_MMS_OUTCOME_UPPER_BOUND_V2 + TELECOM_MMS_POLICY_UPPER_BOUND_V2
)
TELECOM_MMS_PATH_UPPER_BOUND_V2 = 0.14

# Legacy aliases kept for compatibility with older helper imports.
TELECOM_MMS_MAX_RAW_TERMINAL_COST_V1 = TELECOM_MMS_OUTCOME_UPPER_BOUND_V2
TELECOM_MMS_MAX_WEIGHTED_PATH_COST_V1 = TELECOM_MMS_PATH_UPPER_BOUND_V2


def _normalize_id_set(values: Any) -> set[str]:
    if values is None:
        return set()
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def evaluate_terminal_prediction(
    instance: dict[str, Any],
    predicted_stage5_output: dict[str, Any],
    cost_spec: TelecomMMSCostSpec | None = None,
    policy_eval_result: dict[str, Any] | None = None,
) -> dict[str, Any]:
    spec = cost_spec or DEFAULT_COST_SPEC
    oracle = instance["stage5"]["oracle_output"]
    policy_eval = dict(policy_eval_result or {})

    predicted_final_action = predicted_stage5_output.get("final_action", "transfer")
    oracle_final_action = oracle["final_action"]

    predicted_selected = _normalize_id_set(
        predicted_stage5_output.get("selected_blocker_ids")
        or predicted_stage5_output.get("cancelled_reservation_ids")
    )
    predicted_deferred = _normalize_id_set(
        predicted_stage5_output.get("deferred_blocker_ids")
        or predicted_stage5_output.get("refused_reservation_ids")
    )
    oracle_selected = _normalize_id_set(
        oracle.get("selected_blocker_ids") or oracle.get("cancelled_reservation_ids")
    )
    oracle_deferred = _normalize_id_set(
        oracle.get("deferred_blocker_ids") or oracle.get("refused_reservation_ids")
    )

    final_action_mismatch = predicted_final_action != oracle_final_action
    false_selected = sorted(predicted_selected - oracle_selected)
    missed_selected = sorted(oracle_selected - predicted_selected)
    false_deferred = sorted(predicted_deferred - oracle_deferred)
    missed_deferred = sorted(oracle_deferred - predicted_deferred)
    subset_mismatch = bool(
        false_selected or missed_selected or false_deferred or missed_deferred
    )

    invalid_transfer_penalty = 0.0
    if predicted_final_action == "transfer" and oracle_final_action != "transfer":
        invalid_transfer_penalty = spec.invalid_transfer_penalty
    missed_transfer_penalty = 0.0
    if oracle_final_action == "transfer" and predicted_final_action != "transfer":
        missed_transfer_penalty = spec.missed_transfer_penalty

    outcome_cost_breakdown = {
        "final_action_mismatch_penalty": (
            spec.final_action_mismatch_penalty if final_action_mismatch else 0.0
        ),
        "false_selected_penalty": len(false_selected) * spec.false_selected_penalty,
        "missed_selected_penalty": len(missed_selected) * spec.missed_selected_penalty,
        "false_deferred_penalty": len(false_deferred) * spec.false_deferred_penalty,
        "missed_deferred_penalty": len(missed_deferred) * spec.missed_deferred_penalty,
        "invalid_transfer_penalty": invalid_transfer_penalty,
        "missed_transfer_penalty": missed_transfer_penalty,
    }
    raw_outcome_penalty = sum(outcome_cost_breakdown.values())

    policy_action_violation = bool(policy_eval.get("policy_action_violation", False))
    policy_communication_violation = bool(
        policy_eval.get("policy_communication_violation", False)
    )
    policy_nl_assertions_total = int(
        policy_eval.get("policy_nl_assertions_total", 0) or 0
    )
    policy_nl_assertions_failed = int(
        policy_eval.get("policy_nl_assertions_failed", 0) or 0
    )
    policy_violation_count = int(policy_action_violation) + int(
        policy_communication_violation
    ) + policy_nl_assertions_failed

    policy_cost_breakdown = {
        "policy_action_violation_penalty": (
            spec.policy_action_violation_penalty if policy_action_violation else 0.0
        ),
        "policy_communication_violation_penalty": (
            spec.policy_communication_violation_penalty
            if policy_communication_violation
            else 0.0
        ),
        "policy_nl_assertion_violation_penalty": (
            policy_nl_assertions_failed * spec.policy_nl_assertion_violation_penalty
        ),
    }
    raw_policy_penalty = sum(policy_cost_breakdown.values())

    terminal_cost_breakdown = {
        **outcome_cost_breakdown,
        **policy_cost_breakdown,
    }
    raw_terminal_penalty = raw_outcome_penalty + raw_policy_penalty
    normalized_terminal_penalty = min(
        raw_terminal_penalty / TELECOM_MMS_TERMINAL_UPPER_BOUND_V2,
        1.0,
    )
    exact_match = (not final_action_mismatch) and (not subset_mismatch)
    policy_compliant = policy_violation_count == 0

    return {
        "evaluator_version": "telecom_mms_blocker_level_v3_policy",
        "cost_scale_version": TELECOM_MMS_COST_SCALE_VERSION,
        "outcome_cost_upper_bound": TELECOM_MMS_OUTCOME_UPPER_BOUND_V2,
        "policy_cost_upper_bound": TELECOM_MMS_POLICY_UPPER_BOUND_V2,
        "terminal_cost_upper_bound": TELECOM_MMS_TERMINAL_UPPER_BOUND_V2,
        "path_cost_upper_bound": TELECOM_MMS_PATH_UPPER_BOUND_V2,
        "predicted_final_action": predicted_final_action,
        "oracle_final_action": oracle_final_action,
        "final_action_mismatch": final_action_mismatch,
        "subset_mismatch": subset_mismatch,
        "false_selected_blocker_ids": false_selected,
        "missed_selected_blocker_ids": missed_selected,
        "false_deferred_blocker_ids": false_deferred,
        "missed_deferred_blocker_ids": missed_deferred,
        "false_selected_count": len(false_selected),
        "missed_selected_count": len(missed_selected),
        "false_deferred_count": len(false_deferred),
        "missed_deferred_count": len(missed_deferred),
        "raw_outcome_penalty": raw_outcome_penalty,
        "raw_policy_penalty": raw_policy_penalty,
        "raw_terminal_penalty": raw_terminal_penalty,
        "normalized_terminal_penalty": normalized_terminal_penalty,
        "terminal_penalty": raw_terminal_penalty,
        "exact_match": exact_match,
        "policy_compliant": policy_compliant,
        "policy_action_violation": policy_action_violation,
        "policy_communication_violation": policy_communication_violation,
        "policy_nl_assertions_total": policy_nl_assertions_total,
        "policy_nl_assertions_failed": policy_nl_assertions_failed,
        "policy_violation_count": policy_violation_count,
        "policy_violation_breakdown": {
            "policy_action_violation": policy_action_violation,
            "policy_communication_violation": policy_communication_violation,
            "policy_nl_assertions_failed": policy_nl_assertions_failed,
        },
        "outcome_cost_breakdown": outcome_cost_breakdown,
        "policy_cost_breakdown": policy_cost_breakdown,
        "terminal_cost_breakdown": terminal_cost_breakdown,
        "cost_breakdown": terminal_cost_breakdown,
        "bench_action_check_raw": list(policy_eval.get("bench_action_check_raw") or []),
        "bench_communicate_check_raw": list(
            policy_eval.get("bench_communicate_check_raw") or []
        ),
        "bench_nl_assertions_raw": list(policy_eval.get("bench_nl_assertions_raw") or []),
        "bench_action_check": policy_eval.get("bench_action_check"),
        "bench_communicate_check": policy_eval.get("bench_communicate_check"),
        "bench_nl_assertions": policy_eval.get("bench_nl_assertions"),
        "policy_eval_source": policy_eval.get("policy_eval_source"),
        "policy_eval_scope": policy_eval.get("policy_eval_scope"),
        "false_cancelled_ids": false_selected,
        "missed_cancelled_ids": missed_selected,
        "false_refused_ids": false_deferred,
        "missed_refused_ids": missed_deferred,
        "false_cancel_count": len(false_selected),
        "missed_cancel_count": len(missed_selected),
        "false_refuse_count": len(false_deferred),
        "missed_refuse_count": len(missed_deferred),
    }
