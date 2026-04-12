from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TelecomMMSCostSpec:
    final_action_mismatch_penalty: float = 1.0
    false_selected_penalty: float = 1.5
    missed_selected_penalty: float = 1.0
    false_deferred_penalty: float = 1.0
    missed_deferred_penalty: float = 1.0
    invalid_transfer_penalty: float = 1.5
    missed_transfer_penalty: float = 1.5
    path_agent_cost_weight: float = 0.1


DEFAULT_COST_SPEC = TelecomMMSCostSpec()


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
) -> dict[str, Any]:
    spec = cost_spec or DEFAULT_COST_SPEC
    oracle = instance["stage5"]["oracle_output"]

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

    cost_breakdown = {
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
    terminal_penalty = sum(cost_breakdown.values())
    exact_match = (not final_action_mismatch) and (not subset_mismatch)

    return {
        "evaluator_version": "telecom_mms_blocker_level_v2",
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
        "cost_breakdown": cost_breakdown,
        "terminal_penalty": terminal_penalty,
        "exact_match": exact_match,
        "false_cancelled_ids": false_selected,
        "missed_cancelled_ids": missed_selected,
        "false_refused_ids": false_deferred,
        "missed_refused_ids": missed_deferred,
        "false_cancel_count": len(false_selected),
        "missed_cancel_count": len(missed_selected),
        "false_refuse_count": len(false_deferred),
        "missed_refuse_count": len(missed_deferred),
    }
