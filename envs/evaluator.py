"""Reservation-level evaluator for the fixed-tree derived benchmark."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


JsonDict = dict[str, Any]


@dataclass(frozen=True)
class CostSpec:
    final_action_mismatch_penalty: float = 1.0
    false_cancel_penalty: float = 1.5
    missed_cancel_penalty: float = 1.0
    false_refuse_penalty: float = 1.0
    missed_refuse_penalty: float = 1.0
    path_agent_cost_weight: float = 0.1


DEFAULT_COST_SPEC = CostSpec()
EVALUATOR_VERSION = "v2_reservation_level"


def evaluate_terminal_prediction(
    instance: JsonDict,
    predicted_stage5_output: JsonDict,
    cost_spec: CostSpec | None = None,
) -> JsonDict:
    """Evaluate one predicted terminal action against the structured oracle."""

    spec = cost_spec or DEFAULT_COST_SPEC
    oracle_output = instance.get("stage5", {}).get("oracle_output", {})
    metadata = instance.get("metadata", {})

    predicted_final_action = predicted_stage5_output.get("final_action")
    oracle_final_action = oracle_output.get("final_action")

    predicted_cancelled = set(predicted_stage5_output.get("cancelled_reservation_ids", []) or [])
    oracle_cancelled = set(oracle_output.get("cancelled_reservation_ids", []) or [])
    predicted_refused = set(predicted_stage5_output.get("refused_reservation_ids", []) or [])
    oracle_refused = set(oracle_output.get("refused_reservation_ids", []) or [])

    false_cancelled = sorted(predicted_cancelled - oracle_cancelled)
    missed_cancelled = sorted(oracle_cancelled - predicted_cancelled)
    false_refused = sorted(predicted_refused - oracle_refused)
    missed_refused = sorted(oracle_refused - predicted_refused)

    final_action_mismatch = predicted_final_action != oracle_final_action

    cost_breakdown = {
        "final_action_mismatch_penalty": (
            spec.final_action_mismatch_penalty if final_action_mismatch else 0.0
        ),
        "false_cancel_penalty": len(false_cancelled) * spec.false_cancel_penalty,
        "missed_cancel_penalty": len(missed_cancelled) * spec.missed_cancel_penalty,
        "false_refuse_penalty": len(false_refused) * spec.false_refuse_penalty,
        "missed_refuse_penalty": len(missed_refused) * spec.missed_refuse_penalty,
    }
    terminal_penalty = sum(cost_breakdown.values())

    tier = metadata.get("tier")
    subset_mismatch = (
        predicted_cancelled != oracle_cancelled or predicted_refused != oracle_refused
    )
    exact_match = (not final_action_mismatch) and (not subset_mismatch)

    return {
        "evaluator_version": EVALUATOR_VERSION,
        "cost_spec": asdict(spec),
        "tier": tier,
        "predicted_final_action": predicted_final_action,
        "oracle_final_action": oracle_final_action,
        "final_action_mismatch": final_action_mismatch,
        "subset_mismatch": subset_mismatch,
        "false_cancelled_ids": false_cancelled,
        "missed_cancelled_ids": missed_cancelled,
        "false_refused_ids": false_refused,
        "missed_refused_ids": missed_refused,
        "false_cancel_count": len(false_cancelled),
        "missed_cancel_count": len(missed_cancelled),
        "false_refuse_count": len(false_refused),
        "missed_refuse_count": len(missed_refused),
        "cost_breakdown": cost_breakdown,
        "terminal_penalty": terminal_penalty,
        "exact_match": exact_match,
    }
