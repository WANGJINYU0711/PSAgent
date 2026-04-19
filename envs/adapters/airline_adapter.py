"""Adapter from airline derived instances to reusable TaskDescriptor objects."""

from __future__ import annotations

from typing import Any

from tree_family.specs import CAPABILITY_NAMES, TaskDescriptor


class AirlineTaskAdapter:
    def build_task_descriptor(self, raw_instance: dict[str, Any]) -> TaskDescriptor:
        task_id = str(raw_instance.get("instance_id", raw_instance.get("original_task_id", "unknown")))
        metadata = raw_instance.get("metadata", {})
        final_action = raw_instance.get("stage5", {}).get("oracle_output", {}).get("final_action")

        weights = {key: 0.02 for key in CAPABILITY_NAMES}
        stage_difficulty = {stage: 0.2 for stage in ["stage1", "stage2", "stage3", "stage4", "stage5"]}

        tier = metadata.get("tier")
        if tier == "tier1_single_reservation":
            self._boost(weights, ["account_lookup", "line_resolution", "terminal_decision"], 0.12)
            stage_difficulty["stage2"] += 0.10
            stage_difficulty["stage4"] += 0.15
            stage_difficulty["stage5"] += 0.10
        else:
            self._boost(
                weights,
                ["account_lookup", "line_resolution", "repair_execution", "terminal_decision"],
                0.14,
            )
            self._boost(weights, ["verification"], 0.06)
            stage_difficulty["stage2"] += 0.30
            stage_difficulty["stage4"] += 0.22
            stage_difficulty["stage5"] += 0.28

        if metadata.get("contains_user_pressure"):
            self._boost(weights, ["user_grounding", "verification"], 0.10)
            stage_difficulty["stage1"] += 0.12
            stage_difficulty["stage4"] += 0.06

        if metadata.get("requires_multi_reservation_resolution"):
            self._boost(
                weights,
                ["user_grounding", "account_lookup", "line_resolution", "terminal_decision"],
                0.08,
            )
            stage_difficulty["stage2"] += 0.18
            stage_difficulty["stage5"] += 0.12

        if final_action == "cancel_subset":
            self._boost(weights, ["verification", "terminal_decision"], 0.10)
            stage_difficulty["stage5"] += 0.15

        normalized_weights = self._normalize(weights)
        clipped_stage_difficulty = {
            stage: min(1.0, round(value, 3)) for stage, value in stage_difficulty.items()
        }

        return TaskDescriptor(
            task_id=task_id,
            attribute_weights=normalized_weights,
            stage_difficulty=clipped_stage_difficulty,
        )

    def _boost(self, weights: dict[str, float], capability_names: list[str], delta: float) -> None:
        for capability_name in capability_names:
            weights[capability_name] = weights.get(capability_name, 0.0) + delta

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {key: 1.0 / len(weights) for key in weights}
        return {key: round(value / total, 4) for key, value in weights.items()}
