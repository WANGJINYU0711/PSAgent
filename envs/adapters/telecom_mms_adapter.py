"""Adapter from telecom MMS derived instances to reusable TaskDescriptor objects."""

from __future__ import annotations

from typing import Any

from tree_family.specs import TaskDescriptor


class TelecomMMSTaskAdapter:
    ATTRIBUTE_REGISTRY = {
        1: "grounding",
        2: "entity_resolution",
        3: "state_extraction",
        4: "diagnosis",
        5: "subset_selection",
        6: "tool_use",
        7: "robustness",
        8: "long_horizon",
        9: "exception_handling",
        10: "final_execution",
    }

    def build_task_descriptor(self, raw_instance: dict[str, Any]) -> TaskDescriptor:
        task_id = str(raw_instance.get("instance_id", raw_instance.get("original_task_id", "unknown")))
        metadata = raw_instance.get("metadata", {})

        weights = {key: 0.03 for key in self.ATTRIBUTE_REGISTRY}
        stage_difficulty = {
            "stage1": 0.20,
            "stage2": 0.25,
            "stage3": 0.30,
            "stage4": 0.30,
            "stage5": 0.22,
        }

        num_blockers = int(metadata.get("num_blockers", 0))
        if num_blockers >= 6:
            self._boost(weights, [3, 4, 5, 8, 10], 0.10)
            stage_difficulty["stage3"] += 0.12
            stage_difficulty["stage4"] += 0.15
            stage_difficulty["stage5"] += 0.10
        elif num_blockers >= 4:
            self._boost(weights, [3, 4, 8], 0.06)
            stage_difficulty["stage3"] += 0.07
            stage_difficulty["stage4"] += 0.08

        if metadata.get("contains_hybrid_action"):
            self._boost(weights, [2, 4, 6, 10], 0.08)
            stage_difficulty["stage2"] += 0.08
            stage_difficulty["stage4"] += 0.10
            stage_difficulty["stage5"] += 0.08

        if metadata.get("requires_roaming_account_check"):
            self._boost(weights, [2, 3, 4, 6], 0.07)
            stage_difficulty["stage2"] += 0.10
            stage_difficulty["stage3"] += 0.06

        if metadata.get("requires_data_refuel"):
            self._boost(weights, [4, 6, 10], 0.06)
            stage_difficulty["stage3"] += 0.04
            stage_difficulty["stage5"] += 0.05

        if metadata.get("persona_level") == "Hard":
            self._boost(weights, [1, 7, 9], 0.08)
            stage_difficulty["stage1"] += 0.10

        if len(metadata.get("blocker_layers_present", [])) >= 3:
            self._boost(weights, [3, 4, 8], 0.08)
            stage_difficulty["stage3"] += 0.08
            stage_difficulty["stage4"] += 0.08

        normalized_weights = self._normalize(weights)
        clipped_stage_difficulty = {
            stage: min(1.0, round(value, 3)) for stage, value in stage_difficulty.items()
        }
        return TaskDescriptor(
            task_id=task_id,
            attribute_weights=normalized_weights,
            stage_difficulty=clipped_stage_difficulty,
        )

    def _boost(self, weights: dict[int, float], attr_ids: list[int], delta: float) -> None:
        for attr_id in attr_ids:
            weights[attr_id] = weights.get(attr_id, 0.0) + delta

    def _normalize(self, weights: dict[int, float]) -> dict[int, float]:
        total = sum(weights.values())
        if total <= 0:
            return {key: 1.0 / len(weights) for key in weights}
        return {key: round(value / total, 4) for key, value in weights.items()}
