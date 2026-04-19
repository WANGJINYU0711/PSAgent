"""Adapter from telecom MMS derived instances to reusable TaskDescriptor objects."""

from __future__ import annotations

from typing import Any

from tree_family.specs import CAPABILITY_NAMES, TaskDescriptor


class TelecomMMSTaskAdapter:
    def build_task_descriptor(self, raw_instance: dict[str, Any]) -> TaskDescriptor:
        task_id = str(raw_instance.get("instance_id", raw_instance.get("original_task_id", "unknown")))
        stage_requirements = self._extract_stage_capability_requirements(raw_instance)
        stage_deliberation = self._extract_stage_deliberation_requirements(raw_instance)
        if stage_requirements is not None:
            return TaskDescriptor(
                task_id=task_id,
                attribute_weights=self._aggregate_stage_requirements(stage_requirements),
                stage_difficulty=self._build_stage_difficulty_from_requirements(stage_requirements),
                stage_capability_requirements=stage_requirements,
                stage_deliberation_requirements=stage_deliberation
                or self._fallback_stage_deliberation_from_difficulty(stage_requirements),
            )

        metadata = raw_instance.get("metadata", {})

        weights = {key: 0.03 for key in CAPABILITY_NAMES}
        stage_difficulty = {
            "stage1": 0.20,
            "stage2": 0.25,
            "stage3": 0.30,
            "stage4": 0.30,
            "stage5": 0.22,
        }

        num_blockers = int(metadata.get("num_blockers", 0))
        if num_blockers >= 6:
            self._boost(
                weights,
                [
                    "network_diagnosis",
                    "permission_diagnosis",
                    "repair_execution",
                    "verification",
                    "terminal_decision",
                ],
                0.08,
            )
            stage_difficulty["stage3"] += 0.12
            stage_difficulty["stage4"] += 0.15
            stage_difficulty["stage5"] += 0.10
        elif num_blockers >= 4:
            self._boost(
                weights,
                ["network_diagnosis", "repair_execution", "verification"],
                0.06,
            )
            stage_difficulty["stage3"] += 0.07
            stage_difficulty["stage4"] += 0.08

        if metadata.get("contains_hybrid_action"):
            self._boost(
                weights,
                ["account_lookup", "roaming_diagnosis", "repair_execution", "terminal_decision"],
                0.08,
            )
            stage_difficulty["stage2"] += 0.08
            stage_difficulty["stage4"] += 0.10
            stage_difficulty["stage5"] += 0.08

        if metadata.get("requires_roaming_account_check"):
            self._boost(
                weights,
                ["account_lookup", "line_resolution", "roaming_diagnosis", "network_diagnosis"],
                0.07,
            )
            stage_difficulty["stage2"] += 0.10
            stage_difficulty["stage3"] += 0.06

        if metadata.get("requires_data_refuel"):
            self._boost(
                weights,
                ["account_lookup", "network_diagnosis", "terminal_decision"],
                0.06,
            )
            stage_difficulty["stage3"] += 0.04
            stage_difficulty["stage5"] += 0.05

        if metadata.get("persona_level") == "Hard":
            self._boost(weights, ["user_grounding", "verification"], 0.08)
            stage_difficulty["stage1"] += 0.10

        if len(metadata.get("blocker_layers_present", [])) >= 3:
            self._boost(
                weights,
                ["network_diagnosis", "repair_execution", "verification"],
                0.08,
            )
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
            stage_capability_requirements=None,
            stage_deliberation_requirements=None,
        )

    def _extract_stage_capability_requirements(
        self,
        raw_instance: dict[str, Any],
    ) -> dict[str, dict[str, float]] | None:
        stage_names = ["stage1", "stage2", "stage3", "stage4", "stage5"]
        extracted: dict[str, dict[str, float]] = {}
        for stage_name in stage_names:
            stage_payload = raw_instance.get(stage_name, {})
            requirement = stage_payload.get("capability_requirements")
            if requirement is None:
                return None
            extracted[stage_name] = {
                capability_name: float(requirement.get(capability_name, 0.0))
                for capability_name in CAPABILITY_NAMES
            }
        return extracted

    def _aggregate_stage_requirements(
        self,
        stage_requirements: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        totals = {capability_name: 0.0 for capability_name in CAPABILITY_NAMES}
        for requirement in stage_requirements.values():
            for capability_name in CAPABILITY_NAMES:
                totals[capability_name] += requirement.get(capability_name, 0.0)
        return self._normalize(totals)

    def _build_stage_difficulty_from_requirements(
        self,
        stage_requirements: dict[str, dict[str, float]],
    ) -> dict[str, float]:
        out: dict[str, float] = {}
        for stage_name, requirement in stage_requirements.items():
            avg_requirement = sum(requirement.values()) / max(1, len(requirement))
            top_requirement = max(requirement.values(), default=0.0)
            difficulty = 0.15 + (0.35 * avg_requirement) + (0.25 * top_requirement)
            out[stage_name] = min(1.0, round(difficulty, 3))
        return out

    def _extract_stage_deliberation_requirements(
        self,
        raw_instance: dict[str, Any],
    ) -> dict[str, str] | None:
        stage_names = ["stage1", "stage2", "stage3", "stage4", "stage5"]
        extracted: dict[str, str] = {}
        for stage_name in stage_names:
            stage_payload = raw_instance.get(stage_name, {})
            requirement = stage_payload.get("deliberation_requirement")
            if requirement is None:
                return None
            requirement_str = str(requirement).strip().lower()
            extracted[stage_name] = "deep" if requirement_str == "deep" else "fast"
        return extracted

    def _fallback_stage_deliberation_from_difficulty(
        self,
        stage_requirements: dict[str, dict[str, float]],
    ) -> dict[str, str]:
        difficulty = self._build_stage_difficulty_from_requirements(stage_requirements)
        return {
            stage_name: ("deep" if difficulty.get(stage_name, 0.0) >= 0.42 else "fast")
            for stage_name in difficulty
        }

    def _boost(self, weights: dict[str, float], capability_names: list[str], delta: float) -> None:
        for capability_name in capability_names:
            weights[capability_name] = weights.get(capability_name, 0.0) + delta

    def _normalize(self, weights: dict[str, float]) -> dict[str, float]:
        total = sum(weights.values())
        if total <= 0:
            return {key: 1.0 / len(weights) for key in weights}
        return {key: round(value / total, 4) for key, value in weights.items()}
