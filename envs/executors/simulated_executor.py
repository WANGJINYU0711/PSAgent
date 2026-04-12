"""Deterministic simulated executor for tree-family experiments."""

from __future__ import annotations

import hashlib
import math
from copy import deepcopy
from typing import Any

from .base_executor import BaseExecutor
from tree_family.specs import AgentSpec, TaskDescriptor


STAGE_ATTRIBUTE_FOCUS = {
    "stage1": [1, 7, 8, 9],
    "stage2": [2, 6, 8],
    "stage3": [3, 4, 9],
    "stage4": [4, 5, 9],
    "stage5": [5, 10, 6],
}


class SimulatedExecutor(BaseExecutor):
    def run_path(
        self,
        task: TaskDescriptor,
        path: list[str],
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        resolved_ids = self._oracle_reservation_ids(raw_instance)
        stage_outputs: dict[str, dict[str, Any]] = {}
        stage_trace: list[dict[str, Any]] = []
        stage_scores: dict[str, float] = {}
        stage_success: dict[str, bool] = {}

        for stage_name, agent_id in zip(self.stages, path):
            agent = agent_map[agent_id]
            score = self._effective_score(task, stage_name, agent)
            success = score >= self._success_threshold(stage_name)
            stage_scores[stage_name] = score
            stage_success[stage_name] = success

            if stage_name == "stage2":
                resolved_ids = self._apply_stage2_resolution(raw_instance, resolved_ids, score)
            stage_output = {
                "score": round(score, 4),
                "success": success,
                "visible_reservation_ids": list(resolved_ids),
            }
            stage_outputs[stage_name] = {"input": {}, "output": stage_output, "source": "simulated_family"}
            stage_trace.append(
                {
                    "stage_name": stage_name,
                    "agent_id": agent_id,
                    "agent_g": agent.g,
                    "score": round(score, 4),
                    "success": success,
                    "visible_reservation_ids": list(resolved_ids),
                }
            )

        predicted_stage5 = self._build_terminal_prediction(
            raw_instance=raw_instance,
            visible_reservation_ids=resolved_ids,
            stage_scores=stage_scores,
            stage_success=stage_success,
        )
        stage_outputs["stage5"]["output"] = predicted_stage5

        leaf_type = "unshared" if any(agent_map[agent_id].g == 1 for agent_id in path) else "shared"
        path_agent_cost = sum(agent_map[agent_id].base_cost for agent_id in path)

        return {
            "final_action": predicted_stage5["final_action"],
            "cancelled_reservation_ids": predicted_stage5["cancelled_reservation_ids"],
            "refused_reservation_ids": predicted_stage5["refused_reservation_ids"],
            "stage_trace": stage_trace,
            "stage_outputs": stage_outputs,
            "path_agent_cost": path_agent_cost,
            "leaf_type": leaf_type,
        }

    def _effective_score(self, task: TaskDescriptor, stage_name: str, agent: AgentSpec) -> float:
        total_attr_weight = max(1e-9, sum(task.attribute_weights.values()))
        match = sum(
            task.attribute_weights.get(attr_id, 0.0) * agent.attribute_skill.get(attr_id, 0.0)
            for attr_id in task.attribute_weights
        ) / total_attr_weight

        competence_bonus = 0.15 if agent.competence_level == "high" else 0.0
        scope_bonus = self._scope_bonus(task, stage_name, agent)
        difficulty_penalty = 0.35 * task.stage_difficulty.get(stage_name, 0.0)
        noise = self._deterministic_noise(
            task_id=task.task_id,
            stage_name=stage_name,
            agent_id=agent.agent_id,
            sigma=0.03 if agent.stability_level == "stable" else 0.10,
        )

        score = match + competence_bonus + scope_bonus - difficulty_penalty + noise
        return max(0.0, min(1.0, score))

    def _scope_bonus(self, task: TaskDescriptor, stage_name: str, agent: AgentSpec) -> float:
        focus = set(STAGE_ATTRIBUTE_FOCUS[stage_name])
        top_task_attrs = {
            attr_id
            for attr_id, _ in sorted(
                task.attribute_weights.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        }
        top_agent_attrs = {
            attr_id
            for attr_id, _ in sorted(
                agent.attribute_skill.items(),
                key=lambda item: item[1],
                reverse=True,
            )[:3]
        }
        overlap = len((top_task_attrs | focus) & top_agent_attrs)
        if agent.scope_level == "broad":
            return 0.02 * min(overlap, 2)
        return 0.05 * overlap - 0.04 * max(0, 2 - overlap)

    def _deterministic_noise(
        self,
        task_id: str,
        stage_name: str,
        agent_id: str,
        sigma: float,
    ) -> float:
        key = f"{self.seed}|{task_id}|{stage_name}|{agent_id}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        centered = (2.0 * value) - 1.0
        return centered * sigma

    def _success_threshold(self, stage_name: str) -> float:
        if stage_name in {"stage2", "stage4", "stage5"}:
            return 0.58
        return 0.52

    def _oracle_reservation_ids(self, raw_instance: dict[str, Any]) -> list[str]:
        if raw_instance.get("family") == "telecom_mms_recovery":
            rows = (
                raw_instance.get("stage4", {})
                .get("oracle_output", {})
                .get("per_blocker", [])
                or []
            )
            return [row.get("blocker_id") for row in rows if row.get("blocker_id")]
        rows = (
            raw_instance.get("stage4", {})
            .get("oracle_output", {})
            .get("per_reservation", [])
            or []
        )
        return [row.get("reservation_id") for row in rows if row.get("reservation_id")]

    def _apply_stage2_resolution(
        self,
        raw_instance: dict[str, Any],
        current_ids: list[str],
        score: float,
    ) -> list[str]:
        del raw_instance
        if len(current_ids) <= 1:
            return list(current_ids)
        if score >= 0.75:
            return list(current_ids)
        if score >= 0.58:
            return list(current_ids[:-1])
        if score >= 0.42:
            return list(current_ids[: max(1, len(current_ids) // 2)])
        return list(current_ids[:1])

    def _build_terminal_prediction(
        self,
        raw_instance: dict[str, Any],
        visible_reservation_ids: list[str],
        stage_scores: dict[str, float],
        stage_success: dict[str, bool],
    ) -> dict[str, Any]:
        if raw_instance.get("family") == "telecom_mms_recovery":
            return self._build_telecom_terminal_prediction(
                raw_instance=raw_instance,
                visible_blocker_ids=visible_reservation_ids,
                stage_scores=stage_scores,
                stage_success=stage_success,
            )
        oracle_rows = (
            raw_instance.get("stage4", {})
            .get("oracle_output", {})
            .get("per_reservation", [])
            or []
        )
        row_map = {row.get("reservation_id"): deepcopy(row) for row in oracle_rows if row.get("reservation_id")}
        visible_rows = [row_map[rid] for rid in visible_reservation_ids if rid in row_map]

        for row in visible_rows:
            should_cancel = row.get("oracle_execute_decision") == "cancel"
            if should_cancel:
                if not stage_success.get("stage3", True):
                    row["oracle_execute_decision"] = "refuse"
                elif stage_scores.get("stage4", 0.0) < 0.62:
                    row["oracle_execute_decision"] = "refuse"
            else:
                if stage_scores.get("stage4", 1.0) < 0.35:
                    row["oracle_execute_decision"] = "cancel"

        cancelled_ids = [row["reservation_id"] for row in visible_rows if row.get("oracle_execute_decision") == "cancel"]
        refused_ids = [row["reservation_id"] for row in visible_rows if row.get("oracle_execute_decision") != "cancel"]

        if stage_scores.get("stage5", 1.0) < 0.60 and cancelled_ids:
            refused_ids.append(cancelled_ids.pop())

        if cancelled_ids and refused_ids:
            final_action = "cancel_subset"
        elif cancelled_ids:
            final_action = "cancel_all"
        else:
            final_action = "refuse_all"

        return {
            "final_action": final_action,
            "cancelled_reservation_ids": sorted(cancelled_ids),
            "refused_reservation_ids": sorted(refused_ids),
            "response_mode": "simulated_family_executor",
        }

    def _build_telecom_terminal_prediction(
        self,
        raw_instance: dict[str, Any],
        visible_blocker_ids: list[str],
        stage_scores: dict[str, float],
        stage_success: dict[str, bool],
    ) -> dict[str, Any]:
        oracle_rows = (
            raw_instance.get("stage4", {})
            .get("oracle_output", {})
            .get("per_blocker", [])
            or []
        )
        row_map = {row.get("blocker_id"): deepcopy(row) for row in oracle_rows if row.get("blocker_id")}
        visible_rows = [row_map[bid] for bid in visible_blocker_ids if bid in row_map]

        selected_ids = [
            row["blocker_id"]
            for row in visible_rows
            if row.get("oracle_execute_decision") == "repair"
        ]
        deferred_ids: list[str] = []

        if stage_scores.get("stage4", 1.0) < 0.62 and selected_ids:
            deferred_ids.append(selected_ids.pop())
        if not stage_success.get("stage3", True) and selected_ids:
            deferred_ids.append(selected_ids.pop())

        if selected_ids and deferred_ids:
            final_action = "repair_subset"
        elif selected_ids:
            final_action = "repair_all"
        else:
            final_action = "transfer"

        selected_ids = sorted(selected_ids)
        deferred_ids = sorted(deferred_ids)
        return {
            "final_action": final_action,
            "selected_blocker_ids": selected_ids,
            "deferred_blocker_ids": deferred_ids,
            "cancelled_reservation_ids": selected_ids,
            "refused_reservation_ids": deferred_ids,
            "response_mode": "simulated_family_executor",
        }
