"""LLM-backed executor for Stage 2/3 real tool use."""

from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from .bench_backed_executor import BenchBackedExecutor
from tree_family.specs import AgentSpec, TaskDescriptor


DEFAULT_LLM_MODEL = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4.1-2025-04-14")


class LLMBenchExecutor(BenchBackedExecutor):
    def __init__(
        self,
        stages: list[str],
        seed: int = 0,
        model: str = DEFAULT_LLM_MODEL,
        llm_args: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(stages=stages, seed=seed)
        self.model = model
        self.llm_args = llm_args or {"temperature": 0.0}
        self.llm_bridge_script = Path(__file__).with_name("_llm_bench_bridge.py")
        self.tau2_root = self.root / "tau2-bench"

    def _run_stage2(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        system_prompt, user_prompt = self._build_stage2_prompts(task, agent, raw_instance, stage1_output)
        result = self._run_llm_stage_bridge(
            stage_name="stage2",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=["get_user_details", "get_reservation_details"],
            max_rounds=self._max_rounds(agent),
        )
        output = self._normalize_stage2_output(result.get("final_output"), stage1_output)
        trace = {
            "stage_name": "stage2",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "prompt_summary": self._stage2_prompt_summary(agent),
            "llm_raw_output": deepcopy(result.get("llm_messages", [])),
            "planned_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "executed_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "tool_results": deepcopy(result.get("tool_results", [])),
            "tool_errors": deepcopy(result.get("tool_errors", [])),
            "candidate_reservations": list(output.get("candidate_reservations", [])),
            "resolved_reservations": list(output.get("resolved_reservations", [])),
            "db_hash_before": result.get("db_hash_before"),
            "db_hash_after": result.get("db_hash_after"),
            "input": deepcopy(stage1_output),
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
        }
        return {"input": deepcopy(stage1_output), "output": output, "trace": trace}

    def _run_stage3(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
        stage2_output: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        system_prompt, user_prompt = self._build_stage3_prompts(
            task, agent, raw_instance, stage1_output, stage2_output
        )
        result = self._run_llm_stage_bridge(
            stage_name="stage3",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=["get_user_details", "get_reservation_details", "get_flight_status"],
            max_rounds=self._max_rounds(agent),
        )
        output = self._normalize_stage3_output(result.get("final_output"))
        trace = {
            "stage_name": "stage3",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "prompt_summary": self._stage3_prompt_summary(agent),
            "llm_raw_output": deepcopy(result.get("llm_messages", [])),
            "planned_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "executed_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "tool_results": deepcopy(result.get("tool_results", [])),
            "tool_errors": deepcopy(result.get("tool_errors", [])),
            "computed_features": deepcopy(output.get("per_reservation", [])),
            "db_hash_before": result.get("db_hash_before"),
            "db_hash_after": result.get("db_hash_after"),
            "input": {"stage2_output": deepcopy(stage2_output)},
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
        }
        return {"input": {"stage2_output": deepcopy(stage2_output)}, "output": output, "trace": trace}

    def _run_llm_stage_bridge(
        self,
        stage_name: str,
        original_task_id: str,
        system_prompt: str,
        user_prompt: str,
        allowed_tools: list[str],
        max_rounds: int,
    ) -> dict[str, Any]:
        payload = {
            "stage_name": stage_name,
            "original_task_id": original_task_id,
            "model": self.model,
            "llm_args": self.llm_args,
            "max_rounds": max_rounds,
            "allowed_tools": allowed_tools,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }
        proc = subprocess.run(
            [str(self.venv_python), str(self.llm_bridge_script)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
            cwd=str(self.tau2_root),
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LLM bench bridge failed for {stage_name}: "
                + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
            )
        return json.loads(proc.stdout)

    def _max_rounds(self, agent: AgentSpec) -> int:
        rounds = 4 if agent.competence_level == "high" else 3
        if agent.scope_level == "broad":
            rounds += 1
        if agent.stability_level == "unstable":
            rounds -= 1
        return max(2, rounds)

    def _agent_behavior_guidance(self, agent: AgentSpec) -> str:
        competence = (
            "Be careful and verify enough facts before concluding."
            if agent.competence_level == "high"
            else "Keep the investigation lightweight and stop once you have a plausible answer."
        )
        scope = (
            "Search broadly when there may be multiple relevant reservations."
            if agent.scope_level == "broad"
            else "Focus tightly on the most explicit user hints first."
        )
        stability = (
            "You have a strict round budget, so avoid redundant calls."
            if agent.stability_level == "unstable"
            else "You may use the available round budget to double-check key details."
        )
        return " ".join([competence, scope, stability])

    def _build_stage2_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are performing Stage 2: reservation resolution.\n"
            "Goal: identify the candidate reservations and the resolved reservation set for this user request.\n"
            "You may call tools to inspect the user and reservation records.\n"
            "Stop when you have enough evidence to return the final JSON object.\n"
            "Return only JSON with keys: candidate_reservations, resolved_reservations, "
            "resolution_status, primary_target_reservation_id, user_id."
        )
        user_prompt = json.dumps(
            {
                "task_id": task.task_id,
                "agent_profile": {
                    "competence_level": agent.competence_level,
                    "scope_level": agent.scope_level,
                    "stability_level": agent.stability_level,
                    "guidance": self._agent_behavior_guidance(agent),
                },
                "stage_goal": "Resolve which reservation or reservation set the user is asking about.",
                "stage1_output": stage1_output,
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": raw_instance.get("metadata", {}),
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _build_stage3_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
        stage2_output: dict[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are performing Stage 3: eligibility feature extraction.\n"
            "Goal: collect factual reservation and flight-status features for the resolved reservations.\n"
            "Use tools to inspect reservations and flight status records when needed.\n"
            "Return only JSON with key per_reservation, where each row contains: "
            "reservation_id, hours_since_booking, insurance, cabin, membership, passenger_count, "
            "flight_statuses, any_leg_flown, any_leg_cancelled_by_airline, "
            "stated_reason_supported_by_insurance, eligible_by_24h_rule, "
            "eligible_by_airline_cancel_rule, eligible_by_business_rule, eligible_by_insurance_rule."
        )
        user_prompt = json.dumps(
            {
                "task_id": task.task_id,
                "agent_profile": {
                    "competence_level": agent.competence_level,
                    "scope_level": agent.scope_level,
                    "stability_level": agent.stability_level,
                    "guidance": self._agent_behavior_guidance(agent),
                },
                "stage_goal": "Compute structured feature rows for the currently resolved reservations.",
                "stage1_output": stage1_output,
                "stage2_output": stage2_output,
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": raw_instance.get("metadata", {}),
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _normalize_stage2_output(
        self,
        final_output: dict[str, Any] | None,
        stage1_output: dict[str, Any],
    ) -> dict[str, Any]:
        data = final_output or {}
        candidate = self._normalize_reservation_id_list(data.get("candidate_reservations", []))
        resolved = self._normalize_reservation_id_list(data.get("resolved_reservations", []))
        primary = data.get("primary_target_reservation_id")
        return {
            "candidate_reservations": candidate,
            "resolved_reservations": resolved,
            "resolution_status": str(data.get("resolution_status", "llm_missing_output")),
            "primary_target_reservation_id": primary,
            "user_id": data.get("user_id", stage1_output.get("user_id")),
        }

    def _normalize_reservation_id_list(self, values: Any) -> list[str]:
        rows = list(values or [])
        result: list[str] = []
        for value in rows:
            if isinstance(value, str):
                result.append(value)
            elif isinstance(value, dict):
                rid = value.get("reservation_id") or value.get("id")
                if isinstance(rid, str):
                    result.append(rid)
        return sorted(dict.fromkeys(result))

    def _normalize_stage3_output(self, final_output: dict[str, Any] | None) -> dict[str, Any]:
        data = final_output or {}
        rows = list(data.get("per_reservation", []) or [])
        normalized_rows: list[dict[str, Any]] = []
        required = [
            "reservation_id",
            "hours_since_booking",
            "insurance",
            "cabin",
            "membership",
            "passenger_count",
            "flight_statuses",
            "any_leg_flown",
            "any_leg_cancelled_by_airline",
            "stated_reason_supported_by_insurance",
            "eligible_by_24h_rule",
            "eligible_by_airline_cancel_rule",
            "eligible_by_business_rule",
            "eligible_by_insurance_rule",
        ]
        for row in rows:
            normalized = {key: row.get(key) for key in required}
            flight_statuses = list(normalized.get("flight_statuses") or [])
            normalized["flight_statuses"] = [
                item.get("status") if isinstance(item, dict) else item
                for item in flight_statuses
            ]
            insurance_value = normalized.get("insurance")
            if isinstance(insurance_value, str):
                normalized["insurance"] = insurance_value.lower() == "yes"
            normalized_rows.append(normalized)
        return {"per_reservation": normalized_rows}

    def _stage2_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"stage2 resolution; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={self._max_rounds(agent)}"
        )

    def _stage3_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"stage3 feature extraction; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={self._max_rounds(agent)}"
        )
