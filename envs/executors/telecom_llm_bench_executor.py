"""LLM-backed executor for telecom Stage 1/2/3/4/5 execution."""

from __future__ import annotations

import json
import os
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from telecom_mms_specs import (
    build_per_blocker_from_ids,
    first_pass_terminal_decision,
    get_blocker_spec,
    infer_blocker_ids_from_observed_state,
)
from .telecom_bench_backed_executor import TelecomBenchBackedExecutor
from tree_family.specs import AgentSpec, TaskDescriptor


DEFAULT_LLM_MODEL = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4.1-2025-04-14")
STAGE4_REPAIR_TOOLS = [
    "toggle_airplane_mode",
    "toggle_data",
    "enable_roaming",
    "toggle_roaming",
    "refuel_data",
    "set_network_mode_preference",
    "toggle_wifi_calling",
    "reset_apn_settings",
    "reboot_device",
    "grant_app_permission",
    "reseat_sim_card",
    "get_details_by_id",
]
STAGE5_VERIFICATION_TOOLS = [
    "get_details_by_id",
    "check_network_status",
    "check_sim_status",
    "check_network_mode_preference",
    "check_apn_settings",
    "check_wifi_calling_status",
    "check_app_permissions",
    "run_speed_test",
    "can_send_mms",
]
MUTATING_REPAIR_TOOL_NAMES = {
    "toggle_airplane_mode",
    "toggle_data",
    "enable_roaming",
    "toggle_roaming",
    "refuel_data",
    "set_network_mode_preference",
    "toggle_wifi_calling",
    "reset_apn_settings",
    "reboot_device",
    "grant_app_permission",
    "reseat_sim_card",
}


class TelecomLLMBenchExecutor(TelecomBenchBackedExecutor):
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
        self.llm_bridge_script = Path(__file__).with_name("_telecom_llm_bench_bridge.py")
        self.tau2_root = self.root / "tau2-bench"

    def _run_stage1(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        system_prompt, user_prompt = self._build_stage1_prompts(task, agent, raw_instance)
        result = self._run_llm_stage_bridge(
            stage_name="stage1",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=["get_customer_by_phone", "get_details_by_id"],
            max_rounds=min(3, self._max_rounds(agent, task, "stage1")),
        )
        output = self._normalize_stage1_output(
            final_output=result.get("final_output"),
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
            raw_instance=raw_instance,
        )
        trace = {
            "stage_name": "stage1",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "prompt_summary": self._stage1_prompt_summary(agent),
            "llm_raw_output": deepcopy(result.get("llm_messages", [])),
            "planned_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "executed_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "tool_results": deepcopy(result.get("tool_results", [])),
            "tool_errors": deepcopy(result.get("tool_errors", [])),
            "db_hash_before": result.get("db_hash_before"),
            "db_hash_after": result.get("db_hash_after"),
            "input": {
                "user_context": deepcopy(raw_instance.get("user_context", {})),
                "task_metadata": deepcopy(raw_instance.get("metadata", {})),
                "task_id": task.task_id,
            },
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
        }
        return {
            "input": {
                "user_context": deepcopy(raw_instance.get("user_context", {})),
                "task_metadata": deepcopy(raw_instance.get("metadata", {})),
                "task_id": task.task_id,
            },
            "output": output,
            "trace": trace,
        }

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
            allowed_tools=["get_customer_by_phone", "get_details_by_id"],
            max_rounds=self._max_rounds(agent, task, "stage2"),
        )
        output = self._normalize_stage2_output(
            final_output=result.get("final_output"),
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
            stage1_output=stage1_output,
        )
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
            allowed_tools=[
                "get_details_by_id",
                "check_network_status",
                "check_sim_status",
                "check_network_mode_preference",
                "check_apn_settings",
                "check_wifi_calling_status",
                "check_app_permissions",
                "run_speed_test",
                "can_send_mms",
            ],
            max_rounds=self._max_rounds(agent, task, "stage3"),
        )
        diagnostic_fallback = self._maybe_fetch_stage3_diagnostic_fallback(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
            tool_errors=result.get("tool_errors", []),
        )
        fallback_debug = self._maybe_fetch_stage3_account_side_fallback(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
            tool_errors=result.get("tool_errors", []),
        )
        output = self._normalize_stage3_output(
            final_output=result.get("final_output"),
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
            raw_instance=raw_instance,
            stage1_output=stage1_output,
            stage2_output=stage2_output,
        )
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
            "db_hash_before": result.get("db_hash_before"),
            "db_hash_after": result.get("db_hash_after"),
            "input": {
                "stage1_output": deepcopy(stage1_output),
                "stage2_output": deepcopy(stage2_output),
            },
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
            "per_blocker_mode": "inferred_from_observed_state_v2",
            "diagnostic_fallback_used": diagnostic_fallback["used"],
            "diagnostic_fallback_calls": deepcopy(diagnostic_fallback["calls"]),
            "account_side_fallback_used": fallback_debug["used"],
            "account_side_fallback_calls": deepcopy(fallback_debug["calls"]),
        }
        return {
            "input": {
                "stage1_output": deepcopy(stage1_output),
                "stage2_output": deepcopy(stage2_output),
            },
            "output": output,
            "trace": trace,
        }

    def _run_stage4(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        system_prompt, user_prompt = self._build_stage4_prompts(
            task,
            agent,
            raw_instance,
            stage1_output,
            stage2_output,
            stage3_output,
        )
        result = self._run_llm_stage_bridge(
            stage_name="stage4",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=list(STAGE4_REPAIR_TOOLS),
            max_rounds=max(4, min(6, self._max_rounds(agent, task, "stage4") + 1)),
        )
        execution_result = self._execute_stage4_canonical_plan(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            final_output=result.get("final_output"),
        )
        output = self._normalize_stage4_output(
            final_output=result.get("final_output"),
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            executed_tool_calls=execution_result.get("executed_tool_calls", []),
            tool_results=execution_result.get("tool_results", []),
            tool_errors=execution_result.get("tool_errors", []),
            db_hash_before=execution_result.get("db_hash_before", result.get("db_hash_before")),
            db_hash_after=execution_result.get("db_hash_after", result.get("db_hash_after")),
        )
        trace = {
            "stage_name": "stage4",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "prompt_summary": self._stage4_prompt_summary(agent),
            "llm_raw_output": deepcopy(result.get("llm_messages", [])),
            "planned_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "executed_tool_calls": deepcopy(execution_result.get("executed_tool_calls", [])),
            "tool_results": deepcopy(execution_result.get("tool_results", [])),
            "tool_errors": deepcopy(execution_result.get("tool_errors", [])),
            "db_hash_before": execution_result.get("db_hash_before", result.get("db_hash_before")),
            "db_hash_after": execution_result.get("db_hash_after", result.get("db_hash_after")),
            "input": {
                "stage2_output": deepcopy(stage2_output),
                "stage3_output": deepcopy(stage3_output),
            },
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
            "policy_mode": "repair_execution_with_env_mutation",
            "llm_execution_attempts": deepcopy(result.get("executed_tool_calls", [])),
        }
        return {
            "input": {
                "stage2_output": deepcopy(stage2_output),
                "stage3_output": deepcopy(stage3_output),
            },
            "output": output,
            "trace": trace,
        }

    def _run_stage5(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
        stage4_result: dict[str, Any],
    ) -> dict[str, Any]:
        del stage1_output, stage3_output
        agent = agent_map[agent_id]
        stage4_output = stage4_result["output"]
        replay_tool_calls = self._stage5_replay_tool_calls(stage4_result.get("trace", {}))
        system_prompt, user_prompt = self._build_stage5_prompts(
            task,
            agent,
            raw_instance,
            stage2_output,
            stage4_output,
        )
        result = self._run_llm_stage_bridge(
            stage_name="stage5",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=list(STAGE5_VERIFICATION_TOOLS),
            max_rounds=max(3, min(5, self._max_rounds(agent, task, "stage5") + 1)),
            replay_tool_calls=replay_tool_calls,
        )
        verification_fallback = self._maybe_fetch_stage5_verification_fallback(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            stage4_output=stage4_output,
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
            tool_errors=result.get("tool_errors", []),
        )
        output = self._normalize_stage5_output(
            final_output=result.get("final_output"),
            stage4_output=stage4_output,
            stage2_output=stage2_output,
            raw_instance=raw_instance,
            executed_tool_calls=result.get("executed_tool_calls", []),
            tool_results=result.get("tool_results", []),
        )
        trace = {
            "stage_name": "stage5",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "prompt_summary": self._stage5_prompt_summary(agent),
            "llm_raw_output": deepcopy(result.get("llm_messages", [])),
            "planned_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "executed_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "tool_results": deepcopy(result.get("tool_results", [])),
            "tool_errors": deepcopy(result.get("tool_errors", [])),
            "db_hash_before": result.get("db_hash_before"),
            "db_hash_after": result.get("db_hash_after"),
            "db_hash_before_replay": result.get("db_hash_before_replay"),
            "db_hash_after_replay": result.get("db_hash_after_replay"),
            "replay_tool_calls": deepcopy(result.get("replay_tool_calls", [])),
            "replay_tool_results": deepcopy(result.get("replay_tool_results", [])),
            "replay_tool_errors": deepcopy(result.get("replay_tool_errors", [])),
            "input": {
                "stage2_output": deepcopy(stage2_output),
                "stage4_output": deepcopy(stage4_output),
            },
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
            "policy_mode": "verification_then_terminal_decision",
            "verification_fallback_used": verification_fallback["used"],
            "verification_fallback_calls": deepcopy(verification_fallback["calls"]),
        }
        return {
            "input": {
                "stage2_output": deepcopy(stage2_output),
                "stage4_output": deepcopy(stage4_output),
            },
            "output": output,
            "trace": trace,
        }

    def _run_llm_stage_bridge(
        self,
        stage_name: str,
        original_task_id: str,
        system_prompt: str,
        user_prompt: str,
        allowed_tools: list[str],
        max_rounds: int,
        replay_tool_calls: list[dict[str, Any]] | None = None,
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
            "replay_tool_calls": replay_tool_calls or [],
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
                f"Telecom LLM bench bridge failed for {stage_name}: "
                + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
            )
        return json.loads(proc.stdout)

    def _maybe_fetch_stage3_account_side_fallback(
        self,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        tool_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        line_id = stage2_output.get("resolved_line_id")
        plan_id = stage2_output.get("assistant_account_snapshot", {}).get("plan_id")

        seen_line = False
        seen_plan = False
        for call in executed_tool_calls:
            if call.get("name") != "get_details_by_id":
                continue
            arg_id = str(call.get("arguments", {}).get("id", ""))
            if line_id and arg_id == line_id:
                seen_line = True
            if plan_id and arg_id == plan_id:
                seen_plan = True

        fallback_calls: list[dict[str, Any]] = []
        if line_id and not seen_line:
            fallback_calls.append(
                {
                    "id": "stage3_fallback_line",
                    "name": "get_details_by_id",
                    "arguments": {"id": line_id},
                    "requestor": "assistant",
                }
            )
        if plan_id and not seen_plan:
            fallback_calls.append(
                {
                    "id": "stage3_fallback_plan",
                    "name": "get_details_by_id",
                    "arguments": {"id": plan_id},
                    "requestor": "assistant",
                }
            )

        if not fallback_calls:
            return {"used": False, "calls": []}

        bridge_result = self._run_bench_tool_calls(raw_instance, fallback_calls)
        for row in bridge_result.get("responses", []):
            executed_tool_calls.append(row["tool_call"])
            tool_results.append(row["content"])
            if row.get("error"):
                tool_errors.append(row)
        return {"used": True, "calls": fallback_calls}

    def _max_rounds(
        self,
        agent: AgentSpec,
        task: TaskDescriptor | None = None,
        stage_name: str | None = None,
    ) -> int:
        rounds = 3 if getattr(agent, "deliberation_mode", "deep") == "fast" else 5
        if task is not None and stage_name is not None:
            requirement = self._stage_deliberation_requirement(task, stage_name)
            if requirement == "deep":
                rounds += 1
            elif requirement == "fast" and getattr(agent, "deliberation_mode", "deep") == "deep":
                rounds -= 1
        return max(2, rounds)

    def _agent_behavior_guidance(self, agent: AgentSpec) -> str:
        competence = (
            "Be careful, verify facts, and prefer explicit tool evidence."
            if agent.competence_level == "high"
            else "Keep the investigation lightweight and stop after sufficient evidence."
        )
        scope = (
            "Search broadly when customer or line resolution may be ambiguous."
            if agent.scope_level == "broad"
            else "Focus on the most explicit phone-number path first."
        )
        stability = (
            "Avoid redundant calls because the round budget is tight."
            if agent.stability_level == "unstable"
            else "You may use the full round budget to double-check key facts."
        )
        deliberation = (
            "This agent is fast: prioritize short evidence chains, avoid redundant calls, and make compact decisions."
            if getattr(agent, "deliberation_mode", "deep") == "fast"
            else "This agent is deep: spend budget on careful verification when the stage requires detailed reasoning."
        )
        return " ".join([competence, scope, stability, deliberation])

    def _stage_requirement_map(self, task: TaskDescriptor, stage_name: str) -> dict[str, float]:
        if task.stage_capability_requirements and stage_name in task.stage_capability_requirements:
            return task.stage_capability_requirements[stage_name]
        return task.attribute_weights

    def _stage_deliberation_requirement(self, task: TaskDescriptor, stage_name: str) -> str:
        if task.stage_deliberation_requirements and stage_name in task.stage_deliberation_requirements:
            requirement = str(task.stage_deliberation_requirements[stage_name]).strip().lower()
            return "deep" if requirement == "deep" else "fast"
        return "deep" if task.stage_difficulty.get(stage_name, 0.0) >= 0.42 else "fast"

    def _build_agent_capability_profile(self, agent: AgentSpec) -> dict[str, Any]:
        ranked = sorted(agent.attribute_skill.items(), key=lambda item: (item[1], item[0]), reverse=True)
        low_ranked = sorted(agent.attribute_skill.items(), key=lambda item: (item[1], item[0]))
        return {
            "strengths": [[name, round(float(score), 3)] for name, score in ranked[:3]],
            "weaknesses": [[name, round(float(score), 3)] for name, score in low_ranked[:3]],
        }

    def _build_agent_deliberation_profile(self, agent: AgentSpec) -> dict[str, Any]:
        mode = getattr(agent, "deliberation_mode", "deep")
        return {
            "mode": mode,
            "style_guidance": (
                "Prefer short, high-yield evidence chains and avoid low-value extra tool calls."
                if mode == "fast"
                else "Prefer careful cross-checking on high-risk fields before committing to a decision."
            ),
        }

    def _build_stage_requirement_summary(self, task: TaskDescriptor, stage_name: str) -> list[list[Any]]:
        requirement = self._stage_requirement_map(task, stage_name)
        ranked = sorted(requirement.items(), key=lambda item: (item[1], item[0]), reverse=True)
        return [[name, round(float(score), 3)] for name, score in ranked[:3] if score > 0.0]

    def _build_capability_match_summary(
        self,
        task: TaskDescriptor,
        stage_name: str,
        agent: AgentSpec,
    ) -> dict[str, Any]:
        required = [name for name, _ in self._build_stage_requirement_summary(task, stage_name)]
        strengths = {
            name
            for name, _ in sorted(agent.attribute_skill.items(), key=lambda item: (item[1], item[0]), reverse=True)[:3]
        }
        weaknesses = {
            name
            for name, _ in sorted(agent.attribute_skill.items(), key=lambda item: (item[1], item[0]))[:3]
        }
        return {
            "aligned_strengths": [name for name in required if name in strengths],
            "caution_areas": [name for name in required if name in weaknesses],
        }

    def _build_stage_deliberation_summary(
        self,
        task: TaskDescriptor,
        stage_name: str,
    ) -> dict[str, Any]:
        requirement = self._stage_deliberation_requirement(task, stage_name)
        return {
            "mode": requirement,
            "usage_guidance": (
                "This stage rewards detailed reasoning and extra verification on high-risk facts."
                if requirement == "deep"
                else "This stage rewards quick resolution with compact evidence and limited tool use."
            ),
        }

    def _build_deliberation_match_summary(
        self,
        task: TaskDescriptor,
        stage_name: str,
        agent: AgentSpec,
    ) -> dict[str, Any]:
        requirement = self._stage_deliberation_requirement(task, stage_name)
        mode = getattr(agent, "deliberation_mode", "deep")
        aligned = requirement == mode
        return {
            "alignment": "aligned" if aligned else "mismatch",
            "caution": (
                "Agent is fast on a deep-reasoning stage: use the limited budget to verify the highest-risk facts first."
                if requirement == "deep" and mode == "fast"
                else "Agent is deep on a fast stage: avoid redundant tool calls and stop after enough evidence."
                if requirement == "fast" and mode == "deep"
                else "Agent deliberation style is aligned with this stage."
            ),
            "round_budget_hint": self._max_rounds(agent, task, stage_name),
        }

    def _build_stage2_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are performing Stage 2: customer and line resolution for a telecom support case.\n"
            "Goal: identify the customer, resolve the target line, and extract a minimal account snapshot.\n"
            "You are given an agent capability profile, current stage capability requirements, the agent deliberation mode, and the current stage deliberation requirement.\n"
            "Use capability information to focus what evidence matters. Use deliberation information to control how much search to spend: fast agents should keep the search compact, while deep agents may spend more rounds when the stage requires careful reasoning.\n"
            "Use only the allowed tools.\n"
            "Do not do diagnosis. Do not talk about blockers. Do not produce prose.\n"
            "Return only JSON with keys: candidate_customers, resolved_customer_id, candidate_line_ids, "
            "resolved_line_id, target_phone_number, assistant_account_snapshot, resolution_status.\n"
            "assistant_account_snapshot must contain: line_status, roaming_enabled_on_account, plan_id, data_used_gb, data_limit_gb."
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
                "agent_capability_profile": self._build_agent_capability_profile(agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "current_stage_capability_requirements": self._build_stage_requirement_summary(task, "stage2"),
                "current_stage_deliberation_requirement": self._build_stage_deliberation_summary(task, "stage2"),
                "capability_match_summary": self._build_capability_match_summary(task, "stage2", agent),
                "deliberation_match_summary": self._build_deliberation_match_summary(task, "stage2", agent),
                "stage_goal": "Resolve the customer and telecom line only.",
                "stage1_output": stage1_output,
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": raw_instance.get("metadata", {}),
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _build_stage1_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are performing Stage 1: user grounding for a telecom MMS troubleshooting case.\n"
            "Your goal is to transform the user request into a stable structured Stage 1 output.\n"
            "You are given an agent capability profile, current stage capability requirements, the agent deliberation mode, and the current stage deliberation requirement.\n"
            "Use capability information to decide what to ground. Use deliberation information to decide whether to keep the grounding quick or to spend more effort resolving ambiguity.\n"
            "You may use the allowed tools only when needed to confirm customer identity, phone grounding, or line grounding.\n"
            "Keep tool use minimal.\n"
            "Do NOT do diagnosis.\n"
            "Do NOT infer blockers.\n"
            "Do NOT make terminal decisions.\n"
            "Return JSON only.\n"
            "The output must be a JSON object with top-level keys: "
            "domain, problem_family, customer_lookup, line_selector, symptom_report, context_flags, conversation_risk_flags.\n"
            "customer_lookup must contain: full_name, phone_number, lookup_confidence.\n"
            "line_selector must contain: type, value.\n"
            "symptom_report must contain: cannot_send_mms, wants_resolution, target_success_signal.\n"
            "context_flags must contain: is_abroad_claimed, refuel_allowed, max_refuel_gb, plan_change_allowed.\n"
            "conversation_risk_flags must be a list of short strings.\n"
            "Output JSON only. No markdown. No prose outside the JSON object."
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
                "agent_capability_profile": self._build_agent_capability_profile(agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "current_stage_capability_requirements": self._build_stage_requirement_summary(task, "stage1"),
                "current_stage_deliberation_requirement": self._build_stage_deliberation_summary(task, "stage1"),
                "capability_match_summary": self._build_capability_match_summary(task, "stage1", agent),
                "deliberation_match_summary": self._build_deliberation_match_summary(task, "stage1", agent),
                "stage_goal": "Ground the user, phone number, and target telecom line at a minimal level only.",
                "policy_mode": "grounding_only_minimal_lookup",
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": raw_instance.get("metadata", {}),
                "output_contract": {
                    "top_level_keys": [
                        "domain",
                        "problem_family",
                        "customer_lookup",
                        "line_selector",
                        "symptom_report",
                        "context_flags",
                        "conversation_risk_flags",
                    ],
                    "lookup_confidence_values": ["high", "medium", "low"],
                    "line_selector_types": ["phone_number"],
                },
                "normalization_rules": [
                    "Use tools only for minimal customer identity or line grounding",
                    "Prefer the grounded phone number when setting customer_lookup.phone_number and line_selector.value",
                    "Keep line_selector directly usable by Stage 2",
                    "Do not include diagnosis, blockers, repair plans, or terminal actions",
                ],
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
            "You are performing Stage 3: observed-state extraction for a telecom MMS troubleshooting case.\n"
            "Goal: collect factual observed state only. Do not decide terminal actions.\n"
            "You are given an agent capability profile, current stage capability requirements, the agent deliberation mode, and the current stage deliberation requirement.\n"
            "Use capability information to focus diagnostic coverage and be cautious where the agent is weak. Use deliberation information to decide whether to keep diagnosis compact or to spend more effort on careful cross-checking.\n"
            "Use only the allowed tools. Prefer explicit tool evidence over guesses.\n"
            "For MMS diagnosis, permission / APN / network-mode coverage is high priority.\n"
            "Before returning JSON, make sure you have checked messaging app permissions, APN MMS settings, and network mode preference unless the tool results are already present.\n"
            "Return only JSON with key observed_state.\n"
            "observed_state must contain exactly these keys: can_send_mms, service_status, mobile_data_working, "
            "internet_speed_desc, is_abroad, roaming_enabled_on_device, roaming_enabled_on_account, airplane_mode, "
            "sim_status, network_mode_preference, wifi_calling_enabled, apn_mms_ok, "
            "messaging_sms_permission, messaging_storage_permission, data_usage_exceeded.\n"
            "Do not include terminal decisions or free-form explanations."
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
                "agent_capability_profile": self._build_agent_capability_profile(agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "current_stage_capability_requirements": self._build_stage_requirement_summary(task, "stage3"),
                "current_stage_deliberation_requirement": self._build_stage_deliberation_summary(task, "stage3"),
                "capability_match_summary": self._build_capability_match_summary(task, "stage3", agent),
                "deliberation_match_summary": self._build_deliberation_match_summary(task, "stage3", agent),
                "stage_goal": "Produce only factual observed state for the resolved telecom line.",
                "tool_use_checklist": [
                    "check_network_status",
                    "can_send_mms",
                    "check_network_mode_preference",
                    "check_apn_settings",
                    "check_app_permissions(app_name=messaging)",
                ],
                "stage1_output": stage1_output,
                "stage2_output": stage2_output,
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": raw_instance.get("metadata", {}),
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _build_stage4_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
    ) -> tuple[str, str]:
        del stage1_output
        system_prompt = (
            "You are performing Stage 4: blocker adjudication and repair execution for a telecom MMS troubleshooting case.\n"
            "You are given an agent capability profile, current stage capability requirements, the agent deliberation mode, and the current stage deliberation requirement.\n"
            "Use capability information to judge which blockers this agent can handle safely. Use deliberation information to decide whether to make a quick execution plan or to slow down around higher-risk adjudication boundaries.\n"
            "First decide, for each blocker, whether it should be repaired automatically, deferred, or transferred.\n"
            "Then execute canonical repair steps only for blockers with should_repair=true.\n"
            "Use only the allowed repair tools.\n"
            "Do not do fresh diagnosis beyond minimal execution-time grounding.\n"
            "Do not produce customer-facing prose.\n"
            "Return JSON only.\n"
            "Your output must be a JSON object with top-level keys: per_blocker, repairability, transfer_reason, decision_policy_version.\n"
            "per_blocker must include every input blocker_id exactly once and each row must contain blocker_id and should_repair.\n"
            "Allowed repairability values: repairable, partially_repairable, transfer_required.\n"
            "Frozen-policy bias:\n"
            "- If any blocker is marked as hybrid-required, choose transfer_required.\n"
            "- If a blocker is assistant-side-required but can be safely deferred, it may be marked should_repair=false under partially_repairable.\n"
            "- Otherwise prefer should_repair=true for safe auto-repair blockers.\n"
            "Execution rules:\n"
            "- After adjudication, execute canonical_repair_steps in repair_order order for blockers with should_repair=true.\n"
            "- Do not execute repair steps for blockers with should_repair=false.\n"
            "- Do not use tools to override the frozen transfer/defer boundary.\n"
            "If transfer is required, provide a non-null short snake_case transfer_reason. Otherwise use null.\n"
            "Output JSON only. No markdown. No explanation outside the JSON."
        )
        blocker_ids = [
            row.get("blocker_id")
            for row in stage3_output.get("per_blocker", [])
            if row.get("blocker_id")
        ]
        blocker_specs = {
            blocker_id: get_blocker_spec(blocker_id)
            for blocker_id in blocker_ids
        }
        repair_metadata = {
            blocker_id: {
                "assistant_side_required": blocker_specs[blocker_id]["assistant_side_required"],
                "user_side_required": blocker_specs[blocker_id]["user_side_required"],
                "hybrid_required": blocker_specs[blocker_id]["hybrid_required"],
                "can_be_deferred": blocker_specs[blocker_id]["can_be_deferred"],
                "default_priority": blocker_specs[blocker_id]["default_priority"],
                "depends_on": blocker_specs[blocker_id]["depends_on"],
                "canonical_repair_steps": self._build_stage4_rows([blocker_id], stage2_output)[0]["canonical_repair_steps"],
            }
            for blocker_id in blocker_ids
        }
        user_prompt = json.dumps(
            {
                "task_id": task.task_id,
                "agent_profile": {
                    "competence_level": agent.competence_level,
                    "scope_level": agent.scope_level,
                    "stability_level": agent.stability_level,
                    "guidance": self._agent_behavior_guidance(agent),
                },
                "agent_capability_profile": self._build_agent_capability_profile(agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "current_stage_capability_requirements": self._build_stage_requirement_summary(task, "stage4"),
                "current_stage_deliberation_requirement": self._build_stage_deliberation_summary(task, "stage4"),
                "capability_match_summary": self._build_capability_match_summary(task, "stage4", agent),
                "deliberation_match_summary": self._build_deliberation_match_summary(task, "stage4", agent),
                "stage_goal": "Adjudicate blockers under frozen first-pass semantics, then execute canonical repair steps for selected blockers.",
                "policy_mode": "repair_execution_with_env_mutation",
                "output_contract": {
                    "top_level_keys": [
                        "per_blocker",
                        "repairability",
                        "transfer_reason",
                        "decision_policy_version",
                    ],
                    "repairability_values": [
                        "repairable",
                        "partially_repairable",
                        "transfer_required",
                    ],
                },
                "stage2_context": {
                    "resolved_customer_id": stage2_output.get("resolved_customer_id"),
                    "resolved_line_id": stage2_output.get("resolved_line_id"),
                    "target_phone_number": stage2_output.get("target_phone_number"),
                    "assistant_account_snapshot": stage2_output.get("assistant_account_snapshot"),
                },
                "stage3_output": {
                    "observed_state": stage3_output.get("observed_state"),
                    "per_blocker": stage3_output.get("per_blocker"),
                },
                "blocker_specs": blocker_specs,
                "repair_metadata": repair_metadata,
                "task_metadata": raw_instance.get("metadata", {}),
                "normalization_rules": [
                    "Return every blocker_id exactly once",
                    "Set should_repair to a boolean for every blocker row",
                    "Use transfer_reason=null when transfer is not required",
                    "Set decision_policy_version to a short stable string, for example first_pass_v1",
                    "Do not invent blockers that are not in the input per_blocker list",
                    "Execute canonical repair steps only for blockers marked should_repair=true",
                    "Do not execute deferred or transfer-required blocker repairs",
                ],
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _build_stage5_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        stage4_output: dict[str, Any],
    ) -> tuple[str, str]:
        system_prompt = (
            "You are performing Stage 5: post-repair verification and terminal decision for a telecom MMS troubleshooting case.\n"
            "Your job is to verify the current post-repair telecom state using verification tools, then choose the final structured terminal action.\n"
            "You are given an agent capability profile, current stage capability requirements, the agent deliberation mode, and the current stage deliberation requirement.\n"
            "Use capability information to decide where verification and terminal-decision caution matter most. Use deliberation information to balance quick closure against careful post-repair validation.\n"
            "Do not execute repair tools or perform additional repair mutation.\n"
            "Replay has already been applied before your verification step.\n"
            "You must verify after replay before returning JSON.\n"
            "Minimum rule: use can_send_mms plus blocker-matched verification tools when blockers were repaired or selected.\n"
            "Do not produce customer-facing prose.\n"
            "Return JSON only.\n"
            "The only allowed final_action values are: repair_all, repair_subset, transfer.\n"
            "Decision constraints:\n"
            "- repair_all means all blockers must be selected and deferred must be empty.\n"
            "- repair_subset means selected and deferred must form a partition of the input blocker ids.\n"
            "- transfer means selected must be empty and all blockers must be deferred.\n"
            "Output must be a JSON object with at least these top-level keys: "
            "final_action, selected_blocker_ids, deferred_blocker_ids, response_mode, verification_plan, "
            "transfer_reason, cancelled_reservation_ids, refused_reservation_ids.\n"
            "You may also include verification_observed_state, verification_evidence, verification_summary, post_repair_can_send_mms, post_repair_blocker_ids.\n"
            "Do not include explanations outside the JSON."
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
                "agent_capability_profile": self._build_agent_capability_profile(agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "current_stage_capability_requirements": self._build_stage_requirement_summary(task, "stage5"),
                "current_stage_deliberation_requirement": self._build_stage_deliberation_summary(task, "stage5"),
                "capability_match_summary": self._build_capability_match_summary(task, "stage5", agent),
                "deliberation_match_summary": self._build_deliberation_match_summary(task, "stage5", agent),
                "stage_goal": "Verify the post-repair env state and then choose the final terminal decision.",
                "policy_mode": "verification_then_terminal_decision",
                "stage2_context": {
                    "resolved_customer_id": stage2_output.get("resolved_customer_id"),
                    "resolved_line_id": stage2_output.get("resolved_line_id"),
                    "target_phone_number": stage2_output.get("target_phone_number"),
                    "assistant_account_snapshot": stage2_output.get("assistant_account_snapshot"),
                },
                "stage4_output": stage4_output,
                "verification_checklist": self._stage5_verification_checklist(stage4_output),
                "task_metadata": raw_instance.get("metadata", {}),
                "output_contract": {
                    "top_level_keys": [
                        "final_action",
                        "selected_blocker_ids",
                        "deferred_blocker_ids",
                        "response_mode",
                        "verification_plan",
                        "transfer_reason",
                        "cancelled_reservation_ids",
                        "refused_reservation_ids",
                    ],
                    "final_action_values": [
                        "repair_all",
                        "repair_subset",
                        "transfer",
                    ],
                },
                "normalization_rules": [
                    "Only use blocker ids that appear in stage4_output.per_blocker",
                    "Return list fields as lists of blocker ids",
                    "If final_action is transfer, selected_blocker_ids must be empty",
                    "If final_action is repair_all, all blockers must be selected",
                    "If final_action is repair_subset, selected and deferred must partition the blocker ids",
                    "Use verification tools to inspect the current post-repair state before deciding",
                    "If any blocker was replayed or selected, verification should usually include can_send_mms",
                    "Do not execute repair tools in Stage 5",
                    "Do not include tools, prose, or execution details outside the JSON object",
                ],
            },
            ensure_ascii=False,
        )
        return system_prompt, user_prompt

    def _normalize_stage2_output(
        self,
        final_output: dict[str, Any] | None,
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        stage1_output: dict[str, Any],
    ) -> dict[str, Any]:
        data = final_output or {}
        tool_map = self._zip_tool_results(executed_tool_calls, tool_results)

        customer_result = next((content for name, _, content in tool_map if name == "get_customer_by_phone"), None) or {}
        line_results = [
            content
            for name, args, content in tool_map
            if name == "get_details_by_id" and isinstance(content, dict) and str(args.get("id", "")).startswith("L")
        ]
        plan_results = [
            content
            for name, args, content in tool_map
            if name == "get_details_by_id" and isinstance(content, dict) and str(args.get("id", "")).startswith("P")
        ]

        target_phone_number = (
            data.get("target_phone_number")
            or stage1_output.get("line_selector", {}).get("value")
            or stage1_output.get("customer_lookup", {}).get("phone_number")
        )
        customer_fallback = [customer_result.get("customer_id")] if customer_result.get("customer_id") else []
        candidate_customers = self._normalize_str_list(data.get("candidate_customers")) or customer_fallback
        candidate_line_ids = self._merge_unique_str_lists(
            data.get("candidate_line_ids"),
            customer_result.get("line_ids", []),
        )
        matched_line = next(
            (line for line in line_results if line.get("phone_number") == target_phone_number),
            None,
        )
        if matched_line is not None:
            resolved_line = matched_line
        elif data.get("resolved_line_id"):
            resolved_line = next(
                (
                    line
                    for line in line_results
                    if line.get("line_id") == data.get("resolved_line_id")
                ),
                line_results[0] if line_results else {},
            )
        else:
            resolved_line = line_results[0] if line_results else {}
        matched_plan = next(
            (
                plan
                for plan in plan_results
                if plan.get("plan_id") == resolved_line.get("plan_id")
            ),
            None,
        )
        plan = matched_plan or (plan_results[0] if plan_results else {})
        return {
            "candidate_customers": self._normalize_str_list(candidate_customers),
            "resolved_customer_id": data.get("resolved_customer_id") or customer_result.get("customer_id"),
            "candidate_line_ids": self._normalize_str_list(candidate_line_ids),
            "resolved_line_id": resolved_line.get("line_id") or data.get("resolved_line_id"),
            "target_phone_number": target_phone_number,
            "assistant_account_snapshot": {
                "line_status": str(
                    resolved_line.get("status")
                    or data.get("assistant_account_snapshot", {}).get("line_status")
                    or ""
                ).lower()
                or None,
                "roaming_enabled_on_account": self._coalesce_bool(
                    resolved_line.get("roaming_enabled"),
                    data.get("assistant_account_snapshot", {}).get("roaming_enabled_on_account"),
                ),
                "plan_id": resolved_line.get("plan_id") or data.get("assistant_account_snapshot", {}).get("plan_id"),
                "data_used_gb": self._coalesce_number(
                    resolved_line.get("data_used_gb"),
                    data.get("assistant_account_snapshot", {}).get("data_used_gb"),
                ),
                "data_limit_gb": self._coalesce_number(
                    plan.get("data_limit_gb"),
                    data.get("assistant_account_snapshot", {}).get("data_limit_gb"),
                ),
            },
            "resolution_status": data.get("resolution_status") or ("resolved" if resolved_line else "unresolved"),
        }

    def _normalize_stage1_output(
        self,
        final_output: dict[str, Any] | None,
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        data = final_output or {}
        tool_map = self._zip_tool_results(executed_tool_calls, tool_results)
        user_context = raw_instance.get("user_context", {}) or {}
        metadata = raw_instance.get("metadata", {}) or {}
        known_info = str(user_context.get("known_info", "") or "")
        reason_for_call = str(user_context.get("reason_for_call", "") or "")
        task_instructions = str(user_context.get("task_instructions", "") or "")

        customer_result = next(
            (
                content
                for name, _args, content in tool_map
                if name == "get_customer_by_phone" and isinstance(content, dict)
            ),
            {},
        )
        line_result = next(
            (
                content
                for name, _args, content in tool_map
                if name == "get_details_by_id" and isinstance(content, dict) and content.get("phone_number")
            ),
            {},
        )

        llm_customer_lookup = data.get("customer_lookup", {}) if isinstance(data.get("customer_lookup"), dict) else {}
        llm_line_selector = data.get("line_selector", {}) if isinstance(data.get("line_selector"), dict) else {}
        llm_symptom_report = data.get("symptom_report", {}) if isinstance(data.get("symptom_report"), dict) else {}
        llm_context_flags = data.get("context_flags", {}) if isinstance(data.get("context_flags"), dict) else {}

        phone_number = (
            customer_result.get("phone_number")
            or line_result.get("phone_number")
            or llm_customer_lookup.get("phone_number")
            or llm_line_selector.get("value")
            or self._extract_phone(known_info)
        )
        full_name = (
            customer_result.get("full_name")
            or llm_customer_lookup.get("full_name")
            or self._extract_full_name(known_info)
        )

        return {
            "domain": "telecom",
            "problem_family": "mms_issue",
            "customer_lookup": {
                "full_name": full_name,
                "phone_number": phone_number,
                "lookup_confidence": self._normalize_lookup_confidence(
                    llm_customer_lookup.get("lookup_confidence"),
                    used_tools=bool(executed_tool_calls),
                    has_phone=bool(phone_number),
                    has_full_name=bool(full_name),
                ),
            },
            "line_selector": {
                "type": "phone_number",
                "value": phone_number,
            },
            "symptom_report": {
                "cannot_send_mms": self._coalesce_bool(
                    self._normalize_optional_bool(llm_symptom_report.get("cannot_send_mms")),
                    self._infer_cannot_send_mms(reason_for_call),
                ),
                "wants_resolution": self._coalesce_bool(
                    self._normalize_optional_bool(llm_symptom_report.get("wants_resolution")),
                    self._infer_wants_resolution(reason_for_call),
                    True,
                ),
                "target_success_signal": self._normalize_target_success_signal(
                    llm_symptom_report.get("target_success_signal"),
                    metadata.get("target_success_signal"),
                ),
            },
            "context_flags": {
                "is_abroad_claimed": self._coalesce_bool(
                    self._normalize_optional_bool(llm_context_flags.get("is_abroad_claimed")),
                    self._infer_is_abroad(known_info),
                    False,
                ),
                "refuel_allowed": self._coalesce_bool(
                    self._normalize_optional_bool(llm_context_flags.get("refuel_allowed")),
                    self._infer_refuel_allowed(task_instructions),
                    True,
                ),
                "max_refuel_gb": self._coalesce_number(
                    llm_context_flags.get("max_refuel_gb"),
                    self._extract_refuel_gb(task_instructions),
                    2.0,
                ),
                "plan_change_allowed": self._coalesce_bool(
                    self._normalize_optional_bool(llm_context_flags.get("plan_change_allowed")),
                    self._infer_plan_change_allowed(task_instructions),
                    False,
                ),
            },
            "conversation_risk_flags": self._normalize_stage1_risk_flags(
                data.get("conversation_risk_flags"),
                task_instructions=task_instructions,
            ),
        }

    def _normalize_stage3_output(
        self,
        final_output: dict[str, Any] | None,
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
        stage2_output: dict[str, Any],
    ) -> dict[str, Any]:
        data = final_output or {}
        observed_seed = self._normalize_observed_seed(dict(data.get("observed_state") or {}))
        tool_map = self._zip_tool_results(executed_tool_calls, tool_results)

        response_lookup = {name: content for name, _args, content in tool_map}
        line_details = self._latest_tool_result(
            tool_map,
            "get_details_by_id",
            expected_args={"id": stage2_output.get("resolved_line_id")},
        )
        if not isinstance(line_details, dict):
            line_details = {}
        plan_details = self._latest_tool_result(
            tool_map,
            "get_details_by_id",
            expected_args={"id": stage2_output.get("assistant_account_snapshot", {}).get("plan_id")},
        )
        if not isinstance(plan_details, dict):
            plan_details = {}
        messaging_permissions = self._latest_tool_result(
            tool_map,
            "check_app_permissions",
            expected_args={"app_name": "messaging"},
        )

        observed_from_tools = self._normalize_observed_state(
            known_info=raw_instance.get("user_context", {}).get("known_info", ""),
            network_status=response_lookup.get("check_network_status"),
            sim_status=response_lookup.get("check_sim_status"),
            mode_status=response_lookup.get("check_network_mode_preference"),
            apn_status=response_lookup.get("check_apn_settings"),
            wifi_calling_status=response_lookup.get("check_wifi_calling_status"),
            app_permissions=messaging_permissions,
            speed_test=response_lookup.get("run_speed_test"),
            can_send_mms=response_lookup.get("can_send_mms"),
            line_details=line_details,
            plan_details=plan_details,
        )
        observed_state = self._merge_observed_state_tool_first(
            observed_from_tools=observed_from_tools,
            observed_seed=observed_seed,
        )
        inferred_blocker_ids = infer_blocker_ids_from_observed_state(observed_state)
        per_blocker = build_per_blocker_from_ids(inferred_blocker_ids)
        return {
            "observed_state": observed_state,
            "per_blocker": per_blocker,
            "per_blocker_mode": "inferred_from_observed_state_v2",
            "raw_task_blocker_ids": self._raw_task_blocker_ids(raw_instance),
            "inferred_blocker_ids": inferred_blocker_ids,
        }

    def _normalize_stage4_output(
        self,
        final_output: dict[str, Any] | None,
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        tool_errors: list[dict[str, Any]],
        db_hash_before: str | None,
        db_hash_after: str | None,
    ) -> dict[str, Any]:
        normalized_rows, repairability, transfer_reason = self._normalized_stage4_plan(
            final_output=final_output,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
        )
        execution_summary = self._stage4_execution_summary(
            normalized_rows=normalized_rows,
            executed_tool_calls=executed_tool_calls,
            tool_results=tool_results,
            tool_errors=tool_errors,
        )

        for row in normalized_rows:
            blocker_execution = execution_summary["per_blocker_execution"].get(row["blocker_id"], {})
            row["execution_attempted"] = bool(blocker_execution.get("execution_attempted", False))
            row["execution_succeeded"] = bool(blocker_execution.get("execution_succeeded", False))
            row["executed_step_count"] = int(blocker_execution.get("executed_step_count", 0))

        return {
            "per_blocker": normalized_rows,
            "repairability": repairability,
            "transfer_reason": transfer_reason,
            "decision_policy_version": "first_pass_v1",
            "executed_repair_steps": execution_summary["executed_repair_steps"],
            "failed_repair_steps": execution_summary["failed_repair_steps"],
            "skipped_repair_steps": execution_summary["skipped_repair_steps"],
            "executed_blocker_ids": execution_summary["executed_blocker_ids"],
            "deferred_blocker_ids": [
                row["blocker_id"] for row in normalized_rows if not row.get("should_repair")
            ],
            "post_execution_status": {
                "mutation_attempted": bool(execution_summary["mutating_tool_calls"]),
                "successful_mutation_count": len(execution_summary["executed_repair_steps"]),
                "failed_mutation_count": len(execution_summary["failed_repair_steps"]),
                "all_selected_repairs_succeeded": execution_summary["all_selected_repairs_succeeded"],
            },
            "db_hash_before_execution": db_hash_before,
            "db_hash_after_execution": db_hash_after,
        }

    def _normalized_stage4_plan(
        self,
        *,
        final_output: dict[str, Any] | None,
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], str, str | None]:
        blocker_ids = [
            row.get("blocker_id")
            for row in stage3_output.get("per_blocker", [])
            if row.get("blocker_id")
        ]
        decision = first_pass_terminal_decision(blocker_ids)
        fallback_rows = self._build_stage4_rows(blocker_ids, stage2_output)
        llm_rows = final_output.get("per_blocker", []) if isinstance(final_output, dict) else []
        llm_row_map = {
            row.get("blocker_id"): row
            for row in llm_rows
            if isinstance(row, dict) and row.get("blocker_id")
        }

        normalized_rows: list[dict[str, Any]] = []
        for fallback_row in fallback_rows:
            blocker_id = fallback_row["blocker_id"]
            llm_row = llm_row_map.get(blocker_id, {})
            normalized = deepcopy(fallback_row)
            llm_should_repair = llm_row.get("should_repair")
            if llm_should_repair == fallback_row["should_repair"]:
                llm_repair_order = llm_row.get("repair_order")
                if isinstance(llm_repair_order, int) and llm_repair_order > 0:
                    normalized["repair_order"] = llm_repair_order
            normalized_rows.append(normalized)

        normalized_rows.sort(
            key=lambda row: (
                0 if row.get("should_repair") else 1,
                int(row.get("repair_order", 10**6)),
                str(row.get("blocker_id", "")),
            )
        )
        for index, row in enumerate(normalized_rows, start=1):
            row["repair_order"] = index

        repairability = (
            final_output.get("repairability")
            if isinstance(final_output, dict)
            and final_output.get("repairability") == decision["repairability"]
            else decision["repairability"]
        )
        transfer_reason = decision["transfer_reason"] if repairability == "transfer_required" else None
        return normalized_rows, repairability, transfer_reason

    def _execute_stage4_canonical_plan(
        self,
        *,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
        final_output: dict[str, Any] | None,
    ) -> dict[str, Any]:
        normalized_rows, _repairability, _transfer_reason = self._normalized_stage4_plan(
            final_output=final_output,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
        )
        tool_calls: list[dict[str, Any]] = []
        for row in normalized_rows:
            if not row.get("should_repair"):
                continue
            for step_index, step in enumerate(row.get("canonical_repair_steps", []), start=1):
                tool_calls.append(
                    {
                        "id": f"stage4_exec_{row['blocker_id']}_{step_index}",
                        "name": step.get("tool_name"),
                        "arguments": deepcopy(step.get("arguments", {})),
                        "requestor": step.get("requestor", "assistant"),
                    }
                )
        bridge_result = self._run_bench_tool_calls(raw_instance, tool_calls)
        responses = bridge_result.get("responses", [])
        return {
            "db_hash_before": bridge_result.get("db_hash_before"),
            "db_hash_after": bridge_result.get("db_hash_after"),
            "executed_tool_calls": [row["tool_call"] for row in responses],
            "tool_results": [row["content"] for row in responses],
            "tool_errors": [row for row in responses if row.get("error")],
        }

    def _normalize_stage5_output(
        self,
        final_output: dict[str, Any] | None,
        stage4_output: dict[str, Any],
        stage2_output: dict[str, Any],
        raw_instance: dict[str, Any],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
    ) -> dict[str, Any]:
        data = final_output or {}
        blocker_ids = [
            row.get("blocker_id")
            for row in stage4_output.get("per_blocker", [])
            if isinstance(row, dict) and row.get("blocker_id")
        ]
        blocker_id_set = set(blocker_ids)
        verification_summary = self._stage5_verification_summary(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            executed_tool_calls=executed_tool_calls,
            tool_results=tool_results,
        )

        selected_from_output = self._normalize_str_list(
            data.get("selected_blocker_ids") or data.get("cancelled_reservation_ids")
        )
        deferred_from_output = self._normalize_str_list(
            data.get("deferred_blocker_ids") or data.get("refused_reservation_ids")
        )

        selected_clean = [bid for bid in selected_from_output if bid in blocker_id_set]
        deferred_clean = [
            bid for bid in deferred_from_output
            if bid in blocker_id_set and bid not in selected_clean
        ]

        raw_action = str(data.get("final_action", "")).strip().lower()
        if raw_action not in {"repair_all", "repair_subset", "transfer"}:
            if selected_clean and deferred_clean:
                final_action = "repair_subset"
            elif selected_clean and not deferred_clean:
                final_action = "repair_all" if set(selected_clean) == blocker_id_set else "repair_subset"
            elif blocker_ids:
                final_action = "transfer"
            else:
                final_action = "repair_all"
        else:
            final_action = raw_action

        if final_action == "repair_all":
            selected_blocker_ids = list(blocker_ids)
            deferred_blocker_ids: list[str] = []
        elif final_action == "transfer":
            selected_blocker_ids = []
            deferred_blocker_ids = list(blocker_ids)
        else:
            if not selected_clean and deferred_clean:
                selected_clean = [bid for bid in blocker_ids if bid not in deferred_clean]
            if not selected_clean:
                final_action = "transfer"
                selected_blocker_ids = []
                deferred_blocker_ids = list(blocker_ids)
            else:
                selected_blocker_ids = [bid for bid in blocker_ids if bid in set(selected_clean)]
                deferred_blocker_ids = [bid for bid in blocker_ids if bid not in set(selected_blocker_ids)]
                if not deferred_blocker_ids:
                    final_action = "repair_all"
                    selected_blocker_ids = list(blocker_ids)
                elif not selected_blocker_ids:
                    final_action = "transfer"
                    deferred_blocker_ids = list(blocker_ids)

        if final_action == "transfer":
            transfer_reason = self._normalize_optional_short_text(data.get("transfer_reason"))
            if transfer_reason is None:
                transfer_reason = self._normalize_optional_short_text(stage4_output.get("transfer_reason"))
        else:
            transfer_reason = None

        verification_plan = self._normalize_stage5_verification_plan(
            data.get("verification_plan"),
            final_action=final_action,
        )

        return {
            "final_action": final_action,
            "selected_blocker_ids": selected_blocker_ids,
            "deferred_blocker_ids": deferred_blocker_ids,
            "response_mode": "telecom_structured_execution",
            "verification_plan": verification_plan,
            "transfer_reason": transfer_reason,
            "cancelled_reservation_ids": list(selected_blocker_ids),
            "refused_reservation_ids": list(deferred_blocker_ids),
            "verification_observed_state": verification_summary["verification_observed_state"],
            "verification_evidence": verification_summary["verification_evidence"],
            "verification_summary": verification_summary["verification_summary"],
            "post_repair_can_send_mms": verification_summary["verification_observed_state"].get("can_send_mms"),
            "post_repair_blocker_ids": verification_summary["post_repair_blocker_ids"],
        }

    def _maybe_fetch_stage3_diagnostic_fallback(
        self,
        *,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        tool_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        present = {
            (
                str(call.get("name", "")),
                json.dumps(dict(call.get("arguments", {})), ensure_ascii=False, sort_keys=True),
            )
            for call in executed_tool_calls
        }
        fallback_calls: list[dict[str, Any]] = []
        desired = [
            ("check_network_mode_preference", {}),
            ("check_apn_settings", {}),
            ("check_app_permissions", {"app_name": "messaging"}),
        ]
        for name, arguments in desired:
            key = (name, json.dumps(arguments, ensure_ascii=False, sort_keys=True))
            if key in present:
                continue
            fallback_calls.append(
                {
                    "id": f"stage3_diag_fallback_{name}",
                    "name": name,
                    "arguments": deepcopy(arguments),
                    "requestor": "user",
                }
            )
        if not fallback_calls:
            return {"used": False, "calls": []}

        bridge_result = self._run_bench_tool_calls(raw_instance, fallback_calls)
        for row in bridge_result.get("responses", []):
            executed_tool_calls.append(row["tool_call"])
            tool_results.append(row["content"])
            if row.get("error"):
                tool_errors.append(row)
        return {"used": True, "calls": fallback_calls}

    def _stage4_execution_summary(
        self,
        *,
        normalized_rows: list[dict[str, Any]],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        tool_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        error_ids = {
            str(row.get("tool_call", {}).get("id", ""))
            for row in tool_errors
            if isinstance(row, dict)
        }
        mutating_calls = [
            {
                "tool_call": deepcopy(call),
                "tool_result": deepcopy(result),
                "error": str(call.get("id", "")) in error_ids,
            }
            for call, result in zip(executed_tool_calls, tool_results)
            if self._is_mutating_tool_name(call.get("name"))
        ]
        remaining = list(mutating_calls)
        executed_repair_steps: list[dict[str, Any]] = []
        failed_repair_steps: list[dict[str, Any]] = []
        skipped_repair_steps: list[dict[str, Any]] = []
        per_blocker_execution: dict[str, dict[str, Any]] = {}
        executed_blocker_ids: list[str] = []

        for row in normalized_rows:
            blocker_id = str(row.get("blocker_id"))
            canonical_steps = list(row.get("canonical_repair_steps", []))
            successful_steps = 0
            attempted_steps = 0
            all_steps_succeeded = bool(canonical_steps) and bool(row.get("should_repair"))
            if row.get("should_repair"):
                for step_index, expected_step in enumerate(canonical_steps, start=1):
                    matched_index = self._find_matching_executed_step(remaining, expected_step)
                    if matched_index is None:
                        skipped_repair_steps.append(
                            {
                                "blocker_id": blocker_id,
                                "step_index": step_index,
                                "tool_name": expected_step.get("tool_name"),
                                "arguments": deepcopy(expected_step.get("arguments", {})),
                                "requestor": expected_step.get("requestor"),
                                "reason": "not_executed",
                            }
                        )
                        all_steps_succeeded = False
                        continue
                    matched = remaining.pop(matched_index)
                    attempted_steps += 1
                    row_payload = {
                        "blocker_id": blocker_id,
                        "step_index": step_index,
                        "tool_call": deepcopy(matched["tool_call"]),
                        "tool_result": deepcopy(matched["tool_result"]),
                    }
                    if matched["error"]:
                        failed_repair_steps.append(row_payload)
                        all_steps_succeeded = False
                    else:
                        executed_repair_steps.append(row_payload)
                        successful_steps += 1
                if all_steps_succeeded:
                    executed_blocker_ids.append(blocker_id)
            else:
                for step_index, expected_step in enumerate(canonical_steps, start=1):
                    skipped_repair_steps.append(
                        {
                            "blocker_id": blocker_id,
                            "step_index": step_index,
                            "tool_name": expected_step.get("tool_name"),
                            "arguments": deepcopy(expected_step.get("arguments", {})),
                            "requestor": expected_step.get("requestor"),
                            "reason": "deferred_or_transfer",
                        }
                    )
                all_steps_succeeded = False

            per_blocker_execution[blocker_id] = {
                "execution_attempted": attempted_steps > 0,
                "execution_succeeded": all_steps_succeeded,
                "executed_step_count": successful_steps,
            }

        for extra in remaining:
            failed_repair_steps.append(
                {
                    "blocker_id": None,
                    "step_index": None,
                    "tool_call": deepcopy(extra["tool_call"]),
                    "tool_result": deepcopy(extra["tool_result"]),
                    "reason": "unexpected_or_unmapped_mutation",
                }
            )

        selected_blockers = {row["blocker_id"] for row in normalized_rows if row.get("should_repair")}
        all_selected_repairs_succeeded = selected_blockers == set(executed_blocker_ids)
        return {
            "executed_repair_steps": executed_repair_steps,
            "failed_repair_steps": failed_repair_steps,
            "skipped_repair_steps": skipped_repair_steps,
            "executed_blocker_ids": executed_blocker_ids,
            "per_blocker_execution": per_blocker_execution,
            "mutating_tool_calls": mutating_calls,
            "all_selected_repairs_succeeded": all_selected_repairs_succeeded,
        }

    def _stage5_verification_summary(
        self,
        *,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
    ) -> dict[str, Any]:
        if not executed_tool_calls:
            return {
                "verification_observed_state": {},
                "verification_evidence": [],
                "post_repair_blocker_ids": [],
                "verification_summary": {
                    "verification_tool_count": 0,
                    "verification_tools_used": [],
                },
            }
        tool_map = self._zip_tool_results(executed_tool_calls, tool_results)
        response_lookup = {name: content for name, _args, content in tool_map}
        line_details = self._latest_tool_result(
            tool_map,
            "get_details_by_id",
            expected_args={"id": stage2_output.get("resolved_line_id")},
        ) or {}
        plan_details = self._latest_tool_result(
            tool_map,
            "get_details_by_id",
            expected_args={"id": stage2_output.get("assistant_account_snapshot", {}).get("plan_id")},
        ) or {}
        messaging_permissions = self._latest_tool_result(
            tool_map,
            "check_app_permissions",
            expected_args={"app_name": "messaging"},
        )
        observed_state = self._normalize_observed_state(
            known_info=raw_instance.get("user_context", {}).get("known_info", ""),
            network_status=response_lookup.get("check_network_status"),
            sim_status=response_lookup.get("check_sim_status"),
            mode_status=response_lookup.get("check_network_mode_preference"),
            apn_status=response_lookup.get("check_apn_settings"),
            wifi_calling_status=response_lookup.get("check_wifi_calling_status"),
            app_permissions=messaging_permissions,
            speed_test=response_lookup.get("run_speed_test"),
            can_send_mms=response_lookup.get("can_send_mms"),
            line_details=line_details,
            plan_details=plan_details,
        )
        if "can_send_mms" not in response_lookup:
            observed_state["can_send_mms"] = None
        if "check_network_status" not in response_lookup:
            observed_state["service_status"] = None
            observed_state["roaming_enabled_on_device"] = None
            observed_state["airplane_mode"] = None
        if "check_sim_status" not in response_lookup:
            observed_state["sim_status"] = None
        if "check_network_mode_preference" not in response_lookup:
            observed_state["network_mode_preference"] = None
        if "check_apn_settings" not in response_lookup:
            observed_state["apn_mms_ok"] = None
        if "check_wifi_calling_status" not in response_lookup:
            observed_state["wifi_calling_enabled"] = None
        if messaging_permissions is None:
            observed_state["messaging_sms_permission"] = None
            observed_state["messaging_storage_permission"] = None
        if "run_speed_test" not in response_lookup:
            observed_state["mobile_data_working"] = None
            observed_state["internet_speed_desc"] = None
        if not line_details:
            observed_state["roaming_enabled_on_account"] = None
        if not line_details or not plan_details:
            observed_state["data_usage_exceeded"] = None
        evidence = list(dict.fromkeys(str(call.get("name", "")) for call in executed_tool_calls if call.get("name")))
        post_repair_blocker_ids = infer_blocker_ids_from_observed_state(observed_state)
        return {
            "verification_observed_state": observed_state,
            "verification_evidence": evidence,
            "post_repair_blocker_ids": post_repair_blocker_ids,
            "verification_summary": {
                "verification_tool_count": len(executed_tool_calls),
                "verification_tools_used": evidence,
            },
        }

    def _stage5_verification_checklist(self, stage4_output: dict[str, Any]) -> list[str]:
        blocker_ids = [
            row.get("blocker_id")
            for row in stage4_output.get("per_blocker", [])
            if isinstance(row, dict) and row.get("should_repair") and row.get("blocker_id")
        ]
        checklist: list[str] = []
        if any(
            blocker_id in {
                "airplane_mode_on",
                "data_mode_off",
                "data_usage_exceeded",
                "user_abroad_roaming_disabled_on",
                "user_abroad_roaming_enabled_off",
                "user_abroad_roaming_disabled_off",
            }
            for blocker_id in blocker_ids
        ):
            checklist.append("check_network_status")
        if "unseat_sim_card" in blocker_ids:
            checklist.append("check_sim_status")
        if "bad_network_preference" in blocker_ids:
            checklist.append("check_network_mode_preference")
        if "break_apn_mms_setting" in blocker_ids:
            checklist.append("check_apn_settings")
        if "bad_wifi_calling" in blocker_ids:
            checklist.append("check_wifi_calling_status")
        if any(
            blocker_id in {
                "break_app_sms_permission",
                "break_app_storage_permission",
                "break_app_both_permissions",
            }
            for blocker_id in blocker_ids
        ):
            checklist.append("check_app_permissions(app_name=messaging)")
        if blocker_ids:
            checklist.append("can_send_mms")
        return list(dict.fromkeys(checklist))

    def _recommended_stage5_verification_calls(
        self,
        stage4_output: dict[str, Any],
        stage2_output: dict[str, Any],
    ) -> list[dict[str, Any]]:
        blocker_ids = [
            row.get("blocker_id")
            for row in stage4_output.get("per_blocker", [])
            if isinstance(row, dict) and row.get("should_repair") and row.get("blocker_id")
        ]
        calls: list[dict[str, Any]] = []
        if any(
            blocker_id in {
                "airplane_mode_on",
                "data_mode_off",
                "data_usage_exceeded",
                "user_abroad_roaming_disabled_on",
                "user_abroad_roaming_enabled_off",
                "user_abroad_roaming_disabled_off",
            }
            for blocker_id in blocker_ids
        ):
            calls.append({"name": "check_network_status", "arguments": {}, "requestor": "user"})
        if "unseat_sim_card" in blocker_ids:
            calls.append({"name": "check_sim_status", "arguments": {}, "requestor": "user"})
        if "bad_network_preference" in blocker_ids:
            calls.append({"name": "check_network_mode_preference", "arguments": {}, "requestor": "user"})
        if "break_apn_mms_setting" in blocker_ids:
            calls.append({"name": "check_apn_settings", "arguments": {}, "requestor": "user"})
        if "bad_wifi_calling" in blocker_ids:
            calls.append({"name": "check_wifi_calling_status", "arguments": {}, "requestor": "user"})
        if any(
            blocker_id in {
                "break_app_sms_permission",
                "break_app_storage_permission",
                "break_app_both_permissions",
            }
            for blocker_id in blocker_ids
        ):
            calls.append(
                {
                    "name": "check_app_permissions",
                    "arguments": {"app_name": "messaging"},
                    "requestor": "user",
                }
            )
        if any(
            blocker_id in {
                "data_usage_exceeded",
                "user_abroad_roaming_disabled_on",
                "user_abroad_roaming_enabled_off",
                "user_abroad_roaming_disabled_off",
            }
            for blocker_id in blocker_ids
        ):
            line_id = stage2_output.get("resolved_line_id")
            plan_id = stage2_output.get("assistant_account_snapshot", {}).get("plan_id")
            if line_id:
                calls.append(
                    {
                        "name": "get_details_by_id",
                        "arguments": {"id": line_id},
                        "requestor": "assistant",
                    }
                )
            if plan_id:
                calls.append(
                    {
                        "name": "get_details_by_id",
                        "arguments": {"id": plan_id},
                        "requestor": "assistant",
                    }
                )
        if blocker_ids:
            calls.append({"name": "can_send_mms", "arguments": {}, "requestor": "user"})

        deduped: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for call in calls:
            key = (
                str(call["name"]),
                json.dumps(call.get("arguments", {}), ensure_ascii=False, sort_keys=True),
            )
            if key in seen:
                continue
            seen.add(key)
            deduped.append(call)
        return deduped

    def _maybe_fetch_stage5_verification_fallback(
        self,
        *,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        stage4_output: dict[str, Any],
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        tool_errors: list[dict[str, Any]],
    ) -> dict[str, Any]:
        recommended_calls = self._recommended_stage5_verification_calls(stage4_output, stage2_output)
        if not recommended_calls:
            return {"used": False, "calls": []}

        present = {
            (
                str(call.get("name", "")),
                json.dumps(dict(call.get("arguments", {})), ensure_ascii=False, sort_keys=True),
            )
            for call in executed_tool_calls
        }
        fallback_calls: list[dict[str, Any]] = []
        for idx, call in enumerate(recommended_calls, start=1):
            key = (
                str(call["name"]),
                json.dumps(call.get("arguments", {}), ensure_ascii=False, sort_keys=True),
            )
            if key in present:
                continue
            fallback_calls.append(
                {
                    "id": f"stage5_verify_fallback_{idx}",
                    "name": call["name"],
                    "arguments": deepcopy(call.get("arguments", {})),
                    "requestor": call.get("requestor", "assistant"),
                }
            )
        if not fallback_calls:
            return {"used": False, "calls": []}

        bridge_result = self._run_bench_tool_calls(raw_instance, fallback_calls)
        for row in bridge_result.get("responses", []):
            executed_tool_calls.append(row["tool_call"])
            tool_results.append(row["content"])
            if row.get("error"):
                tool_errors.append(row)
        return {"used": True, "calls": fallback_calls}

    def _normalize_stage5_verification_plan(
        self,
        verification_plan: Any,
        *,
        final_action: str,
    ) -> dict[str, Any]:
        if final_action == "repair_all":
            default = {
                "required_postchecks": ["can_send_mms"],
                "success_condition": "can_send_mms_true",
            }
        elif final_action == "repair_subset":
            default = {
                "required_postchecks": [],
                "success_condition": "partial_resolution_only",
            }
        else:
            default = {
                "required_postchecks": [],
                "success_condition": "transfer_required",
            }

        if not isinstance(verification_plan, dict):
            return default

        required_postchecks = self._normalize_str_list(
            verification_plan.get("required_postchecks")
        )
        success_condition = verification_plan.get("success_condition")
        if not isinstance(success_condition, str) or not success_condition.strip():
            success_condition = default["success_condition"]

        return {
            "required_postchecks": required_postchecks,
            "success_condition": success_condition,
        }

    def _normalize_observed_state(
        self,
        known_info: dict[str, Any] | str,
        network_status: Any,
        sim_status: Any,
        mode_status: Any,
        apn_status: Any,
        wifi_calling_status: Any,
        app_permissions: Any,
        speed_test: Any,
        can_send_mms: Any,
        line_details: Any,
        plan_details: Any,
    ) -> dict[str, Any]:
        known_info_text = (
            json.dumps(known_info, ensure_ascii=False)
            if isinstance(known_info, dict)
            else str(known_info)
        )
        network_lines = self._parse_key_value_lines(str(network_status or ""))
        apn_lines = self._parse_key_value_lines(str(apn_status or ""))
        line_details = line_details or {}
        plan_details = plan_details or {}

        speed_text = str(speed_test or "")
        if "Speed test failed:" in speed_text:
            internet_speed_desc = speed_text.split("Speed test failed:", 1)[1].strip().rstrip(".")
            mobile_data_working = False
        else:
            match = re.search(r"\(([^)]+)\)", speed_text)
            internet_speed_desc = match.group(1) if match else "Unknown"
            mobile_data_working = internet_speed_desc not in {"No Connection", "Unknown"}

        sms_permission, storage_permission = self._parse_messaging_permissions(app_permissions)
        mmsc_url = apn_lines.get("MMSC URL (for picture messages)", "")
        data_used_gb = float(line_details.get("data_used_gb", 0.0) or 0.0)
        data_refueling_gb = float(line_details.get("data_refueling_gb", 0.0) or 0.0)
        data_limit_gb = float(plan_details.get("data_limit_gb", 0.0) or 0.0)

        return {
            "can_send_mms": "cannot" not in str(can_send_mms or "").lower(),
            "service_status": network_lines.get("Cellular Connection", "unknown"),
            "mobile_data_working": mobile_data_working,
            "internet_speed_desc": internet_speed_desc,
            "is_abroad": "abroad" in known_info_text.lower(),
            "roaming_enabled_on_device": network_lines.get("Data Roaming Enabled", "No") == "Yes",
            "roaming_enabled_on_account": bool(line_details.get("roaming_enabled")),
            "airplane_mode": network_lines.get("Airplane Mode", "OFF") == "ON",
            "sim_status": self._normalize_sim_status(str(sim_status or "")),
            "network_mode_preference": str(mode_status or "").split(":", 1)[-1].strip(),
            "wifi_calling_enabled": "ON" in str(wifi_calling_status or ""),
            "apn_mms_ok": mmsc_url not in {"Not Set", ""},
            "messaging_sms_permission": sms_permission,
            "messaging_storage_permission": storage_permission,
            "data_usage_exceeded": data_used_gb >= data_limit_gb + data_refueling_gb,
        }

    def _merge_observed_state_tool_first(
        self,
        observed_from_tools: dict[str, Any],
        observed_seed: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(observed_from_tools)
        for key, value in observed_seed.items():
            if value is None:
                continue
            if key in {"messaging_sms_permission", "messaging_storage_permission"}:
                continue
            if key not in merged or merged.get(key) is None:
                merged[key] = value
        return merged

    def _parse_messaging_permissions(self, app_perm_text: Any) -> tuple[bool | None, bool | None]:
        if app_perm_text is None:
            return None, None
        text = " ".join(str(app_perm_text).strip().split())
        if not text:
            return None, None
        lowered = text.lower()

        if "not found on this phone" in lowered:
            return None, None

        if "currently has no permissions granted" in lowered:
            return False, False

        match = re.search(r"has permission for:\s*([a-z,\s]+)\.?", lowered)
        if not match:
            return None, None

        raw_items = match.group(1)
        tokens = {
            token.strip()
            for token in re.split(r"[,\s]+", raw_items)
            if token.strip()
        }
        # Telecom user tools expose the granted permission names verbatim.
        # For messaging blocker recovery we care specifically about sms/storage.
        # If the permission list is present at all, any missing tracked permission
        # should be treated as False rather than "unknown". Example:
        # "has permission for: phone." => sms=False, storage=False.
        sms_aliases = {"sms"}
        storage_aliases = {"storage"}
        sms_granted = any(alias in tokens for alias in sms_aliases)
        storage_granted = any(alias in tokens for alias in storage_aliases)
        return sms_granted, storage_granted

    def _zip_tool_results(
        self,
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
    ) -> list[tuple[str, dict[str, Any], Any]]:
        rows: list[tuple[str, dict[str, Any], Any]] = []
        for call, content in zip(executed_tool_calls, tool_results):
            rows.append((str(call.get("name", "")), dict(call.get("arguments", {})), content))
        return rows

    def _latest_tool_result(
        self,
        tool_map: list[tuple[str, dict[str, Any], Any]],
        tool_name: str,
        expected_args: dict[str, Any] | None = None,
    ) -> Any:
        matches: list[Any] = []
        for name, args, content in tool_map:
            if name != tool_name:
                continue
            if expected_args is not None:
                if not all(args.get(key) == value for key, value in expected_args.items()):
                    continue
            matches.append(content)
        if not matches:
            return None
        return matches[-1]

    def _find_matching_executed_step(
        self,
        remaining: list[dict[str, Any]],
        expected_step: dict[str, Any],
    ) -> int | None:
        expected = self._canonicalize_tool_step(expected_step, call_key="tool_name")
        for idx, candidate in enumerate(remaining):
            actual = self._canonicalize_tool_step(candidate["tool_call"], call_key="name")
            if actual == expected:
                return idx
        return None

    def _canonicalize_tool_step(self, step: dict[str, Any], *, call_key: str) -> tuple[str, str, str]:
        name = str(step.get(call_key, ""))
        requestor = str(step.get("requestor", "assistant"))
        arguments = json.dumps(
            self._normalize_step_arguments(name, dict(step.get("arguments", {}))),
            ensure_ascii=False,
            sort_keys=True,
        )
        return name, requestor, arguments

    def _normalize_step_arguments(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(arguments)
        if tool_name == "grant_app_permission":
            app_name = normalized.get("app_name")
            if isinstance(app_name, str):
                lowered = app_name.lower()
                normalized["app_name"] = "messaging" if lowered in {"messages", "message"} else lowered
        return normalized

    def _is_mutating_tool_name(self, tool_name: Any) -> bool:
        return str(tool_name) in MUTATING_REPAIR_TOOL_NAMES

    def _stage5_replay_tool_calls(self, stage4_trace: dict[str, Any]) -> list[dict[str, Any]]:
        replay_calls: list[dict[str, Any]] = []
        for call in stage4_trace.get("executed_tool_calls", []):
            if self._is_mutating_tool_name(call.get("name")):
                replay_calls.append(deepcopy(call))
        return replay_calls

    def _normalize_str_list(self, values: Any) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            return [values]
        out: list[str] = []
        for value in values:
            if isinstance(value, str):
                out.append(value)
            elif isinstance(value, dict):
                candidate = value.get("id") or value.get("customer_id") or value.get("line_id")
                if isinstance(candidate, str):
                    out.append(candidate)
        return list(dict.fromkeys(out))

    def _merge_unique_str_lists(self, *values: Any) -> list[str]:
        out: list[str] = []
        for value in values:
            out.extend(self._normalize_str_list(value))
        return list(dict.fromkeys(out))

    def _coalesce_bool(self, *values: Any) -> bool | None:
        for value in values:
            if isinstance(value, bool):
                return value
        return None

    def _coalesce_number(self, *values: Any) -> float | None:
        for value in values:
            if isinstance(value, (int, float)):
                return float(value)
        return None

    def _normalize_observed_seed(self, observed_seed: dict[str, Any]) -> dict[str, Any]:
        normalized = dict(observed_seed)

        if "internet_speed_desc" in normalized:
            value = normalized["internet_speed_desc"]
            normalized["internet_speed_desc"] = self._normalize_speed_desc(value)

        if "service_status" in normalized:
            value = str(normalized["service_status"]).strip().lower()
            allowed = {"connected", "searching", "no_service", "emergency_only", "unknown"}
            normalized["service_status"] = value if value in allowed else None

        if "sim_status" in normalized:
            value = str(normalized["sim_status"]).strip().lower()
            allowed = {"active", "missing", "locked_pin", "locked_puk", "unknown"}
            normalized["sim_status"] = value if value in allowed else None

        if "network_mode_preference" in normalized:
            value = str(normalized["network_mode_preference"]).strip().lower()
            allowed = {"2g_only", "3g_only", "4g_only", "4g_5g_preferred", "unknown"}
            normalized["network_mode_preference"] = value if value in allowed else None

        for key in (
            "can_send_mms",
            "mobile_data_working",
            "is_abroad",
            "roaming_enabled_on_device",
            "roaming_enabled_on_account",
            "airplane_mode",
            "wifi_calling_enabled",
            "apn_mms_ok",
            "messaging_sms_permission",
            "messaging_storage_permission",
            "data_usage_exceeded",
        ):
            if key in normalized:
                normalized[key] = self._normalize_optional_bool(normalized[key])

        return normalized

    def _normalize_optional_bool(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "yes", "on"}:
                return True
            if lowered in {"false", "no", "off"}:
                return False
            if lowered in {"unknown", "null", "none", ""}:
                return None
        return None

    def _normalize_speed_desc(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower().replace("-", " ").replace("_", " ")
        mapping = {
            "no connection": "No Connection",
            "very poor": "Very Poor",
            "poor": "Poor",
            "fair": "Fair",
            "good": "Good",
            "excellent": "Excellent",
            "unknown": None,
        }
        return mapping.get(lowered)

    def _normalize_optional_short_text(self, value: Any) -> str | None:
        if not isinstance(value, str):
            return None
        normalized = value.strip()
        return normalized or None

    def _normalize_lookup_confidence(
        self,
        value: Any,
        *,
        used_tools: bool,
        has_phone: bool,
        has_full_name: bool,
    ) -> str:
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"high", "medium", "low"}:
                return normalized
        if used_tools and has_phone:
            return "high"
        if has_phone and has_full_name:
            return "medium"
        return "low"

    def _normalize_target_success_signal(self, *values: Any) -> str:
        for value in values:
            if isinstance(value, str) and value.strip():
                return value.strip()
        return "can_send_mms_true"

    def _normalize_stage1_risk_flags(
        self,
        values: Any,
        *,
        task_instructions: str,
    ) -> list[str]:
        allowed = {
            "mild_frustration_after_unsuccessful_attempt",
            "tool_grounding_required",
        }
        out: list[str] = []
        for value in self._normalize_str_list(values):
            normalized = value.strip().lower()
            if normalized in allowed and normalized not in out:
                out.append(normalized)
        lowered = task_instructions.lower()
        if (
            "mild frustration" in lowered
            and "mild_frustration_after_unsuccessful_attempt" not in out
        ):
            out.append("mild_frustration_after_unsuccessful_attempt")
        if "ground your responses on the results of tool calls" in lowered and "tool_grounding_required" not in out:
            out.append("tool_grounding_required")
        return out

    def _extract_full_name(self, text: str) -> str | None:
        match = re.search(r"you are ([A-Za-z][A-Za-z .'-]+?) with phone number", text, re.IGNORECASE)
        if not match:
            return None
        name = " ".join(match.group(1).split())
        return name.title() if name else None

    def _infer_cannot_send_mms(self, reason_for_call: str) -> bool | None:
        lowered = reason_for_call.lower()
        if "unable to send mms" in lowered or "cannot send mms" in lowered:
            return True
        if "can send mms" in lowered:
            return False
        return None

    def _infer_wants_resolution(self, reason_for_call: str) -> bool | None:
        lowered = reason_for_call.lower()
        if "want to fix" in lowered or "wants to fix" in lowered or "successfully send" in lowered:
            return True
        return None

    def _infer_is_abroad(self, known_info: str) -> bool | None:
        lowered = known_info.lower()
        if "currently abroad" in lowered or "abroad in" in lowered:
            return True
        if "at home" in lowered or "in the united states" in lowered:
            return False
        return None

    def _infer_refuel_allowed(self, task_instructions: str) -> bool | None:
        lowered = task_instructions.lower()
        if "willing to refuel" in lowered:
            return True
        if "do not want to refuel" in lowered:
            return False
        return None

    def _extract_refuel_gb(self, task_instructions: str) -> float | None:
        match = re.search(r"refuel\s+([0-9]+(?:\.[0-9]+)?)\s*gb", task_instructions, re.IGNORECASE)
        if not match:
            return None
        try:
            return float(match.group(1))
        except ValueError:
            return None

    def _infer_plan_change_allowed(self, task_instructions: str) -> bool | None:
        lowered = task_instructions.lower()
        if "do not want to change your mobile data plan" in lowered:
            return False
        if "willing to change your mobile data plan" in lowered:
            return True
        return None

    def _stage1_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"telecom stage1 user grounding; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={min(3, self._max_rounds(agent))}"
        )

    def _stage2_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"telecom stage2 resolution; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={self._max_rounds(agent)}"
        )

    def _stage3_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"telecom stage3 observed-state extraction; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={self._max_rounds(agent)}"
        )

    def _stage4_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"telecom stage4 repair execution; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={max(4, min(6, self._max_rounds(agent) + 1))}"
        )

    def _stage5_prompt_summary(self, agent: AgentSpec) -> str:
        return (
            f"telecom stage5 verification and terminal decision; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"max_rounds={max(3, min(5, self._max_rounds(agent) + 1))}"
        )
