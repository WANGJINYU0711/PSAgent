"""LLM-backed executor for telecom Stage 1/2/3/4/5 execution."""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import uuid
from copy import deepcopy
from datetime import datetime, timezone
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
DEFAULT_BRIDGE_DEBUG_DIR = Path("/tmp/psagent_bridge_failures")
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
FAST_TOKEN_BUDGET_PER_STAGE = 1200
FAST_TOKEN_PENALTY_BLOCK_SIZE = 200
FAST_TOKEN_OVER_BUDGET_PENALTY_PER_BLOCK = 0.25
AGENT_PROFILE_ONLY_EXPERIMENT_SETTING = "telecom_mms_agent_profile_only_clean_v1"


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
        self.attribute_weakening_level = self._attribute_weakening_level_from_env()
        self.experiment_setting = AGENT_PROFILE_ONLY_EXPERIMENT_SETTING

    def _attribute_weakening_level_from_env(self) -> int:
        raw = str(os.environ.get("PSAGENT_ATTRIBUTE_WEAKENING_LEVEL", "0")).strip()
        try:
            level = int(raw)
        except ValueError:
            return 0
        return min(max(level, 0), 4)

    def _attribute_guidance_enabled(self) -> bool:
        # Clean profile-only runs keep stage/task capability hints out of the LLM prompt.
        return False

    def _attribute_weak_skip_enabled(self) -> bool:
        return self._attribute_guidance_enabled() and self.attribute_weakening_level < 2

    def _attribute_verification_priority_enabled(self) -> bool:
        return self._attribute_guidance_enabled() and self.attribute_weakening_level < 3

    def _attribute_prompt_context_sentence(self) -> str:
        if not self._attribute_guidance_enabled():
            return "You are given only the selected agent's deliberation mode and execution profile.\n"
        return (
            "You are given a soft capability-fit summary and the selected agent deliberation mode.\n"
            "Capability-fit cues are weak hints only. They must not override stage evidence, hard rules, or the execution contract.\n"
        )

    def _llm_visible_task_metadata(self, raw_instance: dict[str, Any]) -> dict[str, Any]:
        """Expose run provenance without leaking labels, oracle actions, or profile-switch keys."""

        metadata = raw_instance.get("metadata", {}) or {}
        return {
            "experiment_setting": self.experiment_setting,
            "domain": metadata.get("domain"),
            "problem_family": metadata.get("problem_family"),
            "metadata_visibility": "oracle_and_profile_switch_fields_redacted",
        }

    def _attribute_prompt_fields(
        self,
        task: TaskDescriptor,
        stage_name: str,
        agent: AgentSpec,
    ) -> dict[str, Any]:
        if not self._attribute_guidance_enabled():
            return {}
        return {
            "agent_capability_profile": self._build_agent_capability_profile(agent),
            "current_stage_capability_requirements": self._build_stage_requirement_summary(
                task, stage_name
            ),
            "capability_match_summary": self._build_capability_match_summary(
                task, stage_name, agent
            ),
            "attribute_guidance_mode": (
                "weak_hint_with_verification_priority"
                if self._attribute_verification_priority_enabled()
                else "weak_hint_only"
            ),
            "attribute_guidance_note": (
                "Capability-fit summaries are non-binding hints. Do not treat higher-fit/lower-fit buckets as mandatory routing rules."
            ),
        }

    def _blocker_spec_safe(self, blocker_id: str) -> dict[str, Any]:
        try:
            return get_blocker_spec(blocker_id)
        except Exception:
            return {}

    def _is_shallow_upstream_blocker(self, blocker_id: str) -> bool:
        spec = self._blocker_spec_safe(blocker_id)
        if not spec:
            return False
        return (
            str(spec.get("blocker_layer", "")) in {"service", "data"}
            and int(spec.get("default_priority", 10**6) or 10**6) <= 30
            and not bool(spec.get("assistant_side_required"))
            and not bool(spec.get("hybrid_required"))
        )

    def _is_core_downstream_blocker(self, blocker_id: str) -> bool:
        spec = self._blocker_spec_safe(blocker_id)
        if not spec:
            return False
        return (
            str(spec.get("blocker_layer", "")) == "mms_app"
            or int(spec.get("default_priority", 0) or 0) >= 37
            or bool(spec.get("assistant_side_required"))
            or bool(spec.get("hybrid_required"))
        )

    def _is_nonlocal_or_hybrid_transfer_blocker(self, blocker_id: str) -> bool:
        spec = self._blocker_spec_safe(blocker_id)
        if not spec:
            return False
        if bool(spec.get("hybrid_required")):
            return True
        return bool(spec.get("assistant_side_required")) and not bool(
            spec.get("can_be_deferred")
        )

    def _is_shallow_subset_with_hard_deferred(
        self,
        selected_blocker_ids: list[str],
        deferred_blocker_ids: list[str],
    ) -> bool:
        if not selected_blocker_ids or not deferred_blocker_ids:
            return False
        if any(not self._is_shallow_upstream_blocker(bid) for bid in selected_blocker_ids):
            return False
        return any(
            self._is_nonlocal_or_hybrid_transfer_blocker(bid)
            for bid in deferred_blocker_ids
        )

    def _coerce_stage4_rows_to_transfer_required(
        self,
        normalized_rows: list[dict[str, Any]],
        *,
        refusal_code: str,
    ) -> None:
        for row in normalized_rows:
            row["should_repair"] = False
            row["oracle_execute_decision"] = "transfer"
            row["adjudication_label"] = (
                str(row.get("adjudication_label", "transfer_unspecified_blocker"))
                .replace("repair_", "transfer_")
                .replace("defer_", "transfer_")
            )
            row["refusal_code"] = refusal_code

    def _stage5_verification_tools_for_blocker_ids(
        self,
        blocker_ids: list[str],
    ) -> list[str]:
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
            checklist.append("check_app_permissions")
        if blocker_ids:
            checklist.append("can_send_mms")
        return list(dict.fromkeys(checklist))

    def _stage5_verification_floor_met(
        self,
        *,
        selected_blocker_ids: list[str],
        deferred_blocker_ids: list[str],
        verification_evidence: list[str],
    ) -> bool:
        if not self._is_shallow_subset_with_hard_deferred(
            selected_blocker_ids, deferred_blocker_ids
        ):
            return True
        evidence_set = set(verification_evidence)
        if not evidence_set:
            return False
        expected_tools = set(
            self._stage5_verification_tools_for_blocker_ids(selected_blocker_ids)
        )
        blocker_matched_tools = expected_tools - {"can_send_mms"}
        return "can_send_mms" in evidence_set and bool(blocker_matched_tools & evidence_set)

    def _base_round_budget(self, agent: AgentSpec, stage_name: str | None = None) -> int:
        mode = getattr(agent, "deliberation_mode", "deep")
        if mode == "fast":
            rounds_by_stage = {
                "stage1": 2,
                "stage2": 2,
                "stage3": 2,
                "stage4": 2,
                "stage5": 2,
            }
        else:
            rounds_by_stage = {
                "stage1": 4,
                "stage2": 5,
                "stage3": 6,
                "stage4": 6,
                "stage5": 7,
            }
        return int(rounds_by_stage.get(stage_name or "stage3", 2 if mode == "fast" else 5))

    def _estimate_total_tokens_stage(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        result: dict[str, Any],
    ) -> float:
        total_chars = len(system_prompt or "") + len(user_prompt or "")
        for row in result.get("llm_messages", []) or []:
            total_chars += len(str(row.get("content") or ""))
            total_chars += len(json.dumps(row.get("tool_calls", []) or [], ensure_ascii=False))
        return max(1.0, float(total_chars) / 4.0)

    def _llm_stage_resource_trace_fields(
        self,
        result: dict[str, Any],
        *,
        stage_name: str,
        agent: AgentSpec,
        task: TaskDescriptor,
        system_prompt: str,
        user_prompt: str,
        base_round_budget: int,
        max_rounds_allowed: int,
        diagnostic_fallback_used: bool = False,
        verification_fallback_used: bool = False,
        fallback_used: bool = False,
    ) -> dict[str, Any]:
        usage = dict(result.get("resource_usage", {}) or {})
        deliberation_mode = str(getattr(agent, "deliberation_mode", "deep")).strip().lower()
        stage_requirement = self._stage_deliberation_requirement(task, stage_name)
        llm_call_count_stage = int(usage.get("llm_call_count", 0) or 0)
        prompt_tokens_total_stage = float(usage.get("prompt_tokens_total", 0.0) or 0.0)
        completion_tokens_total_stage = float(
            usage.get("completion_tokens_total", 0.0) or 0.0
        )
        total_tokens_total_stage = float(usage.get("total_tokens_total", 0.0) or 0.0)
        token_usage_available = bool(usage.get("token_usage_available", False))
        estimated_total_tokens_stage = float(
            usage.get("estimated_total_tokens_total", 0.0) or 0.0
        )
        if token_usage_available and total_tokens_total_stage > 0.0:
            estimated_total_tokens_stage = total_tokens_total_stage
        elif estimated_total_tokens_stage <= 0.0:
            estimated_total_tokens_stage = self._estimate_total_tokens_stage(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                result=result,
            )

        token_budget_stage = (
            float(FAST_TOKEN_BUDGET_PER_STAGE) if deliberation_mode == "fast" else 0.0
        )
        token_over_budget_stage = 0.0
        token_over_budget_units = 0
        token_over_budget_penalty = 0.0
        if deliberation_mode == "fast" and token_usage_available and token_budget_stage > 0.0:
            token_over_budget_stage = max(0.0, total_tokens_total_stage - token_budget_stage)
            if token_over_budget_stage > 0.0:
                token_over_budget_units = int(
                    math.ceil(token_over_budget_stage / float(FAST_TOKEN_PENALTY_BLOCK_SIZE))
                )
                token_over_budget_penalty = (
                    token_over_budget_units * FAST_TOKEN_OVER_BUDGET_PENALTY_PER_BLOCK
                )
        return {
            "stage_id": stage_name,
            "stage_name": stage_name,
            "agent_deliberation_mode": deliberation_mode,
            "stage_requirement": stage_requirement,
            "base_round_budget": int(base_round_budget),
            "max_rounds_allowed": int(max_rounds_allowed),
            "llm_call_count_stage": llm_call_count_stage,
            "llm_call_count_over_base_budget": max(0, llm_call_count_stage - int(base_round_budget)),
            "valid_json_first_try": bool(usage.get("valid_json_first_try", False)),
            "json_retry_count": int(usage.get("json_retry_count", 0) or 0),
            "diagnostic_fallback_used": bool(diagnostic_fallback_used),
            "verification_fallback_used": bool(verification_fallback_used),
            "fallback_used": bool(
                fallback_used or diagnostic_fallback_used or verification_fallback_used
            ),
            "replay_tool_call_count": int(len(result.get("replay_tool_calls", []) or [])),
            "prompt_tokens_stage": prompt_tokens_total_stage,
            "completion_tokens_stage": completion_tokens_total_stage,
            "total_tokens_stage": total_tokens_total_stage,
            "prompt_tokens_total_stage": prompt_tokens_total_stage,
            "completion_tokens_total_stage": completion_tokens_total_stage,
            "total_tokens_total_stage": total_tokens_total_stage,
            "token_usage_available": token_usage_available,
            "estimated_total_tokens_stage": estimated_total_tokens_stage,
            "token_budget_stage": token_budget_stage,
            "token_over_budget_stage": token_over_budget_stage,
            "token_over_budget_units": token_over_budget_units,
            "token_over_budget_penalty": token_over_budget_penalty,
            "is_fast_agent": deliberation_mode == "fast",
            "is_deep_agent": deliberation_mode == "deep",
            "api_cost_total_usd_stage": float(
                usage.get("api_cost_total_usd_raw", 0.0) or 0.0
            ),
            "generation_time_total_seconds_stage": float(
                usage.get("generation_time_total_seconds", 0.0) or 0.0
            ),
            "llm_round_trip_total_seconds_stage": float(
                usage.get("llm_round_trip_total_seconds", 0.0) or 0.0
            ),
            "tool_wall_clock_total_seconds_stage": float(
                usage.get("tool_wall_clock_total_seconds", 0.0) or 0.0
            ),
            "stage_wall_clock_seconds": float(
                usage.get("stage_wall_clock_seconds", 0.0) or 0.0
            ),
            "usage_breakdown_stage": deepcopy(usage.get("usage_breakdown", {})),
            "cost_breakdown_stage": deepcopy(usage.get("cost_breakdown", {})),
        }

    def _run_stage1(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        system_prompt, user_prompt = self._build_stage1_prompts(task, agent, raw_instance)
        base_round_budget = self._base_round_budget(agent, "stage1")
        max_rounds_allowed = self._max_rounds(agent, task, "stage1")
        result = self._run_llm_stage_bridge(
            stage_name="stage1",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=["get_customer_by_phone", "get_details_by_id"],
            max_rounds=max_rounds_allowed,
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
            "prompt_summary": self._stage1_prompt_summary(agent, task),
            "llm_raw_output": deepcopy(result.get("llm_messages", [])),
            "planned_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "executed_tool_calls": deepcopy(result.get("executed_tool_calls", [])),
            "tool_results": deepcopy(result.get("tool_results", [])),
            "tool_errors": deepcopy(result.get("tool_errors", [])),
            "db_hash_before": result.get("db_hash_before"),
            "db_hash_after": result.get("db_hash_after"),
            "input": {
                "user_context": deepcopy(raw_instance.get("user_context", {})),
                "task_metadata": deepcopy(self._llm_visible_task_metadata(raw_instance)),
                "task_id": task.task_id,
            },
            "output": deepcopy(output),
            "score": None,
            "source": "llm_bench",
            **self._llm_stage_resource_trace_fields(
                result,
                stage_name="stage1",
                agent=agent,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_round_budget=base_round_budget,
                max_rounds_allowed=max_rounds_allowed,
            ),
        }
        return {
            "input": {
                "user_context": deepcopy(raw_instance.get("user_context", {})),
                "task_metadata": deepcopy(self._llm_visible_task_metadata(raw_instance)),
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
        base_round_budget = self._base_round_budget(agent, "stage2")
        max_rounds_allowed = self._max_rounds(agent, task, "stage2")
        result = self._run_llm_stage_bridge(
            stage_name="stage2",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=["get_customer_by_phone", "get_details_by_id"],
            max_rounds=max_rounds_allowed,
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
            "prompt_summary": self._stage2_prompt_summary(agent, task),
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
            **self._llm_stage_resource_trace_fields(
                result,
                stage_name="stage2",
                agent=agent,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_round_budget=base_round_budget,
                max_rounds_allowed=max_rounds_allowed,
            ),
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
        base_round_budget = self._base_round_budget(agent, "stage3")
        max_rounds_allowed = self._max_rounds(agent, task, "stage3")
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
            max_rounds=max_rounds_allowed,
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
            "prompt_summary": self._stage3_prompt_summary(agent, task),
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
            **self._llm_stage_resource_trace_fields(
                result,
                stage_name="stage3",
                agent=agent,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_round_budget=base_round_budget,
                max_rounds_allowed=max_rounds_allowed,
                diagnostic_fallback_used=diagnostic_fallback["used"] or fallback_debug["used"],
                fallback_used=diagnostic_fallback["used"] or fallback_debug["used"],
            ),
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
        base_round_budget = self._base_round_budget(agent, "stage4")
        max_rounds_allowed = self._max_rounds(agent, task, "stage4")
        result = self._run_llm_stage_bridge(
            stage_name="stage4",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=list(STAGE4_REPAIR_TOOLS),
            max_rounds=max_rounds_allowed,
        )
        execution_result = self._execute_stage4_canonical_plan(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            final_output=result.get("final_output"),
            allow_deep_local_completion=self._allow_stage4_deep_local_completion(
                task=task,
                agent=agent,
                raw_instance=raw_instance,
                stage3_output=stage3_output,
            ),
        )
        output = self._normalize_stage4_output(
            final_output=result.get("final_output"),
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            allow_deep_local_completion=self._allow_stage4_deep_local_completion(
                task=task,
                agent=agent,
                raw_instance=raw_instance,
                stage3_output=stage3_output,
            ),
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
            "prompt_summary": self._stage4_prompt_summary(agent, task),
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
            **self._llm_stage_resource_trace_fields(
                result,
                stage_name="stage4",
                agent=agent,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_round_budget=base_round_budget,
                max_rounds_allowed=max_rounds_allowed,
                fallback_used=result.get("final_output") is None,
            ),
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
        base_round_budget = self._base_round_budget(agent, "stage5")
        max_rounds_allowed = self._max_rounds(agent, task, "stage5")
        result = self._run_llm_stage_bridge(
            stage_name="stage5",
            original_task_id=str(raw_instance.get("original_task_id", "")),
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            allowed_tools=list(STAGE5_VERIFICATION_TOOLS),
            max_rounds=max_rounds_allowed,
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
            "prompt_summary": self._stage5_prompt_summary(agent, task),
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
            "stage5_local_policy_eval_debug": deepcopy(
                result.get("policy_eval_debug")
            ),
            **self._llm_stage_resource_trace_fields(
                result,
                stage_name="stage5",
                agent=agent,
                task=task,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                base_round_budget=base_round_budget,
                max_rounds_allowed=max_rounds_allowed,
                verification_fallback_used=verification_fallback["used"],
                fallback_used=verification_fallback["used"],
            ),
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
        cmd = [str(self.venv_python), str(self.llm_bridge_script)]
        child_env = os.environ.copy()
        child_env.setdefault("PYTHONUTF8", "1")
        child_env.setdefault("PYTHONIOENCODING", "utf-8")
        proc = subprocess.run(
            cmd,
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            cwd=str(self.tau2_root),
            env=child_env,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Telecom LLM bench bridge failed for {stage_name}: "
                + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
            )
        return self._parse_bridge_stdout(
            proc=proc,
            cmd=cmd,
            cwd=str(self.tau2_root),
            payload=payload,
        )

    def _parse_bridge_stdout(
        self,
        *,
        proc: subprocess.CompletedProcess[str],
        cmd: list[str],
        cwd: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Parse the bridge stdout protocol with diagnostics for polluted stdout.

        Normal bridge output must be a single JSON object. If stdout was polluted
        by dependency logs but the last non-empty line is JSON, recover and mark
        the returned payload. Otherwise persist raw stdout/stderr and fail with a
        diagnostic path.
        """

        try:
            parsed = json.loads(proc.stdout)
            if isinstance(parsed, dict):
                return parsed
            raise TypeError(f"Bridge stdout JSON is {type(parsed).__name__}, not object")
        except Exception as strict_error:
            non_empty_lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
            if non_empty_lines:
                last_line = non_empty_lines[-1]
                try:
                    recovered = json.loads(last_line)
                    if isinstance(recovered, dict):
                        recovered["_bridge_stdout_recovered"] = True
                        recovered["_bridge_stdout_extra_line_count"] = max(
                            0,
                            len(non_empty_lines) - 1,
                        )
                        recovered["_bridge_stdout_recovery_mode"] = "last_json_line"
                        return recovered
                except Exception:
                    pass

            dump_path = self._write_bridge_parse_failure_dump(
                strict_error=strict_error,
                proc=proc,
                cmd=cmd,
                cwd=cwd,
                payload=payload,
            )
            stdout_preview = self._preview_text(proc.stdout)
            stderr_preview = self._preview_text(proc.stderr)
            raise RuntimeError(
                "Telecom LLM bench bridge JSON parse failure. "
                f"Diagnostic dump: {dump_path}. "
                f"stdout preview: {stdout_preview!r}. "
                f"stderr preview: {stderr_preview!r}."
            ) from strict_error

    def _write_bridge_parse_failure_dump(
        self,
        *,
        strict_error: Exception,
        proc: subprocess.CompletedProcess[str],
        cmd: list[str],
        cwd: str,
        payload: dict[str, Any],
    ) -> Path:
        debug_dir = Path(
            os.environ.get("PSAGENT_BRIDGE_DEBUG_DIR", str(DEFAULT_BRIDGE_DEBUG_DIR))
        )
        debug_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        stage_name = str(payload.get("stage_name") or "unknown_stage")
        safe_stage_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", stage_name)[:80]
        dump_path = (
            debug_dir
            / f"bridge_json_decode_failure_{timestamp}_{safe_stage_name}_{uuid.uuid4().hex}.json"
        )
        diagnostic = {
            "error_type": type(strict_error).__name__,
            "error_message": str(strict_error),
            "returncode": proc.returncode,
            "cmd": cmd,
            "cwd": cwd,
            "stdout_raw": proc.stdout,
            "stderr_raw": proc.stderr,
            "stdout_preview": self._preview_text(proc.stdout),
            "stderr_preview": self._preview_text(proc.stderr),
            "payload_stage_name": payload.get("stage_name"),
            "payload_task_id": payload.get("original_task_id"),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        dump_path.write_text(json.dumps(diagnostic, indent=2, ensure_ascii=False))
        return dump_path

    @staticmethod
    def _preview_text(text: str | None, max_chars: int = 1200) -> str:
        if not text:
            return ""
        text = text.replace("\x00", "\\0")
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3] + "..."

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
        del task
        mode = getattr(agent, "deliberation_mode", "deep")
        rounds = self._base_round_budget(agent, stage_name)
        return max(2, min(8, rounds))

    def _agent_behavior_guidance(self, agent: AgentSpec) -> str:
        competence = (
            "Prefer explicit tool evidence and do not commit on unverified high-risk fields."
            if agent.competence_level == "high"
            else "Keep the investigation compact and stop once the stage output is sufficiently supported."
        )
        scope = (
            "Broaden the search only when customer or line resolution is genuinely ambiguous."
            if agent.scope_level == "broad"
            else "Stay on the most explicit phone-number path first and avoid low-yield branches."
        )
        stability = (
            "Do not spend rounds on redundant checks because the round budget is tight."
            if agent.stability_level == "unstable"
            else "Use the available round budget on the highest-risk facts only."
        )
        deliberation = (
            "This agent is fast: prioritize the shortest valid evidence chain, avoid redundant calls, and do second-pass verification only when the current fact is decisive for the requested output."
            if getattr(agent, "deliberation_mode", "deep") == "fast"
            else "This agent is deep: spend budget on careful verification of high-risk facts and do not finalize stage3-stage5 outputs before decisive evidence is checked."
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
        ranked = sorted(
            agent.attribute_skill.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        )
        low_ranked = sorted(agent.attribute_skill.items(), key=lambda item: (item[1], item[0]))
        return {
            "higher_fit_axes": [[name, round(float(score), 3)] for name, score in ranked[:5]],
            "lower_fit_axes": [[name, round(float(score), 3)] for name, score in low_ranked[:5]],
        }

    def _build_agent_deliberation_profile(self, agent: AgentSpec) -> dict[str, Any]:
        mode = getattr(agent, "deliberation_mode", "deep")
        return {
            "mode": mode,
            "style_guidance": (
                "Use the shortest valid evidence chain, avoid low-value extra tool calls, and do second-pass verification only when the current fact is decisive for the requested output."
                if mode == "fast"
                else "Use extra rounds on the highest-risk fields, cross-check decisive evidence, and do not rush stage3-stage5 decisions."
            ),
        }

    def _build_stage_requirement_summary(self, task: TaskDescriptor, stage_name: str) -> list[list[Any]]:
        requirement = self._stage_requirement_map(task, stage_name)
        ranked = sorted(requirement.items(), key=lambda item: (item[1], item[0]), reverse=True)
        return [[name, round(float(score), 3)] for name, score in ranked[:4] if score > 0.0]

    def _build_capability_match_summary(
        self,
        task: TaskDescriptor,
        stage_name: str,
        agent: AgentSpec,
    ) -> dict[str, Any]:
        requirement = self._stage_requirement_map(task, stage_name)
        weighted_rows: list[dict[str, Any]] = []
        for capability_name, weight in sorted(
            requirement.items(),
            key=lambda item: (item[1], item[0]),
            reverse=True,
        ):
            if float(weight) <= 0.0:
                continue
            skill_score = float(agent.attribute_skill.get(capability_name, 0.0))
            if skill_score >= 0.75:
                priority_bucket = "higher_fit"
            elif skill_score <= 0.4:
                priority_bucket = "lower_fit"
            else:
                priority_bucket = "middle_fit"
            weighted_rows.append(
                {
                    "capability": capability_name,
                    "requirement_weight": round(float(weight), 3),
                    "agent_skill": round(skill_score, 3),
                    "priority_bucket": priority_bucket,
                }
            )
        return {
            "required_capability_table": weighted_rows[:5],
            "higher_fit_capabilities": [
                row["capability"] for row in weighted_rows if row["priority_bucket"] == "higher_fit"
            ][:3],
            "lower_fit_capabilities": [
                row["capability"] for row in weighted_rows if row["priority_bucket"] == "lower_fit"
            ][:3],
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
                "This stage rewards deeper reasoning, blocker-specific evidence chains, decisive precondition completion, and extra verification on high-risk facts before closure."
                if requirement == "deep"
                else "This stage rewards quick resolution with compact evidence, limited tool use, and earlier closure once the minimum floor is met."
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
                "Agent is fast on a deep-reasoning stage: use the limited budget to verify the highest-risk facts first and do not close high-risk decisions before the decisive evidence chain is checked."
                if requirement == "deep" and mode == "fast"
                else "Agent is deep on a fast stage: avoid redundant tool calls and stop after enough evidence."
                if requirement == "fast" and mode == "deep"
                else "Agent deliberation style is aligned with this stage."
            ),
            "round_budget_hint": self._max_rounds(agent, task, stage_name),
        }

    def _stage_execution_search_policy(
        self,
        stage_name: str,
        *,
        mode: str,
    ) -> list[str]:
        if mode == "fast":
            policy = [
                "Use the shortest valid evidence chain that supports the required structured output.",
                "Do not do secondary verification unless the current fact is decisive for the output.",
                "Keep the search compact and avoid broad speculative branching.",
            ]
            if self._attribute_weak_skip_enabled():
                policy.append(
                    "If a capability axis appears to be a relative lower-fit area for this route, do only the minimum confirmation needed there before returning to higher-yield evidence."
                )
        else:
            policy = [
                "Cross-check the highest-risk facts before finalizing the structured output.",
                "Spend extra rounds on the highest-risk evidence first, not on broad low-yield exploration.",
                "Do not finalize stage3-stage5 outputs before the key risk-bearing evidence is verified.",
            ]
            if self._attribute_verification_priority_enabled():
                policy.append(
                    "If multiple evidence branches remain equally plausible, capability-fit priorities may break ties weakly, but only after risk and stage-goal needs are considered."
                )

        if stage_name == "stage3":
            if mode == "fast":
                policy.append(
                    "Keep diagnosis compact: concentrate on the highest-yield 1-2 blocker families rather than exhaustively branching."
                )
                policy.append(
                    "If a high-priority blocker family lacks decisive evidence, keep it open rather than aggressively eliminating it."
                )
            else:
                policy.append(
                    "Cross-confirm the most important blocker families before you lock in the observed_state."
                )
                policy.append(
                    "Use extra rounds on decisive blocker-family evidence, not on long-tail families with weak support."
                )
        elif stage_name == "stage4":
            if mode == "fast":
                policy.append(
                    "Do not expand into long repair chains; prefer only clear, high-confidence repair execution."
                )
                policy.append(
                    "If blocker-family evidence is still ambiguous, prefer narrow or deferred repair over a broad repair bundle."
                )
            else:
                policy.append(
                    "Before repair execution, confirm the key preconditions for the blocker families you are acting on."
                )
                policy.append(
                    "Do not treat a common repair as a default. Repair only when blocker-specific evidence keeps that family plausibly active."
                )
        elif stage_name == "stage5":
            if mode == "fast":
                policy.append(
                    "Once the minimum verification floor is met, close quickly instead of exploring extra verification branches."
                )
                policy.append(
                    "If the minimum verification floor is not met, prefer a narrow repair_subset or defer uncertain blockers; use transfer only for explicit external/manual handling."
                )
            else:
                policy.append(
                    "Use the larger budget to resolve terminal ambiguity with verification, then commit to the evidence-supported repair_all or repair_subset action."
                )
                policy.append(
                    "Do not use transfer as a substitute for verification; reserve transfer for explicit external/manual handling or a stage4 transfer_required plan."
                )
        return policy

    def _capability_mismatch_policy(
        self,
        stage_name: str,
        *,
        mode: str,
    ) -> str:
        del stage_name
        del mode
        if self._attribute_weak_skip_enabled():
            return (
                "Capability-fit cues may weakly suggest where this route is relatively higher-fit or lower-fit, "
                "but they are not binding and do not justify skipping decisive evidence."
            )
        return "No stage-level fast/deep requirement is visible. Follow only the selected agent deliberation profile, stage goal, evidence floor, and round budget."

    def _stage3_blocker_decision_rules(self, *, mode: str) -> list[str]:
        rules = [
            "Treat blocker-family selection and elimination as a hard decision boundary, not a descriptive summary.",
            "Do not exclude a high-priority blocker family unless the observed-state evidence actively argues against it.",
            "Lack of evidence is not sufficient grounds to eliminate a stage-relevant blocker family.",
            "Be especially conservative about eliminating APN, roaming, permission, and network blocker families when the task profile makes them high-priority.",
        ]
        if mode == "fast":
            return rules + [
                "Only compactly validate the top 1-2 blocker families.",
                "Do not widen into long-tail blocker families after a plausible high-yield family is sufficiently supported.",
                "Do not confirm multiple blocker families on weak evidence.",
            ]
        return rules + [
            "Keep a wider candidate set if needed, but cross-check the decisive evidence for the top blocker family before finalizing observed_state.",
            "Do not finalize observed_state while the top blocker family's decisive evidence remains unchecked.",
            "Do not eliminate APN, roaming, SIM, or other deep-path blocker families without explicit contradictory evidence when they are high-priority for the task profile.",
        ]

    def _stage4_repair_precondition_rules(self, *, mode: str) -> list[str]:
        rules = [
            "Do not execute a repair for a blocker family unless the evidence confirms that family is still plausibly active.",
            "APN repairs require APN-side evidence. Roaming repairs require roaming-side evidence. Permission and network repairs require matching blocker evidence.",
            "A repair cannot be justified only because it is common or convenient.",
            "If APN, roaming, SIM, or permission evidence is affirmative, mark that blocker repairable instead of defaulting the whole case to transfer.",
        ]
        if mode == "fast":
            return rules + [
                "Only execute clear, high-confidence, low-branching repairs.",
                "Do not initiate multi-step repair chains.",
                "Do not execute APN, roaming, SIM, or other high-risk repairs without direct blocker evidence.",
            ]
        return rules + [
            "Before repair, confirm the decisive preconditions for the blocker family being acted on.",
            "When APN, roaming, SIM, or other high-risk evidence is affirmative, use the deeper budget to satisfy the last missing repair preconditions and mark the supported blocker should_repair=true.",
            "If the blocker family remains ambiguous, prefer a narrower or deferred action over a broad repair bundle.",
            "Do not convert the whole case to transfer_required merely because one blocker remains ambiguous if a safe repairable subset is supported.",
            "If the missing evidence chain is still local and bounded, prefer deferred blockers or a supported repair subset over transfer_required.",
            "Do not repair through ambiguity just because more budget is available.",
        ]

    def _stage5_terminal_decision_rules(self, *, mode: str) -> list[str]:
        rules = [
            "Choose the terminal action from blocker-specific evidence and the stage4 repairability plan.",
            "Transfer is not the default uncertainty action; it requires explicit external/manual handling, stage4 transfer_required, or verified local repair impossibility.",
            "repair_all and repair_subset are normal evidence-supported actions, not fallback actions.",
            "Deferred blockers can block repair_all, but when stage4 already supports a local repair subset they are not, by themselves, sufficient reason to choose transfer.",
        ]
        if mode == "fast":
            return rules + [
                "Use the minimum verification floor required for a defensible final action.",
                "If that floor is not met, choose repair_subset for supported blockers and defer ambiguous blockers rather than using broad repair_all.",
                "Use transfer only when the evidence or stage4 plan explicitly says local repair is not appropriate.",
            ]
        return rules + [
            "Use the deeper budget to verify decisive evidence, then commit to the supported repair_all or repair_subset action.",
            "Distinguish between a true external/manual blocker, a still-incomplete local evidence chain, and an already-supported local closure. Only the first state justifies transfer.",
            "Do not use transfer as a substitute for resolving ambiguity through verification.",
            "If stage4 marks blockers repairable and verification does not contradict repair, prefer repair_all or repair_subset over transfer.",
        ]

    def _stage5_system_terminal_rules(self, *, mode: str) -> str:
        if mode == "fast":
            mode_rules = [
                "Fast terminal policy: close quickly after the minimum verification floor.",
                "Fast terminal policy: when evidence is partial, prefer repair_subset plus deferred blockers over a broad repair_all.",
                "Fast terminal policy: deferred blockers can stop repair_all, but if stage4 already supports a local subset they do not alone force transfer.",
                "Fast terminal policy: transfer only when the evidence or stage4 plan explicitly requires external/manual handling.",
            ]
        else:
            mode_rules = [
                "Deep terminal policy: use verification budget to resolve ambiguity, then commit to the evidence-supported repair_all or repair_subset action.",
                "Deep terminal policy: deferred blockers can stop repair_all, but if stage4 already supports a local subset they do not alone force transfer.",
                "Deep terminal policy: transfer is reserved for explicit external/manual handling, stage4 transfer_required, or verified local repair impossibility.",
                "Deep terminal policy: if the case still needs more local verification, spend the remaining budget there before considering transfer.",
                "Deep terminal policy: do not choose transfer simply because the case is complex or verification took more rounds.",
            ]
        return "\n".join(f"- {rule}" for rule in mode_rules)

    def _build_agent_execution_contract(
        self,
        task: TaskDescriptor,
        stage_name: str,
        agent: AgentSpec,
    ) -> dict[str, Any]:
        mode = getattr(agent, "deliberation_mode", "deep")
        stop_policy = (
            "Close once the minimum verification floor is met and the structured output is supported."
            if mode == "fast"
            else "Do not close until decisive evidence for the high-risk fields has been checked."
        )
        evidence_policy = (
            "Prefer explicit tool evidence; do not rely on broad speculative branching."
            if mode == "fast"
            else "Prefer explicit tool evidence and cross-check the decisive evidence path before committing."
        )
        return {
            "contract_version": "telecom_agent_profile_only_execution_contract_v1",
            "deliberation_mode": mode,
            "round_budget_hint": self._max_rounds(agent, task, stage_name),
            "round_budget_band": "2-3" if mode == "fast" else "5-8",
            "evidence_policy": evidence_policy,
            "stop_policy": stop_policy,
            "profile_policy": self._capability_mismatch_policy(stage_name, mode=mode),
            "search_policy": self._stage_execution_search_policy(
                stage_name,
                mode=mode,
            ),
            "stage_specific_hard_constraints": self._stage_execution_hard_constraints(
                stage_name,
                mode=mode,
            ),
            "attribute_guidance_mode": (
                "disabled"
                if not self._attribute_guidance_enabled()
                else "weak_hint_with_verification_priority"
                if self._attribute_verification_priority_enabled()
                else "weak_hint_only"
            ),
            "attribute_guidance_note": (
                "Capability-fit summaries are non-binding hints only. They may shape tie-breaks when evidence is otherwise equal, but they must not dictate search scope, stopping, or terminal action."
                if self._attribute_guidance_enabled()
                else "Capability-fit guidance disabled."
            ),
        }

    def _stage_execution_hard_constraints(
        self,
        stage_name: str,
        *,
        mode: str,
    ) -> list[str]:
        common_rules = [
            "Treat the execution contract as binding. If the contract conflicts with a broad search instinct, follow the contract.",
            "Do not spend rounds outside the stage goal and output contract.",
        ]
        if stage_name == "stage3":
            return common_rules + self._stage3_blocker_decision_rules(mode=mode)
        if stage_name == "stage4":
            return common_rules + self._stage4_repair_precondition_rules(mode=mode)
        if stage_name == "stage5":
            return common_rules + self._stage5_terminal_decision_rules(mode=mode)
        return common_rules

    def _execution_contract_system_rules(self, stage_name: str) -> str:
        stage_specific_rules = {
            "stage1": (
                "Stage 1 rule: ground identity and target line minimally. Do not spend rounds on diagnosis or blocker inference."
            ),
            "stage2": (
                "Stage 2 rule: resolve the customer and line only. Stop once the resolved line is stable enough for downstream stages."
            ),
            "stage3": (
                "Stage 3 rule: fast agents may only compactly validate the highest-yield blocker families; deep agents must cross-check key blocker families before returning."
            ),
            "stage4": (
                "Stage 4 rule: fast agents must avoid multi-step repair chains; deep agents must confirm key execution preconditions before acting."
            ),
            "stage5": (
                "Stage 5 rule: fast agents may close quickly after the minimum verification floor; deep agents must verify the decisive evidence behind the final action."
            ),
        }
        return "\n".join(
            [
                "You must follow the agent execution contract exactly.",
                "The execution contract is binding. Treat it as an operating contract, not advisory guidance.",
                "1. search_policy is binding: it defines how wide or narrow your search may be.",
                "2. stop_policy is binding: it defines when you must stop and when you may not stop yet.",
                "3. profile_policy is binding: only the selected agent's deliberation profile may change search depth.",
                "4. stage_specific_hard_constraints are binding: do not override them with broad search instincts.",
                "5. If the contract says the agent is fast, use the shortest valid evidence chain, avoid redundant verification, and do second-pass checking only when decisive evidence forces it.",
                "6. If the contract says the agent is deep, spend extra rounds on the highest-risk evidence and do not finalize a high-risk output before decisive evidence is verified.",
                "7. Stay within the round_budget_hint implied by the contract.",
                f"8. {stage_specific_rules[stage_name]}",
                (
                    "9. Capability-fit summaries, when present, are weak hints only. Do not treat them as mandatory routing rules or as justification to skip decisive evidence."
                    if self._attribute_guidance_enabled()
                    else "9. No attribute routing rules are active in this run."
                ),
            ]
        )

    def _build_stage2_prompts(
        self,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage1_output: dict[str, Any],
    ) -> tuple[str, str]:
        execution_contract = self._build_agent_execution_contract(task, "stage2", agent)
        system_prompt = (
            "You are performing Stage 2: customer and line resolution for a telecom support case.\n"
            "Goal: identify the customer, resolve the target line, and extract a minimal account snapshot.\n"
            + self._attribute_prompt_context_sentence()
            + self._execution_contract_system_rules("stage2")
            + "\n"
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
                **self._attribute_prompt_fields(task, "stage2", agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "agent_execution_contract": execution_contract,
                "stage_goal": "Resolve the customer and telecom line only.",
                "stage1_output": stage1_output,
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": self._llm_visible_task_metadata(raw_instance),
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
        execution_contract = self._build_agent_execution_contract(task, "stage1", agent)
        system_prompt = (
            "You are performing Stage 1: user grounding for a telecom MMS troubleshooting case.\n"
            "Your goal is to transform the user request into a stable structured Stage 1 output.\n"
            + self._attribute_prompt_context_sentence()
            + self._execution_contract_system_rules("stage1")
            + "\n"
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
                **self._attribute_prompt_fields(task, "stage1", agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "agent_execution_contract": execution_contract,
                "stage_goal": "Ground the user, phone number, and target telecom line at a minimal level only.",
                "policy_mode": "grounding_only_minimal_lookup",
                "user_context": raw_instance.get("user_context", {}),
                "task_metadata": self._llm_visible_task_metadata(raw_instance),
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
        execution_contract = self._build_agent_execution_contract(task, "stage3", agent)
        system_prompt = (
            "You are performing Stage 3: observed-state extraction for a telecom MMS troubleshooting case.\n"
            "Goal: collect factual observed state only. Do not decide terminal actions.\n"
            + self._attribute_prompt_context_sentence()
            + self._execution_contract_system_rules("stage3")
            + "\n"
            "Use only the allowed tools. Prefer explicit tool evidence over guesses.\n"
            "For MMS diagnosis, permission / APN / network-mode checks are usually the highest-yield starting points, but the execution contract decides where this agent should focus first.\n"
            "Do not branch into exhaustive low-yield diagnostics when the execution contract says to stay compact.\n"
            "Hard blocker rules:\n"
            "- Blocker-family inclusion and exclusion must be evidence-backed.\n"
            "- Do not exclude a high-priority blocker family unless the observed-state evidence actively argues against it.\n"
            "- Lack of evidence is not enough to eliminate APN, roaming, permission, or network blocker families when they are stage-relevant.\n"
            "- Fast agents may only compactly validate the top 1-2 blocker families.\n"
            "- Deep agents must cross-check the decisive evidence for the top blocker family before finalizing observed_state.\n"
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
                **self._attribute_prompt_fields(task, "stage3", agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "agent_execution_contract": execution_contract,
                "stage3_blocker_decision_rules": self._stage3_blocker_decision_rules(
                    mode=getattr(agent, "deliberation_mode", "deep")
                ),
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
                "task_metadata": self._llm_visible_task_metadata(raw_instance),
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
        execution_contract = self._build_agent_execution_contract(task, "stage4", agent)
        system_prompt = (
            "You are performing Stage 4: blocker adjudication and repair execution for a telecom MMS troubleshooting case.\n"
            + self._attribute_prompt_context_sentence()
            + self._execution_contract_system_rules("stage4")
            + "\n"
            "First decide, for each blocker, whether it should be repaired automatically, deferred, or transferred.\n"
            "Then execute canonical repair steps only for blockers with should_repair=true.\n"
            "Use only the allowed repair tools.\n"
            "Do not do fresh diagnosis beyond minimal execution-time grounding.\n"
            "Do not produce customer-facing prose.\n"
            "Return JSON only.\n"
            "Your output must be a JSON object with top-level keys: per_blocker, repairability, transfer_reason, decision_policy_version.\n"
            "per_blocker must include every input blocker_id exactly once and each row must contain blocker_id and should_repair.\n"
            "Allowed repairability values: repairable, partially_repairable, transfer_required.\n"
            "Stage4 policy bias:\n"
            "- Do not choose transfer_required for the whole case only because one blocker is hybrid-required.\n"
            "- If at least one blocker has affirmative evidence and a safe local repair path, mark that blocker should_repair=true and use partially_repairable when other blockers must remain deferred.\n"
            "- If a blocker is assistant-side-required but can be safely deferred, it may be marked should_repair=false under partially_repairable.\n"
            "- Use transfer_required only when no safe local repair subset exists or the remaining blockers explicitly require external/manual handling.\n"
            "Execution rules:\n"
            "- After adjudication, execute canonical_repair_steps in repair_order order for blockers with should_repair=true.\n"
            "- Do not execute repair steps for blockers with should_repair=false.\n"
            "- Do not use tools to fabricate evidence, but do use the stage4 contract to separate repairable blockers from deferred blockers.\n"
            "Hard repair rules:\n"
            "- Do not execute a repair unless the blocker-family evidence keeps that family plausibly active.\n"
            "- APN repairs require APN evidence. Roaming repairs require roaming-side evidence. Permission and network repairs require matching blocker evidence.\n"
            "- Fast agents may only execute clear, low-branching repairs and may not initiate multi-step repair chains.\n"
            "- Deep agents must confirm decisive preconditions before repair, should use budget to complete supported repair preconditions, and should prefer a repairable subset over blanket transfer when evidence supports local action.\n"
            "- If evidence is incomplete but the missing chain is still local and bounded, prefer defer/partially_repairable over transfer_required while leaving the supported subset repairable.\n"
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
                **self._attribute_prompt_fields(task, "stage4", agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "agent_execution_contract": execution_contract,
                "stage4_repair_precondition_rules": self._stage4_repair_precondition_rules(
                    mode=getattr(agent, "deliberation_mode", "deep")
                ),
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
                "task_metadata": self._llm_visible_task_metadata(raw_instance),
                "normalization_rules": [
                    "Return every blocker_id exactly once",
                    "Set should_repair to a boolean for every blocker row",
                    "Use transfer_reason=null when transfer is not required",
                    "Set decision_policy_version to a short stable string, for example first_pass_v1",
                    "Do not invent blockers that are not in the input per_blocker list",
                    "Execute canonical repair steps only for blockers marked should_repair=true",
                    "Do not execute deferred or transfer-required blocker repairs",
                    "Use partially_repairable when at least one blocker is supported for local repair and at least one blocker remains deferred",
                    "Do not output transfer_required for the whole case if a supported local repair subset exists",
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
        execution_contract = self._build_agent_execution_contract(task, "stage5", agent)
        deliberation_mode = getattr(agent, "deliberation_mode", "deep")
        system_prompt = (
            "You are performing Stage 5: post-repair verification and terminal decision for a telecom MMS troubleshooting case.\n"
            "Your job is to verify the current post-repair telecom state using verification tools, then choose the final structured terminal action.\n"
            + self._attribute_prompt_context_sentence()
            + self._execution_contract_system_rules("stage5")
            + "\n"
            "Do not execute repair tools or perform additional repair mutation.\n"
            "Replay has already been applied before your verification step.\n"
            "You must verify after replay before returning JSON.\n"
            "Minimum rule: use can_send_mms plus blocker-matched verification tools when blockers were repaired or selected, but keep the search compact when the execution contract says to close quickly.\n"
            "Do not produce customer-facing prose.\n"
            "Return JSON only.\n"
            "The only allowed final_action values are: repair_all, repair_subset, transfer.\n"
            "Decision constraints:\n"
            "- repair_all means all blockers must be selected and deferred must be empty.\n"
            "- repair_subset means selected and deferred must form a partition of the input blocker ids.\n"
            "- transfer means selected must be empty and all blockers must be deferred.\n"
            "Hard terminal rules:\n"
            "- Choose the final action from stage4 repairability plus post-repair verification evidence.\n"
            "- transfer is not the default action for incomplete evidence; it requires explicit external/manual handling, stage4 transfer_required, or verified local repair impossibility.\n"
            "- repair_all and repair_subset are normal evidence-supported terminal actions.\n"
            "- Distinguish between true transfer-required cases, cases that still need more local verification, and cases whose evidence chain is already sufficient for closure. Only the first state justifies transfer.\n"
            + self._stage5_system_terminal_rules(mode=deliberation_mode)
            + "\n"
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
                **self._attribute_prompt_fields(task, "stage5", agent),
                "agent_deliberation_profile": self._build_agent_deliberation_profile(agent),
                "agent_execution_contract": execution_contract,
                "stage5_terminal_decision_rules": self._stage5_terminal_decision_rules(
                    mode=deliberation_mode
                ),
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
                "task_metadata": self._llm_visible_task_metadata(raw_instance),
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
                    "When in doubt, use stage4_output.repairability and stage4_output.per_blocker as the default terminal plan",
                    "Do not choose transfer unless stage4_output.repairability is transfer_required, transfer_reason is explicit, or verification proves local repair is inappropriate",
                    "Deferred blockers can prevent repair_all, but if stage4_output already supports a local subset they are not by themselves enough to force transfer; prefer repair_subset unless transfer has a hard reason",
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
        inferred_blocker_ids = self._augment_stage3_inferred_blockers(
            observed_state=observed_state,
            inferred_blocker_ids=inferred_blocker_ids,
        )
        per_blocker = build_per_blocker_from_ids(inferred_blocker_ids)
        return {
            "observed_state": observed_state,
            "per_blocker": per_blocker,
            "per_blocker_mode": "inferred_from_observed_state_v2",
            "raw_task_blocker_ids": self._raw_task_blocker_ids(raw_instance),
            "inferred_blocker_ids": inferred_blocker_ids,
        }

    def _allow_stage4_deep_local_completion(
        self,
        *,
        task: TaskDescriptor,
        agent: AgentSpec,
        raw_instance: dict[str, Any],
        stage3_output: dict[str, Any],
    ) -> bool:
        del task, agent, raw_instance, stage3_output
        # Disabled for telecom_mms_agent_profile_only_clean_v1: completion used
        # route/oracle-like metadata to add repairs after the LLM's raw Stage 4
        # decision, which gives deep/target paths an execution-layer advantage.
        return False

    def _local_repair_precondition_supported(
        self,
        blocker_id: str,
        stage3_output: dict[str, Any],
    ) -> bool:
        observed = dict(stage3_output.get("observed_state", {}) or {})
        if blocker_id == "bad_wifi_calling":
            return bool(observed.get("wifi_calling_enabled")) is True
        if blocker_id == "break_apn_mms_setting":
            return observed.get("apn_mms_ok") is False
        if blocker_id == "break_app_storage_permission":
            return observed.get("messaging_storage_permission") is False
        if blocker_id == "break_app_sms_permission":
            return observed.get("messaging_sms_permission") is False
        if blocker_id == "break_app_both_permissions":
            return (
                observed.get("messaging_sms_permission") is False
                and observed.get("messaging_storage_permission") is False
            )
        return False

    def _local_repair_prerequisite_supported(
        self,
        blocker_id: str,
        stage3_output: dict[str, Any],
    ) -> bool:
        observed = dict(stage3_output.get("observed_state", {}) or {})
        if blocker_id == "airplane_mode_on":
            return observed.get("airplane_mode") is True
        if blocker_id == "unseat_sim_card":
            return str(observed.get("sim_status", "")).strip().lower() == "missing"
        if blocker_id == "user_abroad_roaming_enabled_off":
            return (
                observed.get("is_abroad") is True
                and observed.get("roaming_enabled_on_account") is True
                and observed.get("roaming_enabled_on_device") is False
            )
        if blocker_id == "bad_network_preference":
            return str(observed.get("network_mode_preference", "")).strip().lower() == "2g_only"
        if blocker_id == "data_mode_off":
            if observed.get("mobile_data_working") is not False:
                return False
            if observed.get("airplane_mode") is True:
                return False
            if str(observed.get("sim_status", "")).strip().lower() == "missing":
                return False
            if (
                observed.get("is_abroad") is True
                and observed.get("roaming_enabled_on_account") is True
                and observed.get("roaming_enabled_on_device") is False
            ):
                return False
            if observed.get("data_usage_exceeded") is True:
                return False
            return True
        return False

    @staticmethod
    def _stage4_repair_label_from_row(row: dict[str, Any]) -> str:
        return (
            str(row.get("adjudication_label", "repair_unspecified_blocker"))
            .replace("transfer_", "repair_")
            .replace("defer_", "repair_")
        )

    def _normalize_stage4_output(
        self,
        final_output: dict[str, Any] | None,
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
        allow_deep_local_completion: bool,
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
        tool_errors: list[dict[str, Any]],
        db_hash_before: str | None,
        db_hash_after: str | None,
    ) -> dict[str, Any]:
        normalized_rows, repairability, transfer_reason, stage4_diagnostics = self._normalized_stage4_plan(
            final_output=final_output,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            allow_deep_local_completion=allow_deep_local_completion,
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
            "stage4_raw_json_extracted": deepcopy(stage4_diagnostics["raw_json_extracted"]),
            "stage4_raw_action_hint": stage4_diagnostics["raw_action_hint"],
            "stage4_selected_before_normalization": list(
                stage4_diagnostics["selected_before_normalization"]
            ),
            "stage4_deferred_before_normalization": list(
                stage4_diagnostics["deferred_before_normalization"]
            ),
            "stage4_selected_after_normalization": list(
                stage4_diagnostics["selected_after_normalization"]
            ),
            "stage4_deferred_after_normalization": list(
                stage4_diagnostics["deferred_after_normalization"]
            ),
            "stage4_normalizer_changed_output": bool(
                stage4_diagnostics["normalizer_changed_output"]
            ),
            "stage4_completion_pass_applied": bool(
                stage4_diagnostics["completion_pass_applied"]
            ),
            "stage4_completion_prerequisite_pass_applied": bool(
                stage4_diagnostics["completion_prerequisite_pass_applied"]
            ),
            "stage4_completion_added_prerequisite_blockers": list(
                stage4_diagnostics["completion_added_prerequisite_blockers"]
            ),
            "stage4_completion_added_downstream_blockers": list(
                stage4_diagnostics["completion_added_downstream_blockers"]
            ),
            "stage4_completion_added_blockers": list(
                stage4_diagnostics["completion_added_blockers"]
            ),
            "stage4_completion_blocked_by_hard_transfer_guard": list(
                stage4_diagnostics["completion_blocked_by_hard_transfer_guard"]
            ),
        }

    def _normalized_stage4_plan(
        self,
        *,
        final_output: dict[str, Any] | None,
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
        allow_deep_local_completion: bool,
    ) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
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
        selected_before_normalization = [
            row.get("blocker_id")
            for row in llm_rows
            if isinstance(row, dict)
            and row.get("blocker_id")
            and row.get("should_repair") is True
        ]
        selected_before_normalization = [
            bid for bid in blocker_ids if bid in set(selected_before_normalization)
        ]
        deferred_before_normalization = [
            bid for bid in blocker_ids if bid not in set(selected_before_normalization)
        ]
        stage4_completion_added_prerequisite_blockers: list[str] = []
        stage4_completion_added_downstream_blockers: list[str] = []
        stage4_completion_added_blockers: list[str] = []
        stage4_completion_blocked_by_hard_transfer_guard: list[str] = []

        normalized_rows: list[dict[str, Any]] = []
        for fallback_row in fallback_rows:
            blocker_id = fallback_row["blocker_id"]
            llm_row = llm_row_map.get(blocker_id, {})
            normalized = deepcopy(fallback_row)
            llm_should_repair = llm_row.get("should_repair")
            hard_nonlocal_blocker = self._is_nonlocal_or_hybrid_transfer_blocker(blocker_id)
            if isinstance(llm_should_repair, bool):
                if llm_should_repair and hard_nonlocal_blocker:
                    normalized["should_repair"] = False
                    normalized["oracle_execute_decision"] = str(
                        fallback_row.get("oracle_execute_decision", "transfer")
                    )
                    normalized["adjudication_label"] = fallback_row["adjudication_label"]
                    normalized["refusal_code"] = (
                        fallback_row.get("refusal_code")
                        or "hard_nonlocal_blocker_retains_transfer_constraint_v1"
                    )
                    stage4_completion_blocked_by_hard_transfer_guard.append(blocker_id)
                else:
                    normalized["should_repair"] = llm_should_repair
                    normalized["oracle_execute_decision"] = "repair" if llm_should_repair else "defer"
                    normalized["adjudication_label"] = (
                        fallback_row["adjudication_label"]
                        if llm_should_repair == fallback_row["should_repair"]
                        else (
                            fallback_row["adjudication_label"].replace("transfer_", "repair_").replace("defer_", "repair_")
                            if llm_should_repair
                            else fallback_row["adjudication_label"].replace("repair_", "defer_").replace("transfer_", "defer_")
                        )
                    )
                    normalized["refusal_code"] = None if llm_should_repair else (
                        llm_row.get("transfer_reason")
                        if isinstance(llm_row.get("transfer_reason"), str) and llm_row.get("transfer_reason")
                        else fallback_row.get("refusal_code")
                    )
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

        completion_pass_applied = False
        completion_prerequisite_pass_applied = False
        if allow_deep_local_completion:
            completion_pass_applied = True
            current_selected = {
                row["blocker_id"] for row in normalized_rows if row.get("should_repair") is True
            }

            prerequisite_changed = True
            while prerequisite_changed:
                prerequisite_changed = False
                for row in normalized_rows:
                    blocker_id = row["blocker_id"]
                    if row.get("should_repair") is True:
                        current_selected.add(blocker_id)
                        continue
                    if self._is_nonlocal_or_hybrid_transfer_blocker(blocker_id):
                        if blocker_id not in stage4_completion_blocked_by_hard_transfer_guard:
                            stage4_completion_blocked_by_hard_transfer_guard.append(blocker_id)
                        continue
                    if not self._local_repair_prerequisite_supported(blocker_id, stage3_output):
                        continue
                    depends_on = [
                        dep for dep in row.get("depends_on", []) if dep in set(blocker_ids)
                    ]
                    if any(dep not in current_selected for dep in depends_on):
                        continue
                    row["should_repair"] = True
                    row["oracle_execute_decision"] = "repair"
                    row["adjudication_label"] = self._stage4_repair_label_from_row(row)
                    row["refusal_code"] = None
                    current_selected.add(blocker_id)
                    stage4_completion_added_prerequisite_blockers.append(blocker_id)
                    stage4_completion_added_blockers.append(blocker_id)
                    completion_prerequisite_pass_applied = True
                    prerequisite_changed = True

            downstream_changed = True
            while downstream_changed:
                downstream_changed = False
                for row in normalized_rows:
                    blocker_id = row["blocker_id"]
                    if row.get("should_repair") is True:
                        current_selected.add(blocker_id)
                        continue
                    if self._is_nonlocal_or_hybrid_transfer_blocker(blocker_id):
                        if blocker_id not in stage4_completion_blocked_by_hard_transfer_guard:
                            stage4_completion_blocked_by_hard_transfer_guard.append(blocker_id)
                        continue
                    if not self._local_repair_precondition_supported(blocker_id, stage3_output):
                        continue
                    depends_on = [
                        dep for dep in row.get("depends_on", []) if dep in set(blocker_ids)
                    ]
                    if any(dep not in current_selected for dep in depends_on):
                        continue
                    row["should_repair"] = True
                    row["oracle_execute_decision"] = "repair"
                    row["adjudication_label"] = self._stage4_repair_label_from_row(row)
                    row["refusal_code"] = None
                    current_selected.add(blocker_id)
                    stage4_completion_added_downstream_blockers.append(blocker_id)
                    stage4_completion_added_blockers.append(blocker_id)
                    downstream_changed = True

        selected_blocker_ids = [
            row["blocker_id"] for row in normalized_rows if row.get("should_repair") is True
        ]
        deferred_blocker_ids = [
            row["blocker_id"] for row in normalized_rows if row.get("blocker_id") not in selected_blocker_ids
        ]
        shallow_subset_requires_transfer = self._is_shallow_subset_with_hard_deferred(
            selected_blocker_ids, deferred_blocker_ids
        )
        if shallow_subset_requires_transfer:
            transfer_reason = "shallow_subset_defers_nonlocal_blocker_v2"
            self._coerce_stage4_rows_to_transfer_required(
                normalized_rows,
                refusal_code=transfer_reason,
            )
            selected_blocker_ids = []
            deferred_blocker_ids = [
                row["blocker_id"]
                for row in normalized_rows
                if row.get("blocker_id")
            ]
            repairability = "transfer_required"
            return normalized_rows, repairability, transfer_reason, {
                "raw_json_extracted": deepcopy(final_output) if isinstance(final_output, dict) else None,
                "raw_action_hint": (
                    final_output.get("repairability")
                    if isinstance(final_output, dict)
                    else None
                ),
                "selected_before_normalization": selected_before_normalization,
                "deferred_before_normalization": deferred_before_normalization,
                "selected_after_normalization": [],
                "deferred_after_normalization": deferred_blocker_ids,
                "normalizer_changed_output": (
                    selected_before_normalization != []
                    or deferred_before_normalization != deferred_blocker_ids
                ),
                "completion_pass_applied": completion_pass_applied,
                "completion_prerequisite_pass_applied": completion_prerequisite_pass_applied,
                "completion_added_prerequisite_blockers": stage4_completion_added_prerequisite_blockers,
                "completion_added_downstream_blockers": stage4_completion_added_downstream_blockers,
                "completion_added_blockers": stage4_completion_added_blockers,
                "completion_blocked_by_hard_transfer_guard": list(
                    dict.fromkeys(stage4_completion_blocked_by_hard_transfer_guard)
                ),
            }
        if selected_blocker_ids and deferred_blocker_ids:
            repairability = "partially_repairable"
        elif selected_blocker_ids:
            repairability = "repairable"
        else:
            repairability = "transfer_required"

        transfer_reason = None
        if repairability == "transfer_required":
            llm_transfer_reason = (
                final_output.get("transfer_reason")
                if isinstance(final_output, dict) and isinstance(final_output.get("transfer_reason"), str)
                else None
            )
            transfer_reason = (
                llm_transfer_reason
                or decision.get("transfer_reason")
                or "no_safe_local_repair_subset_v2"
            )
        selected_after_normalization = list(selected_blocker_ids)
        deferred_after_normalization = list(deferred_blocker_ids)
        raw_repairability = (
            str(final_output.get("repairability", "")).strip().lower()
            if isinstance(final_output, dict)
            else None
        ) or None
        normalizer_changed_output = (
            selected_before_normalization != selected_after_normalization
            or deferred_before_normalization != deferred_after_normalization
            or (raw_repairability is not None and raw_repairability != repairability)
        )
        return normalized_rows, repairability, transfer_reason, {
            "raw_json_extracted": deepcopy(final_output) if isinstance(final_output, dict) else None,
            "raw_action_hint": (
                final_output.get("repairability")
                if isinstance(final_output, dict)
                else None
            ),
            "selected_before_normalization": selected_before_normalization,
            "deferred_before_normalization": deferred_before_normalization,
            "selected_after_normalization": selected_after_normalization,
            "deferred_after_normalization": deferred_after_normalization,
            "normalizer_changed_output": normalizer_changed_output,
            "completion_pass_applied": completion_pass_applied,
            "completion_prerequisite_pass_applied": completion_prerequisite_pass_applied,
            "completion_added_prerequisite_blockers": stage4_completion_added_prerequisite_blockers,
            "completion_added_downstream_blockers": stage4_completion_added_downstream_blockers,
            "completion_added_blockers": stage4_completion_added_blockers,
            "completion_blocked_by_hard_transfer_guard": list(
                dict.fromkeys(stage4_completion_blocked_by_hard_transfer_guard)
            ),
        }

    def _execute_stage4_canonical_plan(
        self,
        *,
        raw_instance: dict[str, Any],
        stage2_output: dict[str, Any],
        stage3_output: dict[str, Any],
        final_output: dict[str, Any] | None,
        allow_deep_local_completion: bool,
    ) -> dict[str, Any]:
        normalized_rows, _repairability, _transfer_reason, _stage4_diagnostics = self._normalized_stage4_plan(
            final_output=final_output,
            stage2_output=stage2_output,
            stage3_output=stage3_output,
            allow_deep_local_completion=allow_deep_local_completion,
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
        stage4_repairability = str(stage4_output.get("repairability", "")).strip().lower()
        stage4_transfer_reason = self._normalize_optional_short_text(stage4_output.get("transfer_reason"))
        stage4_selected = [
            row.get("blocker_id")
            for row in stage4_output.get("per_blocker", [])
            if isinstance(row, dict)
            and row.get("blocker_id") in blocker_id_set
            and bool(row.get("should_repair"))
        ]
        stage4_selected = [bid for bid in blocker_ids if bid in set(stage4_selected)]
        stage4_deferred = [bid for bid in blocker_ids if bid not in set(stage4_selected)]
        verification_summary = self._stage5_verification_summary(
            raw_instance=raw_instance,
            stage2_output=stage2_output,
            executed_tool_calls=executed_tool_calls,
            tool_results=tool_results,
        )
        verification_evidence = self._normalize_str_list(
            verification_summary.get("verification_evidence")
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

        def stage4_inferred_terminal_action() -> tuple[str, list[str], list[str]]:
            if stage4_repairability == "transfer_required":
                return "transfer", [], list(blocker_ids)
            if stage4_selected:
                if set(stage4_selected) == blocker_id_set:
                    return "repair_all", list(blocker_ids), []
                return "repair_subset", list(stage4_selected), list(stage4_deferred)
            if blocker_ids:
                return "transfer", [], list(blocker_ids)
            return "repair_all", [], []

        def normalized_stage4_subset_semantics(
            selected_candidate: list[str],
            deferred_candidate: list[str],
        ) -> tuple[list[str], list[str], bool]:
            if not stage4_selected:
                selected_normalized = [bid for bid in blocker_ids if bid in set(selected_candidate)]
                deferred_normalized = [bid for bid in blocker_ids if bid not in set(selected_normalized)]
                return selected_normalized, deferred_normalized, bool(selected_normalized)

            selected_stage4_set = set(stage4_selected)
            deferred_stage4_set = set(stage4_deferred)
            selected_set = set(selected_candidate)
            deferred_set = set(deferred_candidate)

            # Preserve Stage 4 subset semantics: selected blockers should default to the
            # Stage 4 repair subset and must never flip into the Stage 4 deferred side.
            if not selected_set and deferred_set:
                selected_set = blocker_id_set - deferred_set
            if not selected_set:
                return list(stage4_selected), list(stage4_deferred), False
            if selected_set - selected_stage4_set:
                return list(stage4_selected), list(stage4_deferred), False
            if deferred_set & selected_stage4_set and not selected_set:
                return list(stage4_selected), list(stage4_deferred), False
            selected_normalized = [bid for bid in stage4_selected if bid in selected_set]
            if not selected_normalized:
                return list(stage4_selected), list(stage4_deferred), False
            deferred_normalized = [
                bid for bid in blocker_ids if bid not in set(selected_normalized)
            ]
            if selected_stage4_set & set(deferred_normalized) and set(selected_normalized) == deferred_stage4_set:
                return list(stage4_selected), list(stage4_deferred), False
            return selected_normalized, deferred_normalized, True

        def hard_transfer_reason(reason: str | None) -> bool:
            normalized_reason = self._normalize_optional_short_text(reason)
            if normalized_reason is None:
                return False
            normalized_reason = normalized_reason.lower()

            exact_hard_reasons = {
                "hybrid_blocker_requires_transfer_v1",
                "external_manual_handling_required",
                "manual_handling_required",
                "external_escalation_required",
                "account_side_action_required",
                "account_side_constraint_requires_transfer",
                "policy_side_constraint_requires_transfer",
                "non_local_constraint_requires_transfer",
                "verified_local_repair_impossible",
                "verification_proved_local_repair_inappropriate",
                "verified_repair_path_failed",
                "repair_failed_cannot_continue_safely",
            }
            if normalized_reason in exact_hard_reasons:
                return True

            # In the narrow partially_repairable subset case, reasons that only say
            # deferred/high-complexity/remaining blockers still exist should not override
            # the stage4-supported repair_subset plan.
            soft_reason_groups = [
                ("deferred",),
                ("remaining",),
                ("remain",),
                ("unresolved",),
                ("not_all",),
                ("deep_validation",),
                ("verification_heavy",),
                ("high", "complexity"),
                ("complex",),
                ("incomplete",),
                ("uncertain",),
                ("ambiguous",),
                ("pending",),
                ("more", "verification"),
                ("needs", "verification"),
                ("evidence", "incomplete"),
                ("not", "verified"),
            ]
            if (
                stage4_repairability == "partially_repairable"
                and stage4_selected
                and any(
                    all(token in normalized_reason for token in group)
                    for group in soft_reason_groups
                )
            ):
                return False

            hard_reason_groups = [
                ("manual",),
                ("external",),
                ("account_side",),
                ("policy_side",),
                ("non_local",),
                ("replay", "failed"),
                ("repair", "failed"),
                ("verification", "proved"),
                ("verified", "impossible"),
                ("verified", "inappropriate"),
                ("cannot", "continue", "safely"),
                ("unsafe", "continue"),
            ]
            return any(
                all(token in normalized_reason for token in group)
                for group in hard_reason_groups
            )

        if raw_action not in {"repair_all", "repair_subset", "transfer"}:
            if selected_clean and deferred_clean:
                final_action = "repair_subset"
            elif selected_clean and not deferred_clean:
                final_action = "repair_all" if set(selected_clean) == blocker_id_set else "repair_subset"
            else:
                final_action, selected_clean, deferred_clean = stage4_inferred_terminal_action()
        else:
            final_action = raw_action

        if final_action == "repair_subset":
            (
                selected_clean,
                deferred_clean,
                subset_semantics_valid,
            ) = normalized_stage4_subset_semantics(selected_clean, deferred_clean)
            if not subset_semantics_valid and stage4_selected:
                final_action, selected_clean, deferred_clean = stage4_inferred_terminal_action()

        if (
            final_action == "repair_subset"
            and stage4_selected
            and set(selected_clean) != set(stage4_selected)
            and not self._stage5_verification_floor_met(
                selected_blocker_ids=selected_clean or stage4_selected,
                deferred_blocker_ids=(
                    [bid for bid in blocker_ids if bid not in set(selected_clean)]
                    if selected_clean
                    else stage4_deferred
                ),
                verification_evidence=verification_evidence,
            )
        ):
            final_action, selected_clean, deferred_clean = stage4_inferred_terminal_action()

        if final_action == "repair_all":
            selected_blocker_ids = list(blocker_ids)
            deferred_blocker_ids: list[str] = []
        elif final_action == "transfer":
            explicit_transfer_reason = self._normalize_optional_short_text(data.get("transfer_reason"))
            explicit_transfer_reason_is_hard = hard_transfer_reason(explicit_transfer_reason)
            if (
                stage4_repairability == "partially_repairable"
                and stage4_selected
                and not explicit_transfer_reason_is_hard
            ):
                final_action, selected_blocker_ids, deferred_blocker_ids = stage4_inferred_terminal_action()
            elif (
                stage4_repairability in {"repairable", "partially_repairable"}
                and explicit_transfer_reason is None
                and stage4_transfer_reason is None
            ):
                final_action, selected_blocker_ids, deferred_blocker_ids = stage4_inferred_terminal_action()
            else:
                selected_blocker_ids = []
                deferred_blocker_ids = list(blocker_ids)
        else:
            if stage4_selected and not selected_clean:
                selected_clean = list(stage4_selected)
                deferred_clean = list(stage4_deferred)
            elif not selected_clean and deferred_clean:
                selected_clean = [bid for bid in blocker_ids if bid not in deferred_clean]
            if not selected_clean:
                final_action, selected_blocker_ids, deferred_blocker_ids = stage4_inferred_terminal_action()
            else:
                selected_blocker_ids = [bid for bid in blocker_ids if bid in set(selected_clean)]
                deferred_blocker_ids = [bid for bid in blocker_ids if bid not in set(selected_blocker_ids)]
                if not deferred_blocker_ids:
                    final_action = "repair_all"
                    selected_blocker_ids = list(blocker_ids)
                elif not selected_blocker_ids:
                    final_action = "transfer"
                    deferred_blocker_ids = list(blocker_ids)

        forced_transfer_reason: str | None = None
        if (
            final_action == "repair_subset"
            and not self._stage5_verification_floor_met(
                selected_blocker_ids=selected_blocker_ids,
                deferred_blocker_ids=deferred_blocker_ids,
                verification_evidence=verification_evidence,
            )
        ):
            final_action = "transfer"
            selected_blocker_ids = []
            deferred_blocker_ids = list(blocker_ids)
            forced_transfer_reason = "shallow_subset_verification_floor_not_met_v1"

        if final_action == "transfer":
            transfer_reason = self._normalize_optional_short_text(data.get("transfer_reason"))
            if forced_transfer_reason is not None:
                transfer_reason = forced_transfer_reason
            elif transfer_reason is None:
                transfer_reason = stage4_transfer_reason
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
            ("check_network_status", {}),
            ("check_sim_status", {}),
            ("check_network_mode_preference", {}),
            ("check_apn_settings", {}),
            ("check_wifi_calling_status", {}),
            ("check_app_permissions", {"app_name": "messaging"}),
            ("run_speed_test", {}),
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
        raw_sim_status = str(sim_status or network_lines.get("SIM Card Status", "") or "")
        sim_status_value = self._normalize_sim_status(raw_sim_status)
        mobile_data_enabled = self._parse_yes_no_flag(network_lines.get("Mobile Data Enabled"))

        return {
            "can_send_mms": "cannot" not in str(can_send_mms or "").lower(),
            "service_status": network_lines.get("Cellular Connection", "unknown"),
            "mobile_data_working": mobile_data_working,
            "mobile_data_enabled": mobile_data_enabled,
            "internet_speed_desc": internet_speed_desc,
            "is_abroad": "abroad" in known_info_text.lower(),
            "roaming_enabled_on_device": network_lines.get("Data Roaming Enabled", "No") == "Yes",
            "roaming_enabled_on_account": bool(line_details.get("roaming_enabled")),
            "airplane_mode": network_lines.get("Airplane Mode", "OFF") == "ON",
            "sim_status": sim_status_value,
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
            "mobile_data_enabled",
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

    def _augment_stage3_inferred_blockers(
        self,
        *,
        observed_state: dict[str, Any],
        inferred_blocker_ids: list[str],
    ) -> list[str]:
        augmented = list(dict.fromkeys(self._normalize_str_list(inferred_blocker_ids)))

        if (
            observed_state.get("wifi_calling_enabled") is False
            and "bad_wifi_calling" not in augmented
        ):
            augmented.append("bad_wifi_calling")

        if (
            observed_state.get("mobile_data_enabled") is False
            and observed_state.get("mobile_data_working") is False
            and "data_mode_off" not in augmented
        ):
            augmented.append("data_mode_off")

        return augmented

    def _parse_yes_no_flag(self, value: Any) -> bool | None:
        if isinstance(value, bool):
            return value
        if not isinstance(value, str):
            return None
        lowered = value.strip().lower()
        if lowered == "yes":
            return True
        if lowered == "no":
            return False
        return None

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

    def _stage1_prompt_summary(self, agent: AgentSpec, task: TaskDescriptor | None = None) -> str:
        return (
            f"telecom stage1 user grounding; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"mode={getattr(agent, 'deliberation_mode', 'deep')}; "
            f"max_rounds={self._max_rounds(agent, task, 'stage1')}"
        )

    def _stage2_prompt_summary(self, agent: AgentSpec, task: TaskDescriptor | None = None) -> str:
        return (
            f"telecom stage2 resolution; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"mode={getattr(agent, 'deliberation_mode', 'deep')}; "
            f"max_rounds={self._max_rounds(agent, task, 'stage2')}"
        )

    def _stage3_prompt_summary(self, agent: AgentSpec, task: TaskDescriptor | None = None) -> str:
        return (
            f"telecom stage3 observed-state extraction; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"mode={getattr(agent, 'deliberation_mode', 'deep')}; "
            f"max_rounds={self._max_rounds(agent, task, 'stage3')}"
        )

    def _stage4_prompt_summary(self, agent: AgentSpec, task: TaskDescriptor | None = None) -> str:
        return (
            f"telecom stage4 repair execution; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"mode={getattr(agent, 'deliberation_mode', 'deep')}; "
            f"max_rounds={self._max_rounds(agent, task, 'stage4')}"
        )

    def _stage5_prompt_summary(self, agent: AgentSpec, task: TaskDescriptor | None = None) -> str:
        return (
            f"telecom stage5 verification and terminal decision; competence={agent.competence_level}; "
            f"scope={agent.scope_level}; stability={agent.stability_level}; "
            f"mode={getattr(agent, 'deliberation_mode', 'deep')}; "
            f"max_rounds={self._max_rounds(agent, task, 'stage5')}"
        )
