"""LLM-backed executor for telecom Stage 2/3 real tool use."""

from __future__ import annotations

import json
import os
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from telecom_mms_specs import build_per_blocker_from_ids, infer_blocker_ids_from_observed_state
from .telecom_bench_backed_executor import TelecomBenchBackedExecutor
from tree_family.specs import AgentSpec, TaskDescriptor


DEFAULT_LLM_MODEL = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4.1-2025-04-14")


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
            max_rounds=self._max_rounds(agent),
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
            max_rounds=self._max_rounds(agent),
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

    def _max_rounds(self, agent: AgentSpec) -> int:
        rounds = 4 if agent.competence_level == "high" else 3
        if agent.scope_level == "broad":
            rounds += 1
        if agent.stability_level == "unstable":
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
        return " ".join([competence, scope, stability])

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
                "stage_goal": "Resolve the customer and telecom line only.",
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
            "You are performing Stage 3: observed-state extraction for a telecom MMS troubleshooting case.\n"
            "Goal: collect factual observed state only. Do not decide terminal actions.\n"
            "Use only the allowed tools. Prefer explicit tool evidence over guesses.\n"
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
                "stage_goal": "Produce only factual observed state for the resolved telecom line.",
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
        resolved_line = next(
            (line for line in line_results if line.get("phone_number") == target_phone_number),
            line_results[0] if line_results else {},
        )
        plan = plan_results[0] if plan_results else {}
        return {
            "candidate_customers": self._normalize_str_list(candidate_customers),
            "resolved_customer_id": data.get("resolved_customer_id") or customer_result.get("customer_id"),
            "candidate_line_ids": self._normalize_str_list(candidate_line_ids),
            "resolved_line_id": data.get("resolved_line_id") or resolved_line.get("line_id"),
            "target_phone_number": target_phone_number,
            "assistant_account_snapshot": {
                "line_status": str(
                    data.get("assistant_account_snapshot", {}).get("line_status")
                    or resolved_line.get("status")
                    or ""
                ).lower()
                or None,
                "roaming_enabled_on_account": self._coalesce_bool(
                    data.get("assistant_account_snapshot", {}).get("roaming_enabled_on_account"),
                    resolved_line.get("roaming_enabled"),
                ),
                "plan_id": data.get("assistant_account_snapshot", {}).get("plan_id") or resolved_line.get("plan_id"),
                "data_used_gb": self._coalesce_number(
                    data.get("assistant_account_snapshot", {}).get("data_used_gb"),
                    resolved_line.get("data_used_gb"),
                ),
                "data_limit_gb": self._coalesce_number(
                    data.get("assistant_account_snapshot", {}).get("data_limit_gb"),
                    plan.get("data_limit_gb"),
                ),
            },
            "resolution_status": data.get("resolution_status") or ("resolved" if resolved_line else "unresolved"),
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
        line_details = next(
            (
                content
                for name, args, content in tool_map
                if name == "get_details_by_id" and isinstance(content, dict) and args.get("id") == stage2_output.get("resolved_line_id")
            ),
            {},
        )
        plan_details = next(
            (
                content
                for name, args, content in tool_map
                if name == "get_details_by_id"
                and isinstance(content, dict)
                and args.get("id") == stage2_output.get("assistant_account_snapshot", {}).get("plan_id")
            ),
            {},
        )

        observed_from_tools = self._normalize_observed_state(
            known_info=raw_instance.get("user_context", {}).get("known_info", ""),
            network_status=response_lookup.get("check_network_status"),
            sim_status=response_lookup.get("check_sim_status"),
            mode_status=response_lookup.get("check_network_mode_preference"),
            apn_status=response_lookup.get("check_apn_settings"),
            wifi_calling_status=response_lookup.get("check_wifi_calling_status"),
            app_permissions=response_lookup.get("check_app_permissions"),
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

    def _merge_observed_state_tool_first(
        self,
        observed_from_tools: dict[str, Any],
        observed_seed: dict[str, Any],
    ) -> dict[str, Any]:
        merged = dict(observed_from_tools)
        for key, value in observed_seed.items():
            if value is None:
                continue
            if key not in merged or merged.get(key) is None:
                merged[key] = value
        return merged

    def _zip_tool_results(
        self,
        executed_tool_calls: list[dict[str, Any]],
        tool_results: list[Any],
    ) -> list[tuple[str, dict[str, Any], Any]]:
        rows: list[tuple[str, dict[str, Any], Any]] = []
        for call, content in zip(executed_tool_calls, tool_results):
            rows.append((str(call.get("name", "")), dict(call.get("arguments", {})), content))
        return rows

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
