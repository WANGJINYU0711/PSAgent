"""Bench-backed executor for telecom MMS Stage 2/3 real-tool integration."""

from __future__ import annotations

import json
import re
import subprocess
from copy import deepcopy
from pathlib import Path
from typing import Any

from .base_executor import BaseExecutor
from .simulated_executor import SimulatedExecutor
from telecom_mms_specs import (
    blocker_diagnostic_evidence,
    build_per_blocker_from_ids,
    first_pass_terminal_decision,
    get_blocker_spec,
    infer_blocker_ids_from_observed_state,
    materialize_repair_steps,
)
from tree_family.specs import AgentSpec, TaskDescriptor


PHONE_RE = re.compile(r"(\d{3}-\d{3}-\d{4})")


class TelecomBenchBackedExecutor(BaseExecutor):
    def __init__(self, stages: list[str], seed: int = 0) -> None:
        super().__init__(stages=stages, seed=seed)
        self.score_helper = SimulatedExecutor(stages=stages, seed=seed)
        self.root = Path(__file__).resolve().parents[2]
        self.venv_python = self.root / "tau2-bench" / ".venv" / "bin" / "python"
        self.bridge_script = Path(__file__).with_name("_telecom_bench_tool_bridge.py")

    def run_path(
        self,
        task: TaskDescriptor,
        path: list[str],
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        stage_trace: list[dict[str, Any]] = []
        stage_outputs: dict[str, dict[str, Any]] = {}
        tool_calls_made = 0
        any_tool_error = False
        db_hash_before: str | None = None
        db_hash_after: str | None = None

        stage1_output = self._run_stage1(task, path[0], agent_map, raw_instance)
        stage_outputs["stage1"] = stage1_output
        stage_trace.append(stage1_output["trace"])

        stage2_output = self._run_stage2(task, path[1], agent_map, raw_instance, stage1_output["output"])
        stage_outputs["stage2"] = stage2_output
        stage_trace.append(stage2_output["trace"])
        tool_calls_made += len(stage2_output["trace"]["executed_tool_calls"])
        any_tool_error = any_tool_error or bool(stage2_output["trace"]["tool_errors"])
        db_hash_before = stage2_output["trace"].get("db_hash_before")
        db_hash_after = stage2_output["trace"].get("db_hash_after")

        stage3_output = self._run_stage3(
            task,
            path[2],
            agent_map,
            raw_instance,
            stage1_output["output"],
            stage2_output["output"],
        )
        stage_outputs["stage3"] = stage3_output
        stage_trace.append(stage3_output["trace"])
        tool_calls_made += len(stage3_output["trace"]["executed_tool_calls"])
        any_tool_error = any_tool_error or bool(stage3_output["trace"]["tool_errors"])
        if db_hash_before is None:
            db_hash_before = stage3_output["trace"].get("db_hash_before")
        db_hash_after = stage3_output["trace"].get("db_hash_after", db_hash_after)

        stage4_output = self._run_stage4(
            task,
            path[3],
            agent_map,
            raw_instance,
            stage1_output["output"],
            stage2_output["output"],
            stage3_output["output"],
        )
        stage_outputs["stage4"] = stage4_output
        stage_trace.append(stage4_output["trace"])

        stage5_output = self._run_stage5(
            task,
            path[4],
            agent_map,
            raw_instance,
            stage1_output["output"],
            stage2_output["output"],
            stage3_output["output"],
            stage4_output["output"],
        )
        stage_outputs["stage5"] = stage5_output
        stage_trace.append(stage5_output["trace"])

        leaf_type = "unshared" if any(agent_map[agent_id].g == 1 for agent_id in path) else "shared"
        path_agent_cost = sum(agent_map[agent_id].base_cost for agent_id in path)
        bench_aux_eval = {
            "db_hash_before": db_hash_before,
            "db_hash_after": db_hash_after,
            "tool_calls_made": tool_calls_made,
            "mutating_tool_calls_made": 0,
            "any_tool_error_occurred": any_tool_error,
            "bench_success": "deferred",
            "bench_db_check": db_hash_before == db_hash_after if db_hash_before and db_hash_after else "deferred",
            "bench_action_check": "deferred",
            "bench_communicate_check": "deferred",
            "bench_nl_assertions": "deferred",
        }

        final_output = stage5_output["output"]
        return {
            "final_action": final_output["final_action"],
            "cancelled_reservation_ids": final_output.get("cancelled_reservation_ids", []),
            "refused_reservation_ids": final_output.get("refused_reservation_ids", []),
            "selected_blocker_ids": final_output.get("selected_blocker_ids", []),
            "deferred_blocker_ids": final_output.get("deferred_blocker_ids", []),
            "stage_trace": stage_trace,
            "stage_outputs": stage_outputs,
            "path_agent_cost": path_agent_cost,
            "leaf_type": leaf_type,
            "bench_aux_eval": bench_aux_eval,
        }

    def _run_stage1(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        del task
        agent = agent_map[agent_id]
        output = deepcopy(raw_instance.get("stage1", {}).get("oracle_output", {}))
        return {
            "input": deepcopy(raw_instance.get("user_context", {})),
            "output": output,
            "trace": {
                "stage_name": "stage1",
                "agent_id": agent_id,
                "agent_g": agent.g,
                "planned_tool_calls": [],
                "executed_tool_calls": [],
                "tool_results": [],
                "tool_errors": [],
                "input": deepcopy(raw_instance.get("user_context", {})),
                "output": deepcopy(output),
                "score": 1.0,
                "source": "oracle_like",
            },
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
        phone_number = (
            stage1_output.get("line_selector", {}).get("value")
            or self._extract_phone(raw_instance.get("user_context", {}).get("known_info", ""))
        )
        tool_calls = [
            self._tool_call("stage2_customer", "get_customer_by_phone", {"phone_number": phone_number}, requestor="assistant"),
        ]
        bridge_result = self._run_bench_tool_calls(raw_instance, tool_calls)
        customer_result = bridge_result["responses"][0]["content"] if bridge_result["responses"] else {}
        candidate_line_ids = list((customer_result or {}).get("line_ids", []) or [])

        detail_calls = [
            self._tool_call(f"stage2_line_{idx}", "get_details_by_id", {"id": line_id}, requestor="assistant")
            for idx, line_id in enumerate(candidate_line_ids)
        ]
        detail_result = self._run_bench_tool_calls(raw_instance, detail_calls)
        line_details = [row["content"] for row in detail_result["responses"] if not row["error"]]
        resolved_line = next(
            (row for row in line_details if row.get("phone_number") == phone_number),
            None,
        )
        if resolved_line is None and line_details:
            resolved_line = line_details[0]
        plan_result = {}
        plan_calls: list[dict[str, Any]] = []
        if resolved_line is not None:
            plan_calls = [
                self._tool_call(
                    "stage2_plan",
                    "get_details_by_id",
                    {"id": resolved_line["plan_id"]},
                    requestor="assistant",
                )
            ]
            plan_bridge = self._run_bench_tool_calls(raw_instance, plan_calls)
            if plan_bridge["responses"] and not plan_bridge["responses"][0]["error"]:
                plan_result = plan_bridge["responses"][0]["content"]
            if bridge_result.get("db_hash_before") is None:
                bridge_result["db_hash_before"] = plan_bridge.get("db_hash_before")
            detail_result["db_hash_after"] = plan_bridge.get("db_hash_after", detail_result.get("db_hash_after"))

        output = {
            "candidate_customers": [customer_result.get("customer_id")] if customer_result else [],
            "resolved_customer_id": customer_result.get("customer_id"),
            "candidate_line_ids": candidate_line_ids,
            "resolved_line_id": resolved_line.get("line_id") if resolved_line else None,
            "target_phone_number": phone_number,
            "assistant_account_snapshot": {
                "line_status": str(resolved_line.get("status", "")).lower() if resolved_line else None,
                "roaming_enabled_on_account": resolved_line.get("roaming_enabled") if resolved_line else None,
                "plan_id": resolved_line.get("plan_id") if resolved_line else None,
                "data_used_gb": resolved_line.get("data_used_gb") if resolved_line else None,
                "data_limit_gb": plan_result.get("data_limit_gb"),
            },
            "resolution_status": "resolved" if resolved_line else "unresolved",
        }
        trace = {
            "stage_name": "stage2",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "planned_tool_calls": tool_calls + detail_calls + plan_calls,
            "executed_tool_calls": [row["tool_call"] for row in bridge_result["responses"]]
            + [row["tool_call"] for row in detail_result["responses"]]
            + ([plan_calls[0]] if plan_calls else []),
            "tool_results": [row["content"] for row in bridge_result["responses"]]
            + [row["content"] for row in detail_result["responses"]]
            + ([plan_result] if plan_calls else []),
            "tool_errors": [row for row in bridge_result["responses"] if row["error"]]
            + [row for row in detail_result["responses"] if row["error"]],
            "db_hash_before": bridge_result.get("db_hash_before"),
            "db_hash_after": detail_result.get("db_hash_after", bridge_result.get("db_hash_after")),
            "input": deepcopy(stage1_output),
            "output": deepcopy(output),
            "score": round(self.score_helper._effective_score(task, "stage2", agent), 4),
            "source": "bench_backed",
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
        line_id = stage2_output.get("resolved_line_id")
        plan_id = stage2_output.get("assistant_account_snapshot", {}).get("plan_id")

        tool_calls = [
            self._tool_call("stage3_network", "check_network_status", {}, requestor="user"),
            self._tool_call("stage3_sim", "check_sim_status", {}, requestor="user"),
            self._tool_call("stage3_mode", "check_network_mode_preference", {}, requestor="user"),
            self._tool_call("stage3_apn", "check_apn_settings", {}, requestor="user"),
            self._tool_call("stage3_wifi", "check_wifi_calling_status", {}, requestor="user"),
            self._tool_call("stage3_perm", "check_app_permissions", {"app_name": "messaging"}, requestor="user"),
            self._tool_call("stage3_speed", "run_speed_test", {}, requestor="user"),
            self._tool_call("stage3_mms", "can_send_mms", {}, requestor="user"),
        ]
        if line_id:
            tool_calls.append(
                self._tool_call("stage3_line", "get_details_by_id", {"id": line_id}, requestor="assistant")
            )
        if plan_id:
            tool_calls.append(
                self._tool_call("stage3_plan", "get_details_by_id", {"id": plan_id}, requestor="assistant")
            )
        bridge_result = self._run_bench_tool_calls(raw_instance, tool_calls)
        response_map = {row["tool_call"]["id"]: row for row in bridge_result["responses"]}

        observed_state = self._normalize_observed_state(
            known_info=raw_instance.get("user_context", {}).get("known_info", ""),
            network_status=response_map.get("stage3_network", {}).get("content"),
            sim_status=response_map.get("stage3_sim", {}).get("content"),
            mode_status=response_map.get("stage3_mode", {}).get("content"),
            apn_status=response_map.get("stage3_apn", {}).get("content"),
            wifi_calling_status=response_map.get("stage3_wifi", {}).get("content"),
            app_permissions=response_map.get("stage3_perm", {}).get("content"),
            speed_test=response_map.get("stage3_speed", {}).get("content"),
            can_send_mms=response_map.get("stage3_mms", {}).get("content"),
            line_details=response_map.get("stage3_line", {}).get("content"),
            plan_details=response_map.get("stage3_plan", {}).get("content"),
        )
        inferred_blocker_ids = infer_blocker_ids_from_observed_state(observed_state)
        per_blocker = build_per_blocker_from_ids(inferred_blocker_ids)
        raw_task_blocker_ids = self._raw_task_blocker_ids(raw_instance)
        output = {"observed_state": observed_state, "per_blocker": per_blocker}
        trace = {
            "stage_name": "stage3",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "planned_tool_calls": tool_calls,
            "executed_tool_calls": [row["tool_call"] for row in bridge_result["responses"]],
            "tool_results": [row["content"] for row in bridge_result["responses"]],
            "tool_errors": [row for row in bridge_result["responses"] if row["error"]],
            "db_hash_before": bridge_result.get("db_hash_before"),
            "db_hash_after": bridge_result.get("db_hash_after"),
            "input": {
                "stage1_output": deepcopy(stage1_output),
                "stage2_output": deepcopy(stage2_output),
            },
            "output": deepcopy(output),
            "score": round(self.score_helper._effective_score(task, "stage3", agent), 4),
            "source": "bench_backed",
            "per_blocker_mode": "inferred_from_observed_state_v2",
            "raw_task_blocker_ids": raw_task_blocker_ids,
            "inferred_blocker_ids": inferred_blocker_ids,
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
        del stage1_output, stage2_output
        agent = agent_map[agent_id]
        output = deepcopy(raw_instance.get("stage4", {}).get("oracle_output", {}))
        return {
            "input": deepcopy(stage3_output),
            "output": output,
            "trace": {
                "stage_name": "stage4",
                "agent_id": agent_id,
                "agent_g": agent.g,
                "planned_tool_calls": [],
                "executed_tool_calls": [],
                "tool_results": [],
                "tool_errors": [],
                "input": deepcopy(stage3_output),
                "output": deepcopy(output),
                "score": round(self.score_helper._effective_score(task, "stage4", agent), 4),
                "source": "oracle_like",
            },
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
        stage4_output: dict[str, Any],
    ) -> dict[str, Any]:
        del raw_instance, stage1_output, stage3_output
        agent = agent_map[agent_id]
        output = self._build_stage5_output(stage4_output)
        return {
            "input": deepcopy(stage4_output),
            "output": output,
            "trace": {
                "stage_name": "stage5",
                "agent_id": agent_id,
                "agent_g": agent.g,
                "planned_tool_calls": [],
                "executed_tool_calls": [],
                "tool_results": [],
                "tool_errors": [],
                "input": deepcopy(stage4_output),
                "output": deepcopy(output),
                "score": round(self.score_helper._effective_score(task, "stage5", agent), 4),
                "source": "stage4_derived",
            },
        }

    def _build_stage4_rows(
        self,
        blocker_ids: list[str],
        stage2_output: dict[str, Any],
    ) -> list[dict[str, Any]]:
        decision = first_pass_terminal_decision(blocker_ids)
        present = set(blocker_ids)
        selected = set(decision["selected_blocker_ids"])
        deferred = set(decision["deferred_blocker_ids"])
        ordered = [
            *decision["selected_blocker_ids"],
            *[bid for bid in decision["deferred_blocker_ids"] if bid not in selected],
        ]
        rows: list[dict[str, Any]] = []
        for repair_order, blocker_id in enumerate(ordered, start=1):
            spec = get_blocker_spec(blocker_id)
            if blocker_id in selected:
                should_repair = True
                execute_decision = "repair"
                adjudication_label = f"repair_{spec['blocker_layer']}_blocker"
                refusal_code = None
            elif decision["final_action"] == "transfer":
                should_repair = False
                execute_decision = "transfer"
                adjudication_label = f"transfer_{spec['blocker_layer']}_blocker"
                refusal_code = decision["transfer_reason"]
            elif blocker_id in deferred:
                should_repair = False
                execute_decision = "defer"
                adjudication_label = f"defer_{spec['blocker_layer']}_blocker"
                refusal_code = "deferred_assistant_side_blocker_v1"
            else:
                should_repair = False
                execute_decision = "defer"
                adjudication_label = f"defer_{spec['blocker_layer']}_blocker"
                refusal_code = "deferred_unspecified_v1"
            rows.append(
                {
                    "blocker_id": blocker_id,
                    "should_repair": should_repair,
                    "repair_order": repair_order,
                    "canonical_repair_steps": materialize_repair_steps(
                        blocker_id=blocker_id,
                        resolved_customer_id=stage2_output.get("resolved_customer_id", ""),
                        resolved_line_id=stage2_output.get("resolved_line_id", ""),
                    ),
                    "oracle_execute_decision": execute_decision,
                    "adjudication_label": adjudication_label,
                    "refusal_code": refusal_code,
                    "depends_on": [dep for dep in spec["depends_on"] if dep in present],
                }
            )
        return rows

    def _build_stage5_output(self, stage4_output: dict[str, Any]) -> dict[str, Any]:
        selected_blocker_ids = [
            row.get("blocker_id")
            for row in stage4_output.get("per_blocker", [])
            if row.get("oracle_execute_decision") == "repair" or row.get("should_repair") is True
        ]
        deferred_blocker_ids = [
            row.get("blocker_id")
            for row in stage4_output.get("per_blocker", [])
            if row.get("blocker_id") and row.get("blocker_id") not in selected_blocker_ids
        ]
        repairability = stage4_output.get("repairability")
        if repairability == "transfer_required":
            final_action = "transfer"
            verification_plan = {
                "required_postchecks": [],
                "success_condition": "transfer_required",
            }
        elif deferred_blocker_ids:
            final_action = "repair_subset"
            verification_plan = {
                "required_postchecks": [],
                "success_condition": "partial_resolution_only",
            }
        else:
            final_action = "repair_all"
            verification_plan = {
                "required_postchecks": ["can_send_mms"],
                "success_condition": "can_send_mms_true",
            }
        return {
            "final_action": final_action,
            "selected_blocker_ids": [bid for bid in selected_blocker_ids if bid],
            "deferred_blocker_ids": deferred_blocker_ids,
            "response_mode": "telecom_structured_execution",
            "verification_plan": verification_plan,
            "transfer_reason": stage4_output.get("transfer_reason"),
            "cancelled_reservation_ids": [bid for bid in selected_blocker_ids if bid],
            "refused_reservation_ids": deferred_blocker_ids,
        }

    def _run_bench_tool_calls(
        self,
        raw_instance: dict[str, Any],
        tool_calls: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not tool_calls:
            return {
                "original_task_id": raw_instance.get("original_task_id"),
                "db_hash_before": None,
                "db_hash_after": None,
                "responses": [],
            }
        payload = {
            "original_task_id": raw_instance.get("original_task_id"),
            "tool_calls": tool_calls,
        }
        proc = subprocess.run(
            [str(self.venv_python), str(self.bridge_script)],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                "Telecom bench tool bridge failed: "
                + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
            )
        return json.loads(proc.stdout)

    def _tool_call(
        self,
        call_id: str,
        name: str,
        arguments: dict[str, Any],
        requestor: str,
    ) -> dict[str, Any]:
        return {
            "id": call_id,
            "name": name,
            "arguments": arguments,
            "requestor": requestor,
        }

    def _extract_phone(self, text: str) -> str | None:
        match = PHONE_RE.search(text)
        return match.group(1) if match else None

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
        app_perm_text = str(app_permissions or "")
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

        sms_permission = "sms" in app_perm_text.lower()
        storage_permission = "storage" in app_perm_text.lower()
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

    def _build_per_blocker_from_raw(self, raw_instance: dict[str, Any]) -> list[dict[str, Any]]:
        return build_per_blocker_from_ids(self._raw_task_blocker_ids(raw_instance))

    def _diagnostic_evidence_for(self, blocker_id: str) -> list[str]:
        return blocker_diagnostic_evidence(blocker_id)

    def _raw_task_blocker_ids(self, raw_instance: dict[str, Any]) -> list[str]:
        return [
            row["blocker_id"]
            for row in raw_instance.get("stage4", {}).get("oracle_output", {}).get("per_blocker", [])
        ]

    def _parse_key_value_lines(self, text: str) -> dict[str, str]:
        result: dict[str, str] = {}
        for line in text.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
        return result

    def _normalize_sim_status(self, sim_status_text: str) -> str:
        text = sim_status_text.lower()
        if "active" in text:
            return "active"
        if "no sim" in text:
            return "missing"
        if "pin" in text:
            return "locked_pin"
        if "puk" in text:
            return "locked_puk"
        return "unknown"
