"""Bench-backed executor for Day 8 Stage 2/3 real-tool integration."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

from .base_executor import BaseExecutor
from .simulated_executor import SimulatedExecutor
from tree_family.specs import AgentSpec, TaskDescriptor


CURRENT_TIME = datetime.fromisoformat("2024-05-15T15:00:00")
AIRPORT_CODE_PATTERN = re.compile(r"\b[A-Z]{3}\b")
RESERVATION_ID_PATTERN = re.compile(r"\b[A-Z0-9]{6}\b")


class BenchBackedExecutor(BaseExecutor):
    """Use tau2's real airline Environment for Stage 2/3 read tools."""

    def __init__(self, stages: list[str], seed: int = 0) -> None:
        super().__init__(stages=stages, seed=seed)
        self.score_helper = SimulatedExecutor(stages=stages, seed=seed)
        self.root = Path(__file__).resolve().parents[2]
        self.venv_python = self.root / "tau2-bench" / ".venv" / "bin" / "python"
        self.bridge_script = Path(__file__).with_name("_bench_tool_bridge.py")

    def run_path(
        self,
        task: TaskDescriptor,
        path: list[str],
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
    ) -> dict[str, Any]:
        stage_trace: list[dict[str, Any]] = []
        stage_scores: dict[str, float] = {}
        stage_outputs: dict[str, dict[str, Any]] = {}
        tool_calls_made = 0
        mutating_tool_calls_made = 0
        any_tool_error = False
        db_hash_before: str | None = None
        db_hash_after: str | None = None

        stage1_output = self._run_stage1(task, path[0], agent_map, raw_instance)
        stage_outputs["stage1"] = stage1_output
        stage_trace.append(stage1_output["trace"])
        stage_scores["stage1"] = stage1_output["trace"]["score"]

        stage2_output = self._run_stage2(task, path[1], agent_map, raw_instance, stage1_output["output"])
        stage_outputs["stage2"] = stage2_output
        stage_trace.append(stage2_output["trace"])
        stage_scores["stage2"] = stage2_output["trace"]["score"]
        tool_calls_made += len(stage2_output["trace"]["executed_tool_calls"])
        any_tool_error = any_tool_error or bool(stage2_output["trace"]["tool_errors"])
        db_hash_before = stage2_output["trace"].get("db_hash_before", db_hash_before)
        db_hash_after = stage2_output["trace"].get("db_hash_after", db_hash_after)

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
        stage_scores["stage3"] = stage3_output["trace"]["score"]
        tool_calls_made += len(stage3_output["trace"]["executed_tool_calls"])
        any_tool_error = any_tool_error or bool(stage3_output["trace"]["tool_errors"])
        if db_hash_before is None:
            db_hash_before = stage3_output["trace"].get("db_hash_before")
        db_hash_after = stage3_output["trace"].get("db_hash_after", db_hash_after)

        stage4_output = self._run_stage4(task, path[3], agent_map, raw_instance, stage3_output["output"])
        stage_outputs["stage4"] = stage4_output
        stage_trace.append(stage4_output["trace"])
        stage_scores["stage4"] = stage4_output["trace"]["score"]

        stage5_output = self._run_stage5(task, path[4], agent_map, raw_instance, stage4_output["output"])
        stage_outputs["stage5"] = stage5_output
        stage_trace.append(stage5_output["trace"])
        stage_scores["stage5"] = stage5_output["trace"]["score"]

        leaf_type = "unshared" if any(agent_map[agent_id].g == 1 for agent_id in path) else "shared"
        path_agent_cost = sum(agent_map[agent_id].base_cost for agent_id in path)
        bench_aux_eval = {
            "db_hash_before": db_hash_before,
            "db_hash_after": db_hash_after,
            "tool_calls_made": tool_calls_made,
            "mutating_tool_calls_made": mutating_tool_calls_made,
            "any_tool_error_occurred": any_tool_error,
            "bench_success": "deferred",
            "bench_db_check": db_hash_before == db_hash_after if db_hash_before is not None and db_hash_after is not None else "deferred",
            "bench_action_check": "deferred",
            "bench_communicate_check": "deferred",
            "bench_nl_assertions": "deferred",
        }

        final_output = stage5_output["output"]
        return {
            "final_action": final_output["final_action"],
            "cancelled_reservation_ids": final_output["cancelled_reservation_ids"],
            "refused_reservation_ids": final_output["refused_reservation_ids"],
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
        agent = agent_map[agent_id]
        score = self.score_helper._effective_score(task, "stage1", agent)
        user_context = raw_instance.get("user_context", {})
        text = " ".join(
            [
                str(user_context.get("reason_for_call", "")),
                str(user_context.get("known_info", "")),
                str(user_context.get("task_instructions", "")),
            ]
        )
        user_id = self._extract_user_id(text)
        explicit_res_ids = sorted(set(RESERVATION_ID_PATTERN.findall(text)))
        route_codes = AIRPORT_CODE_PATTERN.findall(text)
        route_pairs = [route_codes[idx : idx + 2] for idx in range(0, max(0, len(route_codes) - 1), 2)]
        output = {
            "user_id": user_id,
            "explicit_reservation_ids": explicit_res_ids,
            "route_pairs": [pair for pair in route_pairs if len(pair) == 2],
            "all_upcoming": "all of your upcoming flights" in text.lower(),
            "single_passenger_only": "only have one passenger" in text.lower(),
            "target_text": text,
        }
        return {
            "input": deepcopy(user_context),
            "output": output,
            "trace": {
                "stage_name": "stage1",
                "agent_id": agent_id,
                "agent_g": agent.g,
                "planned_tool_calls": [],
                "executed_tool_calls": [],
                "tool_results": [],
                "tool_errors": [],
                "input": deepcopy(user_context),
                "output": deepcopy(output),
                "score": round(score, 4),
                "source": "bench_backed",
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
        score = self.score_helper._effective_score(task, "stage2", agent)
        user_id = stage1_output.get("user_id")
        tool_calls = [
            self._tool_call("stage2_user", "get_user_details", {"user_id": user_id}),
        ]
        bridge_result = self._run_bench_tool_calls(raw_instance, tool_calls)
        user_result = bridge_result["responses"][0]["content"] if bridge_result["responses"] else {}
        reservation_ids = list((user_result or {}).get("reservations", []) or [])

        inspect_ids = self._select_reservations_to_inspect(
            reservation_ids=reservation_ids,
            stage1_output=stage1_output,
            score=score,
            task_id=task.task_id,
            agent_id=agent_id,
        )
        reservation_calls = [
            self._tool_call(
                f"stage2_res_{idx}",
                "get_reservation_details",
                {"reservation_id": reservation_id},
            )
            for idx, reservation_id in enumerate(inspect_ids)
        ]
        reservation_result = self._run_bench_tool_calls(raw_instance, reservation_calls)
        reservation_details = {
            row["tool_call"]["arguments"]["reservation_id"]: row["content"]
            for row in reservation_result["responses"]
            if not row["error"]
        }
        resolved_ids, resolution_status = self._resolve_from_real_reservations(
            reservation_details=reservation_details,
            stage1_output=stage1_output,
            score=score,
            task_id=task.task_id,
            agent_id=agent_id,
        )
        output = {
            "candidate_reservations": inspect_ids,
            "resolved_reservations": resolved_ids,
            "resolution_status": resolution_status,
            "primary_target_reservation_id": resolved_ids[0] if len(resolved_ids) == 1 else None,
            "user_id": user_id,
        }
        trace = {
            "stage_name": "stage2",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "planned_tool_calls": tool_calls + reservation_calls,
            "executed_tool_calls": [row["tool_call"] for row in bridge_result["responses"]] + [row["tool_call"] for row in reservation_result["responses"]],
            "tool_results": [row["content"] for row in bridge_result["responses"]] + [row["content"] for row in reservation_result["responses"]],
            "tool_errors": [row for row in bridge_result["responses"] if row["error"]] + [row for row in reservation_result["responses"] if row["error"]],
            "candidate_reservations": inspect_ids,
            "resolved_reservations": resolved_ids,
            "db_hash_before": bridge_result.get("db_hash_before"),
            "db_hash_after": reservation_result.get("db_hash_after", bridge_result.get("db_hash_after")),
            "input": deepcopy(stage1_output),
            "output": deepcopy(output),
            "score": round(score, 4),
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
        score = self.score_helper._effective_score(task, "stage3", agent)
        resolved_ids = list(stage2_output.get("resolved_reservations", []) or [])
        user_id = stage2_output.get("user_id") or stage1_output.get("user_id")

        tool_calls = [self._tool_call("stage3_user", "get_user_details", {"user_id": user_id})]
        for idx, reservation_id in enumerate(resolved_ids):
            tool_calls.append(
                self._tool_call(
                    f"stage3_res_{idx}",
                    "get_reservation_details",
                    {"reservation_id": reservation_id},
                )
            )
        base_result = self._run_bench_tool_calls(raw_instance, tool_calls)
        responses = base_result["responses"]
        user_details = responses[0]["content"] if responses else {}
        reservation_details = {
            row["tool_call"]["arguments"]["reservation_id"]: row["content"]
            for row in responses[1:]
            if not row["error"]
        }

        flight_calls = self._plan_stage3_flight_status_calls(
            reservation_details=reservation_details,
            score=score,
            task_id=task.task_id,
            agent_id=agent_id,
        )
        flight_result = self._run_bench_tool_calls(raw_instance, flight_calls)
        flight_statuses = {
            (row["tool_call"]["arguments"]["flight_number"], row["tool_call"]["arguments"]["date"]): row["content"]
            for row in flight_result["responses"]
            if not row["error"]
        }
        computed_rows = self._compute_stage3_rows(
            raw_instance=raw_instance,
            reservation_details=reservation_details,
            user_details=user_details or {},
            flight_statuses=flight_statuses,
        )
        output = {"per_reservation": computed_rows}
        trace = {
            "stage_name": "stage3",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "planned_tool_calls": tool_calls + flight_calls,
            "executed_tool_calls": [row["tool_call"] for row in responses] + [row["tool_call"] for row in flight_result["responses"]],
            "tool_results": [row["content"] for row in responses] + [row["content"] for row in flight_result["responses"]],
            "tool_errors": [row for row in responses if row["error"]] + [row for row in flight_result["responses"] if row["error"]],
            "computed_features": deepcopy(computed_rows),
            "db_hash_before": base_result.get("db_hash_before"),
            "db_hash_after": flight_result.get("db_hash_after", base_result.get("db_hash_after")),
            "input": {"stage2_output": deepcopy(stage2_output)},
            "output": deepcopy(output),
            "score": round(score, 4),
            "source": "bench_backed",
        }
        return {"input": {"stage2_output": deepcopy(stage2_output)}, "output": output, "trace": trace}

    def _run_stage4(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
        stage3_output: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        score = self.score_helper._effective_score(task, "stage4", agent)
        rows = deepcopy(stage3_output.get("per_reservation", []))
        per_reservation: list[dict[str, Any]] = []
        cancel_rows: list[dict[str, Any]] = []
        for row in rows:
            eligible = bool(
                row.get("eligible_by_24h_rule")
                or row.get("eligible_by_airline_cancel_rule")
                or row.get("eligible_by_business_rule")
                or row.get("eligible_by_insurance_rule")
            )
            adjudication_label = "allow_cancel_refund" if eligible else "deny_policy"
            refusal_code = None if eligible else "policy_not_satisfied"
            if eligible and score < 0.50:
                eligible = False
                adjudication_label = "deny_conservative_agent"
                refusal_code = "conservative_agent"
            elif eligible and score < 0.62 and len(rows) > 1 and row is rows[-1]:
                eligible = False
                adjudication_label = "deny_subset_conservative_agent"
                refusal_code = "subset_conservative_agent"

            result_row = deepcopy(row)
            result_row["policy_eligible_cancel_with_refund"] = eligible
            result_row["policy_adjudication_label"] = adjudication_label
            result_row["policy_refusal_code"] = refusal_code
            result_row["policy_rule_trace"] = [adjudication_label]
            per_reservation.append(result_row)
            if eligible:
                cancel_rows.append(result_row)

        output = {"per_reservation": per_reservation}
        trace = {
            "stage_name": "stage4",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "planned_tool_calls": [],
            "executed_tool_calls": [],
            "tool_results": [],
            "tool_errors": [],
            "input": deepcopy(stage3_output),
            "output": deepcopy(output),
            "score": round(score, 4),
            "source": "structured_logic",
        }
        return {"input": deepcopy(stage3_output), "output": output, "trace": trace}

    def _run_stage5(
        self,
        task: TaskDescriptor,
        agent_id: str,
        agent_map: dict[str, AgentSpec],
        raw_instance: dict[str, Any],
        stage4_output: dict[str, Any],
    ) -> dict[str, Any]:
        agent = agent_map[agent_id]
        score = self.score_helper._effective_score(task, "stage5", agent)
        rows = deepcopy(stage4_output.get("per_reservation", []))
        cancelled_ids = sorted(
            row["reservation_id"] for row in rows if row.get("policy_eligible_cancel_with_refund")
        )
        refused_ids = sorted(
            row["reservation_id"] for row in rows if not row.get("policy_eligible_cancel_with_refund")
        )
        if score < 0.58 and cancelled_ids:
            refused_ids.append(cancelled_ids.pop())
            refused_ids = sorted(refused_ids)
        if cancelled_ids and refused_ids:
            final_action = "cancel_subset"
        elif cancelled_ids:
            final_action = "cancel_all"
        else:
            final_action = "refuse_all"
        output = {
            "final_action": final_action,
            "cancelled_reservation_ids": sorted(cancelled_ids),
            "refused_reservation_ids": sorted(refused_ids),
            "response_mode": "bench_backed_structured_execution",
        }
        trace = {
            "stage_name": "stage5",
            "agent_id": agent_id,
            "agent_g": agent.g,
            "planned_tool_calls": [],
            "executed_tool_calls": [],
            "tool_results": [],
            "tool_errors": [],
            "input": deepcopy(stage4_output),
            "output": deepcopy(output),
            "score": round(score, 4),
            "source": "structured_logic",
        }
        return {"input": deepcopy(stage4_output), "output": output, "trace": trace}

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
                "Bench tool bridge failed: "
                + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
            )
        return json.loads(proc.stdout)

    def _tool_call(self, call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": call_id,
            "name": name,
            "arguments": arguments,
            "requestor": "assistant",
        }

    def _extract_user_id(self, text: str) -> str | None:
        match = re.search(r"user id is ([a-z_0-9]+)", text, flags=re.IGNORECASE)
        return match.group(1) if match else None

    def _quality_noise(self, task_id: str, stage_name: str, agent_id: str) -> float:
        key = f"{self.seed}|{task_id}|{stage_name}|{agent_id}"
        digest = hashlib.md5(key.encode("utf-8")).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        return (2.0 * value) - 1.0

    def _select_reservations_to_inspect(
        self,
        reservation_ids: list[str],
        stage1_output: dict[str, Any],
        score: float,
        task_id: str,
        agent_id: str,
    ) -> list[str]:
        explicit_ids = [rid for rid in stage1_output.get("explicit_reservation_ids", []) if rid in reservation_ids]
        ordered = explicit_ids + [rid for rid in reservation_ids if rid not in explicit_ids]
        if score >= 0.72:
            return ordered
        if score >= 0.56:
            limit = max(len(explicit_ids), max(1, len(ordered) - 1))
            return ordered[:limit]
        if score >= 0.42:
            return ordered[: max(1, min(2, len(ordered)))]
        noise = self._quality_noise(task_id, "stage2", agent_id)
        if noise > 0 and len(ordered) > 2:
            return ordered[:2]
        return ordered[:1]

    def _resolve_from_real_reservations(
        self,
        reservation_details: dict[str, dict[str, Any]],
        stage1_output: dict[str, Any],
        score: float,
        task_id: str,
        agent_id: str,
    ) -> tuple[list[str], str]:
        rows = list(reservation_details.values())
        explicit_ids = [
            rid for rid in stage1_output.get("explicit_reservation_ids", []) if rid in reservation_details
        ]
        if explicit_ids:
            resolved = explicit_ids
        else:
            route_pairs = stage1_output.get("route_pairs", [])
            all_upcoming = bool(stage1_output.get("all_upcoming"))
            single_passenger_only = bool(stage1_output.get("single_passenger_only"))
            resolved = []
            for row in rows:
                reservation_id = row.get("reservation_id")
                if not reservation_id:
                    continue
                keep = False
                if route_pairs:
                    for origin, destination in route_pairs:
                        if row.get("origin") == origin and row.get("destination") == destination:
                            keep = True
                elif all_upcoming:
                    keep = self._reservation_is_upcoming(row)
                else:
                    keep = True
                if keep and single_passenger_only and len(row.get("passengers", []) or []) != 1:
                    keep = False
                if keep:
                    resolved.append(reservation_id)

        if score < 0.55 and len(resolved) > 1:
            resolved = resolved[:-1]
            status = "under_resolved"
        elif score < 0.40 and len(reservation_details) > len(resolved):
            extras = [rid for rid in reservation_details if rid not in resolved]
            if extras:
                resolved = resolved + extras[:1]
            status = "over_broad"
        else:
            status = "resolved" if len(resolved) <= 1 else "resolved_set"

        if not resolved and reservation_details:
            noise = self._quality_noise(task_id, "stage2_resolve", agent_id)
            if noise > 0:
                resolved = [next(iter(reservation_details))]
                status = "fallback_resolved"
            else:
                status = "unresolved"

        return sorted(dict.fromkeys(resolved)), status

    def _reservation_is_upcoming(self, reservation: dict[str, Any]) -> bool:
        for flight in reservation.get("flights", []) or []:
            date_str = flight.get("date")
            if not date_str:
                continue
            try:
                flight_dt = datetime.fromisoformat(f"{date_str}T00:00:00")
            except ValueError:
                continue
            if flight_dt >= CURRENT_TIME.replace(hour=0, minute=0, second=0, microsecond=0):
                return True
        return False

    def _plan_stage3_flight_status_calls(
        self,
        reservation_details: dict[str, dict[str, Any]],
        score: float,
        task_id: str,
        agent_id: str,
    ) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        flight_rows: list[tuple[str, str]] = []
        for reservation in reservation_details.values():
            for flight in reservation.get("flights", []) or []:
                flight_number = flight.get("flight_number")
                date = flight.get("date")
                if flight_number and date:
                    flight_rows.append((flight_number, date))
        selected = list(flight_rows)
        if score < 0.58 and len(selected) > 1:
            selected = selected[:-1]
        if score < 0.42 and len(selected) > 1:
            selected = selected[: max(1, len(selected) // 2)]
        if score < 0.35 and selected:
            noise = self._quality_noise(task_id, "stage3", agent_id)
            if noise < 0:
                selected = selected[:1]

        for idx, (flight_number, date) in enumerate(selected):
            calls.append(
                self._tool_call(
                    f"stage3_flight_{idx}",
                    "get_flight_status",
                    {"flight_number": flight_number, "date": date},
                )
            )
        return calls

    def _compute_stage3_rows(
        self,
        raw_instance: dict[str, Any],
        reservation_details: dict[str, dict[str, Any]],
        user_details: dict[str, Any],
        flight_statuses: dict[tuple[str, str], Any],
    ) -> list[dict[str, Any]]:
        reason_text = " ".join(
            [
                str(raw_instance.get("user_context", {}).get("reason_for_call", "")),
                str(raw_instance.get("user_context", {}).get("task_instructions", "")),
            ]
        ).lower()
        rows: list[dict[str, Any]] = []
        membership = user_details.get("membership", "regular")
        for reservation_id, reservation in reservation_details.items():
            created_at = reservation.get("created_at")
            hours_since_booking = self._hours_since_booking(created_at)
            cabin = reservation.get("cabin")
            insurance = reservation.get("insurance") == "yes"
            passenger_count = len(reservation.get("passengers", []) or [])
            flight_status_list: list[str] = []
            any_leg_flown = False
            any_leg_cancelled = False
            for flight in reservation.get("flights", []) or []:
                key = (flight.get("flight_number"), flight.get("date"))
                status = flight_statuses.get(key, "unknown")
                if isinstance(status, dict):
                    status = status.get("status", "unknown")
                flight_status_list.append(str(status))
                any_leg_flown = any_leg_flown or status in {"flying", "landed"}
                any_leg_cancelled = any_leg_cancelled or status == "cancelled"

            stated_reason_supported_by_insurance = any(
                token in reason_text
                for token in ["medical", "ill", "sick", "death", "weather", "emergency"]
            )
            eligible_by_24h_rule = hours_since_booking is not None and hours_since_booking <= 24.0 and not any_leg_flown
            eligible_by_airline_cancel_rule = any_leg_cancelled
            eligible_by_business_rule = cabin == "business" and not any_leg_flown
            eligible_by_insurance_rule = insurance and stated_reason_supported_by_insurance and not any_leg_flown
            rows.append(
                {
                    "reservation_id": reservation_id,
                    "hours_since_booking": hours_since_booking,
                    "insurance": insurance,
                    "cabin": cabin,
                    "membership": membership,
                    "passenger_count": passenger_count,
                    "flight_statuses": flight_status_list,
                    "any_leg_flown": any_leg_flown,
                    "any_leg_cancelled_by_airline": any_leg_cancelled,
                    "stated_reason_supported_by_insurance": stated_reason_supported_by_insurance,
                    "eligible_by_24h_rule": eligible_by_24h_rule,
                    "eligible_by_airline_cancel_rule": eligible_by_airline_cancel_rule,
                    "eligible_by_business_rule": eligible_by_business_rule,
                    "eligible_by_insurance_rule": eligible_by_insurance_rule,
                }
            )
        return rows

    def _hours_since_booking(self, created_at: str | None) -> float | None:
        if not created_at:
            return None
        try:
            created = datetime.fromisoformat(created_at)
        except ValueError:
            return None
        delta = CURRENT_TIME - created
        return round(delta.total_seconds() / 3600.0, 2)
