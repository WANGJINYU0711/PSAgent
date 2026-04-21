"""Evaluate tau2 telecom policy compliance from a full PSAgent stage trace."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TAU2_ROOT = ROOT / "tau2-bench"
TAU2_SRC = TAU2_ROOT / "src"
os.chdir(TAU2_ROOT)
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.data_model.message import (  # type: ignore  # noqa: E402
    AssistantMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.telecom.environment import get_environment, get_tasks  # type: ignore  # noqa: E402
from tau2.environment.toolkit import ToolType  # type: ignore  # noqa: E402
from tau2.evaluator.evaluator_action import ActionEvaluator  # type: ignore  # noqa: E402
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator  # type: ignore  # noqa: E402
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator  # type: ignore  # noqa: E402


def _load_task_map() -> dict[str, Any]:
    try:
        return {task.id: task for task in get_tasks(task_split_name=None)}
    except Exception:
        return {}


def _tool_content_to_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _reward_info_checks(reward_info: Any, field_name: str) -> list[dict[str, Any]]:
    if reward_info is None:
        return []
    payload = json.loads(reward_info.model_dump_json())
    return list(payload.get(field_name) or [])


def _tool_types_by_name() -> dict[str, ToolType]:
    env = get_environment(policy_type="workflow")
    mapping: dict[str, ToolType] = {}
    if getattr(env, "tools", None) is not None:
        for name in env.tools.tools:
            mapping[name] = env.tools.tool_type(name)
    if getattr(env, "user_tools", None) is not None:
        for name in env.user_tools.tools:
            mapping[name] = env.user_tools.tool_type(name)
    return mapping


def _build_policy_trajectory(stage_trace: list[dict[str, Any]]) -> list[Any]:
    trajectory: list[Any] = []
    for trace in stage_trace:
        llm_messages = trace.get("llm_raw_output", []) or []
        executed_tool_calls = list(trace.get("executed_tool_calls", []) or [])
        tool_results = list(trace.get("tool_results", []) or [])
        tool_errors = {
            str(row.get("tool_call", {}).get("id", "")): bool(row.get("content") is not None)
            for row in (trace.get("tool_errors", []) or [])
        }
        replay_tool_calls = list(trace.get("replay_tool_calls", []) or [])
        replay_tool_results = list(trace.get("replay_tool_results", []) or [])
        replay_tool_errors = {
            str(row.get("tool_call", {}).get("id", "")): bool(row.get("content") is not None)
            for row in (trace.get("replay_tool_errors", []) or [])
        }

        if llm_messages:
            tool_idx = 0
            for message in llm_messages:
                tool_calls = [
                    ToolCall(
                        id=str(row.get("id", "")),
                        name=str(row.get("name", "")),
                        arguments=dict(row.get("arguments", {})),
                        requestor=str(row.get("requestor", "assistant")),
                    )
                    for row in (message.get("tool_calls") or [])
                ]
                trajectory.append(
                    AssistantMessage(
                        role="assistant",
                        content=message.get("content"),
                        tool_calls=tool_calls or None,
                        cost=message.get("cost"),
                        usage=message.get("usage"),
                        generation_time_seconds=message.get("generation_time_seconds"),
                    )
                )
                for tool_call in tool_calls:
                    parsed_content = tool_results[tool_idx] if tool_idx < len(tool_results) else None
                    tool_idx += 1
                    trajectory.append(
                        ToolMessage(
                            id=tool_call.id,
                            role="tool",
                            content=_tool_content_to_text(parsed_content),
                            requestor=tool_call.requestor,
                            error=tool_errors.get(tool_call.id, False),
                        )
                    )
        else:
            requestor_groups: dict[str, list[ToolCall]] = {"assistant": [], "user": []}
            for row in executed_tool_calls:
                requestor = str(row.get("requestor", "assistant"))
                requestor_groups.setdefault(requestor, []).append(
                    ToolCall(
                        id=str(row.get("id", "")),
                        name=str(row.get("name", "")),
                        arguments=dict(row.get("arguments", {})),
                        requestor=requestor,
                    )
                )
            for requestor, tool_calls in requestor_groups.items():
                if not tool_calls:
                    continue
                message_cls = UserMessage if requestor == "user" else AssistantMessage
                trajectory.append(
                    message_cls(
                        role=requestor,
                        content=None,
                        tool_calls=tool_calls,
                    )
                )
            for row, parsed_content in zip(executed_tool_calls, tool_results):
                requestor = str(row.get("requestor", "assistant"))
                tool_id = str(row.get("id", ""))
                trajectory.append(
                    ToolMessage(
                        id=tool_id,
                        role="tool",
                        content=_tool_content_to_text(parsed_content),
                        requestor=requestor,
                        error=tool_errors.get(tool_id, False),
                    )
                )

        for row, parsed_content in zip(replay_tool_calls, replay_tool_results):
            requestor = str(row.get("requestor", "assistant"))
            tool_call = ToolCall(
                id=str(row.get("id", "")),
                name=str(row.get("name", "")),
                arguments=dict(row.get("arguments", {})),
                requestor=requestor,
            )
            message_cls = UserMessage if requestor == "user" else AssistantMessage
            trajectory.append(message_cls(role=requestor, content=None, tool_calls=[tool_call]))
            trajectory.append(
                ToolMessage(
                    id=tool_call.id,
                    role="tool",
                    content=_tool_content_to_text(parsed_content),
                    requestor=requestor,
                    error=replay_tool_errors.get(tool_call.id, False),
                )
            )
    return trajectory


def _missing_task_result() -> dict[str, Any]:
    return {
        "bench_action_check": None,
        "bench_communicate_check": None,
        "bench_nl_assertions": None,
        "bench_action_check_raw": [],
        "bench_communicate_check_raw": [],
        "bench_nl_assertions_raw": [],
        "policy_action_violation": False,
        "policy_communication_violation": False,
        "policy_nl_assertions_total": 0,
        "policy_nl_assertions_failed": 0,
        "policy_violation_count": 0,
        "policy_eval_source": "missing_task",
        "policy_eval_scope": "full_episode_trajectory",
    }


def _evaluate_policy_compliance(
    *,
    raw_instance: dict[str, Any],
    stage_trace: list[dict[str, Any]],
) -> dict[str, Any]:
    task = _load_task_map().get(str(raw_instance.get("original_task_id", "")))
    if task is None:
        return _missing_task_result()

    trajectory = _build_policy_trajectory(stage_trace)
    action_info = ActionEvaluator.calculate_reward(task, trajectory, _tool_types_by_name())
    communicate_info = CommunicateEvaluator.calculate_reward(task, trajectory)
    nl_assertions = (
        getattr(getattr(task, "evaluation_criteria", None), "nl_assertions", None) or []
    )
    nl_info = (
        NLAssertionsEvaluator.calculate_reward(task, trajectory)
        if nl_assertions
        else None
    )

    action_checks = _reward_info_checks(action_info, "action_checks")
    communicate_checks = _reward_info_checks(communicate_info, "communicate_checks")
    nl_checks = _reward_info_checks(nl_info, "nl_assertions")

    policy_action_violation = any(
        not bool(row.get("action_match", False)) for row in action_checks
    )
    policy_communication_violation = any(
        not bool(row.get("met", False)) for row in communicate_checks
    )
    policy_nl_assertions_total = len(nl_checks)
    policy_nl_assertions_failed = sum(
        1 for row in nl_checks if not bool(row.get("met", False))
    )
    return {
        "bench_action_check": not policy_action_violation,
        "bench_communicate_check": not policy_communication_violation,
        "bench_nl_assertions": policy_nl_assertions_failed == 0,
        "bench_action_check_raw": action_checks,
        "bench_communicate_check_raw": communicate_checks,
        "bench_nl_assertions_raw": nl_checks,
        "policy_action_violation": policy_action_violation,
        "policy_communication_violation": policy_communication_violation,
        "policy_nl_assertions_total": policy_nl_assertions_total,
        "policy_nl_assertions_failed": policy_nl_assertions_failed,
        "policy_violation_count": int(policy_action_violation)
        + int(policy_communication_violation)
        + policy_nl_assertions_failed,
        "policy_eval_source": "tau2_reward_evaluators_full_episode",
        "policy_eval_scope": "full_episode_trajectory",
    }


def main() -> None:
    payload = json.load(sys.stdin)
    result = _evaluate_policy_compliance(
        raw_instance=dict(payload.get("raw_instance", {}) or {}),
        stage_trace=list(payload.get("stage_trace", []) or []),
    )
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
