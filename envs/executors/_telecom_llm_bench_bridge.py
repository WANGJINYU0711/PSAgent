"""LLM + real telecom tool bridge for Stage 1/2/3/4/5 execution."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter
from typing import Any


JSON_STDOUT = sys.stdout
# Keep the subprocess stdout protocol clean: every dependency warning, tau2 log,
# accidental print, or traceback should go to stderr. Only the final JSON result
# is written to JSON_STDOUT explicitly.
sys.stdout = sys.stderr

ROOT = Path(__file__).resolve().parents[2]
TAU2_ROOT = ROOT / "tau2-bench"
TAU2_SRC = TAU2_ROOT / "src"
os.chdir(TAU2_ROOT)
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.data_model.message import (  # type: ignore  # noqa: E402
    AssistantMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
    UserMessage,
)
from tau2.domains.telecom.environment import get_environment, get_tasks  # type: ignore  # noqa: E402
from tau2.environment.toolkit import ToolType  # type: ignore  # noqa: E402
from tau2.evaluator.evaluator_action import ActionEvaluator  # type: ignore  # noqa: E402
from tau2.evaluator.evaluator_communicate import CommunicateEvaluator  # type: ignore  # noqa: E402
from tau2.evaluator.evaluator_nl_assertions import NLAssertionsEvaluator  # type: ignore  # noqa: E402
from tau2.utils.llm_utils import generate as llm_generate  # type: ignore  # noqa: E402


def _load_task_map() -> dict[str, Any]:
    try:
        return {task.id: task for task in get_tasks(task_split_name=None)}
    except Exception:
        return {}


def _parse_tool_message_content(content: str | None) -> Any:
    if content is None:
        return None
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        return content


def _extract_json(text: str | None) -> dict[str, Any] | None:
    if not text:
        return None
    text = text.strip()
    for candidate in (text, _extract_codeblock(text), _extract_braced_json(text)):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except Exception:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _extract_codeblock(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else None


def _extract_braced_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


def _extract_usage_value(usage: dict[str, Any] | None, *keys: str) -> float:
    if not isinstance(usage, dict):
        return 0.0
    current: Any = usage
    for key in keys:
        if not isinstance(current, dict):
            return 0.0
        current = current.get(key)
    if isinstance(current, (int, float)):
        return float(current)
    return 0.0


def _normalize_usage(usage: dict[str, Any] | None) -> dict[str, float]:
    prompt_tokens = _extract_usage_value(usage, "prompt_tokens") or _extract_usage_value(
        usage, "input_tokens"
    )
    completion_tokens = _extract_usage_value(
        usage, "completion_tokens"
    ) or _extract_usage_value(usage, "output_tokens")
    total_tokens = _extract_usage_value(usage, "total_tokens")
    if total_tokens <= 0.0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def _assistant_step_to_dict(
    message: Any,
    *,
    round_trip_seconds: float | None = None,
) -> dict[str, Any]:
    usage_breakdown = _normalize_usage(getattr(message, "usage", None))
    return {
        "content": message.content,
        "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
        "cost": message.cost,
        "usage": message.usage,
        "generation_time_seconds": message.generation_time_seconds,
        "round_trip_seconds": round_trip_seconds,
        "usage_breakdown": usage_breakdown,
    }


def _normalize_tool_arguments(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(arguments)
    if name == "check_app_permissions":
        app_name = normalized.get("app_name")
        if isinstance(app_name, str):
            lowered = app_name.lower()
            alias_map = {
                "messages": "messaging",
                "message": "messaging",
            }
            normalized["app_name"] = alias_map.get(lowered, lowered)
    return normalized


def _execute_tool_call(
    env: Any,
    tool_call_data: dict[str, Any],
    fallback_requestor: str = "assistant",
) -> tuple[dict[str, Any], Any, bool]:
    name = str(tool_call_data.get("name", ""))
    normalized_arguments = _normalize_tool_arguments(
        name, dict(tool_call_data.get("arguments", {}))
    )
    explicit_requestor = str(tool_call_data.get("requestor") or "").strip().lower()
    requestor = (
        explicit_requestor
        if explicit_requestor in {"assistant", "user"} and explicit_requestor != "assistant"
        else fallback_requestor
    )
    tool_call = ToolCall(
        id=str(tool_call_data.get("id", "")),
        name=name,
        arguments=normalized_arguments,
        requestor=requestor,
    )
    started_at = perf_counter()
    tool_message = env.get_response(tool_call)
    wall_clock_seconds = perf_counter() - started_at
    parsed_content = _parse_tool_message_content(tool_message.content)
    tool_call_payload = tool_call.model_dump()
    tool_call_payload["wall_clock_seconds"] = wall_clock_seconds
    return tool_call_payload, parsed_content, bool(tool_message.error)


def _tool_content_to_text(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False)


def _build_policy_trajectory(
    messages: list[Any],
    replayed_tool_calls: list[dict[str, Any]],
    replay_tool_results: list[Any],
    replay_tool_errors: list[dict[str, Any]],
) -> list[Any]:
    trajectory: list[Any] = []
    error_by_call_id = {
        str(row.get("tool_call", {}).get("id", "")): bool(row.get("content") is not None)
        for row in replay_tool_errors
    }
    for replay_call, replay_result in zip(replayed_tool_calls, replay_tool_results):
        requestor = str(replay_call.get("requestor", "assistant"))
        tool_call = ToolCall(
            id=str(replay_call.get("id", "")),
            name=str(replay_call.get("name", "")),
            arguments=dict(replay_call.get("arguments", {})),
            requestor=requestor,
        )
        if requestor == "user":
            trajectory.append(
                UserMessage(
                    role="user",
                    content=None,
                    tool_calls=[tool_call],
                )
            )
        else:
            trajectory.append(
                AssistantMessage(
                    role="assistant",
                    content=None,
                    tool_calls=[tool_call],
                )
            )
        trajectory.append(
            ToolMessage(
                id=tool_call.id,
                role="tool",
                content=_tool_content_to_text(replay_result),
                requestor=requestor,
                error=error_by_call_id.get(tool_call.id, False),
            )
        )

    for message in messages:
        if isinstance(message, SystemMessage):
            continue
        trajectory.append(message)
    return trajectory


def _tool_types_by_name(env: Any) -> dict[str, ToolType]:
    mapping: dict[str, ToolType] = {}
    if getattr(env, "tools", None) is not None:
        for name in env.tools.tools:
            mapping[name] = env.tools.tool_type(name)
    if getattr(env, "user_tools", None) is not None:
        for name in env.user_tools.tools:
            mapping[name] = env.user_tools.tool_type(name)
    return mapping


def _reward_info_checks(reward_info: Any, field_name: str) -> list[dict[str, Any]]:
    if reward_info is None:
        return []
    payload = json.loads(reward_info.model_dump_json())
    return list(payload.get(field_name) or [])


def _evaluate_policy_compliance(
    *,
    task: Any,
    trajectory: list[Any],
    tool_types: dict[str, ToolType],
) -> dict[str, Any]:
    if task is None:
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
        }

    action_info = ActionEvaluator.calculate_reward(task, trajectory, tool_types)
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
        "policy_eval_source": "tau2_reward_evaluators_stage5_local_debug",
    }


def _aggregate_resource_usage(
    llm_messages: list[dict[str, Any]],
    *,
    executed_tool_calls: list[dict[str, Any]],
    replayed_tool_calls: list[dict[str, Any]],
    stage_wall_clock_seconds: float,
) -> dict[str, Any]:
    prompt_tokens_total = 0.0
    completion_tokens_total = 0.0
    total_tokens_total = 0.0
    api_cost_total_usd_raw = 0.0
    generation_time_total_seconds = 0.0
    llm_round_trip_total_seconds = 0.0
    llm_call_count = len(llm_messages)

    for message in llm_messages:
        usage_breakdown = _normalize_usage(message.get("usage"))
        prompt_tokens_total += usage_breakdown["prompt_tokens"]
        completion_tokens_total += usage_breakdown["completion_tokens"]
        total_tokens_total += usage_breakdown["total_tokens"]
        api_cost_total_usd_raw += float(message.get("cost") or 0.0)
        generation_time_total_seconds += float(
            message.get("generation_time_seconds") or 0.0
        )
        llm_round_trip_total_seconds += float(message.get("round_trip_seconds") or 0.0)

    tool_wall_clock_total_seconds = sum(
        float(call.get("wall_clock_seconds") or 0.0)
        for call in [*replayed_tool_calls, *executed_tool_calls]
    )

    return {
        "llm_call_count": llm_call_count,
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "api_cost_total_usd_raw": api_cost_total_usd_raw,
        "generation_time_total_seconds": generation_time_total_seconds,
        "llm_round_trip_total_seconds": llm_round_trip_total_seconds,
        "tool_wall_clock_total_seconds": tool_wall_clock_total_seconds,
        "stage_wall_clock_seconds": stage_wall_clock_seconds,
        "usage_breakdown": {
            "prompt_tokens_total": prompt_tokens_total,
            "completion_tokens_total": completion_tokens_total,
            "total_tokens_total": total_tokens_total,
        },
        "cost_breakdown": {
            "api_cost_total_usd_raw": api_cost_total_usd_raw,
        },
    }


def main(*, json_stdout: Any = JSON_STDOUT) -> None:
    payload = json.load(sys.stdin)
    stage_name = payload["stage_name"]
    original_task_id = str(payload.get("original_task_id", ""))
    model = payload["model"]
    llm_args = dict(payload.get("llm_args", {}))
    max_rounds = int(payload.get("max_rounds", 4))
    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]
    allowed_tools = list(payload.get("allowed_tools", []))
    replay_tool_calls = list(payload.get("replay_tool_calls", []))

    stage_started_at = perf_counter()
    env = get_environment(policy_type="workflow")
    task_map = _load_task_map()
    task = task_map.get(original_task_id)
    if task is not None and getattr(task, "initial_state", None) is not None:
        init_actions = getattr(task.initial_state, "initialization_actions", None)
        env.set_state(None, init_actions, [])

    db_hash_before_replay = env.get_db_hash()
    replayed_tool_calls: list[dict[str, Any]] = []
    replay_tool_results: list[Any] = []
    replay_tool_errors: list[dict[str, Any]] = []
    for replay_call in replay_tool_calls:
        replayed_call, replay_result, replay_error = _execute_tool_call(env, replay_call)
        replayed_tool_calls.append(replayed_call)
        replay_tool_results.append(replay_result)
        if replay_error:
            replay_tool_errors.append(
                {
                    "tool_call": replayed_call,
                    "content": replay_result,
                }
            )
    db_hash_after_replay = env.get_db_hash()

    tools, requestor_by_tool = _filter_tools(env, allowed_tools)
    db_hash_before = env.get_db_hash()
    messages: list[Any] = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt),
    ]
    llm_messages: list[dict[str, Any]] = []
    executed_tool_calls: list[dict[str, Any]] = []
    tool_results: list[Any] = []
    tool_errors: list[dict[str, Any]] = []
    final_output: dict[str, Any] | None = None

    for step_idx in range(max_rounds):
        generate_kwargs = {
            "model": model,
            "messages": messages,
            "call_name": f"psagent_{stage_name}_telecom_llm_bench",
            **llm_args,
        }
        if tools:
            generate_kwargs["tools"] = tools
            generate_kwargs["tool_choice"] = "auto"
        llm_started_at = perf_counter()
        assistant = llm_generate(**generate_kwargs)
        llm_round_trip_seconds = perf_counter() - llm_started_at
        llm_messages.append(
            _assistant_step_to_dict(
                assistant,
                round_trip_seconds=llm_round_trip_seconds,
            )
        )
        messages.append(assistant)

        if assistant.tool_calls:
            for tool_call in assistant.tool_calls:
                replayed_call, parsed_content, tool_error = _execute_tool_call(
                    env,
                    tool_call.model_dump(),
                    fallback_requestor=requestor_by_tool.get(tool_call.name, "assistant"),
                )
                executed_tool_calls.append(replayed_call)
                tool_results.append(parsed_content)
                if tool_error:
                    tool_errors.append(
                        {
                            "tool_call": replayed_call,
                            "content": parsed_content,
                        }
                    )
                messages.append(
                    ToolMessage(
                        id=replayed_call["id"],
                        role="tool",
                        content=_tool_content_to_text(parsed_content),
                        requestor=replayed_call.get("requestor", "assistant"),
                        error=tool_error,
                    )
                )
            continue

        final_output = _extract_json(assistant.content)
        if final_output is not None:
            break
        if step_idx < max_rounds - 1:
            messages.append(
                UserMessage(
                    role="user",
                    content=(
                        "Return only a valid JSON object matching the required schema. "
                        "Do not include prose."
                    ),
                )
            )

    stage_wall_clock_seconds = perf_counter() - stage_started_at
    resource_usage = _aggregate_resource_usage(
        llm_messages,
        executed_tool_calls=executed_tool_calls,
        replayed_tool_calls=replayed_tool_calls,
        stage_wall_clock_seconds=stage_wall_clock_seconds,
    )
    policy_eval_debug = None
    if stage_name == "stage5":
        trajectory = _build_policy_trajectory(
            messages,
            replayed_tool_calls,
            replay_tool_results,
            replay_tool_errors,
        )
        policy_eval_debug = _evaluate_policy_compliance(
            task=task,
            trajectory=trajectory,
            tool_types=_tool_types_by_name(env),
        )

    result = {
        "stage_name": stage_name,
        "original_task_id": original_task_id,
        "db_hash_before_replay": db_hash_before_replay,
        "db_hash_after_replay": db_hash_after_replay,
        "db_hash_before": db_hash_before,
        "db_hash_after": env.get_db_hash(),
        "replay_tool_calls": replayed_tool_calls,
        "replay_tool_results": replay_tool_results,
        "replay_tool_errors": replay_tool_errors,
        "llm_messages": llm_messages,
        "executed_tool_calls": executed_tool_calls,
        "tool_results": tool_results,
        "tool_errors": tool_errors,
        "final_output": final_output,
        "resource_usage": resource_usage,
        "policy_eval_debug": policy_eval_debug,
    }
    json.dump(result, json_stdout, ensure_ascii=False, separators=(",", ":"))
    json_stdout.write("\n")
    json_stdout.flush()


def _filter_tools(env: Any, allowed_names: list[str]) -> tuple[list[Any], dict[str, str]]:
    allowed = set(allowed_names)
    tools = []
    requestor_by_tool: dict[str, str] = {}
    assistant_tools = env.get_tools()
    assistant_names = {tool.name for tool in assistant_tools}
    for tool in assistant_tools:
        if tool.name in allowed:
            tools.append(tool)
            requestor_by_tool[tool.name] = "assistant"
    user_include = [name for name in allowed_names if name not in assistant_names]
    user_tools = env.get_user_tools(include=user_include) if user_include else []
    for tool in user_tools:
        if tool.name in allowed and tool.name not in requestor_by_tool:
            tools.append(tool)
            requestor_by_tool[tool.name] = "user"
    return tools, requestor_by_tool


if __name__ == "__main__":
    main(json_stdout=JSON_STDOUT)
