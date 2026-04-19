"""LLM + real telecom tool bridge for Stage 1/2/3/4/5 execution."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TAU2_ROOT = ROOT / "tau2-bench"
TAU2_SRC = TAU2_ROOT / "src"
os.chdir(TAU2_ROOT)
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.data_model.message import SystemMessage, ToolCall, ToolMessage, UserMessage  # type: ignore  # noqa: E402
from tau2.domains.telecom.environment import get_environment, get_tasks  # type: ignore  # noqa: E402
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


def _assistant_step_to_dict(message: Any) -> dict[str, Any]:
    return {
        "content": message.content,
        "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
        "cost": message.cost,
        "usage": message.usage,
        "generation_time_seconds": message.generation_time_seconds,
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


def _execute_tool_call(env: Any, tool_call_data: dict[str, Any], fallback_requestor: str = "assistant") -> tuple[dict[str, Any], Any, bool]:
    name = str(tool_call_data.get("name", ""))
    normalized_arguments = _normalize_tool_arguments(name, dict(tool_call_data.get("arguments", {})))
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
    tool_message = env.get_response(tool_call)
    parsed_content = _parse_tool_message_content(tool_message.content)
    return tool_call.model_dump(), parsed_content, bool(tool_message.error)


def main() -> None:
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
        assistant = llm_generate(**generate_kwargs)
        llm_messages.append(_assistant_step_to_dict(assistant))
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
                        content=json.dumps(parsed_content, ensure_ascii=False) if not isinstance(parsed_content, str) else parsed_content,
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
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
