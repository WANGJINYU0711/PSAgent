"""LLM + real telecom tool bridge for Stage 2/3 execution."""

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

    env = get_environment(policy_type="workflow")
    task_map = _load_task_map()
    task = task_map.get(original_task_id)
    if task is not None and getattr(task, "initial_state", None) is not None:
        init_actions = getattr(task.initial_state, "initialization_actions", None)
        env.set_state(None, init_actions, [])

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
        assistant = llm_generate(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            call_name=f"psagent_{stage_name}_telecom_llm_bench",
            **llm_args,
        )
        llm_messages.append(_assistant_step_to_dict(assistant))
        messages.append(assistant)

        if assistant.tool_calls:
            for tool_call in assistant.tool_calls:
                normalized_arguments = _normalize_tool_arguments(tool_call.name, tool_call.arguments)
                tool_call = ToolCall(
                    id=tool_call.id,
                    name=tool_call.name,
                    arguments=normalized_arguments,
                    requestor=requestor_by_tool.get(tool_call.name, "assistant"),
                )
                tool_message = env.get_response(tool_call)
                parsed_content = _parse_tool_message_content(tool_message.content)
                executed_tool_calls.append(tool_call.model_dump())
                tool_results.append(parsed_content)
                if tool_message.error:
                    tool_errors.append(
                        {
                            "tool_call": tool_call.model_dump(),
                            "content": parsed_content,
                        }
                    )
                messages.append(
                    ToolMessage(
                        id=tool_message.id,
                        role="tool",
                        content=tool_message.content,
                        requestor=tool_message.requestor,
                        error=tool_message.error,
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
        "db_hash_before": db_hash_before,
        "db_hash_after": env.get_db_hash(),
        "llm_messages": llm_messages,
        "executed_tool_calls": executed_tool_calls,
        "tool_results": tool_results,
        "tool_errors": tool_errors,
        "final_output": final_output,
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
