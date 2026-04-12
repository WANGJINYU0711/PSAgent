"""Run telecom bench tool calls inside tau2-bench's virtualenv."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TAU2_SRC = ROOT / "tau2-bench" / "src"
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.data_model.message import ToolCall  # type: ignore  # noqa: E402
from tau2.domains.telecom.environment import get_environment, get_tasks  # type: ignore  # noqa: E402


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


def main() -> None:
    payload = json.load(sys.stdin)
    original_task_id = str(payload.get("original_task_id", ""))
    tool_calls = payload.get("tool_calls", []) or []

    env = get_environment(policy_type="workflow")
    task_map = _load_task_map()
    task = task_map.get(original_task_id)
    if task is not None and getattr(task, "initial_state", None) is not None:
        init_actions = getattr(task.initial_state, "initialization_actions", None)
        env.set_state(None, init_actions, [])

    db_hash_before = env.get_db_hash()
    responses: list[dict[str, Any]] = []
    for row in tool_calls:
        tool_call = ToolCall(
            id=str(row.get("id", "")),
            name=str(row["name"]),
            arguments=dict(row.get("arguments", {})),
            requestor=str(row.get("requestor", "assistant")),
        )
        tool_message = env.get_response(tool_call)
        responses.append(
            {
                "tool_call": {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": tool_call.arguments,
                    "requestor": tool_call.requestor,
                },
                "error": bool(tool_message.error),
                "content": _parse_tool_message_content(tool_message.content),
            }
        )

    result = {
        "original_task_id": original_task_id,
        "db_hash_before": db_hash_before,
        "db_hash_after": env.get_db_hash(),
        "responses": responses,
    }
    json.dump(result, sys.stdout, ensure_ascii=False)


if __name__ == "__main__":
    main()
