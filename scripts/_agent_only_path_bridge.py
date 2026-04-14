"""Small LLM bridge for agent-only path self-selection."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
TAU2_ROOT = ROOT / "tau2-bench"
TAU2_SRC = TAU2_ROOT / "src"
os.chdir(TAU2_ROOT)
if str(TAU2_SRC) not in sys.path:
    sys.path.insert(0, str(TAU2_SRC))

from tau2.data_model.message import SystemMessage, UserMessage  # type: ignore  # noqa: E402
from tau2.utils.llm_utils import generate as llm_generate  # type: ignore  # noqa: E402


def _extract_codeblock(text: str) -> str | None:
    match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    return match.group(1) if match else None


def _extract_braced_json(text: str) -> str | None:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    return text[start : end + 1]


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


def _assistant_step_to_dict(message: Any) -> dict[str, Any]:
    return {
        "content": message.content,
        "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
        "cost": message.cost,
        "usage": message.usage,
        "generation_time_seconds": message.generation_time_seconds,
    }


def main() -> None:
    payload = json.load(sys.stdin)
    model = payload["model"]
    llm_args = dict(payload.get("llm_args", {}))
    system_prompt = payload["system_prompt"]
    user_prompt = payload["user_prompt"]
    max_rounds = int(payload.get("max_rounds", 2))

    messages: list[Any] = [
        SystemMessage(role="system", content=system_prompt),
        UserMessage(role="user", content=user_prompt),
    ]
    llm_messages: list[dict[str, Any]] = []
    final_output: dict[str, Any] | None = None

    for step_idx in range(max_rounds):
        assistant = llm_generate(
            model=model,
            messages=messages,
            call_name="psagent_agent_only_path_select",
            **llm_args,
        )
        llm_messages.append(_assistant_step_to_dict(assistant))
        messages.append(assistant)

        final_output = _extract_json(assistant.content)
        if final_output is not None:
            break
        if step_idx < max_rounds - 1:
            messages.append(
                UserMessage(
                    role="user",
                    content="Return only a valid JSON object with path_id and optional rationale.",
                )
            )

    json.dump(
        {
            "llm_messages": llm_messages,
            "final_output": final_output,
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
