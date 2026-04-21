"""Small LLM bridge for stagewise child selection."""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path
from time import perf_counter
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
    usage = message.usage if isinstance(message.usage, dict) else {}
    prompt_tokens = float(usage.get("prompt_tokens", usage.get("input_tokens", 0.0)) or 0.0)
    completion_tokens = float(
        usage.get("completion_tokens", usage.get("output_tokens", 0.0)) or 0.0
    )
    total_tokens = float(usage.get("total_tokens", 0.0) or 0.0)
    if total_tokens <= 0.0:
        total_tokens = prompt_tokens + completion_tokens
    return {
        "content": message.content,
        "tool_calls": [tc.model_dump() for tc in (message.tool_calls or [])],
        "cost": message.cost,
        "usage": message.usage,
        "generation_time_seconds": message.generation_time_seconds,
        "usage_breakdown": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
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
        started_at = perf_counter()
        assistant = llm_generate(
            model=model,
            messages=messages,
            call_name="psagent_stage_select",
            **llm_args,
        )
        round_trip_seconds = perf_counter() - started_at
        row = _assistant_step_to_dict(assistant)
        row["round_trip_seconds"] = round_trip_seconds
        llm_messages.append(row)
        messages.append(assistant)

        final_output = _extract_json(assistant.content)
        if final_output is not None:
            break
        if step_idx < max_rounds - 1:
            messages.append(
                UserMessage(
                    role="user",
                    content="Return only a valid JSON object with option_alias and optional rationale.",
                )
            )

    prompt_tokens_total = sum(
        float((row.get("usage_breakdown") or {}).get("prompt_tokens", 0.0) or 0.0)
        for row in llm_messages
    )
    completion_tokens_total = sum(
        float((row.get("usage_breakdown") or {}).get("completion_tokens", 0.0) or 0.0)
        for row in llm_messages
    )
    total_tokens_total = sum(
        float((row.get("usage_breakdown") or {}).get("total_tokens", 0.0) or 0.0)
        for row in llm_messages
    )
    api_cost_total_usd_raw = sum(float(row.get("cost") or 0.0) for row in llm_messages)
    generation_time_total_seconds = sum(
        float(row.get("generation_time_seconds") or 0.0) for row in llm_messages
    )
    llm_round_trip_total_seconds = sum(
        float(row.get("round_trip_seconds") or 0.0) for row in llm_messages
    )
    json.dump(
        {
            "llm_messages": llm_messages,
            "final_output": final_output,
            "resource_usage": {
                "llm_call_count": len(llm_messages),
                "prompt_tokens_total": prompt_tokens_total,
                "completion_tokens_total": completion_tokens_total,
                "total_tokens_total": total_tokens_total,
                "api_cost_total_usd_raw": api_cost_total_usd_raw,
                "generation_time_total_seconds": generation_time_total_seconds,
                "llm_round_trip_total_seconds": llm_round_trip_total_seconds,
            },
        },
        sys.stdout,
        ensure_ascii=False,
    )


if __name__ == "__main__":
    main()
