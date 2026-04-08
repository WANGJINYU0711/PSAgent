"""Smoke test for LLM-backed Stage 2/3 on the airline derived dataset."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fixed_tree_env import FixedTreeEnvironment  # noqa: E402


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def choose_path(env: FixedTreeEnvironment, rng: random.Random) -> list[str]:
    path: list[str] = []
    for stage_name in env.STAGE_NAMES:
        path.append(rng.choice(env.agents_by_stage[stage_name]).agent_id)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test llm_bench Stage 2/3.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "llm_bench_stage23_smoke.jsonl",
    )
    parser.add_argument(
        "--family-kinds",
        nargs="+",
        default=["neutral", "moderate", "strong"],
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    instances = load_instances(args.data)
    sampled = rng.sample(instances, k=min(args.num_instances, len(instances)))
    records: list[dict[str, Any]] = []

    for family_kind in args.family_kinds:
        print(f"=== family_kind={family_kind} executor=llm_bench ===")
        env = FixedTreeEnvironment(
            agent_catalog=[],
            family_kind=family_kind,
            family_seed=args.seed,
            executor_name="llm_bench",
        )
        for instance in sampled:
            path = choose_path(env, rng)
            env.reset(instance)
            result = env.run_path(path)
            header = {
                "family_kind": family_kind,
                "instance_id": result.instance_id,
                "original_task_id": instance.get("original_task_id"),
                "selected_path": path,
                "leaf_type": result.leaf_type,
                "final_action": result.final_action,
                "oracle_action": result.oracle_action,
                "terminal_cost": result.terminal_cost,
                "total_cost": result.total_cost,
                "bench_aux_eval": result.episode_log.get("bench_aux_eval", {}),
            }
            print(json.dumps(header, ensure_ascii=False))
            record = {"header": header, "stage_traces": []}
            for trace in result.episode_log.get("stage_trace", []):
                if trace.get("stage_name") not in {"stage2", "stage3"}:
                    continue
                stage_row = {
                    "stage_name": trace.get("stage_name"),
                    "agent_id": trace.get("agent_id"),
                    "prompt_summary": trace.get("prompt_summary"),
                    "source": trace.get("source"),
                    "llm_raw_output": trace.get("llm_raw_output"),
                    "executed_tool_calls": trace.get("executed_tool_calls"),
                    "tool_errors": trace.get("tool_errors"),
                    "output": trace.get("output"),
                }
                print(json.dumps(stage_row, ensure_ascii=False))
                record["stage_traces"].append(stage_row)
            records.append(record)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote_output={args.output}")


if __name__ == "__main__":
    main()
