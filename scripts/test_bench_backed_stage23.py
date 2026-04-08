"""Smoke test for the Day 8 bench-backed Stage 2/3 executor."""

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
        agents = env.agents_by_stage[stage_name]
        path.append(rng.choice(agents).agent_id)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test bench-backed Stage 2/3 execution.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument(
        "--family-kinds",
        nargs="+",
        default=["neutral", "moderate", "strong"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-instances", type=int, default=2)
    args = parser.parse_args()

    instances = load_instances(args.data)
    rng = random.Random(args.seed)
    sampled = rng.sample(instances, k=min(args.num_instances, len(instances)))

    for family_kind in args.family_kinds:
        print(f"=== family_kind={family_kind} executor=bench_backed ===")
        env = FixedTreeEnvironment(
            agent_catalog=[],
            family_kind=family_kind,
            family_seed=args.seed,
            executor_name="bench_backed",
        )
        for instance in sampled:
            path = choose_path(env, rng)
            env.reset(instance)
            result = env.run_path(path)
            print(
                json.dumps(
                    {
                        "instance_id": result.instance_id,
                        "original_task_id": instance.get("original_task_id"),
                        "selected_path": path,
                        "leaf_type": result.leaf_type,
                        "final_action": result.final_action,
                        "oracle_action": result.oracle_action,
                        "terminal_cost": result.terminal_cost,
                        "total_cost": result.total_cost,
                        "bench_aux_eval": result.episode_log.get("bench_aux_eval", {}),
                    },
                    ensure_ascii=False,
                )
            )
            for trace in result.episode_log.get("stage_trace", []):
                if trace.get("stage_name") not in {"stage2", "stage3"}:
                    continue
                print(
                    json.dumps(
                        {
                            "stage_name": trace.get("stage_name"),
                            "agent_id": trace.get("agent_id"),
                            "source": trace.get("source"),
                            "planned_tool_calls": trace.get("planned_tool_calls"),
                            "executed_tool_calls": trace.get("executed_tool_calls"),
                            "tool_errors": trace.get("tool_errors"),
                            "output": trace.get("output"),
                        },
                        ensure_ascii=False,
                    )
                )


if __name__ == "__main__":
    main()
