"""Unified Day 4 baseline runner for the fixed-tree environment."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "envs", ROOT / "baselines"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from agent_catalog import load_catalog  # noqa: E402
from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from fixed_tree_env import FixedTreeEnvironment  # noqa: E402
from full_share import FullSharePolicy  # noqa: E402
from full_unshare import FullUnsharePolicy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402


PolicyFactory = Callable[[int], Any]


POLICY_REGISTRY: dict[str, PolicyFactory] = {
    "full_share": lambda seed: FullSharePolicy(seed=seed),
    "all_share": lambda seed: FullSharePolicy(seed=seed),
    "full_unshare": lambda seed: FullUnsharePolicy(seed=seed),
    "all_unshare": lambda seed: FullUnsharePolicy(seed=seed),
    "naive_mixed": lambda seed: NaiveMixedPolicy(seed=seed),
    "random_path": lambda seed: RandomPathPolicy(seed=seed),
    "direct_multistage_exp3": lambda seed: DirectMultiStageExp3Policy(seed=seed),
    "epsilon_exp3": lambda seed: EpsilonExp3Policy(seed=seed),
}


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Derived dataset must be a JSON list of instances.")
    return data


def make_episode_log(
    method: str,
    seed: int,
    episode_index: int,
    instance: dict[str, Any],
    result: Any,
    policy: Any,
    catalog_preset: str,
) -> dict[str, Any]:
    metadata = instance.get("metadata", {})
    log = {
        "method": method,
        "seed": seed,
        "catalog_preset": catalog_preset,
        "episode_index": episode_index,
        "instance_id": result.instance_id,
        "original_task_id": instance.get("original_task_id"),
        "family": instance.get("family"),
        "metadata_tier": metadata.get("tier"),
        "metadata_contains_user_pressure": metadata.get("contains_user_pressure"),
        "selected_path": list(result.selected_path),
        "leaf_type": result.leaf_type,
        "update_protocol": policy.resolve_feedback_type(result),
        "final_action": result.final_action,
        "oracle_action": result.oracle_action,
        "terminal_cost": result.terminal_cost,
        "path_agent_cost": result.path_agent_cost,
        "total_cost": result.total_cost,
        "success": result.success,
    }
    if hasattr(policy, "get_last_selection_info"):
        selection_info = policy.get_last_selection_info()
        if isinstance(selection_info, dict):
            log.update(selection_info)
    return log


def summarize_logs(method: str, seed: int, logs: list[dict[str, Any]], policy: Any) -> dict[str, Any]:
    num_episodes = len(logs)
    total_cost = sum(log["total_cost"] for log in logs)
    terminal_cost = sum(log["terminal_cost"] for log in logs)
    success_count = sum(1 for log in logs if log["success"])
    shared_count = sum(1 for log in logs if log["leaf_type"] == "shared")
    unshared_count = num_episodes - shared_count
    return {
        "method": method,
        "seed": seed,
        "episodes": num_episodes,
        "mean_total_cost": (total_cost / num_episodes) if num_episodes else 0.0,
        "mean_terminal_cost": (terminal_cost / num_episodes) if num_episodes else 0.0,
        "success_rate": (success_count / num_episodes) if num_episodes else 0.0,
        "leaf_type_counts": {"shared": shared_count, "unshared": unshared_count},
        "policy_state": policy.get_state(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 4 baselines on the fixed-tree env.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument(
        "--method",
        required=True,
        choices=sorted(POLICY_REGISTRY),
        help="Baseline method name.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "day4_smoke",
    )
    parser.add_argument(
        "--catalog-preset",
        type=str,
        default="day3_default",
        help="Agent catalog preset name.",
    )
    args = parser.parse_args()

    instances = load_instances(args.data)
    if not instances:
        raise ValueError("Dataset is empty.")

    rng = random.Random(args.seed)
    policy = POLICY_REGISTRY[args.method](args.seed)
    catalog_preset = args.catalog_preset
    if catalog_preset == "day3_default":
        catalog_preset = policy.preferred_catalog_preset()
    env = FixedTreeEnvironment(load_catalog(catalog_preset))
    policy.bind_env(env)
    policy.reset()

    output_dir = args.output_dir / args.method / f"seed_{args.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs: list[dict[str, Any]] = []

    for episode_index in range(args.episodes):
        instance = rng.choice(instances)
        path = policy.select_path(instance, env)
        env.reset(instance)
        result = env.run_path(path)
        policy.update(result)
        log = make_episode_log(
            args.method,
            args.seed,
            episode_index,
            instance,
            result,
            policy,
            catalog_preset,
        )
        logs.append(log)

    summary = summarize_logs(args.method, args.seed, logs, policy)

    log_path = output_dir / "episode_logs.jsonl"
    with log_path.open("w", encoding="utf-8") as handle:
        for log in logs:
            handle.write(json.dumps(log, ensure_ascii=False) + "\n")

    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print(f"method={args.method}")
    print(f"episodes={summary['episodes']}")
    print(f"mean_total_cost={summary['mean_total_cost']:.4f}")
    print(f"success_rate={summary['success_rate']:.4f}")
    print(f"leaf_type_counts={summary['leaf_type_counts']}")
    print(f"wrote_logs={log_path}")
    print(f"wrote_summary={summary_path}")


if __name__ == "__main__":
    main()
