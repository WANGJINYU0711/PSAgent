"""Smoke test for the minimal fixed-tree environment.

Example:
    python scripts/test_env_single.py \
        --data data/derived/airline_cancellation_fixed_tree/tasks.json \
        --num-paths 5
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "envs") not in sys.path:
    sys.path.insert(0, str(ROOT / "envs"))

from fixed_tree_env import FixedTreeEnvironment, default_agent_catalog  # noqa: E402
from oracle_eval import enumerate_all_paths, find_best_fixed_path  # noqa: E402


def load_instances(path: Path) -> list[dict]:
    with path.open() as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Derived data file must contain a list of instances.")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fixed-tree env smoke tests.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
        help="Path to structured derived dataset JSON.",
    )
    parser.add_argument(
        "--num-paths",
        type=int,
        default=5,
        help="Number of random paths to evaluate on the sampled instance.",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
        help="Number of random smoke-test episodes to run after the single-instance demo.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=7,
        help="Random seed for reproducibility.",
    )
    args = parser.parse_args()

    rng = random.Random(args.seed)
    instances = load_instances(args.data)
    env = FixedTreeEnvironment(default_agent_catalog())

    sampled_instance = rng.choice(instances)
    all_paths = enumerate_all_paths(env)
    num_paths = min(args.num_paths, len(all_paths))
    sampled_paths = rng.sample(all_paths, k=num_paths)

    print("\n=== Single-instance random path runs ===")
    print(f"instance_id={sampled_instance['instance_id']}")
    print(f"original_task_id={sampled_instance.get('original_task_id')}")
    print()

    for idx, path in enumerate(sampled_paths, start=1):
        result = env.reset(sampled_instance)  # type: ignore[assignment]
        del result
        episode = env.run_path(path)
        print(f"[path {idx}]")
        print(f"  path={episode.selected_path}")
        print(f"  leaf_type={episode.leaf_type}")
        print(f"  final_action={episode.final_action}")
        print(f"  oracle_action={episode.oracle_action}")
        print(f"  total_cost={episode.total_cost:.3f}")
        print()

    best_path, best_result = find_best_fixed_path(sampled_instance, env)
    print("=== Oracle best fixed path on sampled instance ===")
    print(f"best_path={best_path}")
    print(f"best_leaf_type={best_result.leaf_type}")
    print(f"best_final_action={best_result.final_action}")
    print(f"best_oracle_action={best_result.oracle_action}")
    print(f"best_total_cost={best_result.total_cost:.3f}")
    print()

    print("=== Random smoke-test episodes ===")
    for episode_idx in range(1, args.episodes + 1):
        instance = rng.choice(instances)
        path = rng.choice(all_paths)
        env.reset(instance)
        result = env.run_path(path)
        print(
            f"[episode {episode_idx}] "
            f"instance_id={result.instance_id} "
            f"leaf_type={result.leaf_type} "
            f"final_action={result.final_action} "
            f"oracle_action={result.oracle_action} "
            f"total_cost={result.total_cost:.3f}"
        )


if __name__ == "__main__":
    main()
