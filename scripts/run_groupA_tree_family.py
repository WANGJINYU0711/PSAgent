"""Run Group A baselines on neutral / moderate / strong tree families."""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
    ROOT / "baselines",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from fixed_tree_env import FixedTreeEnvironment  # noqa: E402
from full_share import FullSharePolicy  # noqa: E402
from full_unshare import FullUnsharePolicy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402


POLICIES = {
    "full_share": lambda seed: FullSharePolicy(seed=seed),
    "full_unshare": lambda seed: FullUnsharePolicy(seed=seed),
    "epsilon_exp3": lambda seed: EpsilonExp3Policy(seed=seed),
    "direct_multistage_exp3": lambda seed: DirectMultiStageExp3Policy(seed=seed),
    "naive_mixed": lambda seed: NaiveMixedPolicy(seed=seed),
}


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Derived dataset must be a JSON list.")
    return data


def run_one(
    instances: list[dict[str, Any]],
    method: str,
    family_kind: str,
    seed: int,
) -> dict[str, Any]:
    env = FixedTreeEnvironment(agent_catalog=[], family_kind=family_kind, family_seed=seed)
    policy = POLICIES[method](seed)
    policy.bind_env(env)
    policy.reset()
    rng = random.Random(seed)

    logs: list[dict[str, Any]] = []
    for _ in range(len(instances)):
        instance = rng.choice(instances)
        path = policy.select_path(instance, env)
        env.reset(instance)
        result = env.run_path(path)
        policy.update(result)
        logs.append(result.episode_log)

    count = len(logs)
    return {
        "family_kind": family_kind,
        "method": method,
        "seed": seed,
        "episodes": count,
        "mean_total_cost": sum(log["total_cost"] for log in logs) / count,
        "mean_terminal_penalty": sum(log["terminal_cost"] for log in logs) / count,
        "mean_path_agent_cost": sum(log["path_agent_cost"] for log in logs) / count,
        "exact_match_rate": sum(1 for log in logs if log["success"]) / count,
        "subset_mismatch_rate": sum(1 for log in logs if log.get("subset_mismatch")) / count,
        "mean_false_cancel_count": sum(log.get("false_cancel_count", 0) for log in logs) / count,
        "mean_missed_cancel_count": sum(log.get("missed_cancel_count", 0) for log in logs) / count,
        "mean_false_refuse_count": sum(log.get("false_refuse_count", 0) for log in logs) / count,
        "mean_missed_refuse_count": sum(log.get("missed_refuse_count", 0) for log in logs) / count,
        "shared_leaf_ratio": sum(1 for log in logs if log["leaf_type"] == "shared") / count,
        "unshared_leaf_ratio": sum(1 for log in logs if log["leaf_type"] == "unshared") / count,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Group A baselines across tree families.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "tree_family_groupA",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--family-kinds",
        nargs="+",
        default=["neutral", "moderate", "strong"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=list(POLICIES),
    )
    args = parser.parse_args()

    instances = load_instances(args.data)
    rows: list[dict[str, Any]] = []
    for family_kind in args.family_kinds:
        for method in args.methods:
            for seed in args.seeds:
                rows.append(run_one(instances, method, family_kind, seed))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "results.json"
    csv_path = args.output_dir / "results.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)

    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
