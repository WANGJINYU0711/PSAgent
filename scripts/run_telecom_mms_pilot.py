"""Run a small fixed telecom MMS pilot on the frozen simulated track."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


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


PolicyFactory = Callable[[int], Any]

POLICIES: dict[str, PolicyFactory] = {
    "direct_multistage_exp3": lambda seed: DirectMultiStageExp3Policy(seed=seed),
    "epsilon_exp3": lambda seed: EpsilonExp3Policy(seed=seed),
    "full_share": lambda seed: FullSharePolicy(seed=seed),
}


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Derived dataset must be a JSON list.")
    return data


def select_pilot_instances(instances: list[dict[str, Any]], per_action: int) -> list[dict[str, Any]]:
    by_action: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for instance in instances:
        action = instance["stage5"]["oracle_output"]["final_action"]
        by_action[action].append(instance)

    selected: list[dict[str, Any]] = []
    for action in ("repair_all", "repair_subset", "transfer"):
        group = sorted(
            by_action[action],
            key=lambda row: (
                row["metadata"].get("num_blockers", 0),
                row["original_task_id"],
            ),
        )
        selected.extend(group[:per_action])
    return selected


def make_episode_row(
    method: str,
    family_kind: str,
    seed: int,
    episode_index: int,
    instance: dict[str, Any],
    result: Any,
) -> dict[str, Any]:
    metadata = instance.get("metadata", {})
    episode_log = result.episode_log or {}
    return {
        "method": method,
        "family_kind": family_kind,
        "seed": seed,
        "episode_index": episode_index,
        "instance_id": instance["instance_id"],
        "original_task_id": instance["original_task_id"],
        "oracle_action": result.oracle_action,
        "final_action": result.final_action,
        "exact_match": bool(result.success),
        "terminal_penalty": float(result.terminal_cost),
        "total_cost": float(result.total_cost),
        "leaf_type": result.leaf_type,
        "selected_path": list(result.selected_path),
        "num_blockers": metadata.get("num_blockers"),
        "subset_mismatch": bool(episode_log.get("subset_mismatch", False)),
        "false_cancel_count": int(episode_log.get("false_cancel_count", 0)),
        "missed_cancel_count": int(episode_log.get("missed_cancel_count", 0)),
        "false_refuse_count": int(episode_log.get("false_refuse_count", 0)),
        "missed_refuse_count": int(episode_log.get("missed_refuse_count", 0)),
    }


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def summarize_rows(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    overall_summary: list[dict[str, Any]] = []
    by_action_summary: list[dict[str, Any]] = []

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["family_kind"])].append(row)

    for (method, family_kind), group_rows in sorted(grouped.items()):
        overall_summary.append(
            {
                "method": method,
                "family_kind": family_kind,
                "episodes": len(group_rows),
                "exact_match_mean": mean([float(r["exact_match"]) for r in group_rows]),
                "terminal_penalty_mean": mean([r["terminal_penalty"] for r in group_rows]),
                "total_cost_mean": mean([r["total_cost"] for r in group_rows]),
                "oracle_action_distribution": dict(Counter(r["oracle_action"] for r in group_rows)),
                "final_action_distribution": dict(Counter(r["final_action"] for r in group_rows)),
            }
        )

        action_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            action_groups[row["oracle_action"]].append(row)
        for oracle_action, action_rows in sorted(action_groups.items()):
            by_action_summary.append(
                {
                    "method": method,
                    "family_kind": family_kind,
                    "oracle_action": oracle_action,
                    "episodes": len(action_rows),
                    "exact_match_mean": mean([float(r["exact_match"]) for r in action_rows]),
                    "terminal_penalty_mean": mean([r["terminal_penalty"] for r in action_rows]),
                    "total_cost_mean": mean([r["total_cost"] for r in action_rows]),
                    "final_action_distribution": dict(Counter(r["final_action"] for r in action_rows)),
                }
            )

    return overall_summary, by_action_summary


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_pilot(
    instances: list[dict[str, Any]],
    methods: list[str],
    family_kinds: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for family_kind in family_kinds:
        for method in methods:
            for seed in seeds:
                env = FixedTreeEnvironment(
                    agent_catalog=[],
                    family_kind=family_kind,
                    family_seed=seed,
                    executor_name="simulated",
                )
                policy = POLICIES[method](seed)
                policy.bind_env(env)
                policy.reset()
                for episode_index, instance in enumerate(instances):
                    path = policy.select_path(instance, env)
                    env.reset(instance)
                    result = env.run_path(path)
                    policy.update(result)
                    rows.append(
                        make_episode_row(
                            method=method,
                            family_kind=family_kind,
                            seed=seed,
                            episode_index=episode_index,
                            instance=instance,
                            result=result,
                        )
                    )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a frozen telecom MMS pilot.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base" / "tasks.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "telecom_mms_pilot",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["direct_multistage_exp3", "epsilon_exp3", "full_share"],
        choices=sorted(POLICIES),
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--family-kinds", nargs="+", default=["neutral", "strong"])
    parser.add_argument("--per-action", type=int, default=3)
    args = parser.parse_args()

    instances = load_instances(args.data)
    pilot_instances = select_pilot_instances(instances, per_action=args.per_action)
    rows = run_pilot(
        instances=pilot_instances,
        methods=args.methods,
        family_kinds=args.family_kinds,
        seeds=args.seeds,
    )
    overall_summary, by_action_summary = summarize_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    freeze_config = {
        "dataset": str(args.data),
        "executor_name": "simulated",
        "methods": args.methods,
        "family_kinds": args.family_kinds,
        "seeds": args.seeds,
        "per_action": args.per_action,
        "selected_instances": [row["original_task_id"] for row in pilot_instances],
        "terminal_action_distribution": dict(
            Counter(instance["stage5"]["oracle_output"]["final_action"] for instance in pilot_instances)
        ),
    }
    write_json(args.output_dir / "pilot_config.json", freeze_config)
    write_json(args.output_dir / "pilot_split.json", pilot_instances)
    write_jsonl(args.output_dir / "episode_logs.jsonl", rows)
    write_json(args.output_dir / "overall_summary.json", overall_summary)
    write_json(args.output_dir / "by_action_summary.json", by_action_summary)
    write_csv(args.output_dir / "overall_summary.csv", overall_summary)
    write_csv(args.output_dir / "by_action_summary.csv", by_action_summary)

    print(json.dumps({
        "episodes": len(rows),
        "pilot_instances": len(pilot_instances),
        "output_dir": str(args.output_dir),
        "methods": args.methods,
        "family_kinds": args.family_kinds,
        "seeds": args.seeds,
    }, indent=2))


if __name__ == "__main__":
    main()
