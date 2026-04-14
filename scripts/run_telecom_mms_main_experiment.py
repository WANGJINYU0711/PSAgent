"""Run the first-pass full-base telecom MMS main experiment."""

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
from full_unshare import FullUnsharePolicy  # noqa: E402
from mechanism_utils import choose_path_with_mechanism  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from oracle_eval import find_best_stationary_path  # noqa: E402
from oracle_policy import OraclePolicy  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402


PolicyFactory = Callable[[int], Any]

POLICIES: dict[str, PolicyFactory] = {
    "risky_ps": lambda seed: RiskyPSPolicy(seed=seed),
    "direct_multistage_exp3": lambda seed: DirectMultiStageExp3Policy(seed=seed),
    "epsilon_exp3": lambda seed: EpsilonExp3Policy(seed=seed),
    "full_share": lambda seed: FullSharePolicy(seed=seed),
    "full_unshare": lambda seed: FullUnsharePolicy(seed=seed),
    "naive_mixed": lambda seed: NaiveMixedPolicy(seed=seed),
    "random_path": lambda seed: RandomPathPolicy(seed=seed),
    "oracle": lambda seed: OraclePolicy(seed=seed),
}

MECHANISMS = ["algorithm_direct", "theta_guided_agent", "agent_only"]


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Derived dataset must be a JSON list.")
    return data


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


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


def compute_stationary_oracle_by_run(
    instances: list[dict[str, Any]],
    family_kinds: list[str],
    seeds: list[int],
) -> dict[tuple[str, int], dict[str, Any]]:
    summaries: dict[tuple[str, int], dict[str, Any]] = {}
    for family_kind in family_kinds:
        for seed in seeds:
            env = FixedTreeEnvironment(
                agent_catalog=[],
                family_kind=family_kind,
                family_seed=seed,
                executor_name="simulated",
            )
            best_path, summary = find_best_stationary_path(instances, env)
            oracle_summary = dict(summary)
            oracle_summary["path"] = list(best_path)
            oracle_summary["episode_total_costs_by_instance_id"] = {
                instance["instance_id"]: cost
                for instance, cost in zip(instances, oracle_summary["episode_total_costs"])
            }
            summaries[(family_kind, seed)] = oracle_summary
    return summaries


def make_episode_row(
    method: str,
    mechanism: str,
    family_kind: str,
    seed: int,
    episode_index: int,
    instance: dict[str, Any],
    result: Any,
    selection_meta: dict[str, Any],
) -> dict[str, Any]:
    metadata = instance.get("metadata", {})
    episode_log = result.episode_log or {}
    return {
        "method": method,
        "mechanism": mechanism,
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
        "selection_signal_summary": selection_meta.get("selection_signal_summary"),
    }


def run_main_experiment(
    instances: list[dict[str, Any]],
    methods: list[str],
    mechanisms: list[str],
    family_kinds: list[str],
    seeds: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for mechanism in mechanisms:
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
                        path, selection_meta, should_update = choose_path_with_mechanism(
                            policy,
                            instance,
                            env,
                            mechanism,
                        )
                        env.reset(instance)
                        result = env.run_path(path)
                        if should_update:
                            policy.update(result)
                        rows.append(
                            make_episode_row(
                                method=method,
                                mechanism=mechanism,
                                family_kind=family_kind,
                                seed=seed,
                                episode_index=episode_index,
                                instance=instance,
                                result=result,
                                selection_meta=selection_meta,
                            )
                        )
    return rows


def summarize_rows(
    rows: list[dict[str, Any]],
    stationary_oracle_by_run: dict[tuple[str, int], dict[str, Any]],
    group_fields: list[str],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped[key].append(row)

    summary_rows: list[dict[str, Any]] = []
    for key, group_rows in sorted(grouped.items()):
        summary_row = {field: value for field, value in zip(group_fields, key)}
        cumulative_total_cost = sum(r["total_cost"] for r in group_rows)
        oracle_cumulative_total_cost = sum(
            stationary_oracle_by_run[(r["family_kind"], r["seed"])]["episode_total_costs_by_instance_id"][r["instance_id"]]
            for r in group_rows
        )
        cumulative_regret = cumulative_total_cost - oracle_cumulative_total_cost
        summary_row.update(
            {
                "episodes": len(group_rows),
                "exact_match_mean": mean([float(r["exact_match"]) for r in group_rows]),
                "terminal_penalty_mean": mean([r["terminal_penalty"] for r in group_rows]),
                "total_cost_mean": mean([r["total_cost"] for r in group_rows]),
                "algorithm_cumulative_total_cost": cumulative_total_cost,
                "oracle_stationary_total_cost": oracle_cumulative_total_cost,
                "cumulative_regret": cumulative_regret,
                "mean_regret": cumulative_regret / len(group_rows),
                "oracle_action_distribution": dict(Counter(r["oracle_action"] for r in group_rows)),
                "final_action_distribution": dict(Counter(r["final_action"] for r in group_rows)),
            }
        )
        summary_rows.append(summary_row)
    return summary_rows


def summarize_by_action(
    rows: list[dict[str, Any]],
    stationary_oracle_by_run: dict[tuple[str, int], dict[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["mechanism"], row["family_kind"], row["oracle_action"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, mechanism, family_kind, oracle_action), group_rows in sorted(grouped.items()):
        cumulative_total_cost = sum(r["total_cost"] for r in group_rows)
        oracle_cumulative_total_cost = sum(
            stationary_oracle_by_run[(r["family_kind"], r["seed"])]["episode_total_costs_by_instance_id"][r["instance_id"]]
            for r in group_rows
        )
        cumulative_regret = cumulative_total_cost - oracle_cumulative_total_cost
        summary_rows.append(
            {
                "method": method,
                "mechanism": mechanism,
                "family_kind": family_kind,
                "oracle_action": oracle_action,
                "episodes": len(group_rows),
                "exact_match_mean": mean([float(r["exact_match"]) for r in group_rows]),
                "terminal_penalty_mean": mean([r["terminal_penalty"] for r in group_rows]),
                "total_cost_mean": mean([r["total_cost"] for r in group_rows]),
                "oracle_stationary_total_cost": oracle_cumulative_total_cost,
                "cumulative_regret": cumulative_regret,
                "mean_regret": cumulative_regret / len(group_rows),
                "final_action_distribution": dict(Counter(r["final_action"] for r in group_rows)),
            }
        )
    return summary_rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the telecom MMS full-base main experiment.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base" / "tasks.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "telecom_mms_main_experiment",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "risky_ps",
            "direct_multistage_exp3",
            "epsilon_exp3",
            "full_share",
            "full_unshare",
            "naive_mixed",
            "random_path",
            "oracle",
        ],
        choices=sorted(POLICIES),
    )
    parser.add_argument(
        "--mechanisms",
        nargs="+",
        default=MECHANISMS,
        choices=MECHANISMS,
    )
    parser.add_argument("--family-kinds", nargs="+", default=["neutral", "moderate", "strong"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    args = parser.parse_args()

    instances = load_instances(args.data)
    stationary_oracle_by_run = compute_stationary_oracle_by_run(
        instances=instances,
        family_kinds=args.family_kinds,
        seeds=args.seeds,
    )
    rows = run_main_experiment(
        instances=instances,
        methods=args.methods,
        mechanisms=args.mechanisms,
        family_kinds=args.family_kinds,
        seeds=args.seeds,
    )
    for row in rows:
        oracle_run = stationary_oracle_by_run[(row["family_kind"], row["seed"])]
        oracle_episode_cost = oracle_run["episode_total_costs_by_instance_id"][row["instance_id"]]
        row["oracle_stationary_episode_cost"] = oracle_episode_cost
        row["episode_regret"] = row["total_cost"] - oracle_episode_cost

    overall_summary = summarize_rows(
        rows,
        stationary_oracle_by_run,
        group_fields=["method", "mechanism", "family_kind"],
    )
    by_action_summary = summarize_by_action(rows, stationary_oracle_by_run)
    by_mechanism_summary = summarize_rows(
        rows,
        stationary_oracle_by_run,
        group_fields=["method", "mechanism"],
    )
    by_family_summary = summarize_rows(
        rows,
        stationary_oracle_by_run,
        group_fields=["method", "family_kind"],
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    experiment_config = {
        "dataset": str(args.data),
        "executor_name": "simulated",
        "methods": args.methods,
        "mechanisms": args.mechanisms,
        "family_kinds": args.family_kinds,
        "seeds": args.seeds,
        "num_instances": len(instances),
        "terminal_action_distribution": dict(
            Counter(instance["stage5"]["oracle_output"]["final_action"] for instance in instances)
        ),
        "regret_definition": "algorithm_cumulative_total_cost - stationary_oracle_cumulative_total_cost",
        "stationary_oracle_note": "formal regret reference uses one fixed path per family_kind/seed run",
        "per_instance_oracle_note": "oracle baseline is a method, not the formal regret comparator",
    }
    write_json(args.output_dir / "experiment_config.json", experiment_config)
    write_jsonl(args.output_dir / "episode_logs.jsonl", rows)
    write_json(
        args.output_dir / "stationary_oracle_summary.json",
        {
            f"{family_kind}::seed_{seed}": {
                "path": summary["path"],
                "cumulative_total_cost": summary["cumulative_total_cost"],
                "mean_total_cost": summary["mean_total_cost"],
            }
            for (family_kind, seed), summary in stationary_oracle_by_run.items()
        },
    )
    write_json(args.output_dir / "overall_summary.json", overall_summary)
    write_json(args.output_dir / "by_action_summary.json", by_action_summary)
    write_json(args.output_dir / "by_mechanism_summary.json", by_mechanism_summary)
    write_json(args.output_dir / "by_family_summary.json", by_family_summary)
    write_csv(args.output_dir / "overall_summary.csv", overall_summary)
    write_csv(args.output_dir / "by_action_summary.csv", by_action_summary)
    write_csv(args.output_dir / "by_mechanism_summary.csv", by_mechanism_summary)
    write_csv(args.output_dir / "by_family_summary.csv", by_family_summary)

    print(
        json.dumps(
            {
                "episodes": len(rows),
                "instances": len(instances),
                "methods": args.methods,
                "mechanisms": args.mechanisms,
                "family_kinds": args.family_kinds,
                "seeds": args.seeds,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
