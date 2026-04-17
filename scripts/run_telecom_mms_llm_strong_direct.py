"""Run the telecom MMS strong/direct slice on the llm_bench executor."""

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
from oracle_eval import find_best_stationary_path  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402


PolicyFactory = Callable[[int], Any]

POLICIES: dict[str, PolicyFactory] = {
    "risky_ps": lambda seed: RiskyPSPolicy(seed=seed),
    "direct_multistage_exp3": lambda seed: DirectMultiStageExp3Policy(seed=seed),
    "epsilon_exp3": lambda seed: EpsilonExp3Policy(seed=seed),
    "full_share": lambda seed: FullSharePolicy(seed=seed),
    "full_unshare": lambda seed: FullUnsharePolicy(seed=seed),
}


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


def compute_stationary_oracle(
    instances: list[dict[str, Any]],
    family_kind: str,
    seed: int,
) -> dict[str, Any]:
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
    return oracle_summary


def _llm_stage_names_from_trace(stage_trace: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for row in stage_trace:
        if row.get("source") == "llm_bench":
            names.append(row["stage_name"])
    return names


def _llm_call_count_from_trace(stage_trace: list[dict[str, Any]]) -> int:
    count = 0
    for row in stage_trace:
        if row.get("source") != "llm_bench":
            continue
        count += len(row.get("llm_raw_output", []))
    return count


def make_episode_row(
    method: str,
    family_kind: str,
    seed: int,
    instance: dict[str, Any],
    episode_index: int,
    result: Any,
) -> dict[str, Any]:
    episode_log = result.episode_log or {}
    stage_trace = episode_log.get("stage_trace", [])
    return {
        "method": method,
        "mechanism": "algorithm_direct",
        "executor_name": "llm_bench",
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
        "llm_stage_names": _llm_stage_names_from_trace(stage_trace),
        "llm_call_count": _llm_call_count_from_trace(stage_trace),
        "bench_aux_eval": episode_log.get("bench_aux_eval"),
        "stage_trace": stage_trace,
    }


def run_slice(
    instances: list[dict[str, Any]],
    methods: list[str],
    family_kind: str,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method in methods:
        env = FixedTreeEnvironment(
            agent_catalog=[],
            family_kind=family_kind,
            family_seed=seed,
            executor_name="llm_bench",
        )
        policy = POLICIES[method](seed)
        policy.bind_env(env)
        policy.reset()
        for episode_index, instance in enumerate(instances):
            path, _selection_meta, should_update = choose_path_with_mechanism(
                policy,
                instance,
                env,
                "algorithm_direct",
            )
            env.reset(instance)
            result = env.run_path(path)
            if should_update:
                policy.update(result)
            rows.append(
                make_episode_row(
                    method=method,
                    family_kind=family_kind,
                    seed=seed,
                    instance=instance,
                    episode_index=episode_index,
                    result=result,
                )
            )
    return rows


def summarize_overall(rows: list[dict[str, Any]], stationary_oracle: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["method"]].append(row)

    summary_rows: list[dict[str, Any]] = []
    for method, group_rows in sorted(grouped.items()):
        cumulative_total_cost = sum(r["total_cost"] for r in group_rows)
        oracle_cumulative_total_cost = sum(
            stationary_oracle["episode_total_costs_by_instance_id"][r["instance_id"]]
            for r in group_rows
        )
        cumulative_regret = cumulative_total_cost - oracle_cumulative_total_cost
        summary_rows.append(
            {
                "method": method,
                "mechanism": "algorithm_direct",
                "executor_name": "llm_bench",
                "family_kind": "strong",
                "seed": group_rows[0]["seed"],
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
                "llm_stage_names": ["stage2", "stage3"],
                "mean_llm_call_count": mean([r["llm_call_count"] for r in group_rows]),
            }
        )
    return summary_rows


def summarize_by_action(rows: list[dict[str, Any]], stationary_oracle: dict[str, Any]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["method"], row["oracle_action"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (method, oracle_action), group_rows in sorted(grouped.items()):
        cumulative_total_cost = sum(r["total_cost"] for r in group_rows)
        oracle_cumulative_total_cost = sum(
            stationary_oracle["episode_total_costs_by_instance_id"][r["instance_id"]]
            for r in group_rows
        )
        cumulative_regret = cumulative_total_cost - oracle_cumulative_total_cost
        summary_rows.append(
            {
                "method": method,
                "mechanism": "algorithm_direct",
                "executor_name": "llm_bench",
                "family_kind": "strong",
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
    parser = argparse.ArgumentParser(description="Run the telecom MMS llm_bench strong/direct slice.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base" / "tasks.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "telecom_mms_llm_strong_direct",
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
        ],
        choices=sorted(POLICIES),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    instances = load_instances(args.data)
    stationary_oracle = compute_stationary_oracle(instances, family_kind="strong", seed=args.seed)
    rows = run_slice(instances, args.methods, family_kind="strong", seed=args.seed)
    for row in rows:
        oracle_episode_cost = stationary_oracle["episode_total_costs_by_instance_id"][row["instance_id"]]
        row["oracle_stationary_episode_cost"] = oracle_episode_cost
        row["episode_regret"] = row["total_cost"] - oracle_episode_cost

    overall_summary = summarize_overall(rows, stationary_oracle)
    by_action_summary = summarize_by_action(rows, stationary_oracle)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / "experiment_config.json",
        {
            "dataset": str(args.data),
            "executor_name": "llm_bench",
            "llm_stages_only": ["stage2", "stage3"],
            "non_llm_stages": ["stage1", "stage4", "stage5"],
            "methods": args.methods,
            "family_kind": "strong",
            "mechanism": "algorithm_direct",
            "seed": args.seed,
            "num_instances": len(instances),
            "terminal_action_distribution": dict(
                Counter(instance["stage5"]["oracle_output"]["final_action"] for instance in instances)
            ),
            "regret_definition": "algorithm_cumulative_total_cost - stationary_oracle_cumulative_total_cost",
            "stationary_oracle_executor_note": "stationary oracle comparator remains the frozen simulated-track reference",
        },
    )
    write_jsonl(args.output_dir / "episode_logs.jsonl", rows)
    write_json(
        args.output_dir / "stationary_oracle_summary.json",
        {
            "family_kind": "strong",
            "seed": args.seed,
            "executor_name": "simulated",
            "path": stationary_oracle["path"],
            "cumulative_total_cost": stationary_oracle["cumulative_total_cost"],
            "mean_total_cost": stationary_oracle["mean_total_cost"],
        },
    )
    write_json(args.output_dir / "overall_summary.json", overall_summary)
    write_json(args.output_dir / "by_action_summary.json", by_action_summary)
    write_csv(args.output_dir / "overall_summary.csv", overall_summary)
    write_csv(args.output_dir / "by_action_summary.csv", by_action_summary)

    print(
        json.dumps(
            {
                "episodes": len(rows),
                "instances": len(instances),
                "methods": args.methods,
                "executor_name": "llm_bench",
                "family_kind": "strong",
                "mechanism": "algorithm_direct",
                "seed": args.seed,
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
