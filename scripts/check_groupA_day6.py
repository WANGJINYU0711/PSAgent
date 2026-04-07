"""Quick checks for Day 6 Group A readiness.

This script does not change algorithm logic. It only runs small validation
checks against the current fixed-tree environment and baseline implementations.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


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
from oracle_eval import find_best_fixed_path  # noqa: E402


METHOD_FACTORIES = {
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
        raise ValueError("Derived dataset must be a JSON list of instances.")
    return data


def sample_instances(
    instances: list[dict[str, Any]],
    num_instances: int,
    seed: int,
) -> list[dict[str, Any]]:
    if num_instances <= 0:
        raise ValueError("--num-instances must be positive.")
    if num_instances >= len(instances):
        return list(instances)
    rng = random.Random(seed)
    return rng.sample(instances, k=num_instances)


def run_oracle_best_path_check(
    instances: list[dict[str, Any]],
    num_instances: int,
    seed: int,
    catalog_preset: str,
) -> dict[str, Any]:
    sampled_instances = sample_instances(instances, num_instances=num_instances, seed=seed)
    env = FixedTreeEnvironment(load_catalog(catalog_preset))

    leaf_type_counts = {"shared": 0, "unshared": 0}
    per_instance: list[dict[str, Any]] = []

    for instance in sampled_instances:
        best_path, best_result = find_best_fixed_path(instance, env)
        leaf_type_counts[best_result.leaf_type] = leaf_type_counts.get(best_result.leaf_type, 0) + 1
        per_instance.append(
            {
                "instance_id": instance.get("instance_id"),
                "original_task_id": instance.get("original_task_id"),
                "best_path": best_path,
                "best_leaf_type": best_result.leaf_type,
                "best_total_cost": best_result.total_cost,
                "best_final_action": best_result.final_action,
                "oracle_action": best_result.oracle_action,
            }
        )

    return {
        "catalog_preset": catalog_preset,
        "num_instances_checked": len(sampled_instances),
        "best_path_leaf_type_counts": leaf_type_counts,
        "per_instance": per_instance,
    }


def run_method_seed_sweep(
    instances: list[dict[str, Any]],
    methods: list[str],
    seeds: list[int],
    episodes: int,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

    for method in methods:
        if method not in METHOD_FACTORIES:
            raise KeyError(f"Unknown method in quick check: {method}")
        for seed in seeds:
            policy = METHOD_FACTORIES[method](seed)
            catalog_preset = policy.preferred_catalog_preset()
            env = FixedTreeEnvironment(load_catalog(catalog_preset))
            policy.bind_env(env)
            policy.reset()
            rng = random.Random(seed)

            logs: list[dict[str, Any]] = []
            for episode_index in range(episodes):
                instance = rng.choice(instances)
                path = policy.select_path(instance, env)
                env.reset(instance)
                result = env.run_path(path)
                policy.update(result)

                logs.append(
                    {
                        "episode_index": episode_index,
                        "instance_id": result.instance_id,
                        "leaf_type": result.leaf_type,
                        "total_cost": result.total_cost,
                        "success": result.success,
                    }
                )

            shared_count = sum(1 for log in logs if log["leaf_type"] == "shared")
            unshared_count = sum(1 for log in logs if log["leaf_type"] == "unshared")
            mean_total_cost = sum(log["total_cost"] for log in logs) / len(logs)
            success_rate = sum(1 for log in logs if log["success"]) / len(logs)

            run_summary = {
                "method": method,
                "seed": seed,
                "catalog_preset": catalog_preset,
                "episodes": episodes,
                "mean_total_cost": mean_total_cost,
                "success_rate": success_rate,
                "leaf_type_counts": {
                    "shared": shared_count,
                    "unshared": unshared_count,
                },
            }
            per_run.append(run_summary)
            by_method[method].append(run_summary)

    method_aggregate: dict[str, dict[str, Any]] = {}
    for method, rows in by_method.items():
        if not rows:
            continue
        method_aggregate[method] = {
            "num_runs": len(rows),
            "mean_total_cost": sum(row["mean_total_cost"] for row in rows) / len(rows),
            "mean_success_rate": sum(row["success_rate"] for row in rows) / len(rows),
            "mean_shared_count": sum(row["leaf_type_counts"]["shared"] for row in rows) / len(rows),
            "mean_unshared_count": sum(row["leaf_type_counts"]["unshared"] for row in rows) / len(rows),
        }

    return {
        "per_run": per_run,
        "by_method": method_aggregate,
    }


def run_endpoint_semantics_check(
    multi_seed_results: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in multi_seed_results["per_run"]:
        grouped.setdefault(row["method"], []).append(row)

    checks: dict[str, dict[str, Any]] = {}

    def summarize_status(method: str, status: str, detail: str) -> dict[str, Any]:
        return {
            "method": method,
            "status": status,
            "detail": detail,
        }

    full_share_rows = grouped.get("full_share", [])
    full_share_ok = all(row["leaf_type_counts"]["unshared"] == 0 for row in full_share_rows)
    checks["full_share"] = summarize_status(
        "full_share",
        "pass" if full_share_ok else "fail",
        "all episodes shared" if full_share_ok else "found unshared leaf in full_share",
    )

    full_unshare_rows = grouped.get("full_unshare", [])
    full_unshare_ok = all(row["leaf_type_counts"]["shared"] == 0 for row in full_unshare_rows)
    checks["full_unshare"] = summarize_status(
        "full_unshare",
        "pass" if full_unshare_ok else "fail",
        "all episodes unshared" if full_unshare_ok else "found shared leaf in full_unshare",
    )

    epsilon_rows = grouped.get("epsilon_exp3", [])
    epsilon_ok = all(
        row["catalog_preset"] == "all_unshare" and row["leaf_type_counts"]["shared"] == 0
        for row in epsilon_rows
    )
    checks["epsilon_exp3"] = summarize_status(
        "epsilon_exp3",
        "pass" if epsilon_ok else "fail",
        "all episodes unshared under all_unshare preset"
        if epsilon_ok
        else "epsilon_exp3 did not stay unshared under all_unshare preset",
    )

    naive_rows = grouped.get("naive_mixed", [])
    naive_shared = sum(row["leaf_type_counts"]["shared"] for row in naive_rows)
    naive_unshared = sum(row["leaf_type_counts"]["unshared"] for row in naive_rows)
    if naive_shared > 0 and naive_unshared > 0:
        checks["naive_mixed"] = summarize_status(
            "naive_mixed",
            "pass",
            "observed both shared and unshared leaves under mixed preset",
        )
    else:
        checks["naive_mixed"] = summarize_status(
            "naive_mixed",
            "warn",
            "mixed preset used, but this small sample did not realize both leaf types",
        )

    return checks


def render_markdown(report: dict[str, Any]) -> str:
    oracle = report["check_1_oracle_best_path_distribution"]
    sweep = report["check_2_multi_seed_groupA"]
    endpoint = report["check_3_endpoint_semantics"]

    lines: list[str] = []
    lines.append("# Group A Day 6 Quick Checks")
    lines.append("")
    lines.append("## Check 1")
    lines.append("")
    lines.append(f"- Catalog preset: `{oracle['catalog_preset']}`")
    lines.append(f"- Instances checked: `{oracle['num_instances_checked']}`")
    lines.append(f"- Best shared count: `{oracle['best_path_leaf_type_counts'].get('shared', 0)}`")
    lines.append(f"- Best unshared count: `{oracle['best_path_leaf_type_counts'].get('unshared', 0)}`")
    lines.append("")
    lines.append("## Check 2")
    lines.append("")
    lines.append("| method | seed | preset | mean_total_cost | success_rate | shared | unshared |")
    lines.append("|---|---:|---|---:|---:|---:|---:|")
    for row in sweep["per_run"]:
        lines.append(
            "| {method} | {seed} | {catalog_preset} | {mean_total_cost:.4f} | {success_rate:.4f} | {shared} | {unshared} |".format(
                method=row["method"],
                seed=row["seed"],
                catalog_preset=row["catalog_preset"],
                mean_total_cost=row["mean_total_cost"],
                success_rate=row["success_rate"],
                shared=row["leaf_type_counts"]["shared"],
                unshared=row["leaf_type_counts"]["unshared"],
            )
        )
    lines.append("")
    lines.append("## Check 3")
    lines.append("")
    for method, result in endpoint.items():
        lines.append(f"- `{method}`: `{result['status']}` - {result['detail']}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Day 6 quick checks for Group A.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "day6_checks",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--num-instances",
        type=int,
        default=15,
    )
    parser.add_argument(
        "--catalog-preset",
        type=str,
        default="mixed",
        help="Catalog preset used only for Check 1.",
    )
    args = parser.parse_args()

    instances = load_instances(args.data)
    methods = [
        "full_share",
        "full_unshare",
        "epsilon_exp3",
        "direct_multistage_exp3",
        "naive_mixed",
    ]

    report = {
        "check_1_oracle_best_path_distribution": run_oracle_best_path_check(
            instances=instances,
            num_instances=args.num_instances,
            seed=args.seeds[0] if args.seeds else 0,
            catalog_preset=args.catalog_preset,
        ),
        "check_2_multi_seed_groupA": run_method_seed_sweep(
            instances=instances,
            methods=methods,
            seeds=args.seeds,
            episodes=args.episodes,
        ),
    }
    report["check_3_endpoint_semantics"] = run_endpoint_semantics_check(
        report["check_2_multi_seed_groupA"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "groupA_day6_checks.json"
    md_path = args.output_dir / "groupA_day6_checks.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))

    oracle = report["check_1_oracle_best_path_distribution"]
    print("Check 1")
    print(
        f"  instances={oracle['num_instances_checked']} "
        f"shared_best={oracle['best_path_leaf_type_counts'].get('shared', 0)} "
        f"unshared_best={oracle['best_path_leaf_type_counts'].get('unshared', 0)}"
    )
    print("Check 2")
    for row in report["check_2_multi_seed_groupA"]["per_run"]:
        print(
            f"  {row['method']} seed={row['seed']} "
            f"cost={row['mean_total_cost']:.4f} "
            f"success={row['success_rate']:.4f} "
            f"shared={row['leaf_type_counts']['shared']} "
            f"unshared={row['leaf_type_counts']['unshared']}"
        )
    print("Check 3")
    for method, result in report["check_3_endpoint_semantics"].items():
        print(f"  {method}: {result['status']} - {result['detail']}")
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
