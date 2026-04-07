"""Enhanced Group A quick checks under evaluator v2."""

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


def _empty_metric_accumulator() -> dict[str, Any]:
    return {
        "count": 0,
        "shared": 0,
        "unshared": 0,
        "sum_total_cost": 0.0,
        "sum_terminal_penalty": 0.0,
        "sum_false_cancel_count": 0.0,
        "sum_missed_cancel_count": 0.0,
        "sum_false_refuse_count": 0.0,
        "sum_missed_refuse_count": 0.0,
        "subset_mismatch_count": 0,
        "exact_match_count": 0,
        "sum_path_agent_cost": 0.0,
    }


def _update_metric_accumulator(acc: dict[str, Any], result: Any, episode_log: dict[str, Any]) -> None:
    acc["count"] += 1
    if result.leaf_type == "shared":
        acc["shared"] += 1
    elif result.leaf_type == "unshared":
        acc["unshared"] += 1
    acc["sum_total_cost"] += float(result.total_cost)
    acc["sum_terminal_penalty"] += float(result.terminal_cost)
    acc["sum_false_cancel_count"] += float(episode_log.get("false_cancel_count", 0))
    acc["sum_missed_cancel_count"] += float(episode_log.get("missed_cancel_count", 0))
    acc["sum_false_refuse_count"] += float(episode_log.get("false_refuse_count", 0))
    acc["sum_missed_refuse_count"] += float(episode_log.get("missed_refuse_count", 0))
    acc["subset_mismatch_count"] += int(bool(episode_log.get("subset_mismatch", False)))
    acc["exact_match_count"] += int(bool(result.success))
    acc["sum_path_agent_cost"] += float(result.path_agent_cost)


def _finalize_metric_accumulator(acc: dict[str, Any]) -> dict[str, Any]:
    count = max(1, acc["count"])
    return {
        "num_examples": acc["count"],
        "leaf_type_counts": {"shared": acc["shared"], "unshared": acc["unshared"]},
        "mean_total_cost": acc["sum_total_cost"] / count,
        "mean_terminal_penalty": acc["sum_terminal_penalty"] / count,
        "mean_false_cancel_count": acc["sum_false_cancel_count"] / count,
        "mean_missed_cancel_count": acc["sum_missed_cancel_count"] / count,
        "mean_false_refuse_count": acc["sum_false_refuse_count"] / count,
        "mean_missed_refuse_count": acc["sum_missed_refuse_count"] / count,
        "subset_mismatch_rate": acc["subset_mismatch_count"] / count,
        "exact_match_rate": acc["exact_match_count"] / count,
        "mean_path_agent_cost": acc["sum_path_agent_cost"] / count,
    }


def run_oracle_best_path_distribution(
    instances: list[dict[str, Any]],
    num_instances: int,
    seed: int,
    catalog_preset: str,
) -> dict[str, Any]:
    sampled_instances = sample_instances(instances, num_instances=num_instances, seed=seed)
    env = FixedTreeEnvironment(load_catalog(catalog_preset))

    overall = _empty_metric_accumulator()
    by_tier_raw: dict[str, dict[str, Any]] = {}
    per_instance: list[dict[str, Any]] = []

    for instance in sampled_instances:
        best_path, best_result = find_best_fixed_path(instance, env)
        episode_log = best_result.episode_log if isinstance(best_result.episode_log, dict) else {}
        tier = instance.get("metadata", {}).get("tier", "unknown")
        _update_metric_accumulator(overall, best_result, episode_log)
        by_tier_raw.setdefault(tier, _empty_metric_accumulator())
        _update_metric_accumulator(by_tier_raw[tier], best_result, episode_log)
        per_instance.append(
            {
                "instance_id": instance.get("instance_id"),
                "original_task_id": instance.get("original_task_id"),
                "tier": tier,
                "best_path": best_path,
                "best_leaf_type": best_result.leaf_type,
                "best_total_cost": best_result.total_cost,
                "best_terminal_penalty": best_result.terminal_cost,
                "cost_breakdown": episode_log.get("cost_breakdown", {}),
                "exact_match": best_result.success,
            }
        )

    summary = _finalize_metric_accumulator(overall)
    summary["catalog_preset"] = catalog_preset
    summary["num_instances_checked"] = len(sampled_instances)
    summary["best_path_leaf_type_counts"] = summary.pop("leaf_type_counts")
    summary["mean_best_total_cost"] = summary.pop("mean_total_cost")
    summary["mean_best_terminal_penalty"] = summary.pop("mean_terminal_penalty")
    summary["mean_best_path_agent_cost"] = summary.pop("mean_path_agent_cost")

    by_tier = {}
    for tier, raw in by_tier_raw.items():
        tier_summary = _finalize_metric_accumulator(raw)
        tier_summary["best_path_leaf_type_counts"] = tier_summary.pop("leaf_type_counts")
        tier_summary["mean_best_total_cost"] = tier_summary.pop("mean_total_cost")
        tier_summary["mean_best_terminal_penalty"] = tier_summary.pop("mean_terminal_penalty")
        tier_summary["mean_best_path_agent_cost"] = tier_summary.pop("mean_path_agent_cost")
        by_tier[tier] = tier_summary

    return {
        **summary,
        "by_tier": by_tier,
        "per_instance": per_instance,
    }


def run_method_seed_sweep(
    instances: list[dict[str, Any]],
    methods: list[str],
    seeds: list[int],
    episodes: int,
) -> dict[str, Any]:
    per_run: list[dict[str, Any]] = []
    by_method_rows: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}

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

            overall = _empty_metric_accumulator()
            by_tier_raw: dict[str, dict[str, Any]] = {}

            for _ in range(episodes):
                instance = rng.choice(instances)
                path = policy.select_path(instance, env)
                env.reset(instance)
                result = env.run_path(path)
                policy.update(result)
                tier = instance.get("metadata", {}).get("tier", "unknown")
                episode_log = result.episode_log if isinstance(result.episode_log, dict) else {}
                _update_metric_accumulator(overall, result, episode_log)
                by_tier_raw.setdefault(tier, _empty_metric_accumulator())
                _update_metric_accumulator(by_tier_raw[tier], result, episode_log)

            run_summary = _finalize_metric_accumulator(overall)
            run_summary.update(
                {
                    "method": method,
                    "seed": seed,
                    "catalog_preset": catalog_preset,
                    "episodes": episodes,
                    "success_rate": run_summary.pop("exact_match_rate"),
                }
            )
            by_tier = {}
            for tier, raw in by_tier_raw.items():
                tier_summary = _finalize_metric_accumulator(raw)
                tier_summary["success_rate"] = tier_summary.pop("exact_match_rate")
                by_tier[tier] = tier_summary
            run_summary["by_tier"] = by_tier
            per_run.append(run_summary)
            by_method_rows[method].append(run_summary)

    by_method: dict[str, dict[str, Any]] = {}
    for method, rows in by_method_rows.items():
        if not rows:
            continue
        count = len(rows)
        by_method[method] = {
            "num_runs": count,
            "mean_total_cost": sum(row["mean_total_cost"] for row in rows) / count,
            "mean_terminal_penalty": sum(row["mean_terminal_penalty"] for row in rows) / count,
            "mean_success_rate": sum(row["success_rate"] for row in rows) / count,
            "mean_false_cancel_count": sum(row["mean_false_cancel_count"] for row in rows) / count,
            "mean_missed_cancel_count": sum(row["mean_missed_cancel_count"] for row in rows) / count,
            "mean_false_refuse_count": sum(row["mean_false_refuse_count"] for row in rows) / count,
            "mean_missed_refuse_count": sum(row["mean_missed_refuse_count"] for row in rows) / count,
            "mean_path_agent_cost": sum(row["mean_path_agent_cost"] for row in rows) / count,
            "mean_shared_count": sum(row["leaf_type_counts"]["shared"] for row in rows) / count,
            "mean_unshared_count": sum(row["leaf_type_counts"]["unshared"] for row in rows) / count,
            "mean_subset_mismatch_rate": sum(row["subset_mismatch_rate"] for row in rows) / count,
        }

    return {
        "per_run": per_run,
        "by_method": by_method,
    }


def run_endpoint_semantics_check(
    multi_seed_results: dict[str, Any],
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in multi_seed_results["per_run"]:
        grouped.setdefault(row["method"], []).append(row)

    checks: dict[str, dict[str, Any]] = {}

    def make_result(method: str, status: str, detail: str, observed_counts: dict[str, int]) -> dict[str, Any]:
        return {
            "method": method,
            "status": status,
            "detail": detail,
            "observed_leaf_type_counts": observed_counts,
        }

    for method in ("full_share", "full_unshare", "epsilon_exp3", "naive_mixed"):
        rows = grouped.get(method, [])
        observed_counts = {
            "shared": sum(row["leaf_type_counts"]["shared"] for row in rows),
            "unshared": sum(row["leaf_type_counts"]["unshared"] for row in rows),
        }
        if method == "full_share":
            ok = observed_counts["unshared"] == 0
            checks[method] = make_result(
                method,
                "pass" if ok else "fail",
                "all episodes shared" if ok else "found unshared leaf in full_share",
                observed_counts,
            )
        elif method == "full_unshare":
            ok = observed_counts["shared"] == 0
            checks[method] = make_result(
                method,
                "pass" if ok else "fail",
                "all episodes unshared" if ok else "found shared leaf in full_unshare",
                observed_counts,
            )
        elif method == "epsilon_exp3":
            ok = all(
                row["catalog_preset"] == "all_unshare" and row["leaf_type_counts"]["shared"] == 0
                for row in rows
            )
            checks[method] = make_result(
                method,
                "pass" if ok else "fail",
                "all episodes unshared under all_unshare preset"
                if ok
                else "epsilon_exp3 did not stay unshared under all_unshare preset",
                observed_counts,
            )
        elif method == "naive_mixed":
            if observed_counts["shared"] > 0 and observed_counts["unshared"] > 0:
                checks[method] = make_result(
                    method,
                    "pass",
                    "observed both shared and unshared leaves under mixed preset",
                    observed_counts,
                )
            else:
                checks[method] = make_result(
                    method,
                    "warn",
                    "mixed preset used, but this sample did not realize both leaf types",
                    observed_counts,
                )

    return checks


def render_markdown(report: dict[str, Any]) -> str:
    oracle_section = report["check_1_oracle_best_path_distribution"]
    sweep = report["check_2_multi_seed_groupA"]
    endpoint = report["check_3_endpoint_semantics"]

    lines: list[str] = []
    lines.append("# Group A Day 7 Evaluator Checks")
    lines.append("")
    lines.append("## Check 1")
    lines.append("")
    for name, oracle in oracle_section.items():
        lines.append(f"### {name}")
        lines.append("")
        lines.append(f"- Catalog preset: `{oracle['catalog_preset']}`")
        lines.append(f"- Instances checked: `{oracle['num_instances_checked']}`")
        lines.append(f"- Best shared count: `{oracle['best_path_leaf_type_counts'].get('shared', 0)}`")
        lines.append(f"- Best unshared count: `{oracle['best_path_leaf_type_counts'].get('unshared', 0)}`")
        lines.append(f"- Mean best total cost: `{oracle['mean_best_total_cost']:.4f}`")
        lines.append(f"- Mean best terminal penalty: `{oracle['mean_best_terminal_penalty']:.4f}`")
        lines.append("")
    lines.append("## Check 2")
    lines.append("")
    lines.append("| method | seed | preset | mean_total_cost | mean_terminal_penalty | success_rate | false_cancel | missed_cancel | false_refuse | missed_refuse | subset_mismatch | shared | unshared |")
    lines.append("|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in sweep["per_run"]:
        lines.append(
            "| {method} | {seed} | {catalog_preset} | {mean_total_cost:.4f} | {mean_terminal_penalty:.4f} | {success_rate:.4f} | {mean_false_cancel_count:.3f} | {mean_missed_cancel_count:.3f} | {mean_false_refuse_count:.3f} | {mean_missed_refuse_count:.3f} | {subset_mismatch_rate:.3f} | {shared} | {unshared} |".format(
                method=row["method"],
                seed=row["seed"],
                catalog_preset=row["catalog_preset"],
                mean_total_cost=row["mean_total_cost"],
                mean_terminal_penalty=row["mean_terminal_penalty"],
                success_rate=row["success_rate"],
                mean_false_cancel_count=row["mean_false_cancel_count"],
                mean_missed_cancel_count=row["mean_missed_cancel_count"],
                mean_false_refuse_count=row["mean_false_refuse_count"],
                mean_missed_refuse_count=row["mean_missed_refuse_count"],
                subset_mismatch_rate=row["subset_mismatch_rate"],
                shared=row["leaf_type_counts"]["shared"],
                unshared=row["leaf_type_counts"]["unshared"],
            )
        )
    lines.append("")
    lines.append("## Check 3")
    lines.append("")
    for method, result in endpoint.items():
        counts = result["observed_leaf_type_counts"]
        lines.append(
            f"- `{method}`: `{result['status']}` - {result['detail']} "
            f"(shared={counts['shared']}, unshared={counts['unshared']})"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run evaluator-v2 quick checks for Group A.")
    parser.add_argument(
        "--data",
        type=Path,
        default=ROOT / "data" / "derived" / "airline_cancellation_fixed_tree" / "tasks.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "day7_eval_checks",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num-instances", type=int, default=15)
    parser.add_argument(
        "--catalog-preset",
        type=str,
        default="mixed",
        help="Primary preset for Check 1 oracle-best-path analysis.",
    )
    parser.add_argument(
        "--compare-richer-preset",
        action="store_true",
        help="Also run Check 1 on mixed_v2_richer when available.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "full_share",
            "full_unshare",
            "epsilon_exp3",
            "direct_multistage_exp3",
            "naive_mixed",
        ],
    )
    args = parser.parse_args()

    instances = load_instances(args.data)

    oracle_checks = {
        args.catalog_preset: run_oracle_best_path_distribution(
            instances=instances,
            num_instances=args.num_instances,
            seed=args.seeds[0] if args.seeds else 0,
            catalog_preset=args.catalog_preset,
        )
    }
    if args.compare_richer_preset:
        richer_preset = "mixed_v2_richer"
        oracle_checks[richer_preset] = run_oracle_best_path_distribution(
            instances=instances,
            num_instances=args.num_instances,
            seed=(args.seeds[0] if args.seeds else 0) + 1,
            catalog_preset=richer_preset,
        )

    report = {
        "check_1_oracle_best_path_distribution": oracle_checks,
        "check_2_multi_seed_groupA": run_method_seed_sweep(
            instances=instances,
            methods=args.methods,
            seeds=args.seeds,
            episodes=args.episodes,
        ),
    }
    report["check_3_endpoint_semantics"] = run_endpoint_semantics_check(
        report["check_2_multi_seed_groupA"]
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "groupA_day7_eval_checks.json"
    md_path = args.output_dir / "groupA_day7_eval_checks.md"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)
    with md_path.open("w", encoding="utf-8") as handle:
        handle.write(render_markdown(report))

    print("Check 1")
    for preset_name, result in oracle_checks.items():
        print(
            f"  preset={preset_name} instances={result['num_instances_checked']} "
            f"shared_best={result['best_path_leaf_type_counts'].get('shared', 0)} "
            f"unshared_best={result['best_path_leaf_type_counts'].get('unshared', 0)} "
            f"mean_terminal_penalty={result['mean_best_terminal_penalty']:.4f}"
        )
    print("Check 2")
    for row in report["check_2_multi_seed_groupA"]["per_run"]:
        print(
            f"  {row['method']} seed={row['seed']} preset={row['catalog_preset']} "
            f"cost={row['mean_total_cost']:.4f} term={row['mean_terminal_penalty']:.4f} "
            f"succ={row['success_rate']:.4f} subset={row['subset_mismatch_rate']:.3f} "
            f"fc={row['mean_false_cancel_count']:.2f} mc={row['mean_missed_cancel_count']:.2f} "
            f"fr={row['mean_false_refuse_count']:.2f} mr={row['mean_missed_refuse_count']:.2f}"
        )
    print("Check 3")
    for method, result in report["check_3_endpoint_semantics"].items():
        counts = result["observed_leaf_type_counts"]
        print(
            f"  {method}: {result['status']} - {result['detail']} "
            f"(shared={counts['shared']}, unshared={counts['unshared']})"
        )
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
