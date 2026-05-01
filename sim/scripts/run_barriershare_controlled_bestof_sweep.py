"""Run a best-of hyperparameter sweep for BarrierShare controlled sim."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_barriershare_controlled_sim.py"

ETA_EPSILON_METHODS = frozenset({"risky_ps_old", "risky_ps_linear", "epsilon_exp3"})
ETA_ONLY_METHODS = frozenset({"direct_multistage_exp3", "direct_multistage_exp3_local"})
FIXED_METHODS = frozenset({"naive_mixed_avg", "naive_mixed", "random_path"})
DEFAULT_PILOT_METHODS = (
    "risky_ps_old",
    "risky_ps_linear",
    "epsilon_exp3",
    "direct_multistage_exp3",
    "naive_mixed_avg",
    "random_path",
)
DEFAULT_ETA_VALUES = (0.5, 1.0, 1.5, 2.0)
DEFAULT_EPSILON_VALUES = (0.0, 0.005, 0.01, 0.02)
PS_FAMILY_METHODS = frozenset({"risky_ps", "risky_ps_old", "risky_ps_linear"})
MUST_BE_BEHIND_PS = frozenset(
    {
        "naive_mixed",
        "naive_mixed_avg",
        "random_path",
        "epsilon_exp3",
        "direct_multistage_exp3",
    }
)


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "| empty |\n| --- |\n| no rows |"
    header = "| " + " | ".join(fields) + " |"
    divider = "| " + " | ".join("---" for _ in fields) + " |"
    body = []
    for row in rows:
        body.append("| " + " | ".join(format_markdown_value(row.get(field)) for field in fields) + " |")
    return "\n".join([header, divider, *body])


def format_markdown_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def best_sort_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        float(row["regret_per_t_mean"]),
        float(row["overall_avg_total_cost_mean"]),
        float(row["post_switch_avg_regret_mean"]),
        str(row["method"]),
    )


def build_run_specs(
    methods: list[str],
    eta_values: list[float],
    epsilon_values: list[float],
) -> list[dict[str, Any]]:
    run_specs: list[dict[str, Any]] = []
    for method in methods:
        if method in ETA_EPSILON_METHODS:
            for eta in eta_values:
                for epsilon in epsilon_values:
                    run_specs.append(
                        {
                            "method": method,
                            "eta": eta,
                            "epsilon": epsilon,
                        }
                    )
            continue
        if method in ETA_ONLY_METHODS:
            for eta in eta_values:
                run_specs.append(
                    {
                        "method": method,
                        "eta": eta,
                        "epsilon": None,
                    }
                )
            continue
        if method in FIXED_METHODS:
            run_specs.append(
                {
                    "method": method,
                    "eta": None,
                    "epsilon": None,
                }
            )
            continue
        raise ValueError(f"Unsupported sweep method: {method}")
    return run_specs


def run_one(
    *,
    python_executable: str,
    run_spec: dict[str, Any],
    output_dir: Path,
    tree_spec: Path,
    role_mode: str,
    cost_mode: str,
    trap_switch_denominator: int,
    horizon: int,
    seeds: list[int],
    cost_noise: float,
    specialist_fraction: float,
) -> dict[str, Any]:
    method = str(run_spec["method"])
    run_name_parts = [method]
    if run_spec.get("eta") is not None:
        run_name_parts.append(f"eta_{run_spec['eta']:g}")
    if run_spec.get("epsilon") is not None:
        run_name_parts.append(f"eps_{run_spec['epsilon']:g}")
    run_dir = output_dir / "runs" / "__".join(run_name_parts)
    run_dir.mkdir(parents=True, exist_ok=True)

    command = [
        python_executable,
        str(RUNNER),
        "--output-dir",
        str(run_dir),
        "--tree-spec",
        str(tree_spec),
        "--tree-spec-role-mode",
        role_mode,
        "--tree-spec-cost-mode",
        cost_mode,
        "--trap-switch-denominator",
        str(trap_switch_denominator),
        "--horizon",
        str(horizon),
        "--cost-noise",
        str(cost_noise),
        "--specialist-fraction",
        str(specialist_fraction),
        "--methods",
        method,
        "--seeds",
        *[str(seed) for seed in seeds],
    ]
    if run_spec.get("eta") is not None:
        command.extend(["--common-eta-override", str(run_spec["eta"])])
    if run_spec.get("epsilon") is not None:
        command.extend(["--common-epsilon-override", str(run_spec["epsilon"])])

    completed = subprocess.run(
        command,
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    summary_rows = json.loads((run_dir / "controlled_sim_compare.json").read_text(encoding="utf-8"))
    if len(summary_rows) != 1:
        raise RuntimeError(
            f"Expected exactly one summary row for {method}, found {len(summary_rows)} in {run_dir}."
        )
    summary_row = dict(summary_rows[0])
    summary_row["run_dir"] = str(run_dir)
    summary_row["sweep_method"] = method
    summary_row["sweep_eta"] = run_spec.get("eta")
    summary_row["sweep_epsilon"] = run_spec.get("epsilon")
    summary_row["stdout_tail"] = "\n".join(completed.stdout.strip().splitlines()[-5:])
    return summary_row


def build_best_of_rows(best_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for rank, row in enumerate(sorted(best_rows, key=best_sort_key), start=1):
        rows.append(
            {
                "rank": rank,
                "method": row["method"],
                "best_eta": row.get("eta"),
                "best_epsilon": row.get("epsilon"),
                "regret/T": row["regret_per_t_mean"],
                "overall avg cost": row["overall_avg_total_cost_mean"],
                "tail20 avg cost": row["tail20_avg_total_cost_mean"],
                "post-switch avg regret": row["post_switch_avg_regret_mean"],
                "target good frac": row["target_good_fraction_mean"],
                "trap frac": row["trap_basin_fraction_mean"],
                "shared path frac": row["shared_path_fraction_mean"],
                "run_dir": row["run_dir"],
            }
        )
    return rows


def goal_status(best_of_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not best_of_rows:
        return {
            "goal_met": False,
            "best_method": None,
            "best_is_ps_family": False,
            "required_methods_behind_best_ps": False,
        }
    best_method = str(best_of_rows[0]["method"])
    best_is_ps_family = best_method in PS_FAMILY_METHODS
    best_rank_by_method = {str(row["method"]): int(row["rank"]) for row in best_of_rows}
    best_ps_rank = min(
        (best_rank_by_method[method] for method in best_rank_by_method if method in PS_FAMILY_METHODS),
        default=None,
    )
    behind_required = (
        best_ps_rank is not None
        and all(
            best_rank_by_method.get(method, 10**9) > best_ps_rank
            for method in MUST_BE_BEHIND_PS
            if method in best_rank_by_method
        )
    )
    return {
        "goal_met": bool(best_is_ps_family and behind_required),
        "best_method": best_method,
        "best_is_ps_family": best_is_ps_family,
        "required_methods_behind_best_ps": behind_required,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a best-of sweep for BarrierShare controlled sim.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs" / "barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v1",
    )
    parser.add_argument(
        "--tree-spec",
        type=Path,
        default=ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json",
    )
    parser.add_argument("--tree-spec-role-mode", default="spec_or_agent_id")
    parser.add_argument("--tree-spec-cost-mode", default="ps_favored_trap_v10_avg_baited")
    parser.add_argument("--trap-switch-denominator", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--cost-noise", type=float, default=0.02)
    parser.add_argument("--specialist-fraction", type=float, default=0.15)
    parser.add_argument("--methods", nargs="+", default=list(DEFAULT_PILOT_METHODS))
    parser.add_argument(
        "--eta-values",
        default=",".join(str(value) for value in DEFAULT_ETA_VALUES),
    )
    parser.add_argument(
        "--epsilon-values",
        default=",".join(str(value) for value in DEFAULT_EPSILON_VALUES),
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    eta_values = parse_float_list(args.eta_values)
    epsilon_values = parse_float_list(args.epsilon_values)
    run_specs = build_run_specs(list(args.methods), eta_values, epsilon_values)

    manifest = {
        "runner": str(RUNNER),
        "tree_spec": str(args.tree_spec),
        "tree_spec_role_mode": args.tree_spec_role_mode,
        "tree_spec_cost_mode": args.tree_spec_cost_mode,
        "trap_switch_denominator": args.trap_switch_denominator,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "cost_noise": args.cost_noise,
        "specialist_fraction": args.specialist_fraction,
        "methods": args.methods,
        "eta_values": eta_values,
        "epsilon_values": epsilon_values,
        "run_specs": run_specs,
    }
    write_json(args.output_dir / "sweep_runs_manifest.json", manifest)

    long_rows: list[dict[str, Any]] = []
    for run_index, run_spec in enumerate(run_specs, start=1):
        print(
            f"[bestof-sweep] {run_index}/{len(run_specs)} method={run_spec['method']} "
            f"eta={run_spec.get('eta')} epsilon={run_spec.get('epsilon')}",
            flush=True,
        )
        row = run_one(
            python_executable=sys.executable,
            run_spec=run_spec,
            output_dir=args.output_dir,
            tree_spec=args.tree_spec,
            role_mode=args.tree_spec_role_mode,
            cost_mode=args.tree_spec_cost_mode,
            trap_switch_denominator=args.trap_switch_denominator,
            horizon=args.horizon,
            seeds=args.seeds,
            cost_noise=args.cost_noise,
            specialist_fraction=args.specialist_fraction,
        )
        row["run_index"] = run_index
        long_rows.append(row)

    long_rows = sorted(long_rows, key=lambda row: (str(row["method"]), best_sort_key(row)))
    write_json(args.output_dir / "sweep_results_long.json", long_rows)
    write_csv(args.output_dir / "sweep_results_long.csv", long_rows)

    best_by_method: list[dict[str, Any]] = []
    for method in args.methods:
        method_rows = [row for row in long_rows if row["method"] == method]
        if not method_rows:
            raise RuntimeError(f"No sweep rows found for method={method}")
        best_by_method.append(dict(sorted(method_rows, key=best_sort_key)[0]))
    best_by_method = sorted(best_by_method, key=best_sort_key)
    write_json(args.output_dir / "best_params_by_method.json", best_by_method)
    write_csv(args.output_dir / "best_params_by_method.csv", best_by_method)

    best_of_rows = build_best_of_rows(best_by_method)
    best_of_payload = {
        "rows": best_of_rows,
        "goal_status": goal_status(best_of_rows),
    }
    write_json(args.output_dir / "best_of_compare.json", best_of_payload)
    write_csv(args.output_dir / "best_of_compare.csv", best_of_rows)
    (args.output_dir / "best_of_compare.md").write_text(
        "\n".join(
            [
                "# BarrierShare controlled sim best-of compare",
                "",
                f"- tree_spec_cost_mode: `{args.tree_spec_cost_mode}`",
                f"- trap_switch_denominator: `{args.trap_switch_denominator}`",
                f"- horizon: `{args.horizon}`",
                f"- seeds: `{args.seeds}`",
                f"- goal_status: `{best_of_payload['goal_status']}`",
                "",
                markdown_table(
                    best_of_rows,
                    [
                        "rank",
                        "method",
                        "best_eta",
                        "best_epsilon",
                        "regret/T",
                        "overall avg cost",
                        "tail20 avg cost",
                        "post-switch avg regret",
                        "target good frac",
                        "trap frac",
                        "shared path frac",
                    ],
                ),
                "",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
