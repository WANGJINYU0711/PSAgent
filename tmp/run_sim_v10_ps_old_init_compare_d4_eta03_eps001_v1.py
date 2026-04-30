#!/usr/bin/env python3
"""Run the d=4 eta=0.3 eps=0.01 PS-old init comparison."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_barriershare_controlled_sim.py"
EXPERIMENT_NAME = "sim_v10_ps_old_init_compare_d4_eta03_eps001_v1"
OUTPUT_DIR = ROOT / "outputs" / EXPERIMENT_NAME
TREE_SPEC = ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json"

HORIZON = 1000
SEEDS = list(range(10))
COST_NOISE = 0.02
SPECIALIST_FRACTION = 0.15
TRAP_SWITCH_DENOMINATOR = 4
ETA = 0.3
EPSILON = 0.01

METHOD_SPECS = [
    {
        "method": "risky_ps_old",
        "init_strategy": "algorithm_subtree_sum",
        "init_description": "W[u] = sum_{leaf in L(u)} exp(eta * theta_leaf)",
    },
    {
        "method": "risky_ps_old_random_init",
        "init_strategy": "random_edge_mass_uniform_0p5_1p5",
        "init_description": "Each exposed shared edge W is sampled independently from Uniform(0.5, 1.5).",
    },
    {
        "method": "risky_ps_old_fixed_init",
        "init_strategy": "fixed_edge_mass_1p0",
        "init_description": "Each exposed shared edge W is initialized to 1.0.",
    },
]

SUMMARY_FIELDS = [
    "rank_post_switch",
    "rank_overall",
    "method",
    "init_strategy",
    "regret_per_t_mean",
    "regret_per_t_std",
    "overall_avg_total_cost_mean",
    "overall_avg_total_cost_std",
    "post_switch_avg_regret_mean",
    "post_switch_avg_regret_std",
    "tail20_avg_total_cost_mean",
    "tail20_avg_total_cost_std",
    "target_good_fraction_mean",
    "target_bad_fraction_mean",
    "trap_basin_fraction_mean",
    "decoy_branch_fraction_mean",
    "calibrated_decoy_fraction_mean",
    "broad_safe_basin_fraction_mean",
    "ordinary_safe_basin_fraction_mean",
    "shared_path_fraction_mean",
    "ps_favored_exact_best_hit_rate_mean",
    "exact_best_path_hit_rate_mean",
    "first_episode_best_hit_rate_mean",
    "shared_update_count_mean",
    "risky_update_count_mean",
    "post_switch_start_episode",
    "post_switch_episode_count",
    "seeds",
    "horizon",
]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def md_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return "| empty |\n| --- |\n| no rows |"
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(md_value(row.get(field)) for field in fields) + " |")
    return "\n".join(lines)


def run_one(spec: dict[str, str], force: bool) -> dict[str, Any]:
    method = spec["method"]
    run_dir = OUTPUT_DIR / "runs" / method
    compare_path = run_dir / "controlled_sim_compare.json"
    log_path = run_dir / "run.log"
    if compare_path.exists() and not force:
        return {"status": "skipped", "method": method, "run_dir": str(run_dir)}

    command = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(run_dir),
        "--tree-spec",
        str(TREE_SPEC),
        "--tree-spec-role-mode",
        "spec_or_agent_id",
        "--tree-spec-cost-mode",
        "ps_favored_trap_v10_avg_baited",
        "--trap-switch-denominator",
        str(TRAP_SWITCH_DENOMINATOR),
        "--horizon",
        str(HORIZON),
        "--seeds",
        *[str(seed) for seed in SEEDS],
        "--cost-noise",
        str(COST_NOISE),
        "--specialist-fraction",
        str(SPECIALIST_FRACTION),
        "--common-eta-override",
        str(ETA),
        "--common-epsilon-override",
        str(EPSILON),
        "--methods",
        method,
    ]
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.time()
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    elapsed = time.time() - started
    log_path.write_text(
        "\n".join(
            [
                "$ " + " ".join(command),
                "",
                f"returncode={completed.returncode}",
                f"elapsed_sec={elapsed:.3f}",
                "",
                "[stdout]",
                completed.stdout,
                "",
                "[stderr]",
                completed.stderr,
            ]
        ),
        encoding="utf-8",
    )
    if completed.returncode != 0:
        return {
            "status": "failed",
            "method": method,
            "run_dir": str(run_dir),
            "elapsed_sec": elapsed,
            "returncode": completed.returncode,
            "stderr_tail": "\n".join(completed.stderr.splitlines()[-30:]),
        }
    return {"status": "completed", "method": method, "run_dir": str(run_dir), "elapsed_sec": elapsed}


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def add_strategy(row: dict[str, Any], spec: dict[str, str]) -> dict[str, Any]:
    out = dict(row)
    out["init_strategy"] = spec["init_strategy"]
    out["init_description"] = spec["init_description"]
    return out


def metric(row: dict[str, Any], field: str) -> float:
    value = row.get(field)
    if value is None:
        return float("inf")
    return float(value)


def aggregate(run_results: list[dict[str, Any]]) -> None:
    spec_by_method = {spec["method"]: spec for spec in METHOD_SPECS}
    summary_rows: list[dict[str, Any]] = []
    per_seed_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] | None = None

    for spec in METHOD_SPECS:
        method = spec["method"]
        run_dir = OUTPUT_DIR / "runs" / method
        summary_payload = load_json(run_dir / "controlled_sim_compare.json")
        per_seed_payload = load_json(run_dir / "per_seed_results.json")
        curve_payload = load_json(run_dir / "regret_curve.json")
        for row in summary_payload:
            summary_rows.append(add_strategy(row, spec))
        for row in per_seed_payload:
            per_seed_rows.append(add_strategy(row, spec))
        for row in curve_payload:
            curve_rows.append(add_strategy(row, spec))
        if diagnostics is None:
            findings = load_json(run_dir / "findings.json")
            diagnostics = findings.get("ps_favored_trap_diagnostics")

    summary_rows = sorted(
        summary_rows,
        key=lambda row: (
            metric(row, "post_switch_avg_regret_mean"),
            metric(row, "tail20_avg_total_cost_mean"),
            metric(row, "regret_per_t_mean"),
        ),
    )
    for rank, row in enumerate(summary_rows, start=1):
        row["rank_post_switch"] = rank
    for rank, row in enumerate(
        sorted(
            summary_rows,
            key=lambda row: (
                metric(row, "regret_per_t_mean"),
                metric(row, "overall_avg_total_cost_mean"),
                metric(row, "post_switch_avg_regret_mean"),
            ),
        ),
        start=1,
    ):
        row["rank_overall"] = rank

    algorithm = next(row for row in summary_rows if row["method"] == "risky_ps_old")
    deltas = []
    for row in summary_rows:
        deltas.append(
            {
                "method": row["method"],
                "init_strategy": row["init_strategy"],
                "delta_regret_per_t_vs_algorithm": metric(row, "regret_per_t_mean")
                - metric(algorithm, "regret_per_t_mean"),
                "delta_post_switch_vs_algorithm": metric(row, "post_switch_avg_regret_mean")
                - metric(algorithm, "post_switch_avg_regret_mean"),
                "delta_tail20_vs_algorithm": metric(row, "tail20_avg_total_cost_mean")
                - metric(algorithm, "tail20_avg_total_cost_mean"),
                "delta_target_good_vs_algorithm": metric(row, "target_good_fraction_mean")
                - metric(algorithm, "target_good_fraction_mean"),
                "delta_exact_best_vs_algorithm": metric(row, "ps_favored_exact_best_hit_rate_mean")
                - metric(algorithm, "ps_favored_exact_best_hit_rate_mean"),
                "delta_trap_vs_algorithm": metric(row, "trap_basin_fraction_mean")
                - metric(algorithm, "trap_basin_fraction_mean"),
            }
        )

    methods = [spec["method"] for spec in METHOD_SPECS]
    rows_by_seed: dict[int, dict[str, dict[str, Any]]] = {}
    for row in per_seed_rows:
        rows_by_seed.setdefault(int(row["seed"]), {})[str(row["method"])] = row

    per_seed_pivot: list[dict[str, Any]] = []
    post_switch_win_counts = {method: 0 for method in methods}
    overall_win_counts = {method: 0 for method in methods}
    for seed, seed_rows in sorted(rows_by_seed.items()):
        best_post = min(methods, key=lambda method: metric(seed_rows[method], "post_switch_avg_regret"))
        best_overall = min(methods, key=lambda method: metric(seed_rows[method], "regret_per_t"))
        post_switch_win_counts[best_post] += 1
        overall_win_counts[best_overall] += 1
        pivot_row: dict[str, Any] = {
            "seed": seed,
            "best_post_switch_method": best_post,
            "best_overall_method": best_overall,
        }
        for method in methods:
            row = seed_rows[method]
            pivot_row[f"{method}_post_switch_avg_regret"] = row["post_switch_avg_regret"]
            pivot_row[f"{method}_regret_per_t"] = row["regret_per_t"]
            pivot_row[f"{method}_target_good_fraction"] = row["target_good_fraction"]
            pivot_row[f"{method}_trap_basin_fraction"] = row["trap_basin_fraction"]
        per_seed_pivot.append(pivot_row)

    paired_delta_stats: list[dict[str, Any]] = []
    for method in methods:
        if method == "risky_ps_old":
            continue
        post_deltas = [
            rows_by_seed[seed][method]["post_switch_avg_regret"]
            - rows_by_seed[seed]["risky_ps_old"]["post_switch_avg_regret"]
            for seed in sorted(rows_by_seed)
        ]
        overall_deltas = [
            rows_by_seed[seed][method]["regret_per_t"]
            - rows_by_seed[seed]["risky_ps_old"]["regret_per_t"]
            for seed in sorted(rows_by_seed)
        ]
        target_good_deltas = [
            rows_by_seed[seed][method]["target_good_fraction"]
            - rows_by_seed[seed]["risky_ps_old"]["target_good_fraction"]
            for seed in sorted(rows_by_seed)
        ]
        paired_delta_stats.append(
            {
                "method": method,
                "init_strategy": spec_by_method[method]["init_strategy"],
                "post_delta_mean": statistics.fmean(post_deltas),
                "post_delta_std": statistics.stdev(post_deltas) if len(post_deltas) > 1 else 0.0,
                "post_delta_min": min(post_deltas),
                "post_delta_max": max(post_deltas),
                "post_delta_positive_seed_count": sum(delta > 0 for delta in post_deltas),
                "overall_delta_mean": statistics.fmean(overall_deltas),
                "overall_delta_std": statistics.stdev(overall_deltas) if len(overall_deltas) > 1 else 0.0,
                "target_good_delta_mean": statistics.fmean(target_good_deltas),
                "target_good_delta_positive_seed_count": sum(
                    delta > 0 for delta in target_good_deltas
                ),
            }
        )

    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "short_name": "ps-old init-compare d4 v1",
        "runner": str(RUNNER),
        "tree_spec": str(TREE_SPEC),
        "tree_spec_role_mode": "spec_or_agent_id",
        "tree_spec_cost_mode": "ps_favored_trap_v10_avg_baited",
        "horizon": HORIZON,
        "seeds": SEEDS,
        "cost_noise": COST_NOISE,
        "specialist_fraction": SPECIALIST_FRACTION,
        "trap_switch_denominator": TRAP_SWITCH_DENOMINATOR,
        "switch_episode": HORIZON // TRAP_SWITCH_DENOMINATOR,
        "eta": ETA,
        "epsilon": EPSILON,
        "methods": METHOD_SPECS,
        "parallelism": "method-level subprocess parallelism",
        "run_results": run_results,
    }
    write_json(OUTPUT_DIR / "manifest.json", manifest)
    write_json(OUTPUT_DIR / "controlled_sim_compare_init_compare.json", summary_rows)
    write_csv(OUTPUT_DIR / "controlled_sim_compare_init_compare.csv", summary_rows)
    write_csv(OUTPUT_DIR / "controlled_sim_compare_init_compare_compact.csv", summary_rows, SUMMARY_FIELDS)
    write_json(OUTPUT_DIR / "per_seed_results_init_compare.json", per_seed_rows)
    write_csv(OUTPUT_DIR / "per_seed_results_init_compare.csv", per_seed_rows)
    write_json(OUTPUT_DIR / "regret_curve_init_compare.json", curve_rows)
    write_json(OUTPUT_DIR / "deltas_vs_algorithm_init.json", deltas)
    write_csv(OUTPUT_DIR / "deltas_vs_algorithm_init.csv", deltas)
    write_json(OUTPUT_DIR / "per_seed_post_switch_pivot.json", per_seed_pivot)
    write_csv(OUTPUT_DIR / "per_seed_post_switch_pivot.csv", per_seed_pivot)
    write_json(OUTPUT_DIR / "paired_delta_stats_vs_algorithm.json", paired_delta_stats)
    write_csv(OUTPUT_DIR / "paired_delta_stats_vs_algorithm.csv", paired_delta_stats)

    table_fields = [
        "rank_post_switch",
        "rank_overall",
        "method",
        "init_strategy",
        "post_switch_avg_regret_mean",
        "post_switch_avg_regret_std",
        "regret_per_t_mean",
        "regret_per_t_std",
        "tail20_avg_total_cost_mean",
        "target_good_fraction_mean",
        "target_bad_fraction_mean",
        "trap_basin_fraction_mean",
        "shared_path_fraction_mean",
        "ps_favored_exact_best_hit_rate_mean",
        "shared_update_count_mean",
        "risky_update_count_mean",
    ]
    delta_fields = [
        "method",
        "init_strategy",
        "delta_post_switch_vs_algorithm",
        "delta_regret_per_t_vs_algorithm",
        "delta_tail20_vs_algorithm",
        "delta_target_good_vs_algorithm",
        "delta_exact_best_vs_algorithm",
        "delta_trap_vs_algorithm",
    ]
    paired_delta_fields = [
        "method",
        "init_strategy",
        "post_delta_mean",
        "post_delta_std",
        "post_delta_min",
        "post_delta_max",
        "post_delta_positive_seed_count",
        "overall_delta_mean",
        "overall_delta_std",
        "target_good_delta_mean",
        "target_good_delta_positive_seed_count",
    ]
    win_rows = [
        {
            "method": method,
            "init_strategy": spec_by_method[method]["init_strategy"],
            "post_switch_seed_wins": post_switch_win_counts[method],
            "overall_seed_wins": overall_win_counts[method],
        }
        for method in methods
    ]
    best = summary_rows[0]
    report_lines = [
        "# sim_v10 PS-old init comparison d4 eta0.3 eps0.01 v1",
        "",
        "## Experiment Name",
        "",
        f"`{EXPERIMENT_NAME}`",
        "",
        "Short name: `ps-old init-compare d4 v1`.",
        "",
        "## Setting",
        "",
        "- Base setting: latest v10 fixed nonzero-eps controlled sim setting.",
        f"- tree_spec: `{TREE_SPEC.relative_to(ROOT)}`",
        "- tree_spec_role_mode: `spec_or_agent_id`",
        "- tree_spec_cost_mode: `ps_favored_trap_v10_avg_baited`",
        f"- horizon: `{HORIZON}`",
        f"- seeds: `{SEEDS}`",
        f"- cost_noise: `{COST_NOISE}`",
        f"- specialist_fraction: `{SPECIALIST_FRACTION}`",
        f"- d / trap_switch_denominator: `{TRAP_SWITCH_DENOMINATOR}`",
        f"- switch episode: `{HORIZON // TRAP_SWITCH_DENOMINATOR}`",
        f"- eta: `{ETA}`",
        f"- epsilon: `{EPSILON}`",
        "- methods: exact `risky_ps_old` algorithm only; treatments differ only in initial shared W.",
        "- parallelism: method-level subprocess parallelism.",
        "",
        "## Init Treatments",
        "",
        markdown_table(
            [
                {
                    "method": spec["method"],
                    "init_strategy": spec["init_strategy"],
                    "description": spec["init_description"],
                }
                for spec in METHOD_SPECS
            ],
            ["method", "init_strategy", "description"],
        ),
        "",
        "## Main Ranking",
        "",
        markdown_table(summary_rows, table_fields),
        "",
        "## Deltas Vs Algorithm Init",
        "",
        "Positive cost deltas are worse than the algorithm initialization; positive target/exact deltas are better.",
        "",
        markdown_table(deltas, delta_fields),
        "",
        "## Per-Seed Stability",
        "",
        markdown_table(
            win_rows,
            ["method", "init_strategy", "post_switch_seed_wins", "overall_seed_wins"],
        ),
        "",
        "## Paired Seed Deltas",
        "",
        "Positive cost deltas mean the treatment is worse than algorithm init on that seed.",
        "",
        markdown_table(paired_delta_stats, paired_delta_fields),
        "",
        "## Diagnostics",
        "",
        f"- diagnostics available: `{diagnostics is not None}`",
        f"- best post-switch treatment: `{best['method']}` / `{best['init_strategy']}`",
        f"- best post-switch avg regret: `{best['post_switch_avg_regret_mean']:.6f}`",
        f"- post-switch episodes per seed: `{best.get('post_switch_episode_count')}`",
        "",
        "## Output Files",
        "",
        "- `controlled_sim_compare_init_compare.csv/json`: all aggregate metrics.",
        "- `controlled_sim_compare_init_compare_compact.csv`: compact aggregate metrics.",
        "- `per_seed_results_init_compare.csv/json`: per-seed rows.",
        "- `regret_curve_init_compare.json`: sampled learning curves.",
        "- `deltas_vs_algorithm_init.csv/json`: paired deltas against the algorithm init.",
        "- `per_seed_post_switch_pivot.csv/json`: seed-level pivot for stability checks.",
        "- `paired_delta_stats_vs_algorithm.csv/json`: paired seed delta statistics.",
        "- `runs/<method>/`: raw runner output for each treatment.",
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


def main() -> None:
    force = "--force" in sys.argv
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=len(METHOD_SPECS)) as executor:
        futures = [executor.submit(run_one, spec, force) for spec in METHOD_SPECS]
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(
                f"[init-compare] {result['status']} method={result['method']} "
                f"elapsed={result.get('elapsed_sec', 0.0):.1f}s",
                flush=True,
            )
            if result["status"] == "failed":
                print(result.get("stderr_tail", ""), flush=True)
    failures = [row for row in results if row["status"] == "failed"]
    if failures:
        write_json(OUTPUT_DIR / "run_failures.json", failures)
        raise SystemExit(1)
    aggregate(sorted(results, key=lambda row: row["method"]))
    print(
        json.dumps(
            {
                "experiment_name": EXPERIMENT_NAME,
                "output_dir": str(OUTPUT_DIR),
                "elapsed_sec": time.time() - started,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
