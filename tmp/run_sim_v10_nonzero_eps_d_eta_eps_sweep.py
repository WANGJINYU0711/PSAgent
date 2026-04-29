#!/usr/bin/env python3
"""Run and aggregate the v10 nonzero-epsilon d/eta/eps sim sweep."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_barriershare_controlled_sim.py"
EXPERIMENT_NAME = "sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1"

METHODS = [
    "risky_ps_safe_conditional",
    "risky_ps_ix",
    "risky_ps_old",
    "risky_ps_linear",
    "risky_ps",
    "risky_ps_direct_cost",
    "risky_ps_safe_conditional_ix",
    "direct_multistage_exp3",
    "direct_multistage_exp3_local",
    "epsilon_exp3",
    "naive_mixed_avg",
    "naive_mixed",
    "random_path",
]

PS_FAMILY = {
    "risky_ps_safe_conditional",
    "risky_ps_ix",
    "risky_ps_old",
    "risky_ps_linear",
    "risky_ps",
    "risky_ps_direct_cost",
    "risky_ps_safe_conditional_ix",
}

PRIMARY_FIELDS = [
    "d",
    "switch_episode",
    "eta_sweep",
    "epsilon_sweep",
    "method",
    "post_switch_rank",
    "overall_rank",
    "post_switch_avg_regret_mean",
    "tail20_avg_total_cost_mean",
    "regret_per_t_mean",
    "overall_avg_total_cost_mean",
    "target_good_fraction_mean",
    "trap_basin_fraction_mean",
    "shared_path_fraction_mean",
    "exact_best_path_hit_rate_mean",
    "ps_favored_exact_best_hit_rate_mean",
    "first_episode_best_hit_rate_mean",
    "shared_update_count_mean",
    "risky_update_count_mean",
    "run_dir",
]


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


def parse_int_list(value: str) -> list[int]:
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def float_tag(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


def combo_run_dir(output_dir: Path, d: int, eta: float, eps: float) -> Path:
    return output_dir / "runs" / f"d_{d:02d}__eta_{float_tag(eta)}__eps_{float_tag(eps)}"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    if fields is None:
        seen: list[str] = []
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.append(key)
        fields = seen
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


def metric(row: dict[str, Any], field: str, default: float = float("inf")) -> float:
    value = row.get(field)
    if value is None:
        return default
    return float(value)


def post_switch_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        metric(row, "post_switch_avg_regret_mean"),
        metric(row, "tail20_avg_total_cost_mean"),
        metric(row, "regret_per_t_mean"),
        str(row.get("method")),
    )


def overall_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
    return (
        metric(row, "regret_per_t_mean"),
        metric(row, "overall_avg_total_cost_mean"),
        metric(row, "post_switch_avg_regret_mean"),
        str(row.get("method")),
    )


def run_one_combo(
    *,
    output_dir: Path,
    tree_spec: Path,
    d: int,
    eta: float,
    eps: float,
    horizon: int,
    seeds: list[int],
    cost_noise: float,
    specialist_fraction: float,
    force: bool,
) -> dict[str, Any]:
    run_dir = combo_run_dir(output_dir, d, eta, eps)
    compare_path = run_dir / "controlled_sim_compare.json"
    log_path = run_dir / "run.log"
    if compare_path.exists() and not force:
        return {
            "status": "skipped",
            "d": d,
            "eta": eta,
            "eps": eps,
            "run_dir": str(run_dir),
        }

    run_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(run_dir),
        "--tree-spec",
        str(tree_spec),
        "--tree-spec-role-mode",
        "spec_or_agent_id",
        "--tree-spec-cost-mode",
        "ps_favored_trap_v10_avg_baited",
        "--trap-switch-denominator",
        str(d),
        "--horizon",
        str(horizon),
        "--seeds",
        *[str(seed) for seed in seeds],
        "--cost-noise",
        str(cost_noise),
        "--specialist-fraction",
        str(specialist_fraction),
        "--common-eta-override",
        str(eta),
        "--common-epsilon-override",
        str(eps),
        "--methods",
        *METHODS,
    ]
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
            "d": d,
            "eta": eta,
            "eps": eps,
            "run_dir": str(run_dir),
            "elapsed_sec": elapsed,
            "returncode": completed.returncode,
            "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        }
    if not compare_path.exists():
        return {
            "status": "failed_missing_compare",
            "d": d,
            "eta": eta,
            "eps": eps,
            "run_dir": str(run_dir),
            "elapsed_sec": elapsed,
        }
    return {
        "status": "completed",
        "d": d,
        "eta": eta,
        "eps": eps,
        "run_dir": str(run_dir),
        "elapsed_sec": elapsed,
    }


def load_rows(output_dir: Path, combos: list[tuple[int, float, float]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for d, eta, eps in combos:
        run_dir = combo_run_dir(output_dir, d, eta, eps)
        compare_path = run_dir / "controlled_sim_compare.json"
        if not compare_path.exists():
            continue
        payload = json.loads(compare_path.read_text(encoding="utf-8"))
        for source_row in payload:
            row = dict(source_row)
            row["experiment_name"] = EXPERIMENT_NAME
            row["d"] = d
            row["switch_episode"] = int(row.get("trap_switch_episode", 1000 // d))
            row["eta_sweep"] = eta
            row["epsilon_sweep"] = eps
            row["run_dir"] = str(run_dir)
            rows.append(row)
    return rows


def add_ranks(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["d"]), float(row["eta_sweep"]), float(row["epsilon_sweep"])), []).append(row)
    for group_rows in grouped.values():
        for rank, row in enumerate(sorted(group_rows, key=post_switch_key), start=1):
            row["post_switch_rank"] = rank
        for rank, row in enumerate(sorted(group_rows, key=overall_key), start=1):
            row["overall_rank"] = rank


def best_by_method(rows: list[dict[str, Any]], key_name: str) -> list[dict[str, Any]]:
    key_func = post_switch_key if key_name == "post_switch" else overall_key
    output = []
    for method in METHODS:
        method_rows = [row for row in rows if row["method"] == method]
        if not method_rows:
            continue
        output.append(dict(sorted(method_rows, key=key_func)[0]))
    output = sorted(output, key=key_func)
    for rank, row in enumerate(output, start=1):
        row[f"best_{key_name}_rank"] = rank
    return output


def combo_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((int(row["d"]), float(row["eta_sweep"]), float(row["epsilon_sweep"])), []).append(row)

    summaries = []
    for (d, eta, eps), group_rows in sorted(grouped.items()):
        by_method = {row["method"]: row for row in group_rows}
        best_post = sorted(group_rows, key=post_switch_key)[0]
        best_overall = sorted(group_rows, key=overall_key)[0]
        ps_rows = [row for row in group_rows if row["method"] in PS_FAMILY]
        best_ps_post = sorted(ps_rows, key=post_switch_key)[0]
        best_ps_overall = sorted(ps_rows, key=overall_key)[0]
        direct = by_method.get("direct_multistage_exp3")
        direct_local = by_method.get("direct_multistage_exp3_local")
        epsilon_exp3 = by_method.get("epsilon_exp3")
        naive_avg = by_method.get("naive_mixed_avg")
        summary = {
            "d": d,
            "switch_episode": int(best_post.get("switch_episode", 1000 // d)),
            "eta": eta,
            "epsilon": eps,
            "best_post_switch_method": best_post["method"],
            "best_post_switch_is_ps": best_post["method"] in PS_FAMILY,
            "best_post_switch_regret": best_post["post_switch_avg_regret_mean"],
            "best_overall_method": best_overall["method"],
            "best_overall_is_ps": best_overall["method"] in PS_FAMILY,
            "best_overall_regret_per_t": best_overall["regret_per_t_mean"],
            "best_ps_post_method": best_ps_post["method"],
            "best_ps_post_rank": best_ps_post["post_switch_rank"],
            "best_ps_post_switch_regret": best_ps_post["post_switch_avg_regret_mean"],
            "best_ps_tail20": best_ps_post["tail20_avg_total_cost_mean"],
            "best_ps_regret_per_t": best_ps_post["regret_per_t_mean"],
            "best_ps_target_good": best_ps_post["target_good_fraction_mean"],
            "best_ps_trap": best_ps_post["trap_basin_fraction_mean"],
            "best_ps_shared_path": best_ps_post["shared_path_fraction_mean"],
            "best_ps_overall_method": best_ps_overall["method"],
            "best_ps_overall_rank": best_ps_overall["overall_rank"],
            "best_ps_overall_regret_per_t": best_ps_overall["regret_per_t_mean"],
        }
        for label, baseline in [
            ("direct", direct),
            ("direct_local", direct_local),
            ("epsilon_exp3", epsilon_exp3),
            ("naive_mixed_avg", naive_avg),
        ]:
            if baseline is None:
                continue
            base_post = float(baseline["post_switch_avg_regret_mean"])
            ps_post = float(best_ps_post["post_switch_avg_regret_mean"])
            base_overall = float(baseline["regret_per_t_mean"])
            ps_overall = float(best_ps_overall["regret_per_t_mean"])
            summary[f"{label}_post_switch_regret"] = base_post
            summary[f"best_ps_post_minus_{label}"] = ps_post - base_post
            summary[f"{label}_regret_per_t"] = base_overall
            summary[f"best_ps_overall_minus_{label}"] = ps_overall - base_overall
        summaries.append(summary)
    return summaries


def aggregate(output_dir: Path, combos: list[tuple[int, float, float]], manifest: dict[str, Any]) -> None:
    rows = load_rows(output_dir, combos)
    add_ranks(rows)
    rows = sorted(rows, key=lambda row: (int(row["d"]), float(row["eta_sweep"]), float(row["epsilon_sweep"]), int(row["post_switch_rank"])))
    write_json(output_dir / "sweep_results_long.json", rows)
    write_csv(output_dir / "sweep_results_long.csv", rows)
    write_csv(output_dir / "sweep_results_long_primary_metrics.csv", rows, PRIMARY_FIELDS)

    by_method_post = best_by_method(rows, "post_switch")
    by_method_overall = best_by_method(rows, "overall")
    write_json(output_dir / "best_by_method_post_switch.json", by_method_post)
    write_csv(output_dir / "best_by_method_post_switch.csv", by_method_post)
    write_json(output_dir / "best_by_method_overall.json", by_method_overall)
    write_csv(output_dir / "best_by_method_overall.csv", by_method_overall)

    summaries = combo_summaries(rows)
    write_json(output_dir / "combo_summaries.json", summaries)
    write_csv(output_dir / "combo_summaries.csv", summaries)

    ps_wins_post = [row for row in summaries if row["best_post_switch_is_ps"]]
    ps_wins_overall = [row for row in summaries if row["best_overall_is_ps"]]
    top_ps_post = sorted(ps_wins_post, key=lambda row: (row["best_ps_post_switch_regret"], row["best_ps_tail20"], row["best_ps_regret_per_t"]))[:40]
    top_post_all = sorted(summaries, key=lambda row: (row["best_post_switch_regret"], row["best_ps_post_switch_regret"]))[:40]
    top_ps_overall = sorted(ps_wins_overall, key=lambda row: row["best_overall_regret_per_t"])[:40]
    write_csv(output_dir / "top_ps_winning_post_switch_configs.csv", top_ps_post)
    write_csv(output_dir / "top_post_switch_configs_all_methods.csv", top_post_all)
    write_csv(output_dir / "top_ps_winning_overall_configs.csv", top_ps_overall)

    compact_by_method_fields = [
        "best_post_switch_rank",
        "method",
        "d",
        "switch_episode",
        "eta_sweep",
        "epsilon_sweep",
        "post_switch_avg_regret_mean",
        "tail20_avg_total_cost_mean",
        "regret_per_t_mean",
        "overall_avg_total_cost_mean",
        "target_good_fraction_mean",
        "trap_basin_fraction_mean",
        "shared_path_fraction_mean",
    ]
    compact_overall_fields = [
        "best_overall_rank",
        "method",
        "d",
        "switch_episode",
        "eta_sweep",
        "epsilon_sweep",
        "regret_per_t_mean",
        "overall_avg_total_cost_mean",
        "post_switch_avg_regret_mean",
        "tail20_avg_total_cost_mean",
        "target_good_fraction_mean",
        "trap_basin_fraction_mean",
        "shared_path_fraction_mean",
    ]
    top_combo_fields = [
        "d",
        "switch_episode",
        "eta",
        "epsilon",
        "best_post_switch_method",
        "best_post_switch_regret",
        "best_ps_post_method",
        "best_ps_post_rank",
        "best_ps_post_switch_regret",
        "best_ps_post_minus_direct",
        "best_ps_post_minus_direct_local",
        "best_ps_post_minus_epsilon_exp3",
        "best_ps_post_minus_naive_mixed_avg",
        "best_ps_tail20",
        "best_ps_regret_per_t",
        "best_ps_target_good",
        "best_ps_trap",
    ]

    report_lines = [
        "# sim_v10 nonzero-eps d/eta/eps sweep full13 v1",
        "",
        "## Setting",
        "",
        f"- experiment_name: `{EXPERIMENT_NAME}`",
        f"- tree_spec: `{manifest['tree_spec']}`",
        "- tree_spec_role_mode: `spec_or_agent_id`",
        "- tree_spec_cost_mode: `ps_favored_trap_v10_avg_baited`",
        f"- d values: `{manifest['d_values']}`",
        f"- eta values: `{manifest['eta_values']}`",
        f"- epsilon values: `{manifest['epsilon_values']}`",
        f"- horizon: `{manifest['horizon']}`",
        f"- seeds: `{manifest['seeds']}`",
        f"- methods: `{len(METHODS)}`",
        "",
        "Primary ranking here is post-switch avg regret, then tail20 avg cost, then regret/T.",
        "",
        "## Coverage",
        "",
        f"- completed combos: `{len(summaries)}` / `{len(combos)}`",
        f"- rows: `{len(rows)}`",
        f"- post-switch PS-winning combos: `{len(ps_wins_post)}` / `{len(summaries)}`",
        f"- overall PS-winning combos: `{len(ps_wins_overall)}` / `{len(summaries)}`",
        "",
        "## Best By Method, Post-Switch Primary",
        "",
        markdown_table(by_method_post, compact_by_method_fields),
        "",
        "## Best By Method, Overall regret/T",
        "",
        markdown_table(by_method_overall, compact_overall_fields),
        "",
        "## Top PS-Winning Post-Switch Configs",
        "",
        markdown_table(top_ps_post[:20], top_combo_fields),
        "",
        "## Top Post-Switch Configs Across All Methods",
        "",
        markdown_table(top_post_all[:20], top_combo_fields),
        "",
        "## Top PS-Winning Overall Configs",
        "",
        markdown_table(
            top_ps_overall[:20],
            [
                "d",
                "switch_episode",
                "eta",
                "epsilon",
                "best_overall_method",
                "best_overall_regret_per_t",
                "best_ps_overall_method",
                "best_ps_overall_rank",
                "best_ps_overall_regret_per_t",
                "best_ps_overall_minus_direct",
                "best_ps_overall_minus_direct_local",
                "best_ps_overall_minus_epsilon_exp3",
                "best_ps_overall_minus_naive_mixed_avg",
            ],
        ),
        "",
        "## Output Files",
        "",
        "- `sweep_results_long.csv/json`: every method at every `(d, eta, eps)`",
        "- `sweep_results_long_primary_metrics.csv`: compact long table",
        "- `combo_summaries.csv/json`: one row per `(d, eta, eps)` with PS-vs-baseline margins",
        "- `best_by_method_post_switch.csv/json`: best setting for each method by post-switch metric",
        "- `best_by_method_overall.csv/json`: best setting for each method by regret/T",
        "- `top_ps_winning_post_switch_configs.csv`: PS-winning configs sorted by post-switch quality",
        "",
    ]
    (output_dir / "sweep_report.md").write_text("\n".join(report_lines), encoding="utf-8")

    status = {
        "experiment_name": EXPERIMENT_NAME,
        "completed_combos": len(summaries),
        "expected_combos": len(combos),
        "rows": len(rows),
        "post_switch_ps_winning_combos": len(ps_wins_post),
        "overall_ps_winning_combos": len(ps_wins_overall),
        "best_by_method_post_switch_top5": by_method_post[:5],
        "best_by_method_overall_top5": by_method_overall[:5],
        "top_ps_winning_post_switch_configs_top10": top_ps_post[:10],
    }
    write_json(output_dir / "sweep_status.json", status)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / EXPERIMENT_NAME)
    parser.add_argument("--tree-spec", type=Path, default=ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json")
    parser.add_argument("--d-values", default="5,6,7,8,10,12,16")
    parser.add_argument("--eta-values", default="0.05,0.1,0.15,0.2,0.3,0.4,0.5")
    parser.add_argument("--epsilon-values", default="0.005,0.01,0.02,0.05,0.1")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cost-noise", type=float, default=0.02)
    parser.add_argument("--specialist-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    d_values = parse_int_list(args.d_values)
    eta_values = parse_float_list(args.eta_values)
    eps_values = parse_float_list(args.epsilon_values)
    combos = [(d, eta, eps) for d in d_values for eta in eta_values for eps in eps_values]
    manifest = {
        "experiment_name": EXPERIMENT_NAME,
        "runner": str(RUNNER),
        "tree_spec": str(args.tree_spec),
        "tree_spec_role_mode": "spec_or_agent_id",
        "tree_spec_cost_mode": "ps_favored_trap_v10_avg_baited",
        "d_values": d_values,
        "eta_values": eta_values,
        "epsilon_values": eps_values,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "cost_noise": args.cost_noise,
        "specialist_fraction": args.specialist_fraction,
        "methods": METHODS,
        "combo_count": len(combos),
        "workers": args.workers,
    }
    write_json(output_dir / "sweep_manifest.json", manifest)

    if not args.aggregate_only:
        print(f"[sweep] experiment={EXPERIMENT_NAME}", flush=True)
        print(f"[sweep] combos={len(combos)} methods={len(METHODS)} workers={args.workers}", flush=True)
        started = time.time()
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_one_combo,
                    output_dir=output_dir,
                    tree_spec=args.tree_spec,
                    d=d,
                    eta=eta,
                    eps=eps,
                    horizon=args.horizon,
                    seeds=args.seeds,
                    cost_noise=args.cost_noise,
                    specialist_fraction=args.specialist_fraction,
                    force=args.force,
                )
                for d, eta, eps in combos
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                status = result["status"]
                print(
                    f"[sweep] {index:03d}/{len(futures)} {status} "
                    f"d={result['d']} eta={result['eta']:g} eps={result['eps']:g} "
                    f"elapsed={result.get('elapsed_sec', 0):.1f}s",
                    flush=True,
                )
                if status.startswith("failed"):
                    print(result.get("stderr_tail", ""), flush=True)
        elapsed = time.time() - started
        write_json(output_dir / "sweep_run_results.json", sorted(results, key=lambda row: (row["d"], row["eta"], row["eps"])))
        failures = [row for row in results if row["status"].startswith("failed")]
        print(f"[sweep] run elapsed={elapsed:.1f}s failures={len(failures)}", flush=True)
        if failures:
            print("[sweep] failures detected; aggregating completed runs and exiting non-zero", flush=True)
            aggregate(output_dir, combos, manifest)
            raise SystemExit(1)

    aggregate(output_dir, combos, manifest)
    print(f"[sweep] wrote report: {output_dir / 'sweep_report.md'}", flush=True)


if __name__ == "__main__":
    main()
