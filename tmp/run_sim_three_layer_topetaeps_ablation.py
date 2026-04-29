#!/usr/bin/env python3
"""Run the three-layer top-eta/eps sim ablation requested on 2026-04-27."""

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
DEFAULT_TREE_SPEC = ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup.json"

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

EXPERIMENTS = [
    {
        "name": "sim_v10_d2_4_top3_etaeps_full13_v1",
        "layer": "layer1_v10_single_switch_d2_4",
        "cost_mode": "ps_favored_trap_v10_avg_baited",
        "control_kind": "d",
        "control_values": [2, 3, 4],
        "description": "补齐 v10 single-switch d=2,3,4；不重复已跑过的 d=5,6。",
    },
    {
        "name": "sim_v11_cyclic_switch1_6_top3_etaeps_full13_v1",
        "layer": "layer2_v11_cyclic_switch1_6",
        "cost_mode": "ps_favored_trap_v11_cyclic_baited",
        "control_kind": "switch_count",
        "control_values": [1, 2, 3, 4, 5, 6],
        "description": "v11 cyclic switch_count=1..6 局部 eta/eps sweep。",
    },
    {
        "name": "sim_v12_gapcompressed_d4_7_top3_etaeps_full13_v1",
        "layer": "layer3_v12_gapcompressed_d4_7",
        "cost_mode": "ps_favored_trap_v12_gap_compressed_baited",
        "control_kind": "d",
        "control_values": [4, 5, 6, 7],
        "description": "v12 compression ablation: d=4,5,6,7，检验是否只是 n=7/eta=0.2/eps=0.01 参数不佳。",
    },
]


def float_tag(value: float) -> str:
    return f"{value:g}".replace(".", "p").replace("-", "m")


def parse_float_list(value: str) -> list[float]:
    return [float(part.strip()) for part in value.split(",") if part.strip()]


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


def metric(row: dict[str, Any], field: str, default: float = float("inf")) -> float:
    value = row.get(field)
    if value in (None, ""):
        return default
    return float(value)


def post_key(row: dict[str, Any]) -> tuple[float, float, float, str]:
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


def run_dir_for(base_output_dir: Path, experiment: dict[str, Any], control_value: int, eta: float, eps: float) -> Path:
    control_kind = experiment["control_kind"]
    return (
        base_output_dir
        / experiment["name"]
        / "runs"
        / f"{control_kind}_{control_value:02d}__eta_{float_tag(eta)}__eps_{float_tag(eps)}"
    )


def build_command(
    *,
    run_dir: Path,
    experiment: dict[str, Any],
    control_value: int,
    eta: float,
    eps: float,
    tree_spec: Path,
    horizon: int,
    seeds: list[int],
    cost_noise: float,
    specialist_fraction: float,
) -> list[str]:
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
        str(experiment["cost_mode"]),
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
    if experiment["control_kind"] == "d":
        command.extend(["--trap-switch-denominator", str(control_value)])
    elif experiment["control_kind"] == "switch_count":
        command.extend(["--cyclic-switch-count", str(control_value)])
    else:
        raise ValueError(f"Unknown control kind: {experiment['control_kind']}")
    return command


def run_one(
    *,
    base_output_dir: Path,
    experiment: dict[str, Any],
    control_value: int,
    eta: float,
    eps: float,
    tree_spec: Path,
    horizon: int,
    seeds: list[int],
    cost_noise: float,
    specialist_fraction: float,
    force: bool,
) -> dict[str, Any]:
    run_dir = run_dir_for(base_output_dir, experiment, control_value, eta, eps)
    compare_path = run_dir / "controlled_sim_compare.json"
    if compare_path.exists() and not force:
        return {
            "status": "skipped",
            "experiment": experiment["name"],
            "layer": experiment["layer"],
            "control_kind": experiment["control_kind"],
            "control_value": control_value,
            "eta": eta,
            "eps": eps,
            "run_dir": str(run_dir),
        }
    run_dir.mkdir(parents=True, exist_ok=True)
    command = build_command(
        run_dir=run_dir,
        experiment=experiment,
        control_value=control_value,
        eta=eta,
        eps=eps,
        tree_spec=tree_spec,
        horizon=horizon,
        seeds=seeds,
        cost_noise=cost_noise,
        specialist_fraction=specialist_fraction,
    )
    started = time.time()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, check=False)
    elapsed = time.time() - started
    (run_dir / "run.log").write_text(
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
            "experiment": experiment["name"],
            "layer": experiment["layer"],
            "control_kind": experiment["control_kind"],
            "control_value": control_value,
            "eta": eta,
            "eps": eps,
            "elapsed_sec": elapsed,
            "run_dir": str(run_dir),
            "stderr_tail": "\n".join(completed.stderr.splitlines()[-20:]),
        }
    if not compare_path.exists():
        return {
            "status": "failed_missing_compare",
            "experiment": experiment["name"],
            "layer": experiment["layer"],
            "control_kind": experiment["control_kind"],
            "control_value": control_value,
            "eta": eta,
            "eps": eps,
            "elapsed_sec": elapsed,
            "run_dir": str(run_dir),
        }
    return {
        "status": "completed",
        "experiment": experiment["name"],
        "layer": experiment["layer"],
        "control_kind": experiment["control_kind"],
        "control_value": control_value,
        "eta": eta,
        "eps": eps,
        "elapsed_sec": elapsed,
        "run_dir": str(run_dir),
    }


def load_long_rows(base_output_dir: Path, combos: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for combo in combos:
        experiment = combo["experiment"]
        run_dir = run_dir_for(base_output_dir, experiment, combo["control_value"], combo["eta"], combo["eps"])
        compare_path = run_dir / "controlled_sim_compare.json"
        if not compare_path.exists():
            continue
        source_rows = json.loads(compare_path.read_text(encoding="utf-8"))
        for source in source_rows:
            row = dict(source)
            row["experiment_name"] = experiment["name"]
            row["layer"] = experiment["layer"]
            row["control_kind"] = experiment["control_kind"]
            row["control_value"] = combo["control_value"]
            row["eta_sweep"] = combo["eta"]
            row["epsilon_sweep"] = combo["eps"]
            row["run_dir"] = str(run_dir)
            row["is_ps_family"] = row.get("method") in PS_FAMILY
            rows.append(row)
    return rows


def add_ranks(rows: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, int, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["experiment_name"]),
                int(row["control_value"]),
                float(row["eta_sweep"]),
                float(row["epsilon_sweep"]),
            ),
            [],
        ).append(row)
    for group_rows in groups.values():
        for rank, row in enumerate(sorted(group_rows, key=post_key), start=1):
            row["post_switch_rank"] = rank
        for rank, row in enumerate(sorted(group_rows, key=overall_key), start=1):
            row["overall_rank"] = rank


def combo_summaries(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int, float, float], list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(
            (
                str(row["experiment_name"]),
                int(row["control_value"]),
                float(row["eta_sweep"]),
                float(row["epsilon_sweep"]),
            ),
            [],
        ).append(row)
    summaries: list[dict[str, Any]] = []
    for (experiment_name, control_value, eta, eps), group_rows in sorted(groups.items()):
        by_method = {row["method"]: row for row in group_rows}
        best_post = sorted(group_rows, key=post_key)[0]
        best_overall = sorted(group_rows, key=overall_key)[0]
        ps_rows = [row for row in group_rows if row["method"] in PS_FAMILY]
        best_ps_post = sorted(ps_rows, key=post_key)[0]
        best_ps_overall = sorted(ps_rows, key=overall_key)[0]
        layer = str(best_post["layer"])
        control_kind = str(best_post["control_kind"])
        summary: dict[str, Any] = {
            "experiment_name": experiment_name,
            "layer": layer,
            "control_kind": control_kind,
            "control_value": control_value,
            "eta": eta,
            "epsilon": eps,
            "trap_switch_episode": best_post.get("trap_switch_episode"),
            "cyclic_switch_count": best_post.get("cyclic_effective_switch_count") or best_post.get("cyclic_switch_count"),
            "cyclic_switch_episodes": best_post.get("cyclic_switch_episodes"),
            "post_switch_start_episode": best_post.get("post_switch_start_episode"),
            "post_switch_episode_count": best_post.get("post_switch_episode_count"),
            "best_post_method": best_post["method"],
            "best_post_is_ps": best_post["method"] in PS_FAMILY,
            "best_post_value": best_post["post_switch_avg_regret_mean"],
            "best_overall_method": best_overall["method"],
            "best_overall_is_ps": best_overall["method"] in PS_FAMILY,
            "best_overall_regret_per_t": best_overall["regret_per_t_mean"],
            "best_ps_method": best_ps_post["method"],
            "best_ps_post_rank": best_ps_post["post_switch_rank"],
            "best_ps_post_value": best_ps_post["post_switch_avg_regret_mean"],
            "best_ps_regret_per_t": best_ps_post["regret_per_t_mean"],
            "best_ps_tail20": best_ps_post["tail20_avg_total_cost_mean"],
            "best_ps_target_good_fraction": best_ps_post["target_good_fraction_mean"],
            "best_ps_trap_basin_fraction": best_ps_post["trap_basin_fraction_mean"],
            "best_ps_shared_path_fraction": best_ps_post["shared_path_fraction_mean"],
            "best_ps_exact_hit": best_ps_post["exact_best_path_hit_rate_mean"],
            "best_ps_overall_method": best_ps_overall["method"],
            "best_ps_overall_rank": best_ps_overall["overall_rank"],
            "best_ps_overall_regret_per_t": best_ps_overall["regret_per_t_mean"],
        }
        nonps_rows = [row for row in group_rows if row["method"] not in PS_FAMILY]
        best_nonps = sorted(nonps_rows, key=post_key)[0]
        summary["best_nonps_method"] = best_nonps["method"]
        summary["best_nonps_post_value"] = best_nonps["post_switch_avg_regret_mean"]
        summary["ps_minus_best_nonps_post"] = float(best_ps_post["post_switch_avg_regret_mean"]) - float(best_nonps["post_switch_avg_regret_mean"])
        for method in ["direct_multistage_exp3", "direct_multistage_exp3_local", "epsilon_exp3", "naive_mixed_avg"]:
            baseline = by_method.get(method)
            if baseline is None:
                continue
            summary[f"{method}_post_rank"] = baseline["post_switch_rank"]
            summary[f"{method}_post_value"] = baseline["post_switch_avg_regret_mean"]
            summary[f"ps_minus_{method}_post"] = float(best_ps_post["post_switch_avg_regret_mean"]) - float(baseline["post_switch_avg_regret_mean"])
            summary[f"{method}_overall_rank"] = baseline["overall_rank"]
            summary[f"{method}_regret_per_t"] = baseline["regret_per_t_mean"]
            summary[f"ps_overall_minus_{method}"] = float(best_ps_overall["regret_per_t_mean"]) - float(baseline["regret_per_t_mean"])
        summaries.append(summary)
    return summaries


def best_by_method(rows: list[dict[str, Any]], experiment_name: str, use_overall: bool = False) -> list[dict[str, Any]]:
    key_fn = overall_key if use_overall else post_key
    exp_rows = [row for row in rows if row["experiment_name"] == experiment_name]
    output: list[dict[str, Any]] = []
    for method in METHODS:
        method_rows = [row for row in exp_rows if row["method"] == method]
        if method_rows:
            output.append(dict(sorted(method_rows, key=key_fn)[0]))
    output = sorted(output, key=key_fn)
    rank_field = "best_overall_rank" if use_overall else "best_post_rank"
    for rank, row in enumerate(output, start=1):
        row[rank_field] = rank
    return output


def aggregate(base_output_dir: Path, combos: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    all_rows = load_long_rows(base_output_dir, combos)
    add_ranks(all_rows)
    all_rows = sorted(
        all_rows,
        key=lambda row: (
            str(row["experiment_name"]),
            int(row["control_value"]),
            float(row["eta_sweep"]),
            float(row["epsilon_sweep"]),
            int(row["post_switch_rank"]),
        ),
    )
    summaries = combo_summaries(all_rows)
    summary_fields = None

    write_json(base_output_dir / "all_layers_long.json", all_rows)
    write_csv(base_output_dir / "all_layers_long.csv", all_rows)
    write_json(base_output_dir / "all_layers_combo_summaries.json", summaries)
    write_csv(base_output_dir / "all_layers_combo_summaries.csv", summaries, summary_fields)

    for experiment in EXPERIMENTS:
        exp_dir = base_output_dir / experiment["name"]
        exp_rows = [row for row in all_rows if row["experiment_name"] == experiment["name"]]
        exp_summaries = [row for row in summaries if row["experiment_name"] == experiment["name"]]
        write_json(exp_dir / "sweep_results_long.json", exp_rows)
        write_csv(exp_dir / "sweep_results_long.csv", exp_rows)
        write_json(exp_dir / "combo_summaries.json", exp_summaries)
        write_csv(exp_dir / "combo_summaries.csv", exp_summaries)
        post_best = best_by_method(all_rows, experiment["name"], use_overall=False)
        overall_best = best_by_method(all_rows, experiment["name"], use_overall=True)
        write_csv(exp_dir / "best_by_method_post_switch.csv", post_best)
        write_json(exp_dir / "best_by_method_post_switch.json", post_best)
        write_csv(exp_dir / "best_by_method_overall.csv", overall_best)
        write_json(exp_dir / "best_by_method_overall.json", overall_best)

    top_fields = [
        "experiment_name",
        "control_kind",
        "control_value",
        "eta",
        "epsilon",
        "best_post_method",
        "best_post_value",
        "best_ps_method",
        "best_ps_post_rank",
        "best_ps_post_value",
        "ps_minus_best_nonps_post",
        "best_nonps_method",
        "ps_minus_direct_multistage_exp3_post",
        "ps_minus_direct_multistage_exp3_local_post",
        "ps_minus_epsilon_exp3_post",
        "ps_minus_naive_mixed_avg_post",
        "best_ps_regret_per_t",
        "best_ps_tail20",
        "best_ps_target_good_fraction",
        "best_ps_trap_basin_fraction",
        "best_ps_shared_path_fraction",
        "best_ps_exact_hit",
    ]
    ps_wins = [row for row in summaries if row["best_post_is_ps"]]
    top_ps_wins = sorted(ps_wins, key=lambda row: (float(row["best_ps_post_value"]), float(row["best_ps_regret_per_t"])))
    all_top = sorted(summaries, key=lambda row: (float(row["best_post_value"]), float(row["best_ps_post_value"])))
    write_csv(base_output_dir / "top_ps_winning_configs.csv", top_ps_wins, top_fields)
    write_csv(base_output_dir / "top_all_configs.csv", all_top, top_fields)

    report_lines = [
        "# Three-layer top eta/eps sim ablation",
        "",
        "## Setting",
        "",
        f"- output_dir: `{base_output_dir}`",
        f"- tree_spec: `{manifest['tree_spec']}`",
        f"- eta_values: `{manifest['eta_values']}`",
        f"- epsilon_values: `{manifest['epsilon_values']}`",
        f"- horizon: `{manifest['horizon']}`",
        f"- seeds: `{manifest['seeds']}`",
        f"- methods: `{len(METHODS)}`",
        f"- completed combo rows: `{len(summaries)}` / `{len(combos)}`",
        f"- PS-winning combos by post-switch: `{len(ps_wins)}` / `{len(summaries)}`",
        "",
        "Primary ranking: post_switch_avg_regret_mean, then tail20_avg_total_cost_mean, then regret_per_t_mean.",
        "",
    ]
    for experiment in EXPERIMENTS:
        exp_summaries = [row for row in summaries if row["experiment_name"] == experiment["name"]]
        exp_ps_wins = [row for row in exp_summaries if row["best_post_is_ps"]]
        report_lines.extend(
            [
                f"## {experiment['name']}",
                "",
                experiment["description"],
                "",
                f"- cost_mode: `{experiment['cost_mode']}`",
                f"- control_kind: `{experiment['control_kind']}`",
                f"- control_values: `{experiment['control_values']}`",
                f"- combos: `{len(exp_summaries)}`",
                f"- PS-winning combos: `{len(exp_ps_wins)}` / `{len(exp_summaries)}`",
                "",
                "### Top PS-winning configs",
                "",
                markdown_table(
                    sorted(exp_ps_wins, key=lambda row: (float(row["best_ps_post_value"]), float(row["best_ps_regret_per_t"])))[:20],
                    top_fields,
                ),
                "",
                "### Top configs across all methods",
                "",
                markdown_table(
                    sorted(exp_summaries, key=lambda row: (float(row["best_post_value"]), float(row["best_ps_post_value"])))[:20],
                    top_fields,
                ),
                "",
                "### Best by method, post-switch",
                "",
                markdown_table(
                    best_by_method(all_rows, experiment["name"], use_overall=False),
                    [
                        "best_post_rank",
                        "method",
                        "control_value",
                        "eta_sweep",
                        "epsilon_sweep",
                        "post_switch_avg_regret_mean",
                        "tail20_avg_total_cost_mean",
                        "regret_per_t_mean",
                        "target_good_fraction_mean",
                        "trap_basin_fraction_mean",
                        "shared_path_fraction_mean",
                        "exact_best_path_hit_rate_mean",
                    ],
                ),
                "",
            ]
        )
    report_lines.extend(
        [
            "## Global Top PS-winning configs",
            "",
            markdown_table(top_ps_wins[:30], top_fields),
            "",
            "## Files",
            "",
            "- `all_layers_long.csv/json`",
            "- `all_layers_combo_summaries.csv/json`",
            "- `top_ps_winning_configs.csv`",
            "- per-layer `combo_summaries.csv`, `best_by_method_post_switch.csv`, `best_by_method_overall.csv`",
            "",
        ]
    )
    (base_output_dir / "three_layer_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    write_json(
        base_output_dir / "three_layer_status.json",
        {
            "combo_count": len(combos),
            "completed_combo_count": len(summaries),
            "row_count": len(all_rows),
            "ps_winning_combo_count": len(ps_wins),
            "ps_winning_by_experiment": {
                experiment["name"]: sum(1 for row in summaries if row["experiment_name"] == experiment["name"] and row["best_post_is_ps"])
                for experiment in EXPERIMENTS
            },
            "top_ps_winning_configs": top_ps_wins[:20],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs" / "sim_three_layer_topetaeps_ablation_v1")
    parser.add_argument("--tree-spec", type=Path, default=DEFAULT_TREE_SPEC)
    parser.add_argument("--eta-values", default="0.2,0.3,0.4")
    parser.add_argument("--epsilon-values", default="0.005,0.01,0.02")
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--cost-noise", type=float, default=0.02)
    parser.add_argument("--specialist-fraction", type=float, default=0.15)
    parser.add_argument("--workers", type=int, default=min(10, max(1, (os.cpu_count() or 4) - 2)))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    args = parser.parse_args()

    eta_values = parse_float_list(args.eta_values)
    epsilon_values = parse_float_list(args.epsilon_values)
    combos: list[dict[str, Any]] = []
    for experiment in EXPERIMENTS:
        for control_value in experiment["control_values"]:
            for eta in eta_values:
                for eps in epsilon_values:
                    combos.append(
                        {
                            "experiment": experiment,
                            "control_value": control_value,
                            "eta": eta,
                            "eps": eps,
                        }
                    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "runner": str(RUNNER),
        "tree_spec": str(args.tree_spec),
        "eta_values": eta_values,
        "epsilon_values": epsilon_values,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "cost_noise": args.cost_noise,
        "specialist_fraction": args.specialist_fraction,
        "methods": METHODS,
        "experiments": EXPERIMENTS,
        "combo_count": len(combos),
        "workers": args.workers,
    }
    write_json(args.output_dir / "three_layer_manifest.json", manifest)

    if not args.aggregate_only:
        print(f"[three-layer] combos={len(combos)} workers={args.workers}", flush=True)
        started = time.time()
        results: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    run_one,
                    base_output_dir=args.output_dir,
                    experiment=combo["experiment"],
                    control_value=combo["control_value"],
                    eta=combo["eta"],
                    eps=combo["eps"],
                    tree_spec=args.tree_spec,
                    horizon=args.horizon,
                    seeds=args.seeds,
                    cost_noise=args.cost_noise,
                    specialist_fraction=args.specialist_fraction,
                    force=args.force,
                )
                for combo in combos
            ]
            for index, future in enumerate(as_completed(futures), start=1):
                result = future.result()
                results.append(result)
                print(
                    f"[three-layer] {index:03d}/{len(futures)} {result['status']} "
                    f"{result['experiment']} {result['control_kind']}={result['control_value']} "
                    f"eta={result['eta']:g} eps={result['eps']:g} "
                    f"elapsed={result.get('elapsed_sec', 0):.1f}s",
                    flush=True,
                )
                if str(result["status"]).startswith("failed"):
                    print(result.get("stderr_tail", ""), flush=True)
        elapsed = time.time() - started
        write_json(args.output_dir / "three_layer_run_results.json", results)
        failures = [row for row in results if str(row["status"]).startswith("failed")]
        print(f"[three-layer] elapsed={elapsed:.1f}s failures={len(failures)}", flush=True)
        if failures:
            aggregate(args.output_dir, combos, manifest)
            raise SystemExit(1)
    aggregate(args.output_dir, combos, manifest)
    print(f"[three-layer] wrote {args.output_dir / 'three_layer_report.md'}", flush=True)


if __name__ == "__main__":
    main()
