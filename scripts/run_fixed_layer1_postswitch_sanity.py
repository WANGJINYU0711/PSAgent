"""Layer-1 post-switch probability-freeze sanity check.

This diagnostic reuses the controlled BarrierShare simulator and freezes only
the root/direct-child sampling distribution after the first switch. Policy
updates continue normally; the wrapper only ignores later root probability
changes while leaving downstream choices adaptive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import MethodType
from typing import Any, Iterable

from run_barriershare_controlled_sim import (
    METHODS,
    PS_FAVORED_COST_MODES,
    ROOT,
    SpecBackedControlledTreeEnv,
    build_instances,
    install_fast_child_prefix_helper,
)


METHOD_SPECS: dict[str, dict[str, Any]] = {
    "eps": {
        "method": "epsilon_exp3",
        "policy_kwargs": {"eta": 0.3, "epsilon": 0.01},
    },
    "exp": {
        "method": "direct_multistage_exp3",
        "policy_kwargs": {"eta": 0.3},
    },
    "ps": {
        "method": "risky_ps_linear",
        "policy_kwargs": {"eta": 0.3, "epsilon": 0.01},
    },
}
REGIMES = ("dynamic", "freeze_layer1_post_switch")


def mean(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.fmean(rows) if rows else 0.0


def stdev(values: Iterable[float]) -> float:
    rows = list(values)
    return statistics.stdev(rows) if len(rows) > 1 else 0.0


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        json.dumps(value, ensure_ascii=False)
                        if isinstance(value, (dict, list, tuple))
                        else value
                    )
                    for key, value in row.items()
                }
            )


def sample_index(probs: list[float], draw: float) -> int:
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if draw <= cumulative:
            return idx
    return len(probs) - 1


def normalize_probs(probs: list[float]) -> list[float]:
    cleaned = [max(0.0, float(prob)) for prob in probs]
    total = sum(cleaned)
    if total <= 0.0:
        return [1.0 / len(cleaned) for _ in cleaned]
    return [prob / total for prob in cleaned]


def root_child_prefixes(policy: Any, env: SpecBackedControlledTreeEnv) -> list[tuple[str, ...]]:
    stage_name = env.STAGE_NAMES[0]
    if hasattr(policy, "_sample_stage_child"):
        agent_ids = policy._legal_agent_ids_for_prefix((), stage_name, env)
        return list(policy._child_prefixes((), agent_ids))
    return list(policy._child_prefixes((), stage_name, env))


def root_distribution(policy: Any, env: SpecBackedControlledTreeEnv) -> tuple[list[tuple[str, ...]], list[float], str]:
    child_prefixes = root_child_prefixes(policy, env)
    if not child_prefixes:
        raise RuntimeError("No root children available for layer-1 freeze diagnostic.")

    if hasattr(policy, "_stage_probs"):
        probs = normalize_probs(list(policy._stage_probs((), child_prefixes)))
        return child_prefixes, probs, "stagewise_marginal_mixture"

    if getattr(policy, "safe_prefixes", {}).get((), False):
        probs = normalize_probs(list(policy._safe_child_probs((), child_prefixes)))
        return child_prefixes, probs, "ps_safe_prefix_mass"

    exploit_probs = list(policy._risky_child_probs((), child_prefixes))
    epsilon = min(1.0, max(0.0, float(getattr(policy, "epsilon", 0.0))))
    uniform = 1.0 / len(child_prefixes)
    probs = normalize_probs(
        [(1.0 - epsilon) * prob + epsilon * uniform for prob in exploit_probs]
    )
    return child_prefixes, probs, "ps_risky_marginal_mixture"


def distribution_rows(
    *,
    env: SpecBackedControlledTreeEnv,
    experiment_name: str,
    seed: int,
    method_short: str,
    method: str,
    regime: str,
    snapshot_episode_index: int,
    child_prefixes: list[tuple[str, ...]],
    probs: list[float],
    distribution_kind: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank, (child_prefix, prob) in enumerate(zip(child_prefixes, probs), start=1):
        child_id = child_prefix[-1]
        rows.append(
            {
                "experiment": experiment_name,
                "seed": seed,
                "method_short": method_short,
                "method": method,
                "regime": regime,
                "snapshot_episode_index": snapshot_episode_index,
                "snapshot_episode_1based": snapshot_episode_index + 1,
                "rank": rank,
                "child_id": child_id,
                "base_alias": getattr(env, "base_alias_by_agent", {}).get(child_id, child_id),
                "prob": prob,
                "distribution_kind": distribution_kind,
            }
        )
    return rows


def install_root_freeze_wrapper(policy: Any) -> None:
    if getattr(policy, "_root_freeze_wrapper_installed", False):
        return

    if hasattr(policy, "_sample_stage_child"):
        original = policy._sample_stage_child

        def sample_stage_child_with_freeze(self: Any, current_prefix: tuple[str, ...], child_prefixes: list[tuple[str, ...]]) -> tuple[tuple[str, ...], dict[str, Any]]:
            if getattr(self, "_freeze_root_active", False) and tuple(current_prefix) == ():
                frozen = getattr(self, "_frozen_root_probs", None)
                if frozen:
                    probs = [float(frozen[tuple(child_prefix)]) for child_prefix in child_prefixes]
                    probs = normalize_probs(probs)
                    selected_idx = sample_index(probs, self.rng.random())
                    selected = child_prefixes[selected_idx]
                    prob = probs[selected_idx]
                    return selected, {
                        "epsilon": getattr(self, "epsilon", None),
                        "epsilon_mode": "F",
                        "selection_mode": "frozen_root_post_switch",
                        "branch_conditional_prob": prob,
                        "conditional_prob": prob,
                        "mixture_conditional_prob": prob,
                        "softmax_conditional_prob": None,
                        "uniform_conditional_prob": None,
                    }
            return original(current_prefix, child_prefixes)

        policy._sample_stage_child = MethodType(sample_stage_child_with_freeze, policy)
    elif hasattr(policy, "_sample_child_prefix"):
        original = policy._sample_child_prefix

        def sample_child_prefix_with_freeze(self: Any, current_prefix: tuple[str, ...], stage_name: str, env: SpecBackedControlledTreeEnv) -> tuple[tuple[str, ...], float, dict[str, Any]]:
            if getattr(self, "_freeze_root_active", False) and tuple(current_prefix) == ():
                child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
                frozen = getattr(self, "_frozen_root_probs", None)
                if frozen:
                    probs = [float(frozen[tuple(child_prefix)]) for child_prefix in child_prefixes]
                    probs = normalize_probs(probs)
                    selected_idx = sample_index(probs, self.rng.random())
                    selected = child_prefixes[selected_idx]
                    prob = probs[selected_idx]
                    return selected, prob, {
                        "epsilon": getattr(self, "epsilon", None),
                        "epsilon_mode": "F",
                        "selection_mode": "frozen_root_post_switch",
                        "branch_conditional_prob": prob,
                        "conditional_prob": prob,
                        "mixture_conditional_prob": prob,
                        "softmax_conditional_prob": None,
                        "uniform_conditional_prob": None,
                        "estimated_loss_denominator": "branch_edge_prob",
                        "estimator_scope": "frozen_root_post_switch_branch_probability",
                    }
            return original(current_prefix, stage_name, env)

        policy._sample_child_prefix = MethodType(sample_child_prefix_with_freeze, policy)
    else:
        raise TypeError(f"Unsupported policy type for root freeze: {type(policy).__name__}")

    policy._root_freeze_wrapper_installed = True
    policy._freeze_root_active = False
    policy._frozen_root_probs = None


def classify_path(env: SpecBackedControlledTreeEnv, path: list[str], episode_index: int, horizon: int) -> dict[str, Any]:
    profile = env.path_profiles[tuple(path)]
    phase = (
        env._ps_favored_cyclic_phase(episode_index, horizon)
        if getattr(env, "tree_spec_cost_mode", "") == "ps_favored_trap_v11_cyclic_baited"
        else None
    )
    is_candidate = env._is_ps_favored_candidate_safe_subtree(profile.base_aliases, profile.gates)
    if phase is None:
        is_target_good = env._is_ps_favored_near_best_good(tuple(path), profile.base_aliases, profile.gates)
        is_exact_best = env._is_ps_favored_exact_best(tuple(path))
    else:
        is_target_good = env._is_ps_favored_v11_phase_target_good(profile.base_aliases, profile.gates, phase)
        is_exact_best = env._is_ps_favored_v11_phase_exact_best(profile.base_aliases, profile.gates, phase)
    return {
        "root_child": path[0] if path else None,
        "root_base_alias": profile.base_aliases[0] if profile.base_aliases else None,
        "base_alias_path": list(profile.base_aliases),
        "family_label": profile.family_label,
        "trap_basin": env._is_ps_favored_trap_basin(profile.base_aliases),
        "target_subtree": is_candidate,
        "target_good": is_target_good,
        "target_bad": bool(is_candidate and not is_target_good),
        "decoy_branch": env._is_ps_favored_decoy_branch(profile.base_aliases, profile.gates),
        "broad_safe_basin": env._is_ps_favored_safe_suffix(profile.base_aliases, profile.gates),
        "exact_best": is_exact_best,
    }


def run_case(args: tuple[Any, ...]) -> dict[str, Any]:
    (
        experiment_name,
        tree_spec,
        tree_spec_cost_mode,
        trap_switch_denominator,
        horizon,
        specialist_fraction,
        cost_noise,
        seed,
        method_short,
        regime,
    ) = args
    spec_path = Path(tree_spec)
    method_spec = METHOD_SPECS[method_short]
    method = method_spec["method"]
    policy_kwargs = dict(method_spec["policy_kwargs"])

    env = SpecBackedControlledTreeEnv(
        spec_path=spec_path,
        seed=seed,
        cost_noise=cost_noise,
        specialist_fraction=specialist_fraction,
        tree_spec_role_mode="spec_or_agent_id",
        tree_spec_cost_mode=tree_spec_cost_mode,
        trap_switch_denominator=trap_switch_denominator,
    )
    env.setting_group = "external_tree"
    env.setting_risky_depth = None

    instances = build_instances(
        horizon=horizon,
        seed=seed,
        specialist_fraction=specialist_fraction,
    )
    oracle = env.oracle_reference(instances)
    policy = METHODS[method](seed=seed, **policy_kwargs)
    policy.bind_env(env)
    install_fast_child_prefix_helper(policy, env)
    policy.reset()
    install_root_freeze_wrapper(policy)

    trap_switch_episode = env._ps_favored_trap_switch_episode(horizon)
    post_start = min(max(trap_switch_episode, 1), horizon)
    episode_rows: list[dict[str, Any]] = []
    freeze_prob_rows: list[dict[str, Any]] = []
    costs: list[float] = []

    for episode_index, instance in enumerate(instances):
        if episode_index == post_start:
            child_prefixes, probs, distribution_kind = root_distribution(policy, env)
            freeze_prob_rows.extend(
                distribution_rows(
                    env=env,
                    experiment_name=experiment_name,
                    seed=seed,
                    method_short=method_short,
                    method=method,
                    regime=regime,
                    snapshot_episode_index=episode_index,
                    child_prefixes=child_prefixes,
                    probs=probs,
                    distribution_kind=distribution_kind,
                )
            )
            if regime == "freeze_layer1_post_switch":
                policy._frozen_root_probs = {
                    tuple(child_prefix): float(prob)
                    for child_prefix, prob in zip(child_prefixes, probs)
                }
                policy._freeze_root_active = True

        path = policy.select_path(instance, env)
        env.reset(instance)
        result = env.run_path(path)
        policy.update(result)
        observed_cost = float(result.total_cost)
        costs.append(observed_cost)
        path_info = classify_path(env, path, episode_index, horizon)
        state = policy.get_state() if hasattr(policy, "get_state") else {}
        selection_info = (
            policy.get_last_selection_info()
            if hasattr(policy, "get_last_selection_info")
            else state
        )
        selected_edges = (
            selection_info.get("selected_edges")
            or selection_info.get("sampled_edges")
            or state.get("last_selected_edges")
            or state.get("last_sampled_edges")
            or []
        )
        root_edge = selected_edges[0] if selected_edges else {}
        episode_rows.append(
            {
                "experiment": experiment_name,
                "seed": seed,
                "method_short": method_short,
                "method": method,
                "regime": regime,
                "episode_index": episode_index,
                "episode_1based": episode_index + 1,
                "phase": "pre_switch" if episode_index < post_start else "post_switch",
                "switch_episode_index": post_start,
                "switch_episode_1based": post_start + 1,
                "cost": observed_cost,
                "oracle_cost": oracle["oracle_episode_costs"][episode_index],
                "regret": observed_cost - float(oracle["oracle_episode_costs"][episode_index]),
                "path": list(path),
                "path_prob": getattr(policy, "last_path_prob", None),
                "root_prob_used": root_edge.get("conditional_prob"),
                "root_selection_mode": root_edge.get("selection_mode"),
                "root_child": path_info["root_child"],
                "root_base_alias": path_info["root_base_alias"],
                "base_alias_path": path_info["base_alias_path"],
                "family_label": path_info["family_label"],
                "trap_basin": path_info["trap_basin"],
                "target_subtree": path_info["target_subtree"],
                "target_good": path_info["target_good"],
                "target_bad": path_info["target_bad"],
                "decoy_branch": path_info["decoy_branch"],
                "broad_safe_basin": path_info["broad_safe_basin"],
                "exact_best": path_info["exact_best"],
            }
        )

    pre_costs = costs[:post_start]
    post_costs = costs[post_start:]
    summary = {
        "experiment": experiment_name,
        "seed": seed,
        "method_short": method_short,
        "method": method,
        "regime": regime,
        "horizon": horizon,
        "pre_episode_count": len(pre_costs),
        "post_episode_count": len(post_costs),
        "switch_episode_index": post_start,
        "switch_episode_1based": post_start + 1,
        "pre_cumulative_cost": sum(pre_costs),
        "pre_cumulative_cost_avg": mean(pre_costs),
        "post_cost": sum(post_costs),
        "post_cost_avg": mean(post_costs),
        "total_cost": sum(costs),
        "total_cost_avg": mean(costs),
        "oracle_pre_cost_avg": mean(oracle["oracle_episode_costs"][:post_start]),
        "oracle_post_cost_avg": mean(oracle["oracle_episode_costs"][post_start:]),
        "oracle_total_cost_avg": mean(oracle["oracle_episode_costs"]),
        "pre_regret_avg": mean(
            row["regret"] for row in episode_rows if row["phase"] == "pre_switch"
        ),
        "post_regret_avg": mean(
            row["regret"] for row in episode_rows if row["phase"] == "post_switch"
        ),
        "total_regret_avg": mean(row["regret"] for row in episode_rows),
        "post_target_good_fraction": mean(
            1.0 if row["target_good"] else 0.0
            for row in episode_rows
            if row["phase"] == "post_switch"
        ),
        "post_target_bad_fraction": mean(
            1.0 if row["target_bad"] else 0.0
            for row in episode_rows
            if row["phase"] == "post_switch"
        ),
        "post_trap_basin_fraction": mean(
            1.0 if row["trap_basin"] else 0.0
            for row in episode_rows
            if row["phase"] == "post_switch"
        ),
    }
    return {
        "summary": summary,
        "episodes": episode_rows,
        "freeze_probs": freeze_prob_rows,
    }


def aggregate_summaries(per_seed: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in per_seed:
        groups.setdefault((row["method_short"], row["regime"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for (method_short, regime), rows in sorted(groups.items()):
        summaries.append(
            {
                "method_short": method_short,
                "method": rows[0]["method"],
                "regime": regime,
                "seeds": len(rows),
                "horizon": rows[0]["horizon"],
                "pre_episode_count": rows[0]["pre_episode_count"],
                "post_episode_count": rows[0]["post_episode_count"],
                "switch_episode_1based": rows[0]["switch_episode_1based"],
                "pre_cumulative_cost_avg_mean": mean(row["pre_cumulative_cost_avg"] for row in rows),
                "pre_cumulative_cost_avg_std": stdev(row["pre_cumulative_cost_avg"] for row in rows),
                "post_cost_avg_mean": mean(row["post_cost_avg"] for row in rows),
                "post_cost_avg_std": stdev(row["post_cost_avg"] for row in rows),
                "total_cost_avg_mean": mean(row["total_cost_avg"] for row in rows),
                "total_cost_avg_std": stdev(row["total_cost_avg"] for row in rows),
                "post_regret_avg_mean": mean(row["post_regret_avg"] for row in rows),
                "total_regret_avg_mean": mean(row["total_regret_avg"] for row in rows),
                "post_target_good_fraction_mean": mean(row["post_target_good_fraction"] for row in rows),
                "post_target_bad_fraction_mean": mean(row["post_target_bad_fraction"] for row in rows),
                "post_trap_basin_fraction_mean": mean(row["post_trap_basin_fraction"] for row in rows),
            }
        )
    return summaries


def add_delta_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, dict[str, dict[str, Any]]] = {}
    for row in summary:
        by_method.setdefault(row["method_short"], {})[row["regime"]] = row
    rows: list[dict[str, Any]] = []
    for method_short, regimes in sorted(by_method.items()):
        dyn = regimes.get("dynamic")
        frozen = regimes.get("freeze_layer1_post_switch")
        if dyn is None or frozen is None:
            continue
        rows.append(
            {
                "method_short": method_short,
                "method": dyn["method"],
                "delta_definition": "freeze_minus_dynamic",
                "pre_cumulative_cost_avg_delta": (
                    frozen["pre_cumulative_cost_avg_mean"]
                    - dyn["pre_cumulative_cost_avg_mean"]
                ),
                "post_cost_avg_delta": (
                    frozen["post_cost_avg_mean"] - dyn["post_cost_avg_mean"]
                ),
                "total_cost_avg_delta": (
                    frozen["total_cost_avg_mean"] - dyn["total_cost_avg_mean"]
                ),
                "post_target_good_fraction_delta": (
                    frozen["post_target_good_fraction_mean"]
                    - dyn["post_target_good_fraction_mean"]
                ),
                "post_target_bad_fraction_delta": (
                    frozen["post_target_bad_fraction_mean"]
                    - dyn["post_target_bad_fraction_mean"]
                ),
                "post_trap_basin_fraction_delta": (
                    frozen["post_trap_basin_fraction_mean"]
                    - dyn["post_trap_basin_fraction_mean"]
                ),
            }
        )
    return rows


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    if not rows:
        return ""
    lines = [
        "| " + " | ".join(fields) + " |",
        "| " + " | ".join("---" for _ in fields) + " |",
    ]
    for row in rows:
        values = []
        for field in fields:
            value = row.get(field)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(
    *,
    path: Path,
    experiment_name: str,
    config: dict[str, Any],
    summary: list[dict[str, Any]],
    deltas: list[dict[str, Any]],
    freeze_probs: list[dict[str, Any]],
) -> None:
    prob_fields = ["method_short", "regime", "seed", "base_alias", "prob", "distribution_kind"]
    prob_preview = [
        row
        for row in freeze_probs
        if row["regime"] == "freeze_layer1_post_switch" and row["seed"] == config["seeds"][0]
    ]
    text = "\n\n".join(
        [
            f"# Fixed Layer-1 Post-Switch Probability Sanity\n\nExperiment name: `{experiment_name}`",
            "## Config\n\n"
            + markdown_table(
                [
                    {"field": key, "value": value}
                    for key, value in config.items()
                    if key
                    in {
                        "source_llm_experiment",
                        "tree_spec",
                        "tree_spec_cost_mode",
                        "horizon",
                        "seeds",
                        "switch_denominator",
                        "switch_episode_1based",
                        "methods",
                        "eta",
                        "epsilon",
                    }
                ],
                ["field", "value"],
            ),
            "## Cost Summary\n\n"
            + markdown_table(
                summary,
                [
                    "method_short",
                    "regime",
                    "seeds",
                    "pre_cumulative_cost_avg_mean",
                    "post_cost_avg_mean",
                    "total_cost_avg_mean",
                    "post_regret_avg_mean",
                    "post_target_good_fraction_mean",
                    "post_target_bad_fraction_mean",
                    "post_trap_basin_fraction_mean",
                ],
            ),
            "## Freeze Minus Dynamic\n\n"
            + markdown_table(
                deltas,
                [
                    "method_short",
                    "post_cost_avg_delta",
                    "total_cost_avg_delta",
                    "post_target_good_fraction_delta",
                    "post_target_bad_fraction_delta",
                    "post_trap_basin_fraction_delta",
                ],
            ),
            "## Switch-Time Layer-1 Probabilities\n\n"
            + markdown_table(prob_preview, prob_fields),
            "## Interpretation\n\n"
            "- `dynamic` is the unmodified policy behavior in the same controlled simulator.\n"
            "- `freeze_layer1_post_switch` snapshots the root/direct-child marginal distribution at the switch boundary and reuses that distribution for every post-switch root choice.\n"
            "- Downstream stages and all policy updates remain active, so this isolates whether continued layer-1 adaptation after the switch is helping or hurting.\n"
            "- A negative `post_cost_avg_delta` means freezing the layer-1 distribution improved post-switch cost; a positive delta means ongoing layer-1 adaptation helped.",
        ]
    )
    path.write_text(text + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "tmp" / "llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods_fixed_layer1_prob_sanity",
    )
    parser.add_argument(
        "--source-llm-experiment",
        default="llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods",
    )
    parser.add_argument(
        "--tree-spec",
        type=Path,
        default=ROOT / "analysis" / "tree_specs" / "shared_basin_strong_4of5_prefix_dedup_profile_switch.json",
    )
    parser.add_argument(
        "--tree-spec-cost-mode",
        choices=sorted(PS_FAVORED_COST_MODES),
        default="ps_favored_trap_v10_avg_baited",
    )
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--switch-denominator", type=int, default=4)
    parser.add_argument("--cost-noise", type=float, default=0.02)
    parser.add_argument("--specialist-fraction", type=float, default=0.15)
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_name = args.output_dir.name
    tasks = [
        (
            experiment_name,
            str(args.tree_spec),
            args.tree_spec_cost_mode,
            args.switch_denominator,
            args.horizon,
            args.specialist_fraction,
            args.cost_noise,
            seed,
            method_short,
            regime,
        )
        for seed in args.seeds
        for method_short in METHOD_SPECS
        for regime in REGIMES
    ]

    results: list[dict[str, Any]] = []
    max_workers = max(1, min(int(args.max_workers), len(tasks)))
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_case, task) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())

    per_seed = sorted(
        [result["summary"] for result in results],
        key=lambda row: (row["method_short"], row["regime"], row["seed"]),
    )
    episodes = sorted(
        [row for result in results for row in result["episodes"]],
        key=lambda row: (row["method_short"], row["regime"], row["seed"], row["episode_index"]),
    )
    freeze_probs = sorted(
        [row for result in results for row in result["freeze_probs"]],
        key=lambda row: (row["method_short"], row["regime"], row["seed"], row["rank"]),
    )
    summary = aggregate_summaries(per_seed)
    deltas = add_delta_rows(summary)

    switch_episode = max(1, args.horizon // args.switch_denominator)
    config = {
        "experiment_name": experiment_name,
        "source_llm_experiment": args.source_llm_experiment,
        "tree_spec": str(args.tree_spec),
        "tree_spec_cost_mode": args.tree_spec_cost_mode,
        "horizon": args.horizon,
        "seeds": args.seeds,
        "switch_denominator": args.switch_denominator,
        "switch_episode_index": switch_episode,
        "switch_episode_1based": switch_episode + 1,
        "methods": {key: value["method"] for key, value in METHOD_SPECS.items()},
        "eta": 0.3,
        "epsilon": 0.01,
        "regimes": REGIMES,
        "max_workers": max_workers,
        "diagnostic": "freeze root/direct-child marginal probabilities after switch",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_config.json", config)
    write_json(args.output_dir / "per_seed_summary.json", per_seed)
    write_csv(args.output_dir / "per_seed_summary.csv", per_seed)
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "summary.csv", summary)
    write_json(args.output_dir / "freeze_minus_dynamic.json", deltas)
    write_csv(args.output_dir / "freeze_minus_dynamic.csv", deltas)
    write_json(args.output_dir / "episodes.json", episodes)
    write_csv(args.output_dir / "episodes.csv", episodes)
    write_json(args.output_dir / "switch_time_layer1_probabilities.json", freeze_probs)
    write_csv(args.output_dir / "switch_time_layer1_probabilities.csv", freeze_probs)
    write_report(
        path=args.output_dir / "report.md",
        experiment_name=experiment_name,
        config=config,
        summary=summary,
        deltas=deltas,
        freeze_probs=freeze_probs,
    )
    print(json.dumps({"output_dir": str(args.output_dir), "summary": summary, "deltas": deltas}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
