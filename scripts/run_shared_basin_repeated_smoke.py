"""Run repeated smoke on shared_basin_strong with full llm_bench.

Scope:
- family_kind = shared_basin_strong
- executor_name = llm_bench
- model = gpt-4o-mini
- smoke10 repeated for a fixed horizon
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


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

from fixed_tree_env import FixedTreeEnvironment  # noqa: E402
from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from oracle_eval import find_best_stationary_path  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402


SMOKE10_INDICES = list(range(10))
DATASET_DEFAULT = (
    ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time" / "tasks.json"
)
SPECIALIST_ANALYSIS_PATH = ROOT / "analysis" / "shared_basin_strong_static_analysis.json"
MODEL_REQUIRED = "gpt-4o-mini"
FAMILY_KIND = "shared_basin_strong"
SEED = 0
EXECUTOR_NAME = "llm_bench"


POLICY_REGISTRY = {
    "risky_ps": RiskyPSPolicy,
    "direct_multistage_exp3": DirectMultiStageExp3Policy,
    "epsilon_exp3": EpsilonExp3Policy,
    "naive_mixed": NaiveMixedPolicy,
    "random_path": RandomPathPolicy,
}


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def load_specialist_task_ids() -> set[str]:
    data = json.loads(SPECIALIST_ANALYSIS_PATH.read_text(encoding="utf-8"))
    return set(data.get("unshared_win_task_ids", []))


def build_env(*, executor_name: str) -> FixedTreeEnvironment:
    return FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=FAMILY_KIND,
        family_seed=SEED,
        executor_name=executor_name,
    )


def build_repeated_selection(
    instances: list[dict[str, Any]],
    *,
    indices: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    for repeat_index in range(repeats):
        for position_in_cycle, dataset_index in enumerate(indices):
            repeated.append(
                {
                    "repeat_index": repeat_index,
                    "position_in_cycle": position_in_cycle,
                    "dataset_index": dataset_index,
                    "instance": instances[dataset_index],
                }
            )
    return repeated


def compute_stationary_oracle(selected: list[dict[str, Any]]) -> dict[str, Any]:
    oracle_env = build_env(executor_name="simulated")
    oracle_path, oracle_summary_raw = find_best_stationary_path(
        [row["instance"] for row in selected],
        oracle_env,
    )
    oracle_summary = dict(oracle_summary_raw)
    oracle_summary["path"] = list(oracle_path)
    return oracle_summary


def flatten_episode(
    *,
    episode_index: int,
    row: dict[str, Any],
    result: Any,
    method: str,
    oracle_summary: dict[str, Any],
    selection_info: dict[str, Any],
    update_info: dict[str, Any],
    specialist_task_ids: set[str],
) -> dict[str, Any]:
    instance = row["instance"]
    log = result.episode_log or {}
    stage_trace = {stage_row["stage_name"]: stage_row for stage_row in log.get("stage_trace", [])}
    stage_sources = {name: stage_row.get("source") for name, stage_row in stage_trace.items()}
    llm_stage_names = [name for name, source in stage_sources.items() if source == "llm_bench"]
    llm_call_count = sum(len(stage_trace[name].get("llm_raw_output", [])) for name in llm_stage_names)
    tool_calls_made = sum(len(stage_row.get("executed_tool_calls", [])) for stage_row in stage_trace.values())
    mutating_tool_calls_made = len(stage_trace.get("stage4", {}).get("executed_tool_calls", []))
    assistant_side_mutating_tool_calls_made = sum(
        1
        for call in stage_trace.get("stage4", {}).get("executed_tool_calls", [])
        if call.get("requestor") == "assistant"
    )
    stage5_trace = stage_trace.get("stage5", {})
    oracle_episode_cost = oracle_summary["episode_total_costs"][episode_index]
    raw_oracle_episode_cost = oracle_summary["episode_raw_total_costs"][episode_index]
    leaf_type = result.leaf_type
    shared_path = leaf_type == "shared"
    specialist_task = instance["original_task_id"] in specialist_task_ids
    shared_updates = update_info.get("shared_safe_suffix_edges_updated", []) or []
    risky_updates = update_info.get("risky_edges_updated", []) or []
    return {
        "method": method,
        "episode_index": episode_index,
        "repeat_index": row["repeat_index"],
        "position_in_cycle": row["position_in_cycle"],
        "dataset_index": row["dataset_index"],
        "instance_id": instance["instance_id"],
        "original_task_id": instance["original_task_id"],
        "is_specialist_task": specialist_task,
        "selected_path": list(result.selected_path),
        "leaf_type": leaf_type,
        "selected_shared_path": shared_path,
        "selected_unshared_path": not shared_path,
        "oracle_action": result.oracle_action,
        "final_action": result.final_action,
        "exact_match": bool(result.success),
        "subset_mismatch": bool(log.get("subset_mismatch", False)),
        "terminal_penalty": float(result.terminal_cost),
        "raw_terminal_penalty": float(result.raw_terminal_penalty),
        "total_cost": float(result.total_cost),
        "raw_total_cost": float(result.raw_total_cost),
        "raw_path_cost_component": float(result.raw_path_cost_component),
        "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
        "reasoning_cost": float(result.reasoning_cost),
        "episode_regret": float(result.total_cost - oracle_episode_cost),
        "raw_episode_regret": float(result.raw_total_cost - raw_oracle_episode_cost),
        "cost_scale_version": str(result.cost_scale_version),
        "stage_sources": stage_sources,
        "llm_stage_names": llm_stage_names,
        "llm_call_count": llm_call_count,
        "tool_calls_made": tool_calls_made,
        "mutating_tool_calls_made": mutating_tool_calls_made,
        "assistant_side_mutating_tool_calls_made": assistant_side_mutating_tool_calls_made,
        "stage5_replay_tool_names": [c.get("name") for c in stage5_trace.get("replay_tool_calls", [])],
        "stage5_executed_tool_names": [c.get("name") for c in stage5_trace.get("executed_tool_calls", [])],
        "selection_path_prob": selection_info.get("path_prob"),
        "shared_branch_triggered": bool(update_info.get("shared_leaf_updated", False)),
        "unshared_branch_triggered": str(update_info.get("leaf_type")) == "unshared",
        "shared_update_count": len(shared_updates),
        "unshared_edge_update_count": len(risky_updates),
        "risky_edge_update_edges": risky_updates,
        "selection_info": selection_info,
        "update_info": update_info,
    }


def add_cumulative_fields(episodes: list[dict[str, Any]]) -> None:
    shared_count = 0
    unshared_count = 0
    shared_branch_count = 0
    unshared_branch_count = 0
    shared_update_count = 0
    unshared_edge_update_count = 0
    window: list[bool] = []
    for idx, row in enumerate(episodes, start=1):
        is_shared = bool(row["selected_shared_path"])
        shared_count += int(is_shared)
        unshared_count += int(not is_shared)
        shared_branch_count += int(bool(row["shared_branch_triggered"]))
        unshared_branch_count += int(bool(row["unshared_branch_triggered"]))
        shared_update_count += int(row["shared_update_count"])
        unshared_edge_update_count += int(row["unshared_edge_update_count"])
        window.append(is_shared)
        if len(window) > 10:
            window.pop(0)
        row["cumulative_shared_path_ratio"] = shared_count / idx
        row["cumulative_unshared_path_ratio"] = unshared_count / idx
        row["rolling_shared_path_ratio_last10"] = sum(window) / len(window)
        row["rolling_unshared_path_ratio_last10"] = 1.0 - row["rolling_shared_path_ratio_last10"]
        row["cumulative_shared_branch_count"] = shared_branch_count
        row["cumulative_unshared_branch_count"] = unshared_branch_count
        row["cumulative_shared_update_count"] = shared_update_count
        row["cumulative_unshared_edge_update_count"] = unshared_edge_update_count


def build_summary(
    *,
    method: str,
    dataset: str,
    repeats: int,
    model: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_source_summary: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        for stage_name, source in episode["stage_sources"].items():
            stage_source_summary[stage_name][str(source)] += 1
    return {
        "test_name": f"{method}_smoke10x{repeats}_{FAMILY_KIND}_full_llm",
        "dataset": dataset,
        "dataset_indices": SMOKE10_INDICES,
        "repeats": repeats,
        "episodes": len(episodes),
        "method": method,
        "mechanism": "algorithm_direct",
        "executor_name": EXECUTOR_NAME,
        "family_kind": FAMILY_KIND,
        "seed": SEED,
        "model": model,
        "stationary_oracle_path": oracle_summary["path"],
        "exact_match_mean": mean([float(ep["exact_match"]) for ep in episodes]),
        "terminal_penalty_mean": mean([ep["terminal_penalty"] for ep in episodes]),
        "raw_terminal_penalty_mean": mean([ep["raw_terminal_penalty"] for ep in episodes]),
        "total_cost_mean": mean([ep["total_cost"] for ep in episodes]),
        "raw_total_cost_mean": mean([ep["raw_total_cost"] for ep in episodes]),
        "reasoning_cost_mean": mean([ep["reasoning_cost"] for ep in episodes]),
        "raw_reasoning_cost_component_mean": mean([ep["raw_reasoning_cost_component"] for ep in episodes]),
        "raw_path_cost_component_mean": mean([ep["raw_path_cost_component"] for ep in episodes]),
        "mean_regret": mean([ep["episode_regret"] for ep in episodes]),
        "raw_mean_regret": mean([ep["raw_episode_regret"] for ep in episodes]),
        "algorithm_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
        "raw_algorithm_cumulative_total_cost": sum(ep["raw_total_cost"] for ep in episodes),
        "oracle_stationary_total_cost": oracle_summary["cumulative_total_cost"],
        "raw_oracle_stationary_total_cost": oracle_summary["raw_cumulative_total_cost"],
        "cumulative_regret": sum(ep["episode_regret"] for ep in episodes),
        "raw_cumulative_regret": sum(ep["raw_episode_regret"] for ep in episodes),
        "raw_terminal_penalty_cumulative": sum(ep["raw_terminal_penalty"] for ep in episodes),
        "raw_path_cost_component_cumulative": sum(ep["raw_path_cost_component"] for ep in episodes),
        "raw_reasoning_cost_component_cumulative": sum(ep["raw_reasoning_cost_component"] for ep in episodes),
        "mean_llm_call_count": mean([ep["llm_call_count"] for ep in episodes]),
        "subset_mismatch_count": sum(1 for ep in episodes if ep["subset_mismatch"]),
        "episodes_with_stage5_verification_tools": sum(1 for ep in episodes if ep["stage5_executed_tool_names"]),
        "shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in episodes]),
        "unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in episodes]),
        "specialist_task_count": sum(1 for ep in episodes if ep["is_specialist_task"]),
        "specialist_task_unshared_fraction": mean(
            [float(ep["selected_unshared_path"]) for ep in episodes if ep["is_specialist_task"]]
        ),
        "stage_source_summary": {k: dict(v) for k, v in stage_source_summary.items()},
    }


def build_specialist_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    specialist = [ep for ep in episodes if ep["is_specialist_task"]]
    return {
        "specialist_episode_count": len(specialist),
        "specialist_shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in specialist]),
        "specialist_unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in specialist]),
        "specialist_exact_match_mean": mean([float(ep["exact_match"]) for ep in specialist]),
        "specialist_total_cost_mean": mean([ep["total_cost"] for ep in specialist]),
        "specialist_raw_terminal_penalty_mean": mean([ep["raw_terminal_penalty"] for ep in specialist]),
        "specialist_raw_path_cost_component_mean": mean([ep["raw_path_cost_component"] for ep in specialist]),
        "specialist_raw_reasoning_cost_component_mean": mean([ep["raw_reasoning_cost_component"] for ep in specialist]),
        "specialist_task_ids": sorted({ep["original_task_id"] for ep in specialist}),
    }


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_policy_method(
    method: str,
    selected: list[dict[str, Any]],
    oracle_summary: dict[str, Any],
    specialist_task_ids: set[str],
) -> tuple[list[dict[str, Any]], str]:
    env = build_env(executor_name=EXECUTOR_NAME)
    policy = POLICY_REGISTRY[method](seed=SEED)
    policy.bind_env(env)
    policy.reset()
    episodes: list[dict[str, Any]] = []
    for episode_index, row in enumerate(selected):
        instance = row["instance"]
        print(
            f"[run] method={method} episode={episode_index + 1}/{len(selected)} "
            f"repeat={row['repeat_index'] + 1} pos={row['position_in_cycle']} dataset_index={row['dataset_index']}",
            flush=True,
        )
        path = policy.select_path(instance, env)
        selection_info = policy.get_last_selection_info() if hasattr(policy, "get_last_selection_info") else {}
        env.reset(instance)
        result = env.run_path(path)
        policy.update(result)
        state = policy.get_state() if hasattr(policy, "get_state") else {}
        update_info = state.get("last_update_info", {}) if isinstance(state, dict) else {}
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                row=row,
                result=result,
                method=method,
                oracle_summary=oracle_summary,
                selection_info=selection_info if isinstance(selection_info, dict) else {},
                update_info=update_info if isinstance(update_info, dict) else {},
                specialist_task_ids=specialist_task_ids,
            )
        )
    add_cumulative_fields(episodes)
    model = getattr(env.family_executor, "model", "unknown")
    return episodes, model


def run_stationary_oracle_method(
    selected: list[dict[str, Any]],
    oracle_summary: dict[str, Any],
    specialist_task_ids: set[str],
) -> tuple[list[dict[str, Any]], str]:
    env = build_env(executor_name=EXECUTOR_NAME)
    path = list(oracle_summary["path"])
    episodes: list[dict[str, Any]] = []
    for episode_index, row in enumerate(selected):
        instance = row["instance"]
        print(
            f"[run] method=oracle_best_fixed_path episode={episode_index + 1}/{len(selected)} "
            f"repeat={row['repeat_index'] + 1} pos={row['position_in_cycle']} dataset_index={row['dataset_index']}",
            flush=True,
        )
        env.reset(instance)
        result = env.run_path(path)
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                row=row,
                result=result,
                method="oracle_best_fixed_path",
                oracle_summary=oracle_summary,
                selection_info={},
                update_info={},
                specialist_task_ids=specialist_task_ids,
            )
        )
    add_cumulative_fields(episodes)
    model = getattr(env.family_executor, "model", "unknown")
    return episodes, model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run repeated shared-basin smoke.")
    parser.add_argument("--data", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--methods",
        nargs="+",
        default=[
            "risky_ps",
            "epsilon_exp3",
            "direct_multistage_exp3",
            "naive_mixed",
            "random_path",
            "oracle_best_fixed_path",
        ],
    )
    args = parser.parse_args()

    model_name = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "")
    if model_name != MODEL_REQUIRED:
        raise SystemExit(
            f"PSAGENT_LLM_BENCH_MODEL must be {MODEL_REQUIRED!r}; got {model_name!r}"
        )

    instances = load_instances(args.data)
    specialist_task_ids = load_specialist_task_ids()
    selected = build_repeated_selection(instances, indices=SMOKE10_INDICES, repeats=args.repeats)
    oracle_summary = compute_stationary_oracle(selected)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = args.output_dir / f"{timestamp}_shared_basin_repeated_smoke"
    root_out.mkdir(parents=True, exist_ok=True)
    write_json(
        root_out / "run_config.json",
        {
            "dataset": str(args.data),
            "dataset_indices": SMOKE10_INDICES,
            "repeats": args.repeats,
            "horizon": len(selected),
            "family_kind": FAMILY_KIND,
            "executor_name": EXECUTOR_NAME,
            "model": model_name,
            "seed": SEED,
            "methods": args.methods,
        },
    )
    write_json(
        root_out / "stationary_oracle_summary.json",
        {
            "path": oracle_summary["path"],
            "cumulative_total_cost": oracle_summary["cumulative_total_cost"],
            "mean_total_cost": oracle_summary["mean_total_cost"],
            "raw_cumulative_total_cost": oracle_summary["raw_cumulative_total_cost"],
            "raw_mean_total_cost": oracle_summary["raw_mean_total_cost"],
            "cost_scale_version": oracle_summary["cost_scale_version"],
            "family_kind": FAMILY_KIND,
            "executor_for_comparator": "simulated",
        },
    )

    for method in args.methods:
        print(f"[method] method={method} start", flush=True)
        method_dir = root_out / method
        method_dir.mkdir(parents=True, exist_ok=True)
        if method == "oracle_best_fixed_path":
            episodes, model = run_stationary_oracle_method(selected, oracle_summary, specialist_task_ids)
        else:
            episodes, model = run_policy_method(method, selected, oracle_summary, specialist_task_ids)
        summary = build_summary(
            method=method,
            dataset=str(args.data),
            repeats=args.repeats,
            model=model,
            oracle_summary=oracle_summary,
            episodes=episodes,
        )
        specialist_summary = build_specialist_summary(episodes)
        write_json(method_dir / "episodes.json", episodes)
        write_json(method_dir / "summary.json", summary)
        write_json(method_dir / "summary_with_oracle.json", summary)
        write_json(method_dir / "specialist_summary.json", specialist_summary)
        (method_dir / "smoke_summary.md").write_text(
            json.dumps({"summary": summary, "specialist_summary": specialist_summary}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        if method == "risky_ps":
            dynamics_rows = [
                {
                    "episode_index": ep["episode_index"],
                    "repeat_index": ep["repeat_index"],
                    "position_in_cycle": ep["position_in_cycle"],
                    "dataset_index": ep["dataset_index"],
                    "original_task_id": ep["original_task_id"],
                    "is_specialist_task": ep["is_specialist_task"],
                    "selected_shared_path": ep["selected_shared_path"],
                    "selected_unshared_path": ep["selected_unshared_path"],
                    "cumulative_shared_path_ratio": ep["cumulative_shared_path_ratio"],
                    "cumulative_unshared_path_ratio": ep["cumulative_unshared_path_ratio"],
                    "rolling_shared_path_ratio_last10": ep["rolling_shared_path_ratio_last10"],
                    "rolling_unshared_path_ratio_last10": ep["rolling_unshared_path_ratio_last10"],
                    "shared_branch_triggered": ep["shared_branch_triggered"],
                    "unshared_branch_triggered": ep["unshared_branch_triggered"],
                    "shared_update_count": ep["shared_update_count"],
                    "unshared_edge_update_count": ep["unshared_edge_update_count"],
                    "cumulative_shared_branch_count": ep["cumulative_shared_branch_count"],
                    "cumulative_unshared_branch_count": ep["cumulative_unshared_branch_count"],
                    "cumulative_shared_update_count": ep["cumulative_shared_update_count"],
                    "cumulative_unshared_edge_update_count": ep["cumulative_unshared_edge_update_count"],
                    "raw_terminal_penalty": ep["raw_terminal_penalty"],
                    "raw_path_cost_component": ep["raw_path_cost_component"],
                    "raw_reasoning_cost_component": ep["raw_reasoning_cost_component"],
                }
                for ep in episodes
            ]
            write_json(root_out / "risky_ps_shared_unshared_dynamics.json", dynamics_rows)
            write_csv(root_out / "risky_ps_shared_unshared_dynamics.csv", dynamics_rows)
        print(
            f"[method] method={method} done mean_regret={summary['mean_regret']:.6f} "
            f"shared_frac={summary['shared_path_fraction']:.3f}",
            flush=True,
        )

    print(str(root_out))


if __name__ == "__main__":
    main()
