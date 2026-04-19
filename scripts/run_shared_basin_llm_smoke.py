"""Run shared_basin_strong full-LLM telecom smoke comparisons.

This is a narrow runner for:
- family_kind = shared_basin_strong
- executor_name = llm_bench
- model = gpt-4o-mini
- smoke3 / smoke10 only
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


SMOKE_SPLITS: dict[str, list[int]] = {
    "smoke3": [0, 2, 3],
    "smoke10": list(range(10)),
}

DATASET_DEFAULT = (
    ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time" / "tasks.json"
)
MODEL_REQUIRED = "gpt-4o-mini"
FAMILY_KIND = "shared_basin_strong"
SEED = 0
EXECUTOR_NAME = "llm_bench"


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


POLICY_REGISTRY = {
    "risky_ps": RiskyPSPolicy,
    "direct_multistage_exp3": DirectMultiStageExp3Policy,
    "epsilon_exp3": EpsilonExp3Policy,
    "naive_mixed": NaiveMixedPolicy,
    "random_path": RandomPathPolicy,
}


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def build_env(*, executor_name: str) -> FixedTreeEnvironment:
    return FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=FAMILY_KIND,
        family_seed=SEED,
        executor_name=executor_name,
    )


def compute_stationary_oracle(selected: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
    oracle_env = build_env(executor_name="simulated")
    oracle_path, oracle_summary_raw = find_best_stationary_path(
        [inst for _, inst in selected],
        oracle_env,
    )
    oracle_summary = dict(oracle_summary_raw)
    oracle_summary["path"] = list(oracle_path)
    oracle_summary["episode_total_costs_by_instance_id"] = {
        inst["instance_id"]: cost
        for (_, inst), cost in zip(selected, oracle_summary["episode_total_costs"])
    }
    oracle_summary["episode_raw_total_costs_by_instance_id"] = {
        inst["instance_id"]: cost
        for (_, inst), cost in zip(selected, oracle_summary["episode_raw_total_costs"])
    }
    return oracle_summary


def flatten_episode(
    *,
    episode_index: int,
    dataset_index: int,
    instance: dict[str, Any],
    result: Any,
    oracle_summary: dict[str, Any],
    method: str,
    stationary_oracle_path: list[str],
) -> dict[str, Any]:
    log = result.episode_log or {}
    stage_trace = {row["stage_name"]: row for row in log.get("stage_trace", [])}
    stage_sources = {name: row.get("source") for name, row in stage_trace.items()}
    llm_stage_names = [name for name, source in stage_sources.items() if source == "llm_bench"]
    llm_call_count = sum(len(stage_trace[name].get("llm_raw_output", [])) for name in llm_stage_names)
    tool_calls_made = sum(len(row.get("executed_tool_calls", [])) for row in stage_trace.values())
    mutating_tool_calls_made = len(stage_trace.get("stage4", {}).get("executed_tool_calls", []))
    assistant_side_mutating_tool_calls_made = sum(
        1
        for call in stage_trace.get("stage4", {}).get("executed_tool_calls", [])
        if call.get("requestor") == "assistant"
    )
    stage2_output = stage_trace.get("stage2", {}).get("output", {})
    stage3_output = stage_trace.get("stage3", {}).get("output", {})
    stage4_output = stage_trace.get("stage4", {}).get("output", {})
    stage5_output = stage_trace.get("stage5", {}).get("output", {})
    stage4_trace = stage_trace.get("stage4", {})
    stage5_trace = stage_trace.get("stage5", {})
    oracle_episode_cost = oracle_summary["episode_total_costs_by_instance_id"][instance["instance_id"]]
    raw_oracle_episode_cost = oracle_summary["episode_raw_total_costs_by_instance_id"][instance["instance_id"]]
    return {
        "method": method,
        "episode_index": episode_index,
        "dataset_index": dataset_index,
        "instance_id": instance["instance_id"],
        "original_task_id": instance["original_task_id"],
        "selected_path": list(result.selected_path),
        "leaf_type": result.leaf_type,
        "oracle_action": result.oracle_action,
        "final_action": result.final_action,
        "exact_match": bool(result.success),
        "subset_mismatch": bool(log.get("subset_mismatch", False)),
        "terminal_penalty": float(result.terminal_cost),
        "normalized_terminal_penalty": float(result.terminal_cost),
        "total_cost": float(result.total_cost),
        "normalized_total_cost": float(result.total_cost),
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
        "bench_db_check": log.get("bench_aux_eval", {}).get("bench_db_check"),
        "bench_success": log.get("bench_aux_eval", {}).get("bench_success"),
        "db_hash_before": log.get("bench_aux_eval", {}).get("db_hash_before"),
        "db_hash_after": log.get("bench_aux_eval", {}).get("db_hash_after"),
        "stage2_resolved_line_id": stage2_output.get("resolved_line_id"),
        "stage3_inferred_blocker_ids": stage3_output.get("inferred_blocker_ids", []),
        "stage3_executed_tool_names": [
            c.get("name") for c in stage_trace.get("stage3", {}).get("executed_tool_calls", [])
        ],
        "stage4_repairability": stage4_output.get("repairability"),
        "stage4_executed_blocker_ids": stage4_output.get("executed_blocker_ids", []),
        "stage4_deferred_blocker_ids": stage4_output.get("deferred_blocker_ids", []),
        "stage4_executed_tool_names": [c.get("name") for c in stage4_trace.get("executed_tool_calls", [])],
        "stage5_selected_blocker_ids": stage5_output.get("selected_blocker_ids", []),
        "stage5_deferred_blocker_ids": stage5_output.get("deferred_blocker_ids", []),
        "stage5_post_repair_blocker_ids": stage5_output.get("post_repair_blocker_ids", []),
        "stage5_verification_evidence": stage5_output.get("verification_evidence", []),
        "stage5_replay_tool_names": [c.get("name") for c in stage5_trace.get("replay_tool_calls", [])],
        "stage5_executed_tool_names": [c.get("name") for c in stage5_trace.get("executed_tool_calls", [])],
        "stage5_db_hash_before_replay": stage5_trace.get("db_hash_before_replay"),
        "stage5_db_hash_after_replay": stage5_trace.get("db_hash_after_replay"),
        "stationary_oracle_path": stationary_oracle_path,
    }


def build_summary(
    *,
    method: str,
    split_name: str,
    dataset: str,
    dataset_indices: list[int],
    model: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_source_summary: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        for stage_name, source in episode["stage_sources"].items():
            stage_source_summary[stage_name][str(source)] += 1
    return {
        "test_name": f"{method}_{split_name}_{FAMILY_KIND}_full_llm",
        "dataset": dataset,
        "dataset_indices": dataset_indices,
        "method": method,
        "mechanism": "algorithm_direct",
        "executor_name": EXECUTOR_NAME,
        "family_kind": FAMILY_KIND,
        "seed": SEED,
        "model": model,
        "episodes": len(episodes),
        "stationary_oracle_path": oracle_summary["path"],
        "exact_match_mean": mean([float(ep["exact_match"]) for ep in episodes]),
        "terminal_penalty_mean": mean([ep["terminal_penalty"] for ep in episodes]),
        "raw_total_cost_mean": mean([ep["raw_total_cost"] for ep in episodes]),
        "total_cost_mean": mean([ep["total_cost"] for ep in episodes]),
        "normalized_total_cost_mean": mean([ep["normalized_total_cost"] for ep in episodes]),
        "reasoning_cost_mean": mean([ep["reasoning_cost"] for ep in episodes]),
        "raw_reasoning_cost_mean": mean([ep["raw_reasoning_cost_component"] for ep in episodes]),
        "mean_regret": mean([ep["episode_regret"] for ep in episodes]),
        "raw_mean_regret": mean([ep["raw_episode_regret"] for ep in episodes]),
        "algorithm_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
        "raw_algorithm_cumulative_total_cost": sum(ep["raw_total_cost"] for ep in episodes),
        "oracle_stationary_total_cost": sum(
            oracle_summary["episode_total_costs_by_instance_id"][ep["instance_id"]] for ep in episodes
        ),
        "raw_oracle_stationary_total_cost": sum(
            oracle_summary["episode_raw_total_costs_by_instance_id"][ep["instance_id"]] for ep in episodes
        ),
        "cumulative_regret": sum(ep["episode_regret"] for ep in episodes),
        "raw_cumulative_regret": sum(ep["raw_episode_regret"] for ep in episodes),
        "mean_llm_call_count": mean([ep["llm_call_count"] for ep in episodes]),
        "oracle_action_distribution": dict(Counter(ep["oracle_action"] for ep in episodes)),
        "final_action_distribution": dict(Counter(ep["final_action"] for ep in episodes)),
        "cost_scale_version": str(episodes[0]["cost_scale_version"]) if episodes else "unknown",
        "stage_source_summary": {k: dict(v) for k, v in stage_source_summary.items()},
        "mean_tool_calls_made": mean([ep["tool_calls_made"] for ep in episodes]),
        "mean_mutating_tool_calls_made": mean([ep["mutating_tool_calls_made"] for ep in episodes]),
        "mean_assistant_side_mutating_tool_calls_made": mean(
            [ep["assistant_side_mutating_tool_calls_made"] for ep in episodes]
        ),
        "episodes_with_stage4_mutation": sum(1 for ep in episodes if ep["stage4_executed_tool_names"]),
        "episodes_with_stage5_replay": sum(1 for ep in episodes if ep["stage5_replay_tool_names"]),
        "episodes_with_stage5_verification_tools": sum(1 for ep in episodes if ep["stage5_executed_tool_names"]),
        "subset_mismatch_count": sum(1 for ep in episodes if ep["subset_mismatch"]),
        "stage2_resolved_line_id_counts": dict(Counter(ep["stage2_resolved_line_id"] for ep in episodes)),
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


def run_policy_method(method: str, selected: list[tuple[int, dict[str, Any]]], oracle_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    env = build_env(executor_name=EXECUTOR_NAME)
    policy = POLICY_REGISTRY[method](seed=SEED)
    policy.bind_env(env)
    policy.reset()
    episodes: list[dict[str, Any]] = []
    for episode_index, (dataset_index, instance) in enumerate(selected):
        print(
            f"[run] method={method} episode={episode_index + 1}/{len(selected)} "
            f"dataset_index={dataset_index} instance_id={instance['instance_id']}",
            flush=True,
        )
        path = policy.select_path(instance, env)
        env.reset(instance)
        result = env.run_path(path)
        policy.update(result)
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                dataset_index=dataset_index,
                instance=instance,
                result=result,
                oracle_summary=oracle_summary,
                method=method,
                stationary_oracle_path=oracle_summary["path"],
            )
        )
    model = getattr(env.family_executor, "model", "unknown")
    return episodes, model


def run_stationary_oracle_method(selected: list[tuple[int, dict[str, Any]]], oracle_summary: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    env = build_env(executor_name=EXECUTOR_NAME)
    path = list(oracle_summary["path"])
    episodes: list[dict[str, Any]] = []
    for episode_index, (dataset_index, instance) in enumerate(selected):
        print(
            f"[run] method=oracle_best_fixed_path episode={episode_index + 1}/{len(selected)} "
            f"dataset_index={dataset_index} instance_id={instance['instance_id']}",
            flush=True,
        )
        env.reset(instance)
        result = env.run_path(path)
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                dataset_index=dataset_index,
                instance=instance,
                result=result,
                oracle_summary=oracle_summary,
                method="oracle_best_fixed_path",
                stationary_oracle_path=path,
            )
        )
    model = getattr(env.family_executor, "model", "unknown")
    return episodes, model


def method_sort_key(row: dict[str, Any]) -> tuple[float, float]:
    return (float(row["mean_regret"]), float(row["total_cost_mean"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run shared_basin_strong full-LLM telecom smoke.")
    parser.add_argument("--data", type=Path, default=DATASET_DEFAULT)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=sorted(SMOKE_SPLITS), default=sorted(SMOKE_SPLITS))
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
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root_out = args.output_dir / f"{timestamp}_shared_basin_full_llm_smoke"
    root_out.mkdir(parents=True, exist_ok=True)

    compare_rows: list[dict[str, Any]] = []
    run_manifest: dict[str, Any] = {
        "dataset": str(args.data),
        "family_kind": FAMILY_KIND,
        "executor_name": EXECUTOR_NAME,
        "model": model_name,
        "seed": SEED,
        "methods": args.methods,
        "splits": {k: SMOKE_SPLITS[k] for k in args.splits},
    }
    write_json(root_out / "run_config.json", run_manifest)

    for split_name in args.splits:
        indices = SMOKE_SPLITS[split_name]
        split_dir = root_out / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        selected = [(idx, instances[idx]) for idx in indices]
        print(f"[split] {split_name} indices={indices}", flush=True)
        oracle_summary = compute_stationary_oracle(selected)
        write_json(
            split_dir / "stationary_oracle_summary.json",
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
            method_dir = split_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            print(f"[method] split={split_name} method={method} start", flush=True)
            if method == "oracle_best_fixed_path":
                episodes, model = run_stationary_oracle_method(selected, oracle_summary)
            else:
                episodes, model = run_policy_method(method, selected, oracle_summary)
            summary = build_summary(
                method=method,
                split_name=split_name,
                dataset=str(args.data),
                dataset_indices=indices,
                model=model,
                oracle_summary=oracle_summary,
                episodes=episodes,
            )
            write_json(method_dir / "episodes.json", episodes)
            write_json(method_dir / "summary.json", summary)
            write_json(method_dir / "summary_with_oracle.json", summary)
            (method_dir / "smoke_summary.md").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            compare_rows.append(
                {
                    "split": split_name,
                    **{k: summary[k] for k in (
                        "method",
                        "family_kind",
                        "executor_name",
                        "model",
                        "cumulative_regret",
                        "mean_regret",
                        "total_cost_mean",
                        "raw_total_cost_mean",
                        "reasoning_cost_mean",
                        "exact_match_mean",
                        "terminal_penalty_mean",
                        "subset_mismatch_count",
                        "episodes_with_stage5_verification_tools",
                        "mean_llm_call_count",
                    )},
                }
            )
            print(
                f"[method] split={split_name} method={method} done "
                f"mean_regret={summary['mean_regret']:.6f} exact_match={summary['exact_match_mean']:.4f}",
                flush=True,
            )

    compare_rows.sort(key=lambda row: (row["split"], method_sort_key(row)))
    write_json(root_out / "multi_method_shared_basin_smoke_compare.json", compare_rows)
    write_csv(root_out / "multi_method_shared_basin_smoke_compare.csv", compare_rows)
    (root_out / "multi_method_shared_basin_smoke_compare.md").write_text(
        json.dumps(compare_rows, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(str(root_out))


if __name__ == "__main__":
    main()
