"""Run a small telecom llm_bench e2e smoke with flattened diagnostics."""

from __future__ import annotations

import argparse
import json
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
from direct_multistage_exp3_local import DirectMultiStageExp3LocalPolicy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from full_unshare import FullUnsharePolicy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from oracle_eval import find_best_stationary_path  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402
from risky_ps_ix import RiskyPSIXPolicy  # noqa: E402
from risky_ps_linear import RiskyPSLinearPolicy  # noqa: E402


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


POLICY_REGISTRY = {
    "risky_ps": RiskyPSPolicy,
    "risky_ps_linear": RiskyPSLinearPolicy,
    "risky_ps_ix": RiskyPSIXPolicy,
    "direct_multistage_exp3": DirectMultiStageExp3Policy,
    "direct_multistage_exp3_local": DirectMultiStageExp3LocalPolicy,
    "epsilon_exp3": EpsilonExp3Policy,
    "full_unshare": FullUnsharePolicy,
    "naive_mixed": NaiveMixedPolicy,
    "random_path": RandomPathPolicy,
}


def load_instances(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def flatten_episode(
    *,
    episode_index: int,
    dataset_index: int,
    instance: dict[str, Any],
    result: Any,
    oracle_summary: dict[str, Any],
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
        "episode_regret": float(result.total_cost - oracle_episode_cost),
        "raw_episode_regret": float(result.raw_total_cost - raw_oracle_episode_cost),
        "cost_scale_version": "telecom_mms_cost_norm_v1",
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
        "stage3_executed_tool_names": [c.get("name") for c in stage_trace.get("stage3", {}).get("executed_tool_calls", [])],
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
        "selection_signal_summary": None,
    }


def build_summary(
    *,
    method: str,
    test_name: str,
    dataset: str,
    dataset_indices: list[int] | None,
    model: str,
    oracle_summary: dict[str, Any] | None,
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_source_summary: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        for stage_name, source in episode["stage_sources"].items():
            stage_source_summary[stage_name][str(source)] += 1
    return {
        "test_name": test_name,
        "dataset": dataset,
        **({"dataset_indices": dataset_indices} if dataset_indices is not None else {}),
        "method": method,
        "mechanism": "algorithm_direct",
        "executor_name": "llm_bench",
        "family_kind": "strong",
        "seed": 0,
        "model": model,
        "episodes": len(episodes),
        "stationary_oracle_path": oracle_summary["path"] if oracle_summary is not None else None,
        "exact_match_mean": mean([float(ep["exact_match"]) for ep in episodes]),
        "terminal_penalty_mean": mean([ep["terminal_penalty"] for ep in episodes]),
        "raw_total_cost_mean": mean([ep["raw_total_cost"] for ep in episodes]),
        "total_cost_mean": mean([ep["total_cost"] for ep in episodes]),
        "normalized_total_cost_mean": mean([ep["normalized_total_cost"] for ep in episodes]),
        "mean_regret": mean([ep["episode_regret"] for ep in episodes]),
        "raw_mean_regret": mean([ep["raw_episode_regret"] for ep in episodes]),
        "normalized_mean_regret": mean([ep["episode_regret"] for ep in episodes]),
        "algorithm_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
        "raw_algorithm_cumulative_total_cost": sum(ep["raw_total_cost"] for ep in episodes),
        "oracle_stationary_total_cost": (
            sum(
                oracle_summary["episode_total_costs_by_instance_id"][ep["instance_id"]]
                for ep in episodes
            )
            if oracle_summary is not None
            else 0.0
        ),
        "raw_oracle_stationary_total_cost": (
            sum(
                oracle_summary["episode_raw_total_costs_by_instance_id"][ep["instance_id"]]
                for ep in episodes
            )
            if oracle_summary is not None
            else 0.0
        ),
        "cumulative_regret": sum(ep["episode_regret"] for ep in episodes),
        "raw_cumulative_regret": sum(ep["raw_episode_regret"] for ep in episodes),
        "mean_llm_call_count": mean([ep["llm_call_count"] for ep in episodes]),
        "oracle_action_distribution": dict(Counter(ep["oracle_action"] for ep in episodes)),
        "final_action_distribution": dict(Counter(ep["final_action"] for ep in episodes)),
        "cost_scale_version": "telecom_mms_cost_norm_v1",
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run telecom llm e2e smoke.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--dataset-indices", type=int, nargs="*")
    parser.add_argument("--test-name", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--skip-stationary-oracle", action="store_true")
    parser.add_argument(
        "--method",
        choices=sorted(POLICY_REGISTRY),
        default="risky_ps",
    )
    args = parser.parse_args()

    instances = load_instances(args.data)
    selected = (
        [(idx, instances[idx]) for idx in args.dataset_indices]
        if args.dataset_indices
        else list(enumerate(instances))
    )

    env = FixedTreeEnvironment(
        agent_catalog=[],
        family_kind="strong",
        family_seed=0,
        executor_name="llm_bench",
    )
    policy = POLICY_REGISTRY[args.method](seed=0)
    policy.bind_env(env)
    policy.reset()

    oracle_summary: dict[str, Any] | None = None
    if not args.skip_stationary_oracle:
        oracle_env = FixedTreeEnvironment(
            agent_catalog=[],
            family_kind="strong",
            family_seed=0,
            executor_name="simulated",
        )
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

    episodes: list[dict[str, Any]] = []
    for episode_index, (dataset_index, instance) in enumerate(selected):
        path, _selection_meta, should_update = policy.select_path(instance, env), None, True
        env.reset(instance)
        result = env.run_path(path)
        if should_update:
            policy.update(result)
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                dataset_index=dataset_index,
                instance=instance,
                result=result,
                oracle_summary=oracle_summary
                or {
                    "episode_total_costs_by_instance_id": {instance["instance_id"]: 0.0},
                    "episode_raw_total_costs_by_instance_id": {instance["instance_id"]: 0.0},
                },
            )
        )

    model = getattr(env.family_executor, "model", "unknown")
    summary = build_summary(
        method=args.method,
        test_name=args.test_name,
        dataset=str(args.data),
        dataset_indices=args.dataset_indices if args.dataset_indices else None,
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = args.output_dir / f"{timestamp}_{args.test_name}"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "episodes.json").open("w", encoding="utf-8") as f:
        json.dump(episodes, f, ensure_ascii=False, indent=2)
    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out / "summary_with_oracle.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with (out / "smoke_summary.md").open("w", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False, indent=2))
    print(str(out))


if __name__ == "__main__":
    main()
