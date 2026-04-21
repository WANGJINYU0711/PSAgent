"""Run repeated smoke on shared_basin_strong for mechanism experiments.

Scope:
- family_kind = shared_basin_strong
- executor_name = llm_bench
- model = gpt-4o-mini
- smoke10 repeated for a fixed horizon
- backbone policy = direct_multistage_exp3
- only the per-stage child decider changes by mechanism
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for extra in (
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
    ROOT / "baselines",
    SCRIPTS_DIR,
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fixed_tree_env import compute_llm_bench_reasoning_components  # noqa: E402
from fixed_tree_env import TELECOM_MMS_TOTAL_UPPER_BOUND_V2_DEFAULT  # noqa: E402
from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from mechanism_utils import choose_path_with_mechanism  # noqa: E402
from run_shared_basin_repeated_smoke import (  # noqa: E402
    DATASET_DEFAULT,
    EXECUTOR_NAME,
    FAMILY_KIND,
    MODEL_REQUIRED,
    ROOT as RUNNER_ROOT,
    SEED,
    SMOKE10_INDICES,
    add_cumulative_fields,
    build_env,
    build_repeated_selection,
    build_specialist_summary,
    compare_rows_to_markdown,
    compute_stationary_oracle,
    ensure_model_env,
    flatten_episode as base_flatten_episode,
    load_instances,
    load_json,
    load_specialist_task_ids,
    materialize_schedule,
    mean,
    serialize_schedule,
    write_bytes_atomic,
    write_csv,
    write_json,
    write_jsonl,
    write_text_atomic,
)


BACKBONE_POLICY = "direct_multistage_exp3"
DEFAULT_MECHANISMS = ["theta_guided_agent"]
MECHANISM_CHOICES = ["theta_guided_agent", "agent_only", "algorithm_direct"]


def validate_mechanisms(mechanisms: list[str]) -> None:
    invalid = [mechanism for mechanism in mechanisms if mechanism not in MECHANISM_CHOICES]
    if invalid:
        raise SystemExit(
            f"Mechanism runner only supports {MECHANISM_CHOICES}; got invalid {invalid}"
        )


def load_run_context(run_dir: Path) -> dict[str, Any]:
    run_config = load_json(run_dir / "run_config.json")
    schedule_rows = load_json(run_dir / "schedule.json")
    oracle_summary = load_json(run_dir / "stationary_oracle_summary.json")
    instances = load_instances(Path(run_config["dataset"]))
    selected = materialize_schedule(instances, schedule_rows)
    specialist_task_ids = load_specialist_task_ids()
    return {
        "run_config": run_config,
        "schedule_rows": schedule_rows,
        "oracle_summary": oracle_summary,
        "selected": selected,
        "specialist_task_ids": specialist_task_ids,
    }


def initialize_run(
    *,
    data_path: Path,
    output_dir: Path,
    repeats: int,
    mechanisms: list[str],
    model_name: str,
) -> Path:
    validate_mechanisms(mechanisms)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    schedule_path = output_dir / "schedule.json"
    oracle_path = output_dir / "stationary_oracle_summary.json"
    specialist_path = output_dir / "specialist_task_ids.json"

    if run_config_path.exists() and schedule_path.exists() and oracle_path.exists():
        return output_dir

    instances = load_instances(data_path)
    selected = build_repeated_selection(instances, indices=SMOKE10_INDICES, repeats=repeats)
    oracle_summary = compute_stationary_oracle(selected)
    schedule_rows = serialize_schedule(selected)
    specialist_task_ids = sorted(load_specialist_task_ids())

    write_json(
        run_config_path,
        {
            "created_at": datetime.now().isoformat(),
            "dataset": str(data_path),
            "dataset_indices": SMOKE10_INDICES,
            "repeats": repeats,
            "horizon": len(selected),
            "family_kind": FAMILY_KIND,
            "executor_name": EXECUTOR_NAME,
            "model": model_name,
            "seed": SEED,
            "backbone_policy": BACKBONE_POLICY,
            "mechanisms": mechanisms,
            "parallelism": "method_only",
        },
    )
    write_json(schedule_path, schedule_rows)
    write_json(oracle_path, oracle_summary)
    write_json(specialist_path, specialist_task_ids)
    return output_dir


def build_progress_payload(
    *,
    mechanism: str,
    completed_count: int,
    total_episodes: int,
    model: str,
    status: str,
) -> dict[str, Any]:
    last_completed = completed_count - 1 if completed_count else None
    return {
        "mechanism": mechanism,
        "backbone_policy": BACKBONE_POLICY,
        "scheduled_episodes": total_episodes,
        "completed_episodes": completed_count,
        "last_completed_episode_index": last_completed,
        "status": status,
        "model": model,
        "updated_at": datetime.now().isoformat(),
    }


def _extract_invalid_flags(stage_choice_trace: list[dict[str, Any]]) -> list[bool]:
    return [bool(row.get("invalid_output", False)) for row in stage_choice_trace]


def flatten_mechanism_episode(
    *,
    episode_index: int,
    row: dict[str, Any],
    result: Any,
    mechanism: str,
    oracle_summary: dict[str, Any],
    selection_info: dict[str, Any],
    update_info: dict[str, Any],
    specialist_task_ids: set[str],
    selection_meta: dict[str, Any],
) -> dict[str, Any]:
    episode = base_flatten_episode(
        episode_index=episode_index,
        row=row,
        result=result,
        method=BACKBONE_POLICY,
        oracle_summary=oracle_summary,
        selection_info=selection_info,
        update_info=update_info,
        specialist_task_ids=specialist_task_ids,
    )
    stage_choice_trace = list(selection_meta.get("stage_choice_trace", []))
    chooser_prompt_tokens_total = float(
        selection_meta.get("chooser_prompt_tokens_total", 0.0) or 0.0
    )
    chooser_completion_tokens_total = float(
        selection_meta.get("chooser_completion_tokens_total", 0.0) or 0.0
    )
    chooser_total_tokens_total = float(
        selection_meta.get("chooser_total_tokens_total", 0.0) or 0.0
    )
    chooser_api_cost_total_usd_raw = float(
        selection_meta.get("chooser_api_cost_total_usd_raw", 0.0) or 0.0
    )
    chooser_generation_time_total_seconds = float(
        selection_meta.get("chooser_generation_time_total_seconds", 0.0) or 0.0
    )
    chooser_llm_round_trip_total_seconds = float(
        selection_meta.get("chooser_llm_round_trip_total_seconds", 0.0) or 0.0
    )
    chooser_episode_wall_clock_seconds = float(
        selection_meta.get("chooser_episode_wall_clock_seconds", 0.0) or 0.0
    )
    chooser_llm_call_count = int(selection_meta.get("chooser_llm_call_count", 0) or 0)
    chooser_reasoning_components = compute_llm_bench_reasoning_components(
        prompt_tokens_total=chooser_prompt_tokens_total,
        completion_tokens_total=chooser_completion_tokens_total,
        api_cost_total_usd_raw=chooser_api_cost_total_usd_raw,
        default_mode=str(episode.get("reasoning_cost_mode_default") or "token"),
    )
    executor_llm_call_count = int(episode.get("llm_call_count", 0) or 0)
    executor_prompt_tokens_total = float(episode.get("prompt_tokens_total", 0.0) or 0.0)
    executor_completion_tokens_total = float(
        episode.get("completion_tokens_total", 0.0) or 0.0
    )
    executor_total_tokens_total = float(episode.get("total_tokens_total", 0.0) or 0.0)
    executor_api_cost_total_usd_raw = float(
        episode.get("api_cost_total_usd_raw", 0.0) or 0.0
    )
    executor_generation_time_total_seconds = float(
        episode.get("generation_time_total_seconds", 0.0) or 0.0
    )
    executor_llm_round_trip_total_seconds = float(
        episode.get("llm_round_trip_total_seconds", 0.0) or 0.0
    )
    executor_episode_wall_clock_seconds = float(
        episode.get("episode_wall_clock_seconds", 0.0) or 0.0
    )
    combined_prompt_tokens_total = executor_prompt_tokens_total + chooser_prompt_tokens_total
    combined_completion_tokens_total = (
        executor_completion_tokens_total + chooser_completion_tokens_total
    )
    combined_total_tokens_total = executor_total_tokens_total + chooser_total_tokens_total
    combined_api_cost_total_usd_raw = (
        executor_api_cost_total_usd_raw + chooser_api_cost_total_usd_raw
    )
    combined_generation_time_total_seconds = (
        executor_generation_time_total_seconds + chooser_generation_time_total_seconds
    )
    combined_llm_round_trip_total_seconds = (
        executor_llm_round_trip_total_seconds + chooser_llm_round_trip_total_seconds
    )
    combined_episode_wall_clock_seconds = (
        executor_episode_wall_clock_seconds + chooser_episode_wall_clock_seconds
    )
    combined_reasoning_components = compute_llm_bench_reasoning_components(
        prompt_tokens_total=combined_prompt_tokens_total,
        completion_tokens_total=combined_completion_tokens_total,
        api_cost_total_usd_raw=combined_api_cost_total_usd_raw,
        default_mode=str(episode.get("reasoning_cost_mode_default") or "token"),
    )
    combined_raw_total_cost_api = (
        float(episode["raw_terminal_penalty"])
        + float(episode["raw_path_cost_component"])
        + float(combined_reasoning_components["raw_reasoning_cost_component_api"])
    )
    combined_raw_total_cost_token = (
        float(episode["raw_terminal_penalty"])
        + float(episode["raw_path_cost_component"])
        + float(combined_reasoning_components["raw_reasoning_cost_component_token"])
    )
    combined_raw_total_cost = (
        combined_raw_total_cost_api
        if combined_reasoning_components["reasoning_cost_mode_default"] == "api"
        else combined_raw_total_cost_token
    )
    combined_total_cost = min(
        combined_raw_total_cost / TELECOM_MMS_TOTAL_UPPER_BOUND_V2_DEFAULT,
        1.0,
    )
    episode.update(
        {
            "mechanism": mechanism,
            "backbone_policy": BACKBONE_POLICY,
            "stage_choice_trace": stage_choice_trace,
            "chosen_child_per_stage": list(selection_meta.get("chosen_child_per_stage", [])),
            "chosen_child_alias_per_stage": list(selection_meta.get("chosen_child_alias_per_stage", [])),
            "chosen_child_per_stage_real": list(selection_meta.get("chosen_child_per_stage_real", [])),
            "candidate_count_per_stage": list(selection_meta.get("candidate_count_per_stage", [])),
            "candidate_aliases_per_stage": list(selection_meta.get("candidate_aliases_per_stage", [])),
            "candidate_real_children_per_stage": list(selection_meta.get("candidate_real_children_per_stage", [])),
            "fallback_used_per_stage": list(selection_meta.get("fallback_used_per_stage", [])),
            "chooser_raw_output_per_stage": list(selection_meta.get("chooser_raw_output_per_stage", [])),
            "theta_of_chosen_child_per_stage": list(selection_meta.get("theta_of_chosen_child_per_stage", [])),
            "max_theta_per_stage": list(selection_meta.get("max_theta_per_stage", [])),
            "followed_max_theta_per_stage": list(selection_meta.get("followed_max_theta_per_stage", [])),
            "invalid_output_per_stage": _extract_invalid_flags(stage_choice_trace),
            "chooser_llm_call_count": chooser_llm_call_count,
            "executor_llm_call_count": executor_llm_call_count,
            "chooser_prompt_tokens_total": chooser_prompt_tokens_total,
            "executor_prompt_tokens_total": executor_prompt_tokens_total,
            "chooser_completion_tokens_total": chooser_completion_tokens_total,
            "executor_completion_tokens_total": executor_completion_tokens_total,
            "chooser_total_tokens_total": chooser_total_tokens_total,
            "executor_total_tokens_total": executor_total_tokens_total,
            "chooser_api_cost_total_usd_raw": chooser_api_cost_total_usd_raw,
            "executor_api_cost_total_usd_raw": executor_api_cost_total_usd_raw,
            "chooser_generation_time_total_seconds": chooser_generation_time_total_seconds,
            "executor_generation_time_total_seconds": executor_generation_time_total_seconds,
            "chooser_llm_round_trip_total_seconds": chooser_llm_round_trip_total_seconds,
            "executor_llm_round_trip_total_seconds": executor_llm_round_trip_total_seconds,
            "chooser_episode_wall_clock_seconds": chooser_episode_wall_clock_seconds,
            "executor_episode_wall_clock_seconds": executor_episode_wall_clock_seconds,
            "chooser_raw_reasoning_cost_component_api": chooser_reasoning_components[
                "raw_reasoning_cost_component_api"
            ],
            "chooser_raw_reasoning_cost_component_token": chooser_reasoning_components[
                "raw_reasoning_cost_component_token"
            ],
            "executor_raw_reasoning_cost_component_api": episode.get(
                "raw_reasoning_cost_component_api"
            ),
            "executor_raw_reasoning_cost_component_token": episode.get(
                "raw_reasoning_cost_component_token"
            ),
            "llm_call_count": chooser_llm_call_count + executor_llm_call_count,
            "prompt_tokens_total": combined_prompt_tokens_total,
            "completion_tokens_total": combined_completion_tokens_total,
            "total_tokens_total": combined_total_tokens_total,
            "api_cost_total_usd_raw": combined_api_cost_total_usd_raw,
            "generation_time_total_seconds": combined_generation_time_total_seconds,
            "llm_round_trip_total_seconds": combined_llm_round_trip_total_seconds,
            "episode_wall_clock_seconds": combined_episode_wall_clock_seconds,
            "raw_reasoning_cost_component": combined_reasoning_components[
                "raw_reasoning_cost_component"
            ],
            "raw_reasoning_cost_component_api": combined_reasoning_components[
                "raw_reasoning_cost_component_api"
            ],
            "raw_reasoning_cost_component_token": combined_reasoning_components[
                "raw_reasoning_cost_component_token"
            ],
            "raw_total_cost": combined_raw_total_cost,
            "raw_total_cost_api": combined_raw_total_cost_api,
            "raw_total_cost_token": combined_raw_total_cost_token,
            "total_cost": combined_total_cost,
            "selection_meta": selection_meta,
        }
    )
    return episode


def build_mechanism_summary_fields(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    followed = [
        float(flag)
        for episode in episodes
        for flag in episode.get("followed_max_theta_per_stage", [])
    ]
    fallback = [
        float(flag)
        for episode in episodes
        for flag in episode.get("fallback_used_per_stage", [])
    ]
    invalid = [
        float(flag)
        for episode in episodes
        for flag in episode.get("invalid_output_per_stage", [])
    ]
    stagewise_decision_count = sum(
        len(episode.get("fallback_used_per_stage", []))
        for episode in episodes
    )
    return {
        "max_theta_follow_rate": mean(followed) if followed else None,
        "fallback_rate": mean(fallback),
        "invalid_output_rate": mean(invalid),
        "stagewise_decision_count": stagewise_decision_count,
        "mean_chooser_llm_call_count": mean(
            [ep.get("chooser_llm_call_count", 0) for ep in episodes]
        ),
        "mean_executor_llm_call_count": mean(
            [ep.get("executor_llm_call_count", 0) for ep in episodes]
        ),
        "mean_chooser_total_tokens": mean(
            [ep.get("chooser_total_tokens_total", 0.0) for ep in episodes]
        ),
        "mean_executor_total_tokens": mean(
            [ep.get("executor_total_tokens_total", 0.0) for ep in episodes]
        ),
        "mean_chooser_api_cost_usd_raw": mean(
            [ep.get("chooser_api_cost_total_usd_raw", 0.0) for ep in episodes]
        ),
        "mean_executor_api_cost_usd_raw": mean(
            [ep.get("executor_api_cost_total_usd_raw", 0.0) for ep in episodes]
        ),
    }


def build_summary(
    *,
    mechanism: str,
    dataset: str,
    repeats: int,
    model: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    from run_shared_basin_repeated_smoke import build_summary as base_build_summary  # noqa: E402

    summary = base_build_summary(
        method=BACKBONE_POLICY,
        dataset=dataset,
        repeats=repeats,
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
    )
    summary["mechanism"] = mechanism
    summary["backbone_policy"] = BACKBONE_POLICY
    summary["test_name"] = f"{BACKBONE_POLICY}_{mechanism}_smoke10x{repeats}_{FAMILY_KIND}_full_llm"
    summary.update(build_mechanism_summary_fields(episodes))
    return summary


def build_partial_summary(
    *,
    mechanism: str,
    dataset: str,
    repeats: int,
    model: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    total_episodes: int,
    status: str = "running",
) -> dict[str, Any]:
    summary = build_summary(
        mechanism=mechanism,
        dataset=dataset,
        repeats=repeats,
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
    )
    summary.update(
        {
            "scheduled_episodes": total_episodes,
            "completed_episodes": len(episodes),
            "status": status,
            "completed_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
            "completed_raw_terminal_penalty": sum(ep["raw_terminal_penalty"] for ep in episodes),
        }
    )
    return summary


def persist_mechanism_state(
    *,
    mechanism_dir: Path,
    mechanism: str,
    episodes: list[dict[str, Any]],
    policy: Any,
    total_episodes: int,
    model: str,
    dataset: str,
    repeats: int,
    oracle_summary: dict[str, Any],
) -> None:
    add_cumulative_fields(episodes)
    checkpoint_payload = {
        "mechanism": mechanism,
        "completed_count": len(episodes),
        "episodes": episodes,
        "model": model,
        "policy": policy,
    }
    write_bytes_atomic(mechanism_dir / "checkpoint.pkl", pickle.dumps(checkpoint_payload))
    write_jsonl(mechanism_dir / "episodes.partial.jsonl", episodes)
    write_json(
        mechanism_dir / "progress.json",
        build_progress_payload(
            mechanism=mechanism,
            completed_count=len(episodes),
            total_episodes=total_episodes,
            model=model,
            status="complete" if len(episodes) == total_episodes else "running",
        ),
    )
    partial_summary = build_partial_summary(
        mechanism=mechanism,
        dataset=dataset,
        repeats=repeats,
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
        total_episodes=total_episodes,
        status="complete" if len(episodes) == total_episodes else "running",
    )
    write_json(mechanism_dir / "summary_partial.json", partial_summary)
    if len(episodes) == total_episodes:
        write_json(mechanism_dir / "episodes.json", episodes)
        write_json(mechanism_dir / "summary.json", partial_summary)
        write_json(mechanism_dir / "summary_with_oracle.json", partial_summary)


def load_mechanism_checkpoint(mechanism_dir: Path) -> dict[str, Any] | None:
    checkpoint_path = mechanism_dir / "checkpoint.pkl"
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def run_mechanism_worker(
    *,
    run_dir: Path,
    mechanism: str,
) -> None:
    ensure_model_env(required=True)
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    selected = context["selected"]
    oracle_summary = context["oracle_summary"]
    specialist_task_ids = context["specialist_task_ids"]
    total_episodes = len(selected)
    mechanism_dir = run_dir / mechanism
    mechanism_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(executor_name=EXECUTOR_NAME)
    checkpoint = load_mechanism_checkpoint(mechanism_dir)
    if checkpoint is not None:
        policy = checkpoint["policy"]
        episodes = list(checkpoint["episodes"])
        model = checkpoint.get("model", getattr(env.family_executor, "model", MODEL_REQUIRED))
    else:
        policy = DirectMultiStageExp3Policy(seed=SEED)
        policy.bind_env(env)
        policy.reset()
        episodes = []
        model = getattr(env.family_executor, "model", MODEL_REQUIRED)

    completed_count = len(episodes)
    if completed_count >= total_episodes:
        persist_mechanism_state(
            mechanism_dir=mechanism_dir,
            mechanism=mechanism,
            episodes=episodes,
            policy=policy,
            total_episodes=total_episodes,
            model=model,
            dataset=run_config["dataset"],
            repeats=int(run_config["repeats"]),
            oracle_summary=oracle_summary,
        )
        return

    for local_offset in range(completed_count, total_episodes):
        row = selected[local_offset]
        episode_index = int(row["episode_index"])
        print(
            f"[run] mechanism={mechanism} episode={episode_index + 1}/{len(selected)} "
            f"repeat={row['repeat_index'] + 1} pos={row['position_in_cycle']} dataset_index={row['dataset_index']}",
            flush=True,
        )
        path, selection_meta, should_update = choose_path_with_mechanism(
            policy,
            row["instance"],
            env,
            mechanism,
        )
        env.reset(row["instance"])
        result = env.run_path(path)
        if should_update:
            policy.update(result)
        if mechanism == "agent_only":
            selection_info: dict[str, Any] = {}
            update_info: dict[str, Any] = {}
        else:
            selection_info = policy.get_last_selection_info() if hasattr(policy, "get_last_selection_info") else {}
            state = policy.get_state() if hasattr(policy, "get_state") else {}
            update_info = state.get("last_update_info", {}) if isinstance(state, dict) else {}
        episodes.append(
            flatten_mechanism_episode(
                episode_index=episode_index,
                row=row,
                result=result,
                mechanism=mechanism,
                oracle_summary=oracle_summary,
                selection_info=selection_info if isinstance(selection_info, dict) else {},
                update_info=update_info if isinstance(update_info, dict) else {},
                specialist_task_ids=specialist_task_ids,
                selection_meta=selection_meta if isinstance(selection_meta, dict) else {},
            )
        )
        persist_mechanism_state(
            mechanism_dir=mechanism_dir,
            mechanism=mechanism,
            episodes=episodes,
            policy=policy,
            total_episodes=total_episodes,
            model=model,
            dataset=run_config["dataset"],
            repeats=int(run_config["repeats"]),
            oracle_summary=oracle_summary,
        )


def merge_mechanism_results(run_dir: Path, mechanism: str) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    oracle_summary = context["oracle_summary"]
    total_episodes = int(run_config["horizon"])
    mechanism_dir = run_dir / mechanism
    model = run_config["model"]
    progress = load_json(mechanism_dir / "progress.json")
    if progress["completed_episodes"] != total_episodes:
        raise RuntimeError(f"Mechanism {mechanism} is incomplete: {progress}")
    episodes = load_json(mechanism_dir / "episodes.json")
    model = progress.get("model", model)

    expected_indices = list(range(total_episodes))
    actual_indices = [int(row["episode_index"]) for row in episodes]
    if actual_indices != expected_indices:
        raise RuntimeError(f"Merged episode indices mismatch for {mechanism}")

    add_cumulative_fields(episodes)
    summary = build_summary(
        mechanism=mechanism,
        dataset=run_config["dataset"],
        repeats=int(run_config["repeats"]),
        model=model,
        oracle_summary=oracle_summary,
        episodes=episodes,
    )
    specialist_summary = build_specialist_summary(episodes)

    merged_dir = mechanism_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    write_json(mechanism_dir / "episodes.json", episodes)
    write_json(mechanism_dir / "summary.json", summary)
    write_json(mechanism_dir / "summary_with_oracle.json", summary)
    write_json(mechanism_dir / "specialist_summary.json", specialist_summary)
    write_json(merged_dir / "episodes.json", episodes)
    write_json(merged_dir / "summary.json", summary)
    write_json(merged_dir / "summary_with_oracle.json", summary)
    write_json(merged_dir / "specialist_summary.json", specialist_summary)
    write_text_atomic(
        mechanism_dir / "smoke_summary.md",
        json.dumps({"summary": summary, "specialist_summary": specialist_summary}, ensure_ascii=False, indent=2),
    )

    return {
        "summary": summary,
        "specialist_summary": specialist_summary,
        "episodes": episodes,
    }


def build_compare_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "mechanism": summary["mechanism"],
            "backbone_policy": summary["backbone_policy"],
            "total_cost_mean": summary["total_cost_mean"],
            "raw_total_cost_mean": summary["raw_total_cost_mean"],
            "raw_total_cost_api_mean": summary["raw_total_cost_api_mean"],
            "raw_total_cost_token_mean": summary["raw_total_cost_token_mean"],
            "raw_outcome_penalty_mean": summary["raw_outcome_penalty_mean"],
            "raw_policy_penalty_mean": summary["raw_policy_penalty_mean"],
            "raw_terminal_penalty_mean": summary["raw_terminal_penalty_mean"],
            "raw_path_cost_component_mean": summary["raw_path_cost_component_mean"],
            "raw_reasoning_cost_component_mean": summary["raw_reasoning_cost_component_mean"],
            "raw_reasoning_cost_component_api_mean": summary[
                "raw_reasoning_cost_component_api_mean"
            ],
            "raw_reasoning_cost_component_token_mean": summary[
                "raw_reasoning_cost_component_token_mean"
            ],
            "exact_match_mean": summary["exact_match_mean"],
            "mean_llm_call_count": summary["mean_llm_call_count"],
            "mean_total_tokens": summary["mean_total_tokens"],
            "mean_api_cost_usd_raw": summary["mean_api_cost_usd_raw"],
            "mean_generation_time_seconds": summary["mean_generation_time_seconds"],
            "mean_episode_wall_clock_seconds": summary[
                "mean_episode_wall_clock_seconds"
            ],
            "mean_chooser_llm_call_count": summary["mean_chooser_llm_call_count"],
            "mean_executor_llm_call_count": summary["mean_executor_llm_call_count"],
            "max_theta_follow_rate": summary["max_theta_follow_rate"],
            "fallback_rate": summary["fallback_rate"],
            "invalid_output_rate": summary["invalid_output_rate"],
        }
        for summary in summaries
    ]
    return sorted(rows, key=lambda row: (row["total_cost_mean"], row["mechanism"]))


def merge_all_results(run_dir: Path) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    summaries: list[dict[str, Any]] = []

    for mechanism in run_config["mechanisms"]:
        summaries.append(load_json(run_dir / mechanism / "summary.json"))

    compare_rows = build_compare_rows(summaries)
    write_json(run_dir / "repeated_smoke_compare.json", compare_rows)
    write_csv(run_dir / "repeated_smoke_compare.csv", compare_rows)
    write_text_atomic(run_dir / "repeated_smoke_compare.md", compare_rows_to_markdown(compare_rows))
    return {"compare_rows": compare_rows}


def orchestrate_run(
    *,
    data_path: Path,
    output_dir: Path,
    repeats: int,
    mechanisms: list[str],
) -> Path:
    model_name = ensure_model_env(required=True)
    validate_mechanisms(mechanisms)
    run_dir = initialize_run(
        data_path=data_path,
        output_dir=output_dir,
        repeats=repeats,
        mechanisms=mechanisms,
        model_name=model_name,
    )
    script_path = Path(__file__).resolve()
    launched: list[tuple[str, subprocess.Popen[Any], Any]] = []

    for mechanism in mechanisms:
        mechanism_dir = run_dir / mechanism
        mechanism_dir.mkdir(parents=True, exist_ok=True)
        log_path = mechanism_dir / "runner.log"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"[launch] {datetime.now().isoformat()} mechanism={mechanism}\n")
        log_handle.flush()
        cmd = [
            sys.executable,
            str(script_path),
            "run-mechanism",
            "--run-dir",
            str(run_dir),
            "--mechanism",
            mechanism,
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(RUNNER_ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        launched.append((mechanism, process, log_handle))

    failures: list[dict[str, Any]] = []
    for mechanism, process, log_handle in launched:
        return_code = process.wait()
        log_handle.write(
            f"[exit] {datetime.now().isoformat()} mechanism={mechanism} return_code={return_code}\n"
        )
        log_handle.close()
        if return_code != 0:
            failures.append({"mechanism": mechanism, "return_code": return_code})

    if failures:
        write_json(run_dir / "orchestrator_failures.json", failures)
        raise SystemExit(f"One or more mechanism runs failed: {failures}")

    for mechanism in mechanisms:
        merge_mechanism_results(run_dir, mechanism)
    merge_all_results(run_dir)
    return run_dir


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated shared-basin mechanism smoke.")
    subparsers = parser.add_subparsers(dest="command")

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--data", type=Path, default=DATASET_DEFAULT)
    common_run.add_argument("--output-dir", type=Path, required=True)
    common_run.add_argument("--repeats", type=int, default=10)
    common_run.add_argument("--mechanisms", nargs="+", default=list(DEFAULT_MECHANISMS))

    setup_parser = subparsers.add_parser("setup", parents=[common_run])
    setup_parser.set_defaults(command="setup")

    orchestrate_parser = subparsers.add_parser("orchestrate", parents=[common_run])
    orchestrate_parser.set_defaults(command="orchestrate")

    mechanism_parser = subparsers.add_parser("run-mechanism")
    mechanism_parser.add_argument("--run-dir", type=Path, required=True)
    mechanism_parser.add_argument("--mechanism", type=str, required=True)
    mechanism_parser.set_defaults(command="run-mechanism")

    merge_mechanism_parser = subparsers.add_parser("merge-mechanism")
    merge_mechanism_parser.add_argument("--run-dir", type=Path, required=True)
    merge_mechanism_parser.add_argument("--mechanism", type=str, required=True)
    merge_mechanism_parser.set_defaults(command="merge-mechanism")

    merge_all_parser = subparsers.add_parser("merge-all")
    merge_all_parser.add_argument("--run-dir", type=Path, required=True)
    merge_all_parser.set_defaults(command="merge-all")
    return parser


def main() -> None:
    parser = build_cli()
    argv = sys.argv[1:]
    known_commands = {"setup", "orchestrate", "run-mechanism", "merge-mechanism", "merge-all"}
    if not argv or argv[0] not in known_commands:
        argv = ["orchestrate", *argv]
    args = parser.parse_args(argv)

    if args.command == "setup":
        model_name = ensure_model_env(required=True)
        validate_mechanisms(args.mechanisms)
        run_dir = initialize_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            mechanisms=args.mechanisms,
            model_name=model_name,
        )
        print(str(run_dir))
        return

    if args.command == "orchestrate":
        validate_mechanisms(args.mechanisms)
        run_dir = orchestrate_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            mechanisms=args.mechanisms,
        )
        print(str(run_dir))
        return

    if args.command == "run-mechanism":
        validate_mechanisms([args.mechanism])
        run_mechanism_worker(run_dir=args.run_dir, mechanism=args.mechanism)
        print(str(args.run_dir / args.mechanism))
        return

    if args.command == "merge-mechanism":
        payload = merge_mechanism_results(args.run_dir, args.mechanism)
        print(json.dumps(payload["summary"], ensure_ascii=False))
        return

    if args.command == "merge-all":
        payload = merge_all_results(args.run_dir)
        print(json.dumps(payload["compare_rows"], ensure_ascii=False))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
