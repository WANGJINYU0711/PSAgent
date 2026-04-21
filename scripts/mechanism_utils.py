"""Mechanism helpers for fixed-tree baseline runners.

This module exposes one runner-facing entrypoint:

- ``choose_path_with_mechanism(...)``

Supported mechanisms:

- ``algorithm_direct``: policy selects the full path directly
- ``theta_guided_agent``: policy exposes raw stage-local theta, then an LLM
  chooses one legal child at each stage
- ``agent_only``: an LLM chooses one legal child at each stage without seeing
  any algorithm signal
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fixed_tree_env import compute_llm_bench_reasoning_components


ROOT = Path(__file__).resolve().parents[1]
TAU2_ROOT = ROOT / "tau2-bench"
TAU2_VENV_PYTHON = TAU2_ROOT / ".venv" / "bin" / "python"
STAGE_CHOICE_BRIDGE = Path(__file__).with_name("_stage_choice_bridge.py")
DEFAULT_STAGE_CHOICE_MODEL = os.environ.get(
    "PSAGENT_STAGE_CHOICE_MODEL",
    os.environ.get(
        "PSAGENT_THETA_GUIDED_MODEL",
        os.environ.get("PSAGENT_LLM_BENCH_MODEL", "gpt-4o-mini"),
    ),
)


def _normalize_usage_breakdown(row: dict[str, Any] | None) -> dict[str, float]:
    if not isinstance(row, dict):
        return {
            "prompt_tokens_total": 0.0,
            "completion_tokens_total": 0.0,
            "total_tokens_total": 0.0,
        }
    prompt_tokens_total = float(
        row.get("prompt_tokens_total", row.get("prompt_tokens", 0.0)) or 0.0
    )
    completion_tokens_total = float(
        row.get("completion_tokens_total", row.get("completion_tokens", 0.0)) or 0.0
    )
    total_tokens_total = float(row.get("total_tokens_total", row.get("total_tokens", 0.0)) or 0.0)
    if total_tokens_total <= 0.0:
        total_tokens_total = prompt_tokens_total + completion_tokens_total
    return {
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
    }


def _chooser_resource_usage_from_llm_messages(raw_output: list[dict[str, Any]]) -> dict[str, float]:
    prompt_tokens_total = 0.0
    completion_tokens_total = 0.0
    total_tokens_total = 0.0
    api_cost_total_usd_raw = 0.0
    generation_time_total_seconds = 0.0
    llm_round_trip_total_seconds = 0.0
    for row in raw_output:
        usage_breakdown = _normalize_usage_breakdown(
            row.get("usage_breakdown") or row.get("usage")
        )
        prompt_tokens_total += usage_breakdown["prompt_tokens_total"]
        completion_tokens_total += usage_breakdown["completion_tokens_total"]
        total_tokens_total += usage_breakdown["total_tokens_total"]
        api_cost_total_usd_raw += float(row.get("cost") or 0.0)
        generation_time_total_seconds += float(row.get("generation_time_seconds") or 0.0)
        llm_round_trip_total_seconds += float(row.get("round_trip_seconds") or 0.0)
    return {
        "llm_call_count": len(raw_output),
        "prompt_tokens_total": prompt_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "total_tokens_total": total_tokens_total,
        "api_cost_total_usd_raw": api_cost_total_usd_raw,
        "generation_time_total_seconds": generation_time_total_seconds,
        "llm_round_trip_total_seconds": llm_round_trip_total_seconds,
    }


def _aggregate_chooser_resource_metrics(
    chooser_raw_output_per_stage: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    per_stage = [
        _chooser_resource_usage_from_llm_messages(raw_output)
        for raw_output in chooser_raw_output_per_stage
    ]
    totals = {
        "chooser_llm_call_count": sum(int(row["llm_call_count"]) for row in per_stage),
        "chooser_prompt_tokens_total": sum(
            float(row["prompt_tokens_total"]) for row in per_stage
        ),
        "chooser_completion_tokens_total": sum(
            float(row["completion_tokens_total"]) for row in per_stage
        ),
        "chooser_total_tokens_total": sum(
            float(row["total_tokens_total"]) for row in per_stage
        ),
        "chooser_api_cost_total_usd_raw": sum(
            float(row["api_cost_total_usd_raw"]) for row in per_stage
        ),
        "chooser_generation_time_total_seconds": sum(
            float(row["generation_time_total_seconds"]) for row in per_stage
        ),
        "chooser_llm_round_trip_total_seconds": sum(
            float(row["llm_round_trip_total_seconds"]) for row in per_stage
        ),
        "chooser_episode_wall_clock_seconds": sum(
            float(row["llm_round_trip_total_seconds"]) for row in per_stage
        ),
        "chooser_stage_prompt_tokens": [
            float(row["prompt_tokens_total"]) for row in per_stage
        ],
        "chooser_stage_completion_tokens": [
            float(row["completion_tokens_total"]) for row in per_stage
        ],
        "chooser_stage_total_tokens": [
            float(row["total_tokens_total"]) for row in per_stage
        ],
        "chooser_stage_api_cost_usd": [
            float(row["api_cost_total_usd_raw"]) for row in per_stage
        ],
        "chooser_stage_generation_time_seconds": [
            float(row["generation_time_total_seconds"]) for row in per_stage
        ],
        "chooser_stage_llm_round_trip_seconds": [
            float(row["llm_round_trip_total_seconds"]) for row in per_stage
        ],
    }
    totals.update(
        compute_llm_bench_reasoning_components(
            prompt_tokens_total=totals["chooser_prompt_tokens_total"],
            completion_tokens_total=totals["chooser_completion_tokens_total"],
            api_cost_total_usd_raw=totals["chooser_api_cost_total_usd_raw"],
        )
    )
    totals["chooser_raw_reasoning_cost_component"] = totals[
        "raw_reasoning_cost_component"
    ]
    totals["chooser_raw_reasoning_cost_component_api"] = totals[
        "raw_reasoning_cost_component_api"
    ]
    totals["chooser_raw_reasoning_cost_component_token"] = totals[
        "raw_reasoning_cost_component_token"
    ]
    return totals


def _sync_full_share_selection(policy: Any, env: Any, path: list[str]) -> None:
    current_prefix: tuple[str, ...] = ()
    policy.last_stage_probs = {}
    policy.last_path_prob = 1.0
    policy.last_estimated_loss = None
    for stage_name, agent_id in zip(env.STAGE_NAMES, path):
        child_prefixes = policy._child_prefixes(current_prefix, stage_name, env)
        child_weights = [
            max(0.0, policy.prefix_weights.get(child_prefix, 0.0))
            for child_prefix in child_prefixes
        ]
        selected_prefix = tuple(list(current_prefix) + [agent_id])
        selected_idx = child_prefixes.index(selected_prefix)
        if sum(child_weights) <= 0:
            prob = 1.0 / len(child_prefixes)
        else:
            prob = child_weights[selected_idx] / sum(child_weights)
        policy.last_stage_probs[stage_name] = prob
        policy.last_path_prob *= prob
        current_prefix = selected_prefix


def _sync_risky_ps_selection(policy: Any, env: Any, path: list[str]) -> None:
    current_prefix: tuple[str, ...] = ()
    prefix_reach_prob = 1.0
    policy.last_stage_probs = {}
    policy.last_path_prob = 1.0
    policy.last_sampled_edges = []
    policy.last_update_info = {}
    for stage_name, agent_id in zip(env.STAGE_NAMES, path):
        child_prefixes = policy._child_prefixes(current_prefix, stage_name, env)
        if policy.safe_prefixes.get(current_prefix, False):
            probs = policy._safe_child_probs(current_prefix, child_prefixes)
        else:
            probs = policy._risky_child_probs(current_prefix, child_prefixes)
        child_prefix = tuple(list(current_prefix) + [agent_id])
        selected_idx = child_prefixes.index(child_prefix)
        conditional_prob = probs[selected_idx]
        policy.last_stage_probs[stage_name] = conditional_prob
        policy.last_path_prob *= conditional_prob
        policy.last_sampled_edges.append(
            {
                "prefix": current_prefix,
                "child_prefix": child_prefix,
                "prefix_reach_prob": prefix_reach_prob,
                "conditional_prob": conditional_prob,
                "edge_prob": prefix_reach_prob * conditional_prob,
                "is_safe_prefix": policy.safe_prefixes.get(current_prefix, False),
            }
        )
        prefix_reach_prob *= conditional_prob
        current_prefix = child_prefix


def _sync_stagewise_exp3_selection(policy: Any, env: Any, path: list[str]) -> None:
    policy.last_path_probs = []
    policy.last_stage_probs = {}
    policy.last_stage_arm_counts = {}
    policy.last_selected_edges = []
    path_prob = 1.0
    current_prefix: tuple[str, ...] = ()
    for stage_name, agent_id in zip(env.STAGE_NAMES, path):
        agent_ids = policy._legal_agent_ids_for_prefix(current_prefix, stage_name, env)
        child_prefixes = policy._child_prefixes(current_prefix, agent_ids)
        probs = policy._stage_probs(current_prefix, child_prefixes)
        child_prefix = tuple(current_prefix + (agent_id,))
        selected_idx = child_prefixes.index(child_prefix)
        prob = probs[selected_idx]
        policy.last_path_probs.append(prob)
        policy.last_stage_probs[stage_name] = prob
        policy.last_stage_arm_counts[stage_name] = len(child_prefixes)
        policy.last_selected_edges.append(
            {
                "stage_name": stage_name,
                "prefix": current_prefix,
                "child_prefix": child_prefix,
                "path_prob": prob,
                "arm_count": len(child_prefixes),
                "weight_before_update": policy._edge_weight(current_prefix, child_prefix),
            }
        )
        path_prob *= prob
        current_prefix = child_prefix
    policy.last_path_prob = path_prob


def sync_policy_selection(policy: Any, env: Any, path: list[str]) -> None:
    if hasattr(policy, "shared_edge_mass") and hasattr(policy, "unshared_edge_mass"):
        _sync_risky_ps_selection(policy, env, path)
        return
    if hasattr(policy, "prefix_weights") and hasattr(policy, "leaf_weights"):
        _sync_full_share_selection(policy, env, path)
        return
    if hasattr(policy, "_stage_probs") and hasattr(policy, "stage_agent_ids"):
        _sync_stagewise_exp3_selection(policy, env, path)
        return


def _trim_text(text: str | None, max_chars: int = 700) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _prefix_alias_history(choice_alias_history: list[str]) -> list[dict[str, Any]]:
    return [
        {
            "step": idx + 1,
            "option_alias": alias,
        }
        for idx, alias in enumerate(choice_alias_history)
    ]


def _stage_task_summary(
    instance: dict[str, Any],
    env: Any,
    stage_name: str,
    choice_alias_history: list[str],
) -> dict[str, Any]:
    stage1_input = instance.get("stage1", {}).get("input", {})
    return {
        "original_task_id": instance.get("original_task_id"),
        "reason_for_call": _trim_text(stage1_input.get("reason_for_call")),
        "known_info": _trim_text(stage1_input.get("known_info")),
        "current_stage": stage_name,
        "previous_choices": _prefix_alias_history(choice_alias_history),
        "stage_names": list(env.STAGE_NAMES),
    }


def _legal_agent_ids_for_prefix_env(
    current_prefix: tuple[str, ...],
    stage_name: str,
    env: Any,
) -> list[str]:
    family_spec = getattr(env, "family_spec", None)
    allowed_children = getattr(family_spec, "allowed_children", None)
    if not allowed_children:
        return list(env.stage_agents[stage_name])
    child_ids = allowed_children.get(current_prefix)
    if child_ids is None:
        return list(env.stage_agents[stage_name])
    return list(child_ids)


def _build_stage_choice_candidates(
    policy: Any,
    env: Any,
    current_prefix: tuple[str, ...],
    stage_name: str,
) -> list[dict[str, Any]]:
    agent_ids = policy._legal_agent_ids_for_prefix(current_prefix, stage_name, env)
    child_prefixes = policy._child_prefixes(current_prefix, agent_ids)
    candidates: list[dict[str, Any]] = []
    for idx, (agent_id, child_prefix) in enumerate(zip(agent_ids, child_prefixes), start=1):
        weight = float(policy._edge_weight(current_prefix, child_prefix))
        theta = math.log(max(weight, 1e-12))
        candidates.append(
            {
                "option_alias": f"option_{idx}",
                "child_id": agent_id,
                "child_prefix": child_prefix,
                "theta": float(theta),
            }
        )
    return candidates


def _build_agent_only_stage_candidates(
    env: Any,
    current_prefix: tuple[str, ...],
    stage_name: str,
) -> list[dict[str, Any]]:
    agent_ids = _legal_agent_ids_for_prefix_env(current_prefix, stage_name, env)
    candidates: list[dict[str, Any]] = []
    for idx, agent_id in enumerate(agent_ids, start=1):
        candidates.append(
            {
                "option_alias": f"option_{idx}",
                "child_id": agent_id,
                "child_prefix": tuple(current_prefix + (agent_id,)),
            }
        )
    return candidates


def _build_theta_guided_stage_prompt(
    instance: dict[str, Any],
    env: Any,
    stage_name: str,
    candidates: list[dict[str, Any]],
    choice_alias_history: list[str],
) -> tuple[str, str]:
    prompt_candidates = [
        {
            "option_alias": row["option_alias"],
            "theta": row["theta"],
        }
        for row in candidates
    ]
    system_prompt = (
        "You are choosing the next option for one stage in a telecom troubleshooting tree. "
        "You must choose exactly one option_alias from the current candidate list. "
        "Theta is the algorithm's current preference score for this option under the current prefix. "
        "Higher theta means the algorithm currently prefers this option more. "
        "Lower theta means it currently prefers this option less. "
        "Theta is not a probability. "
        "Theta values are only comparable among the current options at this stage. "
        "Use theta as one signal, not as a mandatory instruction. "
        "Return only JSON with keys option_alias and optional rationale."
    )
    user_prompt = json.dumps(
        {
            "task_summary": _stage_task_summary(instance, env, stage_name, choice_alias_history),
            "candidate_options": prompt_candidates,
            "instructions": {
                "must_choose_one_option_alias": True,
                "theta_definition": {
                    "meaning": "the algorithm's current preference score for this option under the current prefix",
                    "higher_means": "currently more preferred",
                    "lower_means": "currently less preferred",
                    "not_probability": True,
                    "comparable_scope": "current stage options only",
                    "use_with_context": True,
                },
                "output_schema": {"option_alias": "string", "rationale": "optional short string"},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return system_prompt, user_prompt


def _build_agent_only_stage_prompt(
    instance: dict[str, Any],
    env: Any,
    stage_name: str,
    candidates: list[dict[str, Any]],
    choice_alias_history: list[str],
) -> tuple[str, str]:
    prompt_candidates = [
        {
            "option_alias": row["option_alias"],
        }
        for row in candidates
    ]
    system_prompt = (
        "You are choosing the next option for one stage in a telecom troubleshooting tree. "
        "You must choose exactly one option_alias from the current candidate list. "
        "There is no algorithm score, no theta, and no oracle signal available. "
        "Use only the task context, current stage, previous anonymous choices, and the anonymous candidate list. "
        "Return only JSON with keys option_alias and optional rationale."
    )
    user_prompt = json.dumps(
        {
            "task_summary": _stage_task_summary(instance, env, stage_name, choice_alias_history),
            "candidate_options": prompt_candidates,
            "instructions": {
                "must_choose_one_option_alias": True,
                "output_schema": {"option_alias": "string", "rationale": "optional short string"},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return system_prompt, user_prompt


def _run_stage_choice_llm_selector(
    *,
    model: str,
    system_prompt: str,
    user_prompt: str,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "llm_args": {"temperature": 0.0},
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "max_rounds": 2,
    }
    proc = subprocess.run(
        [str(TAU2_VENV_PYTHON), str(STAGE_CHOICE_BRIDGE)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(TAU2_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "stage-choice bridge failed: "
            + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
        )
    return json.loads(proc.stdout)


def _run_theta_guided_stage_choice_llm_selector(
    instance: dict[str, Any],
    env: Any,
    stage_name: str,
    candidates: list[dict[str, Any]],
    choice_alias_history: list[str],
) -> dict[str, Any]:
    system_prompt, user_prompt = _build_theta_guided_stage_prompt(
        instance,
        env,
        stage_name,
        candidates,
        choice_alias_history,
    )
    return _run_stage_choice_llm_selector(
        model=DEFAULT_STAGE_CHOICE_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _run_agent_only_stage_choice_llm_selector(
    instance: dict[str, Any],
    env: Any,
    stage_name: str,
    candidates: list[dict[str, Any]],
    choice_alias_history: list[str],
) -> dict[str, Any]:
    system_prompt, user_prompt = _build_agent_only_stage_prompt(
        instance,
        env,
        stage_name,
        candidates,
        choice_alias_history,
    )
    return _run_stage_choice_llm_selector(
        model=DEFAULT_STAGE_CHOICE_MODEL,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )


def _resolve_stage_choice(
    candidates: list[dict[str, Any]],
    final_output: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_by_alias = {row["option_alias"]: row for row in candidates}
    max_theta = max(row["theta"] for row in candidates)
    fallback = sorted(
        [
            row
            for row in candidates
            if math.isclose(float(row["theta"]), float(max_theta), rel_tol=1e-9, abs_tol=1e-12)
        ],
        key=lambda row: row["option_alias"],
    )[0]

    if isinstance(final_output, dict):
        option_alias = final_output.get("option_alias")
        if isinstance(option_alias, str) and option_alias in candidate_by_alias:
            chosen = candidate_by_alias[option_alias]
            return chosen, {
                "fallback_used": False,
                "invalid_output": False,
                "rationale": final_output.get("rationale"),
                "max_theta": max_theta,
                "followed_max_theta": math.isclose(
                    float(chosen["theta"]),
                    float(max_theta),
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                ),
            }

    return fallback, {
        "fallback_used": True,
        "invalid_output": True,
        "rationale": "fallback_max_theta_invalid_llm_output",
        "max_theta": max_theta,
        "followed_max_theta": True,
    }


def _resolve_agent_only_stage_choice(
    candidates: list[dict[str, Any]],
    final_output: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    candidate_by_alias = {row["option_alias"]: row for row in candidates}
    fallback = sorted(candidates, key=lambda row: row["option_alias"])[0]

    if isinstance(final_output, dict):
        option_alias = final_output.get("option_alias")
        if isinstance(option_alias, str) and option_alias in candidate_by_alias:
            chosen = candidate_by_alias[option_alias]
            return chosen, {
                "fallback_used": False,
                "invalid_output": False,
                "rationale": final_output.get("rationale"),
            }

    return fallback, {
        "fallback_used": True,
        "invalid_output": True,
        "rationale": "fallback_first_child_invalid_llm_output",
    }


def choose_path_with_mechanism(
    policy: Any,
    instance: dict[str, Any],
    env: Any,
    mechanism: str,
) -> tuple[list[str], dict[str, Any], bool]:
    if mechanism == "algorithm_direct":
        path = policy.select_path(instance, env)
        return path, {"mechanism": "algorithm_direct", "selection_signal_summary": None}, True

    if mechanism == "theta_guided_agent":
        current_prefix: tuple[str, ...] = ()
        chosen_path: list[str] = []
        choice_alias_history: list[str] = []
        stage_choice_trace: list[dict[str, Any]] = []
        chosen_child_alias_per_stage: list[str] = []
        chosen_child_per_stage_real: list[str] = []
        candidate_count_per_stage: list[int] = []
        candidate_aliases_per_stage: list[list[str]] = []
        candidate_real_children_per_stage: list[list[str]] = []
        fallback_used_per_stage: list[bool] = []
        chooser_raw_output_per_stage: list[list[dict[str, Any]]] = []
        theta_of_chosen_child_per_stage: list[float] = []
        max_theta_per_stage: list[float] = []
        followed_max_theta_per_stage: list[bool] = []

        for stage_name in env.STAGE_NAMES:
            prefix_alias_before_choice = list(choice_alias_history)
            candidates = _build_stage_choice_candidates(policy, env, current_prefix, stage_name)
            llm_result = _run_theta_guided_stage_choice_llm_selector(
                instance,
                env,
                stage_name,
                candidates,
                prefix_alias_before_choice,
            )
            chosen_candidate, choice_meta = _resolve_stage_choice(candidates, llm_result.get("final_output"))
            chosen_child = str(chosen_candidate["child_id"])
            chosen_alias = str(chosen_candidate["option_alias"])
            chosen_path.append(chosen_child)
            current_prefix = tuple(chosen_candidate["child_prefix"])
            choice_alias_history.append(chosen_alias)

            candidate_count = len(candidates)
            candidate_aliases = [str(row["option_alias"]) for row in candidates]
            candidate_real_children = [str(row["child_id"]) for row in candidates]
            chosen_theta = float(chosen_candidate["theta"])
            max_theta = float(choice_meta["max_theta"])
            followed_max = bool(choice_meta["followed_max_theta"])
            fallback_used = bool(choice_meta["fallback_used"])
            raw_output = list(llm_result.get("llm_messages", []))
            chooser_resource_usage = _chooser_resource_usage_from_llm_messages(raw_output)

            chosen_child_alias_per_stage.append(chosen_alias)
            chosen_child_per_stage_real.append(chosen_child)
            candidate_count_per_stage.append(candidate_count)
            candidate_aliases_per_stage.append(candidate_aliases)
            candidate_real_children_per_stage.append(candidate_real_children)
            fallback_used_per_stage.append(fallback_used)
            chooser_raw_output_per_stage.append(raw_output)
            theta_of_chosen_child_per_stage.append(chosen_theta)
            max_theta_per_stage.append(max_theta)
            followed_max_theta_per_stage.append(followed_max)
            stage_choice_trace.append(
                {
                    "stage_name": stage_name,
                    "prefix_alias_history": prefix_alias_before_choice,
                    "candidate_options": [
                        {"option_alias": row["option_alias"], "theta": row["theta"]}
                        for row in candidates
                    ],
                    "chosen_child_alias": chosen_alias,
                    "candidate_count": candidate_count,
                    "fallback_used": fallback_used,
                    "invalid_output": bool(choice_meta["invalid_output"]),
                    "theta_of_chosen_child": chosen_theta,
                    "max_theta": max_theta,
                    "followed_max_theta": followed_max,
                    "chooser_resource_usage": chooser_resource_usage,
                    "chooser_raw_output": raw_output,
                    "rationale": choice_meta.get("rationale"),
                }
            )

        sync_policy_selection(policy, env, chosen_path)
        chooser_resource_metrics = _aggregate_chooser_resource_metrics(
            chooser_raw_output_per_stage
        )
        return chosen_path, {
            "mechanism": "theta_guided_agent",
            "selection_signal_summary": {
                "signal_mode": "raw_theta_log_weight",
                "agent_choice_mode": "stagewise_llm_theta_guided_v1",
                "stage_count": len(stage_choice_trace),
            },
            "agent_llm_raw_output": chooser_raw_output_per_stage,
            "stage_choice_trace": stage_choice_trace,
            "chosen_child_per_stage": chosen_child_alias_per_stage,
            "chosen_child_alias_per_stage": chosen_child_alias_per_stage,
            "chosen_child_per_stage_real": chosen_child_per_stage_real,
            "candidate_count_per_stage": candidate_count_per_stage,
            "candidate_aliases_per_stage": candidate_aliases_per_stage,
            "candidate_real_children_per_stage": candidate_real_children_per_stage,
            "fallback_used_per_stage": fallback_used_per_stage,
            "chooser_raw_output_per_stage": chooser_raw_output_per_stage,
            "theta_of_chosen_child_per_stage": theta_of_chosen_child_per_stage,
            "max_theta_per_stage": max_theta_per_stage,
            "followed_max_theta_per_stage": followed_max_theta_per_stage,
            **chooser_resource_metrics,
        }, True

    if mechanism == "agent_only":
        current_prefix: tuple[str, ...] = ()
        chosen_path: list[str] = []
        choice_alias_history: list[str] = []
        stage_choice_trace: list[dict[str, Any]] = []
        chosen_child_alias_per_stage: list[str] = []
        chosen_child_per_stage_real: list[str] = []
        candidate_count_per_stage: list[int] = []
        candidate_aliases_per_stage: list[list[str]] = []
        candidate_real_children_per_stage: list[list[str]] = []
        fallback_used_per_stage: list[bool] = []
        chooser_raw_output_per_stage: list[list[dict[str, Any]]] = []

        for stage_name in env.STAGE_NAMES:
            prefix_alias_before_choice = list(choice_alias_history)
            candidates = _build_agent_only_stage_candidates(env, current_prefix, stage_name)
            llm_result = _run_agent_only_stage_choice_llm_selector(
                instance,
                env,
                stage_name,
                candidates,
                prefix_alias_before_choice,
            )
            chosen_candidate, choice_meta = _resolve_agent_only_stage_choice(
                candidates,
                llm_result.get("final_output"),
            )
            chosen_child = str(chosen_candidate["child_id"])
            chosen_alias = str(chosen_candidate["option_alias"])
            chosen_path.append(chosen_child)
            current_prefix = tuple(chosen_candidate["child_prefix"])
            choice_alias_history.append(chosen_alias)

            candidate_count = len(candidates)
            candidate_aliases = [str(row["option_alias"]) for row in candidates]
            candidate_real_children = [str(row["child_id"]) for row in candidates]
            fallback_used = bool(choice_meta["fallback_used"])
            raw_output = list(llm_result.get("llm_messages", []))
            chooser_resource_usage = _chooser_resource_usage_from_llm_messages(raw_output)

            chosen_child_alias_per_stage.append(chosen_alias)
            chosen_child_per_stage_real.append(chosen_child)
            candidate_count_per_stage.append(candidate_count)
            candidate_aliases_per_stage.append(candidate_aliases)
            candidate_real_children_per_stage.append(candidate_real_children)
            fallback_used_per_stage.append(fallback_used)
            chooser_raw_output_per_stage.append(raw_output)
            stage_choice_trace.append(
                {
                    "stage_name": stage_name,
                    "prefix_alias_history": prefix_alias_before_choice,
                    "candidate_options": [
                        {"option_alias": row["option_alias"]}
                        for row in candidates
                    ],
                    "chosen_child_alias": chosen_alias,
                    "candidate_count": candidate_count,
                    "fallback_used": fallback_used,
                    "invalid_output": bool(choice_meta["invalid_output"]),
                    "chooser_resource_usage": chooser_resource_usage,
                    "chooser_raw_output": raw_output,
                    "rationale": choice_meta.get("rationale"),
                }
            )

        chooser_resource_metrics = _aggregate_chooser_resource_metrics(
            chooser_raw_output_per_stage
        )
        return chosen_path, {
            "mechanism": "agent_only",
            "selection_signal_summary": {
                "signal_mode": "none",
                "agent_choice_mode": "stagewise_llm_agent_only_v1",
                "stage_count": len(stage_choice_trace),
            },
            "agent_llm_raw_output": chooser_raw_output_per_stage,
            "stage_choice_trace": stage_choice_trace,
            "chosen_child_per_stage": chosen_child_alias_per_stage,
            "chosen_child_alias_per_stage": chosen_child_alias_per_stage,
            "chosen_child_per_stage_real": chosen_child_per_stage_real,
            "candidate_count_per_stage": candidate_count_per_stage,
            "candidate_aliases_per_stage": candidate_aliases_per_stage,
            "candidate_real_children_per_stage": candidate_real_children_per_stage,
            "fallback_used_per_stage": fallback_used_per_stage,
            "chooser_raw_output_per_stage": chooser_raw_output_per_stage,
            **chooser_resource_metrics,
        }, False

    raise ValueError(f"Unknown mechanism: {mechanism}")
