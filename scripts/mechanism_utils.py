"""Mechanism helpers for fixed-tree baseline runners.

This module provides a thin runner-level abstraction over three interaction
modes:

- ``algorithm_direct``: policy selects the final path directly
- ``theta_guided_agent``: policy exposes path preferences, then an agent-side
  chooser selects among top-scoring candidates
- ``agent_only``: the runner provides candidate paths, and an LLM agent chooses
  one without seeing any algorithm signal
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from oracle_eval import enumerate_all_paths


ROOT = Path(__file__).resolve().parents[1]
TAU2_ROOT = ROOT / "tau2-bench"
TAU2_VENV_PYTHON = TAU2_ROOT / ".venv" / "bin" / "python"
AGENT_ONLY_BRIDGE = Path(__file__).with_name("_agent_only_path_bridge.py")
DEFAULT_AGENT_ONLY_MODEL = os.environ.get("PSAGENT_AGENT_ONLY_MODEL", "gpt-4.1-2025-04-14")


def _agent_path_heuristic(env: Any, path: list[str]) -> float:
    score = 0.0
    leaf_type = env.compute_leaf_type(path)
    if leaf_type == "shared":
        score += 1.0
    for agent_id in path:
        if "_unstable_" in agent_id:
            score -= 0.3
        elif "_stable_" in agent_id:
            score += 0.3
        if "_high_" in agent_id:
            score += 0.15
    return score


def _extract_algorithm_path_scores(policy: Any, env: Any) -> tuple[dict[tuple[str, ...], float], str]:
    paths = enumerate_all_paths(env)

    if hasattr(policy, "leaf_weights") and isinstance(getattr(policy, "leaf_weights"), dict):
        weights = getattr(policy, "leaf_weights")
        return {tuple(path): float(weights.get(tuple(path), 0.0)) for path in paths}, "leaf_weight"

    if hasattr(policy, "theta") and isinstance(getattr(policy, "theta"), dict):
        theta = getattr(policy, "theta")
        return {tuple(path): float(theta.get(tuple(path), 0.0)) for path in paths}, "leaf_theta"

    if hasattr(policy, "weights") and hasattr(policy, "stage_agent_ids"):
        weights = getattr(policy, "weights")
        scores: dict[tuple[str, ...], float] = {}
        for path in paths:
            score = 1.0
            for agent_id in path:
                score *= float(weights.get(agent_id, 1.0))
            scores[tuple(path)] = score
        return scores, "stage_product_weight"

    return {tuple(path): 1.0 for path in paths}, "uniform"


def _top_path_summary(
    score_map: dict[tuple[str, ...], float],
    env: Any,
    limit: int = 5,
) -> list[dict[str, Any]]:
    ranked = sorted(
        score_map.items(),
        key=lambda item: (-item[1], list(item[0])),
    )[:limit]
    return [
        {
            "path": list(path),
            "algorithm_score": float(score),
            "agent_heuristic": _agent_path_heuristic(env, list(path)),
            "leaf_type": env.compute_leaf_type(list(path)),
        }
        for path, score in ranked
    ]


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
        mass_rows = policy._edge_masses(current_prefix, child_prefixes)
        combined_masses = [row[2] for row in mass_rows]
        num_children = len(child_prefixes)
        if sum(combined_masses) <= 0:
            exploit_probs = [1.0 / num_children for _ in child_prefixes]
        else:
            exploit_probs = [mass / sum(combined_masses) for mass in combined_masses]
        local_epsilon = 0.0 if policy.safe_prefixes.get(current_prefix, False) else policy.epsilon
        probs = [
            (1.0 - local_epsilon) * exploit_prob + local_epsilon * (1.0 / num_children)
            for exploit_prob in exploit_probs
        ]
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
    path_prob = 1.0
    for stage_name, agent_id in zip(env.STAGE_NAMES, path):
        agent_ids = policy.stage_agent_ids[stage_name]
        probs = policy._stage_probs(agent_ids)
        idx = agent_ids.index(agent_id)
        prob = probs[idx]
        policy.last_path_probs.append(prob)
        policy.last_stage_probs[stage_name] = prob
        path_prob *= prob
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


def _path_agent_summary(env: Any, path: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for stage_name, agent_id in zip(env.STAGE_NAMES, path):
        spec = env.agent_catalog[agent_id]
        rows.append(
            {
                "stage": stage_name,
                "agent_id": agent_id,
                "competence": getattr(spec, "competence_level", None),
                "scope": getattr(spec, "scope_level", None),
                "stability": getattr(spec, "stability_level", None),
                "group": getattr(spec, "g", None),
            }
        )
    return rows


def _evenly_spaced_indexes(length: int, count: int) -> list[int]:
    if count <= 0 or length <= 0:
        return []
    if count >= length:
        return list(range(length))
    if count == 1:
        return [0]
    return sorted(
        {
            round(i * (length - 1) / (count - 1))
            for i in range(count)
        }
    )


def _build_agent_only_candidates(env: Any, max_candidates: int = 16) -> list[dict[str, Any]]:
    all_paths = sorted(enumerate_all_paths(env))
    if len(all_paths) <= max_candidates:
        chosen_paths = all_paths
    else:
        grouped: dict[str, list[list[str]]] = {"shared": [], "unshared": []}
        for path in all_paths:
            grouped[env.compute_leaf_type(path)].append(path)

        chosen_paths: list[list[str]] = []
        seen: set[tuple[str, ...]] = set()
        base_quota = max(1, max_candidates // max(1, sum(bool(v) for v in grouped.values())))
        for leaf_type in ("shared", "unshared"):
            bucket = grouped[leaf_type]
            for idx in _evenly_spaced_indexes(len(bucket), min(base_quota, len(bucket))):
                path = bucket[idx]
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    chosen_paths.append(path)
        if len(chosen_paths) < max_candidates:
            for idx in _evenly_spaced_indexes(len(all_paths), max_candidates):
                path = all_paths[idx]
                key = tuple(path)
                if key not in seen:
                    seen.add(key)
                    chosen_paths.append(path)
                if len(chosen_paths) >= max_candidates:
                    break

    return [
        {
            "path_id": f"p{i+1:02d}",
            "path": list(path),
            "leaf_type": env.compute_leaf_type(path),
            "agents": _path_agent_summary(env, path),
        }
        for i, path in enumerate(chosen_paths)
    ]


def _trim_text(text: str | None, max_chars: int = 700) -> str:
    if not text:
        return ""
    text = " ".join(text.split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def _build_agent_only_prompts(instance: dict[str, Any], env: Any, candidates: list[dict[str, Any]]) -> tuple[str, str]:
    stage1_input = instance.get("stage1", {}).get("input", {})
    task_summary = {
        "reason_for_call": _trim_text(stage1_input.get("reason_for_call")),
        "known_info": _trim_text(stage1_input.get("known_info")),
        "family_kind": getattr(env, "family_kind", None),
        "stage_names": list(env.STAGE_NAMES),
        "candidate_count": len(candidates),
    }
    system_prompt = (
        "You are choosing one fixed workflow path for a telecom troubleshooting task. "
        "You must choose exactly one candidate path using only the task context and the candidate "
        "agent profiles. Do not assume access to any algorithm scores, theta values, or oracle labels. "
        "Prefer the candidate path that is most likely to solve the task with low total cost. "
        "Return only JSON with keys path_id and optional rationale."
    )
    user_prompt = json.dumps(
        {
            "task_summary": task_summary,
            "candidate_paths": candidates,
            "instructions": {
                "must_choose_one_path_id": True,
                "forbidden_signals": ["algorithm_score", "theta", "baseline_preference", "oracle_action"],
                "output_schema": {"path_id": "string", "rationale": "optional short string"},
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    return system_prompt, user_prompt


def _run_agent_only_llm_selector(
    instance: dict[str, Any],
    env: Any,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    system_prompt, user_prompt = _build_agent_only_prompts(instance, env, candidates)
    payload = {
        "model": DEFAULT_AGENT_ONLY_MODEL,
        "llm_args": {"temperature": 0.0},
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
    }
    proc = subprocess.run(
        [str(TAU2_VENV_PYTHON), str(AGENT_ONLY_BRIDGE)],
        input=json.dumps(payload, ensure_ascii=False),
        capture_output=True,
        text=True,
        check=False,
        cwd=str(TAU2_ROOT),
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "agent_only LLM bridge failed: "
            + (proc.stderr.strip() or proc.stdout.strip() or f"exit={proc.returncode}")
        )
    return json.loads(proc.stdout)


def _resolve_agent_only_choice(candidates: list[dict[str, Any]], final_output: dict[str, Any] | None) -> tuple[list[str], dict[str, Any]]:
    candidate_by_id = {row["path_id"]: row for row in candidates}
    if isinstance(final_output, dict):
        path_id = final_output.get("path_id")
        if isinstance(path_id, str) and path_id in candidate_by_id:
            row = candidate_by_id[path_id]
            return list(row["path"]), {
                "chosen_path_id": path_id,
                "rationale": final_output.get("rationale"),
                "fallback_used": False,
            }
        path = final_output.get("path")
        if isinstance(path, list):
            normalized = [str(x) for x in path]
            for row in candidates:
                if row["path"] == normalized:
                    return list(normalized), {
                        "chosen_path_id": row["path_id"],
                        "rationale": final_output.get("rationale"),
                        "fallback_used": False,
                    }
    fallback = candidates[0]
    return list(fallback["path"]), {
        "chosen_path_id": fallback["path_id"],
        "rationale": "fallback_first_candidate_invalid_llm_output",
        "fallback_used": True,
    }


def choose_path_with_mechanism(
    policy: Any,
    instance: dict[str, Any],
    env: Any,
    mechanism: str,
) -> tuple[list[str], dict[str, Any], bool]:
    if mechanism == "algorithm_direct":
        path = policy.select_path(instance, env)
        return path, {"selection_signal_summary": None}, True

    all_paths = enumerate_all_paths(env)

    if mechanism == "theta_guided_agent":
        score_map, signal_mode = _extract_algorithm_path_scores(policy, env)
        top_candidates = sorted(
            all_paths,
            key=lambda path: (-score_map.get(tuple(path), 0.0), path),
        )[:5]
        chosen_path = max(
            top_candidates,
            key=lambda path: (
                _agent_path_heuristic(env, path),
                score_map.get(tuple(path), 0.0),
                tuple(path),
            ),
        )
        sync_policy_selection(policy, env, chosen_path)
        return chosen_path, {
            "selection_signal_summary": {
                "signal_mode": signal_mode,
                "top_candidates": _top_path_summary(score_map, env),
                "agent_choice_mode": "heuristic_topk_argmax_v1",
            }
        }, True

    if mechanism == "agent_only":
        candidates = _build_agent_only_candidates(env)
        llm_result = _run_agent_only_llm_selector(instance, env, candidates)
        chosen_path, choice_meta = _resolve_agent_only_choice(candidates, llm_result.get("final_output"))
        return chosen_path, {
            "selection_signal_summary": {
                "signal_mode": "none",
                "agent_choice_mode": "llm_self_select_v1",
                "candidate_count": len(candidates),
                "candidate_ids": [row["path_id"] for row in candidates],
                "chosen_path_id": choice_meta["chosen_path_id"],
                "fallback_used": choice_meta["fallback_used"],
                "rationale": choice_meta.get("rationale"),
                "leaf_type": env.compute_leaf_type(chosen_path),
            },
            "agent_llm_raw_output": llm_result.get("llm_messages", []),
        }, False

    raise ValueError(f"Unknown mechanism: {mechanism}")
