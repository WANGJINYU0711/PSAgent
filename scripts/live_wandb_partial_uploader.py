#!/usr/bin/env python
"""Live W&B uploader for repeated-smoke partial episode files.

This helper is intentionally independent from the main repeated-smoke runner.
It tails ``<run_dir>/<method>/episodes.partial.jsonl`` while a tmux run is in
progress, backfills any completed episodes, and logs cumulative mean cost curves
to one W&B run per method.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from itertools import product
from pathlib import Path
from typing import Any


UPLOADER_SCHEMA_VERSION = 2
PATH_PROFILE_DEPTH = 5
PATH_PROFILE_ORDER = [
    "".join(bits) for bits in product(("f", "d"), repeat=PATH_PROFILE_DEPTH)
]
PATH_PROFILE_TO_CODE = {
    pattern: index for index, pattern in enumerate(PATH_PROFILE_ORDER)
}


def load_json(path: Path) -> Any:
    with path.open() as f:
        return json.load(f)


def load_partial_episodes(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    episodes: list[dict[str, Any]] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                episodes.append(json.loads(line))
            except json.JSONDecodeError:
                # The runner writes one JSON object per line. If we ever catch a
                # partial write, skip it until the next polling pass.
                continue
    episodes.sort(key=lambda ep: int(ep.get("episode_index", -1)))
    return episodes


def wandb_safe_id(text: str, *, max_len: int = 96) -> str:
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")
    digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]
    if len(slug) <= max_len - len(digest) - 1:
        return f"{slug}-{digest}"
    return f"{slug[: max_len - len(digest) - 1]}-{digest}"


def mode_pattern(ep: dict[str, Any]) -> str:
    modes = ep.get("family_deliberation_modes") or []
    if not isinstance(modes, list):
        return "unknown"
    chars: list[str] = []
    for mode in modes:
        text = str(mode).strip().lower()
        if text.startswith("fast"):
            chars.append("f")
        elif text.startswith("deep"):
            chars.append("d")
        else:
            chars.append("u")
    return "".join(chars) or "unknown"


def path_profile_code(pattern: str) -> int | None:
    if pattern in PATH_PROFILE_TO_CODE:
        return PATH_PROFILE_TO_CODE[pattern]
    if not pattern or any(ch not in {"f", "d"} for ch in pattern):
        return None
    code = 0
    for ch in pattern:
        code = code * 2 + (1 if ch == "d" else 0)
    return code


def path_profile_fields(pattern: str) -> dict[str, Any]:
    code = path_profile_code(pattern)
    fields: dict[str, Any] = {
        "path_profile/pattern": pattern,
        "path_profile/code": code,
        "path_profile/code_0fffff_31ddddd": code,
        "path_profile/code_label": (
            None if code is None else f"{code:02d}:{pattern}"
        ),
        "path_profile/deep_fraction": (
            None if not pattern or pattern == "unknown" else pattern.count("d") / len(pattern)
        ),
    }
    if len(pattern) == PATH_PROFILE_DEPTH and all(ch in {"f", "d"} for ch in pattern):
        for index, ch in enumerate(pattern, start=1):
            fields[f"path_profile/stage{index}_is_deep"] = float(ch == "d")
            fields[f"path_profile/stage{index}_is_fast"] = float(ch == "f")
    return fields


def profile_match_labels(ep: dict[str, Any]) -> tuple[str, str]:
    pair = str(ep.get("terminal_majority_pair") or "")
    if pair == "mostly_deep_vs_mostly_deep_required":
        return "matched", "matched_deep"
    if pair == "mostly_fast_vs_mostly_fast_required":
        return "matched", "matched_fast"
    if pair == "mostly_fast_vs_mostly_deep_required":
        return "mismatched", "fast_on_deep"
    if pair == "mostly_deep_vs_mostly_fast_required":
        return "mismatched", "deep_on_fast"
    return "unknown", "unknown"


def episode_analysis_fields(ep: dict[str, Any]) -> dict[str, Any]:
    pattern = mode_pattern(ep)
    deep_count = pattern.count("d")
    fast_count = pattern.count("f")
    match_group, mismatch_direction = profile_match_labels(ep)
    fdddd_group = "fdddd" if pattern == "fdddd" else "non_fdddd"
    return {
        "terminal_majority_pair": ep.get("terminal_majority_pair"),
        "match_group": match_group,
        "mismatch_direction": mismatch_direction,
        "mode_pattern": pattern,
        "path_mode_pattern": pattern,
        "path_deep_count": deep_count,
        "path_fast_count": fast_count,
        "path_depth_balance": deep_count - fast_count,
        "path_is_fdddd": float(pattern == "fdddd"),
        "fdddd_group": fdddd_group,
    }


def analysis_groups(row: dict[str, Any]) -> list[str]:
    groups: list[str] = []
    for key in ("match_group", "mismatch_direction", "task_bucket", "fdddd_group"):
        value = str(row.get(key) or "unknown")
        if value != "unknown":
            groups.append(value)
    task_bucket = str(row.get("task_bucket") or "unknown")
    match_group = str(row.get("match_group") or "unknown")
    if (
        task_bucket in {"target_favoring", "trap_favoring"}
        and match_group in {"matched", "mismatched"}
    ):
        groups.append(f"{task_bucket}_{match_group}")
    return list(dict.fromkeys(groups))


def add_group_aggregates(rows: list[dict[str, Any]]) -> None:
    group_state: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "raw_total_cost": 0.0,
            "raw_terminal_penalty": 0.0,
            "raw_reasoning_cost_component": 0.0,
            "raw_path_cost_component": 0.0,
            "exact_match": 0.0,
        }
    )
    for idx, row in enumerate(rows, start=1):
        groups = analysis_groups(row)
        row["analysis_groups"] = ",".join(groups)
        for group in groups:
            state = group_state[group]
            state["count"] += 1.0
            state["raw_total_cost"] += float(row.get("raw_total_cost", 0.0) or 0.0)
            state["raw_terminal_penalty"] += float(
                row.get("raw_terminal_penalty", 0.0) or 0.0
            )
            state["raw_reasoning_cost_component"] += float(
                row.get("raw_reasoning_cost_component", 0.0) or 0.0
            )
            state["raw_path_cost_component"] += float(
                row.get("raw_path_cost_component", 0.0) or 0.0
            )
            state["exact_match"] += float(row.get("exact_match", 0.0) or 0.0)
        for group, state in group_state.items():
            count = max(1.0, state["count"])
            prefix = f"groups/{group}"
            row[f"{prefix}/count"] = state["count"]
            row[f"{prefix}/rate"] = state["count"] / idx
            row[f"{prefix}/cumulative_raw_total_cost"] = state["raw_total_cost"]
            row[f"{prefix}/mean_raw_total_cost"] = state["raw_total_cost"] / count
            row[f"{prefix}/cumulative_raw_terminal_penalty"] = state[
                "raw_terminal_penalty"
            ]
            row[f"{prefix}/mean_raw_terminal_penalty"] = (
                state["raw_terminal_penalty"] / count
            )
            row[f"{prefix}/mean_raw_reasoning_cost_component"] = (
                state["raw_reasoning_cost_component"] / count
            )
            row[f"{prefix}/mean_raw_path_cost_component"] = (
                state["raw_path_cost_component"] / count
            )
            row[f"{prefix}/exact_match_rate"] = state["exact_match"] / count


def build_episode_table(rows: list[dict[str, Any]], wandb_module: Any) -> Any:
    columns = [
        "method",
        "episode_1based",
        "task_bucket",
        "match_group",
        "mismatch_direction",
        "mode_pattern",
        "path_profile/code_0fffff_31ddddd",
        "path_profile/code_label",
        "selected_path_compact",
        "selected_leaf_agent",
        "leaf_coverage/unique_leaf_count",
        "leaf_coverage/percent",
        "path_deep_count",
        "path_fast_count",
        "path_depth_balance",
        "fdddd_group",
        "raw_total_cost",
        "raw_terminal_penalty",
        "raw_reasoning_cost_component",
        "chooser_raw_reasoning_cost_component",
        "executor_raw_reasoning_cost_component",
        "combined_with_chooser_raw_total_cost",
        "raw_mode_mismatch_cost_component",
        "root_trap_subtree_prob",
        "stage4_trap_child_prob",
        "all_fast_trap_route_prob",
        "selected_trap_like",
        "exact_match",
        "oracle_action",
        "final_action",
        "original_task_id",
    ]
    table = wandb_module.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(column) for column in columns])
    return table


def build_path_profile_legend_table(wandb_module: Any) -> Any:
    table = wandb_module.Table(columns=["code", "pattern", "deep_count"])
    for pattern, code in PATH_PROFILE_TO_CODE.items():
        table.add_data(code, pattern, pattern.count("d"))
    return table


def build_rows(
    *,
    method: str,
    episodes: list[dict[str, Any]],
    seed: int,
    run_group: str,
    total_episodes: int | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_raw_total = 0.0
    cumulative_total_cost = 0.0
    cumulative_terminal = 0.0
    cumulative_reasoning = 0.0
    cumulative_path = 0.0
    cumulative_exact = 0.0
    post_switch_count = 0
    post_switch_cumulative_raw_total = 0.0
    post_switch_cumulative_total_cost = 0.0
    previous_cumulative_total_per_episode: float | None = None
    previous_cumulative_raw_total_per_episode: float | None = None
    previous_post_total_per_episode: float | None = None
    previous_post_raw_per_episode: float | None = None
    seen_leaf_agents: set[str] = set()
    phase_state: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "count": 0.0,
            "total_growth": 0.0,
            "raw_growth": 0.0,
            "post_total_growth": 0.0,
            "post_raw_growth": 0.0,
        }
    )
    for ep in episodes:
        raw_total = float(ep.get("raw_total_cost", 0.0))
        total_cost = float(ep.get("total_cost", 0.0))
        terminal = float(ep.get("raw_terminal_penalty", 0.0))
        reasoning = float(ep.get("raw_reasoning_cost_component", 0.0))
        path_cost = float(ep.get("raw_path_cost_component", 0.0))
        mode_mismatch = float(ep.get("raw_mode_mismatch_cost_component", 0.0) or 0.0)
        selected_path = ep.get("selected_path") or []
        leaf_agent = ""
        selected_path_compact = ""
        if isinstance(selected_path, list) and selected_path:
            leaf_agent = str(selected_path[-1])
            selected_path_compact = " -> ".join(str(node) for node in selected_path)
            seen_leaf_agents.add(leaf_agent)
        configured_leaf_count = int(ep.get("leaf_coverage_denominator", 8) or 8)
        configured_leaf_count = max(1, configured_leaf_count)
        chooser_reasoning = float(
            ep.get("chooser_raw_reasoning_cost_component", 0.0) or 0.0
        )
        executor_reasoning = ep.get("executor_raw_reasoning_cost_component")
        combined_with_chooser_raw_total = ep.get("combined_with_chooser_raw_total_cost")
        combined_with_chooser_total_cost = ep.get("combined_with_chooser_total_cost")
        exact = float(bool(ep.get("exact_match", False)))
        episode_1based = int(ep["episode_index"]) + 1
        pattern = mode_pattern(ep)

        cumulative_raw_total += raw_total
        cumulative_total_cost += total_cost
        cumulative_terminal += terminal
        cumulative_reasoning += reasoning
        cumulative_path += path_cost
        cumulative_exact += exact

        is_post_switch = ep.get("schedule_phase") == "target_post_switch"
        if is_post_switch:
            post_switch_count += 1
            post_switch_cumulative_raw_total += raw_total
            post_switch_cumulative_total_cost += total_cost
        cumulative_total_per_episode = cumulative_total_cost / episode_1based
        cumulative_raw_total_per_episode = cumulative_raw_total / episode_1based
        post_total_per_episode = (
            post_switch_cumulative_total_cost / post_switch_count
            if post_switch_count
            else None
        )
        post_raw_per_episode = (
            post_switch_cumulative_raw_total / post_switch_count
            if post_switch_count
            else None
        )
        cumulative_total_growth = (
            0.0
            if previous_cumulative_total_per_episode is None
            else cumulative_total_per_episode - previous_cumulative_total_per_episode
        )
        cumulative_raw_growth = (
            0.0
            if previous_cumulative_raw_total_per_episode is None
            else cumulative_raw_total_per_episode - previous_cumulative_raw_total_per_episode
        )
        post_total_growth = (
            None
            if post_total_per_episode is None
            else 0.0
            if previous_post_total_per_episode is None
            else post_total_per_episode - previous_post_total_per_episode
        )
        post_raw_growth = (
            None
            if post_raw_per_episode is None
            else 0.0
            if previous_post_raw_per_episode is None
            else post_raw_per_episode - previous_post_raw_per_episode
        )
        progress_pct = min(
            99,
            int((episode_1based - 1) * 100 / max(1, total_episodes or len(episodes))),
        )
        phase_start = (progress_pct // 20) * 20
        phase_label = f"{phase_start:02d}-{phase_start + 20:02d}%"
        phase = phase_state[phase_label]
        phase["count"] += 1.0
        phase["total_growth"] += cumulative_total_growth
        phase["raw_growth"] += cumulative_raw_growth
        if post_total_growth is not None:
            phase["post_total_growth"] += post_total_growth
        if post_raw_growth is not None:
            phase["post_raw_growth"] += post_raw_growth

        row = {
            "method": method,
            "seed": seed,
            "run_group": run_group,
            "episode_index": int(ep["episode_index"]),
            "episode_1based": episode_1based,
            "repeat_index": ep.get("repeat_index"),
            "position_in_cycle": ep.get("position_in_cycle"),
            "dataset_index": ep.get("dataset_index"),
            "instance_id": ep.get("instance_id"),
            "original_task_id": ep.get("original_task_id"),
            "schedule_phase": ep.get("schedule_phase"),
            "task_bucket": ep.get("task_bucket"),
            "family_behavior_archetype": ep.get("family_behavior_archetype"),
            "family_task_bucket": ep.get("family_task_bucket"),
            "family_trap_like_path": float(bool(ep.get("family_trap_like_path"))),
            "family_target_safe_subtree": float(
                bool(ep.get("family_target_safe_subtree"))
            ),
            "family_decoy_path": float(bool(ep.get("family_decoy_path"))),
            "family_fast_stage_count": ep.get("family_fast_stage_count"),
            "selected_path_compact": selected_path_compact,
            "selected_leaf_agent": leaf_agent,
            "leaf_coverage/unique_leaf_count": len(seen_leaf_agents),
            "leaf_coverage/denominator": configured_leaf_count,
            "leaf_coverage/fraction": len(seen_leaf_agents) / configured_leaf_count,
            "leaf_coverage/percent": 100.0 * len(seen_leaf_agents) / configured_leaf_count,
            "selected_shared_path": float(bool(ep.get("selected_shared_path"))),
            "selected_unshared_path": float(bool(ep.get("selected_unshared_path"))),
            "oracle_action": ep.get("oracle_action"),
            "final_action": ep.get("final_action"),
            "exact_match": exact,
            "subset_mismatch": float(bool(ep.get("subset_mismatch"))),
            "raw_total_cost": raw_total,
            "total_cost": total_cost,
            "raw_terminal_penalty": terminal,
            "raw_reasoning_cost_component": reasoning,
            "raw_path_cost_component": path_cost,
            "raw_mode_mismatch_cost_component": mode_mismatch,
            "chooser_llm_call_count": ep.get("chooser_llm_call_count"),
            "executor_llm_call_count": ep.get("executor_llm_call_count"),
            "chooser_prompt_tokens_total": ep.get("chooser_prompt_tokens_total"),
            "executor_prompt_tokens_total": ep.get("executor_prompt_tokens_total"),
            "chooser_completion_tokens_total": ep.get(
                "chooser_completion_tokens_total"
            ),
            "executor_completion_tokens_total": ep.get(
                "executor_completion_tokens_total"
            ),
            "chooser_total_tokens_total": ep.get("chooser_total_tokens_total"),
            "executor_total_tokens_total": ep.get("executor_total_tokens_total"),
            "chooser_api_cost_total_usd_raw": ep.get(
                "chooser_api_cost_total_usd_raw"
            ),
            "executor_api_cost_total_usd_raw": ep.get(
                "executor_api_cost_total_usd_raw"
            ),
            "chooser_raw_reasoning_cost_component": chooser_reasoning,
            "chooser_raw_reasoning_cost_component_api": ep.get(
                "chooser_raw_reasoning_cost_component_api"
            ),
            "chooser_raw_reasoning_cost_component_token": ep.get(
                "chooser_raw_reasoning_cost_component_token"
            ),
            "executor_raw_reasoning_cost_component": executor_reasoning,
            "combined_with_chooser_llm_call_count": ep.get(
                "combined_with_chooser_llm_call_count"
            ),
            "combined_with_chooser_total_tokens_total": ep.get(
                "combined_with_chooser_total_tokens_total"
            ),
            "combined_with_chooser_api_cost_total_usd_raw": ep.get(
                "combined_with_chooser_api_cost_total_usd_raw"
            ),
            "combined_with_chooser_raw_reasoning_cost_component": ep.get(
                "combined_with_chooser_raw_reasoning_cost_component"
            ),
            "combined_with_chooser_raw_total_cost": (
                combined_with_chooser_raw_total
            ),
            "combined_with_chooser_total_cost": combined_with_chooser_total_cost,
            "mechanism": ep.get("mechanism"),
            "backbone_policy": ep.get("backbone_policy"),
            "max_theta_followed_all_stages": (
                None
                if ep.get("followed_max_theta_per_stage") is None
                else float(all(ep.get("followed_max_theta_per_stage") or []))
            ),
            "fallback_stage_count": len(
                [flag for flag in ep.get("fallback_used_per_stage", []) if flag]
            ),
            "invalid_output_stage_count": len(
                [flag for flag in ep.get("invalid_output_per_stage", []) if flag]
            ),
            "root_trap_subtree_prob": ep.get("root_trap_subtree_prob"),
            "stage4_trap_child_prob": ep.get("stage4_trap_child_prob"),
            "all_fast_trap_route_prob": ep.get("all_fast_trap_route_prob"),
            "selected_trap_like": float(bool(ep.get("family_trap_like_path"))),
            "selected_all_fast_trap": float(
                bool(ep.get("family_trap_like_path"))
                and pattern == "fffff"
            ),
            "cumulative_raw_total": cumulative_raw_total,
            "cumulative_total_cost": cumulative_total_cost,
            "cumulative_raw_total_per_episode": cumulative_raw_total_per_episode,
            "cumulative_total_cost_per_episode": cumulative_total_per_episode,
            "cumulative_total_cost_per_episode_growth": cumulative_total_growth,
            "cumulative_raw_total_per_episode_growth": cumulative_raw_growth,
            "cumulative_terminal_penalty_per_episode": cumulative_terminal
            / episode_1based,
            "cumulative_reasoning_cost_per_episode": cumulative_reasoning
            / episode_1based,
            "cumulative_path_cost_per_episode": cumulative_path / episode_1based,
            "cumulative_exact_match_rate": cumulative_exact / episode_1based,
            "post_switch_episode_count": post_switch_count,
            "post_switch_cumulative_raw_total": post_switch_cumulative_raw_total,
            "post_switch_cumulative_total_cost": post_switch_cumulative_total_cost,
            "post_switch_cumulative_raw_total_per_episode": post_raw_per_episode,
            "post_switch_cumulative_total_cost_per_episode": post_total_per_episode,
            "post_switch_cumulative_raw_total_per_episode_growth": post_raw_growth,
            "post_switch_cumulative_total_cost_per_episode_growth": post_total_growth,
            "progress_phase_20pct": phase_label,
        }
        row.update(path_profile_fields(pattern))
        for label, state in phase_state.items():
            count = max(1.0, state["count"])
            prefix = f"growth_phases/{label}"
            row[f"{prefix}/mean_total_growth"] = state["total_growth"] / count
            row[f"{prefix}/mean_raw_growth"] = state["raw_growth"] / count
            row[f"{prefix}/mean_post_total_growth"] = (
                state["post_total_growth"] / count
            )
            row[f"{prefix}/mean_post_raw_growth"] = state["post_raw_growth"] / count
        previous_cumulative_total_per_episode = cumulative_total_per_episode
        previous_cumulative_raw_total_per_episode = cumulative_raw_total_per_episode
        if post_total_per_episode is not None:
            previous_post_total_per_episode = post_total_per_episode
        if post_raw_per_episode is not None:
            previous_post_raw_per_episode = post_raw_per_episode
        row.update(episode_analysis_fields(ep))
        rows.append(row)
    add_group_aggregates(rows)
    return rows


def method_completed(run_dir: Path, method: str) -> bool:
    summary = run_dir / method / "summary.json"
    if summary.exists():
        return True
    partial = run_dir / method / "summary_partial.json"
    if partial.exists():
        try:
            return load_json(partial).get("status") in {"completed", "failed", "error"}
        except Exception:
            return False
    return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--entity")
    parser.add_argument("--run-group")
    parser.add_argument("--run-name-prefix", default="")
    parser.add_argument("--run-id-suffix", default="")
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--finish-when-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import wandb  # type: ignore

    run_config = load_json(args.run_dir / "run_config.json")
    methods = args.methods or list(run_config["methods"])
    seed = int(run_config.get("seed", 0))
    run_group = args.run_group or args.run_dir.name
    base_config = {
        "dataset": run_config.get("dataset"),
        "family_kind": run_config.get("family_kind"),
        "schedule_mode": run_config.get("schedule_mode"),
        "repeats": run_config.get("repeats"),
        "seed": seed,
        "executor_name": run_config.get("executor_name"),
        "horizon": run_config.get("horizon"),
        "methods": methods,
        "run_dir": str(args.run_dir),
        "live_partial_uploader": True,
    }
    state_suffix = f"_{wandb_safe_id(args.run_id_suffix, max_len=48)}" if args.run_id_suffix else ""
    state_path = args.run_dir / f"live_wandb_uploader_state{state_suffix}.json"
    if state_path.exists():
        try:
            state = load_json(state_path)
        except Exception:
            state = {}
    else:
        state = {}
    last_uploaded: dict[str, int] = {
        method: int(state.get(method, {}).get("last_uploaded_episode_index", -1))
        for method in methods
    }
    if int(state.get("_schema_version", 0) or 0) < UPLOADER_SCHEMA_VERSION:
        last_uploaded = {method: -1 for method in methods}

    while True:
        all_complete = True
        any_uploaded = False
        for method in methods:
            episodes = load_partial_episodes(args.run_dir / method / "episodes.partial.jsonl")
            rows = build_rows(
                method=method,
                episodes=episodes,
                seed=seed,
                run_group=run_group,
                total_episodes=int(run_config.get("horizon") or 0) or None,
            )
            new_rows = [
                row
                for row in rows
                if int(row["episode_index"]) > last_uploaded.get(method, -1)
            ]
            if new_rows:
                run_id = wandb_safe_id(
                    f"{run_group}-live-{args.run_id_suffix}-{method}-seed{seed}"
                )
                name_suffix = f"live_{args.run_id_suffix}_" if args.run_id_suffix else "live_"
                wandb_run = wandb.init(
                    project=args.project,
                    entity=args.entity,
                    group=run_group,
                    id=run_id,
                    resume="allow",
                    name=f"{args.run_name_prefix}{name_suffix}{method}_seed{seed}",
                    reinit=True,
                    config={**base_config, "method": method},
                )
                wandb_run.define_metric("episode_index")
                wandb_run.define_metric("*", step_metric="episode_index")
                wandb_run.define_metric("path_profile/code")
                wandb_run.define_metric(
                    "path_profile/code_0fffff_31ddddd",
                    step_metric="episode_index",
                )
                wandb_run.define_metric("leaf_coverage/unique_leaf_count")
                for row in new_rows:
                    episode_index = int(row["episode_index"])
                    wandb_run.log(row, step=episode_index)
                    last_uploaded[method] = episode_index
                wandb_run.log(
                    {
                        "episode_analysis_table": build_episode_table(rows, wandb),
                        "p_legend": build_path_profile_legend_table(wandb),
                    },
                    step=int(new_rows[-1]["episode_index"]),
                )
                wandb_run.finish()
                any_uploaded = True
                state_path.write_text(
                    json.dumps(
                        {
                            "_schema_version": UPLOADER_SCHEMA_VERSION,
                            **{
                                m: {"last_uploaded_episode_index": idx}
                                for m, idx in sorted(last_uploaded.items())
                            },
                        },
                        indent=2,
                        sort_keys=True,
                    )
                    + "\n"
                )
            all_complete = all_complete and method_completed(args.run_dir, method)
        if args.finish_when_complete and all_complete:
            break
        if not any_uploaded:
            time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
