from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
for extra in (
    ROOT,
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
    ROOT / "baselines",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from adapters.telecom_mms_adapter import TelecomMMSTaskAdapter  # noqa: E402
from fixed_tree_env import FixedTreeEnvironment  # noqa: E402
from oracle_eval import enumerate_family_paths  # noqa: E402
from scripts.run_llm_path_sweep_diagnostic import (  # noqa: E402
    aggregate_path_resource_summary,
    build_offline_path_record,
    build_stage_requirements,
    flatten_stage_resource_summary,
    sort_offline_records,
)
from tree_family.generator import TreeFamilyGenerator  # noqa: E402


EXPERIMENT_NAME = "telecom_mms_agent_profile_only_clean_v2_fast_deep_crosscheck_2x2"
DATA_PATH = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_nontransfer_smoke/tasks.json"
)
FAMILY_KIND = "shared_basin_strong_prefix_dedup_profile_switch"
SEED = 0
STAGES = ["stage1", "stage2", "stage3", "stage4", "stage5"]
OUT_DIR = Path(__file__).resolve().parent

FAST_TASK_ID = "[mms_issue]airplane_mode_on|bad_wifi_calling|break_app_storage_permission[PERSONA:Easy]"
DEEP_TASK_ID = (
    "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|"
    "break_apn_mms_setting|break_app_storage_permission|unseat_sim_card[PERSONA:Easy]"
)

RUN_SPECS = [
    {
        "run_id": "fast_requirement__fast_trap_path",
        "task_kind": "fast_requirement",
        "task_id": FAST_TASK_ID,
        "path_selector": "first_pure_trap_all_fast",
    },
    {
        "run_id": "fast_requirement__deep_target_path",
        "task_kind": "fast_requirement",
        "task_id": FAST_TASK_ID,
        "path_selector": "first_pure_target",
    },
    {
        "run_id": "deep_requirement__fast_trap_path",
        "task_kind": "deep_requirement",
        "task_id": DEEP_TASK_ID,
        "path_selector": "first_pure_trap_all_fast",
    },
    {
        "run_id": "deep_requirement__deep_target_path",
        "task_kind": "deep_requirement",
        "task_id": DEEP_TASK_ID,
        "path_selector": "first_pure_target",
    },
]


def json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_rows() -> dict[str, dict[str, Any]]:
    rows = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return {
        str(row.get("original_task_id", row.get("instance_id", "unknown"))): row
        for row in rows
    }


def path_modes(path_agent_ids: list[str], agent_map: dict[str, Any]) -> list[str]:
    return [
        str(getattr(agent_map[agent_id], "deliberation_mode", "unknown"))
        for agent_id in path_agent_ids
    ]


def select_path(
    rankings: list[dict[str, Any]],
    selector: str,
    agent_map: dict[str, Any],
) -> dict[str, Any]:
    for row in rankings:
        modes = path_modes(list(row["path_agent_ids"]), agent_map)
        if selector == "first_pure_target" and row["path_class"] == "pure_target":
            return row
        if (
            selector == "first_pure_trap_all_fast"
            and row["path_class"] == "pure_trap"
            and all(mode == "fast" for mode in modes)
        ):
            return row
    raise RuntimeError(f"No path matched selector={selector}")


def capture_prompt_builders(executor: Any, captured_prompts: dict[str, Any]) -> None:
    for stage_name in STAGES:
        method_name = f"_build_{stage_name}_prompts"
        original = getattr(executor, method_name)

        def wrapper(*args: Any, _original: Any = original, _stage_name: str = stage_name, **kwargs: Any):
            system_prompt, user_prompt = _original(*args, **kwargs)
            captured_prompts[_stage_name] = {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
            return system_prompt, user_prompt

        setattr(executor, method_name, wrapper)


def compact_stage_trace(episode_log: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(stage_row["stage_name"]): stage_row
        for stage_row in episode_log.get("stage_trace", [])
        if isinstance(stage_row, dict) and stage_row.get("stage_name")
    }


def summarize_stage(stage_name: str, trace: dict[str, Any]) -> dict[str, Any]:
    llm_raw = trace.get("llm_raw_output", []) or []
    return {
        "stage_name": stage_name,
        "agent_id": trace.get("agent_id"),
        "agent_deliberation_mode": trace.get("agent_deliberation_mode"),
        "trace_stage_requirement_diagnostic": trace.get("stage_requirement"),
        "llm_call_count": trace.get("llm_call_count_stage"),
        "valid_json_first_try": trace.get("valid_json_first_try"),
        "json_retry_count": trace.get("json_retry_count"),
        "fallback_used": trace.get("fallback_used"),
        "tool_call_count": len(trace.get("executed_tool_calls", []) or []),
        "executed_tool_names": [
            call.get("name")
            for call in trace.get("executed_tool_calls", []) or []
            if isinstance(call, dict)
        ],
        "prompt_tokens": trace.get("prompt_tokens_stage"),
        "completion_tokens": trace.get("completion_tokens_stage"),
        "total_tokens": trace.get("total_tokens_stage"),
        "first_llm_reply": llm_raw[0] if llm_raw else None,
        "final_output": trace.get("output"),
    }


def run_one(
    run_spec: dict[str, Any],
    rows_by_task: dict[str, dict[str, Any]],
    family_spec: Any,
    agent_map: dict[str, Any],
    all_paths: list[tuple[str, ...]],
    adapter: TelecomMMSTaskAdapter,
) -> dict[str, Any]:
    task_id = run_spec["task_id"]
    row = rows_by_task[task_id]
    stage_requirements = build_stage_requirements(row, adapter)
    rankings = sort_offline_records(
        [
            build_offline_path_record(
                task_id=task_id,
                path=path,
                stage_requirements=stage_requirements,
                agent_map=agent_map,
                weakening_level=0,
            )
            for path in all_paths
        ]
    )
    path_row = select_path(rankings, run_spec["path_selector"], agent_map)
    selected_path = list(path_row["path_agent_ids"])

    env = FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=FAMILY_KIND,
        family_seed=SEED,
        executor_name="llm_bench",
    )
    env.reset(row)
    captured_prompts: dict[str, Any] = {}
    capture_prompt_builders(env.family_executor, captured_prompts)
    result = env.run_path(selected_path)
    episode_log = result.episode_log or {}
    stage_trace = compact_stage_trace(episode_log)
    stage_outputs = result.stage_outputs
    stage_resource_summary = flatten_stage_resource_summary(stage_trace)
    path_resource_summary = aggregate_path_resource_summary(stage_resource_summary)

    stage4_output = (
        stage_outputs.get("stage4", {}).get("output", {})
        if isinstance(stage_outputs.get("stage4"), dict)
        else {}
    )
    stage5_output = (
        stage_outputs.get("stage5", {}).get("output", {})
        if isinstance(stage_outputs.get("stage5"), dict)
        else {}
    )
    raw_total_cost_with_token_penalty = (
        float(result.raw_total_cost)
        + float(path_resource_summary.get("fast_token_over_budget_penalty_total", 0.0))
    )
    full_run = {
        "experiment_name": EXPERIMENT_NAME,
        "run_spec": run_spec,
        "task_metadata": {
            "task_id": row.get("instance_id"),
            "original_task_id": task_id,
            "metadata": row.get("metadata", {}),
            "stage_deliberation_requirements": stage_requirements["deliberation"],
        },
        "path": {
            "rank": path_row["rank"],
            "path_class": path_row["path_class"],
            "path_match": path_row["path_match"],
            "path_agent_ids": selected_path,
            "path_agent_modes": path_modes(selected_path, agent_map),
            "path_lane_sequence": path_row["path_lane_sequence"],
            "path_base_cost_sum": path_row["path_base_cost_sum"],
            "path_route_summary": path_row["path_route_summary"],
        },
        "summary": {
            "final_action": result.final_action,
            "oracle_action": result.oracle_action,
            "success": bool(result.success),
            "raw_terminal_penalty": float(result.raw_terminal_penalty),
            "raw_path_cost_component": float(result.raw_path_cost_component),
            "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
            "raw_total_cost": float(result.raw_total_cost),
            "raw_total_cost_with_token_penalty": raw_total_cost_with_token_penalty,
            "prompt_tokens_total": float(result.prompt_tokens_total),
            "completion_tokens_total": float(result.completion_tokens_total),
            "total_tokens_total": float(result.total_tokens_total),
            "api_cost_total_usd_raw": float(result.api_cost_total_usd_raw),
            "selected_blocker_ids": stage5_output.get("selected_blocker_ids", []),
            "deferred_blocker_ids": stage5_output.get("deferred_blocker_ids", []),
            "stage4_repairability": stage4_output.get("repairability"),
            "stage4_selected_after_normalization": stage4_output.get(
                "stage4_selected_after_normalization", []
            ),
            "stage4_deferred_after_normalization": stage4_output.get(
                "stage4_deferred_after_normalization", []
            ),
            "stage4_completion_pass_applied": bool(
                stage4_output.get("stage4_completion_pass_applied", False)
            ),
            "stage4_completion_added_blockers": stage4_output.get(
                "stage4_completion_added_blockers", []
            ),
            **path_resource_summary,
        },
        "stage_summaries": {
            stage_name: summarize_stage(stage_name, stage_trace.get(stage_name, {}))
            for stage_name in STAGES
        },
        "captured_prompts": captured_prompts,
        "stage_outputs": stage_outputs,
        "stage_trace": stage_trace,
        "episode_log": episode_log,
    }
    run_dir = OUT_DIR / run_spec["run_id"]
    write_json(run_dir / "full_run.json", full_run)
    return full_run


def main() -> None:
    rows_by_task = load_rows()
    adapter = TelecomMMSTaskAdapter()
    generator = TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(FAMILY_KIND, seed=SEED)
    validation_errors = generator.validate_family(family_spec, agent_map)
    if validation_errors:
        raise SystemExit(f"Family validation failed: {validation_errors}")
    all_paths = enumerate_family_paths(
        stages=list(family_spec.stages),
        stage_agents=family_spec.stage_agents,
        allowed_children=family_spec.allowed_children,
    )

    completed: list[dict[str, Any]] = []
    for run_spec in RUN_SPECS:
        print(f"[run] {run_spec['run_id']}", flush=True)
        full_run = run_one(
            run_spec,
            rows_by_task,
            family_spec,
            agent_map,
            all_paths,
            adapter,
        )
        completed.append(full_run)
        print(
            "[done] "
            f"{run_spec['run_id']} final={full_run['summary']['final_action']} "
            f"success={full_run['summary']['success']} "
            f"raw_total_cost={full_run['summary']['raw_total_cost']:.6f}",
            flush=True,
        )

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "data_path": str(DATA_PATH),
        "family_kind": FAMILY_KIND,
        "seed": SEED,
        "runs": [
            {
                "run_id": run["run_spec"]["run_id"],
                "task_kind": run["run_spec"]["task_kind"],
                "original_task_id": run["task_metadata"]["original_task_id"],
                "stage_deliberation_requirements": run["task_metadata"][
                    "stage_deliberation_requirements"
                ],
                "path_class": run["path"]["path_class"],
                "path_rank": run["path"]["rank"],
                "path_match": run["path"]["path_match"],
                "path_agent_modes": run["path"]["path_agent_modes"],
                "path_lane_sequence": run["path"]["path_lane_sequence"],
                **run["summary"],
                "stage_modes": {
                    stage_name: run["stage_summaries"][stage_name][
                        "agent_deliberation_mode"
                    ]
                    for stage_name in STAGES
                },
                "stage_tool_counts": {
                    stage_name: run["stage_summaries"][stage_name]["tool_call_count"]
                    for stage_name in STAGES
                },
                "stage4_tool_names": run["stage_summaries"]["stage4"][
                    "executed_tool_names"
                ],
                "stage4_raw_llm_replies": run["stage_trace"]
                .get("stage4", {})
                .get("llm_raw_output", []),
            }
            for run in completed
        ],
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(f"[summary] {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
