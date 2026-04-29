from __future__ import annotations

import concurrent.futures
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "telecom_mms_agent_profile_only_clean_v3_transfer_control_1x2"
V3_SCRIPT = (
    ROOT
    / "tmp/telecom_mms_agent_profile_only_clean_v3_strict_error_propagation_2x2/run_crosscheck.py"
)
DATA_PATH = (
    ROOT
    / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch/tasks.json"
)
TRANSFER_TASK_ID = (
    "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|"
    "break_app_sms_permission|user_abroad_roaming_disabled_off[PERSONA:Easy]"
)

RUN_SPECS = [
    {
        "run_id": "transfer_expected__all_fast_pure_trap",
        "task_kind": "transfer_expected",
        "task_id": TRANSFER_TASK_ID,
        "path_selector": "first_pure_trap_all_fast",
    },
    {
        "run_id": "transfer_expected__all_deep_pure_target",
        "task_kind": "transfer_expected",
        "task_id": TRANSFER_TASK_ID,
        "path_selector": "first_pure_target",
    },
]


def load_v3_module() -> Any:
    spec = importlib.util.spec_from_file_location("v3_crosscheck", V3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load v3 script: {V3_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stage_output(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stage_row = run.get("stage_outputs", {}).get(stage_name, {})
    if isinstance(stage_row, dict) and isinstance(stage_row.get("output"), dict):
        return stage_row["output"]
    return {}


def stage_trace(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    trace = run.get("stage_trace", {}).get(stage_name, {})
    return trace if isinstance(trace, dict) else {}


def raw_llm_replies(run: dict[str, Any], stage_name: str) -> list[Any]:
    replies = stage_trace(run, stage_name).get("llm_raw_output", []) or []
    return replies if isinstance(replies, list) else []


def summarize_run(run: dict[str, Any], base: Any) -> dict[str, Any]:
    stage3 = stage_output(run, "stage3")
    stage4 = stage_output(run, "stage4")
    stage5 = stage_output(run, "stage5")
    stage3_trace = stage_trace(run, "stage3")
    stage4_trace = stage_trace(run, "stage4")
    stage5_trace = stage_trace(run, "stage5")
    return {
        "run_id": run["run_spec"]["run_id"],
        "task_kind": run["run_spec"]["task_kind"],
        "original_task_id": run["task_metadata"]["original_task_id"],
        "task_metadata": run["task_metadata"]["metadata"],
        "stage_deliberation_requirements_for_offline_reference_only": run[
            "task_metadata"
        ]["stage_deliberation_requirements"],
        "path_class": run["path"]["path_class"],
        "path_rank": run["path"]["rank"],
        "path_match": run["path"]["path_match"],
        "path_agent_modes": run["path"]["path_agent_modes"],
        "path_lane_sequence": run["path"]["path_lane_sequence"],
        "path_route_summary": run["path"]["path_route_summary"],
        **run["summary"],
        "stage_modes": {
            stage_name: run["stage_summaries"][stage_name]["agent_deliberation_mode"]
            for stage_name in base.STAGES
        },
        "stage_tool_counts": {
            stage_name: run["stage_summaries"][stage_name]["tool_call_count"]
            for stage_name in base.STAGES
        },
        "stage3_observed_state": stage3.get("observed_state"),
        "stage3_blocker_ids": [
            item.get("blocker_id")
            for item in stage3.get("per_blocker", [])
            if isinstance(item, dict)
        ],
        "stage3_diagnostic_fallback_used": bool(
            stage3_trace.get("diagnostic_fallback_used")
        ),
        "stage4_repairability": stage4.get("repairability"),
        "stage4_transfer_reason": stage4.get("transfer_reason"),
        "stage4_selected_after_normalization": stage4.get(
            "stage4_selected_after_normalization", []
        ),
        "stage4_deferred_after_normalization": stage4.get(
            "stage4_deferred_after_normalization", []
        ),
        "stage4_decision_valid": stage4.get("stage4_decision_valid"),
        "stage4_error_state": stage4.get("stage4_error_state"),
        "stage4_invalid_reason": stage4.get("stage4_invalid_reason"),
        "stage4_repair_decision_source": stage4.get("stage4_repair_decision_source"),
        "stage4_fallback_penalty": stage4.get("stage4_fallback_penalty"),
        "stage4_tool_names": run["stage_summaries"]["stage4"]["executed_tool_names"],
        "stage4_executed_tool_calls": stage4_trace.get("executed_tool_calls", []),
        "stage4_raw_llm_replies": raw_llm_replies(run, "stage4"),
        "stage5_final_action": stage5.get("terminal_action"),
        "stage5_transfer_reason": stage5.get("transfer_reason"),
        "stage5_selected_blocker_ids": stage5.get("selected_blocker_ids", []),
        "stage5_deferred_blocker_ids": stage5.get("deferred_blocker_ids", []),
        "stage5_verification_fallback_used": bool(
            stage5_trace.get("verification_fallback_used")
        ),
    }


def main() -> None:
    v3 = load_v3_module()
    base = v3.load_base_module()
    base.EXPERIMENT_NAME = EXPERIMENT_NAME
    base.DATA_PATH = DATA_PATH
    base.OUT_DIR = OUT_DIR
    base.RUN_SPECS = RUN_SPECS

    rows_by_task = base.load_rows()
    adapter = base.TelecomMMSTaskAdapter()
    generator = base.TreeFamilyGenerator()
    family_spec, agent_map = generator.build_family(base.FAMILY_KIND, seed=base.SEED)
    validation_errors = generator.validate_family(family_spec, agent_map)
    if validation_errors:
        raise SystemExit(f"Family validation failed: {validation_errors}")
    all_paths = base.enumerate_family_paths(
        stages=list(family_spec.stages),
        stage_agents=family_spec.stage_agents,
        allowed_children=family_spec.allowed_children,
    )

    def run_spec(run_spec: dict[str, Any]) -> dict[str, Any]:
        print(f"[run] {run_spec['run_id']}", flush=True)
        run = base.run_one(
            run_spec,
            rows_by_task,
            family_spec,
            agent_map,
            all_paths,
            adapter,
        )
        run["summary"].update(v3.fallback_penalty_for_run(run))
        base.write_json(OUT_DIR / run_spec["run_id"] / "full_run.json", run)
        print(
            "[done] "
            f"{run_spec['run_id']} modes={run['path']['path_agent_modes']} "
            f"final={run['summary']['final_action']} "
            f"success={run['summary']['success']} "
            f"clean_success={run['summary']['clean_success']} "
            f"raw_total={run['summary']['raw_total_cost']:.6f} "
            f"fallback_total={run['summary']['fallback_penalty_total']:.1f}",
            flush=True,
        )
        return run

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(run_spec, RUN_SPECS))

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment_setting": (
            "telecom_mms_agent_profile_only_clean_v3_strict_error_propagation"
        ),
        "data_path": str(DATA_PATH),
        "family_kind": base.FAMILY_KIND,
        "seed": base.SEED,
        "selected_transfer_task_id": TRANSFER_TASK_ID,
        "runs": [summarize_run(run, base) for run in completed],
    }
    base.write_json(OUT_DIR / "summary.json", summary)
    print(f"[summary] {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
