from __future__ import annotations

import concurrent.futures
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "telecom_mms_agent_profile_only_clean_v4_hard_transfer_contract_transfer_control_1x2"
BASE_SCRIPT = (
    ROOT
    / "tmp/telecom_mms_agent_profile_only_clean_v2_fast_deep_crosscheck_2x2/run_crosscheck.py"
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


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v2_crosscheck_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPERIMENT_NAME = EXPERIMENT_NAME
    module.DATA_PATH = DATA_PATH
    module.OUT_DIR = OUT_DIR
    module.RUN_SPECS = RUN_SPECS
    return module


def stage_output(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stage_row = run.get("stage_outputs", {}).get(stage_name, {})
    if isinstance(stage_row, dict) and isinstance(stage_row.get("output"), dict):
        return stage_row["output"]
    return {}


def stage_trace(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    trace = run.get("stage_trace", {}).get(stage_name, {})
    return trace if isinstance(trace, dict) else {}


def fallback_and_clean_metrics(run: dict[str, Any]) -> dict[str, Any]:
    trace = run.get("stage_trace", {}) or {}
    stage3 = trace.get("stage3", {}) or {}
    stage4 = stage_output(run, "stage4")
    stage5 = trace.get("stage5", {}) or {}
    penalty_breakdown = {
        "stage3_diagnostic_fallback_penalty": 20.0
        if stage3.get("diagnostic_fallback_used")
        else 0.0,
        "stage3_account_fallback_penalty": 5.0
        if stage3.get("account_side_fallback_used")
        else 0.0,
        "stage4_fallback_penalty": float(stage4.get("stage4_fallback_penalty", 0.0) or 0.0),
        "stage5_verification_fallback_penalty": 10.0
        if stage5.get("verification_fallback_used")
        else 0.0,
    }
    total = round(sum(penalty_breakdown.values()), 6)
    fallback_or_completion_dirty = bool(
        stage3.get("diagnostic_fallback_used")
        or stage3.get("account_side_fallback_used")
        or stage4.get("stage4_decision_valid") is False
        or stage4.get("stage4_executor_completed_plan")
    )
    safety_guard_applied = bool(stage4.get("hard_transfer_guard_applied"))
    return {
        "fallback_penalty_breakdown": penalty_breakdown,
        "fallback_penalty_total": total,
        "raw_total_cost_with_fallback_penalty": round(
            float(run["summary"]["raw_total_cost"]) + total,
            6,
        ),
        "fallback_or_completion_dirty": fallback_or_completion_dirty,
        "safety_guard_applied": safety_guard_applied,
        "safety_guard_reason": stage4.get("hard_transfer_guard_reason"),
        "safety_guard_blockers": stage4.get("hard_transfer_guard_blockers", []),
        "clean_success_no_fallback": bool(run["summary"]["success"])
        and not fallback_or_completion_dirty,
        "oracle_leak_free_success": bool(run["summary"]["success"])
        and not fallback_or_completion_dirty,
        "legacy_clean_success": bool(run["summary"]["success"])
        and not fallback_or_completion_dirty
        and not bool(stage4.get("stage4_normalizer_changed_output")),
    }


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
        "stage4_normalizer_changed_output": stage4.get("stage4_normalizer_changed_output"),
        "stage4_safety_normalizer_changed_output": stage4.get(
            "stage4_safety_normalizer_changed_output"
        ),
        "hard_transfer_guard_applied": stage4.get("hard_transfer_guard_applied"),
        "hard_transfer_guard_blockers": stage4.get("hard_transfer_guard_blockers", []),
        "hard_transfer_guard_reason": stage4.get("hard_transfer_guard_reason"),
        "stage4_completion_blocked_by_hard_transfer_guard": stage4.get(
            "stage4_completion_blocked_by_hard_transfer_guard", []
        ),
        "stage4_tool_names": run["stage_summaries"]["stage4"]["executed_tool_names"],
        "stage4_executed_tool_calls": stage4_trace.get("executed_tool_calls", []),
        "stage4_raw_json_extracted": stage4.get("stage4_raw_json_extracted"),
        "stage4_raw_llm_replies": stage4_trace.get("llm_raw_output", []) or [],
        "stage5_final_action": stage5.get("final_action"),
        "stage5_terminal_decision_source": stage5.get("terminal_decision_source"),
        "stage5_transfer_reason": stage5.get("transfer_reason"),
        "stage5_selected_blocker_ids": stage5.get("selected_blocker_ids", []),
        "stage5_deferred_blocker_ids": stage5.get("deferred_blocker_ids", []),
        "stage5_verification_fallback_used": bool(
            stage5_trace.get("verification_fallback_used")
        ),
    }


def main() -> None:
    base = load_base_module()
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
        run["summary"].update(fallback_and_clean_metrics(run))
        base.write_json(OUT_DIR / run_spec["run_id"] / "full_run.json", run)
        print(
            "[done] "
            f"{run_spec['run_id']} modes={run['path']['path_agent_modes']} "
            f"final={run['summary']['final_action']} "
            f"success={run['summary']['success']} "
            f"clean_no_fallback={run['summary']['clean_success_no_fallback']} "
            f"safety_guard={run['summary']['safety_guard_applied']} "
            f"raw_total={run['summary']['raw_total_cost']:.6f}",
            flush=True,
        )
        return run

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(run_spec, RUN_SPECS))

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment_setting": "telecom_mms_agent_profile_only_clean_v4_hard_transfer_contract",
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
