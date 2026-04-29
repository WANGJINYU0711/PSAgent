from __future__ import annotations

import concurrent.futures
import importlib.util
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "telecom_mms_agent_profile_only_clean_v3_strict_error_propagation_2x2"
BASE_SCRIPT = (
    ROOT
    / "tmp/telecom_mms_agent_profile_only_clean_v2_fast_deep_crosscheck_2x2/run_crosscheck.py"
)


def load_base_module() -> Any:
    spec = importlib.util.spec_from_file_location("v2_crosscheck_base", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load base script: {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.EXPERIMENT_NAME = EXPERIMENT_NAME
    module.OUT_DIR = OUT_DIR
    return module


def fallback_penalty_for_run(run: dict[str, Any]) -> dict[str, Any]:
    stage_trace = run.get("stage_trace", {}) or {}
    stage_outputs = run.get("stage_outputs", {}) or {}
    stage3 = stage_trace.get("stage3", {}) or {}
    stage4_output = (
        stage_outputs.get("stage4", {}).get("output", {})
        if isinstance(stage_outputs.get("stage4"), dict)
        else {}
    )
    stage5 = stage_trace.get("stage5", {}) or {}
    penalty_breakdown = {
        "stage3_diagnostic_fallback_penalty": 20.0
        if stage3.get("diagnostic_fallback_used")
        else 0.0,
        "stage3_account_fallback_penalty": 5.0
        if stage3.get("account_side_fallback_used")
        else 0.0,
        "stage4_fallback_penalty": float(stage4_output.get("stage4_fallback_penalty", 0.0) or 0.0),
        "stage5_verification_fallback_penalty": 10.0
        if stage5.get("verification_fallback_used")
        else 0.0,
    }
    total = round(sum(penalty_breakdown.values()), 6)
    disqualified = bool(
        stage3.get("diagnostic_fallback_used")
        or stage3.get("account_side_fallback_used")
        or stage4_output.get("stage4_decision_valid") is False
        or stage4_output.get("stage4_executor_completed_plan")
        or stage4_output.get("stage4_normalizer_changed_output")
    )
    return {
        "fallback_penalty_breakdown": penalty_breakdown,
        "fallback_penalty_total": total,
        "raw_total_cost_with_fallback_penalty": round(
            float(run["summary"]["raw_total_cost"]) + total,
            6,
        ),
        "clean_success": bool(run["summary"]["success"]) and not disqualified,
        "disqualified_by_fallback_or_normalizer": disqualified,
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
        penalty = fallback_penalty_for_run(run)
        run["summary"].update(penalty)
        base.write_json(OUT_DIR / run_spec["run_id"] / "full_run.json", run)
        print(
            "[done] "
            f"{run_spec['run_id']} final={run['summary']['final_action']} "
            f"success={run['summary']['success']} "
            f"clean_success={run['summary']['clean_success']} "
            f"raw_total={run['summary']['raw_total_cost']:.6f} "
            f"fallback_total={run['summary']['fallback_penalty_total']:.1f}",
            flush=True,
        )
        return run

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        completed = list(pool.map(run_spec, base.RUN_SPECS))

    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment_setting": "telecom_mms_agent_profile_only_clean_v3_strict_error_propagation",
        "data_path": str(base.DATA_PATH),
        "family_kind": base.FAMILY_KIND,
        "seed": base.SEED,
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
                    for stage_name in base.STAGES
                },
                "stage_tool_counts": {
                    stage_name: run["stage_summaries"][stage_name]["tool_call_count"]
                    for stage_name in base.STAGES
                },
                "stage4_tool_names": run["stage_summaries"]["stage4"][
                    "executed_tool_names"
                ],
                "stage4_decision_valid": run["stage_outputs"]["stage4"]["output"].get(
                    "stage4_decision_valid"
                ),
                "stage4_error_state": run["stage_outputs"]["stage4"]["output"].get(
                    "stage4_error_state"
                ),
                "stage4_invalid_reason": run["stage_outputs"]["stage4"]["output"].get(
                    "stage4_invalid_reason"
                ),
                "stage4_raw_llm_replies": run["stage_trace"]
                .get("stage4", {})
                .get("llm_raw_output", []),
            }
            for run in completed
        ],
    }
    base.write_json(OUT_DIR / "summary.json", summary)
    print(f"[summary] {OUT_DIR / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
