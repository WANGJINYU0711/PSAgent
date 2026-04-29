from __future__ import annotations

import argparse
import concurrent.futures
import importlib
import importlib.util
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
EXPERIMENT_NAME = "telecom_mms_agent_profile_only_transfer_policy_ablation_v3_1_v3_2"
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

VARIANTS = {
    "v3_1_prompt_only_transfer_policy": {
        "experiment_setting": "telecom_mms_agent_profile_only_clean_v3_1_prompt_only_transfer_policy",
        "normalizer": "unchanged_from_v3",
    },
    "v3_2_prompt_plus_safety_normalizer": {
        "experiment_setting": "telecom_mms_agent_profile_only_clean_v3_2_prompt_plus_safety_normalizer",
        "normalizer": "hard_hybrid_transfer_guard",
    },
}

PATH_SPECS = {
    "all_fast_pure_trap": {
        "run_id": "all_fast_pure_trap",
        "path_selector": "first_pure_trap_all_fast",
    },
    "all_deep_pure_target": {
        "run_id": "all_deep_pure_target",
        "path_selector": "first_pure_target",
    },
}


STAGE4_HARD_TRANSFER_CONTRACT = (
    "\nHard transfer contract:\n"
    "- First classify each blocker independently as local_repairable, hard_transfer_required, or ordinary_defer.\n"
    "- A blocker is local_repairable only when it is active in stage3_output, Stage 3 evidence supports it, canonical local repair steps exist, and neither hybrid_required nor non-deferable assistant_side_required is true.\n"
    "- A blocker is hard_transfer_required when it is active and its repair requires hybrid handling, non-deferable assistant-side handling, external/manual handling, or tools not available in this stage.\n"
    "- Do not choose transfer_required merely because a generic depends_on list is long, or because an inactive blocker has a hybrid-looking spec.\n"
    "- If all active unresolved blockers are local_repairable, execute the supported local repair bundle and use repairable or partially_repairable as appropriate.\n"
    "- If at least one active hard_transfer_required blocker remains unresolved, case-level repairability must be transfer_required, even if some local blockers were safely repaired.\n"
    "- Do not label the case partially_repairable merely because local repairs were executed when a hard_transfer_required blocker still blocks MMS success.\n"
    "- If transfer_required is selected, set transfer_reason to hard_hybrid_blocker_requires_transfer_v1 or another short snake_case hard-blocker reason.\n"
)

STAGE5_HARD_TRANSFER_CONTRACT = (
    "\nHard terminal transfer contract:\n"
    "- If stage4_output.repairability is transfer_required, final_action must be transfer.\n"
    "- If stage4_output.transfer_reason names an unresolved hard hybrid/nonlocal blocker, final_action must be transfer.\n"
    "- Do not downgrade transfer_required to repair_subset because some local repair tools succeeded.\n"
    "- repair_subset is valid only when remaining deferred blockers are ordinary defers, not active hard_transfer_required blockers.\n"
)


def json_ready(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_v3_module() -> Any:
    spec = importlib.util.spec_from_file_location("v3_crosscheck", V3_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load v3 script: {V3_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_base_module() -> Any:
    v3 = load_v3_module()
    base = v3.load_base_module()
    base.DATA_PATH = DATA_PATH
    base.OUT_DIR = OUT_DIR
    return base


def _patch_stage4_prompt(cls: type[Any]) -> None:
    original = cls._build_stage4_prompts

    def patched(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        system_prompt, user_prompt = original(self, *args, **kwargs)
        system_prompt = system_prompt.replace(
            "Stage4 policy bias:\n"
            "- Do not choose transfer_required for the whole case only because one blocker is hybrid-required.\n"
            "- If at least one blocker has affirmative evidence and a safe local repair path, mark that blocker should_repair=true and use partially_repairable when other blockers must remain deferred.\n"
            "- If a blocker is assistant-side-required but can be safely deferred, it may be marked should_repair=false under partially_repairable.\n"
            "- Use transfer_required only when no safe local repair subset exists or the remaining blockers explicitly require external/manual handling.\n",
            "Stage4 policy bias:\n"
            "- Separate local repair selection from case-level terminal repairability.\n"
            "- Do not choose transfer_required merely because an inactive or safely deferrable blocker has a hybrid-looking spec.\n"
            "- Use partially_repairable only when the remaining deferred blockers are ordinary defers that do not still require hard transfer handling.\n"
            "- Use transfer_required when an active hard hybrid/nonlocal blocker remains unresolved and still blocks MMS success.\n",
        )
        system_prompt += STAGE4_HARD_TRANSFER_CONTRACT
        data = json.loads(user_prompt)
        data["stage4_hard_transfer_contract"] = {
            "local_repairable_when_all_true": [
                "blocker appears in stage3_output.per_blocker",
                "stage3_output.observed_state contains affirmative evidence for this blocker family",
                "canonical local repair steps are available",
                "hybrid_required is false",
                "non_deferable_assistant_side_required is false",
                "repair can be completed by allowed Stage 4 tools",
            ],
            "hard_transfer_required_when_any_true": [
                "active blocker has hybrid_required=true and remains unresolved",
                "active blocker has assistant_side_required=true and can_be_deferred=false",
                "active blocker requires external/manual/account-side handling not safely completed by allowed Stage 4 tools",
                "active blocker canonical repair chain requires a missing or disallowed tool",
                "post-local-repair success still depends on this unresolved hard blocker",
            ],
            "ordinary_defer_when_any_true": [
                "evidence is missing or contradictory",
                "blocker is not active in stage3_output",
                "blocker is safely deferrable and does not block MMS success",
                "repair is optional or low-confidence under the selected agent profile",
            ],
            "case_level_repairability_rules": [
                "repairable only when all active blockers are locally repaired",
                "partially_repairable only when remaining deferred blockers are ordinary defers, not hard_transfer_required blockers",
                "transfer_required when any active hard_transfer_required blocker remains unresolved",
            ],
        }
        data["stage4_local_repair_decision_table"] = data[
            "stage4_hard_transfer_contract"
        ]
        normalization_rules = [
            rule
            for rule in data.get("normalization_rules", [])
            if rule
            not in {
                "Use partially_repairable when at least one blocker is supported for local repair and at least one blocker remains deferred",
                "Do not output transfer_required for the whole case if a supported local repair subset exists",
            }
        ]
        normalization_rules.extend(
            [
                "Use partially_repairable only when remaining deferred blockers are ordinary defers, not hard_transfer_required blockers",
                "Use transfer_required when any active hard_transfer_required blocker remains unresolved",
                "Do not downgrade a hard transfer case to partially_repairable because local repair tools succeeded",
            ]
        )
        data["normalization_rules"] = normalization_rules
        return system_prompt, json.dumps(data, ensure_ascii=False)

    cls._build_stage4_prompts = patched


def _patch_stage5_prompt(cls: type[Any]) -> None:
    original = cls._build_stage5_prompts

    def patched(self: Any, *args: Any, **kwargs: Any) -> tuple[str, str]:
        system_prompt, user_prompt = original(self, *args, **kwargs)
        system_prompt += STAGE5_HARD_TRANSFER_CONTRACT
        data = json.loads(user_prompt)
        data["stage5_hard_transfer_contract"] = {
            "rules": [
                "If stage4_output.repairability is transfer_required, final_action must be transfer",
                "If stage4_output.transfer_reason names an unresolved hard hybrid/nonlocal blocker, final_action must be transfer",
                "Do not downgrade transfer_required to repair_subset because some local repair tools succeeded",
                "repair_subset is valid only when remaining deferred blockers are ordinary defers, not active hard_transfer_required blockers",
            ]
        }
        normalization_rules = [
            rule
            for rule in data.get("normalization_rules", [])
            if rule
            != "Deferred blockers can prevent repair_all, but if stage4_output already supports a local subset they are not by themselves enough to force transfer; prefer repair_subset unless transfer has a hard reason"
        ]
        normalization_rules.extend(
            [
                "If stage4_output.repairability is transfer_required, final_action must be transfer",
                "If stage4_output.transfer_reason names an unresolved hard hybrid/nonlocal blocker, final_action must be transfer",
                "Do not downgrade transfer_required to repair_subset because some local repair tools succeeded",
                "repair_subset is valid only when remaining deferred blockers are ordinary defers, not active hard_transfer_required blockers",
            ]
        )
        data["normalization_rules"] = normalization_rules
        return system_prompt, json.dumps(data, ensure_ascii=False)

    cls._build_stage5_prompts = patched


def _patch_hard_transfer_normalizer(cls: type[Any]) -> None:
    original = cls._normalized_stage4_plan

    def patched(self: Any, *args: Any, **kwargs: Any) -> tuple[list[dict[str, Any]], str, str | None, dict[str, Any]]:
        normalized_rows, repairability, transfer_reason, diagnostics = original(
            self, *args, **kwargs
        )
        hard_blockers = [
            row.get("blocker_id")
            for row in normalized_rows
            if row.get("blocker_id")
            and self._is_nonlocal_or_hybrid_transfer_blocker(str(row.get("blocker_id")))
            and not bool(row.get("should_repair"))
        ]
        if not hard_blockers:
            diagnostics = dict(diagnostics)
            diagnostics["hard_transfer_guard_applied"] = False
            diagnostics["hard_transfer_guard_blockers"] = []
            return normalized_rows, repairability, transfer_reason, diagnostics

        reason = "hard_hybrid_blocker_requires_transfer_v1"
        coerced_rows = deepcopy(normalized_rows)
        self._coerce_stage4_rows_to_transfer_required(
            coerced_rows,
            refusal_code=reason,
        )
        deferred_blocker_ids = [
            row.get("blocker_id") for row in coerced_rows if row.get("blocker_id")
        ]
        diagnostics = dict(diagnostics)
        diagnostics.update(
            {
                "selected_after_normalization": [],
                "deferred_after_normalization": deferred_blocker_ids,
                "normalizer_changed_output": True,
                "completion_blocked_by_hard_transfer_guard": list(
                    dict.fromkeys(
                        list(diagnostics.get("completion_blocked_by_hard_transfer_guard", []))
                        + hard_blockers
                    )
                ),
                "hard_transfer_guard_applied": True,
                "hard_transfer_guard_blockers": list(dict.fromkeys(hard_blockers)),
                "hard_transfer_guard_reason": reason,
            }
        )
        return coerced_rows, "transfer_required", reason, diagnostics

    cls._normalized_stage4_plan = patched


def apply_variant_patches(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    executor_module = importlib.import_module("executors.telecom_llm_bench_executor")
    cls = executor_module.TelecomLLMBenchExecutor
    setting = VARIANTS[variant]["experiment_setting"]
    executor_module.AGENT_PROFILE_ONLY_EXPERIMENT_SETTING = setting

    original_init = cls.__init__

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        original_init(self, *args, **kwargs)
        self.experiment_setting = setting

    cls.__init__ = patched_init
    cls._strict_error_propagation_enabled = (
        lambda self: self.experiment_setting
        in {
            "telecom_mms_agent_profile_only_clean_v3_strict_error_propagation",
            VARIANTS["v3_1_prompt_only_transfer_policy"]["experiment_setting"],
            VARIANTS["v3_2_prompt_plus_safety_normalizer"]["experiment_setting"],
        }
    )
    _patch_stage4_prompt(cls)
    _patch_stage5_prompt(cls)
    if variant == "v3_2_prompt_plus_safety_normalizer":
        _patch_hard_transfer_normalizer(cls)


def stage_output(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    stage_row = run.get("stage_outputs", {}).get(stage_name, {})
    if isinstance(stage_row, dict) and isinstance(stage_row.get("output"), dict):
        return stage_row["output"]
    return {}


def stage_trace(run: dict[str, Any], stage_name: str) -> dict[str, Any]:
    trace = run.get("stage_trace", {}).get(stage_name, {})
    return trace if isinstance(trace, dict) else {}


def summarize_run(run: dict[str, Any], base: Any, variant: str) -> dict[str, Any]:
    stage3 = stage_output(run, "stage3")
    stage4 = stage_output(run, "stage4")
    stage5 = stage_output(run, "stage5")
    stage3_trace = stage_trace(run, "stage3")
    stage4_trace = stage_trace(run, "stage4")
    stage5_trace = stage_trace(run, "stage5")
    return {
        "variant": variant,
        "variant_experiment_setting": VARIANTS[variant]["experiment_setting"],
        "variant_normalizer": VARIANTS[variant]["normalizer"],
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
        "stage4_completion_blocked_by_hard_transfer_guard": stage4.get(
            "stage4_completion_blocked_by_hard_transfer_guard", []
        ),
        "stage4_tool_names": run["stage_summaries"]["stage4"]["executed_tool_names"],
        "stage4_executed_tool_calls": stage4_trace.get("executed_tool_calls", []),
        "stage4_raw_json_extracted": stage4.get("stage4_raw_json_extracted"),
        "stage4_raw_llm_replies": stage4_trace.get("llm_raw_output", []) or [],
        "stage5_final_action": stage5.get("final_action"),
        "stage5_transfer_reason": stage5.get("transfer_reason"),
        "stage5_selected_blocker_ids": stage5.get("selected_blocker_ids", []),
        "stage5_deferred_blocker_ids": stage5.get("deferred_blocker_ids", []),
        "stage5_verification_fallback_used": bool(
            stage5_trace.get("verification_fallback_used")
        ),
    }


def run_single(variant: str, path_key: str) -> None:
    base = load_base_module()
    apply_variant_patches(variant)
    v3 = load_v3_module()

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

    path_spec = PATH_SPECS[path_key]
    run_id = f"{variant}__{path_spec['run_id']}"
    run_spec = {
        "run_id": run_id,
        "task_kind": "transfer_expected",
        "task_id": TRANSFER_TASK_ID,
        "path_selector": path_spec["path_selector"],
    }
    print(f"[run] {run_id}", flush=True)
    run = base.run_one(
        run_spec,
        rows_by_task,
        family_spec,
        agent_map,
        all_paths,
        adapter,
    )
    run["summary"].update(v3.fallback_penalty_for_run(run))
    write_json(OUT_DIR / run_id / "full_run.json", run)
    write_json(OUT_DIR / run_id / "summary_row.json", summarize_run(run, base, variant))
    print(
        "[done] "
        f"{run_id} final={run['summary']['final_action']} "
        f"oracle={run['summary']['oracle_action']} "
        f"success={run['summary']['success']} "
        f"clean={run['summary']['clean_success']} "
        f"raw={run['summary']['raw_total_cost']:.6f} "
        f"fallback={run['summary']['fallback_penalty_total']:.1f}",
        flush=True,
    )


def aggregate() -> None:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for path_key in PATH_SPECS:
            run_id = f"{variant}__{PATH_SPECS[path_key]['run_id']}"
            p = OUT_DIR / run_id / "summary_row.json"
            if not p.exists():
                raise FileNotFoundError(p)
            rows.append(json.loads(p.read_text(encoding="utf-8")))
    summary = {
        "experiment_name": EXPERIMENT_NAME,
        "base_experiment_setting": "telecom_mms_agent_profile_only_clean_v3_strict_error_propagation",
        "data_path": str(DATA_PATH),
        "selected_transfer_task_id": TRANSFER_TASK_ID,
        "variants": VARIANTS,
        "runs": rows,
    }
    write_json(OUT_DIR / "summary.json", summary)
    print(f"[summary] {OUT_DIR / 'summary.json'}", flush=True)


def run_all() -> None:
    commands = [
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--single",
            "--variant",
            variant,
            "--path-key",
            path_key,
        ]
        for variant in VARIANTS
        for path_key in PATH_SPECS
    ]

    def run_command(cmd: list[str]) -> tuple[int, str]:
        proc = subprocess.run(
            cmd,
            cwd=ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        return proc.returncode, proc.stdout

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(run_command, cmd) for cmd in commands]
        for future in concurrent.futures.as_completed(futures):
            code, output = future.result()
            print(output, end="", flush=True)
            if code != 0:
                raise SystemExit(code)
    aggregate()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single", action="store_true")
    parser.add_argument("--variant", choices=sorted(VARIANTS))
    parser.add_argument("--path-key", choices=sorted(PATH_SPECS))
    args = parser.parse_args()
    if args.single:
        if not args.variant or not args.path_key:
            raise SystemExit("--single requires --variant and --path-key")
        run_single(args.variant, args.path_key)
    else:
        run_all()


if __name__ == "__main__":
    main()
