#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "tmp/llm_v8_seed1_old_vs_probfloor_targeted_diagnostic"

OLD_RUN = ROOT / "tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost"
PF_RUN = ROOT / "tmp/llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_probfloor0002"
TASKS = ROOT / "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json"

FOCUS_DATASETS = {16, 13, 10, 2}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_csv_dict(path: Path, method: str | None = None) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if method is not None and row.get("method") != method:
                continue
            rows[int(row["episode_index"])] = row
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not fieldnames:
        keys: list[str] = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    keys.append(key)
                    seen.add(key)
        fieldnames = keys
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def ffloat(x: Any, default: float = 0.0) -> float:
    if x in (None, ""):
        return default
    return float(x)


def bbool(x: Any) -> bool:
    if isinstance(x, bool):
        return x
    return str(x).lower() == "true"


def mode_pattern(mode_string: str) -> str:
    parts = [p.strip().lower() for p in (mode_string or "").split("/") if p.strip()]
    return "".join("d" if p == "deep" else "f" if p == "fast" else "?" for p in parts)


def pair_label(actual_modes: str, required_modes: str) -> str:
    return f"{mode_pattern(actual_modes)} on {mode_pattern(required_modes)}"


def list_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "|".join(str(v) for v in value)
    return str(value)


def load_episode_bundle(run: Path, label: str) -> dict[int, dict[str, Any]]:
    episodes = {int(r["episode_index"]): r for r in read_json(run / "risky_ps/episodes.json")}
    mode_rows = read_csv_dict(run / "episode_mode_cost_analysis.csv", method="risky_ps")
    for ep, row in episodes.items():
        m = mode_rows.get(ep, {})
        row["_label"] = label
        row["_actual_modes"] = m.get("actual_modes", "/".join(row.get("family_deliberation_modes", [])))
        row["_required_modes"] = m.get("required_modes", "")
        row["_pattern"] = pair_label(row["_actual_modes"], row["_required_modes"])
        row["_majority_pair"] = m.get("majority_pair", row.get("terminal_majority_pair", ""))
        row["_strict_clean_success"] = bbool(m.get("strict_clean_success", row.get("exact_match", False)))
        row["_clear_success_proxy"] = bbool(m.get("clear_success_proxy", row.get("terminal_clear_success_proxy", False)))
        row["_auxiliary_success_proxy"] = bbool(m.get("auxiliary_success_proxy", row.get("terminal_auxiliary_success_proxy", False)))
        row["_route_labels"] = m.get("route_labels", " > ".join(row.get("family_route_labels", [])))
        row["_selected_shared_path"] = bbool(m.get("selected_shared_path", row.get("selected_shared_path", False)))
        row["_selected_unshared_path"] = bbool(m.get("selected_unshared_path", row.get("selected_unshared_path", False)))
    return episodes


def task_oracle(task: dict[str, Any]) -> dict[str, Any]:
    ec = task.get("source_task", {}).get("evaluation_criteria", {})
    stage4 = task.get("stage4", {}).get("oracle_output", {})
    stage5 = task.get("stage5", {}).get("oracle_output", {})
    blocker_to_tools: dict[str, list[str]] = {}
    blocker_to_reason: dict[str, str] = {}
    blocker_to_decision: dict[str, str] = {}
    blocker_to_deps: dict[str, list[str]] = {}
    for item in stage4.get("per_blocker", []):
        bid = item.get("blocker_id")
        if not bid:
            continue
        blocker_to_tools[bid] = [step.get("tool_name", "") for step in item.get("canonical_repair_steps", []) if step.get("tool_name")]
        blocker_to_reason[bid] = item.get("adjudication_label") or item.get("refusal_code") or ""
        blocker_to_decision[bid] = item.get("oracle_execute_decision") or ""
        blocker_to_deps[bid] = item.get("depends_on", [])
    return {
        "original_task_id": task.get("original_task_id", ""),
        "expected_terminal_action": task.get("metadata", {}).get("expected_terminal_action") or ec.get("expected_terminal_action"),
        "repairability": task.get("metadata", {}).get("repairability"),
        "selected_blockers": ec.get("selected_blocker_ids") or stage5.get("selected_blocker_ids", []),
        "deferred_blockers": ec.get("deferred_blocker_ids") or stage5.get("deferred_blocker_ids", []),
        "transfer_reason": ec.get("transfer_reason") or stage5.get("transfer_reason"),
        "oracle_actions": [a.get("name", "") for a in ec.get("actions", []) if a.get("name")],
        "stage4_blocker_reasons": blocker_to_reason,
        "stage4_blocker_decisions": blocker_to_decision,
        "stage4_blocker_deps": blocker_to_deps,
        "blocker_to_tools": blocker_to_tools,
        "stage5_success_condition": stage5.get("verification_plan", {}).get("success_condition") or ec.get("success_mode"),
        "stage5_required_postchecks": stage5.get("verification_plan", {}).get("required_postchecks", []),
    }


def missing_oracle_tools(row: dict[str, Any], oracle: dict[str, Any]) -> tuple[list[str], list[str]]:
    replay = set(row.get("stage5_replay_tool_names") or [])
    required = set()
    missing_blockers = []
    for blocker in oracle["selected_blockers"]:
        tools = oracle["blocker_to_tools"].get(blocker, [])
        if not tools:
            continue
        required.update(tools)
        if any(t not in replay for t in tools):
            missing_blockers.append(blocker)
    missing_tools = sorted(required - replay)
    return sorted(set(missing_blockers)), missing_tools


def transition_type(row: dict[str, Any]) -> str:
    oracle = row.get("oracle_action")
    final = row.get("final_action")
    if oracle == final:
        return "matched_terminal_action"
    if oracle == "repair_all" and final == "repair_subset":
        return "repair_all_to_repair_subset"
    if oracle in {"repair_all", "repair_subset"} and final == "transfer":
        return "local_to_transfer"
    if oracle == "repair_subset" and final == "repair_all":
        return "repair_subset_to_repair_all"
    return f"{oracle}_to_{final}"


def cause_label(row: dict[str, Any], oracle: dict[str, Any]) -> str:
    reasons = row.get("terminal_adjustment_reasons") or []
    final = row.get("final_action")
    oracle_action = row.get("oracle_action")
    missing_blockers, missing_tools = missing_oracle_tools(row, oracle)
    if oracle_action in {"repair_all", "repair_subset"} and final == "transfer":
        return "over_transfer_on_local_oracle"
    if "subset_mismatch_base_plus_linear" in reasons:
        if missing_blockers:
            return "subset_mismatch_with_missing_oracle_repair_tools"
        return "subset_mismatch_terminal_decision"
    if "local_clear_and_aux_failure_floor_12" in reasons:
        return "local_clear_and_aux_failure"
    if "local_clear_failure_floor_10" in reasons:
        return "local_clear_failure"
    if row.get("policy_action_violation"):
        return "policy_action_violation"
    if missing_tools:
        return "missing_oracle_repair_tools_without_subset_floor"
    if ffloat(row.get("raw_terminal_penalty")) >= 10:
        return "high_terminal_uncategorized_flattened_trace"
    return "no_terminal_failure_or_low_penalty"


def compact_row(prefix: str, row: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    missing_blockers, missing_tools = missing_oracle_tools(row, oracle)
    return {
        f"{prefix}_final": row.get("final_action"),
        f"{prefix}_terminal": ffloat(row.get("raw_terminal_penalty")),
        f"{prefix}_total": ffloat(row.get("raw_total_cost")),
        f"{prefix}_reasoning": ffloat(row.get("raw_reasoning_cost_component")),
        f"{prefix}_modecost_report": ffloat(row.get("raw_mode_mismatch_cost_component")),
        f"{prefix}_pair": row.get("_majority_pair"),
        f"{prefix}_pattern": row.get("_pattern"),
        f"{prefix}_actual_modes": mode_pattern(row.get("_actual_modes", "")),
        f"{prefix}_required_modes": mode_pattern(row.get("_required_modes", "")),
        f"{prefix}_clear": row.get("_clear_success_proxy"),
        f"{prefix}_aux": row.get("_auxiliary_success_proxy"),
        f"{prefix}_strict": row.get("_strict_clean_success"),
        f"{prefix}_shared": row.get("_selected_shared_path"),
        f"{prefix}_path_prob": row.get("selection_path_prob"),
        f"{prefix}_stage_probs": json.dumps(row.get("selection_stage_probs", {}), sort_keys=True),
        f"{prefix}_route": row.get("_route_labels"),
        f"{prefix}_path": " > ".join(row.get("selected_path") or []),
        f"{prefix}_suffix": " > ".join((row.get("selected_path") or [])[-2:]),
        f"{prefix}_replay_tools": list_str(row.get("stage5_replay_tool_names")),
        f"{prefix}_executed_tools": list_str(row.get("stage5_executed_tool_names")),
        f"{prefix}_missing_blockers": "|".join(missing_blockers),
        f"{prefix}_missing_tools": "|".join(missing_tools),
        f"{prefix}_terminal_reasons": list_str(row.get("terminal_adjustment_reasons")),
        f"{prefix}_transition": transition_type(row),
        f"{prefix}_cause": cause_label(row, oracle),
    }


def avg(rows: list[dict[str, Any]], key: str) -> float:
    vals = [ffloat(r.get(key)) for r in rows]
    return mean(vals) if vals else 0.0


def rate(rows: list[dict[str, Any]], key: str) -> float:
    return mean([1.0 if r.get(key) else 0.0 for r in rows]) if rows else 0.0


def summarize_group(rows: list[dict[str, Any]], run_label: str, group_key: str) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row[group_key])].append(row)
    out = []
    for value, grp in sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        out.append({
            "run": run_label,
            group_key: value,
            "n": len(grp),
            "terminal": avg(grp, "raw_terminal_penalty"),
            "legacy_terminal": avg(grp, "legacy_raw_terminal_penalty"),
            "reasoning": avg(grp, "raw_reasoning_cost_component"),
            "modecost_report": avg(grp, "raw_mode_mismatch_cost_component"),
            "total": avg(grp, "raw_total_cost"),
            "clear": rate(grp, "_clear_success_proxy"),
            "aux": rate(grp, "_auxiliary_success_proxy"),
            "strict": rate(grp, "_strict_clean_success"),
            "transfer_final_n": sum(1 for r in grp if r.get("final_action") == "transfer"),
            "rtp_ge_10_n": sum(1 for r in grp if ffloat(r.get("raw_terminal_penalty")) >= 10),
            "rtp_ge_14_n": sum(1 for r in grp if ffloat(r.get("raw_terminal_penalty")) >= 14),
            "examples": ",".join(str(r["episode_index"]) for r in grp[:8]),
        })
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], max_rows: int | None = None) -> str:
    rows = rows[:max_rows] if max_rows is not None else rows
    if not rows:
        return "_无记录。_"
    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return f"{v:.3f}"
        s = str(v)
        return s.replace("\n", " ").replace("|", "\\|")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    old = load_episode_bundle(OLD_RUN, "old_seed1_risky_ps")
    pf = load_episode_bundle(PF_RUN, "seed1_probfloor0002")
    tasks = read_json(TASKS)
    oracles = {i: task_oracle(task) for i, task in enumerate(tasks)}

    aligned = []
    for ep in sorted(old):
        o = old[ep]
        p = pf[ep]
        ds = int(o["dataset_index"])
        oracle = oracles[ds]
        assert int(p["dataset_index"]) == ds, (ep, ds, p["dataset_index"])
        row = {
            "episode_index": ep,
            "repeat_index": o.get("repeat_index"),
            "dataset_index": ds,
            "schedule_phase": o.get("family_schedule_phase") or o.get("schedule_phase"),
            "oracle_action": o.get("oracle_action"),
            "expected_terminal_action": oracle["expected_terminal_action"],
            "repairability": oracle["repairability"],
            "original_task_id": o.get("original_task_id"),
            "oracle_selected_blockers": "|".join(oracle["selected_blockers"]),
            "oracle_deferred_blockers": "|".join(oracle["deferred_blockers"]),
            "oracle_actions": "|".join(oracle["oracle_actions"]),
            "stage5_success_condition": oracle["stage5_success_condition"],
            "stage5_required_postchecks": "|".join(oracle["stage5_required_postchecks"]),
        }
        row.update(compact_row("old", o, oracle))
        row.update(compact_row("pf", p, oracle))
        row["delta_terminal_pf_minus_old"] = row["pf_terminal"] - row["old_terminal"]
        row["delta_total_pf_minus_old"] = row["pf_total"] - row["old_total"]
        row["terminal_improved"] = row["delta_terminal_pf_minus_old"] < 0
        row["terminal_worse"] = row["delta_terminal_pf_minus_old"] > 0
        row["pair_change"] = f"{row['old_pattern']} -> {row['pf_pattern']}"
        aligned.append(row)

    write_csv(OUT / "aligned_old_vs_probfloor.csv", aligned)

    post = [r for r in aligned if r["schedule_phase"] == "target_post_switch"]
    improved_fast_on_deep = [
        r for r in post
        if r["pf_pair"] == "mostly_fast_vs_mostly_deep_required" and r["delta_terminal_pf_minus_old"] < 0
    ]
    improved_fast_on_deep.sort(key=lambda r: (r["delta_terminal_pf_minus_old"], r["episode_index"]))
    write_csv(OUT / "probfloor_improved_fast_on_deep_episodes.csv", improved_fast_on_deep)

    all_probfloor_fast_on_deep = [
        r for r in post
        if r["pf_pair"] == "mostly_fast_vs_mostly_deep_required"
    ]
    all_probfloor_fast_on_deep.sort(key=lambda r: (ffloat(r["pf_terminal"]), r["episode_index"]))
    write_csv(OUT / "probfloor_all_fast_on_deep_episodes.csv", all_probfloor_fast_on_deep)

    deep_on_deep_worse = [
        r for r in post
        if r["pf_pair"] == "mostly_deep_vs_mostly_deep_required" and r["delta_terminal_pf_minus_old"] > 0
    ]
    deep_on_deep_worse.sort(key=lambda r: (-r["delta_terminal_pf_minus_old"], r["episode_index"]))
    write_csv(OUT / "probfloor_deep_on_deep_terminal_worse_episodes.csv", deep_on_deep_worse)

    focus = [r for r in aligned if r["dataset_index"] in FOCUS_DATASETS]
    write_csv(OUT / "focus_dataset_episodes_16_13_10_2.csv", focus)

    focus_summary = []
    focus_oracle_rows = []
    for ds in sorted(FOCUS_DATASETS):
        grp = [r for r in focus if r["dataset_index"] == ds]
        if not grp:
            continue
        oracle = oracles[ds]
        focus_oracle_rows.append({
            "dataset_index": ds,
            "oracle_action": grp[0]["oracle_action"],
            "repairability": oracle["repairability"],
            "selected": "|".join(oracle["selected_blockers"]),
            "deferred": "|".join(oracle["deferred_blockers"]),
            "oracle_actions": "|".join(oracle["oracle_actions"]),
            "stage5": f"final={oracle['expected_terminal_action']}; success={oracle['stage5_success_condition']}; postchecks={','.join(oracle['stage5_required_postchecks'])}; transfer={oracle['transfer_reason']}",
            "stage4_repair_reasons": "; ".join(
                f"{b}:{oracle['stage4_blocker_reasons'].get(b)} deps={','.join(oracle['stage4_blocker_deps'].get(b, []))}"
                for b in oracle["selected_blockers"]
            ),
            "stage4_defer_reasons": "; ".join(
                f"{b}:{oracle['stage4_blocker_reasons'].get(b)} deps={','.join(oracle['stage4_blocker_deps'].get(b, []))}"
                for b in oracle["deferred_blockers"]
            ),
        })
        focus_summary.append({
            "dataset_index": ds,
            "n": len(grp),
            "oracle_action": grp[0]["oracle_action"],
            "repairability": oracle["repairability"],
            "selected_blockers": "|".join(oracle["selected_blockers"]),
            "deferred_blockers": "|".join(oracle["deferred_blockers"]),
            "oracle_actions": "|".join(oracle["oracle_actions"]),
            "stage4_reasons": "; ".join(f"{b}:{oracle['stage4_blocker_decisions'].get(b)}/{oracle['stage4_blocker_reasons'].get(b)} deps={','.join(oracle['stage4_blocker_deps'].get(b, []))}" for b in oracle["selected_blockers"] + oracle["deferred_blockers"]),
            "stage5_oracle": f"final={oracle['expected_terminal_action']}; success={oracle['stage5_success_condition']}; postchecks={','.join(oracle['stage5_required_postchecks'])}; transfer_reason={oracle['transfer_reason']}",
            "old_terminal_mean": mean(r["old_terminal"] for r in grp),
            "pf_terminal_mean": mean(r["pf_terminal"] for r in grp),
            "delta_terminal": mean(r["delta_terminal_pf_minus_old"] for r in grp),
            "old_final_counts": dict(Counter(r["old_final"] for r in grp)),
            "pf_final_counts": dict(Counter(r["pf_final"] for r in grp)),
            "old_pattern_counts": dict(Counter(r["old_pattern"] for r in grp)),
            "pf_pattern_counts": dict(Counter(r["pf_pattern"] for r in grp)),
            "old_cause_counts": dict(Counter(r["old_cause"] for r in grp)),
            "pf_cause_counts": dict(Counter(r["pf_cause"] for r in grp)),
        })
    write_csv(OUT / "focus_dataset_summary_16_13_10_2.csv", focus_summary)
    write_csv(OUT / "focus_dataset_stage45_oracle_reasons.csv", focus_oracle_rows)

    transition_rows = []
    for label, bundle in [("old", old), ("probfloor0002", pf)]:
        for ep, row in sorted(bundle.items()):
            if (row.get("family_schedule_phase") or row.get("schedule_phase")) != "target_post_switch":
                continue
            ds = int(row["dataset_index"])
            oracle = oracles[ds]
            missing_blockers, missing_tools = missing_oracle_tools(row, oracle)
            transition_rows.append({
                "run": label,
                "episode_index": ep,
                "dataset_index": ds,
                "oracle_action": row.get("oracle_action"),
                "final_action": row.get("final_action"),
                "transition": transition_type(row),
                "cause": cause_label(row, oracle),
                "terminal": ffloat(row.get("raw_terminal_penalty")),
                "total": ffloat(row.get("raw_total_cost")),
                "pattern": row.get("_pattern"),
                "pair": row.get("_majority_pair"),
                "shared": row.get("_selected_shared_path"),
                "missing_blockers": "|".join(missing_blockers),
                "missing_tools": "|".join(missing_tools),
                "replay_tools": list_str(row.get("stage5_replay_tool_names")),
                "terminal_reasons": list_str(row.get("terminal_adjustment_reasons")),
                "suffix": " > ".join((row.get("selected_path") or [])[-2:]),
                "original_task_id": row.get("original_task_id"),
            })
    write_csv(OUT / "transition_cause_post_episodes.csv", transition_rows)

    pattern_rows = []
    for label, bundle in [("old_seed1_risky_ps", old), ("seed1_probfloor0002", pf)]:
        rows = [r for r in bundle.values() if (r.get("family_schedule_phase") or r.get("schedule_phase")) == "target_post_switch"]
        for row in rows:
            row["pattern"] = row["_pattern"]
        pattern_rows.extend(summarize_group(rows, label, "pattern"))
    write_csv(OUT / "post_stage_pattern_details.csv", pattern_rows)

    suffix_rows = []
    for label, bundle in [("old", old), ("probfloor0002", pf)]:
        high = [
            r for r in bundle.values()
            if (r.get("family_schedule_phase") or r.get("schedule_phase")) == "target_post_switch"
            and ffloat(r.get("raw_terminal_penalty")) >= 10
        ]
        groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
        for r in high:
            suffix = " > ".join((r.get("selected_path") or [])[-2:])
            groups[(r["_pattern"], suffix, str(r.get("_selected_shared_path")))].append(r)
        for (pattern, suffix, shared), grp in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
            suffix_rows.append({
                "run": label,
                "pattern": pattern,
                "suffix": suffix,
                "shared": shared,
                "n_high_terminal": len(grp),
                "terminal_mean": avg(grp, "raw_terminal_penalty"),
                "datasets": ",".join(str(x) for x in sorted(set(int(r["dataset_index"]) for r in grp))),
                "episodes": ",".join(str(r["episode_index"]) for r in grp),
                "final_counts": dict(Counter(r.get("final_action") for r in grp)),
                "cause_counts": dict(Counter(cause_label(r, oracles[int(r["dataset_index"])]) for r in grp)),
            })
    write_csv(OUT / "high_terminal_suffix_concentration.csv", suffix_rows)

    # Compact JSON with full oracle notes for the focus datasets.
    focus_oracle = {}
    for ds in sorted(FOCUS_DATASETS):
        o = oracles[ds]
        focus_oracle[str(ds)] = {
            "original_task_id": o["original_task_id"],
            "expected_terminal_action": o["expected_terminal_action"],
            "repairability": o["repairability"],
            "selected_blockers": o["selected_blockers"],
            "deferred_blockers": o["deferred_blockers"],
            "oracle_actions": o["oracle_actions"],
            "stage4_blocker_reasons": o["stage4_blocker_reasons"],
            "stage4_blocker_decisions": o["stage4_blocker_decisions"],
            "stage4_blocker_deps": o["stage4_blocker_deps"],
            "stage5_success_condition": o["stage5_success_condition"],
            "stage5_required_postchecks": o["stage5_required_postchecks"],
            "transfer_reason": o["transfer_reason"],
        }
    (OUT / "focus_dataset_stage45_oracle_reasons.json").write_text(json.dumps(focus_oracle, indent=2, ensure_ascii=False))

    # Report.
    old_post = [r for r in old.values() if (r.get("family_schedule_phase") or r.get("schedule_phase")) == "target_post_switch"]
    pf_post = [r for r in pf.values() if (r.get("family_schedule_phase") or r.get("schedule_phase")) == "target_post_switch"]
    headline_rows = []
    for label, rows in [("old", old_post), ("probfloor0002", pf_post)]:
        headline_rows.append({
            "run": label,
            "n": len(rows),
            "total": avg(rows, "raw_total_cost"),
            "terminal": avg(rows, "raw_terminal_penalty"),
            "reasoning": avg(rows, "raw_reasoning_cost_component"),
            "modecost_report": avg(rows, "raw_mode_mismatch_cost_component"),
            "clear": rate(rows, "_clear_success_proxy"),
            "aux": rate(rows, "_auxiliary_success_proxy"),
            "strict": rate(rows, "_strict_clean_success"),
            "fast_on_deep_n": sum(1 for r in rows if r["_majority_pair"] == "mostly_fast_vs_mostly_deep_required"),
            "deep_on_deep_n": sum(1 for r in rows if r["_majority_pair"] == "mostly_deep_vs_mostly_deep_required"),
        })

    trans_summary = []
    transition_dataset_rows = []
    for label in ["old", "probfloor0002"]:
        grp = [r for r in transition_rows if r["run"] == label]
        for trans, n in Counter(r["transition"] for r in grp).most_common():
            sub = [r for r in grp if r["transition"] == trans]
            trans_summary.append({
                "run": label,
                "transition": trans,
                "n": n,
                "terminal": mean(r["terminal"] for r in sub),
                "main_causes": dict(Counter(r["cause"] for r in sub).most_common(4)),
            })
        for trans in ["repair_all_to_repair_subset", "local_to_transfer"]:
            sub = [r for r in grp if r["transition"] == trans]
            transition_dataset_rows.append({
                "run": label,
                "transition": trans,
                "n": len(sub),
                "terminal": mean(r["terminal"] for r in sub) if sub else 0.0,
                "dataset_counts": dict(Counter(r["dataset_index"] for r in sub).most_common()),
                "pattern_counts": dict(Counter(r["pattern"] for r in sub).most_common(8)),
            })

    report = []
    report.append("# Seed1 old risky_ps vs probfloor0002 targeted diagnostic\n")
    report.append("## Scope\n")
    report.append("- old: `tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/risky_ps`\n")
    report.append("- probfloor: `tmp/llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_probfloor0002/risky_ps`\n")
    report.append("- 对齐键：`episode_index`，并断言 `dataset_index` 一致。\n")
    report.append("- 注意：这两个 repeated smoke 产物没有保存完整 executor `stage_trace` 或 LLM Stage 4/5 raw JSON。下面的“Stage 4/5 reasons”因此分两类：task oracle 中的 Stage 4/5 blocker/terminal oracle reasons，以及 episode 扁平记录里的 `terminal_adjustment_reasons`、Stage 5 replay/executed tools、path/mode/interface 字段。\n")
    report.append("\n## Headline Post Split\n")
    report.append(markdown_table(headline_rows, ["run", "n", "total", "terminal", "reasoning", "modecost_report", "clear", "aux", "strict", "fast_on_deep_n", "deep_on_deep_n"]))
    report.append("\n\n## Probfloor Improved Fast-On-Deep Episodes\n")
    report.append(markdown_table(improved_fast_on_deep, ["episode_index", "dataset_index", "oracle_action", "old_terminal", "pf_terminal", "delta_terminal_pf_minus_old", "old_pattern", "pf_pattern", "old_final", "pf_final", "old_cause", "pf_cause", "pf_replay_tools"], 20))
    report.append("\n\n## All Probfloor Fast-On-Deep Episodes\n")
    report.append(markdown_table(all_probfloor_fast_on_deep, ["episode_index", "dataset_index", "oracle_action", "old_terminal", "pf_terminal", "delta_terminal_pf_minus_old", "old_pattern", "pf_pattern", "old_final", "pf_final", "old_cause", "pf_cause"], None))
    report.append("\n\n## Probfloor Deep-On-Deep Terminal Worse Episodes\n")
    report.append(markdown_table(deep_on_deep_worse, ["episode_index", "dataset_index", "oracle_action", "old_terminal", "pf_terminal", "delta_terminal_pf_minus_old", "old_pattern", "pf_pattern", "old_final", "pf_final", "old_cause", "pf_cause", "pf_missing_blockers"], 25))
    report.append("\n\n## Focus Datasets 16/13/10/2\n")
    report.append("\n\n### Stage 4/5 Oracle Reasons Available From Tasks\n")
    report.append(markdown_table(focus_oracle_rows, ["dataset_index", "oracle_action", "repairability", "selected", "deferred", "stage5", "stage4_repair_reasons", "stage4_defer_reasons"], None))
    report.append("\n\n### Episode Outcome Summary\n")
    report.append(markdown_table(focus_summary, ["dataset_index", "n", "oracle_action", "repairability", "old_terminal_mean", "pf_terminal_mean", "delta_terminal", "old_final_counts", "pf_final_counts", "old_cause_counts", "pf_cause_counts"], None))
    report.append("\n\n## Post Stage Pattern Details\n")
    report.append(markdown_table(pattern_rows, ["run", "pattern", "n", "terminal", "legacy_terminal", "reasoning", "modecost_report", "total", "clear", "aux", "strict", "transfer_final_n", "rtp_ge_10_n", "rtp_ge_14_n", "examples"], None))
    report.append("\n\n## Terminal Transition Causes\n")
    report.append(markdown_table(trans_summary, ["run", "transition", "n", "terminal", "main_causes"], None))
    report.append("\n\n### Targeted Transition Dataset/Pattern Counts\n")
    report.append(markdown_table(transition_dataset_rows, ["run", "transition", "n", "terminal", "dataset_counts", "pattern_counts"], None))
    report.append("\n\n## High-Terminal Suffix Concentration\n")
    report.append(markdown_table(suffix_rows[:30], ["run", "pattern", "suffix", "shared", "n_high_terminal", "terminal_mean", "datasets", "episodes", "final_counts", "cause_counts"], None))
    report.append("\n\n## Interpretation\n")
    report.append("- `probfloor0002` 的 post terminal 改善主要不是来自 deep-on-deep 质量提升；相反，deep-on-deep 桶里出现了多条 terminal 变差 episode。\n")
    report.append("- fast-on-deep 改善 episode 需要谨慎解读：如果它们从 old 的 `repair_all -> repair_subset` / clear failure 变成 probfloor 的低罚，且 Stage 5 replay tools 覆盖 oracle tools，说明 probfloor 只是把采样推到某些更好执行的接口/路径；这不是“fast-on-deep 普遍更好”。\n")
    report.append("- 高 terminal 如果集中在同一个 shared suffix 且跨 dataset 重复，才支持 shareable suffix failure。若同一 dataset 在不同 pattern/suffix 间好坏切换，或者 high terminal 同时伴随 missing oracle tools、subset mismatch、local clear floor，则更像 path/interface-specific execution failure。\n")
    report.append("- 当前扁平产物无法直接恢复 LLM Stage 4/5 raw rationale；若要定位“为什么 Stage 4 漏 blocker / Stage 5 转 transfer”，下一轮 runner 需要持久化 executor `stage_trace` 中的 `stage4_output`、`stage4_execution_summary`、`stage5_output`、`stage5_verification_summary` 和 raw LLM JSON。\n")
    (OUT / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
