"""Run repeated smoke on shared_basin_strong with method-level incremental persistence.

Scope:
- family_kind = shared_basin_strong
- executor_name = llm_bench
- model = gpt-4o-mini
- smoke10 repeated for a fixed horizon
- repeated-smoke baselines, each run as one stateful T=100 sequence
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import random
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from types import MethodType
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for extra in (
    ROOT / "envs",
    ROOT / "envs" / "adapters",
    ROOT / "envs" / "tree_family",
    ROOT / "envs" / "executors",
    ROOT / "baselines",
):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from fixed_tree_env import FixedTreeEnvironment  # noqa: E402
from direct_multistage_exp3 import DirectMultiStageExp3Policy  # noqa: E402
from direct_multistage_exp3_local import DirectMultiStageExp3LocalPolicy  # noqa: E402
from epsilon_exp3 import EpsilonExp3Policy  # noqa: E402
from naive_mixed import NaiveMixedPolicy  # noqa: E402
from naive_mixed_avg import NaiveMixedAveragePolicy  # noqa: E402
from oracle_eval import find_best_stationary_path  # noqa: E402
from random_path import RandomPathPolicy  # noqa: E402
from risky_ps import RiskyPSPolicy  # noqa: E402
from risky_ps_const_init import RiskyPSConstInitPolicy  # noqa: E402
from risky_ps_const_init_conserve import RiskyPSConstInitConservePolicy  # noqa: E402
from risky_ps_const_init_leaf_ratio_decay import (  # noqa: E402
    RiskyPSConstInitLeafRatioDecayPolicy,
)
from risky_ps_const_init_natural_decay import (  # noqa: E402
    RiskyPSConstInitNaturalDecayPolicy,
)
from risky_ps_direct_cost import RiskyPSDirectCostPolicy  # noqa: E402
from risky_ps_ix import RiskyPSIXPolicy  # noqa: E402
from risky_ps_linear import RiskyPSLinearPolicy  # noqa: E402
from risky_ps_old import RiskyPSOldPolicy  # noqa: E402
from risky_ps_safe_conditional import (  # noqa: E402
    RiskyPSSafeConditionalIXPolicy,
    RiskyPSSafeConditionalPolicy,
)


SMOKE10_INDICES = list(range(10))
DATASET_DEFAULT = (
    ROOT / "data" / "derived" / "telecom_mms_fixed_tree_base_v2_100_capabilities_time" / "tasks.json"
)
SPECIALIST_ANALYSIS_PATH = ROOT / "analysis" / "shared_basin_strong_static_analysis.json"
MODEL_REQUIRED = "gpt-4o-mini"
DEFAULT_FAMILY_KIND = "shared_basin_strong"
SCHEDULE_MODE_STATIONARY = "stationary"
SCHEDULE_MODE_TRAP_SWITCH = "trap_switch"
SCHEDULE_MODE_TRAP_ONLY_RANDOM = "trap_only_random"
TRAP_SWITCH_CYCLE_SOURCE_BUCKET = "bucket"
TRAP_SWITCH_CYCLE_SOURCE_DATASET = "dataset"
SEED = int(os.environ.get("PSAGENT_REPEATED_SMOKE_SEED", "0"))
DEFAULT_EXECUTOR_NAME = "llm_bench"


POLICY_REGISTRY = {
    "risky_ps_old": RiskyPSOldPolicy,
    "risky_ps": RiskyPSPolicy,
    "risky_ps_const_init": RiskyPSConstInitPolicy,
    "risky_ps_const_init_conserve": RiskyPSConstInitConservePolicy,
    "risky_ps_const_init_leaf_ratio_decay": RiskyPSConstInitLeafRatioDecayPolicy,
    "risky_ps_const_init_natural_decay": RiskyPSConstInitNaturalDecayPolicy,
    "risky_ps_linear": RiskyPSLinearPolicy,
    "risky_ps_ix": RiskyPSIXPolicy,
    "risky_ps_safe_conditional": RiskyPSSafeConditionalPolicy,
    "risky_ps_safe_conditional_ix": RiskyPSSafeConditionalIXPolicy,
    "risky_ps_direct_cost": RiskyPSDirectCostPolicy,
    "direct_multistage_exp3": DirectMultiStageExp3Policy,
    "direct_multistage_exp3_local": DirectMultiStageExp3LocalPolicy,
    "epsilon_exp3": EpsilonExp3Policy,
    "naive_mixed": NaiveMixedPolicy,
    "naive_mixed_avg": NaiveMixedAveragePolicy,
    "random_path": RandomPathPolicy,
}
COMMON_ETA_METHODS = frozenset(
    {
        "direct_multistage_exp3",
        "direct_multistage_exp3_local",
        "epsilon_exp3",
        "risky_ps_old",
        "risky_ps",
        "risky_ps_const_init",
        "risky_ps_const_init_conserve",
        "risky_ps_const_init_leaf_ratio_decay",
        "risky_ps_const_init_natural_decay",
        "risky_ps_linear",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    }
)
COMMON_EPSILON_METHODS = frozenset(
    {
        "epsilon_exp3",
        "risky_ps_old",
        "risky_ps",
        "risky_ps_const_init",
        "risky_ps_const_init_conserve",
        "risky_ps_const_init_leaf_ratio_decay",
        "risky_ps_const_init_natural_decay",
        "risky_ps_linear",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    }
)
PS_SHARED_OVERRIDE_METHODS = frozenset(
    {
        "risky_ps_old",
        "risky_ps",
        "risky_ps_const_init",
        "risky_ps_const_init_conserve",
        "risky_ps_const_init_leaf_ratio_decay",
        "risky_ps_const_init_natural_decay",
        "risky_ps_linear",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    }
)
POST_SWITCH_FREEZE_LAYER1_MODE = "root_direct_child_marginal_at_switch"
POST_SWITCH_FREEZE_TREE_MODE = "full_tree_child_marginal_at_switch"

DEFAULT_METHODS = [
    "risky_ps",
    "risky_ps_ix",
    "risky_ps_safe_conditional",
    "risky_ps_safe_conditional_ix",
    "risky_ps_direct_cost",
    "naive_mixed",
    "direct_multistage_exp3",
    "epsilon_exp3",
    "random_path",
]


def resolve_post_switch_probability_freeze_mode(
    *,
    post_switch_fixed_layer1_probs: bool = False,
    post_switch_fixed_tree_probs: bool = False,
) -> str | None:
    if post_switch_fixed_layer1_probs and post_switch_fixed_tree_probs:
        raise ValueError(
            "--post-switch-fixed-layer1-probs and --post-switch-fixed-tree-probs "
            "are mutually exclusive."
        )
    if post_switch_fixed_tree_probs:
        return POST_SWITCH_FREEZE_TREE_MODE
    if post_switch_fixed_layer1_probs:
        return POST_SWITCH_FREEZE_LAYER1_MODE
    return None


def validate_methods(methods: list[str]) -> None:
    invalid = [method for method in methods if method not in POLICY_REGISTRY]
    if invalid:
        raise SystemExit(
            f"Repeated smoke only supports these baselines: {sorted(POLICY_REGISTRY)}. "
            f"Unsupported methods: {invalid}"
        )


def build_policy_kwargs_by_method(
    methods: list[str],
    *,
    common_eta_override: float | None = None,
    common_epsilon_override: float | None = None,
    ps_eta_shared_override: float | None = None,
    ps_loss_clip: float | None = None,
    ps_prob_floor: float | None = None,
) -> dict[str, dict[str, Any]]:
    kwargs_by_method: dict[str, dict[str, Any]] = {}
    for method in methods:
        method_kwargs: dict[str, Any] = {}
        if common_eta_override is not None and method in COMMON_ETA_METHODS:
            method_kwargs["eta"] = common_eta_override
        if common_epsilon_override is not None and method in COMMON_EPSILON_METHODS:
            method_kwargs["epsilon"] = common_epsilon_override
        if method in COMMON_EPSILON_METHODS:
            if ps_eta_shared_override is not None and method in PS_SHARED_OVERRIDE_METHODS:
                method_kwargs["eta_shared"] = ps_eta_shared_override
            if ps_loss_clip is not None:
                method_kwargs["loss_clip"] = ps_loss_clip
            if ps_prob_floor is not None:
                method_kwargs["prob_floor"] = ps_prob_floor
        kwargs_by_method[method] = method_kwargs
    return kwargs_by_method


def mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_text_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        handle.write(text)
    os.replace(tmp_path, path)


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("wb") as handle:
        handle.write(payload)
    os.replace(tmp_path, path)


def write_json(path: Path, data: Any) -> None:
    write_text_atomic(path, json.dumps(data, ensure_ascii=False, indent=2))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
    write_text_atomic(path, payload)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp_path, path)


def load_instances(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list.")
    return data


def load_schedule_buckets(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(f"Schedule bucket file does not exist: {path}")
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError("Schedule bucket JSON must be an object.")
    return data


def load_specialist_task_ids(schedule_buckets: dict[str, Any] | None = None) -> set[str]:
    if schedule_buckets is not None:
        values = schedule_buckets.get("specialist_task_ids", [])
        if not isinstance(values, list):
            raise ValueError("specialist_task_ids must be a JSON list.")
        return {str(value) for value in values}
    data = load_json(SPECIALIST_ANALYSIS_PATH)
    return set(data.get("unshared_win_task_ids", []))


def build_env(*, executor_name: str, family_kind: str) -> FixedTreeEnvironment:
    return FixedTreeEnvironment(
        agent_catalog=[],
        family_kind=family_kind,
        family_seed=SEED,
        executor_name=executor_name,
    )


def build_repeated_selection(
    instances: list[dict[str, Any]],
    *,
    indices: list[int],
    repeats: int,
) -> list[dict[str, Any]]:
    repeated: list[dict[str, Any]] = []
    episode_index = 0
    for repeat_index in range(repeats):
        for position_in_cycle, dataset_index in enumerate(indices):
            repeated.append(
                {
                    "episode_index": episode_index,
                    "repeat_index": repeat_index,
                    "position_in_cycle": position_in_cycle,
                    "dataset_index": dataset_index,
                    "instance": instances[dataset_index],
                    "schedule_phase": SCHEDULE_MODE_STATIONARY,
                    "task_bucket": "stationary",
                    "is_specialist_task": False,
                }
            )
            episode_index += 1
    return repeated


def _instances_by_task_id(instances: list[dict[str, Any]]) -> dict[str, tuple[int, dict[str, Any]]]:
    mapping: dict[str, tuple[int, dict[str, Any]]] = {}
    for dataset_index, instance in enumerate(instances):
        task_id = str(instance["original_task_id"])
        if task_id in mapping:
            raise ValueError(f"Duplicate original_task_id in dataset: {task_id}")
        mapping[task_id] = (dataset_index, instance)
    return mapping


def build_trap_switch_selection(
    instances: list[dict[str, Any]],
    *,
    repeats: int,
    switch_denominator: int,
    schedule_buckets: dict[str, Any],
    trap_switch_cycle_source: str = TRAP_SWITCH_CYCLE_SOURCE_BUCKET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trap_ids_raw = schedule_buckets.get("trap_favoring_task_ids")
    target_ids_raw = schedule_buckets.get("target_favoring_task_ids")
    if not isinstance(trap_ids_raw, list) or not isinstance(target_ids_raw, list):
        raise ValueError(
            "trap_switch schedule requires trap_favoring_task_ids and "
            "target_favoring_task_ids JSON lists."
        )
    trap_ids = [str(value) for value in trap_ids_raw]
    target_ids = [str(value) for value in target_ids_raw]
    if not trap_ids or not target_ids:
        raise ValueError("trap_switch schedule buckets must both be non-empty.")
    if (
        trap_switch_cycle_source == TRAP_SWITCH_CYCLE_SOURCE_BUCKET
        and len(trap_ids) != len(target_ids)
    ):
        raise ValueError(
            "trap_switch schedule currently requires equal-sized trap_favoring_task_ids "
            f"and target_favoring_task_ids. got trap={len(trap_ids)} target={len(target_ids)}"
        )
    if trap_switch_cycle_source not in {
        TRAP_SWITCH_CYCLE_SOURCE_BUCKET,
        TRAP_SWITCH_CYCLE_SOURCE_DATASET,
    }:
        raise ValueError(
            "trap_switch_cycle_source must be "
            f"{TRAP_SWITCH_CYCLE_SOURCE_BUCKET!r} or {TRAP_SWITCH_CYCLE_SOURCE_DATASET!r}; "
            f"got {trap_switch_cycle_source!r}"
        )
    if switch_denominator <= 0:
        raise ValueError("switch_denominator must be a positive integer.")

    instances_by_task_id = _instances_by_task_id(instances)
    missing_ids = [
        task_id
        for task_id in [*trap_ids, *target_ids]
        if task_id not in instances_by_task_id
    ]
    if missing_ids:
        raise ValueError(
            "Schedule buckets reference task IDs not present in dataset: "
            + ", ".join(sorted(set(missing_ids))[:10])
        )

    specialist_task_ids = load_specialist_task_ids(schedule_buckets)
    missing_specialist_ids = [
        task_id for task_id in specialist_task_ids if task_id not in instances_by_task_id
    ]
    if missing_specialist_ids:
        raise ValueError(
            "specialist_task_ids reference task IDs not present in dataset: "
            + ", ".join(sorted(missing_specialist_ids)[:10])
        )
    cycle_length = (
        len(instances)
        if trap_switch_cycle_source == TRAP_SWITCH_CYCLE_SOURCE_DATASET
        else len(trap_ids)
    )
    total_episodes = repeats * cycle_length
    switch_episode = total_episodes // switch_denominator
    if switch_episode <= 0 or switch_episode >= total_episodes:
        raise ValueError(
            "trap_switch schedule must have a non-empty pre-switch and post-switch segment. "
            f"got total_episodes={total_episodes} switch_denominator={switch_denominator} "
            f"switch_episode={switch_episode}"
        )

    selected: list[dict[str, Any]] = []
    trap_phase_index = 0
    target_phase_index = 0
    for episode_index in range(total_episodes):
        repeat_index = episode_index // cycle_length
        position_in_cycle = episode_index % cycle_length
        if episode_index < switch_episode:
            task_id = trap_ids[trap_phase_index % len(trap_ids)]
            trap_phase_index += 1
            schedule_phase = "trap_pre_switch"
            task_bucket = "trap_favoring"
        else:
            task_id = target_ids[target_phase_index % len(target_ids)]
            target_phase_index += 1
            schedule_phase = "target_post_switch"
            task_bucket = "target_favoring"
        dataset_index, instance = instances_by_task_id[task_id]
        selected.append(
            {
                "episode_index": episode_index,
                "repeat_index": repeat_index,
                "position_in_cycle": position_in_cycle,
                "dataset_index": dataset_index,
                "instance": instance,
                "schedule_phase": schedule_phase,
                "task_bucket": task_bucket,
                "is_specialist_task": task_id in specialist_task_ids,
            }
        )

    metadata = {
        "cycle_length": cycle_length,
        "total_episodes": total_episodes,
        "switch_episode": switch_episode,
        "trap_bucket_size": len(trap_ids),
        "target_bucket_size": len(target_ids),
        "specialist_task_count": len(specialist_task_ids),
        "trap_switch_cycle_source": trap_switch_cycle_source,
    }
    return selected, metadata


def build_trap_only_random_selection(
    instances: list[dict[str, Any]],
    *,
    total_episodes: int,
    schedule_buckets: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    trap_ids_raw = schedule_buckets.get("trap_favoring_task_ids")
    if not isinstance(trap_ids_raw, list):
        raise ValueError("trap_only_random schedule requires trap_favoring_task_ids.")
    trap_ids = [str(value) for value in trap_ids_raw]
    if not trap_ids:
        raise ValueError("trap_only_random schedule requires a non-empty trap bucket.")

    instances_by_task_id = _instances_by_task_id(instances)
    missing_ids = [task_id for task_id in trap_ids if task_id not in instances_by_task_id]
    if missing_ids:
        raise ValueError(
            "Trap-only schedule buckets reference task IDs not present in dataset: "
            + ", ".join(sorted(set(missing_ids))[:10])
        )

    specialist_task_ids = load_specialist_task_ids(schedule_buckets)
    rng = random.Random(SEED)
    selected: list[dict[str, Any]] = []
    for episode_index in range(total_episodes):
        task_id = rng.choice(trap_ids)
        dataset_index, instance = instances_by_task_id[task_id]
        selected.append(
            {
                "episode_index": episode_index,
                "repeat_index": episode_index,
                "position_in_cycle": episode_index % len(trap_ids),
                "dataset_index": dataset_index,
                "instance": instance,
                "schedule_phase": "trap_pre_switch",
                "task_bucket": "trap_favoring",
                "is_specialist_task": task_id in specialist_task_ids,
            }
        )

    metadata = {
        "cycle_length": len(trap_ids),
        "total_episodes": total_episodes,
        "switch_episode": None,
        "trap_bucket_size": len(trap_ids),
        "target_bucket_size": 0,
        "specialist_task_count": len(specialist_task_ids),
        "trap_only_random": True,
        "trap_only_random_seed": SEED,
    }
    return selected, metadata


def build_schedule_selection(
    instances: list[dict[str, Any]],
    *,
    repeats: int,
    schedule_mode: str,
    switch_denominator: int,
    schedule_buckets: dict[str, Any] | None,
    trap_switch_cycle_source: str = TRAP_SWITCH_CYCLE_SOURCE_BUCKET,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if schedule_mode == SCHEDULE_MODE_STATIONARY:
        selected = build_repeated_selection(instances, indices=SMOKE10_INDICES, repeats=repeats)
        return selected, {
            "cycle_length": len(SMOKE10_INDICES),
            "total_episodes": len(selected),
            "switch_episode": None,
            "trap_bucket_size": None,
            "target_bucket_size": None,
        }
    if schedule_mode == SCHEDULE_MODE_TRAP_SWITCH:
        if schedule_buckets is None:
            raise FileNotFoundError(
                "trap_switch schedule requires --schedule-buckets and does not fallback silently."
            )
        return build_trap_switch_selection(
            instances,
            repeats=repeats,
            switch_denominator=switch_denominator,
            schedule_buckets=schedule_buckets,
            trap_switch_cycle_source=trap_switch_cycle_source,
        )
    if schedule_mode == SCHEDULE_MODE_TRAP_ONLY_RANDOM:
        if schedule_buckets is None:
            raise FileNotFoundError(
                "trap_only_random schedule requires --schedule-buckets with trap_favoring_task_ids."
            )
        return build_trap_only_random_selection(
            instances,
            total_episodes=repeats,
            schedule_buckets=schedule_buckets,
        )
    raise ValueError(f"Unsupported schedule_mode: {schedule_mode}")


def serialize_schedule(selected: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for episode_index, row in enumerate(selected):
        instance = row["instance"]
        rows.append(
            {
                "episode_index": int(row.get("episode_index", episode_index)),
                "repeat_index": row["repeat_index"],
                "position_in_cycle": row["position_in_cycle"],
                "dataset_index": row["dataset_index"],
                "instance_id": instance["instance_id"],
                "original_task_id": instance["original_task_id"],
                "schedule_phase": row.get("schedule_phase"),
                "task_bucket": row.get("task_bucket"),
                "is_specialist_task": bool(row.get("is_specialist_task", False)),
            }
        )
    return rows


def materialize_schedule(
    instances: list[dict[str, Any]],
    schedule_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for row in schedule_rows:
        dataset_index = int(row["dataset_index"])
        instance = instances[dataset_index]
        if instance["instance_id"] != row["instance_id"]:
            raise ValueError(
                f"Schedule/dataset mismatch at episode {row['episode_index']}: "
                f"expected instance_id={row['instance_id']}, got {instance['instance_id']}"
            )
        selected.append(
            {
                "episode_index": int(row["episode_index"]),
                "repeat_index": int(row["repeat_index"]),
                "position_in_cycle": int(row["position_in_cycle"]),
                "dataset_index": dataset_index,
                "instance": instance,
                "schedule_phase": row.get("schedule_phase", SCHEDULE_MODE_STATIONARY),
                "task_bucket": row.get("task_bucket", "stationary"),
                "is_specialist_task": bool(row.get("is_specialist_task", False)),
            }
        )
    return selected


def attach_schedule_metadata(instance: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    instance_copy = json.loads(json.dumps(instance))
    metadata = instance_copy.setdefault("metadata", {})
    schedule_payload = {
        "schedule_phase": row.get("schedule_phase", SCHEDULE_MODE_STATIONARY),
        "task_bucket": row.get("task_bucket", "stationary"),
        "is_specialist_task": bool(row.get("is_specialist_task", False)),
        "episode_index": int(row["episode_index"]),
        "repeat_index": int(row["repeat_index"]),
        "position_in_cycle": int(row["position_in_cycle"]),
    }
    metadata["psagent_schedule"] = schedule_payload
    return instance_copy


def compute_stationary_oracle(
    selected: list[dict[str, Any]],
    *,
    family_kind: str,
) -> dict[str, Any]:
    oracle_env = build_env(executor_name="simulated", family_kind=family_kind)
    oracle_path, oracle_summary_raw = find_best_stationary_path(
        [attach_schedule_metadata(row["instance"], row) for row in selected],
        oracle_env,
    )
    oracle_summary = {
        "path": list(oracle_path),
        "episode_total_costs": list(oracle_summary_raw["episode_total_costs"]),
        "episode_terminal_costs": list(oracle_summary_raw["episode_terminal_costs"]),
        "episode_raw_total_costs": list(oracle_summary_raw["episode_raw_total_costs"]),
        "episode_normalized_total_costs": list(oracle_summary_raw["episode_normalized_total_costs"]),
        "raw_cumulative_total_cost": float(oracle_summary_raw["raw_cumulative_total_cost"]),
        "raw_mean_total_cost": float(oracle_summary_raw["raw_mean_total_cost"]),
        "normalized_cumulative_total_cost": float(oracle_summary_raw["normalized_cumulative_total_cost"]),
        "normalized_mean_total_cost": float(oracle_summary_raw["normalized_mean_total_cost"]),
        "cost_scale_version": str(oracle_summary_raw["cost_scale_version"]),
        "cumulative_total_cost": float(oracle_summary_raw["cumulative_total_cost"]),
        "mean_total_cost": float(oracle_summary_raw["mean_total_cost"]),
    }
    return oracle_summary


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def mean_present(values: list[Any]) -> float:
    numeric = [float(value) for value in values if value is not None]
    return mean(numeric)


def distribution_with_fraction(labels: list[Any]) -> dict[str, dict[str, Any]]:
    if not labels:
        return {}
    counter = Counter("none" if label in {None, ""} else str(label) for label in labels)
    total = sum(counter.values())
    return {
        key: {
            "count": count,
            "fraction": (count / total) if total else 0.0,
        }
        for key, count in sorted(counter.items())
    }


def mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    width = max(len(vector) for vector in vectors)
    result: list[float] = []
    for idx in range(width):
        result.append(
            mean([
                float(vector[idx])
                for vector in vectors
                if idx < len(vector)
            ])
        )
    return result


def normalize_probs(probs: list[float]) -> list[float]:
    cleaned = [max(0.0, float(prob)) for prob in probs]
    total = sum(cleaned)
    if total <= 0.0:
        return [1.0 / len(cleaned) for _ in cleaned]
    return [prob / total for prob in cleaned]


def sample_index_from_probs(probs: list[float], draw: float) -> int:
    cumulative = 0.0
    for idx, prob in enumerate(probs):
        cumulative += prob
        if draw <= cumulative:
            return idx
    return len(probs) - 1


def direct_child_prefixes(
    policy: Any,
    env: FixedTreeEnvironment,
    prefix: tuple[str, ...],
) -> list[tuple[str, ...]]:
    depth = len(prefix)
    if depth >= len(env.STAGE_NAMES):
        return []
    stage_name = env.STAGE_NAMES[depth]
    if hasattr(policy, "_sample_stage_child"):
        agent_ids = policy._legal_agent_ids_for_prefix(prefix, stage_name, env)
        return list(policy._child_prefixes(prefix, agent_ids))
    return list(policy._child_prefixes(prefix, stage_name, env))


def child_distribution(
    policy: Any,
    env: FixedTreeEnvironment,
    prefix: tuple[str, ...],
) -> tuple[list[tuple[str, ...]], list[float], str]:
    child_prefixes = direct_child_prefixes(policy, env, prefix)
    if not child_prefixes:
        raise RuntimeError(
            "No children available for post-switch probability freeze. "
            f"prefix={list(prefix)}"
        )
    if hasattr(policy, "_stage_probs"):
        probs = normalize_probs(list(policy._stage_probs(prefix, child_prefixes)))
        return child_prefixes, probs, "stagewise_marginal_mixture"
    if getattr(policy, "safe_prefixes", {}).get(prefix, False):
        probs = normalize_probs(list(policy._safe_child_probs(prefix, child_prefixes)))
        return child_prefixes, probs, "ps_safe_prefix_mass"
    exploit_probs = list(policy._risky_child_probs(prefix, child_prefixes))
    epsilon = min(1.0, max(0.0, float(getattr(policy, "epsilon", 0.0))))
    uniform_prob = 1.0 / len(child_prefixes)
    probs = normalize_probs(
        [(1.0 - epsilon) * prob + epsilon * uniform_prob for prob in exploit_probs]
    )
    return child_prefixes, probs, "ps_risky_marginal_mixture"


def snapshot_child_distributions(
    *,
    policy: Any,
    env: FixedTreeEnvironment,
    method: str,
    episode_index: int,
    freeze_mode: str,
) -> tuple[dict[tuple[str, ...], dict[tuple[str, ...], float]], list[dict[str, Any]]]:
    if freeze_mode == POST_SWITCH_FREEZE_LAYER1_MODE:
        prefixes = [()]
    elif freeze_mode == POST_SWITCH_FREEZE_TREE_MODE:
        prefixes: list[tuple[str, ...]] = []
        frontier: list[tuple[str, ...]] = [()]
        while frontier:
            prefix = frontier.pop(0)
            child_prefixes = direct_child_prefixes(policy, env, prefix)
            if not child_prefixes:
                continue
            prefixes.append(prefix)
            frontier.extend(child_prefixes)
    else:
        raise ValueError(f"Unknown post-switch freeze mode: {freeze_mode}")

    frozen: dict[tuple[str, ...], dict[tuple[str, ...], float]] = {}
    rows: list[dict[str, Any]] = []
    for prefix in prefixes:
        child_prefixes, probs, distribution_kind = child_distribution(policy, env, prefix)
        frozen[prefix] = {
            tuple(child_prefix): float(prob)
            for child_prefix, prob in zip(child_prefixes, probs)
        }
        for rank, (child_prefix, prob) in enumerate(zip(child_prefixes, probs), start=1):
            rows.append(
                {
                    "method": method,
                    "snapshot_episode_index": episode_index,
                    "snapshot_episode_1based": episode_index + 1,
                    "prefix_depth": len(prefix),
                    "prefix": list(prefix),
                    "parent_id": prefix[-1] if prefix else "ROOT",
                    "rank": rank,
                    "child_id": child_prefix[-1],
                    "child_prefix": list(child_prefix),
                    "prob": float(prob),
                    "child_count": len(child_prefixes),
                    "distribution_kind": distribution_kind,
                    "freeze_mode": freeze_mode,
                }
            )
    return frozen, rows


def canonical_all_trap_path(env: FixedTreeEnvironment) -> list[str] | None:
    family_agent_map = getattr(env, "family_agent_map", None)
    if not family_agent_map:
        return None
    candidates: list[list[str]] = []
    for path in find_all_env_paths(env):
        labels = [
            str(getattr(family_agent_map[agent_id], "route_label", ""))
            for agent_id in path
        ]
        if labels and all(label.startswith("trap_") for label in labels):
            candidates.append(list(path))
    if len(candidates) != 1:
        return None
    return candidates[0]


def find_all_env_paths(env: FixedTreeEnvironment) -> list[list[str]]:
    # Local import keeps the runner startup path unchanged for commands that do not
    # need path introspection.
    from oracle_eval import enumerate_all_paths

    return [list(path) for path in enumerate_all_paths(env)]


def child_probability_for_policy(
    policy: Any,
    env: FixedTreeEnvironment,
    *,
    prefix: tuple[str, ...],
    child_prefix: tuple[str, ...],
    stage_name: str,
) -> tuple[float | None, int]:
    if hasattr(policy, "_stage_probs") and not hasattr(policy, "_safe_child_probs"):
        agent_ids = policy._legal_agent_ids_for_prefix(prefix, stage_name, env)
        child_prefixes = policy._child_prefixes(prefix, agent_ids)
        probs = list(policy._stage_probs(prefix, child_prefixes))
    elif hasattr(policy, "_safe_child_probs") and hasattr(policy, "_risky_child_probs"):
        child_prefixes = policy._child_prefixes(prefix, stage_name, env)
        if policy.safe_prefixes.get(prefix, False):
            probs = list(policy._safe_child_probs(prefix, child_prefixes))
        else:
            exploit_probs = list(policy._risky_child_probs(prefix, child_prefixes))
            epsilon = min(1.0, max(0.0, float(getattr(policy, "epsilon", 0.0))))
            uniform_prob = 1.0 / len(child_prefixes)
            probs = [
                (1.0 - epsilon) * prob + epsilon * uniform_prob
                for prob in exploit_probs
            ]
    else:
        return None, 0
    try:
        index = child_prefixes.index(child_prefix)
    except ValueError:
        return None, len(child_prefixes)
    return float(probs[index]), len(child_prefixes)


def trap_route_probability_snapshot(
    policy: Any,
    env: FixedTreeEnvironment,
) -> dict[str, Any]:
    trap_path = canonical_all_trap_path(env)
    if not trap_path:
        return {}
    family_agent_map = getattr(env, "family_agent_map", {}) or {}
    route_labels = [
        str(getattr(family_agent_map[agent_id], "route_label", ""))
        for agent_id in trap_path
    ]
    prefix: tuple[str, ...] = ()
    stage_probs: dict[str, float | None] = {}
    arm_counts: dict[str, int] = {}
    product = 1.0
    for depth, stage_name in enumerate(env.STAGE_NAMES):
        child_prefix = tuple(trap_path[: depth + 1])
        prob, arm_count = child_probability_for_policy(
            policy,
            env,
            prefix=prefix,
            child_prefix=child_prefix,
            stage_name=stage_name,
        )
        stage_probs[stage_name] = prob
        arm_counts[stage_name] = arm_count
        if prob is None:
            product = 0.0
        else:
            product *= prob
        prefix = child_prefix
    return {
        "trap_probability_path": list(trap_path),
        "trap_probability_route_labels": route_labels,
        "root_trap_subtree_prob": stage_probs.get("stage1"),
        "stage4_trap_child_prob": stage_probs.get("stage4"),
        "all_fast_trap_route_prob": product,
        "trap_route_stage_probs": stage_probs,
        "trap_route_stage_arm_counts": arm_counts,
    }


def _sample_stage_child_with_probability_freeze(
    self: Any,
    current_prefix: tuple[str, ...],
    child_prefixes: list[tuple[str, ...]],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    prefix = tuple(current_prefix)
    if getattr(self, "_post_switch_probability_freeze_active", False):
        frozen_by_prefix = getattr(self, "_post_switch_frozen_child_probs_by_prefix", None)
        frozen = frozen_by_prefix.get(prefix) if frozen_by_prefix else None
        if frozen is not None:
            probs = normalize_probs(
                [float(frozen[tuple(child_prefix)]) for child_prefix in child_prefixes]
            )
            selected_idx = sample_index_from_probs(probs, self.rng.random())
            selected = child_prefixes[selected_idx]
            prob = probs[selected_idx]
            freeze_mode = getattr(self, "_post_switch_probability_freeze_mode", None)
            return selected, {
                "epsilon": getattr(self, "epsilon", None),
                "epsilon_mode": "F",
                "selection_mode": (
                    "frozen_tree_post_switch"
                    if freeze_mode == POST_SWITCH_FREEZE_TREE_MODE
                    else "frozen_root_post_switch"
                ),
                "branch_conditional_prob": prob,
                "conditional_prob": prob,
                "mixture_conditional_prob": prob,
                "softmax_conditional_prob": None,
                "uniform_conditional_prob": None,
            }
    return type(self)._sample_stage_child(self, current_prefix, child_prefixes)


def _sample_child_prefix_with_probability_freeze(
    self: Any,
    current_prefix: tuple[str, ...],
    stage_name: str,
    env: FixedTreeEnvironment,
) -> tuple[tuple[str, ...], float, dict[str, Any]]:
    prefix = tuple(current_prefix)
    if getattr(self, "_post_switch_probability_freeze_active", False):
        child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
        frozen_by_prefix = getattr(self, "_post_switch_frozen_child_probs_by_prefix", None)
        frozen = frozen_by_prefix.get(prefix) if frozen_by_prefix else None
        if frozen is not None:
            probs = normalize_probs(
                [float(frozen[tuple(child_prefix)]) for child_prefix in child_prefixes]
            )
            selected_idx = sample_index_from_probs(probs, self.rng.random())
            selected = child_prefixes[selected_idx]
            prob = probs[selected_idx]
            freeze_mode = getattr(self, "_post_switch_probability_freeze_mode", None)
            return selected, prob, {
                "epsilon": getattr(self, "epsilon", None),
                "epsilon_mode": "F",
                "selection_mode": (
                    "frozen_tree_post_switch"
                    if freeze_mode == POST_SWITCH_FREEZE_TREE_MODE
                    else "frozen_root_post_switch"
                ),
                "branch_conditional_prob": prob,
                "conditional_prob": prob,
                "mixture_conditional_prob": prob,
                "softmax_conditional_prob": None,
                "uniform_conditional_prob": None,
                "estimated_loss_denominator": "branch_edge_prob",
                "estimator_scope": (
                    "frozen_tree_post_switch_branch_probability"
                    if freeze_mode == POST_SWITCH_FREEZE_TREE_MODE
                    else "frozen_root_post_switch_branch_probability"
                ),
            }
    return type(self)._sample_child_prefix(self, current_prefix, stage_name, env)


def install_post_switch_probability_freeze(policy: Any) -> None:
    if getattr(policy, "_post_switch_probability_freeze_installed", False):
        return
    if hasattr(policy, "_sample_stage_child"):
        policy._sample_stage_child = MethodType(_sample_stage_child_with_probability_freeze, policy)
    elif hasattr(policy, "_sample_child_prefix"):
        policy._sample_child_prefix = MethodType(_sample_child_prefix_with_probability_freeze, policy)
    else:
        raise TypeError(
            f"Unsupported policy type for post-switch probability freeze: {type(policy).__name__}"
        )
    policy._post_switch_probability_freeze_installed = True
    policy._post_switch_probability_freeze_active = False
    policy._post_switch_probability_freeze_mode = None
    policy._post_switch_frozen_child_probs_by_prefix = None


def stage_edge_rows(selection_info: dict[str, Any]) -> list[dict[str, Any]]:
    edges = selection_info.get("selected_edges") or selection_info.get("sampled_edges") or []
    return list(edges) if isinstance(edges, list) else []


def flatten_episode(
    *,
    episode_index: int,
    row: dict[str, Any],
    result: Any,
    method: str,
    oracle_summary: dict[str, Any],
    selection_info: dict[str, Any],
    update_info: dict[str, Any],
    specialist_task_ids: set[str],
) -> dict[str, Any]:
    instance = row["instance"]
    log = result.episode_log or {}
    selected_edges = stage_edge_rows(selection_info)
    root_edge = selected_edges[0] if selected_edges else {}
    stage_trace = {
        stage_row["stage_name"]: stage_row for stage_row in log.get("stage_trace", [])
    }
    stage_sources = {name: stage_row.get("source") for name, stage_row in stage_trace.items()}
    llm_stage_names = [name for name, source in stage_sources.items() if source == "llm_bench"]
    llm_call_count = int(
        log.get(
            "llm_call_count",
            sum(
                int(
                    stage_trace[name].get(
                        "llm_call_count_stage",
                        len(stage_trace[name].get("llm_raw_output", [])),
                    )
                    or 0
                )
                for name in llm_stage_names
            ),
        )
        or 0
    )
    tool_calls_made = sum(
        len(stage_row.get("executed_tool_calls", [])) for stage_row in stage_trace.values()
    )
    mutating_tool_calls_made = len(stage_trace.get("stage4", {}).get("executed_tool_calls", []))
    assistant_side_mutating_tool_calls_made = sum(
        1
        for call in stage_trace.get("stage4", {}).get("executed_tool_calls", [])
        if call.get("requestor") == "assistant"
    )
    stage5_trace = stage_trace.get("stage5", {})
    leaf_type = result.leaf_type
    shared_path = leaf_type == "shared"
    specialist_task = instance["original_task_id"] in specialist_task_ids
    shared_updates = update_info.get("shared_safe_suffix_edges_updated", []) or []
    risky_updates = update_info.get("risky_edges_updated", []) or []
    stage_prompt_tokens = [
        float(stage_trace.get(stage_name, {}).get("prompt_tokens_total_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_completion_tokens = [
        float(
            stage_trace.get(stage_name, {}).get("completion_tokens_total_stage", 0.0) or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_total_tokens = [
        float(stage_trace.get(stage_name, {}).get("total_tokens_total_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_api_cost_usd = [
        float(stage_trace.get(stage_name, {}).get("api_cost_total_usd_stage", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_generation_time_seconds = [
        float(
            stage_trace.get(stage_name, {}).get("generation_time_total_seconds_stage", 0.0)
            or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_llm_round_trip_seconds = [
        float(
            stage_trace.get(stage_name, {}).get("llm_round_trip_total_seconds_stage", 0.0)
            or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_wall_clock_seconds = [
        float(stage_trace.get(stage_name, {}).get("stage_wall_clock_seconds", 0.0) or 0.0)
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    stage_tool_wall_clock_seconds = [
        float(
            stage_trace.get(stage_name, {}).get("tool_wall_clock_total_seconds_stage", 0.0)
            or 0.0
        )
        for stage_name in FixedTreeEnvironment.STAGE_NAMES
    ]
    return {
        "method": method,
        "episode_index": episode_index,
        "repeat_index": row["repeat_index"],
        "position_in_cycle": row["position_in_cycle"],
        "dataset_index": row["dataset_index"],
        "instance_id": instance["instance_id"],
        "original_task_id": instance["original_task_id"],
        "is_specialist_task": specialist_task,
        "selected_path": list(result.selected_path),
        "leaf_type": leaf_type,
        "selected_shared_path": shared_path,
        "selected_unshared_path": not shared_path,
        "schedule_phase": row.get("schedule_phase"),
        "task_bucket": row.get("task_bucket"),
        "family_behavior_archetype": log.get("family_behavior_archetype"),
        "family_schedule_phase": log.get("family_schedule_phase"),
        "family_task_bucket": log.get("family_task_bucket"),
        "family_route_labels": list(log.get("family_route_labels", []) or []),
        "family_deliberation_modes": list(log.get("family_deliberation_modes", []) or []),
        "family_node_semantics": list(log.get("family_node_semantics", []) or []),
        "family_fast_stage_count": log.get("family_fast_stage_count"),
        "family_trap_label_count": log.get("family_trap_label_count"),
        "family_target_label_count": log.get("family_target_label_count"),
        "family_general_label_count": log.get("family_general_label_count"),
        "family_barrier_label_count": log.get("family_barrier_label_count"),
        "family_trap_like_path": log.get("family_trap_like_path"),
        "family_target_safe_subtree": log.get("family_target_safe_subtree"),
        "family_exact_target_good": log.get("family_exact_target_good"),
        "family_decoy_path": log.get("family_decoy_path"),
        "oracle_action": result.oracle_action,
        "final_action": result.final_action,
        "exact_match": bool(result.success),
        "subset_mismatch": bool(log.get("subset_mismatch", False)),
        "terminal_penalty": float(result.terminal_cost),
        "raw_outcome_penalty": float(log.get("raw_outcome_penalty", 0.0) or 0.0),
        "raw_policy_penalty": float(log.get("raw_policy_penalty", 0.0) or 0.0),
        "raw_terminal_penalty": float(result.raw_terminal_penalty),
        "legacy_raw_terminal_penalty": float(
            log.get("legacy_raw_terminal_penalty", result.raw_terminal_penalty) or 0.0
        ),
        "raw_terminal_penalty_exec_clean_v4": (
            float(log.get("raw_terminal_penalty_exec_clean_v4"))
            if log.get("raw_terminal_penalty_exec_clean_v4") is not None
            else None
        ),
        "terminal_adjustment_enabled": bool(
            log.get("terminal_adjustment_enabled", False)
        ),
        "terminal_adjustment_floor": log.get("terminal_adjustment_floor"),
        "terminal_adjustment_reasons": list(
            log.get("terminal_adjustment_reasons", []) or []
        ),
        "terminal_clear_success_proxy": bool(
            log.get("clear_success_proxy", bool(result.success))
        ),
        "terminal_auxiliary_success_proxy": bool(
            log.get(
                "auxiliary_success_proxy",
                int(log.get("policy_violation_count", 0) or 0) == 0,
            )
        ),
        "terminal_majority_pair": log.get("terminal_majority_pair"),
        "total_cost": float(result.total_cost),
        "raw_total_cost": float(result.raw_total_cost),
        "raw_total_cost_api": (
            float(log.get("raw_total_cost_api"))
            if log.get("raw_total_cost_api") is not None
            else None
        ),
        "raw_total_cost_token": (
            float(log.get("raw_total_cost_token"))
            if log.get("raw_total_cost_token") is not None
            else None
        ),
        "raw_path_cost_component": float(result.raw_path_cost_component),
        "raw_reasoning_cost_component": float(result.raw_reasoning_cost_component),
        "raw_mode_mismatch_cost_component": float(
            log.get("raw_mode_mismatch_cost_component", 0.0) or 0.0
        ),
        "mode_mismatch_cost_enabled": bool(
            log.get("mode_mismatch_cost_enabled", False)
        ),
        "mode_mismatch_report_only_enabled": bool(
            log.get("mode_mismatch_report_only_enabled", False)
        ),
        "mode_mismatch_fast_on_deep_cost": float(
            log.get("mode_mismatch_fast_on_deep_cost", 0.0) or 0.0
        ),
        "mode_mismatch_deep_on_fast_cost": float(
            log.get("mode_mismatch_deep_on_fast_cost", 0.0) or 0.0
        ),
        "raw_reasoning_cost_component_api": (
            float(log.get("raw_reasoning_cost_component_api"))
            if log.get("raw_reasoning_cost_component_api") is not None
            else None
        ),
        "raw_reasoning_cost_component_token": (
            float(log.get("raw_reasoning_cost_component_token"))
            if log.get("raw_reasoning_cost_component_token") is not None
            else None
        ),
        "reasoning_cost": float(result.reasoning_cost),
        "reasoning_cost_mode_default": log.get("reasoning_cost_mode_default"),
        "reasoning_weight_calibration_enabled": bool(
            log.get("reasoning_weight_calibration_enabled", False)
        ),
        "policy_eval_source": log.get("policy_eval_source"),
        "policy_eval_scope": log.get("policy_eval_scope"),
        "terminal_cost_upper_bound": log.get("terminal_cost_upper_bound"),
        "path_cost_upper_bound": log.get("path_cost_upper_bound"),
        "reasoning_cost_upper_bound": log.get("reasoning_cost_upper_bound"),
        "total_cost_upper_bound": log.get("total_cost_upper_bound"),
        "cost_scale_version": str(result.cost_scale_version),
        "stage_sources": stage_sources,
        "llm_stage_names": llm_stage_names,
        "llm_call_count": llm_call_count,
        "prompt_tokens_total": float(log.get("prompt_tokens_total", 0.0) or 0.0),
        "completion_tokens_total": float(log.get("completion_tokens_total", 0.0) or 0.0),
        "total_tokens_total": float(log.get("total_tokens_total", 0.0) or 0.0),
        "api_cost_total_usd_raw": float(log.get("api_cost_total_usd_raw", 0.0) or 0.0),
        "generation_time_total_seconds": float(
            log.get("generation_time_total_seconds", 0.0) or 0.0
        ),
        "llm_round_trip_total_seconds": float(
            log.get("llm_round_trip_total_seconds", 0.0) or 0.0
        ),
        "tool_wall_clock_total_seconds": float(
            log.get("tool_wall_clock_total_seconds", 0.0) or 0.0
        ),
        "episode_wall_clock_seconds": float(
            log.get("episode_wall_clock_seconds", 0.0) or 0.0
        ),
        "stage_prompt_tokens": stage_prompt_tokens,
        "stage_completion_tokens": stage_completion_tokens,
        "stage_total_tokens": stage_total_tokens,
        "stage_api_cost_usd": stage_api_cost_usd,
        "stage_generation_time_seconds": stage_generation_time_seconds,
        "stage_llm_round_trip_seconds": stage_llm_round_trip_seconds,
        "stage_tool_wall_clock_seconds": stage_tool_wall_clock_seconds,
        "stage_wall_clock_seconds": stage_wall_clock_seconds,
        "tool_calls_made": tool_calls_made,
        "mutating_tool_calls_made": mutating_tool_calls_made,
        "assistant_side_mutating_tool_calls_made": assistant_side_mutating_tool_calls_made,
        "stage5_replay_tool_names": [c.get("name") for c in stage5_trace.get("replay_tool_calls", [])],
        "stage5_executed_tool_names": [c.get("name") for c in stage5_trace.get("executed_tool_calls", [])],
        "policy_action_violation": bool(log.get("policy_action_violation", False)),
        "policy_communication_violation": bool(
            log.get("policy_communication_violation", False)
        ),
        "policy_nl_assertions_total": int(log.get("policy_nl_assertions_total", 0) or 0),
        "policy_nl_assertions_failed": int(
            log.get("policy_nl_assertions_failed", 0) or 0
        ),
        "policy_violation_count": int(log.get("policy_violation_count", 0) or 0),
        "first_private_barrier_stage": log.get("first_private_barrier_stage"),
        "barrier_stop_depth": log.get("barrier_stop_depth"),
        "candidate_count_per_stage": list(log.get("candidate_count_per_stage", []) or []),
        "legal_child_count_per_stage": list(
            log.get("legal_child_count_per_stage", []) or []
        ),
        "selection_path_prob": selection_info.get("path_prob"),
        "selection_stage_probs": dict(selection_info.get("stage_probs", {}) or {}),
        "root_trap_subtree_prob": selection_info.get("root_trap_subtree_prob"),
        "stage4_trap_child_prob": selection_info.get("stage4_trap_child_prob"),
        "all_fast_trap_route_prob": selection_info.get("all_fast_trap_route_prob"),
        "trap_route_stage_probs": dict(
            selection_info.get("trap_route_stage_probs", {}) or {}
        ),
        "trap_route_stage_arm_counts": dict(
            selection_info.get("trap_route_stage_arm_counts", {}) or {}
        ),
        "root_child_id": result.selected_path[0] if result.selected_path else None,
        "root_selection_mode": root_edge.get("selection_mode"),
        "root_conditional_prob": root_edge.get("conditional_prob"),
        "root_branch_conditional_prob": root_edge.get("branch_conditional_prob"),
        "root_mixture_conditional_prob": root_edge.get("mixture_conditional_prob"),
        "post_switch_probability_freeze_active": bool(
            selection_info.get("post_switch_probability_freeze_active", False)
        ),
        "post_switch_probability_freeze_mode": selection_info.get(
            "post_switch_probability_freeze_mode"
        ),
        "post_switch_layer1_freeze_active": bool(
            selection_info.get("post_switch_layer1_freeze_active", False)
        ),
        "post_switch_layer1_freeze_mode": selection_info.get("post_switch_layer1_freeze_mode"),
        "shared_branch_triggered": bool(update_info.get("shared_leaf_updated", False)),
        "unshared_branch_triggered": str(update_info.get("leaf_type")) == "unshared",
        "shared_update_count": len(shared_updates),
        "unshared_edge_update_count": len(risky_updates),
        "risky_edge_update_edges": risky_updates,
        "selection_info": selection_info,
        "update_info": update_info,
    }


def add_cumulative_fields(episodes: list[dict[str, Any]]) -> None:
    shared_count = 0
    unshared_count = 0
    shared_branch_count = 0
    unshared_branch_count = 0
    shared_update_count = 0
    unshared_edge_update_count = 0
    window: list[bool] = []
    for idx, row in enumerate(episodes, start=1):
        is_shared = bool(row["selected_shared_path"])
        shared_count += int(is_shared)
        unshared_count += int(not is_shared)
        shared_branch_count += int(bool(row["shared_branch_triggered"]))
        unshared_branch_count += int(bool(row["unshared_branch_triggered"]))
        shared_update_count += int(row["shared_update_count"])
        unshared_edge_update_count += int(row["unshared_edge_update_count"])
        window.append(is_shared)
        if len(window) > 10:
            window.pop(0)
        row["cumulative_shared_path_ratio"] = shared_count / idx
        row["cumulative_unshared_path_ratio"] = unshared_count / idx
        row["rolling_shared_path_ratio_last10"] = sum(window) / len(window)
        row["rolling_unshared_path_ratio_last10"] = 1.0 - row["rolling_shared_path_ratio_last10"]
        row["cumulative_shared_branch_count"] = shared_branch_count
        row["cumulative_unshared_branch_count"] = unshared_branch_count
        row["cumulative_shared_update_count"] = shared_update_count
        row["cumulative_unshared_edge_update_count"] = unshared_edge_update_count


def build_summary(
    *,
    method: str,
    dataset: str,
    repeats: int,
    model: str,
    family_kind: str,
    executor_name: str,
    schedule_mode: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
) -> dict[str, Any]:
    stage_source_summary: dict[str, Counter] = defaultdict(Counter)
    for episode in episodes:
        for stage_name, source in episode["stage_sources"].items():
            stage_source_summary[stage_name][str(source)] += 1
    policy_nl_total = sum(ep["policy_nl_assertions_total"] for ep in episodes)
    policy_nl_failed = sum(ep["policy_nl_assertions_failed"] for ep in episodes)
    post_switch_episodes = [
        ep for ep in episodes if ep.get("schedule_phase") == "target_post_switch"
    ]
    return {
        "test_name": f"{method}_repeated_{schedule_mode}_x{repeats}_{family_kind}_full_llm",
        "dataset": dataset,
        "dataset_indices": sorted({int(ep["dataset_index"]) for ep in episodes}),
        "repeats": repeats,
        "episodes": len(episodes),
        "method": method,
        "mechanism": "algorithm_direct",
        "executor_name": executor_name,
        "family_kind": family_kind,
        "schedule_mode": schedule_mode,
        "seed": SEED,
        "model": model,
        "stationary_oracle_path": oracle_summary["path"],
        "exact_match_mean": mean([float(ep["exact_match"]) for ep in episodes]),
        "terminal_penalty_mean": mean([ep["terminal_penalty"] for ep in episodes]),
        "raw_outcome_penalty_mean": mean([ep["raw_outcome_penalty"] for ep in episodes]),
        "raw_policy_penalty_mean": mean([ep["raw_policy_penalty"] for ep in episodes]),
        "raw_terminal_penalty_mean": mean([ep["raw_terminal_penalty"] for ep in episodes]),
        "legacy_raw_terminal_penalty_mean": mean(
            [ep.get("legacy_raw_terminal_penalty", ep["raw_terminal_penalty"]) for ep in episodes]
        ),
        "raw_terminal_penalty_exec_clean_v4_mean": mean_present(
            [ep.get("raw_terminal_penalty_exec_clean_v4") for ep in episodes]
        ),
        "total_cost_mean": mean([ep["total_cost"] for ep in episodes]),
        "raw_total_cost_mean": mean([ep["raw_total_cost"] for ep in episodes]),
        "raw_total_cost_api_mean": mean_present([ep["raw_total_cost_api"] for ep in episodes]),
        "raw_total_cost_token_mean": mean_present(
            [ep["raw_total_cost_token"] for ep in episodes]
        ),
        "reasoning_cost_mean": mean([ep["reasoning_cost"] for ep in episodes]),
        "raw_reasoning_cost_component_mean": mean([ep["raw_reasoning_cost_component"] for ep in episodes]),
        "raw_mode_mismatch_cost_component_mean": mean(
            [ep.get("raw_mode_mismatch_cost_component", 0.0) for ep in episodes]
        ),
        "root_trap_subtree_prob_mean": mean_present(
            [ep.get("root_trap_subtree_prob") for ep in episodes]
        ),
        "stage4_trap_child_prob_mean": mean_present(
            [ep.get("stage4_trap_child_prob") for ep in episodes]
        ),
        "all_fast_trap_route_prob_mean": mean_present(
            [ep.get("all_fast_trap_route_prob") for ep in episodes]
        ),
        "raw_reasoning_cost_component_api_mean": mean_present(
            [ep["raw_reasoning_cost_component_api"] for ep in episodes]
        ),
        "raw_reasoning_cost_component_token_mean": mean_present(
            [ep["raw_reasoning_cost_component_token"] for ep in episodes]
        ),
        "raw_path_cost_component_mean": mean([ep["raw_path_cost_component"] for ep in episodes]),
        "algorithm_cumulative_total_cost": sum(ep["total_cost"] for ep in episodes),
        "raw_algorithm_cumulative_total_cost": sum(ep["raw_total_cost"] for ep in episodes),
        "post_switch_episode_count": len(post_switch_episodes),
        "post_switch_total_cost_mean": mean(
            [ep["total_cost"] for ep in post_switch_episodes]
        ),
        "post_switch_raw_total_cost_mean": mean(
            [ep["raw_total_cost"] for ep in post_switch_episodes]
        ),
        "post_switch_raw_terminal_penalty_mean": mean(
            [ep["raw_terminal_penalty"] for ep in post_switch_episodes]
        ),
        "post_switch_raw_reasoning_cost_component_mean": mean(
            [ep["raw_reasoning_cost_component"] for ep in post_switch_episodes]
        ),
        "post_switch_algorithm_cumulative_total_cost": sum(
            ep["total_cost"] for ep in post_switch_episodes
        ),
        "post_switch_raw_algorithm_cumulative_total_cost": sum(
            ep["raw_total_cost"] for ep in post_switch_episodes
        ),
        "oracle_stationary_total_cost": oracle_summary["cumulative_total_cost"],
        "raw_oracle_stationary_total_cost": oracle_summary["raw_cumulative_total_cost"],
        "raw_outcome_penalty_cumulative": sum(ep["raw_outcome_penalty"] for ep in episodes),
        "raw_policy_penalty_cumulative": sum(ep["raw_policy_penalty"] for ep in episodes),
        "raw_terminal_penalty_cumulative": sum(ep["raw_terminal_penalty"] for ep in episodes),
        "legacy_raw_terminal_penalty_cumulative": sum(
            ep.get("legacy_raw_terminal_penalty", ep["raw_terminal_penalty"])
            for ep in episodes
        ),
        "raw_path_cost_component_cumulative": sum(ep["raw_path_cost_component"] for ep in episodes),
        "raw_reasoning_cost_component_cumulative": sum(ep["raw_reasoning_cost_component"] for ep in episodes),
        "raw_mode_mismatch_cost_component_cumulative": sum(
            ep.get("raw_mode_mismatch_cost_component", 0.0) for ep in episodes
        ),
        "mean_llm_call_count": mean([ep["llm_call_count"] for ep in episodes]),
        "mean_prompt_tokens": mean([ep["prompt_tokens_total"] for ep in episodes]),
        "mean_completion_tokens": mean(
            [ep["completion_tokens_total"] for ep in episodes]
        ),
        "mean_total_tokens": mean([ep["total_tokens_total"] for ep in episodes]),
        "cumulative_total_tokens": sum(ep["total_tokens_total"] for ep in episodes),
        "mean_api_cost_usd_raw": mean([ep["api_cost_total_usd_raw"] for ep in episodes]),
        "cumulative_api_cost_usd_raw": sum(
            ep["api_cost_total_usd_raw"] for ep in episodes
        ),
        "mean_generation_time_seconds": mean(
            [ep["generation_time_total_seconds"] for ep in episodes]
        ),
        "p50_generation_time_seconds": percentile(
            [ep["generation_time_total_seconds"] for ep in episodes],
            0.5,
        ),
        "p90_generation_time_seconds": percentile(
            [ep["generation_time_total_seconds"] for ep in episodes],
            0.9,
        ),
        "mean_llm_round_trip_seconds": mean(
            [ep["llm_round_trip_total_seconds"] for ep in episodes]
        ),
        "mean_episode_wall_clock_seconds": mean(
            [ep["episode_wall_clock_seconds"] for ep in episodes]
        ),
        "p50_episode_wall_clock_seconds": percentile(
            [ep["episode_wall_clock_seconds"] for ep in episodes],
            0.5,
        ),
        "p90_episode_wall_clock_seconds": percentile(
            [ep["episode_wall_clock_seconds"] for ep in episodes],
            0.9,
        ),
        "mean_tool_wall_clock_seconds": mean(
            [ep["tool_wall_clock_total_seconds"] for ep in episodes]
        ),
        "policy_action_violation_rate": mean(
            [float(ep["policy_action_violation"]) for ep in episodes]
        ),
        "policy_communication_violation_rate": mean(
            [float(ep["policy_communication_violation"]) for ep in episodes]
        ),
        "policy_nl_assertion_failure_rate": (
            policy_nl_failed / policy_nl_total if policy_nl_total else 0.0
        ),
        "mean_policy_violation_count": mean(
            [ep["policy_violation_count"] for ep in episodes]
        ),
        "subset_mismatch_count": sum(1 for ep in episodes if ep["subset_mismatch"]),
        "episodes_with_stage5_verification_tools": sum(1 for ep in episodes if ep["stage5_executed_tool_names"]),
        "shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in episodes]),
        "unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in episodes]),
        "mean_barrier_stop_depth": mean_present(
            [ep["barrier_stop_depth"] for ep in episodes]
        ),
        "first_private_barrier_stage_distribution": distribution_with_fraction(
            [ep["first_private_barrier_stage"] for ep in episodes]
        ),
        "mean_candidate_count_per_stage": mean_vector(
            [ep["candidate_count_per_stage"] for ep in episodes]
        ),
        "mean_legal_child_count_per_stage": mean_vector(
            [ep["legal_child_count_per_stage"] for ep in episodes]
        ),
        "specialist_task_count": sum(1 for ep in episodes if ep["is_specialist_task"]),
        "specialist_task_unshared_fraction": mean(
            [float(ep["selected_unshared_path"]) for ep in episodes if ep["is_specialist_task"]]
        ),
        "stage_source_summary": {k: dict(v) for k, v in stage_source_summary.items()},
        "reasoning_cost_mode_default": next(
            (ep["reasoning_cost_mode_default"] for ep in episodes if ep.get("reasoning_cost_mode_default")),
            None,
        ),
    }


def build_specialist_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    specialist = [ep for ep in episodes if ep["is_specialist_task"]]
    return {
        "specialist_episode_count": len(specialist),
        "specialist_shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in specialist]),
        "specialist_unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in specialist]),
        "specialist_exact_match_mean": mean([float(ep["exact_match"]) for ep in specialist]),
        "specialist_total_cost_mean": mean([ep["total_cost"] for ep in specialist]),
        "specialist_raw_outcome_penalty_mean": mean(
            [ep["raw_outcome_penalty"] for ep in specialist]
        ),
        "specialist_raw_policy_penalty_mean": mean(
            [ep["raw_policy_penalty"] for ep in specialist]
        ),
        "specialist_raw_terminal_penalty_mean": mean([ep["raw_terminal_penalty"] for ep in specialist]),
        "specialist_raw_path_cost_component_mean": mean([ep["raw_path_cost_component"] for ep in specialist]),
        "specialist_raw_reasoning_cost_component_mean": mean([ep["raw_reasoning_cost_component"] for ep in specialist]),
        "specialist_raw_reasoning_cost_component_api_mean": mean_present(
            [ep["raw_reasoning_cost_component_api"] for ep in specialist]
        ),
        "specialist_raw_reasoning_cost_component_token_mean": mean_present(
            [ep["raw_reasoning_cost_component_token"] for ep in specialist]
        ),
        "specialist_task_ids": sorted({ep["original_task_id"] for ep in specialist}),
    }


def build_partial_summary(
    *,
    method: str,
    dataset: str,
    repeats: int,
    model: str,
    family_kind: str,
    executor_name: str,
    schedule_mode: str,
    oracle_summary: dict[str, Any],
    episodes: list[dict[str, Any]],
    total_episodes: int,
    status: str = "running",
) -> dict[str, Any]:
    summary = build_summary(
        method=method,
        dataset=dataset,
        repeats=repeats,
        model=model,
        family_kind=family_kind,
        executor_name=executor_name,
        schedule_mode=schedule_mode,
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
            "completed_raw_policy_penalty": sum(ep["raw_policy_penalty"] for ep in episodes),
            "completed_total_tokens": sum(ep["total_tokens_total"] for ep in episodes),
            "completed_api_cost_usd_raw": sum(
                ep["api_cost_total_usd_raw"] for ep in episodes
            ),
        }
    )
    return summary


def build_risky_dynamics_rows(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "episode_index": ep["episode_index"],
            "repeat_index": ep["repeat_index"],
            "position_in_cycle": ep["position_in_cycle"],
            "dataset_index": ep["dataset_index"],
            "original_task_id": ep["original_task_id"],
            "is_specialist_task": ep["is_specialist_task"],
            "selected_shared_path": ep["selected_shared_path"],
            "selected_unshared_path": ep["selected_unshared_path"],
            "cumulative_shared_path_ratio": ep["cumulative_shared_path_ratio"],
            "rolling_shared_path_ratio_last10": ep["rolling_shared_path_ratio_last10"],
            "shared_branch_triggered": ep["shared_branch_triggered"],
            "unshared_branch_triggered": ep["unshared_branch_triggered"],
            "shared_update_count": ep["shared_update_count"],
            "cumulative_shared_update_count": ep["cumulative_shared_update_count"],
            "unshared_edge_update_count": ep["unshared_edge_update_count"],
            "cumulative_unshared_edge_update_count": ep["cumulative_unshared_edge_update_count"],
            "selected_path": ep["selected_path"],
            "selected_shared_path_nodes": ep["selected_path"] if ep["selected_shared_path"] else [],
            "selected_unshared_path_nodes": ep["selected_path"] if ep["selected_unshared_path"] else [],
            "raw_terminal_penalty": ep["raw_terminal_penalty"],
            "total_cost": ep["total_cost"],
        }
        for ep in episodes
    ]


def summarize_window(episodes: list[dict[str, Any]], *, label: str, start: int, end: int) -> dict[str, Any]:
    window = episodes[start:end]
    return {
        "label": label,
        "start_episode_index": start,
        "end_episode_index_exclusive": end,
        "episode_count": len(window),
        "shared_path_fraction": mean([float(ep["selected_shared_path"]) for ep in window]),
        "unshared_path_fraction": mean([float(ep["selected_unshared_path"]) for ep in window]),
        "mean_shared_update_count_per_episode": mean([ep["shared_update_count"] for ep in window]),
        "mean_raw_terminal_penalty": mean([ep["raw_terminal_penalty"] for ep in window]),
        "mean_total_cost": mean([ep["total_cost"] for ep in window]),
    }


def build_risky_dynamics_payload(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "episodes": build_risky_dynamics_rows(episodes),
        "window_summaries": {
            "first20": summarize_window(episodes, label="first20", start=0, end=20),
            "middle20": summarize_window(episodes, label="middle20", start=40, end=60),
            "last20": summarize_window(episodes, label="last20", start=80, end=100),
        },
    }


def build_compare_rows(summaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            "method": summary["method"],
            "total_cost_mean": summary["total_cost_mean"],
            "raw_total_cost_mean": summary["raw_total_cost_mean"],
            "raw_total_cost_api_mean": summary["raw_total_cost_api_mean"],
            "raw_total_cost_token_mean": summary["raw_total_cost_token_mean"],
            "raw_outcome_penalty_mean": summary["raw_outcome_penalty_mean"],
            "raw_policy_penalty_mean": summary["raw_policy_penalty_mean"],
            "raw_terminal_penalty_mean": summary["raw_terminal_penalty_mean"],
            "legacy_raw_terminal_penalty_mean": summary.get(
                "legacy_raw_terminal_penalty_mean",
                summary["raw_terminal_penalty_mean"],
            ),
            "raw_terminal_penalty_exec_clean_v4_mean": summary.get(
                "raw_terminal_penalty_exec_clean_v4_mean",
                0.0,
            ),
            "raw_path_cost_component_mean": summary["raw_path_cost_component_mean"],
            "raw_reasoning_cost_component_mean": summary["raw_reasoning_cost_component_mean"],
            "raw_mode_mismatch_cost_component_mean": summary.get(
                "raw_mode_mismatch_cost_component_mean",
                0.0,
            ),
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
        }
        for summary in summaries
    ]
    return sorted(rows, key=lambda row: (row["total_cost_mean"], row["method"]))


def compare_rows_to_markdown(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return ""
    fieldnames = list(rows[0].keys())
    header = "| " + " | ".join(fieldnames) + " |"
    divider = "| " + " | ".join("---" for _ in fieldnames) + " |"
    body = [
        "| " + " | ".join(f"{row[field]}" for field in fieldnames) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body]) + "\n"


def build_specialist_hit_analysis(
    *,
    merged_episodes_by_method: dict[str, list[dict[str, Any]]],
    specialist_task_ids: set[str],
    schedule_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    specialist_schedule_rows = [
        row for row in schedule_rows if row["original_task_id"] in specialist_task_ids
    ]
    specialist_episode_count = len(specialist_schedule_rows)
    specialist_task_hit_ids = sorted({row["original_task_id"] for row in specialist_schedule_rows})
    payload: dict[str, Any] = {
        "specialist_episode_count": specialist_episode_count,
        "specialist_task_hit_ids": specialist_task_hit_ids,
        "schedule_episode_indices": [row["episode_index"] for row in specialist_schedule_rows],
    }
    if specialist_episode_count == 0:
        payload["methods"] = {}
        return payload

    method_payload: dict[str, Any] = {}
    for method, episodes in merged_episodes_by_method.items():
        specialist_eps = [ep for ep in episodes if ep["is_specialist_task"]]
        method_payload[method] = {
            "specialist_episode_count": len(specialist_eps),
            "specialist_unshared_path_fraction": mean(
                [float(ep["selected_unshared_path"]) for ep in specialist_eps]
            ),
            "specialist_shared_path_fraction": mean(
                [float(ep["selected_shared_path"]) for ep in specialist_eps]
            ),
            "specialist_total_cost_mean": mean([ep["total_cost"] for ep in specialist_eps]),
            "specialist_raw_terminal_penalty_mean": mean(
                [ep["raw_terminal_penalty"] for ep in specialist_eps]
            ),
        }
    payload["methods"] = method_payload
    return payload


def ensure_model_env(required: bool = True) -> str:
    model_name = os.environ.get("PSAGENT_LLM_BENCH_MODEL", "")
    if required and model_name != MODEL_REQUIRED:
        raise SystemExit(f"PSAGENT_LLM_BENCH_MODEL must be {MODEL_REQUIRED!r}; got {model_name!r}")
    return model_name


def resolve_model_name_for_executor(executor_name: str) -> str:
    if executor_name == "llm_bench":
        return ensure_model_env(required=True)
    if executor_name == "simulated":
        return "simulated"
    raise SystemExit(f"Unsupported executor_name: {executor_name!r}")


def _assert_existing_run_compatible(
    *,
    run_config: dict[str, Any],
    data_path: Path,
    repeats: int,
    methods: list[str],
    family_kind: str,
    schedule_mode: str,
    switch_denominator: int,
    schedule_buckets_path: Path | None,
    trap_switch_cycle_source: str,
    policy_kwargs_by_method: dict[str, dict[str, Any]],
    common_eta_override: float | None,
    common_epsilon_override: float | None,
    ps_eta_shared_override: float | None,
    ps_loss_clip: float | None,
    ps_prob_floor: float | None,
    executor_name: str,
    post_switch_fixed_layer1_probs: bool,
    post_switch_fixed_tree_probs: bool,
) -> None:
    mismatches: list[str] = []
    requested_freeze_mode = resolve_post_switch_probability_freeze_mode(
        post_switch_fixed_layer1_probs=post_switch_fixed_layer1_probs,
        post_switch_fixed_tree_probs=post_switch_fixed_tree_probs,
    )
    if str(run_config.get("dataset")) != str(data_path):
        mismatches.append(
            f"dataset existing={run_config.get('dataset')} requested={data_path}"
        )
    if int(run_config.get("repeats", repeats)) != repeats:
        mismatches.append(
            f"repeats existing={run_config.get('repeats')} requested={repeats}"
        )
    existing_methods = list(run_config.get("methods", []))
    if existing_methods and existing_methods != list(methods):
        mismatches.append(
            f"methods existing={existing_methods} requested={list(methods)}"
        )
    if str(run_config.get("family_kind", DEFAULT_FAMILY_KIND)) != family_kind:
        mismatches.append(
            f"family_kind existing={run_config.get('family_kind')} requested={family_kind}"
        )
    if str(run_config.get("executor_name", DEFAULT_EXECUTOR_NAME)) != executor_name:
        mismatches.append(
            f"executor_name existing={run_config.get('executor_name')} requested={executor_name}"
        )
    if bool(run_config.get("post_switch_fixed_layer1_probs", False)) != bool(
        post_switch_fixed_layer1_probs
    ):
        mismatches.append(
            "post_switch_fixed_layer1_probs "
            f"existing={run_config.get('post_switch_fixed_layer1_probs', False)} "
            f"requested={post_switch_fixed_layer1_probs}"
        )
    if bool(run_config.get("post_switch_fixed_tree_probs", False)) != bool(
        post_switch_fixed_tree_probs
    ):
        mismatches.append(
            "post_switch_fixed_tree_probs "
            f"existing={run_config.get('post_switch_fixed_tree_probs', False)} "
            f"requested={post_switch_fixed_tree_probs}"
        )
    existing_freeze_mode = run_config.get("post_switch_probability_freeze_mode")
    if existing_freeze_mode is None and bool(run_config.get("post_switch_fixed_layer1_probs", False)):
        existing_freeze_mode = POST_SWITCH_FREEZE_LAYER1_MODE
    if existing_freeze_mode != requested_freeze_mode:
        mismatches.append(
            "post_switch_probability_freeze_mode "
            f"existing={existing_freeze_mode} requested={requested_freeze_mode}"
        )
    if str(run_config.get("schedule_mode", SCHEDULE_MODE_STATIONARY)) != schedule_mode:
        mismatches.append(
            f"schedule_mode existing={run_config.get('schedule_mode')} requested={schedule_mode}"
        )
    existing_switch_denominator = run_config.get("switch_denominator")
    if schedule_mode in {SCHEDULE_MODE_TRAP_SWITCH, SCHEDULE_MODE_TRAP_ONLY_RANDOM}:
        if int(existing_switch_denominator or switch_denominator) != switch_denominator:
            mismatches.append(
                "switch_denominator "
                f"existing={existing_switch_denominator} requested={switch_denominator}"
            )
        existing_schedule_buckets = run_config.get("schedule_buckets")
        requested_schedule_buckets = str(schedule_buckets_path) if schedule_buckets_path else None
        if str(existing_schedule_buckets) != str(requested_schedule_buckets):
            mismatches.append(
                "schedule_buckets "
                f"existing={existing_schedule_buckets} requested={requested_schedule_buckets}"
            )
        if schedule_mode == SCHEDULE_MODE_TRAP_SWITCH:
            existing_cycle_source = run_config.get(
                "trap_switch_cycle_source",
                run_config.get("schedule_metadata", {}).get(
                    "trap_switch_cycle_source",
                    TRAP_SWITCH_CYCLE_SOURCE_BUCKET,
                ),
            )
            if str(existing_cycle_source) != str(trap_switch_cycle_source):
                mismatches.append(
                    "trap_switch_cycle_source "
                    f"existing={existing_cycle_source} requested={trap_switch_cycle_source}"
                )
    existing_common_eta = run_config.get("common_eta_override")
    if existing_common_eta != common_eta_override:
        mismatches.append(
            f"common_eta_override existing={existing_common_eta} requested={common_eta_override}"
        )
    existing_common_epsilon = run_config.get("common_epsilon_override")
    if existing_common_epsilon != common_epsilon_override:
        mismatches.append(
            "common_epsilon_override "
            f"existing={existing_common_epsilon} requested={common_epsilon_override}"
        )
    if run_config.get("ps_eta_shared_override") != ps_eta_shared_override:
        mismatches.append(
            "ps_eta_shared_override "
            f"existing={run_config.get('ps_eta_shared_override')} requested={ps_eta_shared_override}"
        )
    if run_config.get("ps_loss_clip") != ps_loss_clip:
        mismatches.append(
            f"ps_loss_clip existing={run_config.get('ps_loss_clip')} requested={ps_loss_clip}"
        )
    if run_config.get("ps_prob_floor") != ps_prob_floor:
        mismatches.append(
            f"ps_prob_floor existing={run_config.get('ps_prob_floor')} requested={ps_prob_floor}"
        )
    existing_policy_kwargs = run_config.get("policy_kwargs_by_method")
    if existing_policy_kwargs is None and common_eta_override is None and common_epsilon_override is None:
        existing_policy_kwargs = policy_kwargs_by_method
    if existing_policy_kwargs != policy_kwargs_by_method:
        mismatches.append(
            "policy_kwargs_by_method "
            f"existing={existing_policy_kwargs} requested={policy_kwargs_by_method}"
        )
    if mismatches:
        raise RuntimeError(
            "Existing run directory is incompatible with requested setup: "
            + "; ".join(mismatches)
        )


def load_run_context(run_dir: Path) -> dict[str, Any]:
    run_config = load_json(run_dir / "run_config.json")
    schedule_rows = load_json(run_dir / "schedule.json")
    oracle_summary = load_json(run_dir / "stationary_oracle_summary.json")
    instances = load_instances(Path(run_config["dataset"]))
    selected = materialize_schedule(instances, schedule_rows)
    specialist_path = run_dir / "specialist_task_ids.json"
    if specialist_path.exists():
        specialist_task_ids = {str(value) for value in load_json(specialist_path)}
    else:
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
    methods: list[str],
    model_name: str,
    family_kind: str,
    schedule_mode: str,
    switch_denominator: int,
    schedule_buckets_path: Path | None,
    trap_switch_cycle_source: str,
    common_eta_override: float | None,
    common_epsilon_override: float | None,
    ps_eta_shared_override: float | None,
    ps_loss_clip: float | None,
    ps_prob_floor: float | None,
    executor_name: str,
    post_switch_fixed_layer1_probs: bool,
    post_switch_fixed_tree_probs: bool,
) -> Path:
    validate_methods(methods)
    post_switch_probability_freeze_mode = resolve_post_switch_probability_freeze_mode(
        post_switch_fixed_layer1_probs=post_switch_fixed_layer1_probs,
        post_switch_fixed_tree_probs=post_switch_fixed_tree_probs,
    )
    policy_kwargs_by_method = build_policy_kwargs_by_method(
        methods,
        common_eta_override=common_eta_override,
        common_epsilon_override=common_epsilon_override,
        ps_eta_shared_override=ps_eta_shared_override,
        ps_loss_clip=ps_loss_clip,
        ps_prob_floor=ps_prob_floor,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    run_config_path = output_dir / "run_config.json"
    schedule_path = output_dir / "schedule.json"
    oracle_path = output_dir / "stationary_oracle_summary.json"
    specialist_path = output_dir / "specialist_task_ids.json"

    if run_config_path.exists() and schedule_path.exists() and oracle_path.exists():
        _assert_existing_run_compatible(
            run_config=load_json(run_config_path),
            data_path=data_path,
            repeats=repeats,
            methods=methods,
            family_kind=family_kind,
            schedule_mode=schedule_mode,
            switch_denominator=switch_denominator,
            schedule_buckets_path=schedule_buckets_path,
            trap_switch_cycle_source=trap_switch_cycle_source,
            policy_kwargs_by_method=policy_kwargs_by_method,
            common_eta_override=common_eta_override,
            common_epsilon_override=common_epsilon_override,
            ps_eta_shared_override=ps_eta_shared_override,
            ps_loss_clip=ps_loss_clip,
            ps_prob_floor=ps_prob_floor,
            executor_name=executor_name,
            post_switch_fixed_layer1_probs=post_switch_fixed_layer1_probs,
            post_switch_fixed_tree_probs=post_switch_fixed_tree_probs,
        )
        return output_dir

    instances = load_instances(data_path)
    schedule_buckets = load_schedule_buckets(schedule_buckets_path)
    selected, schedule_metadata = build_schedule_selection(
        instances,
        repeats=repeats,
        schedule_mode=schedule_mode,
        switch_denominator=switch_denominator,
        schedule_buckets=schedule_buckets,
        trap_switch_cycle_source=trap_switch_cycle_source,
    )
    oracle_summary = compute_stationary_oracle(selected, family_kind=family_kind)
    schedule_rows = serialize_schedule(selected)
    specialist_task_ids = sorted(
        load_specialist_task_ids(
            schedule_buckets
            if schedule_mode in {SCHEDULE_MODE_TRAP_SWITCH, SCHEDULE_MODE_TRAP_ONLY_RANDOM}
            else None
        )
    )

    write_json(
        run_config_path,
        {
            "created_at": datetime.now().isoformat(),
            "dataset": str(data_path),
            "dataset_indices": sorted({row["dataset_index"] for row in selected}),
            "repeats": repeats,
            "horizon": len(selected),
            "family_kind": family_kind,
            "executor_name": executor_name,
            "schedule_mode": schedule_mode,
            "switch_denominator": (
                switch_denominator
                if schedule_mode in {SCHEDULE_MODE_TRAP_SWITCH, SCHEDULE_MODE_TRAP_ONLY_RANDOM}
                else None
            ),
            "schedule_buckets": (
                str(schedule_buckets_path) if schedule_buckets_path is not None else None
            ),
            "trap_switch_cycle_source": (
                trap_switch_cycle_source
                if schedule_mode == SCHEDULE_MODE_TRAP_SWITCH
                else None
            ),
            "schedule_metadata": schedule_metadata,
            "model": model_name,
            "seed": SEED,
            "methods": methods,
            "policy_kwargs_by_method": policy_kwargs_by_method,
            "common_eta_override": common_eta_override,
            "common_epsilon_override": common_epsilon_override,
            "ps_eta_shared_override": ps_eta_shared_override,
            "ps_loss_clip": ps_loss_clip,
            "ps_prob_floor": ps_prob_floor,
            "post_switch_fixed_layer1_probs": post_switch_fixed_layer1_probs,
            "post_switch_fixed_tree_probs": post_switch_fixed_tree_probs,
            "post_switch_probability_freeze_mode": post_switch_probability_freeze_mode,
            "post_switch_layer1_freeze_mode": (
                POST_SWITCH_FREEZE_LAYER1_MODE
                if post_switch_fixed_layer1_probs
                else None
            ),
            "parallelism": "method_only",
        },
    )
    write_json(schedule_path, schedule_rows)
    write_json(oracle_path, oracle_summary)
    write_json(specialist_path, specialist_task_ids)
    return output_dir


def build_progress_payload(
    *,
    method: str,
    completed_count: int,
    total_episodes: int,
    model: str,
    status: str,
) -> dict[str, Any]:
    last_completed = completed_count - 1 if completed_count else None
    return {
        "method": method,
        "scheduled_episodes": total_episodes,
        "completed_episodes": completed_count,
        "last_completed_episode_index": last_completed,
        "status": status,
        "model": model,
        "updated_at": datetime.now().isoformat(),
    }


def persist_method_state(
    *,
    method_dir: Path,
    method: str,
    episodes: list[dict[str, Any]],
    policy: Any | None,
    total_episodes: int,
    model: str,
    dataset: str,
    repeats: int,
    family_kind: str,
        executor_name: str,
    schedule_mode: str,
    oracle_summary: dict[str, Any],
) -> None:
    add_cumulative_fields(episodes)
    checkpoint_payload = {
        "method": method,
        "completed_count": len(episodes),
        "episodes": episodes,
        "model": model,
        "policy": policy,
    }
    write_bytes_atomic(method_dir / "checkpoint.pkl", pickle.dumps(checkpoint_payload))
    write_jsonl(method_dir / "episodes.partial.jsonl", episodes)
    write_json(
        method_dir / "progress.json",
        build_progress_payload(
            method=method,
            completed_count=len(episodes),
            total_episodes=total_episodes,
            model=model,
            status="complete" if len(episodes) == total_episodes else "running",
        ),
    )
    partial_summary = build_partial_summary(
        method=method,
        dataset=dataset,
        repeats=repeats,
        model=model,
        family_kind=family_kind,
        executor_name=executor_name,
        schedule_mode=schedule_mode,
        oracle_summary=oracle_summary,
        episodes=episodes,
        total_episodes=total_episodes,
        status="complete" if len(episodes) == total_episodes else "running",
    )
    write_json(method_dir / "summary_partial.json", partial_summary)
    if len(episodes) == total_episodes:
        write_json(method_dir / "episodes.json", episodes)
        write_json(method_dir / "summary.json", partial_summary)
        write_json(method_dir / "summary_with_oracle.json", partial_summary)


def load_method_checkpoint(method_dir: Path) -> dict[str, Any] | None:
    checkpoint_path = method_dir / "checkpoint.pkl"
    if not checkpoint_path.exists():
        return None
    with checkpoint_path.open("rb") as handle:
        return pickle.load(handle)


def run_policy_method(
    *,
    run_dir: Path,
    method: str,
) -> None:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    resolve_model_name_for_executor(str(run_config.get("executor_name", DEFAULT_EXECUTOR_NAME)))
    selected = context["selected"]
    oracle_summary = context["oracle_summary"]
    specialist_task_ids = context["specialist_task_ids"]
    total_episodes = len(selected)
    method_dir = run_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)

    env = build_env(
        executor_name=str(run_config.get("executor_name", DEFAULT_EXECUTOR_NAME)),
        family_kind=str(run_config.get("family_kind", DEFAULT_FAMILY_KIND)),
    )
    checkpoint = load_method_checkpoint(method_dir)
    if checkpoint is not None:
        policy = checkpoint["policy"]
        episodes = list(checkpoint["episodes"])
        model = checkpoint.get("model", getattr(env.family_executor, "model", MODEL_REQUIRED))
    else:
        episodes = []
        model = str(run_config.get("model", getattr(env.family_executor, "model", MODEL_REQUIRED)))
        policy_kwargs_by_method = run_config.get("policy_kwargs_by_method", {})
        policy_kwargs = dict(policy_kwargs_by_method.get(method, {}))
        policy = POLICY_REGISTRY[method](seed=SEED, **policy_kwargs)
        policy.bind_env(env)
        policy.reset()
    post_switch_probability_freeze_mode = run_config.get("post_switch_probability_freeze_mode")
    if post_switch_probability_freeze_mode is None and bool(
        run_config.get("post_switch_fixed_layer1_probs", False)
    ):
        post_switch_probability_freeze_mode = POST_SWITCH_FREEZE_LAYER1_MODE
    if post_switch_probability_freeze_mode is not None:
        install_post_switch_probability_freeze(policy)

    completed_count = len(episodes)
    if completed_count >= total_episodes:
        persist_method_state(
            method_dir=method_dir,
            method=method,
            episodes=episodes,
            policy=policy,
            total_episodes=total_episodes,
            model=model,
            dataset=run_config["dataset"],
            repeats=int(run_config["repeats"]),
            family_kind=str(run_config.get("family_kind", DEFAULT_FAMILY_KIND)),
            executor_name=str(run_config.get("executor_name", DEFAULT_EXECUTOR_NAME)),
            schedule_mode=str(run_config.get("schedule_mode", SCHEDULE_MODE_STATIONARY)),
            oracle_summary=oracle_summary,
        )
        return

    for local_offset in range(completed_count, total_episodes):
        row = selected[local_offset]
        episode_index = int(row["episode_index"])
        runtime_instance = attach_schedule_metadata(row["instance"], row)
        if post_switch_probability_freeze_mode is not None:
            switch_episode = int(
                (run_config.get("schedule_metadata") or {}).get(
                    "switch_episode",
                    max(1, total_episodes // int(run_config.get("switch_denominator") or 3)),
                )
            )
            if episode_index == switch_episode and not getattr(
                policy,
                "_post_switch_probability_freeze_active",
                False,
            ):
                frozen, snapshot_rows = snapshot_child_distributions(
                    policy=policy,
                    env=env,
                    method=method,
                    episode_index=episode_index,
                    freeze_mode=str(post_switch_probability_freeze_mode),
                )
                policy._post_switch_frozen_child_probs_by_prefix = frozen
                policy._post_switch_probability_freeze_active = True
                policy._post_switch_probability_freeze_mode = post_switch_probability_freeze_mode
                write_json(method_dir / "post_switch_probability_snapshot.json", snapshot_rows)
                write_csv(method_dir / "post_switch_probability_snapshot.csv", snapshot_rows)
                if post_switch_probability_freeze_mode == POST_SWITCH_FREEZE_TREE_MODE:
                    write_json(method_dir / "post_switch_tree_probability_snapshot.json", snapshot_rows)
                    write_csv(method_dir / "post_switch_tree_probability_snapshot.csv", snapshot_rows)
                elif post_switch_probability_freeze_mode == POST_SWITCH_FREEZE_LAYER1_MODE:
                    write_json(method_dir / "post_switch_layer1_probability_snapshot.json", snapshot_rows)
                    write_csv(method_dir / "post_switch_layer1_probability_snapshot.csv", snapshot_rows)
        print(
            f"[run] method={method} episode={episode_index + 1}/{len(selected)} "
            f"repeat={row['repeat_index'] + 1} pos={row['position_in_cycle']} dataset_index={row['dataset_index']}",
            flush=True,
        )
        trap_probability_info = trap_route_probability_snapshot(policy, env)
        path = policy.select_path(runtime_instance, env)
        selection_info = policy.get_last_selection_info() if hasattr(policy, "get_last_selection_info") else {}
        if isinstance(selection_info, dict):
            selection_info = dict(selection_info)
            selection_info.update(trap_probability_info)
            freeze_active = bool(
                getattr(policy, "_post_switch_probability_freeze_active", False)
            )
            active_freeze_mode = (
                getattr(policy, "_post_switch_probability_freeze_mode", None)
                if freeze_active
                else None
            )
            selection_info["post_switch_probability_freeze_active"] = freeze_active
            selection_info["post_switch_probability_freeze_mode"] = active_freeze_mode
            selection_info["post_switch_layer1_freeze_active"] = bool(
                freeze_active and active_freeze_mode == POST_SWITCH_FREEZE_LAYER1_MODE
            )
            selection_info["post_switch_layer1_freeze_mode"] = (
                POST_SWITCH_FREEZE_LAYER1_MODE
                if freeze_active and active_freeze_mode == POST_SWITCH_FREEZE_LAYER1_MODE
                else None
            )
        env.reset(runtime_instance)
        result = env.run_path(path)
        policy.update(result)
        state = policy.get_state() if hasattr(policy, "get_state") else {}
        update_info = state.get("last_update_info", {}) if isinstance(state, dict) else {}
        episodes.append(
            flatten_episode(
                episode_index=episode_index,
                row=row,
                result=result,
                method=method,
                oracle_summary=oracle_summary,
                selection_info=selection_info if isinstance(selection_info, dict) else {},
                update_info=update_info if isinstance(update_info, dict) else {},
                specialist_task_ids=specialist_task_ids,
            )
        )
        persist_method_state(
            method_dir=method_dir,
            method=method,
            episodes=episodes,
            policy=policy,
            total_episodes=total_episodes,
            model=model,
            dataset=run_config["dataset"],
            repeats=int(run_config["repeats"]),
            family_kind=str(run_config.get("family_kind", DEFAULT_FAMILY_KIND)),
            executor_name=str(run_config.get("executor_name", DEFAULT_EXECUTOR_NAME)),
            schedule_mode=str(run_config.get("schedule_mode", SCHEDULE_MODE_STATIONARY)),
            oracle_summary=oracle_summary,
        )


def merge_method_results(run_dir: Path, method: str) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    oracle_summary = context["oracle_summary"]
    specialist_task_ids = context["specialist_task_ids"]
    total_episodes = int(run_config["horizon"])
    method_dir = run_dir / method
    model = run_config["model"]
    progress = load_json(method_dir / "progress.json")
    if progress["completed_episodes"] != total_episodes:
        raise RuntimeError(f"Method {method} is incomplete: {progress}")
    merged_episodes = load_json(method_dir / "episodes.json")
    model = progress.get("model", model)

    expected_indices = list(range(total_episodes))
    actual_indices = [int(row["episode_index"]) for row in merged_episodes]
    if actual_indices != expected_indices:
        raise RuntimeError(
            f"Merged episode indices mismatch for {method}. "
            f"expected={expected_indices[:3]}...{expected_indices[-3:]}, "
            f"actual={actual_indices[:3]}...{actual_indices[-3:]}"
        )
    add_cumulative_fields(merged_episodes)
    summary = build_summary(
        method=method,
        dataset=run_config["dataset"],
        repeats=int(run_config["repeats"]),
        model=model,
        family_kind=str(run_config.get("family_kind", DEFAULT_FAMILY_KIND)),
        executor_name=str(run_config.get("executor_name", DEFAULT_EXECUTOR_NAME)),
        schedule_mode=str(run_config.get("schedule_mode", SCHEDULE_MODE_STATIONARY)),
        oracle_summary=oracle_summary,
        episodes=merged_episodes,
    )
    specialist_summary = build_specialist_summary(merged_episodes)

    merged_dir = method_dir / "merged"
    merged_dir.mkdir(parents=True, exist_ok=True)

    write_json(method_dir / "episodes.json", merged_episodes)
    write_json(method_dir / "summary.json", summary)
    write_json(method_dir / "summary_with_oracle.json", summary)
    write_json(method_dir / "specialist_summary.json", specialist_summary)
    write_json(merged_dir / "episodes.json", merged_episodes)
    write_json(merged_dir / "summary.json", summary)
    write_json(merged_dir / "summary_with_oracle.json", summary)
    write_json(merged_dir / "specialist_summary.json", specialist_summary)
    write_text_atomic(
        method_dir / "smoke_summary.md",
        json.dumps({"summary": summary, "specialist_summary": specialist_summary}, ensure_ascii=False, indent=2),
    )

    if method in {
        "risky_ps_old",
        "risky_ps",
        "risky_ps_ix",
        "risky_ps_safe_conditional",
        "risky_ps_safe_conditional_ix",
        "risky_ps_direct_cost",
    }:
        dynamics_payload = build_risky_dynamics_payload(merged_episodes)
        write_json(run_dir / f"{method}_shared_unshared_dynamics.json", dynamics_payload)
        write_csv(run_dir / f"{method}_shared_unshared_dynamics.csv", dynamics_payload["episodes"])

    return {
        "summary": summary,
        "specialist_summary": specialist_summary,
        "episodes": merged_episodes,
        "specialist_task_ids": specialist_task_ids,
    }


def merge_all_results(run_dir: Path) -> dict[str, Any]:
    context = load_run_context(run_dir)
    run_config = context["run_config"]
    specialist_task_ids = context["specialist_task_ids"]
    summaries: list[dict[str, Any]] = []
    merged_episodes_by_method: dict[str, list[dict[str, Any]]] = {}

    for method in run_config["methods"]:
        method_summary = load_json(run_dir / method / "summary_with_oracle.json")
        summaries.append(method_summary)
        merged_episodes_by_method[method] = load_json(run_dir / method / "episodes.json")

    compare_rows = build_compare_rows(summaries)
    write_json(run_dir / "repeated_smoke_compare.json", compare_rows)
    write_csv(run_dir / "repeated_smoke_compare.csv", compare_rows)
    write_text_atomic(run_dir / "repeated_smoke_compare.md", compare_rows_to_markdown(compare_rows))

    specialist_payload = build_specialist_hit_analysis(
        merged_episodes_by_method=merged_episodes_by_method,
        specialist_task_ids=specialist_task_ids,
        schedule_rows=context["schedule_rows"],
    )
    write_json(run_dir / "specialist_unshared_hit_analysis.json", specialist_payload)
    return {
        "compare_rows": compare_rows,
        "specialist_analysis": specialist_payload,
        "merged_episodes_by_method": merged_episodes_by_method,
    }


def wandb_mode_pattern(ep: dict[str, Any]) -> str:
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


def wandb_profile_match_labels(ep: dict[str, Any]) -> tuple[str, str]:
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


def build_wandb_episode_analysis_fields(ep: dict[str, Any]) -> dict[str, Any]:
    mode_pattern = wandb_mode_pattern(ep)
    deep_count = mode_pattern.count("d")
    fast_count = mode_pattern.count("f")
    match_group, mismatch_direction = wandb_profile_match_labels(ep)
    task_bucket = str(ep.get("task_bucket") or "unknown")
    fdddd_group = "fdddd" if mode_pattern == "fdddd" else "non_fdddd"
    return {
        "task_bucket": task_bucket,
        "schedule_phase": ep.get("schedule_phase"),
        "family_task_bucket": ep.get("family_task_bucket"),
        "terminal_majority_pair": ep.get("terminal_majority_pair"),
        "match_group": match_group,
        "mismatch_direction": mismatch_direction,
        "mode_pattern": mode_pattern,
        "path_mode_pattern": mode_pattern,
        "path_deep_count": deep_count,
        "path_fast_count": fast_count,
        "path_depth_balance": deep_count - fast_count,
        "path_is_fdddd": float(mode_pattern == "fdddd"),
        "fdddd_group": fdddd_group,
        "oracle_action": ep.get("oracle_action"),
        "final_action": ep.get("final_action"),
        "original_task_id": ep.get("original_task_id"),
        "family_behavior_archetype": ep.get("family_behavior_archetype"),
    }


def wandb_analysis_groups(row: dict[str, Any]) -> list[str]:
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


def add_wandb_group_aggregates(rows: list[dict[str, Any]]) -> None:
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
        groups = wandb_analysis_groups(row)
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


def build_wandb_episode_rows(
    *,
    method: str,
    episodes: list[dict[str, Any]],
    seed: int,
    run_group: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    cumulative_raw_total = 0.0
    cumulative_total_cost = 0.0
    post_switch_count = 0
    post_switch_cumulative_raw_total = 0.0
    post_switch_cumulative_total_cost = 0.0
    previous_cumulative_total_per_episode: float | None = None
    previous_cumulative_raw_total_per_episode: float | None = None
    previous_post_total_per_episode: float | None = None
    previous_post_raw_per_episode: float | None = None
    for ep in episodes:
        raw_total = float(ep["raw_total_cost"])
        total_cost = float(ep["total_cost"])
        cumulative_raw_total += raw_total
        cumulative_total_cost += total_cost
        is_post_switch = ep.get("schedule_phase") == "target_post_switch"
        if is_post_switch:
            post_switch_count += 1
            post_switch_cumulative_raw_total += raw_total
            post_switch_cumulative_total_cost += total_cost
        episode_1based = int(ep["episode_index"]) + 1
        cumulative_raw_total_per_episode = cumulative_raw_total / max(1, episode_1based)
        cumulative_total_cost_per_episode = cumulative_total_cost / max(1, episode_1based)
        post_raw_per_episode = (
            post_switch_cumulative_raw_total / post_switch_count
            if post_switch_count
            else None
        )
        post_total_per_episode = (
            post_switch_cumulative_total_cost / post_switch_count
            if post_switch_count
            else None
        )
        row = {
            "method": method,
            "seed": seed,
            "episode_index": int(ep["episode_index"]),
            "episode_1based": episode_1based,
            "repeat_index": ep.get("repeat_index"),
            "position_in_cycle": ep.get("position_in_cycle"),
            "dataset_index": ep.get("dataset_index"),
            "instance_id": ep.get("instance_id"),
            "cumulative_raw_total": cumulative_raw_total,
            "cumulative_total_cost": cumulative_total_cost,
            "cumulative_raw_total_per_episode": cumulative_raw_total_per_episode,
            "cumulative_total_cost_per_episode": cumulative_total_cost_per_episode,
            "cumulative_raw_total_per_episode_growth": (
                0.0
                if previous_cumulative_raw_total_per_episode is None
                else cumulative_raw_total_per_episode
                - previous_cumulative_raw_total_per_episode
            ),
            "cumulative_total_cost_per_episode_growth": (
                0.0
                if previous_cumulative_total_per_episode is None
                else cumulative_total_cost_per_episode
                - previous_cumulative_total_per_episode
            ),
            "post_switch_episode_count": post_switch_count,
            "post_switch_cumulative_raw_total": post_switch_cumulative_raw_total,
            "post_switch_cumulative_total_cost": post_switch_cumulative_total_cost,
            "post_switch_cumulative_raw_total_per_episode": post_raw_per_episode,
            "post_switch_cumulative_total_cost_per_episode": post_total_per_episode,
            "post_switch_cumulative_raw_total_per_episode_growth": (
                None
                if post_raw_per_episode is None
                else 0.0
                if previous_post_raw_per_episode is None
                else post_raw_per_episode - previous_post_raw_per_episode
            ),
            "post_switch_cumulative_total_cost_per_episode_growth": (
                None
                if post_total_per_episode is None
                else 0.0
                if previous_post_total_per_episode is None
                else post_total_per_episode - previous_post_total_per_episode
            ),
            "raw_total_cost": raw_total,
            "total_cost": total_cost,
            "raw_terminal_penalty": float(ep["raw_terminal_penalty"]),
            "raw_reasoning_cost_component": float(ep["raw_reasoning_cost_component"]),
            "raw_path_cost_component": float(ep["raw_path_cost_component"]),
            "raw_mode_mismatch_cost_component": float(
                ep.get("raw_mode_mismatch_cost_component", 0.0) or 0.0
            ),
            "exact_match": float(ep["exact_match"]),
            "selected_shared_path": float(ep["selected_shared_path"]),
            "selected_unshared_path": float(ep["selected_unshared_path"]),
            "root_trap_subtree_prob": ep.get("root_trap_subtree_prob"),
            "stage4_trap_child_prob": ep.get("stage4_trap_child_prob"),
            "all_fast_trap_route_prob": ep.get("all_fast_trap_route_prob"),
            "selected_trap_like": float(bool(ep.get("family_trap_like_path"))),
            "run_group": run_group,
        }
        previous_cumulative_raw_total_per_episode = cumulative_raw_total_per_episode
        previous_cumulative_total_per_episode = cumulative_total_cost_per_episode
        if post_raw_per_episode is not None:
            previous_post_raw_per_episode = post_raw_per_episode
        if post_total_per_episode is not None:
            previous_post_total_per_episode = post_total_per_episode
        row.update(build_wandb_episode_analysis_fields(ep))
        rows.append(row)
    add_wandb_group_aggregates(rows)
    return rows


def wandb_episode_log_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if key
        not in {
            "method",
            "seed",
            "run_group",
        }
    }


def build_wandb_episode_table(rows: list[dict[str, Any]], wandb_module: Any) -> Any:
    columns = [
        "method",
        "episode_1based",
        "task_bucket",
        "match_group",
        "mismatch_direction",
        "mode_pattern",
        "path_deep_count",
        "path_fast_count",
        "path_depth_balance",
        "fdddd_group",
        "raw_total_cost",
        "raw_terminal_penalty",
        "raw_reasoning_cost_component",
        "raw_mode_mismatch_cost_component",
        "exact_match",
        "oracle_action",
        "final_action",
        "original_task_id",
    ]
    table = wandb_module.Table(columns=columns)
    for row in rows:
        table.add_data(*[row.get(column) for column in columns])
    return table


def log_wandb_rows(
    wandb_run: Any,
    rows: list[dict[str, Any]],
    *,
    include_table: bool,
    wandb_module: Any,
) -> None:
    wandb_run.define_metric("episode_index")
    wandb_run.define_metric("*", step_metric="episode_index")
    for row in rows:
        wandb_run.log(
            wandb_episode_log_payload(row),
            step=int(row["episode_index"]),
        )
    if include_table:
        wandb_run.log(
            {
                "episode_analysis_table": build_wandb_episode_table(
                    rows,
                    wandb_module,
                )
            }
        )


def log_results_to_wandb(
    run_dir: Path,
    *,
    project: str,
    entity: str | None,
    run_name_prefix: str,
    run_group: str,
) -> None:
    try:
        import wandb  # type: ignore
    except Exception as exc:  # pragma: no cover
        raise SystemExit(
            "wandb is required for --wandb-project logging, but it is not installed."
        ) from exc

    context = load_run_context(run_dir)
    run_config = context["run_config"]
    base_config = {
        "dataset": run_config.get("dataset"),
        "family_kind": run_config.get("family_kind"),
        "schedule_mode": run_config.get("schedule_mode"),
        "repeats": run_config.get("repeats"),
        "seed": run_config.get("seed"),
        "executor_name": run_config.get("executor_name"),
        "horizon": run_config.get("horizon"),
        "methods": run_config.get("methods"),
        "run_dir": str(run_dir),
    }

    for method in run_config["methods"]:
        episodes = load_json(run_dir / method / "episodes.json")
        rows = build_wandb_episode_rows(
            method=method,
            episodes=episodes,
            seed=int(run_config.get("seed", 0)),
            run_group=run_group,
        )
        wandb_run = wandb.init(
            project=project,
            entity=entity,
            group=run_group,
            name=f"{run_name_prefix}{method}_seed{run_config.get('seed')}",
            reinit=True,
            config={**base_config, "method": method},
        )
        log_wandb_rows(wandb_run, rows, include_table=True, wandb_module=wandb)
        wandb_run.finish()


def orchestrate_run(
    *,
    data_path: Path,
    output_dir: Path,
    repeats: int,
    methods: list[str],
    family_kind: str,
    schedule_mode: str,
    switch_denominator: int,
    schedule_buckets_path: Path | None,
    trap_switch_cycle_source: str,
    common_eta_override: float | None,
    common_epsilon_override: float | None,
    ps_eta_shared_override: float | None,
    ps_loss_clip: float | None,
    ps_prob_floor: float | None,
    executor_name: str,
    post_switch_fixed_layer1_probs: bool,
    post_switch_fixed_tree_probs: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_run_group: str | None,
    wandb_run_name_prefix: str,
) -> Path:
    model_name = resolve_model_name_for_executor(executor_name)
    validate_methods(methods)
    run_dir = initialize_run(
        data_path=data_path,
        output_dir=output_dir,
        repeats=repeats,
        methods=methods,
        model_name=model_name,
        family_kind=family_kind,
        schedule_mode=schedule_mode,
        switch_denominator=switch_denominator,
        schedule_buckets_path=schedule_buckets_path,
        trap_switch_cycle_source=trap_switch_cycle_source,
        common_eta_override=common_eta_override,
        common_epsilon_override=common_epsilon_override,
        ps_eta_shared_override=ps_eta_shared_override,
        ps_loss_clip=ps_loss_clip,
        ps_prob_floor=ps_prob_floor,
        executor_name=executor_name,
        post_switch_fixed_layer1_probs=post_switch_fixed_layer1_probs,
        post_switch_fixed_tree_probs=post_switch_fixed_tree_probs,
    )
    script_path = Path(__file__).resolve()
    launched: list[tuple[str, subprocess.Popen[Any], Any]] = []

    for method in methods:
        method_dir = run_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        log_path = method_dir / "runner.log"
        log_handle = log_path.open("a", encoding="utf-8")
        log_handle.write(f"[launch] {datetime.now().isoformat()} method={method}\n")
        log_handle.flush()
        cmd = [
            sys.executable,
            str(script_path),
            "run-method",
            "--run-dir",
            str(run_dir),
            "--method",
            method,
        ]
        process = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
        )
        launched.append((method, process, log_handle))

    failures: list[dict[str, Any]] = []
    for method, process, log_handle in launched:
        return_code = process.wait()
        log_handle.write(
            f"[exit] {datetime.now().isoformat()} method={method} return_code={return_code}\n"
        )
        log_handle.close()
        if return_code != 0:
            failures.append({"method": method, "return_code": return_code})

    if failures:
        write_json(run_dir / "orchestrator_failures.json", failures)
        raise SystemExit(f"One or more method runs failed: {failures}")

    for method in methods:
        merge_method_results(run_dir, method)
    merge_all_results(run_dir)
    if wandb_project:
        log_results_to_wandb(
            run_dir,
            project=wandb_project,
            entity=wandb_entity,
            run_name_prefix=wandb_run_name_prefix,
            run_group=wandb_run_group or run_dir.name,
        )
    return run_dir


def build_cli() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run repeated shared-basin smoke with method-level persistence.")
    subparsers = parser.add_subparsers(dest="command")

    common_run = argparse.ArgumentParser(add_help=False)
    common_run.add_argument("--data", type=Path, default=DATASET_DEFAULT)
    common_run.add_argument("--output-dir", type=Path, required=True)
    common_run.add_argument("--repeats", type=int, default=10)
    common_run.add_argument("--family-kind", type=str, default=DEFAULT_FAMILY_KIND)
    common_run.add_argument(
        "--schedule-mode",
        type=str,
        default=SCHEDULE_MODE_STATIONARY,
        choices=[
            SCHEDULE_MODE_STATIONARY,
            SCHEDULE_MODE_TRAP_SWITCH,
            SCHEDULE_MODE_TRAP_ONLY_RANDOM,
        ],
    )
    common_run.add_argument("--switch-denominator", type=int, default=3)
    common_run.add_argument("--schedule-buckets", type=Path)
    common_run.add_argument(
        "--trap-switch-cycle-source",
        type=str,
        default=TRAP_SWITCH_CYCLE_SOURCE_BUCKET,
        choices=[TRAP_SWITCH_CYCLE_SOURCE_BUCKET, TRAP_SWITCH_CYCLE_SOURCE_DATASET],
        help=(
            "For trap_switch schedules, choose whether repeats multiply the bucket "
            "length or the full dataset length."
        ),
    )
    common_run.add_argument("--common-eta-override", type=float)
    common_run.add_argument("--common-epsilon-override", type=float)
    common_run.add_argument(
        "--ps-eta-shared-override",
        type=float,
        help="Override eta_shared for PS-family methods only.",
    )
    common_run.add_argument(
        "--ps-loss-clip",
        type=float,
        help="Clip PS-family importance-weighted estimated losses to this value.",
    )
    common_run.add_argument(
        "--ps-prob-floor",
        type=float,
        help="Floor PS-family importance-weight denominators at this probability.",
    )
    common_run.add_argument(
        "--post-switch-fixed-layer1-probs",
        action="store_true",
        help=(
            "At the trap-switch boundary, snapshot each policy's root/direct-child "
            "marginal distribution and reuse it for all post-switch root choices."
        ),
    )
    common_run.add_argument(
        "--post-switch-fixed-tree-probs",
        action="store_true",
        help=(
            "At the trap-switch boundary, snapshot each policy's distribution from "
            "every internal prefix to its direct children and reuse those frozen "
            "distributions for all post-switch tree choices."
        ),
    )
    common_run.add_argument(
        "--executor-name",
        type=str,
        default=DEFAULT_EXECUTOR_NAME,
        choices=["llm_bench", "simulated"],
    )
    common_run.add_argument("--methods", nargs="+", default=list(DEFAULT_METHODS))
    common_run.add_argument(
        "--wandb-project",
        type=str,
        help="Optional W&B project name for per-episode cumulative raw total logging.",
    )
    common_run.add_argument(
        "--wandb-entity",
        type=str,
        help="Optional W&B entity/team name.",
    )
    common_run.add_argument(
        "--wandb-run-group",
        type=str,
        help="Optional W&B group; defaults to the output directory name.",
    )
    common_run.add_argument(
        "--wandb-run-name-prefix",
        type=str,
        default="",
        help="Prefix added to each per-method W&B run name.",
    )

    setup_parser = subparsers.add_parser("setup", parents=[common_run])
    setup_parser.set_defaults(command="setup")

    orchestrate_parser = subparsers.add_parser("orchestrate", parents=[common_run])
    orchestrate_parser.set_defaults(command="orchestrate")

    method_parser = subparsers.add_parser("run-method")
    method_parser.add_argument("--run-dir", type=Path, required=True)
    method_parser.add_argument("--method", type=str, required=True)
    method_parser.set_defaults(command="run-method")

    merge_method_parser = subparsers.add_parser("merge-method")
    merge_method_parser.add_argument("--run-dir", type=Path, required=True)
    merge_method_parser.add_argument("--method", type=str, required=True)
    merge_method_parser.set_defaults(command="merge-method")

    merge_all_parser = subparsers.add_parser("merge-all")
    merge_all_parser.add_argument("--run-dir", type=Path, required=True)
    merge_all_parser.set_defaults(command="merge-all")
    return parser


def main() -> None:
    parser = build_cli()
    argv = sys.argv[1:]
    known_commands = {"setup", "orchestrate", "run-method", "merge-method", "merge-all"}
    if not argv or argv[0] not in known_commands:
        argv = ["orchestrate", *argv]
    args = parser.parse_args(argv)
    if hasattr(args, "post_switch_fixed_layer1_probs"):
        resolve_post_switch_probability_freeze_mode(
            post_switch_fixed_layer1_probs=args.post_switch_fixed_layer1_probs,
            post_switch_fixed_tree_probs=args.post_switch_fixed_tree_probs,
        )

    if args.command == "setup":
        model_name = resolve_model_name_for_executor(args.executor_name)
        validate_methods(args.methods)
        run_dir = initialize_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            methods=args.methods,
            model_name=model_name,
            family_kind=args.family_kind,
            schedule_mode=args.schedule_mode,
            switch_denominator=args.switch_denominator,
            schedule_buckets_path=args.schedule_buckets,
            trap_switch_cycle_source=args.trap_switch_cycle_source,
            common_eta_override=args.common_eta_override,
            common_epsilon_override=args.common_epsilon_override,
            ps_eta_shared_override=args.ps_eta_shared_override,
            ps_loss_clip=args.ps_loss_clip,
            ps_prob_floor=args.ps_prob_floor,
            executor_name=args.executor_name,
            post_switch_fixed_layer1_probs=args.post_switch_fixed_layer1_probs,
            post_switch_fixed_tree_probs=args.post_switch_fixed_tree_probs,
        )
        print(str(run_dir))
        return

    if args.command == "orchestrate":
        validate_methods(args.methods)
        run_dir = orchestrate_run(
            data_path=args.data,
            output_dir=args.output_dir,
            repeats=args.repeats,
            methods=args.methods,
            family_kind=args.family_kind,
            schedule_mode=args.schedule_mode,
            switch_denominator=args.switch_denominator,
            schedule_buckets_path=args.schedule_buckets,
            trap_switch_cycle_source=args.trap_switch_cycle_source,
            common_eta_override=args.common_eta_override,
            common_epsilon_override=args.common_epsilon_override,
            ps_eta_shared_override=args.ps_eta_shared_override,
            ps_loss_clip=args.ps_loss_clip,
            ps_prob_floor=args.ps_prob_floor,
            executor_name=args.executor_name,
            post_switch_fixed_layer1_probs=args.post_switch_fixed_layer1_probs,
            post_switch_fixed_tree_probs=args.post_switch_fixed_tree_probs,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_group=args.wandb_run_group,
            wandb_run_name_prefix=args.wandb_run_name_prefix,
        )
        print(str(run_dir))
        return

    if args.command == "run-method":
        if args.method not in POLICY_REGISTRY:
            raise SystemExit(f"Unknown method for run-method: {args.method}")
        run_policy_method(run_dir=args.run_dir, method=args.method)
        print(str(args.run_dir / args.method))
        return

    if args.command == "merge-method":
        payload = merge_method_results(args.run_dir, args.method)
        print(
            json.dumps(
                {
                    "method": args.method,
                    "total_cost_mean": payload["summary"]["total_cost_mean"],
                    "shared_path_fraction": payload["summary"]["shared_path_fraction"],
                },
                ensure_ascii=False,
            )
        )
        return

    if args.command == "merge-all":
        payload = merge_all_results(args.run_dir)
        print(json.dumps(payload["compare_rows"], ensure_ascii=False))
        return

    raise SystemExit(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()
