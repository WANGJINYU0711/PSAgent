"""Minimal fixed-tree environment for the derived airline cancellation benchmark.

This module intentionally does not depend on tau2's orchestrator. It operates
over the structured derived instances in ``data/derived/.../tasks.json`` and
supports:

- reset(instance)
- run_path(path)
- leaf start-condition typing plus layered shared-upload barrier helpers
- simple terminal-cost computation
- default oracle-like and noisy rule-based stage executors
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Sequence

from evaluator import DEFAULT_COST_SPEC, evaluate_terminal_prediction
from adapters.airline_adapter import AirlineTaskAdapter
from adapters.telecom_mms_adapter import TelecomMMSTaskAdapter
from executors.bench_backed_executor import BenchBackedExecutor
from executors.llm_bench_executor import LLMBenchExecutor
from executors.simulated_executor import SimulatedExecutor
from executors.telecom_bench_backed_executor import TelecomBenchBackedExecutor
from executors.telecom_llm_bench_executor import TelecomLLMBenchExecutor
from telecom_mms_evaluator import (
    DEFAULT_COST_SPEC as TELECOM_DEFAULT_COST_SPEC,
    TELECOM_MMS_COST_SCALE_VERSION,
    TELECOM_MMS_PATH_UPPER_BOUND_V2,
    TELECOM_MMS_TERMINAL_UPPER_BOUND_V2,
    evaluate_terminal_prediction as evaluate_telecom_terminal_prediction,
)
from tree_family.generator import TreeFamilyGenerator


JsonDict = dict[str, Any]
StageExecutor = Callable[["FixedTreeEnvironment", "AgentSpec", JsonDict], JsonDict]
PrefixKey = tuple[str, ...]
EdgeKey = tuple[PrefixKey, PrefixKey]

LLM_BENCH_REASONING_ALPHA_API = 100.0
LLM_BENCH_REASONING_ALPHA_IN = 0.0001
LLM_BENCH_REASONING_ALPHA_OUT = 0.0004
LLM_BENCH_REASONING_DEFAULT_MODE = "token"
LLM_BENCH_REASONING_MATCH_DISCOUNT = 0.85
LLM_BENCH_REASONING_MISMATCH_PENALTY_DEEP_REQUIRED = 1.35
LLM_BENCH_REASONING_MISMATCH_PENALTY_FAST_REQUIRED = 1.15
LLM_BENCH_REASONING_CALIBRATED_MATCH_DISCOUNT = 0.70
LLM_BENCH_REASONING_CALIBRATED_MISMATCH_PENALTY_DEEP_REQUIRED = 1.55
LLM_BENCH_REASONING_CALIBRATED_MISMATCH_PENALTY_FAST_REQUIRED = 1.25
TELECOM_EXEC_CLEAN_V4_TERMINAL_UPPER_BOUND = 32.0
TELECOM_MODE_MISMATCH_FAST_ON_DEEP_COST_V2 = 1.5
TELECOM_MODE_MISMATCH_DEEP_ON_FAST_COST_V2 = 0.5
TELECOM_MMS_REASONING_INPUT_TOKEN_BUDGET_V2 = 20_000.0
TELECOM_MMS_REASONING_OUTPUT_TOKEN_BUDGET_V2 = 7_500.0
TELECOM_MMS_REASONING_API_COST_BUDGET_USD_V2 = 0.05
TELECOM_MMS_REASONING_UPPER_BOUND_TOKEN_V2 = (
    (LLM_BENCH_REASONING_ALPHA_IN * TELECOM_MMS_REASONING_INPUT_TOKEN_BUDGET_V2)
    + (LLM_BENCH_REASONING_ALPHA_OUT * TELECOM_MMS_REASONING_OUTPUT_TOKEN_BUDGET_V2)
)
TELECOM_MMS_REASONING_UPPER_BOUND_API_V2 = (
    LLM_BENCH_REASONING_ALPHA_API * TELECOM_MMS_REASONING_API_COST_BUDGET_USD_V2
)
if LLM_BENCH_REASONING_DEFAULT_MODE == "api":
    TELECOM_MMS_REASONING_UPPER_BOUND_DEFAULT_V2 = TELECOM_MMS_REASONING_UPPER_BOUND_API_V2
else:
    TELECOM_MMS_REASONING_UPPER_BOUND_DEFAULT_V2 = (
        TELECOM_MMS_REASONING_UPPER_BOUND_TOKEN_V2
    )
TELECOM_MMS_TOTAL_UPPER_BOUND_V2_DEFAULT = (
    TELECOM_MMS_TERMINAL_UPPER_BOUND_V2
    + TELECOM_MMS_PATH_UPPER_BOUND_V2
    + TELECOM_MMS_REASONING_UPPER_BOUND_DEFAULT_V2
)


def _env_flag(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


def leaf_starts_shared_upload(
    path: Sequence[str],
    agent_lookup: Mapping[str, Any],
) -> bool:
    """Return whether the sampled leaf can start a shared upload."""

    if not path:
        return False
    return int(getattr(agent_lookup[path[-1]], "g", 1)) == 0


def compute_shared_upload_edges(
    path: Sequence[str],
    agent_lookup: Mapping[str, Any],
) -> list[EdgeKey]:
    """Return the parent-child edges traversed by one shared upload.

    The sampled leaf must have ``g=0`` to start an upload. Once started, the
    update walks upward edge-by-edge until it reaches the first internal node
    whose ``g=1``; that node can receive the update from its children but cannot
    forward it to its own parent.
    """

    if not leaf_starts_shared_upload(path, agent_lookup):
        return []

    child_prefix: PrefixKey = tuple(path)
    edges: list[EdgeKey] = []
    while child_prefix:
        parent_prefix = child_prefix[:-1]
        edges.append((parent_prefix, child_prefix))
        if not parent_prefix:
            break
        if int(getattr(agent_lookup[parent_prefix[-1]], "g", 1)) != 0:
            break
        child_prefix = parent_prefix
    return edges


def compute_shared_upload_stop_prefix(
    path: Sequence[str],
    agent_lookup: Mapping[str, Any],
) -> PrefixKey | None:
    """Return the first internal barrier prefix that stops upward upload."""

    upload_edges = compute_shared_upload_edges(path, agent_lookup)
    if not upload_edges:
        return None
    stop_prefix = upload_edges[-1][0]
    if not stop_prefix:
        return None
    if int(getattr(agent_lookup[stop_prefix[-1]], "g", 1)) == 1:
        return stop_prefix
    return None


def compute_first_private_barrier_depth(
    path: Sequence[str],
    agent_lookup: Mapping[str, Any],
) -> int | None:
    """Return the 1-indexed stage depth of the first ``g=1`` node on a path."""

    for depth, agent_id in enumerate(path, start=1):
        if int(getattr(agent_lookup[agent_id], "g", 1)) == 1:
            return depth
    return None


def _normalize_usage_breakdown(row: Mapping[str, Any] | None) -> JsonDict:
    if not isinstance(row, Mapping):
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


def compute_llm_bench_reasoning_components(
    *,
    prompt_tokens_total: float,
    completion_tokens_total: float,
    api_cost_total_usd_raw: float,
    default_mode: str = LLM_BENCH_REASONING_DEFAULT_MODE,
) -> JsonDict:
    raw_reasoning_cost_component_api = LLM_BENCH_REASONING_ALPHA_API * float(
        api_cost_total_usd_raw
    )
    raw_reasoning_cost_component_token = (
        (LLM_BENCH_REASONING_ALPHA_IN * float(prompt_tokens_total))
        + (LLM_BENCH_REASONING_ALPHA_OUT * float(completion_tokens_total))
    )
    if default_mode == "api":
        raw_reasoning_cost_component = raw_reasoning_cost_component_api
    else:
        raw_reasoning_cost_component = raw_reasoning_cost_component_token
    return {
        "alpha_api": LLM_BENCH_REASONING_ALPHA_API,
        "alpha_in": LLM_BENCH_REASONING_ALPHA_IN,
        "alpha_out": LLM_BENCH_REASONING_ALPHA_OUT,
        "reasoning_cost_mode_default": default_mode,
        "raw_reasoning_cost_component": raw_reasoning_cost_component,
        "raw_reasoning_cost_component_api": raw_reasoning_cost_component_api,
        "raw_reasoning_cost_component_token": raw_reasoning_cost_component_token,
    }


@dataclass(frozen=True)
class AgentSpec:
    """Lightweight candidate-agent description for one stage."""

    agent_id: str
    stage_name: str
    g: int
    kind: str
    cost: float = 0.0


@dataclass
class EpisodeResult:
    """Result object returned after running one full fixed-tree path."""

    instance_id: str
    selected_path: list[str]
    leaf_type: str
    stage_outputs: dict[str, JsonDict]
    final_action: Optional[str]
    oracle_action: Optional[str]
    terminal_cost: float
    raw_terminal_penalty: float
    raw_path_cost_component: float
    raw_reasoning_cost_component: float
    raw_total_cost: float
    normalized_terminal_penalty: float
    success: bool
    path_agent_cost: float
    reasoning_cost: float
    total_cost: float
    total_cost_upper_bound: float
    cost_scale_version: str
    raw_outcome_penalty: float = 0.0
    raw_policy_penalty: float = 0.0
    raw_reasoning_cost_component_api: float | None = None
    raw_reasoning_cost_component_token: float | None = None
    raw_total_cost_api: float | None = None
    raw_total_cost_token: float | None = None
    prompt_tokens_total: float = 0.0
    completion_tokens_total: float = 0.0
    total_tokens_total: float = 0.0
    api_cost_total_usd_raw: float = 0.0
    generation_time_total_seconds: float = 0.0
    llm_round_trip_total_seconds: float = 0.0
    tool_wall_clock_total_seconds: float = 0.0
    episode_wall_clock_seconds: float = 0.0
    episode_log: JsonDict = field(default_factory=dict)

    def to_dict(self) -> JsonDict:
        """Return a JSON-serializable dictionary."""

        return asdict(self)


def default_agent_catalog() -> list[AgentSpec]:
    """Return a small default catalog with oracle-like and noisy agents."""

    stage_costs = {
        "stage1": (0.2, 0.4),
        "stage2": (0.2, 0.5),
        "stage3": (0.2, 0.5),
        "stage4": (0.2, 0.5),
        "stage5": (0.2, 0.5),
    }
    stage_ids = {
        "stage1": ("ground_oracle_g0", "ground_noisy_g1"),
        "stage2": ("resolve_oracle_g0", "resolve_noisy_g1"),
        "stage3": ("feature_oracle_g0", "feature_noisy_g1"),
        "stage4": ("adjudicate_oracle_g0", "adjudicate_noisy_g1"),
        "stage5": ("execute_oracle_g0", "execute_noisy_g1"),
    }

    catalog: list[AgentSpec] = []
    for stage_name in FixedTreeEnvironment.STAGE_NAMES:
        oracle_id, noisy_id = stage_ids[stage_name]
        oracle_cost, noisy_cost = stage_costs[stage_name]
        catalog.append(
            AgentSpec(
                agent_id=oracle_id,
                stage_name=stage_name,
                g=0,
                kind="rule",
                cost=oracle_cost,
            )
        )
        catalog.append(
            AgentSpec(
                agent_id=noisy_id,
                stage_name=stage_name,
                g=1,
                kind="simulated",
                cost=noisy_cost,
            )
        )
    return catalog


class FixedTreeEnvironment:
    """Minimal fixed-tree environment operating over structured derived tasks."""

    STAGE_NAMES = ["stage1", "stage2", "stage3", "stage4", "stage5"]

    def __init__(
        self,
        agent_catalog: Iterable[AgentSpec],
        agent_executors: Optional[dict[str, StageExecutor]] = None,
        family_kind: str | None = None,
        family_seed: int = 0,
        executor_name: str = "simulated",
    ) -> None:
        self.family_kind = family_kind
        self.family_seed = family_seed
        self.executor_name = executor_name
        self.family_spec = None
        self.family_agent_map = None
        self.family_executor = None
        self.task_adapter = None
        self.current_task_descriptor = None
        self._family_stages: list[str] | None = None

        runtime_catalog = list(agent_catalog)
        if family_kind is not None:
            runtime_catalog = self._build_family_runtime_catalog(family_kind, family_seed)

        self.agent_catalog: dict[str, AgentSpec] = {
            agent.agent_id: agent for agent in runtime_catalog
        }
        self.agents_by_stage: dict[str, list[AgentSpec]] = {stage: [] for stage in self.STAGE_NAMES}
        for agent in self.agent_catalog.values():
            if agent.stage_name not in self.STAGE_NAMES:
                raise ValueError(f"Unknown stage_name in catalog: {agent.stage_name}")
            self.agents_by_stage[agent.stage_name].append(agent)

        self.agent_executors = agent_executors or self._build_default_executors()
        self.current_instance: Optional[JsonDict] = None
        self.current_instance_id: Optional[str] = None
        self._last_episode_log: Optional[JsonDict] = None

    def _build_family_runtime_catalog(self, family_kind: str, family_seed: int) -> list[AgentSpec]:
        generator = TreeFamilyGenerator()
        family_spec, family_agent_map = generator.build_family(family_kind, seed=family_seed)
        validation_errors = generator.validate_family(family_spec, family_agent_map)
        if validation_errors:
            raise ValueError(
                "Invalid family specification: " + "; ".join(validation_errors)
            )

        self.family_spec = family_spec
        self.family_agent_map = family_agent_map
        self.task_adapter = None
        self._family_stages = list(family_spec.stages)
        if self.executor_name == "simulated":
            self.family_executor = SimulatedExecutor(
                stages=list(family_spec.stages),
                seed=family_seed,
            )
        elif self.executor_name == "bench_backed":
            self.family_executor = BenchBackedExecutor(
                stages=list(family_spec.stages),
                seed=family_seed,
            )
        elif self.executor_name == "llm_bench":
            self.family_executor = LLMBenchExecutor(
                stages=list(family_spec.stages),
                seed=family_seed,
            )
        else:
            raise ValueError(f"Unsupported executor_name: {self.executor_name}")

        flat_profile_switch_path_cost = (
            family_kind == "shared_basin_strong_prefix_dedup_profile_switch"
            and str(os.environ.get("PSAGENT_PROFILE_SWITCH_FLAT_PATH_COST", "")).strip()
            in {"1", "true", "True", "yes", "on"}
        )
        flat_stage_costs: dict[str, float] = {}
        if flat_profile_switch_path_cost:
            for stage_name in family_spec.stages:
                stage_costs = [
                    float(family_agent_map[agent_id].base_cost)
                    for agent_id in family_spec.stage_agents[stage_name]
                ]
                flat_stage_costs[stage_name] = (
                    round(sum(stage_costs) / len(stage_costs), 6)
                    if stage_costs
                    else 0.0
                )

        runtime_catalog: list[AgentSpec] = []
        for stage_name in family_spec.stages:
            for agent_id in family_spec.stage_agents[stage_name]:
                family_agent = family_agent_map[agent_id]
                runtime_catalog.append(
                    AgentSpec(
                        agent_id=family_agent.agent_id,
                        stage_name=stage_name,
                        g=family_agent.g,
                        kind="family",
                        cost=(
                            flat_stage_costs[stage_name]
                            if flat_profile_switch_path_cost
                            else family_agent.base_cost
                        ),
                    )
                )
        return runtime_catalog

    def _build_default_executors(self) -> dict[str, StageExecutor]:
        return {
            "stage1": _run_stage1,
            "stage2": _run_stage2,
            "stage3": _run_stage3,
            "stage4": _run_stage4,
            "stage5": _run_stage5,
        }

    def reset(self, instance: JsonDict) -> JsonDict:
        """Load a new derived instance into the environment."""

        if not isinstance(instance, dict):
            raise TypeError("Instance must be a dict-like derived sample.")
        if "instance_id" not in instance:
            raise ValueError("Instance is missing required field 'instance_id'.")
        for stage_name in self.STAGE_NAMES:
            if stage_name not in instance:
                raise ValueError(f"Instance is missing required stage: {stage_name}")

        self.current_instance = deepcopy(instance)
        self.current_instance_id = str(instance["instance_id"])
        self._last_episode_log = None
        if self.family_kind is not None:
            self._ensure_family_executor_for_instance(self.current_instance)
            self.task_adapter = self._select_task_adapter(self.current_instance)
            assert self.task_adapter is not None
            self.current_task_descriptor = self.task_adapter.build_task_descriptor(
                self.current_instance
            )
        else:
            self.current_task_descriptor = None
        return deepcopy(self.current_instance)

    def run_path(self, path: list[str]) -> EpisodeResult:
        """Execute one complete stage-wise path over the current instance."""

        if self.current_instance is None:
            raise RuntimeError("Environment has no loaded instance. Call reset(instance) first.")

        self._validate_path(path)
        instance = self.current_instance

        if self.family_kind is not None:
            return self._run_family_path(path)

        stage_outputs: dict[str, JsonDict] = {}
        stage_trace: list[JsonDict] = []

        for stage_name, agent_id in zip(self.STAGE_NAMES, path):
            agent = self.agent_catalog[agent_id]
            executor = self.agent_executors.get(stage_name)
            if executor is None:
                raise KeyError(f"No executor registered for stage: {stage_name}")

            stage_bundle = executor(self, agent, stage_outputs)
            stage_outputs[stage_name] = stage_bundle
            stage_trace.append(
                {
                    "stage_name": stage_name,
                    "agent_id": agent_id,
                    "agent_kind": agent.kind,
                    "agent_g": agent.g,
                    "input": deepcopy(stage_bundle.get("input", {})),
                    "output": deepcopy(stage_bundle.get("output", {})),
                }
            )

        leaf_type = self.compute_leaf_type(path)
        evaluator_result = self.evaluate_terminal_outcome(stage_outputs, path)
        path_agent_cost = sum(self.agent_catalog[agent_id].cost for agent_id in path)
        reasoning_metrics = {
            "raw_reasoning_cost_component": 0.0,
            "raw_reasoning_cost_component_api": None,
            "raw_reasoning_cost_component_token": None,
            "reasoning_cost_mode_default": None,
            "prompt_tokens_total": 0.0,
            "completion_tokens_total": 0.0,
            "total_tokens_total": 0.0,
            "api_cost_total_usd_raw": 0.0,
            "generation_time_total_seconds": 0.0,
            "llm_round_trip_total_seconds": 0.0,
            "tool_wall_clock_total_seconds": 0.0,
            "episode_wall_clock_seconds": 0.0,
            "stage_prompt_tokens": [0.0] * len(self.STAGE_NAMES),
            "stage_completion_tokens": [0.0] * len(self.STAGE_NAMES),
            "stage_total_tokens": [0.0] * len(self.STAGE_NAMES),
            "stage_api_cost_usd": [0.0] * len(self.STAGE_NAMES),
            "stage_generation_time_seconds": [0.0] * len(self.STAGE_NAMES),
            "stage_llm_round_trip_seconds": [0.0] * len(self.STAGE_NAMES),
            "stage_tool_wall_clock_seconds": [0.0] * len(self.STAGE_NAMES),
            "stage_wall_clock_seconds": [0.0] * len(self.STAGE_NAMES),
            "reasoning_resource_breakdown": {
                "prompt_tokens_total": 0.0,
                "completion_tokens_total": 0.0,
                "total_tokens_total": 0.0,
                "api_cost_total_usd_raw": 0.0,
            },
            "latency_breakdown": {
                "generation_time_total_seconds": 0.0,
                "llm_round_trip_total_seconds": 0.0,
                "tool_wall_clock_total_seconds": 0.0,
                "episode_wall_clock_seconds": 0.0,
            },
            "trace": [],
        }
        cost_metrics = self._build_cost_metrics(
            evaluator_result=evaluator_result,
            path_agent_cost=path_agent_cost,
            reasoning_metrics=reasoning_metrics,
        )

        final_action = (
            stage_outputs.get("stage5", {}).get("output", {}).get("final_action")
        )
        oracle_action = (
            instance.get("stage5", {})
            .get("oracle_output", {})
            .get("final_action")
        )
        success = bool(evaluator_result["exact_match"])

        first_private_barrier_stage = self._first_private_barrier_stage_label(path)
        barrier_stop_prefix = self.compute_shared_upload_stop_prefix(path)
        legal_child_count_per_stage = self._legal_child_count_per_stage(path)
        annotated_stage_trace = self._annotate_stage_trace_with_terminal_details(
            stage_trace,
            evaluator_result,
        )
        episode_log = {
            "instance_id": self.current_instance_id,
            "selected_path": list(path),
            "leaf_type": leaf_type,
            "stage_trace": annotated_stage_trace,
            "final_action": final_action,
            "oracle_action": oracle_action,
            "terminal_cost": cost_metrics["raw_terminal_penalty"],
            "path_agent_cost": path_agent_cost,
            "total_cost": cost_metrics["normalized_total_cost"],
            "success": success,
            "evaluator_version": evaluator_result["evaluator_version"],
            "false_cancel_count": evaluator_result["false_cancel_count"],
            "missed_cancel_count": evaluator_result["missed_cancel_count"],
            "false_refuse_count": evaluator_result["false_refuse_count"],
            "missed_refuse_count": evaluator_result["missed_refuse_count"],
            "subset_mismatch": evaluator_result["subset_mismatch"],
            "cost_breakdown": deepcopy(evaluator_result["cost_breakdown"]),
            "outcome_cost_breakdown": deepcopy(
                evaluator_result.get("outcome_cost_breakdown", {})
            ),
            "policy_cost_breakdown": deepcopy(
                evaluator_result.get("policy_cost_breakdown", {})
            ),
            "terminal_cost_breakdown": deepcopy(
                evaluator_result.get("terminal_cost_breakdown", {})
            ),
            "raw_outcome_penalty": cost_metrics["raw_outcome_penalty"],
            "raw_policy_penalty": cost_metrics["raw_policy_penalty"],
            "raw_terminal_penalty": cost_metrics["raw_terminal_penalty"],
            "legacy_raw_terminal_penalty": cost_metrics["legacy_raw_terminal_penalty"],
            "raw_terminal_penalty_exec_clean_v4": cost_metrics[
                "raw_terminal_penalty_exec_clean_v4"
            ],
            "terminal_adjustment": deepcopy(cost_metrics["terminal_adjustment"]),
            "terminal_adjustment_enabled": bool(
                cost_metrics["terminal_adjustment"].get("enabled")
            ),
            "terminal_adjustment_floor": cost_metrics["terminal_adjustment"].get(
                "applied_floor"
            ),
            "terminal_adjustment_reasons": list(
                cost_metrics["terminal_adjustment"].get("applied_floor_reasons", [])
            ),
            "clear_success_proxy": bool(
                cost_metrics["terminal_adjustment"].get("clear_success_proxy", success)
            ),
            "auxiliary_success_proxy": bool(
                cost_metrics["terminal_adjustment"].get("auxiliary_success_proxy", True)
            ),
            "terminal_majority_pair": cost_metrics["terminal_adjustment"].get(
                "majority_pair"
            ),
            "raw_path_cost_component": cost_metrics["raw_path_cost_component"],
            "raw_reasoning_cost_component_api": cost_metrics[
                "raw_reasoning_cost_component_api"
            ],
            "raw_reasoning_cost_component_token": cost_metrics[
                "raw_reasoning_cost_component_token"
            ],
            "raw_total_cost": cost_metrics["raw_total_cost"],
            "raw_total_cost_api": cost_metrics["raw_total_cost_api"],
            "raw_total_cost_token": cost_metrics["raw_total_cost_token"],
            "raw_reasoning_cost_component": cost_metrics["raw_reasoning_cost_component"],
            "terminal_cost_upper_bound": cost_metrics["terminal_cost_upper_bound"],
            "path_cost_upper_bound": cost_metrics["path_cost_upper_bound"],
            "reasoning_cost_upper_bound": cost_metrics["reasoning_cost_upper_bound"],
            "normalized_terminal_penalty": cost_metrics["normalized_terminal_penalty"],
            "normalized_total_cost": cost_metrics["normalized_total_cost"],
            "total_cost_upper_bound": cost_metrics["total_cost_upper_bound"],
            "cost_scale_version": cost_metrics["cost_scale_version"],
            "reasoning_cost": cost_metrics["raw_reasoning_cost_component"],
            "reasoning_cost_mode_default": cost_metrics["reasoning_cost_mode_default"],
            "reasoning_weight_calibration_enabled": bool(
                reasoning_metrics.get("reasoning_weight_calibration_enabled", False)
            ),
            "raw_mode_mismatch_cost_component": float(
                reasoning_metrics.get("raw_mode_mismatch_cost_component", 0.0) or 0.0
            ),
            "mode_mismatch_cost_enabled": bool(
                reasoning_metrics.get("mode_mismatch_cost_enabled", False)
            ),
            "mode_mismatch_report_only_enabled": bool(
                reasoning_metrics.get("mode_mismatch_report_only_enabled", False)
            ),
            "mode_mismatch_fast_on_deep_cost": float(
                reasoning_metrics.get("mode_mismatch_fast_on_deep_cost", 0.0) or 0.0
            ),
            "mode_mismatch_deep_on_fast_cost": float(
                reasoning_metrics.get("mode_mismatch_deep_on_fast_cost", 0.0) or 0.0
            ),
            "prompt_tokens_total": reasoning_metrics["prompt_tokens_total"],
            "completion_tokens_total": reasoning_metrics["completion_tokens_total"],
            "total_tokens_total": reasoning_metrics["total_tokens_total"],
            "api_cost_total_usd_raw": reasoning_metrics["api_cost_total_usd_raw"],
            "reasoning_resource_breakdown": deepcopy(
                reasoning_metrics["reasoning_resource_breakdown"]
            ),
            "generation_time_total_seconds": reasoning_metrics[
                "generation_time_total_seconds"
            ],
            "llm_round_trip_total_seconds": reasoning_metrics[
                "llm_round_trip_total_seconds"
            ],
            "tool_wall_clock_total_seconds": reasoning_metrics[
                "tool_wall_clock_total_seconds"
            ],
            "episode_wall_clock_seconds": reasoning_metrics["episode_wall_clock_seconds"],
            "latency_breakdown": deepcopy(reasoning_metrics["latency_breakdown"]),
            "policy_action_violation": bool(
                evaluator_result.get("policy_action_violation", False)
            ),
            "policy_communication_violation": bool(
                evaluator_result.get("policy_communication_violation", False)
            ),
            "policy_nl_assertions_total": int(
                evaluator_result.get("policy_nl_assertions_total", 0) or 0
            ),
            "policy_nl_assertions_failed": int(
                evaluator_result.get("policy_nl_assertions_failed", 0) or 0
            ),
            "policy_violation_count": int(
                evaluator_result.get("policy_violation_count", 0) or 0
            ),
            "policy_violation_breakdown": deepcopy(
                evaluator_result.get("policy_violation_breakdown", {})
            ),
            "policy_eval_source": evaluator_result.get("policy_eval_source"),
            "policy_eval_scope": evaluator_result.get("policy_eval_scope"),
            "selected_shared_path": leaf_type == "shared",
            "selected_unshared_path": leaf_type != "shared",
            "first_private_barrier_stage": first_private_barrier_stage,
            "barrier_stop_depth": len(barrier_stop_prefix) if barrier_stop_prefix else None,
            "legal_child_count_per_stage": legal_child_count_per_stage,
            "candidate_count_per_stage": list(legal_child_count_per_stage),
        }
        self._last_episode_log = deepcopy(episode_log)

        return EpisodeResult(
            instance_id=self.current_instance_id or "unknown_instance",
            selected_path=list(path),
            leaf_type=leaf_type,
            stage_outputs=stage_outputs,
            final_action=final_action,
            oracle_action=oracle_action,
            terminal_cost=cost_metrics["raw_terminal_penalty"],
            raw_terminal_penalty=cost_metrics["raw_terminal_penalty"],
            raw_path_cost_component=cost_metrics["raw_path_cost_component"],
            raw_reasoning_cost_component=cost_metrics["raw_reasoning_cost_component"],
            raw_total_cost=cost_metrics["raw_total_cost"],
            normalized_terminal_penalty=cost_metrics["normalized_terminal_penalty"],
            success=success,
            path_agent_cost=path_agent_cost,
            reasoning_cost=cost_metrics["raw_reasoning_cost_component"],
            total_cost=cost_metrics["normalized_total_cost"],
            total_cost_upper_bound=cost_metrics["total_cost_upper_bound"],
            cost_scale_version=cost_metrics["cost_scale_version"],
            raw_outcome_penalty=cost_metrics["raw_outcome_penalty"],
            raw_policy_penalty=cost_metrics["raw_policy_penalty"],
            raw_reasoning_cost_component_api=cost_metrics[
                "raw_reasoning_cost_component_api"
            ],
            raw_reasoning_cost_component_token=cost_metrics[
                "raw_reasoning_cost_component_token"
            ],
            raw_total_cost_api=cost_metrics["raw_total_cost_api"],
            raw_total_cost_token=cost_metrics["raw_total_cost_token"],
            prompt_tokens_total=reasoning_metrics["prompt_tokens_total"],
            completion_tokens_total=reasoning_metrics["completion_tokens_total"],
            total_tokens_total=reasoning_metrics["total_tokens_total"],
            api_cost_total_usd_raw=reasoning_metrics["api_cost_total_usd_raw"],
            generation_time_total_seconds=reasoning_metrics[
                "generation_time_total_seconds"
            ],
            llm_round_trip_total_seconds=reasoning_metrics[
                "llm_round_trip_total_seconds"
            ],
            tool_wall_clock_total_seconds=reasoning_metrics[
                "tool_wall_clock_total_seconds"
            ],
            episode_wall_clock_seconds=reasoning_metrics["episode_wall_clock_seconds"],
            episode_log=episode_log,
        )

    def _run_family_path(self, path: list[str]) -> EpisodeResult:
        if self.current_instance is None or self.current_task_descriptor is None:
            raise RuntimeError("Family mode requires a loaded instance and task descriptor.")
        if self.family_agent_map is None or self.family_executor is None:
            raise RuntimeError("Family mode is not fully initialized.")

        execution = self.family_executor.run_path(
            task=self.current_task_descriptor,
            path=path,
            agent_map=self.family_agent_map,
            raw_instance=self.current_instance,
        )
        stage_outputs = self._family_stage_outputs_from_execution(execution)
        evaluator_result = self.evaluate_terminal_outcome(
            stage_outputs,
            path,
            execution=execution,
        )
        if self._profile_switch_flat_path_cost_enabled():
            path_agent_cost = sum(self.agent_catalog[agent_id].cost for agent_id in path)
        else:
            path_agent_cost = float(execution["path_agent_cost"])
        reasoning_metrics = self._compute_family_reasoning_cost(
            path,
            stage_trace=execution.get("stage_trace", []),
        )
        cost_metrics = self._build_cost_metrics(
            evaluator_result=evaluator_result,
            path_agent_cost=path_agent_cost,
            reasoning_metrics=reasoning_metrics,
        )
        final_action = execution.get("final_action")
        oracle_action = (
            self.current_instance.get("stage5", {})
            .get("oracle_output", {})
            .get("final_action")
        )
        success = bool(evaluator_result["exact_match"])
        leaf_type = self.compute_leaf_type(path)

        first_private_barrier_stage = self._first_private_barrier_stage_label(path)
        barrier_stop_prefix = self.compute_shared_upload_stop_prefix(path)
        legal_child_count_per_stage = self._legal_child_count_per_stage(path)
        annotated_stage_trace = self._annotate_stage_trace_with_terminal_details(
            execution.get("stage_trace", []),
            evaluator_result,
        )
        behavior_context = self._prefix_dedup_behavior_context(path)
        family_path_metadata = self._family_path_metadata(path, behavior_context)
        episode_log = {
            "instance_id": self.current_instance_id,
            "selected_path": list(path),
            "leaf_type": leaf_type,
            "stage_trace": annotated_stage_trace,
            "final_action": final_action,
            "oracle_action": oracle_action,
            "terminal_cost": cost_metrics["raw_terminal_penalty"],
            "path_agent_cost": path_agent_cost,
            "total_cost": cost_metrics["normalized_total_cost"],
            "success": success,
            "evaluator_version": evaluator_result["evaluator_version"],
            "false_cancel_count": evaluator_result["false_cancel_count"],
            "missed_cancel_count": evaluator_result["missed_cancel_count"],
            "false_refuse_count": evaluator_result["false_refuse_count"],
            "missed_refuse_count": evaluator_result["missed_refuse_count"],
            "subset_mismatch": evaluator_result["subset_mismatch"],
            "cost_breakdown": deepcopy(evaluator_result["cost_breakdown"]),
            "outcome_cost_breakdown": deepcopy(
                evaluator_result.get("outcome_cost_breakdown", {})
            ),
            "policy_cost_breakdown": deepcopy(
                evaluator_result.get("policy_cost_breakdown", {})
            ),
            "terminal_cost_breakdown": deepcopy(
                evaluator_result.get("terminal_cost_breakdown", {})
            ),
            "family_kind": self.family_kind,
            **family_path_metadata,
            "raw_outcome_penalty": cost_metrics["raw_outcome_penalty"],
            "raw_policy_penalty": cost_metrics["raw_policy_penalty"],
            "raw_terminal_penalty": cost_metrics["raw_terminal_penalty"],
            "legacy_raw_terminal_penalty": cost_metrics["legacy_raw_terminal_penalty"],
            "raw_terminal_penalty_exec_clean_v4": cost_metrics[
                "raw_terminal_penalty_exec_clean_v4"
            ],
            "terminal_adjustment": deepcopy(cost_metrics["terminal_adjustment"]),
            "terminal_adjustment_enabled": bool(
                cost_metrics["terminal_adjustment"].get("enabled")
            ),
            "terminal_adjustment_floor": cost_metrics["terminal_adjustment"].get(
                "applied_floor"
            ),
            "terminal_adjustment_reasons": list(
                cost_metrics["terminal_adjustment"].get("applied_floor_reasons", [])
            ),
            "clear_success_proxy": bool(
                cost_metrics["terminal_adjustment"].get("clear_success_proxy", success)
            ),
            "auxiliary_success_proxy": bool(
                cost_metrics["terminal_adjustment"].get("auxiliary_success_proxy", True)
            ),
            "terminal_majority_pair": cost_metrics["terminal_adjustment"].get(
                "majority_pair"
            ),
            "raw_path_cost_component": cost_metrics["raw_path_cost_component"],
            "raw_reasoning_cost_component_api": cost_metrics[
                "raw_reasoning_cost_component_api"
            ],
            "raw_reasoning_cost_component_token": cost_metrics[
                "raw_reasoning_cost_component_token"
            ],
            "raw_total_cost": cost_metrics["raw_total_cost"],
            "raw_total_cost_api": cost_metrics["raw_total_cost_api"],
            "raw_total_cost_token": cost_metrics["raw_total_cost_token"],
            "raw_reasoning_cost_component": cost_metrics["raw_reasoning_cost_component"],
            "terminal_cost_upper_bound": cost_metrics["terminal_cost_upper_bound"],
            "path_cost_upper_bound": cost_metrics["path_cost_upper_bound"],
            "reasoning_cost_upper_bound": cost_metrics["reasoning_cost_upper_bound"],
            "normalized_terminal_penalty": cost_metrics["normalized_terminal_penalty"],
            "normalized_total_cost": cost_metrics["normalized_total_cost"],
            "total_cost_upper_bound": cost_metrics["total_cost_upper_bound"],
            "cost_scale_version": cost_metrics["cost_scale_version"],
            "reasoning_cost": cost_metrics["raw_reasoning_cost_component"],
            "reasoning_trace": deepcopy(reasoning_metrics["trace"]),
            "reasoning_cost_mode_default": cost_metrics["reasoning_cost_mode_default"],
            "reasoning_weight_calibration_enabled": bool(
                reasoning_metrics.get("reasoning_weight_calibration_enabled", False)
            ),
            "raw_mode_mismatch_cost_component": float(
                reasoning_metrics.get("raw_mode_mismatch_cost_component", 0.0) or 0.0
            ),
            "mode_mismatch_cost_enabled": bool(
                reasoning_metrics.get("mode_mismatch_cost_enabled", False)
            ),
            "mode_mismatch_report_only_enabled": bool(
                reasoning_metrics.get("mode_mismatch_report_only_enabled", False)
            ),
            "mode_mismatch_fast_on_deep_cost": float(
                reasoning_metrics.get("mode_mismatch_fast_on_deep_cost", 0.0) or 0.0
            ),
            "mode_mismatch_deep_on_fast_cost": float(
                reasoning_metrics.get("mode_mismatch_deep_on_fast_cost", 0.0) or 0.0
            ),
            "prompt_tokens_total": reasoning_metrics["prompt_tokens_total"],
            "completion_tokens_total": reasoning_metrics["completion_tokens_total"],
            "total_tokens_total": reasoning_metrics["total_tokens_total"],
            "api_cost_total_usd_raw": reasoning_metrics["api_cost_total_usd_raw"],
            "reasoning_resource_breakdown": deepcopy(
                reasoning_metrics["reasoning_resource_breakdown"]
            ),
            "generation_time_total_seconds": reasoning_metrics[
                "generation_time_total_seconds"
            ],
            "llm_round_trip_total_seconds": reasoning_metrics[
                "llm_round_trip_total_seconds"
            ],
            "tool_wall_clock_total_seconds": reasoning_metrics[
                "tool_wall_clock_total_seconds"
            ],
            "episode_wall_clock_seconds": reasoning_metrics["episode_wall_clock_seconds"],
            "latency_breakdown": deepcopy(reasoning_metrics["latency_breakdown"]),
            "policy_action_violation": bool(
                evaluator_result.get("policy_action_violation", False)
            ),
            "policy_communication_violation": bool(
                evaluator_result.get("policy_communication_violation", False)
            ),
            "policy_nl_assertions_total": int(
                evaluator_result.get("policy_nl_assertions_total", 0) or 0
            ),
            "policy_nl_assertions_failed": int(
                evaluator_result.get("policy_nl_assertions_failed", 0) or 0
            ),
            "policy_violation_count": int(
                evaluator_result.get("policy_violation_count", 0) or 0
            ),
            "policy_violation_breakdown": deepcopy(
                evaluator_result.get("policy_violation_breakdown", {})
            ),
            "policy_eval_source": evaluator_result.get("policy_eval_source"),
            "policy_eval_scope": evaluator_result.get("policy_eval_scope"),
            "selected_shared_path": leaf_type == "shared",
            "selected_unshared_path": leaf_type != "shared",
            "first_private_barrier_stage": first_private_barrier_stage,
            "barrier_stop_depth": len(barrier_stop_prefix) if barrier_stop_prefix else None,
            "legal_child_count_per_stage": legal_child_count_per_stage,
            "candidate_count_per_stage": list(legal_child_count_per_stage),
        }
        if isinstance(execution.get("bench_aux_eval"), dict):
            episode_log["bench_aux_eval"] = deepcopy(execution["bench_aux_eval"])
        if isinstance(execution.get("family_behavior"), dict):
            episode_log["family_behavior"] = deepcopy(execution["family_behavior"])
        self._last_episode_log = deepcopy(episode_log)

        return EpisodeResult(
            instance_id=self.current_instance_id or "unknown_instance",
            selected_path=list(path),
            leaf_type=leaf_type,
            stage_outputs=stage_outputs,
            final_action=final_action,
            oracle_action=oracle_action,
            terminal_cost=cost_metrics["raw_terminal_penalty"],
            raw_terminal_penalty=cost_metrics["raw_terminal_penalty"],
            raw_path_cost_component=cost_metrics["raw_path_cost_component"],
            raw_reasoning_cost_component=cost_metrics["raw_reasoning_cost_component"],
            raw_total_cost=cost_metrics["raw_total_cost"],
            normalized_terminal_penalty=cost_metrics["normalized_terminal_penalty"],
            success=success,
            path_agent_cost=path_agent_cost,
            reasoning_cost=cost_metrics["raw_reasoning_cost_component"],
            total_cost=cost_metrics["normalized_total_cost"],
            total_cost_upper_bound=cost_metrics["total_cost_upper_bound"],
            cost_scale_version=cost_metrics["cost_scale_version"],
            raw_outcome_penalty=cost_metrics["raw_outcome_penalty"],
            raw_policy_penalty=cost_metrics["raw_policy_penalty"],
            raw_reasoning_cost_component_api=cost_metrics[
                "raw_reasoning_cost_component_api"
            ],
            raw_reasoning_cost_component_token=cost_metrics[
                "raw_reasoning_cost_component_token"
            ],
            raw_total_cost_api=cost_metrics["raw_total_cost_api"],
            raw_total_cost_token=cost_metrics["raw_total_cost_token"],
            prompt_tokens_total=reasoning_metrics["prompt_tokens_total"],
            completion_tokens_total=reasoning_metrics["completion_tokens_total"],
            total_tokens_total=reasoning_metrics["total_tokens_total"],
            api_cost_total_usd_raw=reasoning_metrics["api_cost_total_usd_raw"],
            generation_time_total_seconds=reasoning_metrics[
                "generation_time_total_seconds"
            ],
            llm_round_trip_total_seconds=reasoning_metrics[
                "llm_round_trip_total_seconds"
            ],
            tool_wall_clock_total_seconds=reasoning_metrics[
                "tool_wall_clock_total_seconds"
            ],
            episode_wall_clock_seconds=reasoning_metrics["episode_wall_clock_seconds"],
            episode_log=episode_log,
        )

    def _family_stage_outputs_from_execution(self, execution: JsonDict) -> dict[str, JsonDict]:
        stage_outputs: dict[str, JsonDict] = {}
        for row in execution.get("stage_trace", []):
            stage_name = row["stage_name"]
            stage_outputs[stage_name] = {
                "input": deepcopy(row.get("input", {})),
                "output": deepcopy(row.get("output", {})),
                "source": "simulated_executor",
            }
        stage_outputs["stage5"] = {
            "input": deepcopy(stage_outputs.get("stage5", {}).get("input", {})),
            "output": {
                "final_action": execution.get("final_action"),
                "cancelled_reservation_ids": list(
                    execution.get("cancelled_reservation_ids", [])
                ),
                "refused_reservation_ids": list(
                    execution.get("refused_reservation_ids", [])
                ),
                "selected_blocker_ids": list(
                    execution.get("selected_blocker_ids", execution.get("cancelled_reservation_ids", []))
                ),
                "deferred_blocker_ids": list(
                    execution.get("deferred_blocker_ids", execution.get("refused_reservation_ids", []))
                ),
            },
            "source": "simulated_executor",
        }
        return stage_outputs

    def compute_leaf_type(self, path: list[str]) -> str:
        """Compute shared/unshared leaf type from the sampled leaf start condition."""

        self._validate_path(path)
        return "shared" if self.leaf_starts_shared_upload(path) else "unshared"

    def leaf_starts_shared_upload(self, path: list[str]) -> bool:
        """Return whether the sampled leaf can start a shared upload."""

        self._validate_path(path)
        return leaf_starts_shared_upload(path, self.agent_catalog)

    def compute_shared_upload_edges(self, path: list[str]) -> list[EdgeKey]:
        """Return the upward shared-upload edges for one sampled path."""

        self._validate_path(path)
        return compute_shared_upload_edges(path, self.agent_catalog)

    def compute_shared_upload_stop_prefix(self, path: list[str]) -> PrefixKey | None:
        """Return the internal barrier node where upward upload stops, if any."""

        self._validate_path(path)
        return compute_shared_upload_stop_prefix(path, self.agent_catalog)

    def compute_terminal_cost(
        self,
        stage_outputs: dict[str, JsonDict],
        path: list[str],
    ) -> float:
        """Compute terminal cost using the reservation-level evaluator."""

        del path
        return float(self.evaluate_terminal_outcome(stage_outputs, []).get("terminal_penalty", 0.0))

    def evaluate_terminal_outcome(
        self,
        stage_outputs: dict[str, JsonDict],
        path: list[str],
        execution: JsonDict | None = None,
    ) -> JsonDict:
        """Run evaluator v2 on the terminal stage output."""

        del path
        if self.current_instance is None:
            raise RuntimeError("No current instance loaded.")
        predicted_stage5_output = stage_outputs.get("stage5", {}).get("output", {})
        family = self.current_instance.get("family")
        if family == "telecom_mms_recovery":
            policy_eval = None
            if isinstance((execution or {}).get("bench_aux_eval"), dict):
                policy_eval = (execution or {}).get("bench_aux_eval")
            return evaluate_telecom_terminal_prediction(
                instance=self.current_instance,
                predicted_stage5_output=predicted_stage5_output,
                policy_eval_result=policy_eval,
            )
        return evaluate_terminal_prediction(
            instance=self.current_instance,
            predicted_stage5_output=predicted_stage5_output,
        )

    def _select_task_adapter(self, instance: JsonDict):
        family = instance.get("family")
        if family == "telecom_mms_recovery":
            return TelecomMMSTaskAdapter()
        return AirlineTaskAdapter()

    def _path_agent_cost_weight(self) -> float:
        override = os.environ.get("PSAGENT_PATH_AGENT_COST_WEIGHT")
        if override is not None and override.strip():
            value = float(override)
            if value < 0:
                raise ValueError("PSAGENT_PATH_AGENT_COST_WEIGHT must be non-negative.")
            return value
        if self.current_instance and self.current_instance.get("family") == "telecom_mms_recovery":
            return TELECOM_DEFAULT_COST_SPEC.path_agent_cost_weight
        return DEFAULT_COST_SPEC.path_agent_cost_weight

    def _first_private_barrier_stage_label(self, path: list[str]) -> str | None:
        depth = compute_first_private_barrier_depth(path, self.agent_catalog)
        if depth is None:
            return None
        return self.STAGE_NAMES[depth - 1]

    def _legal_child_count_per_stage(self, path: list[str]) -> list[int]:
        counts: list[int] = []
        prefix: tuple[str, ...] = ()
        allowed_children = getattr(self.family_spec, "allowed_children", None)
        for stage_name, agent_id in zip(self.STAGE_NAMES, path):
            if allowed_children:
                legal_children = allowed_children.get(prefix)
                if legal_children is None:
                    legal_children = [agent.agent_id for agent in self.agents_by_stage[stage_name]]
            else:
                legal_children = [agent.agent_id for agent in self.agents_by_stage[stage_name]]
            counts.append(len(legal_children))
            prefix = prefix + (agent_id,)
        return counts

    def _stage_resource_snapshot(self, stage_row: Mapping[str, Any]) -> JsonDict:
        llm_raw_output = list(stage_row.get("llm_raw_output", []) or [])
        usage_totals = _normalize_usage_breakdown(stage_row.get("usage_breakdown_stage"))
        if usage_totals["total_tokens_total"] <= 0.0 and llm_raw_output:
            for message in llm_raw_output:
                usage_breakdown = _normalize_usage_breakdown(
                    message.get("usage_breakdown") or message.get("usage")
                )
                usage_totals["prompt_tokens_total"] += usage_breakdown["prompt_tokens_total"]
                usage_totals["completion_tokens_total"] += usage_breakdown[
                    "completion_tokens_total"
                ]
                usage_totals["total_tokens_total"] += usage_breakdown["total_tokens_total"]

        llm_call_count = int(
            stage_row.get("llm_call_count_stage", len(llm_raw_output)) or 0
        )
        api_cost_total_usd_stage = float(
            stage_row.get(
                "api_cost_total_usd_stage",
                sum(float(message.get("cost") or 0.0) for message in llm_raw_output),
            )
            or 0.0
        )
        generation_time_total_seconds_stage = float(
            stage_row.get(
                "generation_time_total_seconds_stage",
                sum(
                    float(message.get("generation_time_seconds") or 0.0)
                    for message in llm_raw_output
                ),
            )
            or 0.0
        )
        llm_round_trip_total_seconds_stage = float(
            stage_row.get(
                "llm_round_trip_total_seconds_stage",
                sum(float(message.get("round_trip_seconds") or 0.0) for message in llm_raw_output),
            )
            or 0.0
        )
        tool_wall_clock_total_seconds_stage = float(
            stage_row.get(
                "tool_wall_clock_total_seconds_stage",
                sum(
                    float(call.get("wall_clock_seconds") or 0.0)
                    for call in [
                        *(stage_row.get("executed_tool_calls", []) or []),
                        *(stage_row.get("replay_tool_calls", []) or []),
                    ]
                ),
            )
            or 0.0
        )
        stage_wall_clock_seconds = float(
            stage_row.get(
                "stage_wall_clock_seconds",
                llm_round_trip_total_seconds_stage + tool_wall_clock_total_seconds_stage,
            )
            or 0.0
        )
        return {
            "llm_call_count_stage": llm_call_count,
            "prompt_tokens_total_stage": usage_totals["prompt_tokens_total"],
            "completion_tokens_total_stage": usage_totals["completion_tokens_total"],
            "total_tokens_total_stage": usage_totals["total_tokens_total"],
            "api_cost_total_usd_stage": api_cost_total_usd_stage,
            "generation_time_total_seconds_stage": generation_time_total_seconds_stage,
            "llm_round_trip_total_seconds_stage": llm_round_trip_total_seconds_stage,
            "tool_wall_clock_total_seconds_stage": tool_wall_clock_total_seconds_stage,
            "stage_wall_clock_seconds": stage_wall_clock_seconds,
            "usage_breakdown_stage": {
                "prompt_tokens_total": usage_totals["prompt_tokens_total"],
                "completion_tokens_total": usage_totals["completion_tokens_total"],
                "total_tokens_total": usage_totals["total_tokens_total"],
            },
            "cost_breakdown_stage": {
                "api_cost_total_usd_raw": api_cost_total_usd_stage,
            },
        }

    def _aggregate_stage_resource_metrics(self, stage_trace: Sequence[Mapping[str, Any]]) -> JsonDict:
        stage_trace_map = {row["stage_name"]: row for row in stage_trace if "stage_name" in row}
        stage_prompt_tokens: list[float] = []
        stage_completion_tokens: list[float] = []
        stage_total_tokens: list[float] = []
        stage_api_cost_usd: list[float] = []
        stage_generation_time_seconds: list[float] = []
        stage_llm_round_trip_seconds: list[float] = []
        stage_tool_wall_clock_seconds: list[float] = []
        stage_wall_clock_seconds: list[float] = []
        reasoning_trace: list[JsonDict] = []
        totals = {
            "llm_call_count": 0,
            "prompt_tokens_total": 0.0,
            "completion_tokens_total": 0.0,
            "total_tokens_total": 0.0,
            "api_cost_total_usd_raw": 0.0,
            "generation_time_total_seconds": 0.0,
            "llm_round_trip_total_seconds": 0.0,
            "tool_wall_clock_total_seconds": 0.0,
            "episode_wall_clock_seconds": 0.0,
        }
        for stage_name in self.STAGE_NAMES:
            snapshot = self._stage_resource_snapshot(stage_trace_map.get(stage_name, {}))
            stage_prompt_tokens.append(snapshot["prompt_tokens_total_stage"])
            stage_completion_tokens.append(snapshot["completion_tokens_total_stage"])
            stage_total_tokens.append(snapshot["total_tokens_total_stage"])
            stage_api_cost_usd.append(snapshot["api_cost_total_usd_stage"])
            stage_generation_time_seconds.append(snapshot["generation_time_total_seconds_stage"])
            stage_llm_round_trip_seconds.append(
                snapshot["llm_round_trip_total_seconds_stage"]
            )
            stage_tool_wall_clock_seconds.append(
                snapshot["tool_wall_clock_total_seconds_stage"]
            )
            stage_wall_clock_seconds.append(snapshot["stage_wall_clock_seconds"])
            totals["llm_call_count"] += int(snapshot["llm_call_count_stage"])
            totals["prompt_tokens_total"] += snapshot["prompt_tokens_total_stage"]
            totals["completion_tokens_total"] += snapshot["completion_tokens_total_stage"]
            totals["total_tokens_total"] += snapshot["total_tokens_total_stage"]
            totals["api_cost_total_usd_raw"] += snapshot["api_cost_total_usd_stage"]
            totals["generation_time_total_seconds"] += snapshot[
                "generation_time_total_seconds_stage"
            ]
            totals["llm_round_trip_total_seconds"] += snapshot[
                "llm_round_trip_total_seconds_stage"
            ]
            totals["tool_wall_clock_total_seconds"] += snapshot[
                "tool_wall_clock_total_seconds_stage"
            ]
            totals["episode_wall_clock_seconds"] += snapshot["stage_wall_clock_seconds"]
            reasoning_trace.append(
                {
                    "stage_name": stage_name,
                    **snapshot,
                }
            )
        return {
            **totals,
            "stage_prompt_tokens": stage_prompt_tokens,
            "stage_completion_tokens": stage_completion_tokens,
            "stage_total_tokens": stage_total_tokens,
            "stage_api_cost_usd": stage_api_cost_usd,
            "stage_generation_time_seconds": stage_generation_time_seconds,
            "stage_llm_round_trip_seconds": stage_llm_round_trip_seconds,
            "stage_tool_wall_clock_seconds": stage_tool_wall_clock_seconds,
            "stage_wall_clock_seconds": stage_wall_clock_seconds,
            "reasoning_trace": reasoning_trace,
            "reasoning_resource_breakdown": {
                "prompt_tokens_total": totals["prompt_tokens_total"],
                "completion_tokens_total": totals["completion_tokens_total"],
                "total_tokens_total": totals["total_tokens_total"],
                "api_cost_total_usd_raw": totals["api_cost_total_usd_raw"],
            },
            "latency_breakdown": {
                "generation_time_total_seconds": totals["generation_time_total_seconds"],
                "llm_round_trip_total_seconds": totals["llm_round_trip_total_seconds"],
                "tool_wall_clock_total_seconds": totals["tool_wall_clock_total_seconds"],
                "episode_wall_clock_seconds": totals["episode_wall_clock_seconds"],
            },
        }

    def _annotate_stage_trace_with_terminal_details(
        self,
        stage_trace: Sequence[Mapping[str, Any]],
        evaluator_result: Mapping[str, Any],
    ) -> list[JsonDict]:
        annotated = [deepcopy(dict(row)) for row in stage_trace]
        for row in annotated:
            row.update(self._stage_resource_snapshot(row))
            if row.get("stage_name") == "stage5":
                row["bench_action_check_raw"] = deepcopy(
                    evaluator_result.get("bench_action_check_raw", row.get("bench_action_check_raw", []))
                )
                row["bench_communicate_check_raw"] = deepcopy(
                    evaluator_result.get(
                        "bench_communicate_check_raw",
                        row.get("bench_communicate_check_raw", []),
                    )
                )
                row["bench_nl_assertions_raw"] = deepcopy(
                    evaluator_result.get(
                        "bench_nl_assertions_raw",
                        row.get("bench_nl_assertions_raw", []),
                    )
                )
                row["policy_action_violation_stage5"] = bool(
                    evaluator_result.get("policy_action_violation", False)
                )
                row["policy_communication_violation_stage5"] = bool(
                    evaluator_result.get("policy_communication_violation", False)
                )
                row["policy_nl_assertions_total_stage5"] = int(
                    evaluator_result.get("policy_nl_assertions_total", 0) or 0
                )
                row["policy_nl_assertions_failed_stage5"] = int(
                    evaluator_result.get("policy_nl_assertions_failed", 0) or 0
                )
                row["raw_policy_penalty_stage5"] = float(
                    evaluator_result.get("raw_policy_penalty", 0.0) or 0.0
                )
                row["raw_outcome_penalty_stage5"] = float(
                    evaluator_result.get("raw_outcome_penalty", 0.0) or 0.0
                )
                row["policy_eval_source_stage5"] = evaluator_result.get(
                    "policy_eval_source"
                )
        return annotated

    def _build_cost_metrics(
        self,
        *,
        evaluator_result: JsonDict,
        path_agent_cost: float,
        reasoning_metrics: JsonDict,
    ) -> JsonDict:
        raw_outcome_penalty = float(evaluator_result.get("raw_outcome_penalty", 0.0) or 0.0)
        raw_policy_penalty = float(evaluator_result.get("raw_policy_penalty", 0.0) or 0.0)
        legacy_raw_terminal_penalty = float(
            evaluator_result.get(
                "raw_terminal_penalty",
                evaluator_result.get("terminal_penalty", raw_outcome_penalty + raw_policy_penalty),
            )
        )
        raw_terminal_penalty = legacy_raw_terminal_penalty
        terminal_adjustment: JsonDict = {
            "enabled": False,
            "version": "legacy",
            "legacy_raw_terminal_penalty": legacy_raw_terminal_penalty,
            "raw_terminal_penalty_exec_clean_v4": None,
            "applied_floor": None,
            "applied_floor_reasons": [],
            "subset_mismatch_base_penalty": 0.0,
            "clear_success_proxy": bool(evaluator_result.get("exact_match", False))
            and not bool(evaluator_result.get("subset_mismatch", False)),
            "auxiliary_success_proxy": int(
                evaluator_result.get("policy_violation_count", 0) or 0
            )
            == 0,
            "actual_majority_mode": None,
            "required_majority_mode": None,
            "majority_pair": None,
        }

        if (
            self.current_instance
            and self.current_instance.get("family") == "telecom_mms_recovery"
            and _env_flag("PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4")
        ):
            trace = list(reasoning_metrics.get("trace", []) or [])
            actual_modes = [
                self._normalize_deliberation_mode(row.get("deliberation_mode", "deep"))
                for row in trace
            ]
            required_modes = [
                self._normalize_deliberation_mode(
                    row.get("deliberation_requirement", "deep")
                )
                for row in trace
            ]
            actual_majority = (
                "mostly_fast"
                if actual_modes.count("fast") > actual_modes.count("deep")
                else "mostly_deep"
            )
            required_majority = (
                "mostly_fast_required"
                if required_modes.count("fast") > required_modes.count("deep")
                else "mostly_deep_required"
            )
            majority_pair = f"{actual_majority}_vs_{required_majority}"
            clear_success = bool(evaluator_result.get("exact_match", False)) and not bool(
                evaluator_result.get("subset_mismatch", False)
            )
            auxiliary_success = int(evaluator_result.get("policy_violation_count", 0) or 0) == 0
            oracle_action = str(evaluator_result.get("oracle_final_action", ""))
            predicted_action = str(evaluator_result.get("predicted_final_action", ""))
            local_repair_task = oracle_action in {"repair_all", "repair_subset"}
            subset_mismatch = bool(evaluator_result.get("subset_mismatch", False))

            candidate_terminal = legacy_raw_terminal_penalty
            floor = 0.0
            reasons: list[str] = []
            subset_mismatch_base_penalty = 0.0

            if local_repair_task and subset_mismatch:
                subset_mismatch_base_penalty = 4.0
                candidate_terminal += subset_mismatch_base_penalty
                reasons.append("subset_mismatch_base_plus_linear")
            if local_repair_task and not clear_success:
                floor = max(floor, 10.0)
                reasons.append("local_clear_failure_floor_10")
            if local_repair_task and (not clear_success) and (not auxiliary_success):
                floor = max(floor, 12.0)
                reasons.append("local_clear_and_aux_failure_floor_12")
            if (
                local_repair_task
                and required_majority == "mostly_deep_required"
                and actual_majority == "mostly_fast"
                and not clear_success
            ):
                floor = max(floor, 14.0)
                reasons.append("fast_path_on_deep_required_clear_failure_floor_14")
            if local_repair_task and predicted_action == "transfer":
                floor = max(floor, 18.0)
                reasons.append("invalid_local_transfer_floor_18")
            if local_repair_task and clear_success and not auxiliary_success:
                floor = max(floor, 6.0)
                reasons.append("clear_but_aux_failure_floor_6")

            raw_terminal_penalty = max(candidate_terminal, floor)
            terminal_adjustment = {
                "enabled": True,
                "version": "exec_clean_terminal_v4",
                "legacy_raw_terminal_penalty": legacy_raw_terminal_penalty,
                "raw_terminal_penalty_exec_clean_v4": raw_terminal_penalty,
                "applied_floor": floor if floor > 0.0 else None,
                "applied_floor_reasons": reasons,
                "subset_mismatch_base_penalty": subset_mismatch_base_penalty,
                "clear_success_proxy": clear_success,
                "auxiliary_success_proxy": auxiliary_success,
                "actual_majority_mode": actual_majority,
                "required_majority_mode": required_majority,
                "majority_pair": majority_pair,
            }
        raw_path_cost_component = self._path_agent_cost_weight() * float(path_agent_cost)
        raw_reasoning_cost_component = float(
            reasoning_metrics.get("raw_reasoning_cost_component", 0.0) or 0.0
        )
        raw_reasoning_cost_component_api = reasoning_metrics.get(
            "raw_reasoning_cost_component_api"
        )
        raw_reasoning_cost_component_token = reasoning_metrics.get(
            "raw_reasoning_cost_component_token"
        )
        raw_total_cost_api = (
            raw_terminal_penalty
            + raw_path_cost_component
            + float(raw_reasoning_cost_component_api)
            if raw_reasoning_cost_component_api is not None
            else None
        )
        raw_total_cost_token = (
            raw_terminal_penalty
            + raw_path_cost_component
            + float(raw_reasoning_cost_component_token)
            if raw_reasoning_cost_component_token is not None
            else None
        )
        raw_total_cost = raw_terminal_penalty + raw_path_cost_component + raw_reasoning_cost_component
        terminal_cost_upper_bound = None
        path_cost_upper_bound = None
        reasoning_cost_upper_bound = None

        if self.current_instance and self.current_instance.get("family") == "telecom_mms_recovery":
            terminal_cost_upper_bound = float(
                evaluator_result.get(
                    "terminal_cost_upper_bound",
                    TELECOM_MMS_TERMINAL_UPPER_BOUND_V2,
                )
            )
            if terminal_adjustment.get("enabled"):
                terminal_cost_upper_bound = max(
                    terminal_cost_upper_bound,
                    TELECOM_EXEC_CLEAN_V4_TERMINAL_UPPER_BOUND,
                )
            path_cost_upper_bound = TELECOM_MMS_PATH_UPPER_BOUND_V2
            if reasoning_metrics.get("reasoning_cost_mode_default") == "api":
                reasoning_cost_upper_bound = TELECOM_MMS_REASONING_UPPER_BOUND_API_V2
            else:
                reasoning_cost_upper_bound = TELECOM_MMS_REASONING_UPPER_BOUND_TOKEN_V2
            normalized_terminal_penalty = min(
                raw_terminal_penalty / terminal_cost_upper_bound,
                1.0,
            )
            total_cost_upper_bound = (
                terminal_cost_upper_bound
                + path_cost_upper_bound
                + reasoning_cost_upper_bound
            )
            normalized_total_cost = min(raw_total_cost / total_cost_upper_bound, 1.0)
            cost_scale_version = (
                f"{TELECOM_MMS_COST_SCALE_VERSION}_{reasoning_metrics.get('reasoning_cost_mode_default', 'default')}"
            )
            if terminal_adjustment.get("enabled"):
                cost_scale_version = f"{cost_scale_version}_exec_clean_terminal_v4"
            if reasoning_metrics.get("mode_mismatch_cost_enabled"):
                cost_scale_version = f"{cost_scale_version}_mode_mismatch_cost_v2"
        else:
            normalized_terminal_penalty = raw_terminal_penalty
            normalized_total_cost = raw_total_cost
            total_cost_upper_bound = raw_total_cost
            cost_scale_version = "raw_cost_unscaled"

        return {
            "raw_outcome_penalty": raw_outcome_penalty,
            "raw_policy_penalty": raw_policy_penalty,
            "raw_terminal_penalty": raw_terminal_penalty,
            "legacy_raw_terminal_penalty": legacy_raw_terminal_penalty,
            "raw_terminal_penalty_exec_clean_v4": terminal_adjustment.get(
                "raw_terminal_penalty_exec_clean_v4"
            ),
            "terminal_adjustment": terminal_adjustment,
            "raw_path_cost_component": raw_path_cost_component,
            "raw_reasoning_cost_component": raw_reasoning_cost_component,
            "raw_reasoning_cost_component_api": raw_reasoning_cost_component_api,
            "raw_reasoning_cost_component_token": raw_reasoning_cost_component_token,
            "raw_total_cost": raw_total_cost,
            "raw_total_cost_api": raw_total_cost_api,
            "raw_total_cost_token": raw_total_cost_token,
            "normalized_terminal_penalty": normalized_terminal_penalty,
            "normalized_total_cost": normalized_total_cost,
            "terminal_cost_upper_bound": terminal_cost_upper_bound,
            "path_cost_upper_bound": path_cost_upper_bound,
            "reasoning_cost_upper_bound": reasoning_cost_upper_bound,
            "total_cost_upper_bound": total_cost_upper_bound,
            "cost_scale_version": cost_scale_version,
            "reasoning_cost_mode_default": reasoning_metrics.get("reasoning_cost_mode_default"),
        }

    def _compute_family_reasoning_cost(
        self,
        path: list[str],
        *,
        stage_trace: Sequence[Mapping[str, Any]] | None = None,
    ) -> JsonDict:
        if self.family_agent_map is None or self.current_task_descriptor is None:
            return {
                "raw_reasoning_cost_component": 0.0,
                "raw_reasoning_cost_component_api": None,
                "raw_reasoning_cost_component_token": None,
                "reasoning_cost_mode_default": None,
                "trace": [],
            }

        if self._family_stages is None:
            return {
                "raw_reasoning_cost_component": 0.0,
                "raw_reasoning_cost_component_api": None,
                "raw_reasoning_cost_component_token": None,
                "reasoning_cost_mode_default": None,
                "trace": [],
            }

        if self.executor_name == "llm_bench":
            resource_metrics = self._aggregate_stage_resource_metrics(stage_trace or [])
            stage_trace_map = {
                str(row["stage_name"]): row for row in (stage_trace or []) if "stage_name" in row
            }
            reasoning_trace: list[JsonDict] = []
            raw_reasoning_cost_component_api = 0.0
            raw_reasoning_cost_component_token = 0.0
            mode_mismatch_cost_enabled = _env_flag(
                "PSAGENT_TELECOM_MODE_MISMATCH_COST_V2"
            )
            mode_mismatch_report_only_enabled = _env_flag(
                "PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2"
            )
            raw_mode_mismatch_cost_component = 0.0

            for stage_name, agent_id in zip(self._family_stages, path):
                snapshot = self._stage_resource_snapshot(stage_trace_map.get(stage_name, {}))
                agent = self.family_agent_map[agent_id]
                requirement = self._stage_deliberation_requirement(
                    self.current_task_descriptor,
                    stage_name,
                )
                realized_mode = getattr(agent, "deliberation_mode", "deep")
                multiplier = self._reasoning_match_multiplier(
                    requirement=requirement,
                    actual_mode=realized_mode,
                )
                stage_reasoning_components = compute_llm_bench_reasoning_components(
                    prompt_tokens_total=snapshot["prompt_tokens_total_stage"],
                    completion_tokens_total=snapshot["completion_tokens_total_stage"],
                    api_cost_total_usd_raw=snapshot["api_cost_total_usd_stage"],
                )
                base_stage_api = float(
                    stage_reasoning_components.get("raw_reasoning_cost_component_api", 0.0) or 0.0
                )
                base_stage_token = float(
                    stage_reasoning_components.get("raw_reasoning_cost_component_token", 0.0) or 0.0
                )
                normalized_requirement = self._normalize_deliberation_mode(requirement)
                normalized_mode = self._normalize_deliberation_mode(realized_mode)
                mode_mismatch_stage_cost = 0.0
                if mode_mismatch_cost_enabled or mode_mismatch_report_only_enabled:
                    if normalized_requirement == "deep" and normalized_mode == "fast":
                        mode_mismatch_stage_cost = TELECOM_MODE_MISMATCH_FAST_ON_DEEP_COST_V2
                    elif normalized_requirement == "fast" and normalized_mode == "deep":
                        mode_mismatch_stage_cost = TELECOM_MODE_MISMATCH_DEEP_ON_FAST_COST_V2
                weighted_stage_api = round(base_stage_api * multiplier, 6)
                weighted_stage_token = round(base_stage_token * multiplier, 6)
                if mode_mismatch_cost_enabled and mode_mismatch_stage_cost:
                    weighted_stage_api = round(weighted_stage_api + mode_mismatch_stage_cost, 6)
                    weighted_stage_token = round(
                        weighted_stage_token + mode_mismatch_stage_cost,
                        6,
                    )
                raw_reasoning_cost_component_api += weighted_stage_api
                raw_reasoning_cost_component_token += weighted_stage_token
                raw_mode_mismatch_cost_component += mode_mismatch_stage_cost
                reasoning_trace.append(
                    {
                        "stage_name": stage_name,
                        "agent_id": agent_id,
                        "deliberation_requirement": normalized_requirement,
                        "deliberation_mode": normalized_mode,
                        "reasoning_match_multiplier": multiplier,
                        "mode_mismatch_cost_enabled": mode_mismatch_cost_enabled,
                        "mode_mismatch_report_only_enabled": (
                            mode_mismatch_report_only_enabled
                        ),
                        "mode_mismatch_stage_cost": mode_mismatch_stage_cost,
                        "base_reasoning_cost_api": round(base_stage_api, 6),
                        "base_reasoning_cost_token": round(base_stage_token, 6),
                        "weighted_reasoning_cost_api": weighted_stage_api,
                        "weighted_reasoning_cost_token": weighted_stage_token,
                        **snapshot,
                    }
                )

            reasoning_default_mode = LLM_BENCH_REASONING_DEFAULT_MODE
            raw_reasoning_cost_component = (
                raw_reasoning_cost_component_api
                if reasoning_default_mode == "api"
                else raw_reasoning_cost_component_token
            )
            reasoning_components = {
                "alpha_api": LLM_BENCH_REASONING_ALPHA_API,
                "alpha_in": LLM_BENCH_REASONING_ALPHA_IN,
                "alpha_out": LLM_BENCH_REASONING_ALPHA_OUT,
                "reasoning_cost_mode_default": reasoning_default_mode,
                "reasoning_weight_calibration_enabled": _env_flag(
                    "PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3"
                ),
                "raw_reasoning_cost_component": round(raw_reasoning_cost_component, 6),
                "raw_reasoning_cost_component_api": round(
                    raw_reasoning_cost_component_api, 6
                ),
                "raw_reasoning_cost_component_token": round(
                    raw_reasoning_cost_component_token, 6
                ),
                "raw_mode_mismatch_cost_component": round(
                    raw_mode_mismatch_cost_component,
                    6,
                ),
                "mode_mismatch_cost_enabled": mode_mismatch_cost_enabled,
                "mode_mismatch_report_only_enabled": mode_mismatch_report_only_enabled,
                "mode_mismatch_fast_on_deep_cost": (
                    TELECOM_MODE_MISMATCH_FAST_ON_DEEP_COST_V2
                    if mode_mismatch_cost_enabled or mode_mismatch_report_only_enabled
                    else 0.0
                ),
                "mode_mismatch_deep_on_fast_cost": (
                    TELECOM_MODE_MISMATCH_DEEP_ON_FAST_COST_V2
                    if mode_mismatch_cost_enabled or mode_mismatch_report_only_enabled
                    else 0.0
                ),
            }
            return {
                **resource_metrics,
                **reasoning_components,
                "trace": reasoning_trace,
            }

        if self.executor_name != "simulated":
            return {
                "raw_reasoning_cost_component": 0.0,
                "raw_reasoning_cost_component_api": None,
                "raw_reasoning_cost_component_token": None,
                "reasoning_cost_mode_default": None,
                "trace": [],
                "prompt_tokens_total": 0.0,
                "completion_tokens_total": 0.0,
                "total_tokens_total": 0.0,
                "api_cost_total_usd_raw": 0.0,
                "generation_time_total_seconds": 0.0,
                "llm_round_trip_total_seconds": 0.0,
                "tool_wall_clock_total_seconds": 0.0,
                "episode_wall_clock_seconds": 0.0,
                "stage_prompt_tokens": [0.0] * len(self.STAGE_NAMES),
                "stage_completion_tokens": [0.0] * len(self.STAGE_NAMES),
                "stage_total_tokens": [0.0] * len(self.STAGE_NAMES),
                "stage_api_cost_usd": [0.0] * len(self.STAGE_NAMES),
                "stage_generation_time_seconds": [0.0] * len(self.STAGE_NAMES),
                "stage_llm_round_trip_seconds": [0.0] * len(self.STAGE_NAMES),
                "stage_tool_wall_clock_seconds": [0.0] * len(self.STAGE_NAMES),
                "stage_wall_clock_seconds": [0.0] * len(self.STAGE_NAMES),
                "reasoning_resource_breakdown": {
                    "prompt_tokens_total": 0.0,
                    "completion_tokens_total": 0.0,
                    "total_tokens_total": 0.0,
                    "api_cost_total_usd_raw": 0.0,
                },
                "latency_breakdown": {
                    "generation_time_total_seconds": 0.0,
                    "llm_round_trip_total_seconds": 0.0,
                    "tool_wall_clock_total_seconds": 0.0,
                    "episode_wall_clock_seconds": 0.0,
                },
            }

        trace: list[JsonDict] = []
        total = 0.0
        for stage_name, agent_id in zip(self._family_stages, path):
            agent = self.family_agent_map[agent_id]
            requirement = self._stage_deliberation_requirement(self.current_task_descriptor, stage_name)
            stage_cost = 0.012 if getattr(agent, "deliberation_mode", "deep") == "fast" else 0.028
            mismatch_penalty = 0.0
            if requirement == "deep" and getattr(agent, "deliberation_mode", "deep") == "fast":
                mismatch_penalty = 0.022 + (0.01 * self.current_task_descriptor.stage_difficulty.get(stage_name, 0.0))
            elif requirement == "fast" and getattr(agent, "deliberation_mode", "deep") == "deep":
                mismatch_penalty = 0.012
            stage_total = round(stage_cost + mismatch_penalty, 6)
            total += stage_total
            trace.append(
                {
                    "stage_name": stage_name,
                    "agent_id": agent_id,
                    "deliberation_mode": getattr(agent, "deliberation_mode", "deep"),
                    "deliberation_requirement": requirement,
                    "base_reasoning_cost": round(stage_cost, 6),
                    "mismatch_penalty": round(mismatch_penalty, 6),
                    "stage_reasoning_cost": stage_total,
                }
            )

        return {
            "raw_reasoning_cost_component": round(total, 6),
            "raw_reasoning_cost_component_api": None,
            "raw_reasoning_cost_component_token": None,
            "reasoning_cost_mode_default": "simulated_proxy",
            "trace": trace,
            "prompt_tokens_total": 0.0,
            "completion_tokens_total": 0.0,
            "total_tokens_total": 0.0,
            "api_cost_total_usd_raw": 0.0,
            "generation_time_total_seconds": 0.0,
            "llm_round_trip_total_seconds": 0.0,
            "tool_wall_clock_total_seconds": 0.0,
            "episode_wall_clock_seconds": 0.0,
            "stage_prompt_tokens": [0.0] * len(self._family_stages),
            "stage_completion_tokens": [0.0] * len(self._family_stages),
            "stage_total_tokens": [0.0] * len(self._family_stages),
            "stage_api_cost_usd": [0.0] * len(self._family_stages),
            "stage_generation_time_seconds": [0.0] * len(self._family_stages),
            "stage_llm_round_trip_seconds": [0.0] * len(self._family_stages),
            "stage_tool_wall_clock_seconds": [0.0] * len(self._family_stages),
            "stage_wall_clock_seconds": [0.0] * len(self._family_stages),
            "reasoning_resource_breakdown": {
                "prompt_tokens_total": 0.0,
                "completion_tokens_total": 0.0,
                "total_tokens_total": 0.0,
                "api_cost_total_usd_raw": 0.0,
            },
            "latency_breakdown": {
                "generation_time_total_seconds": 0.0,
                "llm_round_trip_total_seconds": 0.0,
                "tool_wall_clock_total_seconds": 0.0,
                "episode_wall_clock_seconds": 0.0,
            },
        }

    def _stage_deliberation_requirement(self, task: Any, stage_name: str) -> str:
        stage_requirements = getattr(task, "stage_deliberation_requirements", None)
        if isinstance(stage_requirements, dict) and stage_name in stage_requirements:
            return str(stage_requirements[stage_name])
        difficulty = getattr(task, "stage_difficulty", {}).get(stage_name, 0.0)
        return "deep" if difficulty >= 0.42 else "fast"

    def _normalize_deliberation_mode(self, value: Any) -> str:
        return "deep" if str(value).strip().lower() == "deep" else "fast"

    def _reasoning_match_multiplier(
        self,
        *,
        requirement: Any,
        actual_mode: Any,
    ) -> float:
        required_mode = self._normalize_deliberation_mode(requirement)
        realized_mode = self._normalize_deliberation_mode(actual_mode)
        if _env_flag("PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3"):
            if required_mode == realized_mode:
                return LLM_BENCH_REASONING_CALIBRATED_MATCH_DISCOUNT
            if required_mode == "deep":
                return LLM_BENCH_REASONING_CALIBRATED_MISMATCH_PENALTY_DEEP_REQUIRED
            return LLM_BENCH_REASONING_CALIBRATED_MISMATCH_PENALTY_FAST_REQUIRED
        if required_mode == realized_mode:
            return LLM_BENCH_REASONING_MATCH_DISCOUNT
        if required_mode == "deep":
            return LLM_BENCH_REASONING_MISMATCH_PENALTY_DEEP_REQUIRED
        return LLM_BENCH_REASONING_MISMATCH_PENALTY_FAST_REQUIRED

    def _prefix_dedup_behavior_context(self, path: list[str]) -> JsonDict | None:
        if (
            self.family_kind
            not in {
                "shared_basin_strong_prefix_dedup",
                "shared_basin_strong_prefix_dedup_profile_switch",
            }
            or self.current_instance is None
            or self.family_agent_map is None
        ):
            return None

        metadata = dict(self.current_instance.get("metadata", {}) or {})
        schedule_meta = dict(metadata.get("psagent_schedule", {}) or {})
        schedule_phase = str(schedule_meta.get("schedule_phase", "stationary") or "stationary")
        task_bucket = str(schedule_meta.get("task_bucket", "") or "")
        is_specialist_task = bool(schedule_meta.get("is_specialist_task", False))

        family_agents = [self.family_agent_map[agent_id] for agent_id in path]
        base_aliases = [self._prefix_dedup_base_alias(agent_id) for agent_id in path]
        route_labels = [str(getattr(agent, "route_label", "")) for agent in family_agents]
        node_semantics = [str(getattr(agent, "node_semantic", "")) for agent in family_agents]
        safe_prefix = node_semantics[0] == "safe_core"
        trap_like_path = base_aliases[0] == "stage1_n4" or route_labels[0] == "mixed_stage1_intake"
        if self.family_kind == "shared_basin_strong_prefix_dedup_profile_switch":
            trap_like_path = trap_like_path or route_labels[0] == "trap_stage1_intake"
            target_stage5_labels = {"target_stage5_verify", "target_stage5_decision"}
            profile_shared_or_target_prefix = route_labels[0] in {
                "general_stage1_intake",
                "general_stage1_verify",
                "target_stage1_handoff",
            }
            profile_clean_target_route = (
                profile_shared_or_target_prefix
                and not any(label.startswith("trap_") for label in route_labels)
                and not any(label.startswith("barrier_") for label in route_labels)
            )
            exact_target_good = (
                profile_clean_target_route
                and route_labels[3] == "target_stage4_repair"
                and route_labels[4] in target_stage5_labels
            )
            target_safe_subtree = exact_target_good or (
                profile_clean_target_route
                and route_labels[3] in {"general_stage4_repair", "general_stage4_verify"}
                and route_labels[4] in target_stage5_labels
            )
            decoy_path = (
                route_labels[3].startswith("barrier_stage4_")
                or route_labels[4].startswith("barrier_stage5_")
            )
        else:
            target_safe_subtree = (
                safe_prefix
                and route_labels[3] in {"public_stage4_core", "public_stage4_verify"}
                and route_labels[4] in {"public_stage5_verify", "public_stage5_decision"}
            )
            exact_target_good = (
                target_safe_subtree
                and route_labels[3] == "public_stage4_verify"
                and route_labels[4] == "public_stage5_verify"
            )
            decoy_path = (
                base_aliases[0] in {"stage1_n2", "stage1_n3", "stage1_n5"}
                and (
                    route_labels[3] == "mixed_stage4_lane"
                    or route_labels[4] in {"mixed_stage5_transfer", "private_stage5_edge"}
                )
            )

        is_trap_task = schedule_phase == "trap_pre_switch" or task_bucket == "trap_favoring"
        is_target_task = (
            schedule_phase == "target_post_switch" or task_bucket == "target_favoring"
        )

        archetype = "neutral"
        if is_trap_task and trap_like_path:
            archetype = "trap_like_good"
        elif is_target_task and trap_like_path:
            archetype = "trap_like_bad"
        elif is_target_task and exact_target_good and is_specialist_task:
            archetype = "target_safe_specialist_good"
        elif is_target_task and decoy_path:
            archetype = "target_decoy_medium"
        elif is_target_task and target_safe_subtree:
            archetype = "target_safe_majority_bad"
        elif is_trap_task and target_safe_subtree:
            archetype = "trap_safe_overcautious"

        return {
            "archetype": archetype,
            "schedule_phase": schedule_phase,
            "task_bucket": task_bucket,
            "is_specialist_task": is_specialist_task,
            "base_aliases": list(base_aliases),
            "route_labels": list(route_labels),
            "node_semantics": list(node_semantics),
            "trap_like_path": trap_like_path,
            "target_safe_subtree": target_safe_subtree,
            "exact_target_good": exact_target_good,
            "decoy_path": decoy_path,
        }

    def _profile_switch_flat_path_cost_enabled(self) -> bool:
        return (
            self.family_kind == "shared_basin_strong_prefix_dedup_profile_switch"
            and str(os.environ.get("PSAGENT_PROFILE_SWITCH_FLAT_PATH_COST", "")).strip()
            in {"1", "true", "True", "yes", "on"}
        )

    def _family_path_metadata(
        self,
        path: list[str],
        behavior_context: JsonDict | None = None,
    ) -> JsonDict:
        if self.family_agent_map is None:
            return {}
        family_agents = [self.family_agent_map[agent_id] for agent_id in path]
        route_labels = [str(getattr(agent, "route_label", "")) for agent in family_agents]
        deliberation_modes = [
            self._normalize_deliberation_mode(getattr(agent, "deliberation_mode", "deep"))
            for agent in family_agents
        ]
        node_semantics = [str(getattr(agent, "node_semantic", "")) for agent in family_agents]
        payload: JsonDict = {
            "family_route_labels": route_labels,
            "family_deliberation_modes": deliberation_modes,
            "family_node_semantics": node_semantics,
            "family_fast_stage_count": sum(mode == "fast" for mode in deliberation_modes),
            "family_trap_label_count": sum(label.startswith("trap_") for label in route_labels),
            "family_target_label_count": sum(label.startswith("target_") for label in route_labels),
            "family_general_label_count": sum(label.startswith("general_") for label in route_labels),
            "family_barrier_label_count": sum(label.startswith("barrier_") for label in route_labels),
        }
        if behavior_context is not None:
            payload.update(
                {
                    "family_behavior_archetype": behavior_context.get("archetype"),
                    "family_schedule_phase": behavior_context.get("schedule_phase"),
                    "family_task_bucket": behavior_context.get("task_bucket"),
                    "family_trap_like_path": behavior_context.get("trap_like_path"),
                    "family_target_safe_subtree": behavior_context.get("target_safe_subtree"),
                    "family_exact_target_good": behavior_context.get("exact_target_good"),
                    "family_decoy_path": behavior_context.get("decoy_path"),
                }
            )
        return payload

    def _prefix_dedup_base_alias(self, agent_id: str) -> str:
        parts = agent_id.split("__from__", 1)
        return parts[0] if parts else agent_id

    def _ensure_family_executor_for_instance(self, instance: JsonDict) -> None:
        if self.family_kind is None or self._family_stages is None:
            return
        if self.executor_name != "bench_backed":
            if self.executor_name != "llm_bench":
                return
        family = instance.get("family")
        if self.executor_name == "bench_backed":
            if family == "telecom_mms_recovery":
                if not isinstance(self.family_executor, TelecomBenchBackedExecutor):
                    self.family_executor = TelecomBenchBackedExecutor(
                        stages=list(self._family_stages),
                        seed=self.family_seed,
                    )
            else:
                if not isinstance(self.family_executor, BenchBackedExecutor):
                    self.family_executor = BenchBackedExecutor(
                        stages=list(self._family_stages),
                        seed=self.family_seed,
                    )
            return
        if family == "telecom_mms_recovery":
            if not isinstance(self.family_executor, TelecomLLMBenchExecutor):
                self.family_executor = TelecomLLMBenchExecutor(
                    stages=list(self._family_stages),
                    seed=self.family_seed,
                )
        else:
            if not isinstance(self.family_executor, LLMBenchExecutor):
                self.family_executor = LLMBenchExecutor(
                    stages=list(self._family_stages),
                    seed=self.family_seed,
                )

    def get_episode_log(self) -> JsonDict:
        """Return the most recent episode log."""

        if self._last_episode_log is None:
            return {}
        return deepcopy(self._last_episode_log)

    def _validate_path(self, path: list[str]) -> None:
        if len(path) != len(self.STAGE_NAMES):
            raise ValueError(
                f"Path length must be {len(self.STAGE_NAMES)}. Got {len(path)}."
            )
        prefix: tuple[str, ...] = ()
        allowed_children = getattr(self.family_spec, "allowed_children", None)
        for expected_stage, agent_id in zip(self.STAGE_NAMES, path):
            if agent_id not in self.agent_catalog:
                raise KeyError(f"Unknown agent_id in path: {agent_id}")
            agent = self.agent_catalog[agent_id]
            if agent.stage_name != expected_stage:
                raise ValueError(
                    f"Agent {agent_id} belongs to {agent.stage_name}, "
                    f"but path position expects {expected_stage}."
                )
            if allowed_children:
                legal_children = allowed_children.get(prefix)
                if legal_children is not None and agent_id not in legal_children:
                    raise ValueError(
                        "Path violates family continuation topology. "
                        f"prefix={list(prefix)} agent_id={agent_id}"
                    )
            prefix = prefix + (agent_id,)


def _oracle_stage_bundle(env: FixedTreeEnvironment, stage_name: str) -> JsonDict:
    assert env.current_instance is not None
    stage = env.current_instance[stage_name]
    return {
        "input": deepcopy(stage.get("input", {})),
        "output": deepcopy(stage.get("oracle_output", {})),
        "source": "oracle",
    }


def _is_richer_catalog(env: FixedTreeEnvironment) -> bool:
    return any(
        ("_specialist_" in agent_id) or ("_weak_" in agent_id)
        for agent_id in env.agent_catalog
    )


def _is_tier2_instance(env: FixedTreeEnvironment) -> bool:
    assert env.current_instance is not None
    return env.current_instance.get("metadata", {}).get("tier") == "tier2_multi_resolution"


def _reservation_ids_from_stage2(previous_outputs: JsonDict) -> list[str]:
    return list(
        previous_outputs.get("stage2", {})
        .get("output", {})
        .get("resolved_reservations", [])
        or []
    )


def _filter_rows_by_reservation_ids(rows: list[JsonDict], reservation_ids: list[str]) -> list[JsonDict]:
    if not reservation_ids:
        return deepcopy(rows)
    allowed = set(reservation_ids)
    return [deepcopy(row) for row in rows if row.get("reservation_id") in allowed]


def _oracle_stage4_rows_for_reservations(env: FixedTreeEnvironment, reservation_ids: list[str]) -> list[JsonDict]:
    assert env.current_instance is not None
    rows = (
        env.current_instance.get("stage4", {})
        .get("oracle_output", {})
        .get("per_reservation", [])
        or []
    )
    return _filter_rows_by_reservation_ids(rows, reservation_ids)


def _cancel_candidate_rows(rows: list[JsonDict]) -> list[JsonDict]:
    return [row for row in rows if row.get("oracle_execute_decision") == "cancel"]


def _mark_row_refused(row: JsonDict, code: str) -> None:
    row["policy_eligible_cancel_with_refund"] = False
    row["policy_adjudication_label"] = code
    row["policy_refusal_code"] = code
    row["policy_rule_trace"] = list(row.get("policy_rule_trace", [])) + [code]


def _build_stage5_output_from_stage4_rows(rows: list[JsonDict]) -> JsonDict:
    cancelled_ids: list[str] = []
    refused_ids: list[str] = []
    for row in rows:
        reservation_id = row.get("reservation_id")
        if not reservation_id:
            continue
        if row.get("policy_eligible_cancel_with_refund"):
            cancelled_ids.append(reservation_id)
        else:
            refused_ids.append(reservation_id)

    if cancelled_ids and refused_ids:
        final_action = "cancel_subset"
    elif cancelled_ids:
        final_action = "cancel_all"
    else:
        final_action = "refuse_all"

    return {
        "final_action": final_action,
        "cancelled_reservation_ids": cancelled_ids,
        "refused_reservation_ids": refused_ids,
        "response_mode": "stage4_derived_execution",
    }


def _run_stage1(
    env: FixedTreeEnvironment,
    agent: AgentSpec,
    previous_outputs: JsonDict,
) -> JsonDict:
    del previous_outputs
    bundle = _oracle_stage_bundle(env, "stage1")
    if agent.kind == "simulated":
        output = deepcopy(bundle["output"])
        pressure_signals = list(output.get("pressure_signals", []))
        if pressure_signals:
            pressure_signals = pressure_signals[:-1]
        output["pressure_signals"] = pressure_signals
        bundle["output"] = output
        bundle["source"] = "simulated_noisy"
    return bundle


def _run_stage2(
    env: FixedTreeEnvironment,
    agent: AgentSpec,
    previous_outputs: JsonDict,
) -> JsonDict:
    del previous_outputs
    bundle = _oracle_stage_bundle(env, "stage2")
    if _is_richer_catalog(env) and _is_tier2_instance(env):
        if "specialist_g1" in agent.agent_id:
            return bundle
        if "weak_g0" in agent.agent_id:
            output = deepcopy(bundle["output"])
            resolved = list(output.get("resolved_reservations", []))
            if len(resolved) > 1:
                output["resolved_reservations"] = resolved[:1]
                output["resolution_status"] = "under_resolved_multi_candidate"
            bundle["output"] = output
            bundle["source"] = "richer_tier2_weak"
            return bundle
        if agent.agent_id == "resolve_oracle_g0":
            output = deepcopy(bundle["output"])
            resolved = list(output.get("resolved_reservations", []))
            if len(resolved) > 2:
                output["resolved_reservations"] = resolved[:-1]
                output["resolution_status"] = "partial_multi_resolution"
            bundle["output"] = output
            bundle["source"] = "richer_tier2_conservative"
            return bundle
    if agent.kind == "simulated":
        output = deepcopy(bundle["output"])
        resolved = list(output.get("resolved_reservations", []))
        if len(resolved) > 1:
            output["resolved_reservations"] = resolved[:-1]
            output["resolution_status"] = "partially_resolved"
        elif len(resolved) == 1:
            output["resolution_status"] = "under_resolved_but_single_candidate"
        bundle["output"] = output
        bundle["source"] = "simulated_noisy"
    return bundle


def _run_stage3(
    env: FixedTreeEnvironment,
    agent: AgentSpec,
    previous_outputs: JsonDict,
) -> JsonDict:
    bundle = _oracle_stage_bundle(env, "stage3")
    if _is_richer_catalog(env) and _is_tier2_instance(env):
        resolved_ids = _reservation_ids_from_stage2(previous_outputs)
        output = deepcopy(bundle["output"])
        per_reservation = _filter_rows_by_reservation_ids(
            deepcopy(output.get("per_reservation", [])),
            resolved_ids,
        )
        if "specialist_g1" in agent.agent_id:
            output["per_reservation"] = per_reservation
            bundle["output"] = output
            bundle["source"] = "richer_tier2_specialist"
            return bundle

        cancel_rows = [row for row in per_reservation if row.get("oracle_execute_decision") == "cancel"]
        if "weak_g0" in agent.agent_id:
            for row in cancel_rows:
                row["eligible_by_business_rule"] = False
                row["eligible_by_insurance_rule"] = False
                row["stated_reason_supported_by_insurance"] = False
                row["richer_feature_failure"] = "weak_g0_drops_cancel_support"
            output["per_reservation"] = per_reservation
            bundle["output"] = output
            bundle["source"] = "richer_tier2_weak"
            return bundle

        if agent.agent_id == "feature_oracle_g0" and cancel_rows:
            row = cancel_rows[-1]
            row["eligible_by_business_rule"] = False
            row["richer_feature_failure"] = "oracle_g0_soft_subset_drop"
            output["per_reservation"] = per_reservation
            bundle["output"] = output
            bundle["source"] = "richer_tier2_conservative"
            return bundle

        output["per_reservation"] = per_reservation
        bundle["output"] = output
        return bundle

    if agent.kind == "simulated":
        output = deepcopy(bundle["output"])
        per_reservation = deepcopy(output.get("per_reservation", []))
        for row in per_reservation:
            if "eligible_by_24h_rule" in row:
                row["eligible_by_24h_rule"] = False
            if "stated_reason_supported_by_insurance" in row:
                row["stated_reason_supported_by_insurance"] = False
        output["per_reservation"] = per_reservation
        bundle["output"] = output
        bundle["source"] = "simulated_noisy"
    return bundle


def _run_stage4(
    env: FixedTreeEnvironment,
    agent: AgentSpec,
    previous_outputs: JsonDict,
) -> JsonDict:
    if "stage3" not in previous_outputs:
        raise RuntimeError("Stage4 requires stage3 output.")

    bundle = _oracle_stage_bundle(env, "stage4")
    if _is_richer_catalog(env) and _is_tier2_instance(env):
        stage3_rows = (
            previous_outputs.get("stage3", {})
            .get("output", {})
            .get("per_reservation", [])
            or []
        )
        reservation_ids = [row.get("reservation_id") for row in stage3_rows if row.get("reservation_id")]
        oracle_rows = _oracle_stage4_rows_for_reservations(env, reservation_ids)
        row_map = {row.get("reservation_id"): deepcopy(row) for row in oracle_rows}
        feature_rows = {row.get("reservation_id"): row for row in stage3_rows}
        per_reservation: list[JsonDict] = []
        for reservation_id in reservation_ids:
            oracle_row = row_map.get(reservation_id)
            if not oracle_row:
                continue
            feature_row = feature_rows.get(reservation_id, {})
            failure_code = feature_row.get("richer_feature_failure")
            if failure_code == "weak_g0_drops_cancel_support":
                _mark_row_refused(oracle_row, str(failure_code))
            elif failure_code == "oracle_g0_soft_subset_drop":
                if "specialist_g1" not in agent.agent_id:
                    _mark_row_refused(oracle_row, str(failure_code))
            per_reservation.append(oracle_row)

        if "specialist_g1" in agent.agent_id:
            bundle["output"] = {"per_reservation": per_reservation}
            bundle["source"] = "richer_tier2_specialist"
            return bundle

        cancel_rows = _cancel_candidate_rows(per_reservation)
        if "weak_g0" in agent.agent_id:
            for row in cancel_rows:
                _mark_row_refused(row, "weak_g0_subset_failure")
            bundle["output"] = {"per_reservation": per_reservation}
            bundle["source"] = "richer_tier2_weak"
            return bundle

        if agent.agent_id == "adjudicate_oracle_g0" and len(cancel_rows) > 1:
            _mark_row_refused(cancel_rows[-1], "oracle_g0_subset_conservative")
            bundle["output"] = {"per_reservation": per_reservation}
            bundle["source"] = "richer_tier2_conservative"
            return bundle

        bundle["output"] = {"per_reservation": per_reservation}
        return bundle
    if agent.kind == "simulated":
        output = deepcopy(bundle["output"])
        per_reservation = deepcopy(output.get("per_reservation", []))
        for row in per_reservation:
            row["policy_eligible_cancel_with_refund"] = False
            if row.get("policy_adjudication_label") == "allow_cancel_refund":
                row["policy_adjudication_label"] = "deny_simulated_noise"
                row["policy_refusal_code"] = "simulated_false_negative"
                row["policy_rule_trace"] = list(row.get("policy_rule_trace", [])) + [
                    "simulated_false_negative"
                ]
        output["per_reservation"] = per_reservation
        bundle["output"] = output
        bundle["source"] = "simulated_noisy"
    return bundle


def _run_stage5(
    env: FixedTreeEnvironment,
    agent: AgentSpec,
    previous_outputs: JsonDict,
) -> JsonDict:
    if "stage4" not in previous_outputs:
        raise RuntimeError("Stage5 requires stage4 output.")

    bundle = _oracle_stage_bundle(env, "stage5")
    if _is_richer_catalog(env) and _is_tier2_instance(env):
        stage4_rows = (
            previous_outputs.get("stage4", {})
            .get("output", {})
            .get("per_reservation", [])
            or []
        )
        output = _build_stage5_output_from_stage4_rows(deepcopy(stage4_rows))
        if "specialist_g1" in agent.agent_id:
            bundle["output"] = output
            bundle["source"] = "richer_tier2_specialist"
            return bundle

        if agent.agent_id == "execute_oracle_g0":
            if output["final_action"] == "cancel_subset" and len(output["cancelled_reservation_ids"]) > 1:
                dropped = output["cancelled_reservation_ids"].pop()
                output["refused_reservation_ids"].append(dropped)
            bundle["output"] = output
            bundle["source"] = "richer_tier2_conservative"
            return bundle

        if "weak_g0" in agent.agent_id:
            if output["cancelled_reservation_ids"]:
                dropped = output["cancelled_reservation_ids"].pop()
                output["refused_reservation_ids"].append(dropped)
                if output["cancelled_reservation_ids"] and output["refused_reservation_ids"]:
                    output["final_action"] = "cancel_subset"
                elif output["cancelled_reservation_ids"]:
                    output["final_action"] = "cancel_all"
                else:
                    output["final_action"] = "refuse_all"
            bundle["output"] = output
            bundle["source"] = "richer_tier2_weak"
            return bundle

    if agent.kind == "rule":
        return bundle

    output = deepcopy(bundle["output"])
    final_action = output.get("final_action")
    if final_action == "refuse_all":
        output["final_action"] = "cancel_all"
    elif final_action == "cancel_all":
        output["final_action"] = "refuse_all"
    elif final_action == "cancel_subset":
        output["final_action"] = "refuse_all"
        output["cancelled_reservation_ids"] = []
        stage4_rows = (
            previous_outputs.get("stage4", {})
            .get("output", {})
            .get("per_reservation", [])
        )
        output["refused_reservation_ids"] = [
            row.get("reservation_id") for row in stage4_rows
        ]
    bundle["output"] = output
    bundle["source"] = "simulated_noisy"
    return bundle
