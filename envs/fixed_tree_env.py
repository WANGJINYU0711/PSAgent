"""Minimal fixed-tree environment for the derived airline cancellation benchmark.

This module intentionally does not depend on tau2's orchestrator. It operates
over the structured derived instances in ``data/derived/.../tasks.json`` and
supports:

- reset(instance)
- run_path(path)
- shared/unshared leaf typing via z(l)=max_h g(a_h)
- simple terminal-cost computation
- default oracle-like and noisy rule-based stage executors
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, Iterable, Optional

from evaluator import DEFAULT_COST_SPEC, evaluate_terminal_prediction
from adapters.airline_adapter import AirlineTaskAdapter
from executors.bench_backed_executor import BenchBackedExecutor
from executors.llm_bench_executor import LLMBenchExecutor
from executors.simulated_executor import SimulatedExecutor
from tree_family.generator import TreeFamilyGenerator


JsonDict = dict[str, Any]
StageExecutor = Callable[["FixedTreeEnvironment", "AgentSpec", JsonDict], JsonDict]


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
    success: bool
    path_agent_cost: float
    total_cost: float
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
        self.task_adapter = AirlineTaskAdapter()
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
                        cost=family_agent.base_cost,
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
        terminal_cost = evaluator_result["terminal_penalty"]
        path_agent_cost = sum(self.agent_catalog[agent_id].cost for agent_id in path)
        total_cost = terminal_cost + DEFAULT_COST_SPEC.path_agent_cost_weight * path_agent_cost

        final_action = (
            stage_outputs.get("stage5", {}).get("output", {}).get("final_action")
        )
        oracle_action = (
            instance.get("stage5", {})
            .get("oracle_output", {})
            .get("final_action")
        )
        success = bool(evaluator_result["exact_match"])

        episode_log = {
            "instance_id": self.current_instance_id,
            "selected_path": list(path),
            "leaf_type": leaf_type,
            "stage_trace": stage_trace,
            "final_action": final_action,
            "oracle_action": oracle_action,
            "terminal_cost": terminal_cost,
            "path_agent_cost": path_agent_cost,
            "total_cost": total_cost,
            "success": success,
            "evaluator_version": evaluator_result["evaluator_version"],
            "false_cancel_count": evaluator_result["false_cancel_count"],
            "missed_cancel_count": evaluator_result["missed_cancel_count"],
            "false_refuse_count": evaluator_result["false_refuse_count"],
            "missed_refuse_count": evaluator_result["missed_refuse_count"],
            "subset_mismatch": evaluator_result["subset_mismatch"],
            "cost_breakdown": deepcopy(evaluator_result["cost_breakdown"]),
        }
        self._last_episode_log = deepcopy(episode_log)

        return EpisodeResult(
            instance_id=self.current_instance_id or "unknown_instance",
            selected_path=list(path),
            leaf_type=leaf_type,
            stage_outputs=stage_outputs,
            final_action=final_action,
            oracle_action=oracle_action,
            terminal_cost=terminal_cost,
            success=success,
            path_agent_cost=path_agent_cost,
            total_cost=total_cost,
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
        evaluator_result = self.evaluate_terminal_outcome(stage_outputs, path)
        terminal_cost = evaluator_result["terminal_penalty"]
        path_agent_cost = float(execution["path_agent_cost"])
        total_cost = terminal_cost + DEFAULT_COST_SPEC.path_agent_cost_weight * path_agent_cost
        final_action = execution.get("final_action")
        oracle_action = (
            self.current_instance.get("stage5", {})
            .get("oracle_output", {})
            .get("final_action")
        )
        success = bool(evaluator_result["exact_match"])

        episode_log = {
            "instance_id": self.current_instance_id,
            "selected_path": list(path),
            "leaf_type": execution["leaf_type"],
            "stage_trace": deepcopy(execution.get("stage_trace", [])),
            "final_action": final_action,
            "oracle_action": oracle_action,
            "terminal_cost": terminal_cost,
            "path_agent_cost": path_agent_cost,
            "total_cost": total_cost,
            "success": success,
            "evaluator_version": evaluator_result["evaluator_version"],
            "false_cancel_count": evaluator_result["false_cancel_count"],
            "missed_cancel_count": evaluator_result["missed_cancel_count"],
            "false_refuse_count": evaluator_result["false_refuse_count"],
            "missed_refuse_count": evaluator_result["missed_refuse_count"],
            "subset_mismatch": evaluator_result["subset_mismatch"],
            "cost_breakdown": deepcopy(evaluator_result["cost_breakdown"]),
            "family_kind": self.family_kind,
        }
        if isinstance(execution.get("bench_aux_eval"), dict):
            episode_log["bench_aux_eval"] = deepcopy(execution["bench_aux_eval"])
        self._last_episode_log = deepcopy(episode_log)

        return EpisodeResult(
            instance_id=self.current_instance_id or "unknown_instance",
            selected_path=list(path),
            leaf_type=execution["leaf_type"],
            stage_outputs=stage_outputs,
            final_action=final_action,
            oracle_action=oracle_action,
            terminal_cost=terminal_cost,
            success=success,
            path_agent_cost=path_agent_cost,
            total_cost=total_cost,
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
        stage_outputs.setdefault(
            "stage5",
            {
                "input": {},
                "output": {
                    "final_action": execution.get("final_action"),
                    "cancelled_reservation_ids": list(
                        execution.get("cancelled_reservation_ids", [])
                    ),
                    "refused_reservation_ids": list(
                        execution.get("refused_reservation_ids", [])
                    ),
                },
                "source": "simulated_executor",
            },
        )
        return stage_outputs

    def compute_leaf_type(self, path: list[str]) -> str:
        """Compute shared/unshared leaf type from z(l)=max_h g(a_h)."""

        self._validate_path(path)
        z_value = max(self.agent_catalog[agent_id].g for agent_id in path)
        return "shared" if z_value == 0 else "unshared"

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
    ) -> JsonDict:
        """Run evaluator v2 on the terminal stage output."""

        del path
        if self.current_instance is None:
            raise RuntimeError("No current instance loaded.")
        predicted_stage5_output = stage_outputs.get("stage5", {}).get("output", {})
        return evaluate_terminal_prediction(
            instance=self.current_instance,
            predicted_stage5_output=predicted_stage5_output,
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
        for expected_stage, agent_id in zip(self.STAGE_NAMES, path):
            if agent_id not in self.agent_catalog:
                raise KeyError(f"Unknown agent_id in path: {agent_id}")
            agent = self.agent_catalog[agent_id]
            if agent.stage_name != expected_stage:
                raise ValueError(
                    f"Agent {agent_id} belongs to {agent.stage_name}, "
                    f"but path position expects {expected_stage}."
                )


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
