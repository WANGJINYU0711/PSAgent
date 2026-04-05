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
    ) -> None:
        self.agent_catalog: dict[str, AgentSpec] = {
            agent.agent_id: agent for agent in agent_catalog
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
        return deepcopy(self.current_instance)

    def run_path(self, path: list[str]) -> EpisodeResult:
        """Execute one complete stage-wise path over the current instance."""

        if self.current_instance is None:
            raise RuntimeError("Environment has no loaded instance. Call reset(instance) first.")

        self._validate_path(path)
        instance = self.current_instance
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
        terminal_cost = self.compute_terminal_cost(stage_outputs, path)
        path_agent_cost = sum(self.agent_catalog[agent_id].cost for agent_id in path)
        total_cost = terminal_cost + 0.1 * path_agent_cost

        final_action = (
            stage_outputs.get("stage5", {}).get("output", {}).get("final_action")
        )
        oracle_action = (
            instance.get("stage5", {})
            .get("oracle_output", {})
            .get("final_action")
        )
        success = final_action == oracle_action

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
        """Compute the v1 terminal action-mismatch cost."""

        del path  # Reserved for future richer cost models.
        if self.current_instance is None:
            raise RuntimeError("No current instance loaded.")

        predicted_action = (
            stage_outputs.get("stage5", {}).get("output", {}).get("final_action")
        )
        oracle_action = (
            self.current_instance.get("stage5", {})
            .get("oracle_output", {})
            .get("final_action")
        )
        return 0.0 if predicted_action == oracle_action else 1.0

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
    del previous_outputs
    bundle = _oracle_stage_bundle(env, "stage3")
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
