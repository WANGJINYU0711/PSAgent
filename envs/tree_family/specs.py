"""Core reusable dataclasses for tree-family experiments."""

from __future__ import annotations

from dataclasses import dataclass


CAPABILITY_NAMES: tuple[str, ...] = (
    "user_grounding",
    "account_lookup",
    "line_resolution",
    "network_diagnosis",
    "permission_diagnosis",
    "apn_diagnosis",
    "roaming_diagnosis",
    "repair_execution",
    "verification",
    "terminal_decision",
)

CAPABILITY_DESCRIPTIONS: dict[str, str] = {
    "user_grounding": "Extract and ground the user's problem statement and context.",
    "account_lookup": "Retrieve account-side data needed to identify or validate the target line.",
    "line_resolution": "Disambiguate the correct line/target entity for downstream work.",
    "network_diagnosis": "Diagnose network, service, data, and connectivity state.",
    "permission_diagnosis": "Diagnose app permission blockers affecting messaging behavior.",
    "apn_diagnosis": "Diagnose APN and MMS configuration blockers.",
    "roaming_diagnosis": "Diagnose roaming-related account/device state blockers.",
    "repair_execution": "Execute or orchestrate blocker-specific repair actions.",
    "verification": "Verify whether intermediate or final recovery succeeded.",
    "terminal_decision": "Choose the correct terminal action such as repair_all/subset/transfer.",
}


@dataclass
class TaskDescriptor:
    task_id: str
    # Kept as attribute_weights for compatibility with existing consumers, but the
    # keys are now semantic capability names rather than anonymous integer ids.
    attribute_weights: dict[str, float]
    stage_difficulty: dict[str, float]
    stage_capability_requirements: dict[str, dict[str, float]] | None = None
    stage_deliberation_requirements: dict[str, str] | None = None


@dataclass
class AgentSpec:
    agent_id: str
    g: int
    base_cost: float
    competence_level: str
    scope_level: str
    stability_level: str
    # Kept as attribute_skill for compatibility with existing consumers, but the
    # keys are now semantic capability names rather than anonymous integer ids.
    attribute_skill: dict[str, float]
    deliberation_mode: str = "deep"


@dataclass
class FamilySpec:
    family_name: str
    stages: list[str]
    stage_agents: dict[str, list[str]]
    allowed_children: dict[tuple[str, ...], list[str]] | None = None
