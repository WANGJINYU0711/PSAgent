"""Brute-force path enumeration and fixed-path evaluation helpers."""

from __future__ import annotations

from itertools import product
from typing import Iterable

from fixed_tree_env import EpisodeResult, FixedTreeEnvironment


def enumerate_all_paths(env: FixedTreeEnvironment) -> list[list[str]]:
    """Enumerate all complete fixed-tree paths from the environment catalog."""

    stage_agent_ids: list[list[str]] = []
    for stage_name in env.STAGE_NAMES:
        agent_ids = [agent.agent_id for agent in env.agents_by_stage[stage_name]]
        if not agent_ids:
            raise ValueError(f"No agents registered for stage {stage_name}.")
        stage_agent_ids.append(agent_ids)
    return [list(path) for path in product(*stage_agent_ids)]


def evaluate_path(
    instance: dict,
    path: list[str],
    env: FixedTreeEnvironment,
) -> EpisodeResult:
    """Reset the environment on an instance and evaluate one fixed path."""

    env.reset(instance)
    return env.run_path(path)


def find_best_fixed_path(
    instance: dict,
    env: FixedTreeEnvironment,
) -> tuple[list[str], EpisodeResult]:
    """Return the lowest-cost path under brute-force enumeration."""

    best_path: list[str] | None = None
    best_result: EpisodeResult | None = None

    for path in enumerate_all_paths(env):
        result = evaluate_path(instance, path, env)
        if best_result is None or result.total_cost < best_result.total_cost:
            best_path = list(path)
            best_result = result

    if best_path is None or best_result is None:
        raise RuntimeError("Failed to find a best path.")
    return best_path, best_result
