"""Brute-force path enumeration helpers for per-instance and stationary oracles."""

from __future__ import annotations

from itertools import product
from typing import Any, Iterable

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
    """Return the per-instance lowest-cost fixed path.

    This is a single-instance oracle and should be treated as a supporting
    upper-bound helper, not as the formal regret baseline for a whole run.
    """

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


def evaluate_stationary_path(
    instances: list[dict[str, Any]],
    path: list[str],
    env: FixedTreeEnvironment,
) -> dict[str, Any]:
    """Evaluate one fixed path across a whole run.

    The same path is used for every instance in the provided horizon. This is
    the object needed for a stationary-oracle regret baseline.
    """

    episode_results: list[EpisodeResult] = []
    for instance in instances:
        episode_results.append(evaluate_path(instance, path, env))

    total_costs = [float(result.total_cost) for result in episode_results]
    terminal_costs = [float(result.terminal_cost) for result in episode_results]
    return {
        "path": list(path),
        "episode_results": episode_results,
        "episode_total_costs": total_costs,
        "episode_terminal_costs": terminal_costs,
        "cumulative_total_cost": sum(total_costs),
        "mean_total_cost": (sum(total_costs) / len(total_costs)) if total_costs else 0.0,
    }


def find_best_stationary_path(
    instances: list[dict[str, Any]],
    env: FixedTreeEnvironment,
) -> tuple[list[str], dict[str, Any]]:
    """Return the best fixed path over a whole run.

    This is the formal stationary oracle: one leaf/path is chosen once and then
    reused for the entire instance horizon. Ties are broken by the first path
    encountered in the deterministic enumeration order.
    """

    best_summary: dict[str, Any] | None = None
    best_path: list[str] | None = None
    for path in enumerate_all_paths(env):
        summary = evaluate_stationary_path(instances, path, env)
        if (
            best_summary is None
            or summary["cumulative_total_cost"] < best_summary["cumulative_total_cost"]
        ):
            best_summary = summary
            best_path = list(path)

    if best_path is None or best_summary is None:
        raise RuntimeError("Failed to find a stationary best path.")
    return best_path, best_summary
