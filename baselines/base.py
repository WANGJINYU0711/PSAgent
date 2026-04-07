"""Shared baseline interfaces and lightweight helpers."""

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod
from typing import Any

from agent_catalog import group_agent_ids_by_stage
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import enumerate_all_paths


class BasePolicy(ABC):
    """Minimal policy interface used by the Day 4 runner."""

    def __init__(self, seed: int = 0, protocol_mode: str = "actual_leaf") -> None:
        self.seed = seed
        self.rng = random.Random(seed)
        self.protocol_mode = protocol_mode
        self.stage_agent_ids: dict[str, list[str]] = {}

    def reset(self) -> None:
        """Reset any internal state between runs if needed."""

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        """Cache catalog layout for path selection."""

        self.stage_agent_ids = group_agent_ids_by_stage(env.agent_catalog.values())
        missing = [stage for stage in env.STAGE_NAMES if stage not in self.stage_agent_ids]
        if missing:
            raise ValueError(f"Policy cannot bind env. Missing stages: {missing}")

    @abstractmethod
    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        """Select a complete path for the next episode."""

    @abstractmethod
    def update(self, episode_result: EpisodeResult) -> None:
        """Update policy state after one episode."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Stable method name used in logs."""

    def resolve_feedback_type(self, episode_result: EpisodeResult) -> str:
        """Return the effective feedback protocol used by this policy."""

        if self.protocol_mode == "force_shared":
            return "shared"
        if self.protocol_mode == "force_unshared":
            return "unshared"
        return episode_result.leaf_type

    def get_state(self) -> dict[str, Any]:
        """Expose lightweight debug state for logging and inspection."""

        return {"protocol_mode": self.protocol_mode}

    def preferred_catalog_preset(self) -> str:
        """Return the catalog preset that matches this policy's endpoint semantics."""

        return "mixed"

    def get_last_selection_info(self) -> dict[str, Any]:
        """Expose per-episode sampling diagnostics for optional logging."""

        return {}

    def _sample_index(self, weights: list[float]) -> int:
        """Sample an index from non-negative weights."""

        if not weights:
            raise ValueError("Cannot sample from an empty weight list.")
        total = sum(max(0.0, weight) for weight in weights)
        if total <= 0:
            return self.rng.randrange(len(weights))

        draw = self.rng.random() * total
        cumulative = 0.0
        for idx, weight in enumerate(weights):
            cumulative += max(0.0, weight)
            if draw <= cumulative:
                return idx
        return len(weights) - 1


class StagewiseScorePolicy(BasePolicy):
    """Simple stagewise average-cost selector used for shared-style baselines."""

    def __init__(
        self,
        seed: int = 0,
        protocol_mode: str = "force_shared",
        epsilon: float = 0.1,
    ) -> None:
        super().__init__(seed=seed, protocol_mode=protocol_mode)
        self.epsilon = epsilon
        self.counts: dict[str, int] = {}
        self.total_costs: dict[str, float] = {}

    def _mean_cost(self, agent_id: str) -> float:
        count = self.counts.get(agent_id, 0)
        if count == 0:
            return 0.0
        return self.total_costs[agent_id] / count

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.stage_agent_ids:
            self.bind_env(env)

        path: list[str] = []
        for stage_name in env.STAGE_NAMES:
            agent_ids = self.stage_agent_ids[stage_name]
            if self.rng.random() < self.epsilon:
                path.append(self.rng.choice(agent_ids))
                continue

            ranked = sorted(
                agent_ids,
                key=lambda agent_id: (self._mean_cost(agent_id), self.counts.get(agent_id, 0)),
            )
            best_mean = self._mean_cost(ranked[0])
            tied = [agent_id for agent_id in ranked if math.isclose(self._mean_cost(agent_id), best_mean)]
            path.append(self.rng.choice(tied))
        return path

    def update(self, episode_result: EpisodeResult) -> None:
        observed_cost = episode_result.total_cost
        for agent_id in episode_result.selected_path:
            self.counts[agent_id] = self.counts.get(agent_id, 0) + 1
            self.total_costs[agent_id] = self.total_costs.get(agent_id, 0.0) + observed_cost

    def get_state(self) -> dict[str, Any]:
        return {
            "protocol_mode": self.protocol_mode,
            "epsilon": self.epsilon,
            "counts": dict(self.counts),
            "mean_costs": {
                agent_id: self._mean_cost(agent_id) for agent_id in sorted(self.total_costs)
            },
        }


class StagewiseExp3Policy(BasePolicy):
    """A compact stagewise EXP3-style selector.

    This is intentionally lightweight: each stage maintains an independent set
    of adversarial-bandit weights and updates only the sampled child.
    """

    def __init__(
        self,
        seed: int = 0,
        protocol_mode: str = "force_unshared",
        gamma: float = 0.2,
        epsilon: float = 0.0,
        estimator_type: str = "reward",
        update_type: str = "stagewise_exp3",
    ) -> None:
        super().__init__(seed=seed, protocol_mode=protocol_mode)
        self.gamma = gamma
        self.epsilon = epsilon
        self.estimator_type = estimator_type
        self.update_type = update_type
        self.weights: dict[str, float] = {}
        self.last_path_probs: list[float] = []
        self.last_stage_probs: dict[str, float] = {}
        self.last_path_prob: float = 0.0

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        for agent_ids in self.stage_agent_ids.values():
            for agent_id in agent_ids:
                self.weights.setdefault(agent_id, 1.0)

    def _stage_probs(self, agent_ids: list[str]) -> list[float]:
        weight_sum = sum(self.weights[agent_id] for agent_id in agent_ids)
        if weight_sum <= 0:
            base = 1.0 / len(agent_ids)
            return [base for _ in agent_ids]

        probs: list[float] = []
        num_arms = len(agent_ids)
        for agent_id in agent_ids:
            exploit = self.weights[agent_id] / weight_sum
            mixed = (1.0 - self.gamma) * exploit + self.gamma / num_arms
            mixed = (1.0 - self.epsilon) * mixed + self.epsilon / num_arms
            probs.append(mixed)
        return probs

    def _sample_from_probs(self, agent_ids: list[str], probs: list[float]) -> tuple[str, float]:
        draw = self.rng.random()
        cumulative = 0.0
        for agent_id, prob in zip(agent_ids, probs):
            cumulative += prob
            if draw <= cumulative:
                return agent_id, prob
        return agent_ids[-1], probs[-1]

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.stage_agent_ids:
            self.bind_env(env)

        path: list[str] = []
        self.last_path_probs = []
        self.last_stage_probs = {}
        for stage_name in env.STAGE_NAMES:
            agent_ids = self.stage_agent_ids[stage_name]
            probs = self._stage_probs(agent_ids)
            agent_id, prob = self._sample_from_probs(agent_ids, probs)
            path.append(agent_id)
            self.last_path_probs.append(prob)
            self.last_stage_probs[stage_name] = prob
        path_prob = 1.0
        for prob in self.last_path_probs:
            path_prob *= prob
        self.last_path_prob = path_prob
        return path

    def update(self, episode_result: EpisodeResult) -> None:
        for stage_name, agent_id, prob in zip(
            self.stage_agent_ids.keys(),
            episode_result.selected_path,
            self.last_path_probs,
        ):
            arms = max(1, len(self.stage_agent_ids[self._stage_for_agent(agent_id)]))
            if self.estimator_type == "loss":
                estimated_signal = episode_result.total_cost / max(prob, 1e-12)
                self.weights[agent_id] *= math.exp(
                    -self.gamma * estimated_signal / arms
                )
            elif self.estimator_type == "reward":
                reward = 1.0 / (1.0 + episode_result.total_cost)
                estimated_signal = reward / max(prob, 1e-12)
                self.weights[agent_id] *= math.exp(
                    self.gamma * estimated_signal / arms
                )
            else:
                raise ValueError(
                    f"Unknown estimator_type for StagewiseExp3Policy: {self.estimator_type}"
                )
            self.last_stage_probs[stage_name] = prob

    def _stage_for_agent(self, agent_id: str) -> str:
        for stage_name, agent_ids in self.stage_agent_ids.items():
            if agent_id in agent_ids:
                return stage_name
        raise KeyError(f"Unknown agent id in policy state: {agent_id}")

    def get_state(self) -> dict[str, Any]:
        return {
            "protocol_mode": self.protocol_mode,
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "estimator_type": self.estimator_type,
            "update_type": self.update_type,
            "weights": {agent_id: round(weight, 6) for agent_id, weight in sorted(self.weights.items())},
        }

    def get_last_selection_info(self) -> dict[str, Any]:
        return {
            "stage_probs": dict(self.last_stage_probs),
            "path_prob": self.last_path_prob,
            "estimator_type": self.estimator_type,
            "update_type": self.update_type,
        }


class PathCentricUnsharedPolicy(BasePolicy):
    """Path-centric unshared-bandit policy over complete leaves.

    The basic objects are full root-to-leaf paths and their induced subtree
    aggregates. Selection walks the prefix tree explicitly; update only touches
    the sampled leaf, then repairs prefix sums via the corresponding weight
    delta.
    """

    def __init__(
        self,
        seed: int = 0,
        protocol_mode: str = "force_unshared",
        eta: float = 0.2,
        epsilon: float = 0.0,
        update_type: str = "path_centric_unshared",
    ) -> None:
        super().__init__(seed=seed, protocol_mode=protocol_mode)
        self.eta = eta
        self.epsilon = epsilon
        self.update_type = update_type
        self.paths: list[list[str]] = []
        self.theta: dict[tuple[str, ...], float] = {}
        self.leaf_weights: dict[tuple[str, ...], float] = {}
        self.prefix_weights: dict[tuple[str, ...], float] = {}
        self.last_stage_probs: dict[str, float] = {}
        self.last_path_prob: float = 0.0

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.paths = enumerate_all_paths(env)
        self.theta = {}
        self.leaf_weights = {}
        self.prefix_weights = {}
        self.last_stage_probs = {}
        self.last_path_prob = 0.0

        for path in self.paths:
            leaf = tuple(path)
            self.theta[leaf] = 0.0
            self.leaf_weights[leaf] = 1.0
            for prefix in self._prefixes(leaf):
                self.prefix_weights[prefix] = self.prefix_weights.get(prefix, 0.0) + 1.0

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.paths:
            self.bind_env(env)

        current_prefix: tuple[str, ...] = ()
        self.last_stage_probs = {}
        path_prob = 1.0

        for stage_name in env.STAGE_NAMES:
            current_prefix, conditional_prob = self._sample_child_prefix(current_prefix, stage_name, env)
            self.last_stage_probs[stage_name] = conditional_prob
            path_prob *= conditional_prob

        if len(current_prefix) != len(env.STAGE_NAMES):
            raise RuntimeError(
                "PathCentricUnsharedPolicy failed to sample a complete leaf path. "
                f"Got prefix of length {len(current_prefix)}."
            )
        self.last_path_prob = path_prob
        return list(current_prefix)

    def update(self, episode_result: EpisodeResult) -> None:
        leaf = tuple(episode_result.selected_path)
        if leaf not in self.theta or leaf not in self.leaf_weights:
            raise KeyError(f"Selected path is unknown to {self.__class__.__name__}: {leaf}")

        observed_cost = episode_result.total_cost
        path_prob = max(self.last_path_prob, 1e-12)
        estimated_loss = observed_cost / path_prob

        old_weight = self.leaf_weights[leaf]
        self.theta[leaf] = self.theta[leaf] - self.eta * estimated_loss
        new_weight = math.exp(self.theta[leaf])
        self.leaf_weights[leaf] = new_weight

        delta = new_weight - old_weight
        for prefix in self._prefixes(leaf):
            self.prefix_weights[prefix] = self.prefix_weights.get(prefix, 0.0) + delta

    def _prefixes(self, leaf: tuple[str, ...]) -> list[tuple[str, ...]]:
        return [tuple(leaf[:depth]) for depth in range(0, len(leaf) + 1)]

    def _child_prefixes(
        self,
        current_prefix: tuple[str, ...],
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> list[tuple[str, ...]]:
        expected_depth = len(current_prefix)
        if expected_depth >= len(env.STAGE_NAMES):
            return []
        if env.STAGE_NAMES[expected_depth] != stage_name:
            raise ValueError(
                f"Prefix depth {expected_depth} expects stage {env.STAGE_NAMES[expected_depth]}, "
                f"but got {stage_name}."
            )

        child_prefixes: set[tuple[str, ...]] = set()
        for path in self.paths:
            leaf = tuple(path)
            if leaf[:expected_depth] != current_prefix:
                continue
            if len(leaf) <= expected_depth:
                continue
            child_prefixes.add(tuple(leaf[: expected_depth + 1]))
        return sorted(child_prefixes)

    def _child_sampling_distribution(
        self,
        child_prefixes: list[tuple[str, ...]],
    ) -> list[float]:
        if not child_prefixes:
            raise ValueError("Cannot build a child distribution from an empty prefix set.")

        subtree_weights = [
            max(0.0, self.prefix_weights.get(child_prefix, 0.0))
            for child_prefix in child_prefixes
        ]
        total_weight = sum(subtree_weights)
        num_children = len(child_prefixes)

        if total_weight <= 0:
            return [1.0 / num_children for _ in child_prefixes]

        probs: list[float] = []
        for subtree_weight in subtree_weights:
            exploit_prob = subtree_weight / total_weight
            mixed_prob = (1.0 - self.epsilon) * exploit_prob + self.epsilon * (1.0 / num_children)
            probs.append(mixed_prob)
        return probs

    def _sample_child_prefix(
        self,
        current_prefix: tuple[str, ...],
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> tuple[tuple[str, ...], float]:
        child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
        if not child_prefixes:
            raise RuntimeError(
                "No legal child prefixes found during subtree sampling. "
                f"current_prefix={list(current_prefix)} stage_name={stage_name}"
            )

        probs = self._child_sampling_distribution(child_prefixes)
        selected_idx = self._sample_index(probs)
        return child_prefixes[selected_idx], probs[selected_idx]

    def get_state(self) -> dict[str, Any]:
        top_leaves = sorted(
            self.leaf_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        return {
            "protocol_mode": self.protocol_mode,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "estimator_type": "loss",
            "update_type": self.update_type,
            "num_leaves": len(self.leaf_weights),
            "root_weight": round(self.prefix_weights.get((), 0.0), 6),
            "top_leaf_weights": [
                {"path": list(path), "weight": round(weight, 6), "theta": round(self.theta[path], 6)}
                for path, weight in top_leaves
            ],
        }

    def get_last_selection_info(self) -> dict[str, Any]:
        return {
            "stage_probs": dict(self.last_stage_probs),
            "path_prob": self.last_path_prob,
            "estimator_type": "loss",
            "update_type": self.update_type,
        }
