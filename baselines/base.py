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

    def _legal_agent_ids_for_prefix(
        self,
        current_prefix: tuple[str, ...],
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> list[str]:
        """Return the legal next-stage agent ids under optional family topology.

        Most legacy families are plain cartesian products, so every stage agent is
        legal. Topology-aware families may expose an explicit continuation map in
        ``family_spec.allowed_children``; in that case we must sample only from the
        children reachable under the current prefix.
        """

        stage_agent_ids = self.stage_agent_ids[stage_name]
        family_spec = getattr(env, "family_spec", None)
        allowed_children = getattr(family_spec, "allowed_children", None)
        if not allowed_children:
            return list(stage_agent_ids)
        child_ids = allowed_children.get(current_prefix)
        if child_ids is None:
            return list(stage_agent_ids)
        return list(child_ids)

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
        current_prefix: tuple[str, ...] = ()
        for stage_name in env.STAGE_NAMES:
            agent_ids = self._legal_agent_ids_for_prefix(current_prefix, stage_name, env)
            if self.rng.random() < self.epsilon:
                chosen = self.rng.choice(agent_ids)
                path.append(chosen)
                current_prefix = current_prefix + (chosen,)
                continue

            ranked = sorted(
                agent_ids,
                key=lambda agent_id: (self._mean_cost(agent_id), self.counts.get(agent_id, 0)),
            )
            best_mean = self._mean_cost(ranked[0])
            tied = [agent_id for agent_id in ranked if math.isclose(self._mean_cost(agent_id), best_mean)]
            chosen = self.rng.choice(tied)
            path.append(chosen)
            current_prefix = current_prefix + (chosen,)
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
    """Strict tree-local stagewise Exp3 over parent-child interfaces.

    The learned objects are local edges, not global stage agents: even if the
    family reuses the same child agent id under multiple prefixes, each
    ``(prefix, child_prefix)`` pair is its own arm. Selection uses
    ``softmax(eta * theta)`` under the current prefix. The optional epsilon
    mixture is used only by the epsilon-EXP3 subclass.

    ``conditional_prob`` is the branch-conditioned local probability of
    choosing the direct child after reaching the current prefix. If
    ``epsilon > 0``, the policy first samples mode ``U`` with probability
    epsilon or mode ``E`` otherwise; the update denominator uses the selected
    branch probability, not the marginal mixture probability.
    ``prefix_reach_prob`` (also logged as ``path_prob_so_far``) is the
    probability of reaching that prefix, matching ``v[i,t]`` in the paper
    figure. The multilevel EXP3 estimator updates each selected edge with
    denominator ``edge_prob = prefix_reach_prob * conditional_prob``.
    """

    def __init__(
        self,
        seed: int = 0,
        protocol_mode: str = "force_unshared",
        eta: float = 0.2,
        epsilon: float = 0.0,
        estimator_type: str = "loss",
        update_type: str = "stagewise_exp3_theta_loss",
    ) -> None:
        super().__init__(seed=seed, protocol_mode=protocol_mode)
        self.eta = eta
        self.epsilon = epsilon
        self.estimator_type = estimator_type
        self.update_type = update_type
        self.theta: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}
        # Legacy/debug alias populated from theta for callers that inspect
        # derived multiplicative weights. Updates are driven by theta.
        self.weights: dict[tuple[tuple[str, ...], tuple[str, ...]], float] = {}
        self.last_conditional_probs: list[float] = []
        self.last_path_probs: list[float] = []
        self.last_stage_probs: dict[str, float] = {}
        self.last_stage_arm_counts: dict[str, int] = {}
        self.last_path_prob: float = 0.0
        self.last_selected_edges: list[dict[str, Any]] = []
        self.last_update_info: dict[str, Any] = {}

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.last_conditional_probs = []
        self.last_path_probs = []
        self.last_stage_probs = {}
        self.last_stage_arm_counts = {}
        self.last_path_prob = 0.0
        self.last_selected_edges = []
        self.last_update_info = {}

    def _edge_key(
        self,
        current_prefix: tuple[str, ...],
        child_prefix: tuple[str, ...],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        return (tuple(current_prefix), tuple(child_prefix))

    def _edge_theta(
        self,
        current_prefix: tuple[str, ...],
        child_prefix: tuple[str, ...],
    ) -> float:
        return self.theta.setdefault(self._edge_key(current_prefix, child_prefix), 0.0)

    def _edge_weight(
        self,
        current_prefix: tuple[str, ...],
        child_prefix: tuple[str, ...],
    ) -> float:
        edge_key = self._edge_key(current_prefix, child_prefix)
        theta = self.theta.setdefault(edge_key, 0.0)
        weight = math.exp(max(-60.0, min(60.0, self.eta * theta)))
        self.weights[edge_key] = weight
        return weight

    def _child_prefixes(
        self,
        current_prefix: tuple[str, ...],
        agent_ids: list[str],
    ) -> list[tuple[str, ...]]:
        return [tuple(current_prefix + (agent_id,)) for agent_id in agent_ids]

    def _softmax_child_probs(
        self,
        current_prefix: tuple[str, ...],
        child_prefixes: list[tuple[str, ...]],
    ) -> list[float]:
        if not child_prefixes:
            raise ValueError("Cannot build stage probabilities for an empty child set.")

        local_theta = [
            self._edge_theta(current_prefix, child_prefix)
            for child_prefix in child_prefixes
        ]
        max_theta = max(local_theta, default=0.0)
        exp_values = [
            math.exp(max(-60.0, min(60.0, self.eta * (theta - max_theta))))
            for theta in local_theta
        ]
        total = sum(exp_values)
        if total <= 0.0:
            return [1.0 / len(child_prefixes) for _ in child_prefixes]
        return [value / total for value in exp_values]

    def _stage_probs(
        self,
        current_prefix: tuple[str, ...],
        child_prefixes: list[tuple[str, ...]],
    ) -> list[float]:
        """Return the marginal mixture distribution for diagnostics/sync only."""

        exploit_probs = self._softmax_child_probs(current_prefix, child_prefixes)
        epsilon = min(1.0, max(0.0, float(self.epsilon)))
        if epsilon <= 0.0:
            return exploit_probs
        uniform = 1.0 / len(child_prefixes)
        return [
            (1.0 - epsilon) * exploit_prob + epsilon * uniform
            for exploit_prob in exploit_probs
        ]

    def _sample_stage_child(
        self,
        current_prefix: tuple[str, ...],
        child_prefixes: list[tuple[str, ...]],
    ) -> tuple[tuple[str, ...], dict[str, Any]]:
        exploit_probs = self._softmax_child_probs(current_prefix, child_prefixes)
        epsilon = min(1.0, max(0.0, float(self.epsilon)))
        arm_count = len(child_prefixes)
        uniform_prob = 1.0 / arm_count
        mixture_probs = [
            (1.0 - epsilon) * exploit_prob + epsilon * uniform_prob
            for exploit_prob in exploit_probs
        ]

        if epsilon > 0.0 and self.rng.random() < epsilon:
            epsilon_mode = "U"
            selection_mode = "epsilon_uniform"
            branch_probs = [uniform_prob for _ in child_prefixes]
        else:
            epsilon_mode = "E"
            selection_mode = "epsilon_exploit" if epsilon > 0.0 else "direct_exploit"
            branch_probs = exploit_probs

        child_prefix, branch_prob = self._sample_from_probs(child_prefixes, branch_probs)
        selected_idx = child_prefixes.index(child_prefix)
        return child_prefix, {
            "epsilon": epsilon,
            "epsilon_mode": epsilon_mode,
            "selection_mode": selection_mode,
            "branch_conditional_prob": branch_prob,
            "conditional_prob": branch_prob,
            "mixture_conditional_prob": mixture_probs[selected_idx],
            "softmax_conditional_prob": exploit_probs[selected_idx],
            "uniform_conditional_prob": uniform_prob,
        }

    def _sample_from_probs(
        self,
        child_prefixes: list[tuple[str, ...]],
        probs: list[float],
    ) -> tuple[tuple[str, ...], float]:
        draw = self.rng.random()
        cumulative = 0.0
        for child_prefix, prob in zip(child_prefixes, probs):
            cumulative += prob
            if draw <= cumulative:
                return child_prefix, prob
        return child_prefixes[-1], probs[-1]

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.stage_agent_ids:
            self.bind_env(env)

        path: list[str] = []
        self.last_conditional_probs = []
        self.last_path_probs = []
        self.last_stage_probs = {}
        self.last_stage_arm_counts = {}
        self.last_selected_edges = []
        self.last_update_info = {}
        current_prefix: tuple[str, ...] = ()
        path_prob_so_far = 1.0
        for stage_name in env.STAGE_NAMES:
            agent_ids = self._legal_agent_ids_for_prefix(current_prefix, stage_name, env)
            child_prefixes = self._child_prefixes(current_prefix, agent_ids)
            child_prefix, selection_meta = self._sample_stage_child(current_prefix, child_prefixes)
            prob = float(selection_meta["branch_conditional_prob"])
            path.append(child_prefix[-1])
            self.last_conditional_probs.append(prob)
            self.last_path_probs.append(prob)
            self.last_stage_probs[stage_name] = prob
            self.last_stage_arm_counts[stage_name] = len(child_prefixes)
            self.last_selected_edges.append(
                {
                    "stage_name": stage_name,
                    "prefix": current_prefix,
                    "child_prefix": child_prefix,
                    "prefix_reach_prob": path_prob_so_far,
                    "epsilon": selection_meta["epsilon"],
                    "epsilon_mode": selection_meta["epsilon_mode"],
                    "selection_mode": selection_meta["selection_mode"],
                    "branch_conditional_prob": selection_meta["branch_conditional_prob"],
                    "conditional_prob": prob,
                    "mixture_conditional_prob": selection_meta["mixture_conditional_prob"],
                    "softmax_conditional_prob": selection_meta["softmax_conditional_prob"],
                    "uniform_conditional_prob": selection_meta["uniform_conditional_prob"],
                    "path_prob_so_far": path_prob_so_far,
                    "edge_prob": path_prob_so_far * prob,
                    "estimated_loss_denominator": "branch_edge_prob",
                    "estimator_scope": "branch_conditioned_multilevel_edge_probability",
                    "arm_count": len(child_prefixes),
                    "theta_before_update": self._edge_theta(current_prefix, child_prefix),
                    "weight_before_update": self._edge_weight(current_prefix, child_prefix),
                }
            )
            path_prob_so_far *= prob
            current_prefix = child_prefix
        path_prob = 1.0
        for prob in self.last_conditional_probs:
            path_prob *= prob
        self.last_path_prob = path_prob
        return path

    def update(self, episode_result: EpisodeResult) -> None:
        if len(self.last_selected_edges) != len(episode_result.selected_path):
            raise RuntimeError(
                "StagewiseExp3Policy update called without matching select_path metadata. "
                f"selected_path_len={len(episode_result.selected_path)} "
                f"last_selected_edges={len(self.last_selected_edges)}"
            )

        if self.estimator_type != "loss":
            raise ValueError(
                "StagewiseExp3Policy now implements loss-style theta updates only. "
                f"Got estimator_type={self.estimator_type!r}."
            )

        observed_loss = max(0.0, float(episode_result.total_cost))
        edge_updates: list[dict[str, Any]] = []
        for selected_edge in self.last_selected_edges:
            stage_name = str(selected_edge["stage_name"])
            current_prefix = tuple(selected_edge["prefix"])
            child_prefix = tuple(selected_edge["child_prefix"])
            conditional_prob = float(selected_edge["conditional_prob"])
            prefix_reach_prob = float(
                selected_edge.get(
                    "prefix_reach_prob",
                    selected_edge.get("path_prob_so_far", 1.0),
                )
            )
            edge_prob = float(
                selected_edge.get("edge_prob", prefix_reach_prob * conditional_prob)
            )
            edge_key = self._edge_key(current_prefix, child_prefix)
            estimated_loss = observed_loss / max(edge_prob, 1e-12)
            theta_before = self.theta.setdefault(edge_key, 0.0)
            theta_after = theta_before - estimated_loss
            self.theta[edge_key] = theta_after
            self.weights[edge_key] = self._edge_weight(current_prefix, child_prefix)
            self.last_stage_probs[stage_name] = conditional_prob
            selected_edge["estimated_loss"] = estimated_loss
            selected_edge["observed_loss"] = observed_loss
            selected_edge["theta_before_update"] = theta_before
            selected_edge["theta_after_update"] = theta_after
            selected_edge["estimated_loss_denominator"] = selected_edge.get(
                "estimated_loss_denominator",
                "branch_edge_prob",
            )
            selected_edge["estimator_scope"] = selected_edge.get(
                "estimator_scope",
                "branch_conditioned_multilevel_edge_probability",
            )
            edge_updates.append(
                {
                    "stage_name": stage_name,
                    "prefix": list(current_prefix),
                    "child_prefix": list(child_prefix),
                    "prefix_reach_prob": prefix_reach_prob,
                    "conditional_prob": conditional_prob,
                    "branch_conditional_prob": float(
                        selected_edge.get("branch_conditional_prob", conditional_prob)
                    ),
                    "mixture_conditional_prob": selected_edge.get("mixture_conditional_prob"),
                    "epsilon": selected_edge.get("epsilon", self.epsilon),
                    "epsilon_mode": selected_edge.get("epsilon_mode"),
                    "selection_mode": selected_edge.get("selection_mode"),
                    "path_prob_so_far": prefix_reach_prob,
                    "edge_prob": edge_prob,
                    "estimated_loss_denominator": selected_edge["estimated_loss_denominator"],
                    "estimator_scope": selected_edge["estimator_scope"],
                    "arm_count": int(selected_edge["arm_count"]),
                    "observed_loss": observed_loss,
                    "estimated_loss": estimated_loss,
                    "theta_before_update": theta_before,
                    "theta_after_update": theta_after,
                    "update_type": self.update_type,
                }
            )

        self.last_update_info = {
            "update_type": self.update_type,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "observed_loss": observed_loss,
            "edge_updates": edge_updates,
        }

    def get_state(self) -> dict[str, Any]:
        top_local_edges = sorted(self.theta.items(), key=lambda item: item[1], reverse=True)[:10]
        return {
            "protocol_mode": self.protocol_mode,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "estimator_type": self.estimator_type,
            "update_type": self.update_type,
            "num_local_edge_arms": len(self.theta),
            "theta": [
                {
                    "prefix": list(prefix),
                    "child_prefix": list(child_prefix),
                    "theta": round(theta, 6),
                    "weight": round(math.exp(max(-60.0, min(60.0, self.eta * theta))), 6),
                }
                for (prefix, child_prefix), theta in top_local_edges
            ],
            "last_update_info": dict(self.last_update_info),
        }

    def get_last_selection_info(self) -> dict[str, Any]:
        return {
            "stage_probs": dict(self.last_stage_probs),
            "stage_conditional_probs": dict(self.last_stage_probs),
            "selected_edges": [
                {
                    "stage_name": str(edge["stage_name"]),
                    "prefix": list(edge["prefix"]),
                    "child_prefix": list(edge["child_prefix"]),
                    "prefix_reach_prob": float(
                        edge.get("prefix_reach_prob", edge.get("path_prob_so_far", 1.0))
                    ),
                    "epsilon": edge.get("epsilon", self.epsilon),
                    "epsilon_mode": edge.get("epsilon_mode"),
                    "selection_mode": edge.get("selection_mode"),
                    "branch_conditional_prob": float(
                        edge.get("branch_conditional_prob", edge["conditional_prob"])
                    ),
                    "conditional_prob": float(edge["conditional_prob"]),
                    "mixture_conditional_prob": edge.get("mixture_conditional_prob"),
                    "path_prob_so_far": float(edge.get("path_prob_so_far", 1.0)),
                    "edge_prob": float(
                        edge.get(
                            "edge_prob",
                            float(edge.get("path_prob_so_far", 1.0))
                            * float(edge["conditional_prob"]),
                        )
                    ),
                    "arm_count": int(edge["arm_count"]),
                    "theta_before_update": float(edge["theta_before_update"]),
                    "theta_after_update": edge.get("theta_after_update"),
                    "estimated_loss": edge.get("estimated_loss"),
                    "estimated_loss_denominator": edge.get(
                        "estimated_loss_denominator",
                        "branch_edge_prob",
                    ),
                    "estimator_scope": edge.get(
                        "estimator_scope",
                        "branch_conditioned_multilevel_edge_probability",
                    ),
                    "observed_loss": edge.get("observed_loss"),
                    "weight_before_update": float(edge["weight_before_update"]),
                }
                for edge in self.last_selected_edges
            ],
            "path_prob": self.last_path_prob,
            "conditional_prob_product": self.last_path_prob,
            "estimator_type": self.estimator_type,
            "update_type": self.update_type,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "last_update_info": dict(self.last_update_info),
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
