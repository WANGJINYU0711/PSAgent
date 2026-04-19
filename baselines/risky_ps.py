"""Barrier-style Partial-Share / Risky-PS baseline.

This implementation aligns the internal policy mechanics with the barrier
semantics described in ``notes/PartialShare_for_Codex_clean.md``:

- ``theta[leaf]`` stores shared-leaf scores only
- ``shared_edge_mass[(prefix, child_prefix)]`` stores safe-subtree aggregates
- ``risky_theta[(prefix, child_prefix)]`` stores local risky-node parameters
- selection uses recursive safe aggregates on safe prefixes and local softmax on
  risky prefixes
- every sampled risky ancestor receives an importance-weighted update
- shared-leaf deltas propagate only through the sampled safe suffix and stop at
  the first risky ancestor barrier
"""

from __future__ import annotations

import math
from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import enumerate_all_paths


LeafKey = tuple[str, ...]
PrefixKey = tuple[str, ...]
EdgeKey = tuple[PrefixKey, PrefixKey]


class RiskyPSPolicy(BasePolicy):
    """Barrier-style Risky-PS baseline."""

    def __init__(self, seed: int = 0, eta: float = 0.2, epsilon: float = 0.2) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")
        self.eta = eta
        self.epsilon = epsilon
        self.paths: list[list[str]] = []
        self.theta: dict[LeafKey, float] = {}
        self.leaf_types: dict[LeafKey, str] = {}
        self.safe_prefixes: dict[PrefixKey, bool] = {}
        self.shared_edge_mass: dict[EdgeKey, float] = {}
        self.unshared_edge_mass: dict[EdgeKey, float] = {}
        self.unshared_edge_count: dict[EdgeKey, int] = {}
        self.risky_theta: dict[EdgeKey, float] = {}
        self.last_stage_probs: dict[str, float] = {}
        self.last_path_prob: float = 0.0
        self.last_sampled_edges: list[dict[str, Any]] = []
        self.last_update_info: dict[str, Any] = {}

    @property
    def name(self) -> str:
        return "risky_ps"

    def preferred_catalog_preset(self) -> str:
        return "mixed"

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.paths = enumerate_all_paths(env)
        self.theta = {}
        self.leaf_types = {}
        self.safe_prefixes = {}
        self.shared_edge_mass = {}
        self.unshared_edge_mass = {}
        self.unshared_edge_count = {}
        self.risky_theta = {}
        self.last_stage_probs = {}
        self.last_path_prob = 0.0
        self.last_sampled_edges = []
        self.last_update_info = {}

        descendant_types: dict[PrefixKey, set[str]] = {}
        subtree_nonleaf_all_share: dict[PrefixKey, bool] = {}
        shared_prefix_mass: dict[EdgeKey, float] = {}
        unshared_descendant_counts: dict[EdgeKey, int] = {}
        all_prefixes: set[PrefixKey] = {()}

        for path in self.paths:
            leaf = tuple(path)
            leaf_type = env.compute_leaf_type(path)
            self.leaf_types[leaf] = leaf_type
            self.theta[leaf] = 0.0

            leaf_prefixes = self._prefixes(leaf)
            for prefix in leaf_prefixes:
                all_prefixes.add(prefix)
                descendant_types.setdefault(prefix, set()).add(leaf_type)

            for depth in range(len(leaf)):
                prefix = tuple(leaf[:depth])
                child_prefix = tuple(leaf[: depth + 1])
                edge = (prefix, child_prefix)
                all_prefixes.add(child_prefix)
                if leaf_type == "unshared":
                    unshared_descendant_counts[edge] = unshared_descendant_counts.get(edge, 0) + 1

            for prefix in leaf_prefixes:
                if len(prefix) == 0 or len(prefix) == len(leaf):
                    continue
                gate_is_share = env.agent_catalog[prefix[-1]].g == 0
                subtree_nonleaf_all_share[prefix] = (
                    subtree_nonleaf_all_share.get(prefix, True) and gate_is_share
                )

        subtree_nonleaf_all_share.setdefault((), True)
        for prefix in all_prefixes:
            subtree_nonleaf_all_share.setdefault(prefix, True)

        for prefix in all_prefixes:
            descendant_leaf_types = descendant_types.get(prefix, set())
            self.safe_prefixes[prefix] = (
                bool(descendant_leaf_types)
                and descendant_leaf_types == {"shared"}
                and subtree_nonleaf_all_share.get(prefix, True)
            )

        for path in self.paths:
            leaf = tuple(path)
            if self.leaf_types[leaf] != "shared":
                continue
            leaf_weight = self._shared_leaf_weight(leaf)
            for depth in range(len(leaf)):
                prefix = tuple(leaf[:depth])
                child_prefix = tuple(leaf[: depth + 1])
                if not self.safe_prefixes.get(prefix, False):
                    continue
                edge = (prefix, child_prefix)
                shared_prefix_mass[edge] = shared_prefix_mass.get(edge, 0.0) + leaf_weight

        for path in self.paths:
            leaf = tuple(path)
            for depth in range(len(leaf)):
                prefix = tuple(leaf[:depth])
                child_prefix = tuple(leaf[: depth + 1])
                edge = (prefix, child_prefix)
                self.shared_edge_mass[edge] = shared_prefix_mass.get(edge, 0.0)
                self.unshared_edge_count[edge] = unshared_descendant_counts.get(edge, 0)
                # Keep this debug mass around for inspection, but it is no longer
                # the risky exploit state.
                self.unshared_edge_mass[edge] = float(self.unshared_edge_count[edge])
                if not self.safe_prefixes.get(prefix, False):
                    self.risky_theta.setdefault(edge, 0.0)

    def select_path(self, instance: dict[str, Any], env: FixedTreeEnvironment) -> list[str]:
        del instance
        if not self.paths:
            self.bind_env(env)

        current_prefix: PrefixKey = ()
        self.last_stage_probs = {}
        self.last_path_prob = 1.0
        self.last_sampled_edges = []
        self.last_update_info = {}
        prefix_reach_prob = 1.0

        for stage_name in env.STAGE_NAMES:
            child_prefix, conditional_prob = self._sample_child_prefix(current_prefix, stage_name, env)
            self.last_stage_probs[stage_name] = conditional_prob
            self.last_path_prob *= conditional_prob
            self.last_sampled_edges.append(
                {
                    "prefix": current_prefix,
                    "child_prefix": child_prefix,
                    "prefix_reach_prob": prefix_reach_prob,
                    "conditional_prob": conditional_prob,
                    "edge_prob": prefix_reach_prob * conditional_prob,
                    "is_safe_prefix": self.safe_prefixes.get(current_prefix, False),
                }
            )
            prefix_reach_prob *= conditional_prob
            current_prefix = child_prefix

        if len(current_prefix) != len(env.STAGE_NAMES):
            raise RuntimeError(
                "RiskyPSPolicy failed to sample a complete path. "
                f"Got prefix length {len(current_prefix)}."
            )
        return list(current_prefix)

    def update(self, episode_result: EpisodeResult) -> None:
        leaf = tuple(episode_result.selected_path)
        if leaf not in self.leaf_types:
            raise KeyError(f"Unknown selected path in RiskyPSPolicy: {leaf}")

        risky_edge_updates = self._apply_risky_edge_updates(episode_result)
        shared_leaf_updated = False
        shared_safe_suffix_edges_updated: list[dict[str, Any]] = []
        barrier_stop_prefix: list[str] | None = None
        shared_estimated_loss: float | None = None
        shared_delta: float | None = None

        if self.leaf_types.get(leaf) == "shared":
            shared_leaf_updated = True
            shared_estimated_loss = self._shared_leaf_estimated_loss(episode_result)
            shared_delta = self._apply_shared_leaf_update(leaf, shared_estimated_loss)
            touched_edges, barrier_prefix = self._propagate_barrier_limited_shared_delta(leaf, shared_delta)
            shared_safe_suffix_edges_updated = [
                {"prefix": list(prefix), "child_prefix": list(child_prefix)}
                for prefix, child_prefix in touched_edges
            ]
            barrier_stop_prefix = list(barrier_prefix) if barrier_prefix is not None else None

        self.last_update_info = {
            "update_type": "risky_ps_barrier_v1",
            "leaf_type": self.leaf_types.get(leaf),
            "observed_cost": episode_result.total_cost,
            "risky_edges_updated": risky_edge_updates,
            "shared_leaf_updated": shared_leaf_updated,
            "shared_leaf_estimated_loss": shared_estimated_loss,
            "shared_delta": shared_delta,
            "shared_safe_suffix_edges_updated": shared_safe_suffix_edges_updated,
            "barrier_stop_prefix": barrier_stop_prefix,
        }

    def _prefixes(self, leaf: LeafKey) -> list[PrefixKey]:
        return [tuple(leaf[:depth]) for depth in range(0, len(leaf) + 1)]

    def _shared_leaf_weight(self, leaf: LeafKey) -> float:
        return math.exp(self.eta * self.theta[leaf])

    def _child_prefixes(
        self,
        current_prefix: PrefixKey,
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> list[PrefixKey]:
        expected_depth = len(current_prefix)
        if expected_depth >= len(env.STAGE_NAMES):
            return []
        if env.STAGE_NAMES[expected_depth] != stage_name:
            raise ValueError(
                f"Prefix depth {expected_depth} expects stage {env.STAGE_NAMES[expected_depth]}, "
                f"but got {stage_name}."
            )

        child_prefixes: set[PrefixKey] = set()
        for path in self.paths:
            leaf = tuple(path)
            if leaf[:expected_depth] != current_prefix:
                continue
            if len(leaf) <= expected_depth:
                continue
            child_prefixes.add(tuple(leaf[: expected_depth + 1]))
        return sorted(child_prefixes)

    def _safe_child_probs(self, prefix: PrefixKey, child_prefixes: list[PrefixKey]) -> list[float]:
        child_masses = [
            max(0.0, self.shared_edge_mass.get((prefix, child_prefix), 0.0))
            for child_prefix in child_prefixes
        ]
        total_mass = sum(child_masses)
        if total_mass <= 0:
            return [1.0 / len(child_prefixes) for _ in child_prefixes]
        return [mass / total_mass for mass in child_masses]

    def _risky_child_probs(
        self,
        prefix: PrefixKey,
        child_prefixes: list[PrefixKey],
    ) -> list[float]:
        logits = [self.risky_theta.get((prefix, child_prefix), 0.0) for child_prefix in child_prefixes]
        max_logit = max(logits, default=0.0)
        exp_values = [math.exp(self.eta * (logit - max_logit)) for logit in logits]
        total = sum(exp_values)
        if total <= 0:
            exploit_probs = [1.0 / len(child_prefixes) for _ in child_prefixes]
        else:
            exploit_probs = [value / total for value in exp_values]

        local_epsilon = self._local_epsilon(prefix, child_prefixes)
        uniform = 1.0 / len(child_prefixes)
        return [
            (1.0 - local_epsilon) * exploit_prob + local_epsilon * uniform
            for exploit_prob in exploit_probs
        ]

    def _local_epsilon(self, prefix: PrefixKey, child_prefixes: list[PrefixKey]) -> float:
        del prefix
        if not child_prefixes:
            return 0.0
        # Match the barrier note's epsilon_i = 0 on the last risky branching
        # layer whose children are leaves.
        if len(child_prefixes[0]) == len(self.paths[0]):
            return 0.0
        return self.epsilon

    def _sample_child_prefix(
        self,
        current_prefix: PrefixKey,
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> tuple[PrefixKey, float]:
        child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
        if not child_prefixes:
            raise RuntimeError(
                "No legal child prefixes found in RiskyPS selection. "
                f"current_prefix={list(current_prefix)} stage_name={stage_name}"
            )

        if self.safe_prefixes.get(current_prefix, False):
            probs = self._safe_child_probs(current_prefix, child_prefixes)
        else:
            probs = self._risky_child_probs(current_prefix, child_prefixes)

        selected_idx = self._sample_index(probs)
        return child_prefixes[selected_idx], probs[selected_idx]

    def _shared_leaf_estimated_loss(self, episode_result: EpisodeResult) -> float:
        return episode_result.total_cost / max(self.last_path_prob, 1e-12)

    def _apply_shared_leaf_update(self, leaf: LeafKey, estimated_loss: float) -> float:
        old_weight = self._shared_leaf_weight(leaf)
        self.theta[leaf] = self.theta[leaf] - estimated_loss
        new_weight = self._shared_leaf_weight(leaf)
        return new_weight - old_weight

    def _propagate_barrier_limited_shared_delta(
        self,
        leaf: LeafKey,
        delta: float,
    ) -> tuple[list[EdgeKey], PrefixKey | None]:
        touched_edges: list[EdgeKey] = []
        barrier_stop_prefix: PrefixKey | None = None
        for depth in range(len(leaf) - 1, -1, -1):
            prefix = tuple(leaf[:depth])
            child_prefix = tuple(leaf[: depth + 1])
            if not self.safe_prefixes.get(prefix, False):
                barrier_stop_prefix = prefix
                break
            edge = (prefix, child_prefix)
            self.shared_edge_mass[edge] = self.shared_edge_mass.get(edge, 0.0) + delta
            touched_edges.append(edge)
        touched_edges.reverse()
        return touched_edges, barrier_stop_prefix

    def _sampled_risky_edge_infos(self) -> list[dict[str, Any]]:
        return [
            edge_info
            for edge_info in self.last_sampled_edges
            if not self.safe_prefixes.get(edge_info["prefix"], False)
        ]

    def _apply_risky_edge_updates(self, episode_result: EpisodeResult) -> list[dict[str, Any]]:
        updated_edges: list[dict[str, Any]] = []
        for edge_info in self._sampled_risky_edge_infos():
            prefix = edge_info["prefix"]
            child_prefix = edge_info["child_prefix"]
            edge = (prefix, child_prefix)
            denom = max(edge_info["edge_prob"], 1e-12)
            old_theta = self.risky_theta.get(edge, 0.0)
            new_theta = old_theta - (episode_result.total_cost / denom)
            self.risky_theta[edge] = new_theta
            updated_edges.append(
                {
                    "prefix": list(prefix),
                    "child_prefix": list(child_prefix),
                    "prefix_reach_prob": edge_info["prefix_reach_prob"],
                    "conditional_prob": edge_info["conditional_prob"],
                    "edge_prob": edge_info["edge_prob"],
                    "update_denominator": denom,
                    "old_risky_theta": old_theta,
                    "new_risky_theta": new_theta,
                }
            )
        return updated_edges

    def get_state(self) -> dict[str, Any]:
        shared_mass_total = sum(self.shared_edge_mass.values())
        return {
            "protocol_mode": self.protocol_mode,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "num_paths": len(self.paths),
            "num_safe_prefixes": sum(1 for is_safe in self.safe_prefixes.values() if is_safe),
            "num_risky_prefixes": sum(1 for is_safe in self.safe_prefixes.values() if not is_safe),
            "num_safe_aggregate_edges": sum(1 for value in self.shared_edge_mass.values() if value > 0),
            "num_risky_theta_edges": len(self.risky_theta),
            "shared_mass_total": round(shared_mass_total, 6),
            "debug_unshared_descendant_edge_count": sum(1 for value in self.unshared_edge_count.values() if value > 0),
            "max_unshared_descendant_count": max(self.unshared_edge_count.values(), default=0),
            "last_update_info": dict(self.last_update_info),
        }

    def get_last_selection_info(self) -> dict[str, Any]:
        return {
            "stage_probs": dict(self.last_stage_probs),
            "path_prob": self.last_path_prob,
            "sampled_edges": [
                {
                    "prefix": list(edge["prefix"]),
                    "child_prefix": list(edge["child_prefix"]),
                    "prefix_reach_prob": edge["prefix_reach_prob"],
                    "conditional_prob": edge["conditional_prob"],
                    "edge_prob": edge["edge_prob"],
                    "is_safe_prefix": edge["is_safe_prefix"],
                }
                for edge in self.last_sampled_edges
            ],
            "update_type": "risky_ps_barrier_v1",
            "last_update_info": dict(self.last_update_info),
        }
