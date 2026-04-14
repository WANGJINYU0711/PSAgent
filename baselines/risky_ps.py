"""Partial-Share baseline with explicit shared / unshared edge aggregates.

This version is a closer structural match to Algorithm 1 in
``notes/PartialShare.md``:

- ``theta[leaf]`` stores leaf scores
- ``leaf_types[leaf]`` stores shared / unshared endpoint semantics
- ``safe_prefixes[prefix]`` marks prefixes whose descendant leaves are all shared
- ``shared_edge_mass[(prefix, child_prefix)]`` stores the shared subtree aggregate
- ``unshared_edge_mass[(prefix, child_prefix)]`` stores a separate unshared mass

Selection uses the combined edge mass ``Sshr + Sunshr`` with risky-only local
epsilon exploration. Update is split into explicit shared / unshared branches:

- shared leaves update ``theta`` and push the induced delta through the shared
  channel along the sampled ancestors
- unshared leaves apply multiplicative updates directly to the unshared channel
  on sampled risky ancestor edges
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
    """First-pass Adaptive Partial-Share baseline."""

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
        self.last_stage_probs = {}
        self.last_path_prob = 0.0
        self.last_sampled_edges = []
        self.last_update_info = {}

        descendant_types: dict[PrefixKey, set[str]] = {}
        unshared_descendant_counts: dict[EdgeKey, int] = {}
        for path in self.paths:
            leaf = tuple(path)
            leaf_type = env.compute_leaf_type(path)
            self.theta[leaf] = 0.0
            self.leaf_types[leaf] = leaf_type
            for prefix in self._prefixes(leaf):
                descendant_types.setdefault(prefix, set()).add(leaf_type)
            if leaf_type == "unshared":
                for depth in range(len(leaf)):
                    prefix = tuple(leaf[:depth])
                    child_prefix = tuple(leaf[: depth + 1])
                    edge = (prefix, child_prefix)
                    unshared_descendant_counts[edge] = unshared_descendant_counts.get(edge, 0) + 1

        for prefix, leaf_type_set in descendant_types.items():
            self.safe_prefixes[prefix] = leaf_type_set == {"shared"}

        for path in self.paths:
            leaf = tuple(path)
            leaf_weight = self._shared_leaf_weight(leaf)
            for depth in range(len(leaf)):
                prefix = tuple(leaf[:depth])
                child_prefix = tuple(leaf[: depth + 1])
                edge = (prefix, child_prefix)
                if self.leaf_types[leaf] == "shared":
                    self.shared_edge_mass[edge] = self.shared_edge_mass.get(edge, 0.0) + leaf_weight
                    self.unshared_edge_mass.setdefault(edge, 0.0)
                else:
                    self.shared_edge_mass.setdefault(edge, 0.0)
                    self.unshared_edge_mass.setdefault(edge, float(unshared_descendant_counts.get(edge, 0)))
                    self.unshared_edge_count[edge] = unshared_descendant_counts.get(edge, 0)

        for edge in set(self.shared_edge_mass) | set(self.unshared_edge_mass):
            self.shared_edge_mass.setdefault(edge, 0.0)
            self.unshared_edge_mass.setdefault(edge, 0.0)
            self.unshared_edge_count.setdefault(edge, 0)

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
        if leaf not in self.theta:
            raise KeyError(f"Unknown selected path in RiskyPSPolicy: {leaf}")

        if self.leaf_types.get(leaf) == "shared":
            self.last_update_info = self._update_shared_branch(leaf, episode_result)
        else:
            self.last_update_info = self._update_unshared_branch(leaf, episode_result)

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

    def _edge_masses(self, prefix: PrefixKey, child_prefixes: list[PrefixKey]) -> list[tuple[float, float, float]]:
        masses: list[tuple[float, float, float]] = []
        for child_prefix in child_prefixes:
            edge = (prefix, child_prefix)
            shared_mass = max(0.0, self.shared_edge_mass.get(edge, 0.0))
            unshared_mass = max(0.0, self.unshared_edge_mass.get(edge, 0.0))
            masses.append((shared_mass, unshared_mass, shared_mass + unshared_mass))
        return masses

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

        mass_rows = self._edge_masses(current_prefix, child_prefixes)
        combined_masses = [row[2] for row in mass_rows]
        total_mass = sum(combined_masses)
        num_children = len(child_prefixes)
        if total_mass <= 0:
            exploit_probs = [1.0 / num_children for _ in child_prefixes]
        else:
            exploit_probs = [mass / total_mass for mass in combined_masses]

        local_epsilon = 0.0 if self.safe_prefixes.get(current_prefix, False) else self.epsilon
        probs = [
            (1.0 - local_epsilon) * exploit_prob + local_epsilon * (1.0 / num_children)
            for exploit_prob in exploit_probs
        ]
        selected_idx = self._sample_index(probs)
        return child_prefixes[selected_idx], probs[selected_idx]

    def _shared_leaf_estimated_loss(self, episode_result: EpisodeResult) -> float:
        # Align with Algorithm 1 line 26 in notes/PartialShare.md:
        # the sampled shared leaf uploads the importance-weighted estimator
        # c_t / Pi_t(leaf_t), where Pi_t(leaf_t) is the full sampled path
        # probability.
        return episode_result.total_cost / max(self.last_path_prob, 1e-12)

    def _apply_shared_leaf_update(self, leaf: LeafKey, estimated_loss: float) -> float:
        old_weight = self._shared_leaf_weight(leaf)
        self.theta[leaf] = self.theta[leaf] - estimated_loss
        new_weight = self._shared_leaf_weight(leaf)
        return new_weight - old_weight

    def _propagate_shared_delta(self, leaf: LeafKey, delta: float) -> list[EdgeKey]:
        touched_edges: list[EdgeKey] = []
        for depth in range(len(leaf)):
            prefix = tuple(leaf[:depth])
            child_prefix = tuple(leaf[: depth + 1])
            edge = (prefix, child_prefix)
            self.shared_edge_mass[edge] = self.shared_edge_mass.get(edge, 0.0) + delta
            touched_edges.append(edge)
        return touched_edges

    def _sampled_risky_edge_infos(self) -> list[dict[str, Any]]:
        return [
            edge_info
            for edge_info in self.last_sampled_edges
            if not self.safe_prefixes.get(edge_info["prefix"], False)
        ]

    def _update_shared_branch(self, leaf: LeafKey, episode_result: EpisodeResult) -> dict[str, Any]:
        observed_cost = episode_result.total_cost
        shared_path_prob = max(self.last_path_prob, 1e-12)
        estimated_loss = self._shared_leaf_estimated_loss(episode_result)
        delta = self._apply_shared_leaf_update(leaf, estimated_loss)
        touched_edges = self._propagate_shared_delta(leaf, delta)
        return {
            "branch_type": "shared",
            "observed_cost": observed_cost,
            "shared_path_prob": shared_path_prob,
            "shared_estimated_loss": estimated_loss,
            "shared_delta": delta,
            "shared_edges_updated": [
                {"prefix": list(prefix), "child_prefix": list(child_prefix)}
                for prefix, child_prefix in touched_edges
            ],
            "risky_edges_updated": [],
        }

    def _update_unshared_branch(self, leaf: LeafKey, episode_result: EpisodeResult) -> dict[str, Any]:
        risky_edge_infos = self._sampled_risky_edge_infos()
        updated_edges = self._apply_unshared_edge_update(episode_result, risky_edge_infos)
        return {
            "branch_type": "unshared",
            "unshared_estimated_loss": episode_result.total_cost,
            "unshared_update_mode": "sampled_risky_edge_multiplicative_v2",
            "shared_delta": None,
            "shared_edges_updated": [],
            "risky_edges_updated": updated_edges,
            "selected_unshared_leaf": list(leaf),
        }

    def _apply_unshared_edge_update(
        self,
        episode_result: EpisodeResult,
        risky_edge_infos: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Update maintained unshared aggregate mass on sampled risky edges only.

        This is still a first-pass approximation, but it is closer to the note's
        ``Sunshr`` semantics than the old leaf-only view:

        - the updated object is the edge-level unshared aggregate state
        - only sampled risky ancestors are touched
        - the multiplicative shrink uses the sampled edge probability as the
          denominator, matching the importance-weighted flavor of Algorithm 1
        """

        observed_cost = episode_result.total_cost
        updated_edges: list[dict[str, Any]] = []
        for edge_info in risky_edge_infos:
            prefix = edge_info["prefix"]
            edge = (prefix, edge_info["child_prefix"])
            denom = max(edge_info["edge_prob"], 1e-12)
            old_mass = max(self.unshared_edge_mass.get(edge, 0.0), 1e-12)
            new_mass = old_mass * math.exp(-self.eta * observed_cost / denom)
            self.unshared_edge_mass[edge] = new_mass
            updated_edges.append(
                {
                    "prefix": list(prefix),
                    "child_prefix": list(edge_info["child_prefix"]),
                    "prefix_reach_prob": edge_info["prefix_reach_prob"],
                    "conditional_prob": edge_info["conditional_prob"],
                    "edge_prob": edge_info["edge_prob"],
                    "update_denominator": denom,
                    "unshared_descendant_count": self.unshared_edge_count.get(edge, 0),
                    "old_unshared_mass": old_mass,
                    "new_unshared_mass": new_mass,
                }
            )
        return updated_edges

    def get_state(self) -> dict[str, Any]:
        shared_mass_total = sum(self.shared_edge_mass.values())
        unshared_mass_total = sum(self.unshared_edge_mass.values())
        return {
            "protocol_mode": self.protocol_mode,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "unshared_init_mode": "unshared_descendant_count_v2",
            "num_paths": len(self.paths),
            "num_safe_prefixes": sum(1 for is_safe in self.safe_prefixes.values() if is_safe),
            "num_risky_prefixes": sum(1 for is_safe in self.safe_prefixes.values() if not is_safe),
            "num_shared_edges": sum(1 for value in self.shared_edge_mass.values() if value > 0),
            "num_unshared_edges": sum(1 for value in self.unshared_edge_mass.values() if value > 0),
            "shared_mass_total": round(shared_mass_total, 6),
            "unshared_mass_total": round(unshared_mass_total, 6),
            "max_unshared_descendant_count": max(self.unshared_edge_count.values(), default=0),
            "last_update_info": dict(self.last_update_info),
        }

    def get_last_selection_info(self) -> dict[str, Any]:
        return {
            "stage_probs": dict(self.last_stage_probs),
            "path_prob": self.last_path_prob,
            "unshared_init_mode": "unshared_descendant_count_v2",
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
            "update_type": "risky_ps_dual_mass_v3",
            "last_update_info": dict(self.last_update_info),
        }
