"""Layered-barrier Partial-Share / Risky-PS baseline.

This implementation follows the updated upload semantics:

- the sampled leaf starts shared upload iff its own ``g=0``
- ``shared_edge_mass[(prefix, child_prefix)]`` is the aggregate mass of
  upload-reachable shared leaves in that child subtree; ``shared_edge_weight``
  is a backward-compatible mirror of the same aggregate mass
- a child is treated as full-share for a parent iff every reachable leaf in
  that child subtree can upload all the way to that parent
- a parent uses shared aggregation iff all of its legal children are full-share
  children for that parent
- otherwise that parent is handled locally by the same branch-conditioned
  epsilon-EXP3 update as ``epsilon_exp3``
- risky/mixed edge updates use the branch-conditioned multilevel EXP3
  denominator ``edge_prob = prefix_reach_prob * branch_conditional_prob``;
  safe prefixes do not receive risky updates and do not sample epsilon modes
- shared leaf weights use ``eta_shared`` so full-path importance-weighted
  shared updates can be stabilized independently from local risky updates
- shared deltas propagate only along the sampled leaf's upload-reachable edges
  and stop at the first internal ``g=1`` barrier
"""

from __future__ import annotations

import math
from typing import Any

from base import BasePolicy
from fixed_tree_env import (
    EpisodeResult,
    FixedTreeEnvironment,
    compute_shared_upload_edges,
    compute_shared_upload_stop_prefix,
)
from oracle_eval import enumerate_all_paths


LeafKey = tuple[str, ...]
PrefixKey = tuple[str, ...]
EdgeKey = tuple[PrefixKey, PrefixKey]


class RiskyPSPolicy(BasePolicy):
    """Barrier-style Risky-PS baseline."""

    def __init__(
        self,
        seed: int = 0,
        eta: float = 0.2,
        eta_shared: float | None = 0.05,
        epsilon: float = 0.1,
        loss_clip: float | None = None,
        prob_floor: float = 0.0,
    ) -> None:
        super().__init__(seed=seed, protocol_mode="actual_leaf")
        self.eta = eta
        self.eta_shared = eta if eta_shared is None else eta_shared
        self.epsilon = epsilon
        self.loss_clip = loss_clip
        self.prob_floor = prob_floor
        self.completed_updates = 0
        self.paths: list[list[str]] = []
        self.theta: dict[LeafKey, float] = {}
        self.leaf_types: dict[LeafKey, str] = {}
        self.safe_prefixes: dict[PrefixKey, bool] = {}
        self.mixed_prefixes: dict[PrefixKey, bool] = {}
        # Aggregate upload-reachable shared leaf mass for each parent-child
        # interface. This is not a per-child normalized score.
        self.shared_edge_mass: dict[EdgeKey, float] = {}
        # Backward-compatible alias for diagnostics; mirrors shared_edge_mass.
        self.shared_edge_weight: dict[EdgeKey, float] = {}
        self.full_share_child_edges: dict[EdgeKey, bool] = {}
        self.descendant_leaf_count: dict[EdgeKey, int] = {}
        self.shared_reachable_leaf_count: dict[EdgeKey, int] = {}
        self.unshared_edge_mass: dict[EdgeKey, float] = {}
        self.unshared_edge_count: dict[EdgeKey, int] = {}
        self.risky_theta: dict[EdgeKey, float] = {}
        self.upload_edges_by_leaf: dict[LeafKey, list[EdgeKey]] = {}
        self.barrier_stop_prefix_by_leaf: dict[LeafKey, PrefixKey | None] = {}
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
        self.mixed_prefixes = {}
        self.shared_edge_weight = {}
        self.shared_edge_mass = {}
        self.full_share_child_edges = {}
        self.descendant_leaf_count = {}
        self.shared_reachable_leaf_count = {}
        self.unshared_edge_mass = {}
        self.unshared_edge_count = {}
        self.risky_theta = {}
        self.upload_edges_by_leaf = {}
        self.barrier_stop_prefix_by_leaf = {}
        self.last_stage_probs = {}
        self.last_path_prob = 0.0
        self.last_sampled_edges = []
        self.last_update_info = {}

        parent_children: dict[PrefixKey, set[PrefixKey]] = {}
        total_descendant_counts: dict[EdgeKey, int] = {}
        shared_reachable_counts: dict[EdgeKey, int] = {}
        initial_shared_edge_mass: dict[EdgeKey, float] = {}

        for path in self.paths:
            leaf = tuple(path)
            leaf_type = env.compute_leaf_type(path)
            self.leaf_types[leaf] = leaf_type
            self.theta[leaf] = 0.0
            leaf_weight = self._shared_leaf_weight(leaf)

            for depth in range(len(leaf)):
                prefix = tuple(leaf[:depth])
                child_prefix = tuple(leaf[: depth + 1])
                edge = (prefix, child_prefix)
                parent_children.setdefault(prefix, set()).add(child_prefix)
                total_descendant_counts[edge] = total_descendant_counts.get(edge, 0) + 1

            upload_edges = compute_shared_upload_edges(path, env.agent_catalog)
            self.upload_edges_by_leaf[leaf] = upload_edges
            self.barrier_stop_prefix_by_leaf[leaf] = compute_shared_upload_stop_prefix(
                path,
                env.agent_catalog,
            )
            for edge in upload_edges:
                shared_reachable_counts[edge] = shared_reachable_counts.get(edge, 0) + 1
                initial_shared_edge_mass[edge] = initial_shared_edge_mass.get(edge, 0.0) + leaf_weight

        for prefix, child_prefixes in parent_children.items():
            full_share_children: list[bool] = []
            for child_prefix in child_prefixes:
                edge = (prefix, child_prefix)
                descendant_count = total_descendant_counts.get(edge, 0)
                shared_reachable_count = shared_reachable_counts.get(edge, 0)
                is_full_share_child = (
                    descendant_count > 0 and shared_reachable_count == descendant_count
                )
                blocked_count = max(0, descendant_count - shared_reachable_count)

                self.descendant_leaf_count[edge] = descendant_count
                self.shared_reachable_leaf_count[edge] = shared_reachable_count
                self.full_share_child_edges[edge] = is_full_share_child
                self.unshared_edge_count[edge] = blocked_count
                # Debug-only: descendants that do not expose shared aggregate across
                # this parent-child interface.
                self.unshared_edge_mass[edge] = float(blocked_count)
                aggregate_mass = float(initial_shared_edge_mass.get(edge, 0.0))
                self.shared_edge_mass[edge] = aggregate_mass
                # Compatibility alias: shared_edge_weight mirrors aggregate
                # upload-reachable shared leaf mass, not a normalized edge score.
                self.shared_edge_weight[edge] = aggregate_mass
                full_share_children.append(is_full_share_child)

            self.safe_prefixes[prefix] = bool(full_share_children) and all(full_share_children)
            self.mixed_prefixes[prefix] = (
                bool(full_share_children)
                and any(full_share_children)
                and not all(full_share_children)
            )

        for edge in total_descendant_counts:
            if not self.safe_prefixes.get(edge[0], False):
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
            child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
            child_prefix, conditional_prob, selection_meta = self._sample_child_prefix(
                current_prefix,
                stage_name,
                env,
            )
            edge = (current_prefix, child_prefix)
            is_safe_prefix = self.safe_prefixes.get(current_prefix, False)
            is_mixed_prefix = self.mixed_prefixes.get(current_prefix, False)
            if is_safe_prefix:
                theta_before_update = self._shared_edge_theta(edge)
            else:
                theta_before_update = self.risky_theta.get(edge, 0.0)
            self.last_stage_probs[stage_name] = conditional_prob
            self.last_path_prob *= conditional_prob
            self.last_sampled_edges.append(
                {
                    "prefix": current_prefix,
                    "child_prefix": child_prefix,
                    "prefix_reach_prob": prefix_reach_prob,
                    "epsilon": selection_meta.get("epsilon"),
                    "epsilon_mode": selection_meta.get("epsilon_mode"),
                    "selection_mode": selection_meta.get("selection_mode"),
                    "branch_conditional_prob": selection_meta.get(
                        "branch_conditional_prob",
                        conditional_prob,
                    ),
                    "conditional_prob": conditional_prob,
                    "mixture_conditional_prob": selection_meta.get("mixture_conditional_prob"),
                    "softmax_conditional_prob": selection_meta.get("softmax_conditional_prob"),
                    "uniform_conditional_prob": selection_meta.get("uniform_conditional_prob"),
                    "path_prob_so_far": prefix_reach_prob,
                    "edge_prob": prefix_reach_prob * conditional_prob,
                    "estimated_loss_denominator": selection_meta.get("estimated_loss_denominator"),
                    "estimator_scope": selection_meta.get("estimator_scope"),
                    "arm_count": len(child_prefixes),
                    "theta_before_update": theta_before_update,
                    "is_safe_prefix": is_safe_prefix,
                    "safe_prefix_selected": is_safe_prefix,
                    "is_full_share_parent": is_safe_prefix,
                    "is_mixed_parent": is_mixed_prefix,
                    "mixed_prefix_selected": is_mixed_prefix and not is_safe_prefix,
                    "risky_prefix_selected": not is_safe_prefix,
                    "is_full_share_child": self.full_share_child_edges.get(
                        edge,
                        False,
                    ),
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

        observed_loss = self._observed_loss(episode_result)
        risky_edge_updates = self._apply_risky_edge_updates(
            episode_result,
            observed_loss=observed_loss,
        )
        shared_leaf_updated = False
        shared_safe_suffix_edges_updated: list[dict[str, Any]] = []
        barrier_stop_prefix: list[str] | None = None
        shared_estimated_loss: float | None = None
        shared_delta: float | None = None
        shared_theta_before: float | None = None
        shared_theta_after: float | None = None
        shared_weight_before: float | None = None
        shared_weight_after: float | None = None

        if self.leaf_types.get(leaf) == "shared":
            shared_leaf_updated = True
            shared_estimated_loss = self._shared_leaf_estimated_loss(
                episode_result,
                observed_loss=observed_loss,
            )
            shared_theta_before = self.theta[leaf]
            shared_weight_before = self._shared_leaf_weight(leaf)
            shared_delta = self._apply_shared_leaf_update(leaf, shared_estimated_loss)
            shared_theta_after = self.theta[leaf]
            shared_weight_after = self._shared_leaf_weight(leaf)
            touched_edges, barrier_prefix = self._propagate_barrier_limited_shared_delta(leaf, shared_delta)
            shared_safe_suffix_edges_updated = [
                {"prefix": list(prefix), "child_prefix": list(child_prefix)}
                for prefix, child_prefix in touched_edges
            ]
            barrier_stop_prefix = list(barrier_prefix) if barrier_prefix is not None else None

        self.last_update_info = {
            "update_type": "risky_ps_layered_barrier_theta_loss",
            "leaf_type": self.leaf_types.get(leaf),
            "observed_loss": observed_loss,
            "observed_cost": episode_result.total_cost,
            "eta": self.eta,
            "eta_shared": self.eta_shared,
            "shared_leaf_weight_eta": self.eta_shared,
            "epsilon": self.epsilon,
            "loss_clip": getattr(self, "loss_clip", None),
            "prob_floor": getattr(self, "prob_floor", 0.0),
            "completed_updates": int(getattr(self, "completed_updates", 0)),
            "risky_edges_updated": risky_edge_updates,
            "shared_leaf_updated": shared_leaf_updated,
            "shared_leaf_estimated_loss": shared_estimated_loss,
            "max_shared_estimated_loss": shared_estimated_loss,
            "shared_leaf_theta_before_update": shared_theta_before,
            "shared_leaf_theta_after_update": shared_theta_after,
            "shared_leaf_weight_before_update": shared_weight_before,
            "shared_leaf_weight_after_update": shared_weight_after,
            "shared_delta_t": shared_delta,
            "shared_delta": shared_delta,
            "shared_safe_suffix_edges_updated": shared_safe_suffix_edges_updated,
            "shared_upload_edges_updated": list(shared_safe_suffix_edges_updated),
            "barrier_stop_prefix": barrier_stop_prefix,
            "safe_prefix_selected": any(
                bool(edge.get("safe_prefix_selected", False))
                for edge in self.last_sampled_edges
            ),
            "mixed_prefix_selected": any(
                bool(edge.get("mixed_prefix_selected", False))
                for edge in self.last_sampled_edges
            ),
            "risky_prefix_selected": any(
                bool(edge.get("risky_prefix_selected", False))
                for edge in self.last_sampled_edges
            ),
        }
        self.completed_updates = int(getattr(self, "completed_updates", 0)) + 1

    def _prefixes(self, leaf: LeafKey) -> list[PrefixKey]:
        return [tuple(leaf[:depth]) for depth in range(0, len(leaf) + 1)]

    def _shared_leaf_weight(self, leaf: LeafKey) -> float:
        exponent = self.eta_shared * self.theta[leaf]
        return math.exp(max(-60.0, min(60.0, exponent)))

    def _shared_edge_theta(self, edge: EdgeKey) -> float:
        mass = max(self.shared_edge_mass.get(edge, self.shared_edge_weight.get(edge, 0.0)), 1e-12)
        return math.log(mass) / max(self.eta_shared, 1e-12)

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
        if total_mass <= 0.0:
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
        return exploit_probs

    def _risky_branch_selection_meta(
        self,
        *,
        prefix: PrefixKey,
        child_prefixes: list[PrefixKey],
        is_mixed_prefix: bool,
    ) -> tuple[list[float], dict[str, Any]]:
        exploit_probs = self._risky_child_probs(prefix, child_prefixes)
        epsilon = min(1.0, max(0.0, float(self.epsilon)))
        arm_count = len(child_prefixes)
        uniform_prob = 1.0 / arm_count
        mixture_probs = [
            (1.0 - epsilon) * exploit_prob + epsilon * uniform_prob
            for exploit_prob in exploit_probs
        ]

        if epsilon > 0.0 and self.rng.random() < epsilon:
            epsilon_mode = "U"
            branch_probs = [uniform_prob for _ in child_prefixes]
            selection_mode = "risky_mixed_uniform" if is_mixed_prefix else "risky_uniform"
        else:
            epsilon_mode = "E"
            branch_probs = exploit_probs
            selection_mode = "risky_mixed_exploit" if is_mixed_prefix else "risky_exploit"

        return branch_probs, {
            "epsilon": epsilon,
            "epsilon_mode": epsilon_mode,
            "selection_mode": selection_mode,
            "mixture_probs": mixture_probs,
            "softmax_probs": exploit_probs,
            "uniform_prob": uniform_prob,
            "estimated_loss_denominator": "branch_edge_prob",
            "estimator_scope": "branch_conditioned_multilevel_edge_probability",
        }

    def _mixed_child_probs(
        self,
        prefix: PrefixKey,
        child_prefixes: list[PrefixKey],
    ) -> list[float]:
        return self._risky_child_probs(prefix, child_prefixes)

    def _sample_child_prefix(
        self,
        current_prefix: PrefixKey,
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> tuple[PrefixKey, float, dict[str, Any]]:
        child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
        if not child_prefixes:
            raise RuntimeError(
                "No legal child prefixes found in RiskyPS selection. "
                f"current_prefix={list(current_prefix)} stage_name={stage_name}"
            )

        if self.safe_prefixes.get(current_prefix, False):
            probs = self._safe_child_probs(current_prefix, child_prefixes)
            selected_idx = self._sample_index(probs)
            selected_prob = probs[selected_idx]
            return child_prefixes[selected_idx], selected_prob, {
                "epsilon": None,
                "epsilon_mode": None,
                "selection_mode": "shared_safe",
                "branch_conditional_prob": selected_prob,
                "conditional_prob": selected_prob,
                "mixture_conditional_prob": selected_prob,
                "softmax_conditional_prob": None,
                "uniform_conditional_prob": None,
                "estimated_loss_denominator": None,
                "estimator_scope": "shared_safe_prefix",
            }

        is_mixed_prefix = self.mixed_prefixes.get(current_prefix, False)
        branch_probs, branch_meta = self._risky_branch_selection_meta(
            prefix=current_prefix,
            child_prefixes=child_prefixes,
            is_mixed_prefix=is_mixed_prefix,
        )
        selected_idx = self._sample_index(branch_probs)
        selected_prob = branch_probs[selected_idx]
        return child_prefixes[selected_idx], selected_prob, {
            "epsilon": branch_meta["epsilon"],
            "epsilon_mode": branch_meta["epsilon_mode"],
            "selection_mode": branch_meta["selection_mode"],
            "branch_conditional_prob": selected_prob,
            "conditional_prob": selected_prob,
            "mixture_conditional_prob": branch_meta["mixture_probs"][selected_idx],
            "softmax_conditional_prob": branch_meta["softmax_probs"][selected_idx],
            "uniform_conditional_prob": branch_meta["uniform_prob"],
            "estimated_loss_denominator": branch_meta["estimated_loss_denominator"],
            "estimator_scope": branch_meta["estimator_scope"],
        }

    def _shared_leaf_estimated_loss(
        self,
        episode_result: EpisodeResult,
        *,
        observed_loss: float | None = None,
    ) -> float:
        return self._importance_weighted_loss(
            observed_loss=(
                observed_loss
                if observed_loss is not None
                else self._observed_loss(episode_result)
            ),
            denominator=self.last_path_prob,
        )

    def _observed_loss(self, episode_result: EpisodeResult) -> float:
        return max(0.0, float(episode_result.total_cost))

    def _importance_weighted_loss(self, *, observed_loss: float, denominator: float) -> float:
        floor = max(0.0, float(getattr(self, "prob_floor", 0.0)))
        denom = max(float(denominator), floor, 1e-12)
        estimated_loss = observed_loss / denom
        clip = getattr(self, "loss_clip", None)
        if clip is not None:
            estimated_loss = min(estimated_loss, float(clip))
        return estimated_loss

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
        touched_edges = list(self.upload_edges_by_leaf.get(leaf, []))
        for edge in touched_edges:
            updated_mass = max(1e-12, self.shared_edge_mass.get(edge, 0.0) + delta)
            self.shared_edge_mass[edge] = updated_mass
            self.shared_edge_weight[edge] = updated_mass
        return touched_edges, self.barrier_stop_prefix_by_leaf.get(leaf)

    def _sampled_risky_edge_infos(self) -> list[dict[str, Any]]:
        return [
            edge_info
            for edge_info in self.last_sampled_edges
            if not self.safe_prefixes.get(edge_info["prefix"], False)
        ]

    def _apply_risky_edge_updates(
        self,
        episode_result: EpisodeResult,
        *,
        observed_loss: float | None = None,
    ) -> list[dict[str, Any]]:
        del episode_result
        updated_edges: list[dict[str, Any]] = []
        observed_loss = max(0.0, float(observed_loss or 0.0))
        for edge_info in self._sampled_risky_edge_infos():
            prefix = edge_info["prefix"]
            child_prefix = edge_info["child_prefix"]
            edge = (prefix, child_prefix)
            conditional_prob = float(edge_info["conditional_prob"])
            prefix_reach_prob = float(
                edge_info.get("prefix_reach_prob", edge_info.get("path_prob_so_far", 1.0))
            )
            edge_prob = float(edge_info.get("edge_prob", prefix_reach_prob * conditional_prob))
            estimated_loss = self._importance_weighted_loss(
                observed_loss=observed_loss,
                denominator=edge_prob,
            )
            old_theta = self.risky_theta.get(edge, 0.0)
            new_theta = old_theta - estimated_loss
            self.risky_theta[edge] = new_theta
            edge_info["estimated_loss"] = estimated_loss
            edge_info["estimated_loss_denominator"] = edge_info.get(
                "estimated_loss_denominator",
                "branch_edge_prob",
            )
            edge_info["estimator_scope"] = edge_info.get(
                "estimator_scope",
                "branch_conditioned_multilevel_edge_probability",
            )
            edge_info["observed_loss"] = observed_loss
            edge_info["theta_before_update"] = old_theta
            edge_info["theta_after_update"] = new_theta
            updated_edges.append(
                {
                    "prefix": list(prefix),
                    "child_prefix": list(child_prefix),
                    "prefix_reach_prob": prefix_reach_prob,
                    "path_prob_so_far": edge_info.get("path_prob_so_far", prefix_reach_prob),
                    "conditional_prob": conditional_prob,
                    "branch_conditional_prob": edge_info.get(
                        "branch_conditional_prob",
                        conditional_prob,
                    ),
                    "mixture_conditional_prob": edge_info.get("mixture_conditional_prob"),
                    "epsilon": edge_info.get("epsilon", self.epsilon),
                    "epsilon_mode": edge_info.get("epsilon_mode"),
                    "selection_mode": edge_info.get("selection_mode"),
                    "edge_prob": edge_prob,
                    "estimated_loss_denominator": edge_info["estimated_loss_denominator"],
                    "estimator_scope": edge_info["estimator_scope"],
                    "arm_count": int(edge_info.get("arm_count", 1)),
                    "observed_loss": observed_loss,
                    "estimated_loss": estimated_loss,
                    "theta_before_update": old_theta,
                    "theta_after_update": new_theta,
                    "old_risky_theta": old_theta,
                    "new_risky_theta": new_theta,
                    "update_type": "risky_ps_risky_edge_theta_loss",
                }
            )
        return updated_edges

    def get_state(self) -> dict[str, Any]:
        shared_edge_masses = list(self.shared_edge_mass.values())
        shared_mass_total = sum(shared_edge_masses)
        return {
            "protocol_mode": self.protocol_mode,
            "eta": self.eta,
            "eta_shared": self.eta_shared,
            "shared_leaf_weight_eta": self.eta_shared,
            "epsilon": self.epsilon,
            "update_type": "risky_ps_layered_barrier_theta_loss",
            "loss_clip": getattr(self, "loss_clip", None),
            "prob_floor": getattr(self, "prob_floor", 0.0),
            "completed_updates": int(getattr(self, "completed_updates", 0)),
            "num_paths": len(self.paths),
            "num_full_share_parents": sum(1 for is_safe in self.safe_prefixes.values() if is_safe),
            "num_mixed_parents": sum(1 for is_mixed in self.mixed_prefixes.values() if is_mixed),
            "num_risky_prefixes": sum(1 for is_safe in self.safe_prefixes.values() if not is_safe),
            "num_full_share_child_edges": sum(
                1 for is_full_share in self.full_share_child_edges.values() if is_full_share
            ),
            "num_exposed_shared_edges": sum(1 for value in self.shared_edge_mass.values() if value > 0),
            "num_risky_theta_edges": len(self.risky_theta),
            "shared_edge_mass_total": round(shared_mass_total, 6),
            "shared_mass_total": round(shared_mass_total, 6),
            "shared_edge_weight_total": round(shared_mass_total, 6),
            "min_shared_edge_mass": min(shared_edge_masses, default=0.0),
            "max_shared_edge_mass": max(shared_edge_masses, default=0.0),
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
                    "path_prob_so_far": edge.get("path_prob_so_far", edge["prefix_reach_prob"]),
                    "epsilon": edge.get("epsilon"),
                    "epsilon_mode": edge.get("epsilon_mode"),
                    "selection_mode": edge.get("selection_mode"),
                    "branch_conditional_prob": edge.get(
                        "branch_conditional_prob",
                        edge["conditional_prob"],
                    ),
                    "conditional_prob": edge["conditional_prob"],
                    "mixture_conditional_prob": edge.get("mixture_conditional_prob"),
                    "edge_prob": edge["edge_prob"],
                    "estimated_loss_denominator": edge.get(
                        "estimated_loss_denominator",
                        "branch_edge_prob",
                    ),
                    "estimator_scope": edge.get(
                        "estimator_scope",
                        "branch_conditioned_multilevel_edge_probability",
                    ),
                    "arm_count": edge.get("arm_count"),
                    "theta_before_update": edge.get("theta_before_update"),
                    "theta_after_update": edge.get("theta_after_update"),
                    "estimated_loss": edge.get("estimated_loss"),
                    "observed_loss": edge.get("observed_loss"),
                    "is_safe_prefix": edge.get("is_safe_prefix", False),
                    "safe_prefix_selected": edge.get("safe_prefix_selected", False),
                    "is_full_share_parent": edge.get("is_full_share_parent", False),
                    "is_mixed_parent": edge.get("is_mixed_parent", False),
                    "mixed_prefix_selected": edge.get("mixed_prefix_selected", False),
                    "risky_prefix_selected": edge.get("risky_prefix_selected", False),
                    "is_full_share_child": edge.get("is_full_share_child", False),
                }
                for edge in self.last_sampled_edges
            ],
            "update_type": "risky_ps_layered_barrier_theta_loss",
            "eta": self.eta,
            "eta_shared": self.eta_shared,
            "shared_leaf_weight_eta": self.eta_shared,
            "epsilon": self.epsilon,
            "last_update_info": dict(self.last_update_info),
        }
