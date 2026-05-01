"""Full-Share baseline with recursive subtree aggregation.

This baseline keeps the Full-Share program state explicitly at the leaf level
and derives every internal-node quantity from subtree aggregates:

- ``theta[leaf]``: leaf score
- ``leaf_weights[leaf]``: ``exp(eta * theta[leaf])``
- ``prefix_weights[prefix]``: subtree aggregate rooted at ``prefix``

Selection uses only the subtree aggregates, mirroring Algorithm 2 in
``notes/PartialShare.md``: at each internal node we sample a child in
proportion to that child subtree's aggregate mass. Update is a single-leaf
change followed by ancestor delta propagation:

1. estimate the sampled leaf loss
2. update ``theta[leaf]``
3. recompute ``leaf_weights[leaf]``
4. propagate ``delta = new_weight - old_weight`` to every ancestor prefix

Under the current ``force_shared`` feedback protocol we still treat the sampled
leaf as the shared endpoint, but the uploaded loss is now the standard
leaf-bandit importance-weighted estimator ``observed_cost / path_prob``. This
keeps the recursive aggregation structure unchanged while making the leaf update
closer to the estimator semantics implied by ``notes/PartialShare.md``.
"""

from __future__ import annotations

import math
from typing import Any

from base import BasePolicy
from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from oracle_eval import enumerate_all_paths


LeafKey = tuple[str, ...]
PrefixKey = tuple[str, ...]


class FullSharePolicy(BasePolicy):
    def __init__(self, seed: int = 0, eta: float = 1.0) -> None:
        super().__init__(seed=seed, protocol_mode="force_shared")
        self.eta = eta
        self.paths: list[list[str]] = []
        self.theta: dict[LeafKey, float] = {}
        self.leaf_weights: dict[LeafKey, float] = {}
        self.prefix_weights: dict[PrefixKey, float] = {}
        self.last_stage_probs: dict[str, float] = {}
        self.last_path_prob: float = 0.0
        self.last_estimated_loss: float | None = None

    @property
    def name(self) -> str:
        return "full_share"

    def preferred_catalog_preset(self) -> str:
        return "all_share"

    def reset(self) -> None:
        """Keep learned shared state across episodes in one run."""

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self.paths = enumerate_all_paths(env)
        self.theta = {}
        self.leaf_weights = {}
        self.prefix_weights = {}
        self.last_stage_probs = {}
        self.last_path_prob = 0.0
        self.last_estimated_loss = None

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

        current_prefix: PrefixKey = ()
        self.last_stage_probs = {}
        self.last_path_prob = 1.0
        for stage_name in env.STAGE_NAMES:
            current_prefix, conditional_prob = self._sample_child_prefix(current_prefix, stage_name, env)
            self.last_stage_probs[stage_name] = conditional_prob
            self.last_path_prob *= conditional_prob
        if len(current_prefix) != len(env.STAGE_NAMES):
            raise RuntimeError(
                "FullSharePolicy failed to sample a complete leaf path. "
                f"Got prefix of length {len(current_prefix)}."
            )
        return list(current_prefix)

    def update(self, episode_result: EpisodeResult) -> None:
        leaf = tuple(episode_result.selected_path)
        if leaf not in self.theta or leaf not in self.leaf_weights:
            raise KeyError(f"Selected path is unknown to FullSharePolicy: {leaf}")

        estimated_loss = self._leaf_estimated_loss(episode_result)
        self.last_estimated_loss = estimated_loss
        delta = self._apply_leaf_update(leaf, estimated_loss)
        self._propagate_delta_to_prefixes(leaf, delta)

    def _prefixes(self, leaf: LeafKey) -> list[PrefixKey]:
        return [tuple(leaf[:depth]) for depth in range(0, len(leaf) + 1)]

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

    def _sample_child_prefix(
        self,
        current_prefix: PrefixKey,
        stage_name: str,
        env: FixedTreeEnvironment,
    ) -> tuple[PrefixKey, float]:
        child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
        if not child_prefixes:
            raise RuntimeError(
                "No legal child prefixes found during subtree sampling. "
                f"current_prefix={list(current_prefix)} stage_name={stage_name}"
            )

        child_weights = [
            max(0.0, self.prefix_weights.get(child_prefix, 0.0))
            for child_prefix in child_prefixes
        ]
        if sum(child_weights) <= 0:
            selected_idx = self.rng.randrange(len(child_prefixes))
            return child_prefixes[selected_idx], 1.0 / len(child_prefixes)

        # Recursive aggregation: each child competes only through its subtree mass.
        selected_idx = self._sample_index(child_weights)
        total_weight = sum(child_weights)
        conditional_prob = child_weights[selected_idx] / total_weight
        return child_prefixes[selected_idx], conditional_prob

    def _leaf_estimated_loss(self, episode_result: EpisodeResult) -> float:
        """Return the sampled leaf's importance-weighted loss estimate.

        This is the standard leaf-bandit first-pass estimator:

        ``estimated_loss = observed_cost / sampled_leaf_probability``

        It is unbiased for the sampled leaf loss under the recursive sampling
        distribution induced by subtree aggregation.
        """

        return episode_result.total_cost / max(self.last_path_prob, 1e-12)

    def _apply_leaf_update(self, leaf: LeafKey, estimated_loss: float) -> float:
        """Update a single leaf and return its induced weight delta."""

        old_weight = self.leaf_weights[leaf]
        self.theta[leaf] = self.theta[leaf] - estimated_loss
        new_weight = math.exp(self.eta * self.theta[leaf])
        self.leaf_weights[leaf] = new_weight
        return new_weight - old_weight

    def _propagate_delta_to_prefixes(self, leaf: LeafKey, delta: float) -> None:
        """Push a leaf weight change to every ancestor subtree aggregate."""

        for prefix in self._prefixes(leaf):
            self.prefix_weights[prefix] = self.prefix_weights.get(prefix, 0.0) + delta

    def get_state(self) -> dict[str, Any]:
        top_leaves = sorted(
            self.leaf_weights.items(),
            key=lambda item: item[1],
            reverse=True,
        )[:5]
        return {
            "protocol_mode": self.protocol_mode,
            "eta": self.eta,
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
            "estimated_loss": self.last_estimated_loss,
            "update_type": "full_share_iw_leaf_v2",
        }
