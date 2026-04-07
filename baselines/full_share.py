"""All-share endpoint baseline with leaf-centric subtree aggregation.

This version treats each complete root-to-leaf path as a leaf arm. It maintains:

- ``theta[leaf]``: leaf-level score
- ``leaf_weights[leaf]``: current leaf weight
- ``prefix_weights[prefix]``: aggregate subtree weight for every prefix

Selection is implemented by explicit prefix/subtree sampling. Starting from the
root prefix, the policy repeatedly compares the aggregate weights of the legal
child prefixes and samples the next child until a complete leaf is reached. The
update follows the shared endpoint mechanics:

1. update the selected leaf score
2. recompute the selected leaf weight
3. compute ``delta = new_w - old_w``
4. add that delta to every ancestor prefix aggregate
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
        for stage_name in env.STAGE_NAMES:
            current_prefix = self._sample_child_prefix(current_prefix, stage_name, env)
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

        observed_cost = episode_result.total_cost
        old_weight = self.leaf_weights[leaf]

        # Shared endpoint update: the leaf score moves by the observed shared cost,
        # then the induced weight delta is pushed to every ancestor prefix.
        self.theta[leaf] = self.theta[leaf] - self.eta * observed_cost
        new_weight = math.exp(self.theta[leaf])
        self.leaf_weights[leaf] = new_weight

        delta = new_weight - old_weight
        for prefix in self._prefixes(leaf):
            self.prefix_weights[prefix] = self.prefix_weights.get(prefix, 0.0) + delta

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
    ) -> PrefixKey:
        child_prefixes = self._child_prefixes(current_prefix, stage_name, env)
        if not child_prefixes:
            raise RuntimeError(
                "No legal child prefixes found during subtree sampling. "
                f"current_prefix={list(current_prefix)} stage_name={stage_name}"
            )

        weights = [
            max(0.0, self.prefix_weights.get(child_prefix, 0.0))
            for child_prefix in child_prefixes
        ]
        if sum(weights) <= 0:
            return child_prefixes[self.rng.randrange(len(child_prefixes))]

        selected_idx = self._sample_index(weights)
        return child_prefixes[selected_idx]

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
