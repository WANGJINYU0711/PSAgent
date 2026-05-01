"""Structure-agnostic shared-init variants for PS-family policies.

These variants keep the original sampling and update rules intact and only
replace the initial shared edge aggregate with a structure-agnostic prior.  The
goal is to remove the subtree-size prior from the safe shared suffix while
preserving the same downstream update mechanics.
"""

from __future__ import annotations

import random
from typing import Any

from fixed_tree_env import EpisodeResult, FixedTreeEnvironment
from risky_ps import EdgeKey, RiskyPSPolicy
from risky_ps_ix import RiskyPSIXPolicy
from risky_ps_linear import RiskyPSLinearPolicy
from risky_ps_old import RiskyPSOldPolicy


class _ConstantSharedInitMixin:
    """Overwrite shared edge aggregates with a constant non-structural prior."""

    def __init__(
        self,
        *args: Any,
        shared_edge_init: float = 1.0,
        shared_edge_floor: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        self.shared_edge_init = float(shared_edge_init)
        self.shared_edge_floor = max(0.0, float(shared_edge_floor))
        super().__init__(*args, **kwargs)

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self._apply_constant_shared_edge_init()

    def update(self, episode_result: EpisodeResult) -> None:
        super().update(episode_result)
        if hasattr(self, "last_update_info") and isinstance(self.last_update_info, dict):
            self.last_update_info["shared_init_mode"] = "constant_edge_mass"
            self.last_update_info["shared_edge_init"] = self._constant_shared_edge_mass()

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["shared_init_mode"] = "constant_edge_mass"
        state["shared_edge_init"] = self._constant_shared_edge_mass()
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info["shared_init_mode"] = "constant_edge_mass"
        info["shared_edge_init"] = self._constant_shared_edge_mass()
        return info

    def _constant_shared_edge_mass(self) -> float:
        return max(self.shared_edge_floor, self.shared_edge_init)

    def _apply_constant_shared_edge_init(self) -> None:
        constant_mass = self._constant_shared_edge_mass()
        for edge in list(self.shared_edge_mass):
            if self.shared_reachable_leaf_count.get(edge, 0) > 0:
                updated_mass = constant_mass
            else:
                updated_mass = 0.0
            self.shared_edge_mass[edge] = updated_mass
            self.shared_edge_weight[edge] = updated_mass


class RiskyPSConstInitPolicy(_ConstantSharedInitMixin, RiskyPSPolicy):
    @property
    def name(self) -> str:
        return "risky_ps_const_init"


class RiskyPSOldConstInitPolicy(_ConstantSharedInitMixin, RiskyPSOldPolicy):
    @property
    def name(self) -> str:
        return "risky_ps_old_const_init"


class RiskyPSOldFixedInitPolicy(RiskyPSOldConstInitPolicy):
    """Alias for the exact old Risky-PS algorithm with fixed W=1 shared init."""

    @property
    def name(self) -> str:
        return "risky_ps_old_fixed_init"


class _RandomSharedInitMixin:
    """Overwrite shared edge aggregates with deterministic random edge priors."""

    def __init__(
        self,
        *args: Any,
        random_edge_init_low: float = 0.5,
        random_edge_init_high: float = 1.5,
        random_edge_floor: float = 1e-12,
        **kwargs: Any,
    ) -> None:
        self.random_edge_init_low = float(random_edge_init_low)
        self.random_edge_init_high = float(random_edge_init_high)
        self.random_edge_floor = max(0.0, float(random_edge_floor))
        if self.random_edge_init_high < self.random_edge_init_low:
            raise ValueError("random_edge_init_high must be >= random_edge_init_low")
        super().__init__(*args, **kwargs)

    def bind_env(self, env: FixedTreeEnvironment) -> None:
        super().bind_env(env)
        self._apply_random_shared_edge_init()

    def update(self, episode_result: EpisodeResult) -> None:
        super().update(episode_result)
        if hasattr(self, "last_update_info") and isinstance(self.last_update_info, dict):
            self.last_update_info["shared_init_mode"] = "random_edge_mass_uniform"
            self.last_update_info["random_edge_init_low"] = self.random_edge_init_low
            self.last_update_info["random_edge_init_high"] = self.random_edge_init_high

    def get_state(self) -> dict[str, Any]:
        state = super().get_state()
        state["shared_init_mode"] = "random_edge_mass_uniform"
        state["random_edge_init_low"] = self.random_edge_init_low
        state["random_edge_init_high"] = self.random_edge_init_high
        return state

    def get_last_selection_info(self) -> dict[str, Any]:
        info = super().get_last_selection_info()
        info["shared_init_mode"] = "random_edge_mass_uniform"
        info["random_edge_init_low"] = self.random_edge_init_low
        info["random_edge_init_high"] = self.random_edge_init_high
        return info

    def _apply_random_shared_edge_init(self) -> None:
        init_rng = random.Random(int(getattr(self, "seed", 0)) + 9_172_663)
        span = self.random_edge_init_high - self.random_edge_init_low
        for edge in sorted(self.shared_edge_mass):
            if self.shared_reachable_leaf_count.get(edge, 0) > 0:
                updated_mass = self.random_edge_init_low + span * init_rng.random()
                updated_mass = max(self.random_edge_floor, updated_mass)
            else:
                updated_mass = 0.0
            self.shared_edge_mass[edge] = updated_mass
            self.shared_edge_weight[edge] = updated_mass


class RiskyPSOldRandomInitPolicy(_RandomSharedInitMixin, RiskyPSOldPolicy):
    """Exact old Risky-PS algorithm with random structure-agnostic shared W init."""

    @property
    def name(self) -> str:
        return "risky_ps_old_random_init"


class RiskyPSIXConstInitPolicy(_ConstantSharedInitMixin, RiskyPSIXPolicy):
    @property
    def name(self) -> str:
        return "risky_ps_ix_const_init"


class RiskyPSLinearConstInitPolicy(_ConstantSharedInitMixin, RiskyPSLinearPolicy):
    def __init__(
        self,
        *args: Any,
        shared_edge_init: float | None = None,
        shared_edge_floor: float | None = None,
        **kwargs: Any,
    ) -> None:
        shared_leaf_init = float(kwargs.get("shared_leaf_init", 1.0))
        shared_leaf_floor = float(kwargs.get("shared_leaf_floor", 1e-12))
        super().__init__(
            *args,
            shared_edge_init=(
                shared_leaf_init if shared_edge_init is None else float(shared_edge_init)
            ),
            shared_edge_floor=(
                shared_leaf_floor if shared_edge_floor is None else float(shared_edge_floor)
            ),
            **kwargs,
        )

    @property
    def name(self) -> str:
        return "risky_ps_linear_const_init"
