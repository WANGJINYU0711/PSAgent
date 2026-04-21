"""Strict tree-local stagewise bandit without explicit exploration."""

from __future__ import annotations

from base import StagewiseExp3Policy


class DirectMultiStageExp3Policy(StagewiseExp3Policy):
    """Prefix-local stagewise multiplicative bandit over child interfaces.

    This baseline does not share weights across different parent prefixes:
    ``(prefix=C3, child=D2)`` and ``(prefix=C4, child=D2)`` are distinct arms
    even when the underlying child agent id is reused by the family.
    Selection normalizes only the local child weights under the current prefix.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__(
            seed=seed,
            protocol_mode="actual_leaf",
            gamma=0.2,
            epsilon=0.0,
            estimator_type="loss",
            update_type="direct_stagewise_exp3_loss",
        )

    @property
    def name(self) -> str:
        return "direct_multistage_exp3"

    def preferred_catalog_preset(self) -> str:
        return "all_unshare"
