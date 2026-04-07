"""Stagewise EXP3 baseline without risky/safe structure awareness."""

from __future__ import annotations

from base import StagewiseExp3Policy


class DirectMultiStageExp3Policy(StagewiseExp3Policy):
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
