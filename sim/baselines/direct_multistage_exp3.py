"""Strict tree-local stagewise Exp3 without explicit exploration."""

from __future__ import annotations

from base import StagewiseExp3Policy


class DirectMultiStageExp3Policy(StagewiseExp3Policy):
    """Prefix-local stagewise Exp3 over child interfaces.

    This baseline does not share theta across different parent prefixes:
    ``(prefix=C3, child=D2)`` and ``(prefix=C4, child=D2)`` are distinct arms
    even when the underlying child agent id is reused by the family.
    Selection uses ``softmax(eta * theta)`` under the current prefix and has no
    explicit epsilon exploration.
    """

    def __init__(self, seed: int = 0, eta: float = 0.2) -> None:
        super().__init__(
            seed=seed,
            protocol_mode="actual_leaf",
            eta=eta,
            epsilon=0.0,
            estimator_type="loss",
            update_type="direct_stagewise_exp3_theta_loss",
        )

    @property
    def name(self) -> str:
        return "direct_multistage_exp3"

    def preferred_catalog_preset(self) -> str:
        return "mixed"
