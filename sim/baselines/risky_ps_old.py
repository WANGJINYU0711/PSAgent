"""Pre-eta_shared-split Risky-PS baseline.

This variant preserves the older shared-update temperature behavior where
shared leaf weights use the same eta as local risky/mixed epsilon-EXP3 edges.
It is kept as a separate registration target for controlled-simulation
comparisons and does not modify the current ``risky_ps`` default.
"""

from __future__ import annotations

from risky_ps import RiskyPSPolicy


class RiskyPSOldPolicy(RiskyPSPolicy):
    """Risky-PS with shared eta tied to the risky eta."""

    def __init__(
        self,
        seed: int = 0,
        eta: float = 0.2,
        epsilon: float = 0.1,
        loss_clip: float | None = None,
        prob_floor: float = 0.0,
    ) -> None:
        super().__init__(
            seed=seed,
            eta=eta,
            eta_shared=eta,
            epsilon=epsilon,
            loss_clip=loss_clip,
            prob_floor=prob_floor,
        )

    @property
    def name(self) -> str:
        return "risky_ps_old"
