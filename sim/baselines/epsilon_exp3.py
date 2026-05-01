"""Branch-conditioned stagewise epsilon-EXP3 baseline."""

from __future__ import annotations

from base import StagewiseExp3Policy


class EpsilonExp3Policy(StagewiseExp3Policy):
    """Direct stagewise Exp3 plus branch-conditioned epsilon exploration.

    This policy uses the same local ``theta[(prefix, child_prefix)]`` update as
    ``direct_multistage_exp3``. The only algorithmic difference is that each
    prefix first samples a mode: ``U`` with probability epsilon, then uniform
    over legal direct children, or ``E`` otherwise, then softmax over
    ``eta * theta``. The update denominator is the current branch probability
    ``prefix_reach_prob * branch_conditional_prob``; the marginal mixture
    probability is logged only for diagnostics.
    """

    def __init__(self, seed: int = 0, eta: float = 0.2, epsilon: float = 0.1) -> None:
        super().__init__(
            seed=seed,
            protocol_mode="actual_leaf",
            eta=eta,
            epsilon=epsilon,
            estimator_type="loss",
            update_type="epsilon_stagewise_exp3_theta_loss",
        )

    @property
    def name(self) -> str:
        return "epsilon_exp3"

    def preferred_catalog_preset(self) -> str:
        return "mixed"
