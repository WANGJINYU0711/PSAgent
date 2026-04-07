"""Path-centric epsilon-EXP3 style baseline for the fixed-tree environment."""

from __future__ import annotations

from base import PathCentricUnsharedPolicy


class EpsilonExp3Policy(PathCentricUnsharedPolicy):
    """Path/prefix-centric unshared baseline with explicit uniform mixing."""

    def __init__(self, seed: int = 0, eta: float = 0.2, epsilon: float = 0.1) -> None:
        super().__init__(
            seed=seed,
            protocol_mode="force_unshared",
            eta=eta,
            epsilon=epsilon,
            update_type="epsilon_exp3_path_uniform_mixing",
        )

    @property
    def name(self) -> str:
        return "epsilon_exp3"

    def preferred_catalog_preset(self) -> str:
        return "all_unshare"
