"""All-unshare endpoint baseline with path-centric bandit updates."""

from __future__ import annotations

from base import PathCentricUnsharedPolicy


class FullUnsharePolicy(PathCentricUnsharedPolicy):
    """The all-unshare endpoint of the fixed-tree / partial-share family."""

    def __init__(self, seed: int = 0, eta: float = 0.2) -> None:
        super().__init__(
            seed=seed,
            protocol_mode="force_unshared",
            eta=eta,
            epsilon=0.0,
            update_type="all_unshare_endpoint",
        )

    @property
    def name(self) -> str:
        return "full_unshare"

    def preferred_catalog_preset(self) -> str:
        return "all_unshare"
