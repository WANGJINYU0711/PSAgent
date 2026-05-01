"""Local-prefix Exp3 variant of direct multistage Exp3."""

from __future__ import annotations

from direct_multistage_exp3 import DirectMultiStageExp3Policy
from fixed_tree_env import EpisodeResult


class DirectMultiStageExp3LocalPolicy(DirectMultiStageExp3Policy):
    """Direct stagewise Exp3 with per-prefix local importance weighting only."""

    def __init__(self, seed: int = 0, eta: float = 0.2) -> None:
        super().__init__(seed=seed, eta=eta)
        self.update_type = "direct_stagewise_exp3_local_theta_loss"

    @property
    def name(self) -> str:
        return "direct_multistage_exp3_local"

    def update(self, episode_result: EpisodeResult) -> None:
        if len(self.last_selected_edges) != len(episode_result.selected_path):
            raise RuntimeError(
                "DirectMultiStageExp3LocalPolicy update called without matching select_path metadata. "
                f"selected_path_len={len(episode_result.selected_path)} "
                f"last_selected_edges={len(self.last_selected_edges)}"
            )

        observed_loss = max(0.0, float(episode_result.total_cost))
        edge_updates: list[dict[str, object]] = []
        for selected_edge in self.last_selected_edges:
            stage_name = str(selected_edge["stage_name"])
            current_prefix = tuple(selected_edge["prefix"])
            child_prefix = tuple(selected_edge["child_prefix"])
            conditional_prob = float(selected_edge["conditional_prob"])
            prefix_reach_prob = float(
                selected_edge.get(
                    "prefix_reach_prob",
                    selected_edge.get("path_prob_so_far", 1.0),
                )
            )
            edge_prob = float(
                selected_edge.get("edge_prob", prefix_reach_prob * conditional_prob)
            )
            edge_key = self._edge_key(current_prefix, child_prefix)
            estimated_loss = observed_loss / max(conditional_prob, 1e-12)
            theta_before = self.theta.setdefault(edge_key, 0.0)
            theta_after = theta_before - estimated_loss
            self.theta[edge_key] = theta_after
            self.weights[edge_key] = self._edge_weight(current_prefix, child_prefix)
            self.last_stage_probs[stage_name] = conditional_prob
            selected_edge["estimated_loss"] = estimated_loss
            selected_edge["observed_loss"] = observed_loss
            selected_edge["theta_before_update"] = theta_before
            selected_edge["theta_after_update"] = theta_after
            selected_edge["estimated_loss_denominator"] = "local_branch_prob"
            selected_edge["estimator_scope"] = "local_prefix_bandit_probability"
            edge_updates.append(
                {
                    "stage_name": stage_name,
                    "prefix": list(current_prefix),
                    "child_prefix": list(child_prefix),
                    "prefix_reach_prob": prefix_reach_prob,
                    "conditional_prob": conditional_prob,
                    "branch_conditional_prob": float(
                        selected_edge.get("branch_conditional_prob", conditional_prob)
                    ),
                    "mixture_conditional_prob": selected_edge.get("mixture_conditional_prob"),
                    "epsilon": selected_edge.get("epsilon", self.epsilon),
                    "epsilon_mode": selected_edge.get("epsilon_mode"),
                    "selection_mode": selected_edge.get("selection_mode"),
                    "path_prob_so_far": prefix_reach_prob,
                    "edge_prob": edge_prob,
                    "estimated_loss_denominator": "local_branch_prob",
                    "estimator_scope": "local_prefix_bandit_probability",
                    "arm_count": int(selected_edge["arm_count"]),
                    "observed_loss": observed_loss,
                    "estimated_loss": estimated_loss,
                    "theta_before_update": theta_before,
                    "theta_after_update": theta_after,
                    "update_type": self.update_type,
                }
            )

        self.last_update_info = {
            "update_type": self.update_type,
            "eta": self.eta,
            "epsilon": self.epsilon,
            "observed_loss": observed_loss,
            "edge_updates": edge_updates,
        }
