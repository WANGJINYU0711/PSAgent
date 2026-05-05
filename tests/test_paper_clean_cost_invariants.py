from __future__ import annotations

import math
import os
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "envs"))

from envs.fixed_tree_env import FixedTreeEnvironment, default_agent_catalog
from envs.tree_family.generator import TreeFamilyGenerator


class PaperCleanCostInvariantTest(unittest.TestCase):
    def test_family_agents_do_not_expose_attribute_profiles(self) -> None:
        generator = TreeFamilyGenerator()
        _, agent_map = generator.build_family(
            "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v4_binary_mixed_stage45_4of5",
            seed=0,
        )

        self.assertTrue(agent_map)
        self.assertTrue(all(not hasattr(agent, "attribute_skill") for agent in agent_map.values()))

    def test_raw_total_cost_excludes_path_and_mode_mismatch_cost(self) -> None:
        old_env = {
            key: os.environ.get(key)
            for key in (
                "PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4",
                "PSAGENT_TELECOM_MODE_MISMATCH_COST_V2",
                "PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2",
            )
        }
        try:
            os.environ["PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4"] = "1"
            os.environ["PSAGENT_TELECOM_MODE_MISMATCH_COST_V2"] = "1"
            os.environ["PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2"] = "1"

            env = FixedTreeEnvironment(agent_catalog=default_agent_catalog())
            env.current_instance = {"family": "telecom_mms_recovery"}
            metrics = env._build_cost_metrics(
                evaluator_result={
                    "raw_terminal_penalty": 3.25,
                    "exact_match": True,
                    "subset_mismatch": False,
                    "policy_violation_count": 0,
                    "oracle_final_action": "repair_all",
                    "predicted_final_action": "repair_all",
                },
                path_agent_cost=999.0,
                reasoning_metrics={
                    "raw_reasoning_cost_component": 2.5,
                    "raw_reasoning_cost_component_api": 1.25,
                    "raw_reasoning_cost_component_token": 2.5,
                    "reasoning_cost_mode_default": "token",
                    "trace": [
                        {
                            "deliberation_mode": "fast",
                            "deliberation_requirement": "deep",
                        }
                    ],
                },
            )

            self.assertNotIn("raw_path_cost_component", metrics)
            self.assertNotIn("raw_mode_mismatch_cost_component", metrics)
            self.assertTrue(math.isclose(metrics["raw_total_cost"], 3.25 + 2.5))
            self.assertTrue(math.isclose(metrics["raw_total_cost_api"], 3.25 + 1.25))
            self.assertTrue(math.isclose(metrics["raw_total_cost_token"], 3.25 + 2.5))
        finally:
            for key, value in old_env.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value


if __name__ == "__main__":
    unittest.main()
