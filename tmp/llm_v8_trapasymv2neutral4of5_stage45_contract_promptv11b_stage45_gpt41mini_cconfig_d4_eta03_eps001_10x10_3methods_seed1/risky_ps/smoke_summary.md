{
  "summary": {
    "test_name": "risky_ps_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json",
    "dataset_indices": [
      1,
      2,
      3,
      6,
      9,
      10,
      13,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      32,
      33,
      34,
      35,
      36
    ],
    "repeats": 10,
    "episodes": 100,
    "method": "risky_ps",
    "mechanism": "algorithm_direct",
    "executor_name": "llm_bench",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n3__from__root__c03",
      "stage2_n1__from__n0003__c01",
      "stage3_n2__from__n0011__c02",
      "stage4_n3__from__n0034__c03",
      "stage5_n4__from__n0103__c04"
    ],
    "exact_match_mean": 0.79,
    "terminal_penalty_mean": 3.36,
    "raw_outcome_penalty_mean": 1.18,
    "raw_policy_penalty_mean": 0.56,
    "raw_terminal_penalty_mean": 3.36,
    "legacy_raw_terminal_penalty_mean": 1.74,
    "raw_terminal_penalty_exec_clean_v4_mean": 3.36,
    "total_cost_mean": 0.23820139203015617,
    "raw_total_cost_mean": 8.8467997,
    "raw_total_cost_api_mean": 4.7242836399999995,
    "raw_total_cost_token_mean": 8.8467997,
    "reasoning_cost_mean": 5.4203297,
    "raw_reasoning_cost_component_mean": 5.4203297,
    "raw_mode_mismatch_cost_component_mean": 2.16,
    "raw_reasoning_cost_component_api_mean": 1.29781364,
    "raw_reasoning_cost_component_token_mean": 5.4203297,
    "raw_path_cost_component_mean": 0.06647,
    "algorithm_cumulative_total_cost": 23.820139203015618,
    "raw_algorithm_cumulative_total_cost": 884.67997,
    "oracle_stationary_total_cost": 20.81367285945073,
    "raw_oracle_stationary_total_cost": 773.0198100000001,
    "raw_outcome_penalty_cumulative": 118.0,
    "raw_policy_penalty_cumulative": 56.0,
    "raw_terminal_penalty_cumulative": 336.0,
    "legacy_raw_terminal_penalty_cumulative": 174.0,
    "raw_path_cost_component_cumulative": 6.647,
    "raw_reasoning_cost_component_cumulative": 542.03297,
    "raw_mode_mismatch_cost_component_cumulative": 216.0,
    "mean_llm_call_count": 12.55,
    "mean_prompt_tokens": 52682.98,
    "mean_completion_tokens": 1176.53,
    "mean_total_tokens": 53859.51,
    "cumulative_total_tokens": 5385951.0,
    "mean_api_cost_usd_raw": 0.013798793,
    "cumulative_api_cost_usd_raw": 1.3798793,
    "mean_generation_time_seconds": 63.54969016464427,
    "p50_generation_time_seconds": 46.20884278137237,
    "p90_generation_time_seconds": 114.98965560216459,
    "mean_llm_round_trip_seconds": 63.5903641789034,
    "mean_episode_wall_clock_seconds": 66.19409837268293,
    "p50_episode_wall_clock_seconds": 48.97803922556341,
    "p90_episode_wall_clock_seconds": 117.49235558547082,
    "mean_tool_wall_clock_seconds": 0.011438904702663422,
    "policy_action_violation_rate": 0.28,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.28,
    "subset_mismatch_count": 21,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.86,
    "unshared_path_fraction": 0.14,
    "mean_barrier_stop_depth": 3.024390243902439,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 45,
        "fraction": 0.45
      },
      "stage1": {
        "count": 16,
        "fraction": 0.16
      },
      "stage2": {
        "count": 9,
        "fraction": 0.09
      },
      "stage3": {
        "count": 17,
        "fraction": 0.17
      },
      "stage4": {
        "count": 7,
        "fraction": 0.07
      },
      "stage5": {
        "count": 6,
        "fraction": 0.06
      }
    },
    "mean_candidate_count_per_stage": [
      4.0,
      2.74,
      2.87,
      2.93,
      2.73
    ],
    "mean_legal_child_count_per_stage": [
      4.0,
      2.74,
      2.87,
      2.93,
      2.73
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.125,
    "stage_source_summary": {
      "stage1": {
        "llm_bench": 100
      },
      "stage2": {
        "llm_bench": 100
      },
      "stage3": {
        "llm_bench": 100
      },
      "stage4": {
        "llm_bench": 100
      },
      "stage5": {
        "llm_bench": 100
      }
    },
    "reasoning_cost_mode_default": "token"
  },
  "specialist_summary": {
    "specialist_episode_count": 32,
    "specialist_shared_path_fraction": 0.875,
    "specialist_unshared_path_fraction": 0.125,
    "specialist_exact_match_mean": 0.40625,
    "specialist_total_cost_mean": 0.43581439317447496,
    "specialist_raw_outcome_penalty_mean": 3.4375,
    "specialist_raw_policy_penalty_mean": 1.625,
    "specialist_raw_terminal_penalty_mean": 9.625,
    "specialist_raw_path_cost_component_mean": 0.064734375,
    "specialist_raw_reasoning_cost_component_mean": 6.4964121875,
    "specialist_raw_reasoning_cost_component_api_mean": 1.6431945,
    "specialist_raw_reasoning_cost_component_token_mean": 6.4964121875,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}