{
  "summary": {
    "test_name": "risky_ps_linear_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_linear",
    "mechanism": "algorithm_direct",
    "executor_name": "llm_bench",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n5__from__root__c05",
      "stage2_n5__from__n0005__c03",
      "stage3_n5__from__n0022__c03",
      "stage4_n3__from__n0073__c02",
      "stage5_n4__from__n0213__c04"
    ],
    "exact_match_mean": 0.68,
    "terminal_penalty_mean": 6.2,
    "raw_outcome_penalty_mean": 3.86,
    "raw_policy_penalty_mean": 0.46,
    "raw_terminal_penalty_mean": 6.2,
    "legacy_raw_terminal_penalty_mean": 4.32,
    "raw_terminal_penalty_exec_clean_v4_mean": 6.2,
    "total_cost_mean": 0.3471618255250404,
    "raw_total_cost_mean": 12.8935902,
    "raw_total_cost_api_mean": 8.30954956,
    "raw_total_cost_token_mean": 12.8935902,
    "reasoning_cost_mean": 6.6227122,
    "raw_reasoning_cost_component_mean": 6.6227122,
    "raw_mode_mismatch_cost_component_mean": 1.54,
    "raw_reasoning_cost_component_api_mean": 2.03867156,
    "raw_reasoning_cost_component_token_mean": 6.6227122,
    "raw_path_cost_component_mean": 0.07087800000000001,
    "algorithm_cumulative_total_cost": 34.71618255250404,
    "raw_algorithm_cumulative_total_cost": 1289.35902,
    "oracle_stationary_total_cost": 20.49191733979537,
    "raw_oracle_stationary_total_cost": 761.0698100000001,
    "raw_outcome_penalty_cumulative": 386.0,
    "raw_policy_penalty_cumulative": 46.0,
    "raw_terminal_penalty_cumulative": 620.0,
    "legacy_raw_terminal_penalty_cumulative": 432.0,
    "raw_path_cost_component_cumulative": 7.0878000000000005,
    "raw_reasoning_cost_component_cumulative": 662.27122,
    "raw_mode_mismatch_cost_component_cumulative": 154.0,
    "mean_llm_call_count": 12.9,
    "mean_prompt_tokens": 47173.14,
    "mean_completion_tokens": 1344.69,
    "mean_total_tokens": 48517.83,
    "cumulative_total_tokens": 4851783.0,
    "mean_api_cost_usd_raw": 0.0051331530000000005,
    "cumulative_api_cost_usd_raw": 0.5133153,
    "mean_generation_time_seconds": 35.350166911445555,
    "p50_generation_time_seconds": 34.41878291312605,
    "p90_generation_time_seconds": 40.68421188984067,
    "mean_llm_round_trip_seconds": 35.39296560464427,
    "mean_episode_wall_clock_seconds": 38.326068079415705,
    "p50_episode_wall_clock_seconds": 37.43661053292453,
    "p90_episode_wall_clock_seconds": 43.54243321828544,
    "mean_tool_wall_clock_seconds": 0.011903360337018967,
    "policy_action_violation_rate": 0.23,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.23,
    "subset_mismatch_count": 32,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.93,
    "unshared_path_fraction": 0.07,
    "mean_barrier_stop_depth": 2.5434782608695654,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 47,
        "fraction": 0.47
      },
      "stage1": {
        "count": 33,
        "fraction": 0.33
      },
      "stage2": {
        "count": 5,
        "fraction": 0.05
      },
      "stage3": {
        "count": 12,
        "fraction": 0.12
      },
      "stage4": {
        "count": 3,
        "fraction": 0.03
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.4,
      3.0,
      2.86,
      3.02
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.4,
      3.0,
      2.86,
      3.02
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.15625,
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
    "specialist_shared_path_fraction": 0.84375,
    "specialist_unshared_path_fraction": 0.15625,
    "specialist_exact_match_mean": 0.3125,
    "specialist_total_cost_mean": 0.5744430070341949,
    "specialist_raw_outcome_penalty_mean": 9.125,
    "specialist_raw_policy_penalty_mean": 1.4375,
    "specialist_raw_terminal_penalty_mean": 14.546875,
    "specialist_raw_path_cost_component_mean": 0.071353125,
    "specialist_raw_reasoning_cost_component_mean": 6.71658515625,
    "specialist_raw_reasoning_cost_component_api_mean": 2.00355915625,
    "specialist_raw_reasoning_cost_component_token_mean": 6.71658515625,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}