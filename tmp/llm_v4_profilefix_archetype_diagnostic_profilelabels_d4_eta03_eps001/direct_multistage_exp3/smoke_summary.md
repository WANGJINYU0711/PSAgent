{
  "summary": {
    "test_name": "direct_multistage_exp3_repeated_trap_switch_x2_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json",
    "dataset_indices": [
      7,
      11,
      13,
      16,
      17,
      19,
      25,
      35,
      36,
      40,
      45,
      46,
      51,
      65,
      67
    ],
    "repeats": 2,
    "episodes": 20,
    "method": "direct_multistage_exp3",
    "mechanism": "algorithm_direct",
    "executor_name": "llm_bench",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n1__from__root__c01",
      "stage2_n2__from__n0001__c02",
      "stage3_n2__from__n0007__c01",
      "stage4_n1__from__n0025__c01",
      "stage5_n2__from__n0076__c02"
    ],
    "exact_match_mean": 0.2,
    "terminal_penalty_mean": 6.8,
    "raw_outcome_penalty_mean": 5.5,
    "raw_policy_penalty_mean": 1.3,
    "raw_terminal_penalty_mean": 6.8,
    "total_cost_mean": 0.3618840841882921,
    "raw_total_cost_mean": 11.992838549999998,
    "raw_total_cost_api_mean": 7.497656549999999,
    "raw_total_cost_token_mean": 11.992838549999998,
    "reasoning_cost_mean": 5.12312355,
    "raw_reasoning_cost_component_mean": 5.12312355,
    "raw_reasoning_cost_component_api_mean": 0.62794155,
    "raw_reasoning_cost_component_token_mean": 5.12312355,
    "raw_path_cost_component_mean": 0.069715,
    "algorithm_cumulative_total_cost": 7.2376816837658415,
    "raw_algorithm_cumulative_total_cost": 239.85677099999998,
    "oracle_stationary_total_cost": 0.546657242003621,
    "raw_oracle_stationary_total_cost": 18.116221,
    "raw_outcome_penalty_cumulative": 110.0,
    "raw_policy_penalty_cumulative": 26.0,
    "raw_terminal_penalty_cumulative": 136.0,
    "raw_path_cost_component_cumulative": 1.3943,
    "raw_reasoning_cost_component_cumulative": 102.462471,
    "mean_llm_call_count": 12.0,
    "mean_prompt_tokens": 46528.893,
    "mean_completion_tokens": 1449.0385,
    "mean_total_tokens": 47977.9315,
    "cumulative_total_tokens": 959558.63,
    "mean_api_cost_usd_raw": 0.0064051575,
    "cumulative_api_cost_usd_raw": 0.12810315,
    "mean_generation_time_seconds": 39.17454095527674,
    "p50_generation_time_seconds": 37.98554659153771,
    "p90_generation_time_seconds": 42.941377670272374,
    "mean_llm_round_trip_seconds": 39.213209757056504,
    "mean_episode_wall_clock_seconds": 41.9486495690148,
    "p50_episode_wall_clock_seconds": 40.68197193273217,
    "p90_episode_wall_clock_seconds": 45.985587910894935,
    "mean_tool_wall_clock_seconds": 0.011298410549646616,
    "policy_action_violation_rate": 0.65,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.65,
    "subset_mismatch_count": 16,
    "episodes_with_stage5_verification_tools": 20,
    "shared_path_fraction": 0.75,
    "unshared_path_fraction": 0.25,
    "mean_barrier_stop_depth": 2.4444444444444446,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 6,
        "fraction": 0.3
      },
      "stage1": {
        "count": 6,
        "fraction": 0.3
      },
      "stage2": {
        "count": 1,
        "fraction": 0.05
      },
      "stage3": {
        "count": 4,
        "fraction": 0.2
      },
      "stage4": {
        "count": 3,
        "fraction": 0.15
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.3,
      3.1,
      3.1,
      3.25
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.3,
      3.1,
      3.1,
      3.25
    ],
    "specialist_task_count": 14,
    "specialist_task_unshared_fraction": 0.2857142857142857,
    "stage_source_summary": {
      "stage1": {
        "llm_bench": 20
      },
      "stage2": {
        "llm_bench": 20
      },
      "stage3": {
        "llm_bench": 20
      },
      "stage4": {
        "llm_bench": 20
      },
      "stage5": {
        "llm_bench": 20
      }
    },
    "reasoning_cost_mode_default": "token"
  },
  "specialist_summary": {
    "specialist_episode_count": 14,
    "specialist_shared_path_fraction": 0.7142857142857143,
    "specialist_unshared_path_fraction": 0.2857142857142857,
    "specialist_exact_match_mean": 0.14285714285714285,
    "specialist_total_cost_mean": 0.42804966160875935,
    "specialist_raw_outcome_penalty_mean": 7.142857142857143,
    "specialist_raw_policy_penalty_mean": 1.7142857142857142,
    "specialist_raw_terminal_penalty_mean": 8.857142857142858,
    "specialist_raw_path_cost_component_mean": 0.06888571428571429,
    "specialist_raw_reasoning_cost_component_mean": 5.259537214285714,
    "specialist_raw_reasoning_cost_component_api_mean": 0.6578285,
    "specialist_raw_reasoning_cost_component_token_mean": 5.259537214285714,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]"
    ]
  }
}