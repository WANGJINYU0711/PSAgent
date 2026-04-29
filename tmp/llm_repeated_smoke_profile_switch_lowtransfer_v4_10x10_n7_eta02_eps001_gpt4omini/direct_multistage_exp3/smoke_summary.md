{
  "summary": {
    "test_name": "direct_multistage_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json",
    "dataset_indices": [
      0,
      7,
      11,
      13,
      16,
      17,
      18,
      19,
      23,
      24,
      25,
      35,
      36,
      40,
      45,
      46,
      51,
      61,
      65,
      67
    ],
    "repeats": 10,
    "episodes": 100,
    "method": "direct_multistage_exp3",
    "mechanism": "algorithm_direct",
    "executor_name": "llm_bench",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n5__from__root__c05",
      "stage2_n4__from__n0005__c02",
      "stage3_n5__from__n0021__c03",
      "stage4_n3__from__n0070__c02",
      "stage5_n3__from__n0203__c03"
    ],
    "exact_match_mean": 0.25,
    "terminal_penalty_mean": 9.075,
    "raw_outcome_penalty_mean": 7.755,
    "raw_policy_penalty_mean": 1.32,
    "raw_terminal_penalty_mean": 9.075,
    "total_cost_mean": 0.42811958509354253,
    "raw_total_cost_mean": 14.18788305,
    "raw_total_cost_api_mean": 9.65456051,
    "raw_total_cost_token_mean": 14.18788305,
    "reasoning_cost_mean": 5.04199205,
    "raw_reasoning_cost_component_mean": 5.04199205,
    "raw_reasoning_cost_component_api_mean": 0.50866951,
    "raw_reasoning_cost_component_token_mean": 5.04199205,
    "raw_path_cost_component_mean": 0.070891,
    "algorithm_cumulative_total_cost": 42.811958509354255,
    "raw_algorithm_cumulative_total_cost": 1418.788305,
    "oracle_stationary_total_cost": 6.946650573325287,
    "raw_oracle_stationary_total_cost": 230.21200000000002,
    "raw_outcome_penalty_cumulative": 775.5,
    "raw_policy_penalty_cumulative": 132.0,
    "raw_terminal_penalty_cumulative": 907.5,
    "raw_path_cost_component_cumulative": 7.0891,
    "raw_reasoning_cost_component_cumulative": 504.199205,
    "mean_llm_call_count": 12.57,
    "mean_prompt_tokens": 48079.75,
    "mean_completion_tokens": 1430.33,
    "mean_total_tokens": 49510.08,
    "cumulative_total_tokens": 4951008.0,
    "mean_api_cost_usd_raw": 0.005395984499999999,
    "cumulative_api_cost_usd_raw": 0.53959845,
    "mean_generation_time_seconds": 41.54121611239388,
    "p50_generation_time_seconds": 39.67635476682335,
    "p90_generation_time_seconds": 49.02936396524311,
    "mean_llm_round_trip_seconds": 41.58553791785613,
    "mean_episode_wall_clock_seconds": 44.84293498747051,
    "p50_episode_wall_clock_seconds": 42.91224482096732,
    "p90_episode_wall_clock_seconds": 52.36276224087924,
    "mean_tool_wall_clock_seconds": 0.012416150905191898,
    "policy_action_violation_rate": 0.66,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.66,
    "subset_mismatch_count": 75,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.89,
    "unshared_path_fraction": 0.11,
    "mean_barrier_stop_depth": 2.7083333333333335,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 65,
        "fraction": 0.65
      },
      "stage1": {
        "count": 12,
        "fraction": 0.12
      },
      "stage2": {
        "count": 4,
        "fraction": 0.04
      },
      "stage3": {
        "count": 14,
        "fraction": 0.14
      },
      "stage4": {
        "count": 5,
        "fraction": 0.05
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.52,
      2.86,
      2.4,
      2.78
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.52,
      2.86,
      2.4,
      2.78
    ],
    "specialist_task_count": 78,
    "specialist_task_unshared_fraction": 0.10256410256410256,
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
    "specialist_episode_count": 78,
    "specialist_shared_path_fraction": 0.8974358974358975,
    "specialist_unshared_path_fraction": 0.10256410256410256,
    "specialist_exact_match_mean": 0.24358974358974358,
    "specialist_total_cost_mean": 0.49470850741995887,
    "specialist_raw_outcome_penalty_mean": 9.621794871794872,
    "specialist_raw_policy_penalty_mean": 1.564102564102564,
    "specialist_raw_terminal_penalty_mean": 11.185897435897436,
    "specialist_raw_path_cost_component_mean": 0.07153205128205128,
    "specialist_raw_reasoning_cost_component_mean": 5.137210448717949,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5203284743589744,
    "specialist_raw_reasoning_cost_component_token_mean": 5.137210448717949,
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