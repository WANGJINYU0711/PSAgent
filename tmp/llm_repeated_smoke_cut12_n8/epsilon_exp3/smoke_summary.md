{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x8_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data\\derived\\telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch\\tasks.json",
    "dataset_indices": [
      0,
      1,
      8,
      9,
      10,
      11,
      14,
      17,
      18,
      20,
      25,
      26,
      27,
      29,
      35,
      36,
      37,
      38,
      44,
      51,
      53,
      54,
      61,
      68,
      69,
      70,
      77,
      84,
      85,
      87,
      92,
      98
    ],
    "repeats": 8,
    "episodes": 128,
    "method": "epsilon_exp3",
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
      "stage4_n5__from__n0073__c04",
      "stage5_n5__from__n0215__c03"
    ],
    "exact_match_mean": 0.40625,
    "terminal_penalty_mean": 8.125,
    "raw_outcome_penalty_mean": 6.46875,
    "raw_policy_penalty_mean": 1.65625,
    "raw_terminal_penalty_mean": 8.125,
    "total_cost_mean": 0.3873310835659324,
    "raw_total_cost_mean": 12.836152109375,
    "raw_total_cost_api_mean": 8.789411515625,
    "raw_total_cost_token_mean": 12.836152109375,
    "reasoning_cost_mean": 4.640663828125,
    "raw_reasoning_cost_component_mean": 4.640663828125,
    "raw_reasoning_cost_component_api_mean": 0.5939232343749999,
    "raw_reasoning_cost_component_token_mean": 4.640663828125,
    "raw_path_cost_component_mean": 0.07048828125,
    "algorithm_cumulative_total_cost": 49.578378696439344,
    "raw_algorithm_cumulative_total_cost": 1643.02747,
    "oracle_stationary_total_cost": 14.762776101388052,
    "raw_oracle_stationary_total_cost": 489.23840000000007,
    "raw_outcome_penalty_cumulative": 828.0,
    "raw_policy_penalty_cumulative": 212.0,
    "raw_terminal_penalty_cumulative": 1040.0,
    "raw_path_cost_component_cumulative": 9.0225,
    "raw_reasoning_cost_component_cumulative": 594.00497,
    "mean_llm_call_count": 11.03125,
    "mean_prompt_tokens": 41706.5625,
    "mean_completion_tokens": 1255.5390625,
    "mean_total_tokens": 42962.1015625,
    "cumulative_total_tokens": 5499149.0,
    "mean_api_cost_usd_raw": 0.0059904265625,
    "cumulative_api_cost_usd_raw": 0.7667746,
    "mean_generation_time_seconds": 32.02509123896016,
    "p50_generation_time_seconds": 31.850333500420675,
    "p90_generation_time_seconds": 35.45090255052783,
    "mean_llm_round_trip_seconds": 32.04917379688777,
    "mean_episode_wall_clock_seconds": 34.71386011085997,
    "p50_episode_wall_clock_seconds": 34.334878199733794,
    "p90_episode_wall_clock_seconds": 38.38100968981162,
    "mean_tool_wall_clock_seconds": 0.005124308652739273,
    "policy_action_violation_rate": 0.828125,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.828125,
    "subset_mismatch_count": 76,
    "episodes_with_stage5_verification_tools": 128,
    "shared_path_fraction": 0.953125,
    "unshared_path_fraction": 0.046875,
    "mean_barrier_stop_depth": 2.693877551020408,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 73,
        "fraction": 0.5703125
      },
      "stage1": {
        "count": 21,
        "fraction": 0.1640625
      },
      "stage2": {
        "count": 5,
        "fraction": 0.0390625
      },
      "stage3": {
        "count": 24,
        "fraction": 0.1875
      },
      "stage4": {
        "count": 4,
        "fraction": 0.03125
      },
      "stage5": {
        "count": 1,
        "fraction": 0.0078125
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.265625,
      3.1171875,
      2.65625,
      2.828125
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.265625,
      3.1171875,
      2.65625,
      2.828125
    ],
    "specialist_task_count": 91,
    "specialist_task_unshared_fraction": 0.06593406593406594,
    "stage_source_summary": {
      "stage1": {
        "llm_bench": 128
      },
      "stage2": {
        "llm_bench": 128
      },
      "stage3": {
        "llm_bench": 128
      },
      "stage4": {
        "llm_bench": 128
      },
      "stage5": {
        "llm_bench": 128
      }
    },
    "reasoning_cost_mode_default": "token"
  },
  "specialist_summary": {
    "specialist_episode_count": 91,
    "specialist_shared_path_fraction": 0.9340659340659341,
    "specialist_unshared_path_fraction": 0.06593406593406594,
    "specialist_exact_match_mean": 0.4175824175824176,
    "specialist_total_cost_mean": 0.4479441198511808,
    "specialist_raw_outcome_penalty_mean": 8.225274725274724,
    "specialist_raw_policy_penalty_mean": 1.8241758241758241,
    "specialist_raw_terminal_penalty_mean": 10.04945054945055,
    "specialist_raw_path_cost_component_mean": 0.07068681318681319,
    "specialist_raw_reasoning_cost_component_mean": 4.724730769230769,
    "specialist_raw_reasoning_cost_component_api_mean": 0.6001973296703297,
    "specialist_raw_reasoning_cost_component_token_mean": 4.724730769230769,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]"
    ]
  }
}