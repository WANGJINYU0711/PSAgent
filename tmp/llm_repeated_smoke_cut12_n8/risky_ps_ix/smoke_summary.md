{
  "summary": {
    "test_name": "risky_ps_ix_repeated_trap_switch_x8_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_ix",
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
    "terminal_penalty_mean": 7.87109375,
    "raw_outcome_penalty_mean": 6.26171875,
    "raw_policy_penalty_mean": 1.609375,
    "raw_terminal_penalty_mean": 7.87109375,
    "total_cost_mean": 0.37400517336489136,
    "raw_total_cost_mean": 12.3945314453125,
    "raw_total_cost_api_mean": 8.5047615234375,
    "raw_total_cost_token_mean": 12.3945314453125,
    "reasoning_cost_mean": 4.4523806640625,
    "raw_reasoning_cost_component_mean": 4.4523806640625,
    "raw_reasoning_cost_component_api_mean": 0.5626107421875,
    "raw_reasoning_cost_component_token_mean": 4.4523806640625,
    "raw_path_cost_component_mean": 0.07105703125,
    "algorithm_cumulative_total_cost": 47.872662190706095,
    "raw_algorithm_cumulative_total_cost": 1586.500025,
    "oracle_stationary_total_cost": 14.762776101388052,
    "raw_oracle_stationary_total_cost": 489.23840000000007,
    "raw_outcome_penalty_cumulative": 801.5,
    "raw_policy_penalty_cumulative": 206.0,
    "raw_terminal_penalty_cumulative": 1007.5,
    "raw_path_cost_component_cumulative": 9.0953,
    "raw_reasoning_cost_component_cumulative": 569.904725,
    "mean_llm_call_count": 11.046875,
    "mean_prompt_tokens": 41562.4140625,
    "mean_completion_tokens": 1271.625,
    "mean_total_tokens": 42834.0390625,
    "cumulative_total_tokens": 5482757.0,
    "mean_api_cost_usd_raw": 0.0059101328125,
    "cumulative_api_cost_usd_raw": 0.756497,
    "mean_generation_time_seconds": 32.01212756719906,
    "p50_generation_time_seconds": 31.847577499458566,
    "p90_generation_time_seconds": 36.18166438983753,
    "mean_llm_round_trip_seconds": 32.035976139843115,
    "mean_episode_wall_clock_seconds": 34.68783132576209,
    "p50_episode_wall_clock_seconds": 34.818850399693474,
    "p90_episode_wall_clock_seconds": 38.644050660496575,
    "mean_tool_wall_clock_seconds": 0.0051365797662583645,
    "policy_action_violation_rate": 0.8046875,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.8046875,
    "subset_mismatch_count": 76,
    "episodes_with_stage5_verification_tools": 128,
    "shared_path_fraction": 0.9375,
    "unshared_path_fraction": 0.0625,
    "mean_barrier_stop_depth": 2.685185185185185,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 66,
        "fraction": 0.515625
      },
      "stage1": {
        "count": 37,
        "fraction": 0.2890625
      },
      "stage2": {
        "count": 5,
        "fraction": 0.0390625
      },
      "stage3": {
        "count": 17,
        "fraction": 0.1328125
      },
      "stage4": {
        "count": 3,
        "fraction": 0.0234375
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.78125,
      2.9609375,
      2.6953125,
      2.9921875
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.78125,
      2.9609375,
      2.6953125,
      2.9921875
    ],
    "specialist_task_count": 91,
    "specialist_task_unshared_fraction": 0.04395604395604396,
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
    "specialist_shared_path_fraction": 0.9560439560439561,
    "specialist_unshared_path_fraction": 0.04395604395604396,
    "specialist_exact_match_mean": 0.4065934065934066,
    "specialist_total_cost_mean": 0.4395595989707335,
    "specialist_raw_outcome_penalty_mean": 8.093406593406593,
    "specialist_raw_policy_penalty_mean": 1.8021978021978022,
    "specialist_raw_terminal_penalty_mean": 9.895604395604396,
    "specialist_raw_path_cost_component_mean": 0.07093406593406594,
    "specialist_raw_reasoning_cost_component_mean": 4.600466648351649,
    "specialist_raw_reasoning_cost_component_api_mean": 0.583831,
    "specialist_raw_reasoning_cost_component_token_mean": 4.600466648351649,
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