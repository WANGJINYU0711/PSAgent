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
    "terminal_penalty_mean": 7.92,
    "raw_outcome_penalty_mean": 6.86,
    "raw_policy_penalty_mean": 1.06,
    "raw_terminal_penalty_mean": 7.92,
    "total_cost_mean": 0.394301025950513,
    "raw_total_cost_mean": 13.067136,
    "raw_total_cost_api_mean": 8.47908307,
    "raw_total_cost_token_mean": 13.067136,
    "reasoning_cost_mean": 5.075851,
    "raw_reasoning_cost_component_mean": 5.075851,
    "raw_reasoning_cost_component_api_mean": 0.48779807,
    "raw_reasoning_cost_component_token_mean": 5.075851,
    "raw_path_cost_component_mean": 0.071285,
    "algorithm_cumulative_total_cost": 39.430102595051295,
    "raw_algorithm_cumulative_total_cost": 1306.7136,
    "oracle_stationary_total_cost": 6.449607724803863,
    "raw_oracle_stationary_total_cost": 213.74,
    "raw_outcome_penalty_cumulative": 686.0,
    "raw_policy_penalty_cumulative": 106.0,
    "raw_terminal_penalty_cumulative": 792.0,
    "raw_path_cost_component_cumulative": 7.128500000000001,
    "raw_reasoning_cost_component_cumulative": 507.5851,
    "mean_llm_call_count": 12.66,
    "mean_prompt_tokens": 47882.88,
    "mean_completion_tokens": 1382.26,
    "mean_total_tokens": 49265.14,
    "cumulative_total_tokens": 4926514.0,
    "mean_api_cost_usd_raw": 0.005116908,
    "cumulative_api_cost_usd_raw": 0.5116908,
    "mean_generation_time_seconds": 38.37051753588021,
    "p50_generation_time_seconds": 38.99302090611309,
    "p90_generation_time_seconds": 44.09817567430437,
    "mean_llm_round_trip_seconds": 38.414404452387245,
    "mean_episode_wall_clock_seconds": 41.783439884893596,
    "p50_episode_wall_clock_seconds": 42.34352925699204,
    "p90_episode_wall_clock_seconds": 47.611237914673985,
    "mean_tool_wall_clock_seconds": 0.011272590197622777,
    "policy_action_violation_rate": 0.53,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.53,
    "subset_mismatch_count": 75,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.92,
    "unshared_path_fraction": 0.08,
    "mean_barrier_stop_depth": 2.6153846153846154,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 66,
        "fraction": 0.66
      },
      "stage1": {
        "count": 10,
        "fraction": 0.1
      },
      "stage2": {
        "count": 4,
        "fraction": 0.04
      },
      "stage3": {
        "count": 15,
        "fraction": 0.15
      },
      "stage4": {
        "count": 4,
        "fraction": 0.04
      },
      "stage5": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.54,
      2.84,
      2.29,
      2.71
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.54,
      2.84,
      2.29,
      2.71
    ],
    "specialist_task_count": 68,
    "specialist_task_unshared_fraction": 0.04411764705882353,
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
    "specialist_episode_count": 68,
    "specialist_shared_path_fraction": 0.9558823529411765,
    "specialist_unshared_path_fraction": 0.04411764705882353,
    "specialist_exact_match_mean": 0.22058823529411764,
    "specialist_total_cost_mean": 0.48994694744222367,
    "specialist_raw_outcome_penalty_mean": 9.602941176470589,
    "specialist_raw_policy_penalty_mean": 1.411764705882353,
    "specialist_raw_terminal_penalty_mean": 11.014705882352942,
    "specialist_raw_path_cost_component_mean": 0.07137941176470589,
    "specialist_raw_reasoning_cost_component_mean": 5.150756544117647,
    "specialist_raw_reasoning_cost_component_api_mean": 0.49843302941176465,
    "specialist_raw_reasoning_cost_component_token_mean": 5.150756544117647,
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