{
  "summary": {
    "test_name": "risky_ps_safe_conditional_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_safe_conditional",
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
    "exact_match_mean": 0.18,
    "terminal_penalty_mean": 8.875,
    "raw_outcome_penalty_mean": 7.735,
    "raw_policy_penalty_mean": 1.14,
    "raw_terminal_penalty_mean": 8.875,
    "total_cost_mean": 0.42429568949909474,
    "raw_total_cost_mean": 14.061159150000002,
    "raw_total_cost_api_mean": 9.45804903,
    "raw_total_cost_token_mean": 14.061159150000002,
    "reasoning_cost_mean": 5.1172961500000005,
    "raw_reasoning_cost_component_mean": 5.1172961500000005,
    "raw_reasoning_cost_component_api_mean": 0.51418603,
    "raw_reasoning_cost_component_token_mean": 5.1172961500000005,
    "raw_path_cost_component_mean": 0.06886300000000001,
    "algorithm_cumulative_total_cost": 42.42956894990947,
    "raw_algorithm_cumulative_total_cost": 1406.115915,
    "oracle_stationary_total_cost": 6.449607724803863,
    "raw_oracle_stationary_total_cost": 213.74,
    "raw_outcome_penalty_cumulative": 773.5,
    "raw_policy_penalty_cumulative": 114.0,
    "raw_terminal_penalty_cumulative": 887.5,
    "raw_path_cost_component_cumulative": 6.8863,
    "raw_reasoning_cost_component_cumulative": 511.729615,
    "mean_llm_call_count": 11.95,
    "mean_prompt_tokens": 46371.07,
    "mean_completion_tokens": 1325.46,
    "mean_total_tokens": 47696.53,
    "cumulative_total_tokens": 4769653.0,
    "mean_api_cost_usd_raw": 0.005183128499999999,
    "cumulative_api_cost_usd_raw": 0.5183128499999999,
    "mean_generation_time_seconds": 36.17542877342552,
    "p50_generation_time_seconds": 35.89930465631187,
    "p90_generation_time_seconds": 41.92149457931519,
    "mean_llm_round_trip_seconds": 36.218155718054625,
    "mean_episode_wall_clock_seconds": 39.29072454761714,
    "p50_episode_wall_clock_seconds": 38.92725957930088,
    "p90_episode_wall_clock_seconds": 44.97136500142515,
    "mean_tool_wall_clock_seconds": 0.011103965751826763,
    "policy_action_violation_rate": 0.57,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.57,
    "subset_mismatch_count": 82,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.89,
    "unshared_path_fraction": 0.11,
    "mean_barrier_stop_depth": 2.9705882352941178,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 55,
        "fraction": 0.55
      },
      "stage1": {
        "count": 10,
        "fraction": 0.1
      },
      "stage2": {
        "count": 5,
        "fraction": 0.05
      },
      "stage3": {
        "count": 19,
        "fraction": 0.19
      },
      "stage4": {
        "count": 7,
        "fraction": 0.07
      },
      "stage5": {
        "count": 4,
        "fraction": 0.04
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.44,
      3.05,
      3.02,
      3.07
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.44,
      3.05,
      3.02,
      3.07
    ],
    "specialist_task_count": 68,
    "specialist_task_unshared_fraction": 0.11764705882352941,
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
    "specialist_shared_path_fraction": 0.8823529411764706,
    "specialist_unshared_path_fraction": 0.11764705882352941,
    "specialist_exact_match_mean": 0.11764705882352941,
    "specialist_total_cost_mean": 0.530035074017537,
    "specialist_raw_outcome_penalty_mean": 10.816176470588236,
    "specialist_raw_policy_penalty_mean": 1.5588235294117647,
    "specialist_raw_terminal_penalty_mean": 12.375,
    "specialist_raw_path_cost_component_mean": 0.06807058823529412,
    "specialist_raw_reasoning_cost_component_mean": 5.122291764705882,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5209115441176471,
    "specialist_raw_reasoning_cost_component_token_mean": 5.122291764705882,
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