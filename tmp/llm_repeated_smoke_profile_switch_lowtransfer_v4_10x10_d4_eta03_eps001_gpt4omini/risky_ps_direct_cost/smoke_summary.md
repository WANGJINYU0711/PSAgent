{
  "summary": {
    "test_name": "risky_ps_direct_cost_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_direct_cost",
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
    "exact_match_mean": 0.19,
    "terminal_penalty_mean": 8.92,
    "raw_outcome_penalty_mean": 7.8,
    "raw_policy_penalty_mean": 1.12,
    "raw_terminal_penalty_mean": 8.92,
    "total_cost_mean": 0.4276509354254677,
    "raw_total_cost_mean": 14.172352,
    "raw_total_cost_api_mean": 9.49810887,
    "raw_total_cost_token_mean": 14.172352,
    "reasoning_cost_mean": 5.181774000000001,
    "raw_reasoning_cost_component_mean": 5.181774000000001,
    "raw_reasoning_cost_component_api_mean": 0.50753087,
    "raw_reasoning_cost_component_token_mean": 5.181774000000001,
    "raw_path_cost_component_mean": 0.070578,
    "algorithm_cumulative_total_cost": 42.76509354254677,
    "raw_algorithm_cumulative_total_cost": 1417.2352,
    "oracle_stationary_total_cost": 6.449607724803863,
    "raw_oracle_stationary_total_cost": 213.74,
    "raw_outcome_penalty_cumulative": 780.0,
    "raw_policy_penalty_cumulative": 112.0,
    "raw_terminal_penalty_cumulative": 892.0,
    "raw_path_cost_component_cumulative": 7.0578,
    "raw_reasoning_cost_component_cumulative": 518.1774,
    "mean_llm_call_count": 12.65,
    "mean_prompt_tokens": 48210.38,
    "mean_completion_tokens": 1414.36,
    "mean_total_tokens": 49624.74,
    "cumulative_total_tokens": 4962474.0,
    "mean_api_cost_usd_raw": 0.005267948999999999,
    "cumulative_api_cost_usd_raw": 0.5267949,
    "mean_generation_time_seconds": 38.26055683810264,
    "p50_generation_time_seconds": 38.113848767243326,
    "p90_generation_time_seconds": 42.84449876099825,
    "mean_llm_round_trip_seconds": 38.30318078428507,
    "mean_episode_wall_clock_seconds": 41.33832784609869,
    "p50_episode_wall_clock_seconds": 41.277318521402776,
    "p90_episode_wall_clock_seconds": 45.95493183005601,
    "mean_tool_wall_clock_seconds": 0.011634540800005198,
    "policy_action_violation_rate": 0.56,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.56,
    "subset_mismatch_count": 81,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.94,
    "unshared_path_fraction": 0.06,
    "mean_barrier_stop_depth": 2.361111111111111,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 58,
        "fraction": 0.58
      },
      "stage1": {
        "count": 24,
        "fraction": 0.24
      },
      "stage2": {
        "count": 3,
        "fraction": 0.03
      },
      "stage3": {
        "count": 12,
        "fraction": 0.12
      },
      "stage4": {
        "count": 2,
        "fraction": 0.02
      },
      "stage5": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.28,
      3.05,
      2.63,
      2.87
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.28,
      3.05,
      2.63,
      2.87
    ],
    "specialist_task_count": 68,
    "specialist_task_unshared_fraction": 0.058823529411764705,
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
    "specialist_shared_path_fraction": 0.9411764705882353,
    "specialist_unshared_path_fraction": 0.058823529411764705,
    "specialist_exact_match_mean": 0.1323529411764706,
    "specialist_total_cost_mean": 0.5392832191416096,
    "specialist_raw_outcome_penalty_mean": 11.014705882352942,
    "specialist_raw_policy_penalty_mean": 1.5,
    "specialist_raw_terminal_penalty_mean": 12.514705882352942,
    "specialist_raw_path_cost_component_mean": 0.07006176470588237,
    "specialist_raw_reasoning_cost_component_mean": 5.287078235294118,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5278284264705883,
    "specialist_raw_reasoning_cost_component_token_mean": 5.287078235294118,
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