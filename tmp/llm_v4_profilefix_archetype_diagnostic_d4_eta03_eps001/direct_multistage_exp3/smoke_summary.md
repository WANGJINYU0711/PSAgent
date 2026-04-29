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
      "stage1_n5__from__root__c05",
      "stage2_n4__from__n0005__c02",
      "stage3_n5__from__n0021__c03",
      "stage4_n3__from__n0070__c02",
      "stage5_n3__from__n0203__c03"
    ],
    "exact_match_mean": 0.15,
    "terminal_penalty_mean": 12.075,
    "raw_outcome_penalty_mean": 10.775,
    "raw_policy_penalty_mean": 1.3,
    "raw_terminal_penalty_mean": 12.075,
    "total_cost_mean": 0.5241586692818346,
    "raw_total_cost_mean": 17.3706183,
    "raw_total_cost_api_mean": 12.8149895,
    "raw_total_cost_token_mean": 17.3706183,
    "reasoning_cost_mean": 5.2260033,
    "raw_reasoning_cost_component_mean": 5.2260033,
    "raw_reasoning_cost_component_api_mean": 0.6703745,
    "raw_reasoning_cost_component_token_mean": 5.2260033,
    "raw_path_cost_component_mean": 0.06961500000000001,
    "algorithm_cumulative_total_cost": 10.483173385636693,
    "raw_algorithm_cumulative_total_cost": 347.412366,
    "oracle_stationary_total_cost": 1.3019915509957756,
    "raw_oracle_stationary_total_cost": 43.148,
    "raw_outcome_penalty_cumulative": 215.5,
    "raw_policy_penalty_cumulative": 26.0,
    "raw_terminal_penalty_cumulative": 241.5,
    "raw_path_cost_component_cumulative": 1.3923,
    "raw_reasoning_cost_component_cumulative": 104.520066,
    "mean_llm_call_count": 12.4,
    "mean_prompt_tokens": 47798.083,
    "mean_completion_tokens": 1461.147,
    "mean_total_tokens": 49259.229999999996,
    "cumulative_total_tokens": 985184.6,
    "mean_api_cost_usd_raw": 0.0068462275000000005,
    "cumulative_api_cost_usd_raw": 0.13692455,
    "mean_generation_time_seconds": 41.980949334146004,
    "p50_generation_time_seconds": 43.268512272275984,
    "p90_generation_time_seconds": 47.700673025097885,
    "mean_llm_round_trip_seconds": 42.02035395878135,
    "mean_episode_wall_clock_seconds": 44.74269902536899,
    "p50_episode_wall_clock_seconds": 45.94328447151929,
    "p90_episode_wall_clock_seconds": 50.37535269456979,
    "mean_tool_wall_clock_seconds": 0.010684759163218736,
    "policy_action_violation_rate": 0.65,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.65,
    "subset_mismatch_count": 17,
    "episodes_with_stage5_verification_tools": 20,
    "shared_path_fraction": 0.8,
    "unshared_path_fraction": 0.2,
    "mean_barrier_stop_depth": 2.125,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 8,
        "fraction": 0.4
      },
      "stage1": {
        "count": 5,
        "fraction": 0.25
      },
      "stage2": {
        "count": 1,
        "fraction": 0.05
      },
      "stage3": {
        "count": 3,
        "fraction": 0.15
      },
      "stage4": {
        "count": 3,
        "fraction": 0.15
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.4,
      3.1,
      2.8,
      3.1
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.4,
      3.1,
      2.8,
      3.1
    ],
    "specialist_task_count": 14,
    "specialist_task_unshared_fraction": 0.21428571428571427,
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
    "specialist_shared_path_fraction": 0.7857142857142857,
    "specialist_unshared_path_fraction": 0.21428571428571427,
    "specialist_exact_match_mean": 0.07142857142857142,
    "specialist_total_cost_mean": 0.670763828778343,
    "specialist_raw_outcome_penalty_mean": 14.964285714285714,
    "specialist_raw_policy_penalty_mean": 1.7142857142857142,
    "specialist_raw_terminal_penalty_mean": 16.678571428571427,
    "specialist_raw_path_cost_component_mean": 0.0689357142857143,
    "specialist_raw_reasoning_cost_component_mean": 5.481606142857143,
    "specialist_raw_reasoning_cost_component_api_mean": 0.6962084285714286,
    "specialist_raw_reasoning_cost_component_token_mean": 5.481606142857143,
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