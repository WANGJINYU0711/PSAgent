{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x2_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "epsilon_exp3",
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
    "exact_match_mean": 0.2,
    "terminal_penalty_mean": 9.7,
    "raw_outcome_penalty_mean": 8.8,
    "raw_policy_penalty_mean": 0.9,
    "raw_terminal_penalty_mean": 9.7,
    "total_cost_mean": 0.45512054315027156,
    "raw_total_cost_mean": 15.082694799999999,
    "raw_total_cost_api_mean": 10.41950635,
    "raw_total_cost_token_mean": 15.082694799999999,
    "reasoning_cost_mean": 5.3128148,
    "raw_reasoning_cost_component_mean": 5.3128148,
    "raw_reasoning_cost_component_api_mean": 0.64962635,
    "raw_reasoning_cost_component_token_mean": 5.3128148,
    "raw_path_cost_component_mean": 0.06988000000000001,
    "algorithm_cumulative_total_cost": 9.102410863005431,
    "raw_algorithm_cumulative_total_cost": 301.653896,
    "oracle_stationary_total_cost": 1.3019915509957756,
    "raw_oracle_stationary_total_cost": 43.148,
    "raw_outcome_penalty_cumulative": 176.0,
    "raw_policy_penalty_cumulative": 18.0,
    "raw_terminal_penalty_cumulative": 194.0,
    "raw_path_cost_component_cumulative": 1.3976000000000002,
    "raw_reasoning_cost_component_cumulative": 106.256296,
    "mean_llm_call_count": 12.95,
    "mean_prompt_tokens": 49970.5055,
    "mean_completion_tokens": 1387.729,
    "mean_total_tokens": 51358.2345,
    "cumulative_total_tokens": 1027164.69,
    "mean_api_cost_usd_raw": 0.0067573975,
    "cumulative_api_cost_usd_raw": 0.13514795,
    "mean_generation_time_seconds": 42.27408851746383,
    "p50_generation_time_seconds": 40.13491881079972,
    "p90_generation_time_seconds": 49.92050761636347,
    "mean_llm_round_trip_seconds": 42.31530944313916,
    "mean_episode_wall_clock_seconds": 44.96772383635441,
    "p50_episode_wall_clock_seconds": 42.88038252107799,
    "p90_episode_wall_clock_seconds": 52.98999896291643,
    "mean_tool_wall_clock_seconds": 0.011282067546868325,
    "policy_action_violation_rate": 0.45,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.45,
    "subset_mismatch_count": 16,
    "episodes_with_stage5_verification_tools": 20,
    "shared_path_fraction": 1.0,
    "unshared_path_fraction": 0.0,
    "mean_barrier_stop_depth": 2.0,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 13,
        "fraction": 0.65
      },
      "stage1": {
        "count": 5,
        "fraction": 0.25
      },
      "stage3": {
        "count": 1,
        "fraction": 0.05
      },
      "stage4": {
        "count": 1,
        "fraction": 0.05
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.3,
      3.1,
      2.35,
      2.65
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.3,
      3.1,
      2.35,
      2.65
    ],
    "specialist_task_count": 14,
    "specialist_task_unshared_fraction": 0.0,
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
    "specialist_shared_path_fraction": 1.0,
    "specialist_unshared_path_fraction": 0.0,
    "specialist_exact_match_mean": 0.07142857142857142,
    "specialist_total_cost_mean": 0.5718007802396758,
    "specialist_raw_outcome_penalty_mean": 12.035714285714286,
    "specialist_raw_policy_penalty_mean": 1.2857142857142858,
    "specialist_raw_terminal_penalty_mean": 13.321428571428571,
    "specialist_raw_path_cost_component_mean": 0.07062142857142857,
    "specialist_raw_reasoning_cost_component_mean": 5.557427857142857,
    "specialist_raw_reasoning_cost_component_api_mean": 0.6733312142857143,
    "specialist_raw_reasoning_cost_component_token_mean": 5.557427857142857,
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