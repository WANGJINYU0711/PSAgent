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
    "exact_match_mean": 0.19,
    "terminal_penalty_mean": 8.885,
    "raw_outcome_penalty_mean": 7.785,
    "raw_policy_penalty_mean": 1.1,
    "raw_terminal_penalty_mean": 8.885,
    "total_cost_mean": 0.4281730310802655,
    "raw_total_cost_mean": 14.18965425,
    "raw_total_cost_api_mean": 9.48560115,
    "raw_total_cost_token_mean": 14.18965425,
    "reasoning_cost_mean": 5.23359825,
    "raw_reasoning_cost_component_mean": 5.23359825,
    "raw_reasoning_cost_component_api_mean": 0.52954515,
    "raw_reasoning_cost_component_token_mean": 5.23359825,
    "raw_path_cost_component_mean": 0.07105600000000001,
    "algorithm_cumulative_total_cost": 42.81730310802655,
    "raw_algorithm_cumulative_total_cost": 1418.965425,
    "oracle_stationary_total_cost": 6.449607724803863,
    "raw_oracle_stationary_total_cost": 213.74,
    "raw_outcome_penalty_cumulative": 778.5,
    "raw_policy_penalty_cumulative": 110.0,
    "raw_terminal_penalty_cumulative": 888.5,
    "raw_path_cost_component_cumulative": 7.105600000000001,
    "raw_reasoning_cost_component_cumulative": 523.359825,
    "mean_llm_call_count": 12.87,
    "mean_prompt_tokens": 49219.47,
    "mean_completion_tokens": 1398.87,
    "mean_total_tokens": 50618.34,
    "cumulative_total_tokens": 5061834.0,
    "mean_api_cost_usd_raw": 0.0055156185000000005,
    "cumulative_api_cost_usd_raw": 0.55156185,
    "mean_generation_time_seconds": 40.168368040639905,
    "p50_generation_time_seconds": 39.4836770221591,
    "p90_generation_time_seconds": 46.922988527454436,
    "mean_llm_round_trip_seconds": 40.20977248450741,
    "mean_episode_wall_clock_seconds": 43.1734951454401,
    "p50_episode_wall_clock_seconds": 42.691113874316216,
    "p90_episode_wall_clock_seconds": 49.82178905438632,
    "mean_tool_wall_clock_seconds": 0.011801877673715353,
    "policy_action_violation_rate": 0.55,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.55,
    "subset_mismatch_count": 81,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.91,
    "unshared_path_fraction": 0.09,
    "mean_barrier_stop_depth": 2.611111111111111,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 55,
        "fraction": 0.55
      },
      "stage1": {
        "count": 13,
        "fraction": 0.13
      },
      "stage2": {
        "count": 9,
        "fraction": 0.09
      },
      "stage3": {
        "count": 19,
        "fraction": 0.19
      },
      "stage4": {
        "count": 2,
        "fraction": 0.02
      },
      "stage5": {
        "count": 2,
        "fraction": 0.02
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.64,
      2.97,
      2.52,
      2.81
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.64,
      2.97,
      2.52,
      2.81
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
    "specialist_total_cost_mean": 0.5350245637935318,
    "specialist_raw_outcome_penalty_mean": 10.889705882352942,
    "specialist_raw_policy_penalty_mean": 1.5,
    "specialist_raw_terminal_penalty_mean": 12.389705882352942,
    "specialist_raw_path_cost_component_mean": 0.07150588235294118,
    "specialist_raw_reasoning_cost_component_mean": 5.269502279411765,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5163733970588236,
    "specialist_raw_reasoning_cost_component_token_mean": 5.269502279411765,
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