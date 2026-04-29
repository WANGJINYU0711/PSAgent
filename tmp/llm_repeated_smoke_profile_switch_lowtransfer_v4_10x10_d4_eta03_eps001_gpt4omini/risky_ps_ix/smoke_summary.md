{
  "summary": {
    "test_name": "risky_ps_ix_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_ix",
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
    "exact_match_mean": 0.22,
    "terminal_penalty_mean": 8.045,
    "raw_outcome_penalty_mean": 6.965,
    "raw_policy_penalty_mean": 1.08,
    "raw_terminal_penalty_mean": 8.045,
    "total_cost_mean": 0.3979275452625226,
    "raw_total_cost_mean": 13.187318849999999,
    "raw_total_cost_api_mean": 8.59249892,
    "raw_total_cost_token_mean": 13.187318849999999,
    "reasoning_cost_mean": 5.07175885,
    "raw_reasoning_cost_component_mean": 5.07175885,
    "raw_reasoning_cost_component_api_mean": 0.47693892,
    "raw_reasoning_cost_component_token_mean": 5.07175885,
    "raw_path_cost_component_mean": 0.07056000000000001,
    "algorithm_cumulative_total_cost": 39.79275452625226,
    "raw_algorithm_cumulative_total_cost": 1318.731885,
    "oracle_stationary_total_cost": 6.449607724803863,
    "raw_oracle_stationary_total_cost": 213.74,
    "raw_outcome_penalty_cumulative": 696.5,
    "raw_policy_penalty_cumulative": 108.0,
    "raw_terminal_penalty_cumulative": 804.5,
    "raw_path_cost_component_cumulative": 7.056000000000001,
    "raw_reasoning_cost_component_cumulative": 507.175885,
    "mean_llm_call_count": 12.5,
    "mean_prompt_tokens": 47297.97,
    "mean_completion_tokens": 1402.12,
    "mean_total_tokens": 48700.09,
    "cumulative_total_tokens": 4870009.0,
    "mean_api_cost_usd_raw": 0.0049516565,
    "cumulative_api_cost_usd_raw": 0.49516565,
    "mean_generation_time_seconds": 38.32897527649999,
    "p50_generation_time_seconds": 38.56668552290648,
    "p90_generation_time_seconds": 44.049944267421964,
    "mean_llm_round_trip_seconds": 38.37165355425328,
    "mean_episode_wall_clock_seconds": 41.74882440321147,
    "p50_episode_wall_clock_seconds": 42.248305413872004,
    "p90_episode_wall_clock_seconds": 47.57508591059596,
    "mean_tool_wall_clock_seconds": 0.011219900846481324,
    "policy_action_violation_rate": 0.54,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.54,
    "subset_mismatch_count": 78,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.94,
    "unshared_path_fraction": 0.06,
    "mean_barrier_stop_depth": 2.441860465116279,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 51,
        "fraction": 0.51
      },
      "stage1": {
        "count": 20,
        "fraction": 0.2
      },
      "stage2": {
        "count": 16,
        "fraction": 0.16
      },
      "stage3": {
        "count": 12,
        "fraction": 0.12
      },
      "stage4": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.64,
      3.08,
      2.7,
      2.9
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.64,
      3.08,
      2.7,
      2.9
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
    "specialist_exact_match_mean": 0.17647058823529413,
    "specialist_total_cost_mean": 0.49371400741950366,
    "specialist_raw_outcome_penalty_mean": 9.698529411764707,
    "specialist_raw_policy_penalty_mean": 1.4411764705882353,
    "specialist_raw_terminal_penalty_mean": 11.139705882352942,
    "specialist_raw_path_cost_component_mean": 0.07015882352941177,
    "specialist_raw_reasoning_cost_component_mean": 5.1518175,
    "specialist_raw_reasoning_cost_component_api_mean": 0.4879513529411765,
    "specialist_raw_reasoning_cost_component_token_mean": 5.1518175,
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