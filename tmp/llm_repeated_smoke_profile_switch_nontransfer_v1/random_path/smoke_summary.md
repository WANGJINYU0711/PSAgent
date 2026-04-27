{
  "summary": {
    "test_name": "random_path_repeated_trap_switch_x2_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_nontransfer_smoke/tasks.json",
    "dataset_indices": [
      0,
      7,
      9,
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
      65
    ],
    "repeats": 2,
    "episodes": 20,
    "method": "random_path",
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
      "stage5_n4__from__n0203__c04"
    ],
    "exact_match_mean": 0.1,
    "terminal_penalty_mean": 6.975,
    "raw_outcome_penalty_mean": 5.375,
    "raw_policy_penalty_mean": 1.6,
    "raw_terminal_penalty_mean": 6.975,
    "total_cost_mean": 0.33298366777308386,
    "raw_total_cost_mean": 11.03507875,
    "raw_total_cost_api_mean": 7.51428155,
    "raw_total_cost_token_mean": 11.03507875,
    "reasoning_cost_mean": 3.99099875,
    "raw_reasoning_cost_component_mean": 3.99099875,
    "raw_reasoning_cost_component_api_mean": 0.47020154999999997,
    "raw_reasoning_cost_component_token_mean": 3.99099875,
    "raw_path_cost_component_mean": 0.06908,
    "algorithm_cumulative_total_cost": 6.659673355461678,
    "raw_algorithm_cumulative_total_cost": 220.701575,
    "oracle_stationary_total_cost": 0.9413705491852746,
    "raw_oracle_stationary_total_cost": 31.197020000000002,
    "raw_outcome_penalty_cumulative": 107.5,
    "raw_policy_penalty_cumulative": 32.0,
    "raw_terminal_penalty_cumulative": 139.5,
    "raw_path_cost_component_cumulative": 1.3816000000000002,
    "raw_reasoning_cost_component_cumulative": 79.819975,
    "mean_llm_call_count": 10.5,
    "mean_prompt_tokens": 36027.45,
    "mean_completion_tokens": 1042.05,
    "mean_total_tokens": 37069.5,
    "cumulative_total_tokens": 741390.0,
    "mean_api_cost_usd_raw": 0.0047160675,
    "cumulative_api_cost_usd_raw": 0.09432135,
    "mean_generation_time_seconds": 30.146890570409596,
    "p50_generation_time_seconds": 30.506086301989853,
    "p90_generation_time_seconds": 32.804219991713765,
    "mean_llm_round_trip_seconds": 30.179552642349154,
    "mean_episode_wall_clock_seconds": 33.25846100607887,
    "p50_episode_wall_clock_seconds": 33.95755217038095,
    "p90_episode_wall_clock_seconds": 35.91837396919727,
    "mean_tool_wall_clock_seconds": 0.007772572617977857,
    "policy_action_violation_rate": 0.8,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.8,
    "subset_mismatch_count": 18,
    "episodes_with_stage5_verification_tools": 20,
    "shared_path_fraction": 0.9,
    "unshared_path_fraction": 0.1,
    "mean_barrier_stop_depth": 2.625,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 10,
        "fraction": 0.5
      },
      "stage1": {
        "count": 9,
        "fraction": 0.45
      },
      "stage3": {
        "count": 1,
        "fraction": 0.05
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.3,
      2.85,
      2.95,
      3.05
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.3,
      2.85,
      2.95,
      3.05
    ],
    "specialist_task_count": 9,
    "specialist_task_unshared_fraction": 0.2222222222222222,
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
    "specialist_episode_count": 9,
    "specialist_shared_path_fraction": 0.7777777777777778,
    "specialist_unshared_path_fraction": 0.2222222222222222,
    "specialist_exact_match_mean": 0.0,
    "specialist_total_cost_mean": 0.4149295245758734,
    "specialist_raw_outcome_penalty_mean": 7.333333333333333,
    "specialist_raw_policy_penalty_mean": 2.0,
    "specialist_raw_terminal_penalty_mean": 9.333333333333334,
    "specialist_raw_path_cost_component_mean": 0.07136666666666668,
    "specialist_raw_reasoning_cost_component_mean": 4.346064444444445,
    "specialist_raw_reasoning_cost_component_api_mean": 0.47992,
    "specialist_raw_reasoning_cost_component_token_mean": 4.346064444444445,
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