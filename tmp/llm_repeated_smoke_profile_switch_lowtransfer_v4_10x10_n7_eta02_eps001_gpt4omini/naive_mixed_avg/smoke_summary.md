{
  "summary": {
    "test_name": "naive_mixed_avg_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "naive_mixed_avg",
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
    "terminal_penalty_mean": 10.045,
    "raw_outcome_penalty_mean": 8.985,
    "raw_policy_penalty_mean": 1.06,
    "raw_terminal_penalty_mean": 10.045,
    "total_cost_mean": 0.4700317803258901,
    "raw_total_cost_mean": 15.5768532,
    "raw_total_cost_api_mean": 10.63676199,
    "raw_total_cost_token_mean": 15.5768532,
    "reasoning_cost_mean": 5.459089199999999,
    "raw_reasoning_cost_component_mean": 5.459089199999999,
    "raw_reasoning_cost_component_api_mean": 0.51899799,
    "raw_reasoning_cost_component_token_mean": 5.459089199999999,
    "raw_path_cost_component_mean": 0.07276400000000001,
    "algorithm_cumulative_total_cost": 47.00317803258901,
    "raw_algorithm_cumulative_total_cost": 1557.68532,
    "oracle_stationary_total_cost": 6.946650573325287,
    "raw_oracle_stationary_total_cost": 230.21200000000002,
    "raw_outcome_penalty_cumulative": 898.5,
    "raw_policy_penalty_cumulative": 106.0,
    "raw_terminal_penalty_cumulative": 1004.5,
    "raw_path_cost_component_cumulative": 7.276400000000001,
    "raw_reasoning_cost_component_cumulative": 545.90892,
    "mean_llm_call_count": 13.9,
    "mean_prompt_tokens": 52311.68,
    "mean_completion_tokens": 1523.67,
    "mean_total_tokens": 53835.35,
    "cumulative_total_tokens": 5383535.0,
    "mean_api_cost_usd_raw": 0.0055523460000000005,
    "cumulative_api_cost_usd_raw": 0.5552346,
    "mean_generation_time_seconds": 45.55465810578316,
    "p50_generation_time_seconds": 42.646795274689794,
    "p90_generation_time_seconds": 55.75654380582273,
    "mean_llm_round_trip_seconds": 45.601374047771095,
    "mean_episode_wall_clock_seconds": 48.90296441141516,
    "p50_episode_wall_clock_seconds": 46.10533315502107,
    "p90_episode_wall_clock_seconds": 59.244301362521945,
    "mean_tool_wall_clock_seconds": 0.013616450019180775,
    "policy_action_violation_rate": 0.53,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.53,
    "subset_mismatch_count": 80,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.99,
    "unshared_path_fraction": 0.01,
    "mean_barrier_stop_depth": 1.8,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 74,
        "fraction": 0.74
      },
      "stage1": {
        "count": 22,
        "fraction": 0.22
      },
      "stage3": {
        "count": 3,
        "fraction": 0.03
      },
      "stage4": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.06,
      2.61,
      1.94,
      2.3
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.06,
      2.61,
      1.94,
      2.3
    ],
    "specialist_task_count": 78,
    "specialist_task_unshared_fraction": 0.01282051282051282,
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
    "specialist_episode_count": 78,
    "specialist_shared_path_fraction": 0.9871794871794872,
    "specialist_unshared_path_fraction": 0.01282051282051282,
    "specialist_exact_match_mean": 0.1794871794871795,
    "specialist_total_cost_mean": 0.5456756031134425,
    "specialist_raw_outcome_penalty_mean": 11.173076923076923,
    "specialist_raw_policy_penalty_mean": 1.358974358974359,
    "specialist_raw_terminal_penalty_mean": 12.532051282051283,
    "specialist_raw_path_cost_component_mean": 0.07273846153846154,
    "specialist_raw_reasoning_cost_component_mean": 5.478899743589744,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5203995384615384,
    "specialist_raw_reasoning_cost_component_token_mean": 5.478899743589744,
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