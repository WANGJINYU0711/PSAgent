{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "exact_match_mean": 0.18,
    "terminal_penalty_mean": 9.695,
    "raw_outcome_penalty_mean": 8.535,
    "raw_policy_penalty_mean": 1.16,
    "raw_terminal_penalty_mean": 9.695,
    "total_cost_mean": 0.4518225558237779,
    "raw_total_cost_mean": 14.973399500000001,
    "raw_total_cost_api_mean": 10.264954809999999,
    "raw_total_cost_token_mean": 14.973399500000001,
    "reasoning_cost_mean": 5.2077515000000005,
    "raw_reasoning_cost_component_mean": 5.2077515000000005,
    "raw_reasoning_cost_component_api_mean": 0.49930681,
    "raw_reasoning_cost_component_token_mean": 5.2077515000000005,
    "raw_path_cost_component_mean": 0.070648,
    "algorithm_cumulative_total_cost": 45.18225558237779,
    "raw_algorithm_cumulative_total_cost": 1497.33995,
    "oracle_stationary_total_cost": 6.946650573325287,
    "raw_oracle_stationary_total_cost": 230.21200000000002,
    "raw_outcome_penalty_cumulative": 853.5,
    "raw_policy_penalty_cumulative": 116.0,
    "raw_terminal_penalty_cumulative": 969.5,
    "raw_path_cost_component_cumulative": 7.064800000000001,
    "raw_reasoning_cost_component_cumulative": 520.77515,
    "mean_llm_call_count": 12.65,
    "mean_prompt_tokens": 49531.16,
    "mean_completion_tokens": 1437.26,
    "mean_total_tokens": 50968.42,
    "cumulative_total_tokens": 5096842.0,
    "mean_api_cost_usd_raw": 0.0052787285,
    "cumulative_api_cost_usd_raw": 0.52787285,
    "mean_generation_time_seconds": 42.03706231512129,
    "p50_generation_time_seconds": 39.33614917751402,
    "p90_generation_time_seconds": 50.1623470954597,
    "mean_llm_round_trip_seconds": 42.08206097951159,
    "mean_episode_wall_clock_seconds": 45.40971732027829,
    "p50_episode_wall_clock_seconds": 42.82032359857112,
    "p90_episode_wall_clock_seconds": 53.82066912036389,
    "mean_tool_wall_clock_seconds": 0.01273399943485856,
    "policy_action_violation_rate": 0.58,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.58,
    "subset_mismatch_count": 82,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.92,
    "unshared_path_fraction": 0.08,
    "mean_barrier_stop_depth": 2.682926829268293,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 51,
        "fraction": 0.51
      },
      "stage1": {
        "count": 21,
        "fraction": 0.21
      },
      "stage2": {
        "count": 7,
        "fraction": 0.07
      },
      "stage3": {
        "count": 13,
        "fraction": 0.13
      },
      "stage4": {
        "count": 6,
        "fraction": 0.06
      },
      "stage5": {
        "count": 2,
        "fraction": 0.02
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.5,
      2.95,
      2.62,
      2.77
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.5,
      2.95,
      2.62,
      2.77
    ],
    "specialist_task_count": 78,
    "specialist_task_unshared_fraction": 0.10256410256410256,
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
    "specialist_shared_path_fraction": 0.8974358974358975,
    "specialist_unshared_path_fraction": 0.10256410256410256,
    "specialist_exact_match_mean": 0.16666666666666666,
    "specialist_total_cost_mean": 0.5234104111539235,
    "specialist_raw_outcome_penalty_mean": 10.551282051282051,
    "specialist_raw_policy_penalty_mean": 1.4102564102564104,
    "specialist_raw_terminal_penalty_mean": 11.961538461538462,
    "specialist_raw_path_cost_component_mean": 0.07105641025641027,
    "specialist_raw_reasoning_cost_component_mean": 5.313226153846154,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5063369871794872,
    "specialist_raw_reasoning_cost_component_token_mean": 5.313226153846154,
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