{
  "summary": {
    "test_name": "direct_multistage_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json",
    "dataset_indices": [
      1,
      2,
      3,
      6,
      9,
      10,
      13,
      15,
      16,
      17,
      18,
      19,
      20,
      21,
      22,
      32,
      33,
      34,
      35,
      36
    ],
    "repeats": 10,
    "episodes": 100,
    "method": "direct_multistage_exp3",
    "mechanism": "algorithm_direct",
    "executor_name": "llm_bench",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch",
    "schedule_mode": "trap_switch",
    "seed": 1,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n5__from__root__c05",
      "stage2_n5__from__n0005__c03",
      "stage3_n5__from__n0022__c03",
      "stage4_n3__from__n0073__c02",
      "stage5_n4__from__n0213__c04"
    ],
    "exact_match_mean": 0.84,
    "terminal_penalty_mean": 2.63,
    "raw_outcome_penalty_mean": 0.59,
    "raw_policy_penalty_mean": 0.48,
    "raw_terminal_penalty_mean": 2.63,
    "legacy_raw_terminal_penalty_mean": 1.07,
    "raw_terminal_penalty_exec_clean_v4_mean": 2.63,
    "total_cost_mean": 0.21524839660743134,
    "raw_total_cost_mean": 7.99432545,
    "raw_total_cost_api_mean": 4.05855097,
    "raw_total_cost_token_mean": 7.99432545,
    "reasoning_cost_mean": 5.29336545,
    "raw_reasoning_cost_component_mean": 5.29336545,
    "raw_mode_mismatch_cost_component_mean": 1.595,
    "raw_reasoning_cost_component_api_mean": 1.35759097,
    "raw_reasoning_cost_component_token_mean": 5.29336545,
    "raw_path_cost_component_mean": 0.07096,
    "algorithm_cumulative_total_cost": 21.524839660743133,
    "raw_algorithm_cumulative_total_cost": 799.432545,
    "oracle_stationary_total_cost": 16.72670463112547,
    "raw_oracle_stationary_total_cost": 621.22981,
    "raw_outcome_penalty_cumulative": 59.0,
    "raw_policy_penalty_cumulative": 48.0,
    "raw_terminal_penalty_cumulative": 263.0,
    "legacy_raw_terminal_penalty_cumulative": 107.0,
    "raw_path_cost_component_cumulative": 7.096,
    "raw_reasoning_cost_component_cumulative": 529.336545,
    "raw_mode_mismatch_cost_component_cumulative": 159.5,
    "mean_llm_call_count": 12.77,
    "mean_prompt_tokens": 55985.02,
    "mean_completion_tokens": 1217.97,
    "mean_total_tokens": 57202.99,
    "cumulative_total_tokens": 5720299.0,
    "mean_api_cost_usd_raw": 0.016032342499999998,
    "cumulative_api_cost_usd_raw": 1.6032342499999999,
    "mean_generation_time_seconds": 58.04892241952941,
    "p50_generation_time_seconds": 45.948481323197484,
    "p90_generation_time_seconds": 104.13256386611614,
    "mean_llm_round_trip_seconds": 58.09246344080195,
    "mean_episode_wall_clock_seconds": 61.00116706555709,
    "p50_episode_wall_clock_seconds": 49.02612320613116,
    "p90_episode_wall_clock_seconds": 106.94864281788473,
    "mean_tool_wall_clock_seconds": 0.011265357565134764,
    "policy_action_violation_rate": 0.24,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.24,
    "subset_mismatch_count": 16,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.93,
    "unshared_path_fraction": 0.07,
    "mean_barrier_stop_depth": 2.735294117647059,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 59,
        "fraction": 0.59
      },
      "stage1": {
        "count": 18,
        "fraction": 0.18
      },
      "stage2": {
        "count": 6,
        "fraction": 0.06
      },
      "stage3": {
        "count": 14,
        "fraction": 0.14
      },
      "stage4": {
        "count": 3,
        "fraction": 0.03
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.6,
      2.99,
      2.5,
      2.78
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.6,
      2.99,
      2.5,
      2.78
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.0625,
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
    "specialist_episode_count": 32,
    "specialist_shared_path_fraction": 0.9375,
    "specialist_unshared_path_fraction": 0.0625,
    "specialist_exact_match_mean": 0.5625,
    "specialist_total_cost_mean": 0.3685933292945611,
    "specialist_raw_outcome_penalty_mean": 1.65625,
    "specialist_raw_policy_penalty_mean": 1.375,
    "specialist_raw_terminal_penalty_mean": 7.40625,
    "specialist_raw_path_cost_component_mean": 0.070628125,
    "specialist_raw_reasoning_cost_component_mean": 6.212678125,
    "specialist_raw_reasoning_cost_component_api_mean": 1.7258362812499999,
    "specialist_raw_reasoning_cost_component_token_mean": 6.212678125,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}