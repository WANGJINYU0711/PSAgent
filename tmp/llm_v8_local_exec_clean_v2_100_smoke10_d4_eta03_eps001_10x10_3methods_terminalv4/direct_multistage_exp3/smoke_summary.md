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
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n5__from__root__c05",
      "stage2_n5__from__n0005__c03",
      "stage3_n5__from__n0022__c03",
      "stage4_n3__from__n0073__c02",
      "stage5_n4__from__n0213__c04"
    ],
    "exact_match_mean": 0.63,
    "terminal_penalty_mean": 7.14,
    "raw_outcome_penalty_mean": 4.665,
    "raw_policy_penalty_mean": 0.52,
    "raw_terminal_penalty_mean": 7.14,
    "legacy_raw_terminal_penalty_mean": 5.185,
    "raw_terminal_penalty_exec_clean_v4_mean": 7.14,
    "total_cost_mean": 0.32513332525578886,
    "raw_total_cost_mean": 12.0754517,
    "raw_total_cost_api_mean": 7.71638992,
    "raw_total_cost_token_mean": 12.0754517,
    "reasoning_cost_mean": 4.8643257,
    "raw_reasoning_cost_component_mean": 4.8643257,
    "raw_mode_mismatch_cost_component_mean": 0.0,
    "raw_reasoning_cost_component_api_mean": 0.50526392,
    "raw_reasoning_cost_component_token_mean": 4.8643257,
    "raw_path_cost_component_mean": 0.07112600000000001,
    "algorithm_cumulative_total_cost": 32.51333252557889,
    "raw_algorithm_cumulative_total_cost": 1207.54517,
    "oracle_stationary_total_cost": 20.49191733979537,
    "raw_oracle_stationary_total_cost": 761.0698100000001,
    "raw_outcome_penalty_cumulative": 466.5,
    "raw_policy_penalty_cumulative": 52.0,
    "raw_terminal_penalty_cumulative": 714.0,
    "legacy_raw_terminal_penalty_cumulative": 518.5,
    "raw_path_cost_component_cumulative": 7.1126000000000005,
    "raw_reasoning_cost_component_cumulative": 486.43257,
    "raw_mode_mismatch_cost_component_cumulative": 0.0,
    "mean_llm_call_count": 12.65,
    "mean_prompt_tokens": 45663.1,
    "mean_completion_tokens": 1323.55,
    "mean_total_tokens": 46986.65,
    "cumulative_total_tokens": 4698665.0,
    "mean_api_cost_usd_raw": 0.00525818,
    "cumulative_api_cost_usd_raw": 0.525818,
    "mean_generation_time_seconds": 35.29753709094599,
    "p50_generation_time_seconds": 35.220918592996895,
    "p90_generation_time_seconds": 40.026522553525865,
    "mean_llm_round_trip_seconds": 35.33824384450912,
    "mean_episode_wall_clock_seconds": 38.25578596070409,
    "p50_episode_wall_clock_seconds": 38.194217641837895,
    "p90_episode_wall_clock_seconds": 42.94385291524232,
    "mean_tool_wall_clock_seconds": 0.011329810321331023,
    "policy_action_violation_rate": 0.26,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.26,
    "subset_mismatch_count": 37,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.92,
    "unshared_path_fraction": 0.08,
    "mean_barrier_stop_depth": 2.65625,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 60,
        "fraction": 0.6
      },
      "stage1": {
        "count": 18,
        "fraction": 0.18
      },
      "stage2": {
        "count": 2,
        "fraction": 0.02
      },
      "stage3": {
        "count": 12,
        "fraction": 0.12
      },
      "stage4": {
        "count": 7,
        "fraction": 0.07
      },
      "stage5": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.34,
      2.86,
      2.48,
      2.76
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.34,
      2.86,
      2.48,
      2.76
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.0,
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
    "specialist_shared_path_fraction": 1.0,
    "specialist_unshared_path_fraction": 0.0,
    "specialist_exact_match_mean": 0.15625,
    "specialist_total_cost_mean": 0.6190901824178783,
    "specialist_raw_outcome_penalty_mean": 11.71875,
    "specialist_raw_policy_penalty_mean": 1.4375,
    "specialist_raw_terminal_penalty_mean": 17.5625,
    "specialist_raw_path_cost_component_mean": 0.07160625000000001,
    "specialist_raw_reasoning_cost_component_mean": 5.358903125,
    "specialist_raw_reasoning_cost_component_api_mean": 0.54910675,
    "specialist_raw_reasoning_cost_component_token_mean": 5.358903125,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}