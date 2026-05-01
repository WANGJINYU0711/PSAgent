{
  "summary": {
    "test_name": "risky_ps_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps",
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
    "exact_match_mean": 0.82,
    "terminal_penalty_mean": 2.95,
    "raw_outcome_penalty_mean": 0.505,
    "raw_policy_penalty_mean": 0.56,
    "raw_terminal_penalty_mean": 2.95,
    "legacy_raw_terminal_penalty_mean": 1.065,
    "raw_terminal_penalty_exec_clean_v4_mean": 2.95,
    "total_cost_mean": 0.21967048599892297,
    "raw_total_cost_mean": 8.15856185,
    "raw_total_cost_api_mean": 4.21270873,
    "raw_total_cost_token_mean": 8.15856185,
    "reasoning_cost_mean": 5.13712285,
    "raw_reasoning_cost_component_mean": 5.13712285,
    "raw_mode_mismatch_cost_component_mean": 1.345,
    "raw_reasoning_cost_component_api_mean": 1.1912697300000001,
    "raw_reasoning_cost_component_token_mean": 5.13712285,
    "raw_path_cost_component_mean": 0.071439,
    "algorithm_cumulative_total_cost": 21.967048599892298,
    "raw_algorithm_cumulative_total_cost": 815.856185,
    "oracle_stationary_total_cost": 16.72670463112547,
    "raw_oracle_stationary_total_cost": 621.22981,
    "raw_outcome_penalty_cumulative": 50.5,
    "raw_policy_penalty_cumulative": 56.0,
    "raw_terminal_penalty_cumulative": 295.0,
    "legacy_raw_terminal_penalty_cumulative": 106.5,
    "raw_path_cost_component_cumulative": 7.1439,
    "raw_reasoning_cost_component_cumulative": 513.712285,
    "raw_mode_mismatch_cost_component_cumulative": 134.5,
    "mean_llm_call_count": 12.85,
    "mean_prompt_tokens": 55594.21,
    "mean_completion_tokens": 1242.5,
    "mean_total_tokens": 56836.71,
    "cumulative_total_tokens": 5683671.0,
    "mean_api_cost_usd_raw": 0.014464877499999999,
    "cumulative_api_cost_usd_raw": 1.44648775,
    "mean_generation_time_seconds": 60.11722071960568,
    "p50_generation_time_seconds": 47.01573165785521,
    "p90_generation_time_seconds": 107.01204560864717,
    "mean_llm_round_trip_seconds": 60.16127396190539,
    "mean_episode_wall_clock_seconds": 63.06253335621208,
    "p50_episode_wall_clock_seconds": 49.865970375947654,
    "p90_episode_wall_clock_seconds": 110.12743340916931,
    "mean_tool_wall_clock_seconds": 0.011474493741989136,
    "policy_action_violation_rate": 0.28,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.28,
    "subset_mismatch_count": 18,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.96,
    "unshared_path_fraction": 0.04,
    "mean_barrier_stop_depth": 2.586206896551724,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 67,
        "fraction": 0.67
      },
      "stage1": {
        "count": 16,
        "fraction": 0.16
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
        "count": 3,
        "fraction": 0.03
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.2,
      2.79,
      2.26,
      2.75
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.2,
      2.79,
      2.26,
      2.75
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.09375,
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
    "specialist_shared_path_fraction": 0.90625,
    "specialist_unshared_path_fraction": 0.09375,
    "specialist_exact_match_mean": 0.5625,
    "specialist_total_cost_mean": 0.3693869017568659,
    "specialist_raw_outcome_penalty_mean": 0.9375,
    "specialist_raw_policy_penalty_mean": 1.5625,
    "specialist_raw_terminal_penalty_mean": 7.65625,
    "specialist_raw_path_cost_component_mean": 0.071990625,
    "specialist_raw_reasoning_cost_component_mean": 5.99078890625,
    "specialist_raw_reasoning_cost_component_api_mean": 1.49799915625,
    "specialist_raw_reasoning_cost_component_token_mean": 5.99078890625,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}