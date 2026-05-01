{
  "summary": {
    "test_name": "risky_ps_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v1_full_llm",
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
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v1",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n1__from__root__c01",
      "stage2_n1__from__n0001__c01",
      "stage3_n3__from__n0005__c03",
      "stage4_n3__from__n0017__c03",
      "stage5_n4__from__n0052__c04"
    ],
    "exact_match_mean": 0.83,
    "terminal_penalty_mean": 2.52,
    "raw_outcome_penalty_mean": 0.44,
    "raw_policy_penalty_mean": 0.46,
    "raw_terminal_penalty_mean": 2.52,
    "legacy_raw_terminal_penalty_mean": 0.9,
    "raw_terminal_penalty_exec_clean_v4_mean": 2.52,
    "total_cost_mean": 0.21830973747980614,
    "raw_total_cost_mean": 8.10802365,
    "raw_total_cost_api_mean": 3.75672868,
    "raw_total_cost_token_mean": 8.10802365,
    "reasoning_cost_mean": 5.51808265,
    "raw_reasoning_cost_component_mean": 5.51808265,
    "raw_mode_mismatch_cost_component_mean": 1.395,
    "raw_reasoning_cost_component_api_mean": 1.16678768,
    "raw_reasoning_cost_component_token_mean": 5.51808265,
    "raw_path_cost_component_mean": 0.069941,
    "algorithm_cumulative_total_cost": 21.830973747980615,
    "raw_algorithm_cumulative_total_cost": 810.802365,
    "oracle_stationary_total_cost": 17.508880183091005,
    "raw_oracle_stationary_total_cost": 650.27981,
    "raw_outcome_penalty_cumulative": 44.0,
    "raw_policy_penalty_cumulative": 46.0,
    "raw_terminal_penalty_cumulative": 252.0,
    "legacy_raw_terminal_penalty_cumulative": 90.0,
    "raw_path_cost_component_cumulative": 6.9941,
    "raw_reasoning_cost_component_cumulative": 551.808265,
    "raw_mode_mismatch_cost_component_cumulative": 139.5,
    "mean_llm_call_count": 13.77,
    "mean_prompt_tokens": 58347.26,
    "mean_completion_tokens": 1286.62,
    "mean_total_tokens": 59633.88,
    "cumulative_total_tokens": 5963388.0,
    "mean_api_cost_usd_raw": 0.013805383,
    "cumulative_api_cost_usd_raw": 1.3805383,
    "mean_generation_time_seconds": 87.10126011084765,
    "p50_generation_time_seconds": 52.84439567476511,
    "p90_generation_time_seconds": 172.95784777216616,
    "mean_llm_round_trip_seconds": 87.14444037873298,
    "mean_episode_wall_clock_seconds": 89.71619146795943,
    "p50_episode_wall_clock_seconds": 55.369736853055656,
    "p90_episode_wall_clock_seconds": 175.52046064082535,
    "mean_tool_wall_clock_seconds": 0.012545546125620604,
    "policy_action_violation_rate": 0.23,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.23,
    "subset_mismatch_count": 17,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 1.0,
    "unshared_path_fraction": 0.0,
    "mean_barrier_stop_depth": 0.0,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 100,
        "fraction": 1.0
      }
    },
    "mean_candidate_count_per_stage": [
      4.0,
      2.96,
      3.45,
      3.2,
      3.18
    ],
    "mean_legal_child_count_per_stage": [
      4.0,
      2.96,
      3.45,
      3.2,
      3.18
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
    "specialist_exact_match_mean": 0.5,
    "specialist_total_cost_mean": 0.3694514253500269,
    "specialist_raw_outcome_penalty_mean": 1.28125,
    "specialist_raw_policy_penalty_mean": 1.375,
    "specialist_raw_terminal_penalty_mean": 7.4375,
    "specialist_raw_path_cost_component_mean": 0.07005625,
    "specialist_raw_reasoning_cost_component_mean": 6.2138696875,
    "specialist_raw_reasoning_cost_component_api_mean": 1.40884725,
    "specialist_raw_reasoning_cost_component_token_mean": 6.2138696875,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}