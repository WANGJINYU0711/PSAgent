{
  "summary": {
    "test_name": "risky_ps_old_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_old",
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
    "exact_match_mean": 0.66,
    "terminal_penalty_mean": 5.89,
    "raw_outcome_penalty_mean": 3.61,
    "raw_policy_penalty_mean": 0.48,
    "raw_terminal_penalty_mean": 5.89,
    "legacy_raw_terminal_penalty_mean": 4.09,
    "raw_terminal_penalty_exec_clean_v4_mean": 5.89,
    "total_cost_mean": 0.2802193228325256,
    "raw_total_cost_mean": 10.40734565,
    "raw_total_cost_api_mean": 6.38920888,
    "raw_total_cost_token_mean": 10.40734565,
    "reasoning_cost_mean": 4.446070649999999,
    "raw_reasoning_cost_component_mean": 4.446070649999999,
    "raw_mode_mismatch_cost_component_mean": 1.465,
    "raw_reasoning_cost_component_api_mean": 0.42793388,
    "raw_reasoning_cost_component_token_mean": 4.446070649999999,
    "raw_path_cost_component_mean": 0.071275,
    "algorithm_cumulative_total_cost": 28.021932283252557,
    "raw_algorithm_cumulative_total_cost": 1040.734565,
    "oracle_stationary_total_cost": 20.49191733979537,
    "raw_oracle_stationary_total_cost": 761.0698100000001,
    "raw_outcome_penalty_cumulative": 361.0,
    "raw_policy_penalty_cumulative": 48.0,
    "raw_terminal_penalty_cumulative": 589.0,
    "legacy_raw_terminal_penalty_cumulative": 409.0,
    "raw_path_cost_component_cumulative": 7.1275,
    "raw_reasoning_cost_component_cumulative": 444.607065,
    "raw_mode_mismatch_cost_component_cumulative": 146.5,
    "mean_llm_call_count": 12.58,
    "mean_prompt_tokens": 45092.88,
    "mean_completion_tokens": 1309.74,
    "mean_total_tokens": 46402.62,
    "cumulative_total_tokens": 4640262.0,
    "mean_api_cost_usd_raw": 0.004802351999999999,
    "cumulative_api_cost_usd_raw": 0.4802352,
    "mean_generation_time_seconds": 38.38461485803127,
    "p50_generation_time_seconds": 38.457417896948755,
    "p90_generation_time_seconds": 44.18794975988567,
    "mean_llm_round_trip_seconds": 38.42726842770353,
    "mean_episode_wall_clock_seconds": 41.610303419400005,
    "p50_episode_wall_clock_seconds": 41.83793554548174,
    "p90_episode_wall_clock_seconds": 47.132300592400135,
    "mean_tool_wall_clock_seconds": 0.011914417892694474,
    "policy_action_violation_rate": 0.24,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.24,
    "subset_mismatch_count": 34,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.95,
    "unshared_path_fraction": 0.05,
    "mean_barrier_stop_depth": 2.292682926829268,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 54,
        "fraction": 0.54
      },
      "stage1": {
        "count": 32,
        "fraction": 0.32
      },
      "stage2": {
        "count": 2,
        "fraction": 0.02
      },
      "stage3": {
        "count": 9,
        "fraction": 0.09
      },
      "stage4": {
        "count": 3,
        "fraction": 0.03
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.2,
      3.09,
      2.52,
      2.95
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.2,
      3.09,
      2.52,
      2.95
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
    "specialist_exact_match_mean": 0.21875,
    "specialist_total_cost_mean": 0.5161723377759828,
    "specialist_raw_outcome_penalty_mean": 8.9375,
    "specialist_raw_policy_penalty_mean": 1.5,
    "specialist_raw_terminal_penalty_mean": 14.5625,
    "specialist_raw_path_cost_component_mean": 0.07079375,
    "specialist_raw_reasoning_cost_component_mean": 4.537346875,
    "specialist_raw_reasoning_cost_component_api_mean": 0.432728,
    "specialist_raw_reasoning_cost_component_token_mean": 4.537346875,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}