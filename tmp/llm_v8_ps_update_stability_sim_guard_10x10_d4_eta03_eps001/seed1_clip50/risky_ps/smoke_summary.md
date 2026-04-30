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
    "executor_name": "simulated",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch",
    "schedule_mode": "trap_switch",
    "seed": 1,
    "model": "simulated",
    "stationary_oracle_path": [
      "stage1_n5__from__root__c05",
      "stage2_n5__from__n0005__c03",
      "stage3_n5__from__n0022__c03",
      "stage4_n3__from__n0073__c02",
      "stage5_n4__from__n0213__c04"
    ],
    "exact_match_mean": 0.16,
    "terminal_penalty_mean": 3.42,
    "raw_outcome_penalty_mean": 3.42,
    "raw_policy_penalty_mean": 0.0,
    "raw_terminal_penalty_mean": 3.42,
    "legacy_raw_terminal_penalty_mean": 3.42,
    "raw_terminal_penalty_exec_clean_v4_mean": 0.0,
    "total_cost_mean": 0.10971183464091733,
    "raw_total_cost_mean": 3.6358501999999997,
    "raw_total_cost_api_mean": 0.0,
    "raw_total_cost_token_mean": 0.0,
    "reasoning_cost_mean": 0.14460119999999999,
    "raw_reasoning_cost_component_mean": 0.14460119999999999,
    "raw_mode_mismatch_cost_component_mean": 0.0,
    "raw_reasoning_cost_component_api_mean": 0.0,
    "raw_reasoning_cost_component_token_mean": 0.0,
    "raw_path_cost_component_mean": 0.071249,
    "algorithm_cumulative_total_cost": 10.971183464091732,
    "raw_algorithm_cumulative_total_cost": 363.58502,
    "oracle_stationary_total_cost": 3.658111345805673,
    "raw_oracle_stationary_total_cost": 121.22981,
    "raw_outcome_penalty_cumulative": 342.0,
    "raw_policy_penalty_cumulative": 0.0,
    "raw_terminal_penalty_cumulative": 342.0,
    "legacy_raw_terminal_penalty_cumulative": 342.0,
    "raw_path_cost_component_cumulative": 7.1249,
    "raw_reasoning_cost_component_cumulative": 14.46012,
    "raw_mode_mismatch_cost_component_cumulative": 0.0,
    "mean_llm_call_count": 0.0,
    "mean_prompt_tokens": 0.0,
    "mean_completion_tokens": 0.0,
    "mean_total_tokens": 0.0,
    "cumulative_total_tokens": 0.0,
    "mean_api_cost_usd_raw": 0.0,
    "cumulative_api_cost_usd_raw": 0.0,
    "mean_generation_time_seconds": 0.0,
    "p50_generation_time_seconds": 0.0,
    "p90_generation_time_seconds": 0.0,
    "mean_llm_round_trip_seconds": 0.0,
    "mean_episode_wall_clock_seconds": 0.0,
    "p50_episode_wall_clock_seconds": 0.0,
    "p90_episode_wall_clock_seconds": 0.0,
    "mean_tool_wall_clock_seconds": 0.0,
    "policy_action_violation_rate": 0.0,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.0,
    "subset_mismatch_count": 84,
    "episodes_with_stage5_verification_tools": 0,
    "shared_path_fraction": 0.92,
    "unshared_path_fraction": 0.08,
    "mean_barrier_stop_depth": 2.725,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 52,
        "fraction": 0.52
      },
      "stage1": {
        "count": 27,
        "fraction": 0.27
      },
      "stage2": {
        "count": 3,
        "fraction": 0.03
      },
      "stage3": {
        "count": 14,
        "fraction": 0.14
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
      3.3,
      2.98,
      2.7,
      2.98
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.3,
      2.98,
      2.7,
      2.98
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.09375,
    "stage_source_summary": {
      "stage1": {
        "None": 100
      },
      "stage2": {
        "None": 100
      },
      "stage3": {
        "None": 100
      },
      "stage4": {
        "None": 100
      },
      "stage5": {
        "None": 100
      }
    },
    "reasoning_cost_mode_default": "simulated_proxy"
  },
  "specialist_summary": {
    "specialist_episode_count": 32,
    "specialist_shared_path_fraction": 0.90625,
    "specialist_unshared_path_fraction": 0.09375,
    "specialist_exact_match_mean": 0.03125,
    "specialist_total_cost_mean": 0.13234067592033796,
    "specialist_raw_outcome_penalty_mean": 4.171875,
    "specialist_raw_policy_penalty_mean": 0.0,
    "specialist_raw_terminal_penalty_mean": 4.171875,
    "specialist_raw_path_cost_component_mean": 0.0719,
    "specialist_raw_reasoning_cost_component_mean": 0.141995,
    "specialist_raw_reasoning_cost_component_api_mean": 0.0,
    "specialist_raw_reasoning_cost_component_token_mean": 0.0,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}