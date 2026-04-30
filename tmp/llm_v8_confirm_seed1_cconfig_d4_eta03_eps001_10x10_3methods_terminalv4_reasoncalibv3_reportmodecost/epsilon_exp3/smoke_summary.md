{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "epsilon_exp3",
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
    "exact_match_mean": 0.72,
    "terminal_penalty_mean": 5.03,
    "raw_outcome_penalty_mean": 2.645,
    "raw_policy_penalty_mean": 0.5,
    "raw_terminal_penalty_mean": 5.03,
    "legacy_raw_terminal_penalty_mean": 3.145,
    "raw_terminal_penalty_exec_clean_v4_mean": 5.03,
    "total_cost_mean": 0.25881379913839525,
    "raw_total_cost_mean": 9.6123445,
    "raw_total_cost_api_mean": 5.5767529,
    "raw_total_cost_token_mean": 9.6123445,
    "reasoning_cost_mean": 4.5116715,
    "raw_reasoning_cost_component_mean": 4.5116715,
    "raw_mode_mismatch_cost_component_mean": 1.49,
    "raw_reasoning_cost_component_api_mean": 0.4760799,
    "raw_reasoning_cost_component_token_mean": 4.5116715,
    "raw_path_cost_component_mean": 0.070673,
    "algorithm_cumulative_total_cost": 25.881379913839528,
    "raw_algorithm_cumulative_total_cost": 961.23445,
    "oracle_stationary_total_cost": 16.72670463112547,
    "raw_oracle_stationary_total_cost": 621.22981,
    "raw_outcome_penalty_cumulative": 264.5,
    "raw_policy_penalty_cumulative": 50.0,
    "raw_terminal_penalty_cumulative": 503.0,
    "legacy_raw_terminal_penalty_cumulative": 314.5,
    "raw_path_cost_component_cumulative": 7.0673,
    "raw_reasoning_cost_component_cumulative": 451.16715,
    "raw_mode_mismatch_cost_component_cumulative": 149.0,
    "mean_llm_call_count": 12.45,
    "mean_prompt_tokens": 45610.75,
    "mean_completion_tokens": 1316.6,
    "mean_total_tokens": 46927.35,
    "cumulative_total_tokens": 4692735.0,
    "mean_api_cost_usd_raw": 0.0053354445,
    "cumulative_api_cost_usd_raw": 0.53354445,
    "mean_generation_time_seconds": 38.91219589950517,
    "p50_generation_time_seconds": 38.3687307741493,
    "p90_generation_time_seconds": 46.65482952389866,
    "mean_llm_round_trip_seconds": 38.95247030105442,
    "mean_episode_wall_clock_seconds": 41.62292250066996,
    "p50_episode_wall_clock_seconds": 41.05697301682085,
    "p90_episode_wall_clock_seconds": 49.264564597420396,
    "mean_tool_wall_clock_seconds": 0.011680777184665203,
    "policy_action_violation_rate": 0.25,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.25,
    "subset_mismatch_count": 28,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.93,
    "unshared_path_fraction": 0.07,
    "mean_barrier_stop_depth": 2.857142857142857,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 65,
        "fraction": 0.65
      },
      "stage1": {
        "count": 12,
        "fraction": 0.12
      },
      "stage2": {
        "count": 3,
        "fraction": 0.03
      },
      "stage3": {
        "count": 15,
        "fraction": 0.15
      },
      "stage4": {
        "count": 5,
        "fraction": 0.05
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.8,
      2.88,
      2.49,
      2.68
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.8,
      2.88,
      2.49,
      2.68
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
    "specialist_exact_match_mean": 0.375,
    "specialist_total_cost_mean": 0.46861300989499194,
    "specialist_raw_outcome_penalty_mean": 6.546875,
    "specialist_raw_policy_penalty_mean": 1.5,
    "specialist_raw_terminal_penalty_mean": 12.640625,
    "specialist_raw_path_cost_component_mean": 0.07159375,
    "specialist_raw_reasoning_cost_component_mean": 4.6920684375,
    "specialist_raw_reasoning_cost_component_api_mean": 0.51342671875,
    "specialist_raw_reasoning_cost_component_token_mean": 4.6920684375,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}