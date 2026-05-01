{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v1_full_llm",
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
    "exact_match_mean": 0.74,
    "terminal_penalty_mean": 4.0,
    "raw_outcome_penalty_mean": 1.39,
    "raw_policy_penalty_mean": 0.64,
    "raw_terminal_penalty_mean": 4.0,
    "legacy_raw_terminal_penalty_mean": 2.03,
    "raw_terminal_penalty_exec_clean_v4_mean": 4.0,
    "total_cost_mean": 0.2566277207862143,
    "raw_total_cost_mean": 9.53115355,
    "raw_total_cost_api_mean": 5.2532371399999995,
    "raw_total_cost_token_mean": 9.53115355,
    "reasoning_cost_mean": 5.46591355,
    "raw_reasoning_cost_component_mean": 5.46591355,
    "raw_mode_mismatch_cost_component_mean": 2.245,
    "raw_reasoning_cost_component_api_mean": 1.18799714,
    "raw_reasoning_cost_component_token_mean": 5.46591355,
    "raw_path_cost_component_mean": 0.06524,
    "algorithm_cumulative_total_cost": 25.662772078621433,
    "raw_algorithm_cumulative_total_cost": 953.115355,
    "oracle_stationary_total_cost": 17.508880183091005,
    "raw_oracle_stationary_total_cost": 650.27981,
    "raw_outcome_penalty_cumulative": 139.0,
    "raw_policy_penalty_cumulative": 64.0,
    "raw_terminal_penalty_cumulative": 400.0,
    "legacy_raw_terminal_penalty_cumulative": 203.0,
    "raw_path_cost_component_cumulative": 6.524000000000001,
    "raw_reasoning_cost_component_cumulative": 546.591355,
    "raw_mode_mismatch_cost_component_cumulative": 224.5,
    "mean_llm_call_count": 12.55,
    "mean_prompt_tokens": 53375.86,
    "mean_completion_tokens": 1158.59,
    "mean_total_tokens": 54534.45,
    "cumulative_total_tokens": 5453445.0,
    "mean_api_cost_usd_raw": 0.012508242999999999,
    "cumulative_api_cost_usd_raw": 1.2508243,
    "mean_generation_time_seconds": 77.60437505830079,
    "p50_generation_time_seconds": 57.641682118177414,
    "p90_generation_time_seconds": 151.3430531706661,
    "mean_llm_round_trip_seconds": 77.64473844442517,
    "mean_episode_wall_clock_seconds": 80.17372467411683,
    "p50_episode_wall_clock_seconds": 60.39105499815196,
    "p90_episode_wall_clock_seconds": 153.8166598957032,
    "mean_tool_wall_clock_seconds": 0.011491625811904669,
    "policy_action_violation_rate": 0.32,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.32,
    "subset_mismatch_count": 26,
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
      2.52,
      2.65,
      2.72,
      2.58
    ],
    "mean_legal_child_count_per_stage": [
      4.0,
      2.52,
      2.65,
      2.72,
      2.58
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
    "specialist_exact_match_mean": 0.4375,
    "specialist_total_cost_mean": 0.4289240163906839,
    "specialist_raw_outcome_penalty_mean": 3.15625,
    "specialist_raw_policy_penalty_mean": 1.5,
    "specialist_raw_terminal_penalty_mean": 9.0625,
    "specialist_raw_path_cost_component_mean": 0.066990625,
    "specialist_raw_reasoning_cost_component_mean": 6.80074734375,
    "specialist_raw_reasoning_cost_component_api_mean": 1.4640225,
    "specialist_raw_reasoning_cost_component_token_mean": 6.80074734375,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}