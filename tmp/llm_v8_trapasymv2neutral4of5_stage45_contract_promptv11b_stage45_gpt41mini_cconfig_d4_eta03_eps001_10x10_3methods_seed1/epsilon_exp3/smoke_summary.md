{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5_full_llm",
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
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n3__from__root__c03",
      "stage2_n1__from__n0003__c01",
      "stage3_n2__from__n0011__c02",
      "stage4_n3__from__n0034__c03",
      "stage5_n4__from__n0103__c04"
    ],
    "exact_match_mean": 0.76,
    "terminal_penalty_mean": 3.75,
    "raw_outcome_penalty_mean": 0.8,
    "raw_policy_penalty_mean": 0.66,
    "raw_terminal_penalty_mean": 3.75,
    "legacy_raw_terminal_penalty_mean": 1.46,
    "raw_terminal_penalty_exec_clean_v4_mean": 3.75,
    "total_cost_mean": 0.2486101588583737,
    "raw_total_cost_mean": 9.2333813,
    "raw_total_cost_api_mean": 5.09502634,
    "raw_total_cost_token_mean": 9.2333813,
    "reasoning_cost_mean": 5.4164533,
    "raw_reasoning_cost_component_mean": 5.4164533,
    "raw_mode_mismatch_cost_component_mean": 1.705,
    "raw_reasoning_cost_component_api_mean": 1.2780983399999999,
    "raw_reasoning_cost_component_token_mean": 5.4164533,
    "raw_path_cost_component_mean": 0.066928,
    "algorithm_cumulative_total_cost": 24.86101588583737,
    "raw_algorithm_cumulative_total_cost": 923.33813,
    "oracle_stationary_total_cost": 20.81367285945073,
    "raw_oracle_stationary_total_cost": 773.0198100000001,
    "raw_outcome_penalty_cumulative": 80.0,
    "raw_policy_penalty_cumulative": 66.0,
    "raw_terminal_penalty_cumulative": 375.0,
    "legacy_raw_terminal_penalty_cumulative": 146.0,
    "raw_path_cost_component_cumulative": 6.6928,
    "raw_reasoning_cost_component_cumulative": 541.64533,
    "raw_mode_mismatch_cost_component_cumulative": 170.5,
    "mean_llm_call_count": 13.22,
    "mean_prompt_tokens": 55410.73,
    "mean_completion_tokens": 1212.4,
    "mean_total_tokens": 56623.13,
    "cumulative_total_tokens": 5662313.0,
    "mean_api_cost_usd_raw": 0.0143154915,
    "cumulative_api_cost_usd_raw": 1.43154915,
    "mean_generation_time_seconds": 61.21446707043797,
    "p50_generation_time_seconds": 47.297855604439974,
    "p90_generation_time_seconds": 113.03514612447478,
    "mean_llm_round_trip_seconds": 61.256537179872396,
    "mean_episode_wall_clock_seconds": 63.88279249131679,
    "p50_episode_wall_clock_seconds": 49.89314994215965,
    "p90_episode_wall_clock_seconds": 115.74202408231804,
    "mean_tool_wall_clock_seconds": 0.012291824333369733,
    "policy_action_violation_rate": 0.33,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.33,
    "subset_mismatch_count": 24,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.84,
    "unshared_path_fraction": 0.16,
    "mean_barrier_stop_depth": 2.66,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 34,
        "fraction": 0.34
      },
      "stage1": {
        "count": 23,
        "fraction": 0.23
      },
      "stage2": {
        "count": 26,
        "fraction": 0.26
      },
      "stage3": {
        "count": 9,
        "fraction": 0.09
      },
      "stage4": {
        "count": 4,
        "fraction": 0.04
      },
      "stage5": {
        "count": 4,
        "fraction": 0.04
      }
    },
    "mean_candidate_count_per_stage": [
      4.0,
      2.7,
      2.85,
      2.9,
      2.71
    ],
    "mean_legal_child_count_per_stage": [
      4.0,
      2.7,
      2.85,
      2.9,
      2.71
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.1875,
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
    "specialist_shared_path_fraction": 0.8125,
    "specialist_unshared_path_fraction": 0.1875,
    "specialist_exact_match_mean": 0.5625,
    "specialist_total_cost_mean": 0.3698844153877221,
    "specialist_raw_outcome_penalty_mean": 1.3125,
    "specialist_raw_policy_penalty_mean": 1.4375,
    "specialist_raw_terminal_penalty_mean": 7.40625,
    "specialist_raw_path_cost_component_mean": 0.068065625,
    "specialist_raw_reasoning_cost_component_mean": 6.2631915625,
    "specialist_raw_reasoning_cost_component_api_mean": 1.53188228125,
    "specialist_raw_reasoning_cost_component_token_mean": 6.2631915625,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}