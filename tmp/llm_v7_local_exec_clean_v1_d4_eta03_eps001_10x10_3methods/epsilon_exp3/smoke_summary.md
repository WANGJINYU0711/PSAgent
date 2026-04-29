{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v1/tasks.json",
    "dataset_indices": [
      1,
      3,
      6,
      9,
      10,
      13,
      15,
      16
    ],
    "repeats": 10,
    "episodes": 40,
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
    "exact_match_mean": 0.4,
    "terminal_penalty_mean": 7.1875,
    "raw_outcome_penalty_mean": 5.9875,
    "raw_policy_penalty_mean": 1.2,
    "raw_terminal_penalty_mean": 7.1875,
    "total_cost_mean": 0.3794314348219674,
    "raw_total_cost_mean": 12.57435775,
    "raw_total_cost_api_mean": 7.7776079,
    "raw_total_cost_token_mean": 12.57435775,
    "reasoning_cost_mean": 5.31587775,
    "raw_reasoning_cost_component_mean": 5.31587775,
    "raw_reasoning_cost_component_api_mean": 0.5191279,
    "raw_reasoning_cost_component_token_mean": 5.31587775,
    "raw_path_cost_component_mean": 0.07098000000000002,
    "algorithm_cumulative_total_cost": 15.177257392878696,
    "raw_algorithm_cumulative_total_cost": 502.97431,
    "oracle_stationary_total_cost": 2.543633071816536,
    "raw_oracle_stationary_total_cost": 84.296,
    "raw_outcome_penalty_cumulative": 239.5,
    "raw_policy_penalty_cumulative": 48.0,
    "raw_terminal_penalty_cumulative": 287.5,
    "raw_path_cost_component_cumulative": 2.8392000000000004,
    "raw_reasoning_cost_component_cumulative": 212.63511,
    "mean_llm_call_count": 12.8,
    "mean_prompt_tokens": 49442.75,
    "mean_completion_tokens": 1459.4,
    "mean_total_tokens": 50902.15,
    "cumulative_total_tokens": 2036086.0,
    "mean_api_cost_usd_raw": 0.0053880525,
    "cumulative_api_cost_usd_raw": 0.2155221,
    "mean_generation_time_seconds": 35.2224795514252,
    "p50_generation_time_seconds": 35.06826335191727,
    "p90_generation_time_seconds": 40.44950089864433,
    "mean_llm_round_trip_seconds": 35.26553306956775,
    "mean_episode_wall_clock_seconds": 38.53961834283545,
    "p50_episode_wall_clock_seconds": 38.748856937512755,
    "p90_episode_wall_clock_seconds": 43.32647393252701,
    "mean_tool_wall_clock_seconds": 0.013165645906701684,
    "policy_action_violation_rate": 0.6,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.6,
    "subset_mismatch_count": 24,
    "episodes_with_stage5_verification_tools": 40,
    "shared_path_fraction": 0.925,
    "unshared_path_fraction": 0.075,
    "mean_barrier_stop_depth": 2.4615384615384617,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 24,
        "fraction": 0.6
      },
      "stage1": {
        "count": 7,
        "fraction": 0.175
      },
      "stage2": {
        "count": 2,
        "fraction": 0.05
      },
      "stage3": {
        "count": 5,
        "fraction": 0.125
      },
      "stage4": {
        "count": 1,
        "fraction": 0.025
      },
      "stage5": {
        "count": 1,
        "fraction": 0.025
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.3,
      3.05,
      2.55,
      2.6
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.3,
      3.05,
      2.55,
      2.6
    ],
    "specialist_task_count": 30,
    "specialist_task_unshared_fraction": 0.1,
    "stage_source_summary": {
      "stage1": {
        "llm_bench": 40
      },
      "stage2": {
        "llm_bench": 40
      },
      "stage3": {
        "llm_bench": 40
      },
      "stage4": {
        "llm_bench": 40
      },
      "stage5": {
        "llm_bench": 40
      }
    },
    "reasoning_cost_mode_default": "token"
  },
  "specialist_summary": {
    "specialist_episode_count": 30,
    "specialist_shared_path_fraction": 0.9,
    "specialist_unshared_path_fraction": 0.1,
    "specialist_exact_match_mean": 0.2,
    "specialist_total_cost_mean": 0.4515910028163347,
    "specialist_raw_outcome_penalty_mean": 7.983333333333333,
    "specialist_raw_policy_penalty_mean": 1.6,
    "specialist_raw_terminal_penalty_mean": 9.583333333333334,
    "specialist_raw_path_cost_component_mean": 0.07128000000000001,
    "specialist_raw_reasoning_cost_component_mean": 5.311112499999999,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5180091,
    "specialist_raw_reasoning_cost_component_token_mean": 5.311112499999999,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}