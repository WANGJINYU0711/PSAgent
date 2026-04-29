{
  "summary": {
    "test_name": "epsilon_exp3_repeated_trap_switch_x2_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json",
    "dataset_indices": [
      7,
      11,
      13,
      16,
      17,
      19,
      25,
      35,
      36,
      40,
      45,
      46,
      51,
      65,
      67
    ],
    "repeats": 2,
    "episodes": 20,
    "method": "epsilon_exp3",
    "mechanism": "algorithm_direct",
    "executor_name": "llm_bench",
    "family_kind": "shared_basin_strong_prefix_dedup_profile_switch",
    "schedule_mode": "trap_switch",
    "seed": 0,
    "model": "gpt-4o-mini",
    "stationary_oracle_path": [
      "stage1_n1__from__root__c01",
      "stage2_n2__from__n0001__c02",
      "stage3_n2__from__n0007__c01",
      "stage4_n1__from__n0025__c01",
      "stage5_n2__from__n0076__c02"
    ],
    "exact_match_mean": 0.45,
    "terminal_penalty_mean": 5.5,
    "raw_outcome_penalty_mean": 4.6,
    "raw_policy_penalty_mean": 0.9,
    "raw_terminal_penalty_mean": 5.5,
    "total_cost_mean": 0.32832076191913095,
    "raw_total_cost_mean": 10.88055005,
    "raw_total_cost_api_mean": 6.1590658,
    "raw_total_cost_token_mean": 10.88055005,
    "reasoning_cost_mean": 5.31040005,
    "raw_reasoning_cost_component_mean": 5.31040005,
    "raw_reasoning_cost_component_api_mean": 0.5889158,
    "raw_reasoning_cost_component_token_mean": 5.31040005,
    "raw_path_cost_component_mean": 0.07015,
    "algorithm_cumulative_total_cost": 6.5664152383826195,
    "raw_algorithm_cumulative_total_cost": 217.611001,
    "oracle_stationary_total_cost": 0.546657242003621,
    "raw_oracle_stationary_total_cost": 18.116221,
    "raw_outcome_penalty_cumulative": 92.0,
    "raw_policy_penalty_cumulative": 18.0,
    "raw_terminal_penalty_cumulative": 110.0,
    "raw_path_cost_component_cumulative": 1.403,
    "raw_reasoning_cost_component_cumulative": 106.208001,
    "mean_llm_call_count": 12.7,
    "mean_prompt_tokens": 50501.0805,
    "mean_completion_tokens": 1424.946,
    "mean_total_tokens": 51926.0265,
    "cumulative_total_tokens": 1038520.53,
    "mean_api_cost_usd_raw": 0.006237335,
    "cumulative_api_cost_usd_raw": 0.1247467,
    "mean_generation_time_seconds": 40.65377025787162,
    "p50_generation_time_seconds": 40.409091000000004,
    "p90_generation_time_seconds": 48.0995818,
    "mean_llm_round_trip_seconds": 40.69477633073828,
    "mean_episode_wall_clock_seconds": 43.4070957931937,
    "p50_episode_wall_clock_seconds": 43.298033000000004,
    "p90_episode_wall_clock_seconds": 51.022906,
    "mean_tool_wall_clock_seconds": 0.011054253456145526,
    "policy_action_violation_rate": 0.45,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.45,
    "subset_mismatch_count": 11,
    "episodes_with_stage5_verification_tools": 20,
    "shared_path_fraction": 1.0,
    "unshared_path_fraction": 0.0,
    "mean_barrier_stop_depth": 2.142857142857143,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 13,
        "fraction": 0.65
      },
      "stage1": {
        "count": 4,
        "fraction": 0.2
      },
      "stage2": {
        "count": 1,
        "fraction": 0.05
      },
      "stage3": {
        "count": 1,
        "fraction": 0.05
      },
      "stage4": {
        "count": 1,
        "fraction": 0.05
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.3,
      3.05,
      2.25,
      2.65
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.3,
      3.05,
      2.25,
      2.65
    ],
    "specialist_task_count": 14,
    "specialist_task_unshared_fraction": 0.0,
    "stage_source_summary": {
      "stage1": {
        "llm_bench": 20
      },
      "stage2": {
        "llm_bench": 20
      },
      "stage3": {
        "llm_bench": 20
      },
      "stage4": {
        "llm_bench": 20
      },
      "stage5": {
        "llm_bench": 20
      }
    },
    "reasoning_cost_mode_default": "token"
  },
  "specialist_summary": {
    "specialist_episode_count": 14,
    "specialist_shared_path_fraction": 1.0,
    "specialist_unshared_path_fraction": 0.0,
    "specialist_exact_match_mean": 0.35714285714285715,
    "specialist_total_cost_mean": 0.3971838305026295,
    "specialist_raw_outcome_penalty_mean": 6.214285714285714,
    "specialist_raw_policy_penalty_mean": 1.2857142857142858,
    "specialist_raw_terminal_penalty_mean": 7.5,
    "specialist_raw_path_cost_component_mean": 0.07100714285714287,
    "specialist_raw_reasoning_cost_component_mean": 5.591665,
    "specialist_raw_reasoning_cost_component_api_mean": 0.6332783571428572,
    "specialist_raw_reasoning_cost_component_token_mean": 5.591665,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]"
    ]
  }
}