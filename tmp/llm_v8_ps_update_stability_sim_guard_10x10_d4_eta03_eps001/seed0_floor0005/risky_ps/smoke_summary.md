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
    "seed": 0,
    "model": "simulated",
    "stationary_oracle_path": [
      "stage1_n5__from__root__c05",
      "stage2_n5__from__n0005__c03",
      "stage3_n5__from__n0022__c03",
      "stage4_n3__from__n0073__c02",
      "stage5_n4__from__n0213__c04"
    ],
    "exact_match_mean": 0.17,
    "terminal_penalty_mean": 3.33,
    "raw_outcome_penalty_mean": 3.33,
    "raw_policy_penalty_mean": 0.0,
    "raw_terminal_penalty_mean": 3.33,
    "legacy_raw_terminal_penalty_mean": 3.33,
    "raw_terminal_penalty_exec_clean_v4_mean": 0.0,
    "total_cost_mean": 0.10697102896801448,
    "raw_total_cost_mean": 3.5450199,
    "raw_total_cost_api_mean": 0.0,
    "raw_total_cost_token_mean": 0.0,
    "reasoning_cost_mean": 0.1444559,
    "raw_reasoning_cost_component_mean": 0.1444559,
    "raw_mode_mismatch_cost_component_mean": 0.0,
    "raw_reasoning_cost_component_api_mean": 0.0,
    "raw_reasoning_cost_component_token_mean": 0.0,
    "raw_path_cost_component_mean": 0.07056400000000002,
    "algorithm_cumulative_total_cost": 10.697102896801448,
    "raw_algorithm_cumulative_total_cost": 354.50199,
    "oracle_stationary_total_cost": 4.075733554616777,
    "raw_oracle_stationary_total_cost": 135.06981000000002,
    "raw_outcome_penalty_cumulative": 333.0,
    "raw_policy_penalty_cumulative": 0.0,
    "raw_terminal_penalty_cumulative": 333.0,
    "legacy_raw_terminal_penalty_cumulative": 333.0,
    "raw_path_cost_component_cumulative": 7.056400000000001,
    "raw_reasoning_cost_component_cumulative": 14.44559,
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
    "subset_mismatch_count": 83,
    "episodes_with_stage5_verification_tools": 0,
    "shared_path_fraction": 0.91,
    "unshared_path_fraction": 0.09,
    "mean_barrier_stop_depth": 2.8157894736842106,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 53,
        "fraction": 0.53
      },
      "stage1": {
        "count": 16,
        "fraction": 0.16
      },
      "stage2": {
        "count": 7,
        "fraction": 0.07
      },
      "stage3": {
        "count": 20,
        "fraction": 0.2
      },
      "stage4": {
        "count": 3,
        "fraction": 0.03
      },
      "stage5": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.56,
      3.03,
      2.69,
      2.92
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.56,
      3.03,
      2.69,
      2.92
    ],
    "specialist_task_count": 32,
    "specialist_task_unshared_fraction": 0.0,
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
    "specialist_shared_path_fraction": 1.0,
    "specialist_unshared_path_fraction": 0.0,
    "specialist_exact_match_mean": 0.0,
    "specialist_total_cost_mean": 0.1634100030175015,
    "specialist_raw_outcome_penalty_mean": 5.203125,
    "specialist_raw_policy_penalty_mean": 0.0,
    "specialist_raw_terminal_penalty_mean": 5.203125,
    "specialist_raw_path_cost_component_mean": 0.070503125,
    "specialist_raw_reasoning_cost_component_mean": 0.141779375,
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