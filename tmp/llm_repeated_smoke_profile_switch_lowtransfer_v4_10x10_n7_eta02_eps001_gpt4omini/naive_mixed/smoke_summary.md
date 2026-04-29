{
  "summary": {
    "test_name": "naive_mixed_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json",
    "dataset_indices": [
      0,
      7,
      11,
      13,
      16,
      17,
      18,
      19,
      23,
      24,
      25,
      35,
      36,
      40,
      45,
      46,
      51,
      61,
      65,
      67
    ],
    "repeats": 10,
    "episodes": 100,
    "method": "naive_mixed",
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
    "exact_match_mean": 0.16,
    "terminal_penalty_mean": 10.205,
    "raw_outcome_penalty_mean": 8.965,
    "raw_policy_penalty_mean": 1.24,
    "raw_terminal_penalty_mean": 10.205,
    "total_cost_mean": 0.47189945835847913,
    "raw_total_cost_mean": 15.638748049999998,
    "raw_total_cost_api_mean": 10.771901680000001,
    "raw_total_cost_token_mean": 15.638748049999998,
    "reasoning_cost_mean": 5.36304405,
    "raw_reasoning_cost_component_mean": 5.36304405,
    "raw_reasoning_cost_component_api_mean": 0.49619768000000003,
    "raw_reasoning_cost_component_token_mean": 5.36304405,
    "raw_path_cost_component_mean": 0.070704,
    "algorithm_cumulative_total_cost": 47.189945835847915,
    "raw_algorithm_cumulative_total_cost": 1563.874805,
    "oracle_stationary_total_cost": 6.946650573325287,
    "raw_oracle_stationary_total_cost": 230.21200000000002,
    "raw_outcome_penalty_cumulative": 896.5,
    "raw_policy_penalty_cumulative": 124.0,
    "raw_terminal_penalty_cumulative": 1020.5,
    "raw_path_cost_component_cumulative": 7.0704,
    "raw_reasoning_cost_component_cumulative": 536.304405,
    "mean_llm_call_count": 13.15,
    "mean_prompt_tokens": 50714.35,
    "mean_completion_tokens": 1466.44,
    "mean_total_tokens": 52180.79,
    "cumulative_total_tokens": 5218079.0,
    "mean_api_cost_usd_raw": 0.005229736499999999,
    "cumulative_api_cost_usd_raw": 0.52297365,
    "mean_generation_time_seconds": 42.635321589242665,
    "p50_generation_time_seconds": 40.637412674725056,
    "p90_generation_time_seconds": 52.365752607211476,
    "mean_llm_round_trip_seconds": 42.681037620790306,
    "mean_episode_wall_clock_seconds": 45.950752655025575,
    "p50_episode_wall_clock_seconds": 44.11783457081765,
    "p90_episode_wall_clock_seconds": 55.284799136780215,
    "mean_tool_wall_clock_seconds": 0.013656167779117823,
    "policy_action_violation_rate": 0.62,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.62,
    "subset_mismatch_count": 84,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.98,
    "unshared_path_fraction": 0.02,
    "mean_barrier_stop_depth": 2.6,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 63,
        "fraction": 0.63
      },
      "stage1": {
        "count": 11,
        "fraction": 0.11
      },
      "stage2": {
        "count": 5,
        "fraction": 0.05
      },
      "stage3": {
        "count": 20,
        "fraction": 0.2
      },
      "stage4": {
        "count": 1,
        "fraction": 0.01
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.42,
      2.86,
      2.59,
      2.53
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.42,
      2.86,
      2.59,
      2.53
    ],
    "specialist_task_count": 78,
    "specialist_task_unshared_fraction": 0.02564102564102564,
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
    "specialist_episode_count": 78,
    "specialist_shared_path_fraction": 0.9743589743589743,
    "specialist_unshared_path_fraction": 0.02564102564102564,
    "specialist_exact_match_mean": 0.14102564102564102,
    "specialist_total_cost_mean": 0.5475953917335933,
    "specialist_raw_outcome_penalty_mean": 11.166666666666666,
    "specialist_raw_policy_penalty_mean": 1.5128205128205128,
    "specialist_raw_terminal_penalty_mean": 12.679487179487179,
    "specialist_raw_path_cost_component_mean": 0.07022692307692309,
    "specialist_raw_reasoning_cost_component_mean": 5.39759717948718,
    "specialist_raw_reasoning_cost_component_api_mean": 0.49827707692307693,
    "specialist_raw_reasoning_cost_component_token_mean": 5.39759717948718,
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