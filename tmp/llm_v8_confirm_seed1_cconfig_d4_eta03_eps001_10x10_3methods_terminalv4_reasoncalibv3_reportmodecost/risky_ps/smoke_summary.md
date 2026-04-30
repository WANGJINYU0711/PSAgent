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
    "exact_match_mean": 0.65,
    "terminal_penalty_mean": 5.985,
    "raw_outcome_penalty_mean": 3.345,
    "raw_policy_penalty_mean": 0.6,
    "raw_terminal_penalty_mean": 5.985,
    "legacy_raw_terminal_penalty_mean": 3.945,
    "raw_terminal_penalty_exec_clean_v4_mean": 5.985,
    "total_cost_mean": 0.2828183912224017,
    "raw_total_cost_mean": 10.50387505,
    "raw_total_cost_api_mean": 6.55829443,
    "raw_total_cost_token_mean": 10.50387505,
    "reasoning_cost_mean": 4.4491010499999994,
    "raw_reasoning_cost_component_mean": 4.4491010499999994,
    "raw_mode_mismatch_cost_component_mean": 1.665,
    "raw_reasoning_cost_component_api_mean": 0.50352043,
    "raw_reasoning_cost_component_token_mean": 4.4491010499999994,
    "raw_path_cost_component_mean": 0.069774,
    "algorithm_cumulative_total_cost": 28.28183912224017,
    "raw_algorithm_cumulative_total_cost": 1050.387505,
    "oracle_stationary_total_cost": 16.72670463112547,
    "raw_oracle_stationary_total_cost": 621.22981,
    "raw_outcome_penalty_cumulative": 334.5,
    "raw_policy_penalty_cumulative": 60.0,
    "raw_terminal_penalty_cumulative": 598.5,
    "legacy_raw_terminal_penalty_cumulative": 394.5,
    "raw_path_cost_component_cumulative": 6.9774,
    "raw_reasoning_cost_component_cumulative": 444.910105,
    "raw_mode_mismatch_cost_component_cumulative": 166.5,
    "mean_llm_call_count": 12.2,
    "mean_prompt_tokens": 44156.76,
    "mean_completion_tokens": 1284.56,
    "mean_total_tokens": 45441.32,
    "cumulative_total_tokens": 4544132.0,
    "mean_api_cost_usd_raw": 0.005524074,
    "cumulative_api_cost_usd_raw": 0.5524074,
    "mean_generation_time_seconds": 38.22119025841355,
    "p50_generation_time_seconds": 37.01965196430683,
    "p90_generation_time_seconds": 46.6166033025831,
    "mean_llm_round_trip_seconds": 38.261437864545734,
    "mean_episode_wall_clock_seconds": 40.92889930807054,
    "p50_episode_wall_clock_seconds": 39.75198673643172,
    "p90_episode_wall_clock_seconds": 49.13131626155228,
    "mean_tool_wall_clock_seconds": 0.011352266445755958,
    "policy_action_violation_rate": 0.3,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.3,
    "subset_mismatch_count": 35,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.97,
    "unshared_path_fraction": 0.03,
    "mean_barrier_stop_depth": 2.6,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 62,
        "fraction": 0.62
      },
      "stage1": {
        "count": 17,
        "fraction": 0.17
      },
      "stage2": {
        "count": 5,
        "fraction": 0.05
      },
      "stage3": {
        "count": 13,
        "fraction": 0.13
      },
      "stage4": {
        "count": 3,
        "fraction": 0.03
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.62,
      2.96,
      2.54,
      2.86
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.62,
      2.96,
      2.54,
      2.86
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
    "specialist_exact_match_mean": 0.1875,
    "specialist_total_cost_mean": 0.5483294838785676,
    "specialist_raw_outcome_penalty_mean": 9.109375,
    "specialist_raw_policy_penalty_mean": 1.6875,
    "specialist_raw_terminal_penalty_mean": 15.515625,
    "specialist_raw_path_cost_component_mean": 0.0701375,
    "specialist_raw_reasoning_cost_component_mean": 4.77919453125,
    "specialist_raw_reasoning_cost_component_api_mean": 0.54937809375,
    "specialist_raw_reasoning_cost_component_token_mean": 4.77919453125,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}