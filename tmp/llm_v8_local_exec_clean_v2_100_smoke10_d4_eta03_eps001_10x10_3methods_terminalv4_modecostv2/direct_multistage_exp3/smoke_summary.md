{
  "summary": {
    "test_name": "direct_multistage_exp3_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "direct_multistage_exp3",
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
    "exact_match_mean": 0.7,
    "terminal_penalty_mean": 5.81,
    "raw_outcome_penalty_mean": 3.305,
    "raw_policy_penalty_mean": 0.56,
    "raw_terminal_penalty_mean": 5.81,
    "legacy_raw_terminal_penalty_mean": 3.865,
    "raw_terminal_penalty_exec_clean_v4_mean": 5.81,
    "total_cost_mean": 0.3259097913301023,
    "raw_total_cost_mean": 12.10428965,
    "raw_total_cost_api_mean": 7.73206073,
    "raw_total_cost_token_mean": 12.10428965,
    "reasoning_cost_mean": 6.22298465,
    "raw_reasoning_cost_component_mean": 6.22298465,
    "raw_mode_mismatch_cost_component_mean": 1.38,
    "raw_reasoning_cost_component_api_mean": 1.85075573,
    "raw_reasoning_cost_component_token_mean": 6.22298465,
    "raw_path_cost_component_mean": 0.07130500000000001,
    "algorithm_cumulative_total_cost": 32.59097913301023,
    "raw_algorithm_cumulative_total_cost": 1210.428965,
    "oracle_stationary_total_cost": 20.49191733979537,
    "raw_oracle_stationary_total_cost": 761.0698100000001,
    "raw_outcome_penalty_cumulative": 330.5,
    "raw_policy_penalty_cumulative": 56.0,
    "raw_terminal_penalty_cumulative": 581.0,
    "legacy_raw_terminal_penalty_cumulative": 386.5,
    "raw_path_cost_component_cumulative": 7.1305000000000005,
    "raw_reasoning_cost_component_cumulative": 622.298465,
    "raw_mode_mismatch_cost_component_cumulative": 138.0,
    "mean_llm_call_count": 12.54,
    "mean_prompt_tokens": 45447.05,
    "mean_completion_tokens": 1320.58,
    "mean_total_tokens": 46767.63,
    "cumulative_total_tokens": 4676763.0,
    "mean_api_cost_usd_raw": 0.0049003815,
    "cumulative_api_cost_usd_raw": 0.49003815,
    "mean_generation_time_seconds": 34.90980449974537,
    "p50_generation_time_seconds": 34.09459034912288,
    "p90_generation_time_seconds": 40.963652287982406,
    "mean_llm_round_trip_seconds": 34.9515522518754,
    "mean_episode_wall_clock_seconds": 38.014345619063825,
    "p50_episode_wall_clock_seconds": 37.19303463958204,
    "p90_episode_wall_clock_seconds": 44.19276649262756,
    "mean_tool_wall_clock_seconds": 0.011429771948605776,
    "policy_action_violation_rate": 0.28,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.28,
    "subset_mismatch_count": 30,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.91,
    "unshared_path_fraction": 0.09,
    "mean_barrier_stop_depth": 2.7096774193548385,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 60,
        "fraction": 0.6
      },
      "stage1": {
        "count": 18,
        "fraction": 0.18
      },
      "stage2": {
        "count": 4,
        "fraction": 0.04
      },
      "stage3": {
        "count": 12,
        "fraction": 0.12
      },
      "stage4": {
        "count": 6,
        "fraction": 0.06
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.56,
      2.86,
      2.4,
      2.72
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.56,
      2.86,
      2.4,
      2.72
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
    "specialist_exact_match_mean": 0.34375,
    "specialist_total_cost_mean": 0.5639450937331718,
    "specialist_raw_outcome_penalty_mean": 8.875,
    "specialist_raw_policy_penalty_mean": 1.4375,
    "specialist_raw_terminal_penalty_mean": 14.515625,
    "specialist_raw_path_cost_component_mean": 0.072646875,
    "specialist_raw_reasoning_cost_component_mean": 6.35664890625,
    "specialist_raw_reasoning_cost_component_api_mean": 1.5501085000000001,
    "specialist_raw_reasoning_cost_component_token_mean": 6.35664890625,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]"
    ]
  }
}