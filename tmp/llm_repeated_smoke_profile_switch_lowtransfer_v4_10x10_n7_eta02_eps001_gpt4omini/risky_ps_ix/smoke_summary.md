{
  "summary": {
    "test_name": "risky_ps_ix_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_ix",
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
    "exact_match_mean": 0.24,
    "terminal_penalty_mean": 9.61,
    "raw_outcome_penalty_mean": 8.37,
    "raw_policy_penalty_mean": 1.24,
    "raw_terminal_penalty_mean": 9.61,
    "total_cost_mean": 0.4469016369945685,
    "raw_total_cost_mean": 14.81032025,
    "raw_total_cost_api_mean": 10.174554930000001,
    "raw_total_cost_token_mean": 14.81032025,
    "reasoning_cost_mean": 5.12940925,
    "raw_reasoning_cost_component_mean": 5.12940925,
    "raw_reasoning_cost_component_api_mean": 0.49364393,
    "raw_reasoning_cost_component_token_mean": 5.12940925,
    "raw_path_cost_component_mean": 0.070911,
    "algorithm_cumulative_total_cost": 44.690163699456846,
    "raw_algorithm_cumulative_total_cost": 1481.032025,
    "oracle_stationary_total_cost": 6.946650573325287,
    "raw_oracle_stationary_total_cost": 230.21200000000002,
    "raw_outcome_penalty_cumulative": 837.0,
    "raw_policy_penalty_cumulative": 124.0,
    "raw_terminal_penalty_cumulative": 961.0,
    "raw_path_cost_component_cumulative": 7.091100000000001,
    "raw_reasoning_cost_component_cumulative": 512.940925,
    "mean_llm_call_count": 12.62,
    "mean_prompt_tokens": 49055.99,
    "mean_completion_tokens": 1444.36,
    "mean_total_tokens": 50500.35,
    "cumulative_total_tokens": 5050035.0,
    "mean_api_cost_usd_raw": 0.005254294499999999,
    "cumulative_api_cost_usd_raw": 0.52542945,
    "mean_generation_time_seconds": 41.84399675548077,
    "p50_generation_time_seconds": 39.21376620605588,
    "p90_generation_time_seconds": 51.422275625355546,
    "mean_llm_round_trip_seconds": 41.889361689537765,
    "mean_episode_wall_clock_seconds": 45.250618108659985,
    "p50_episode_wall_clock_seconds": 42.80527164693922,
    "p90_episode_wall_clock_seconds": 55.12424877453596,
    "mean_tool_wall_clock_seconds": 0.012784160617738962,
    "policy_action_violation_rate": 0.62,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.62,
    "subset_mismatch_count": 76,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.92,
    "unshared_path_fraction": 0.08,
    "mean_barrier_stop_depth": 2.619047619047619,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 50,
        "fraction": 0.5
      },
      "stage1": {
        "count": 27,
        "fraction": 0.27
      },
      "stage2": {
        "count": 6,
        "fraction": 0.06
      },
      "stage3": {
        "count": 14,
        "fraction": 0.14
      },
      "stage4": {
        "count": 3,
        "fraction": 0.03
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.6,
      2.97,
      2.69,
      3.05
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.6,
      2.97,
      2.69,
      3.05
    ],
    "specialist_task_count": 78,
    "specialist_task_unshared_fraction": 0.07692307692307693,
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
    "specialist_shared_path_fraction": 0.9230769230769231,
    "specialist_unshared_path_fraction": 0.07692307692307693,
    "specialist_exact_match_mean": 0.23076923076923078,
    "specialist_total_cost_mean": 0.5187280302678613,
    "specialist_raw_outcome_penalty_mean": 10.41025641025641,
    "specialist_raw_policy_penalty_mean": 1.4871794871794872,
    "specialist_raw_terminal_penalty_mean": 11.897435897435898,
    "specialist_raw_path_cost_component_mean": 0.07123461538461538,
    "specialist_raw_reasoning_cost_component_mean": 5.221976410256411,
    "specialist_raw_reasoning_cost_component_api_mean": 0.503762,
    "specialist_raw_reasoning_cost_component_token_mean": 5.221976410256411,
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