{
  "summary": {
    "test_name": "risky_ps_safe_conditional_ix_repeated_trap_switch_x10_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
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
    "method": "risky_ps_safe_conditional_ix",
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
    "exact_match_mean": 0.23,
    "terminal_penalty_mean": 8.545,
    "raw_outcome_penalty_mean": 7.385,
    "raw_policy_penalty_mean": 1.16,
    "raw_terminal_penalty_mean": 8.545,
    "total_cost_mean": 0.4178230416415208,
    "raw_total_cost_mean": 13.846655599999998,
    "raw_total_cost_api_mean": 9.11704437,
    "raw_total_cost_token_mean": 13.846655599999998,
    "reasoning_cost_mean": 5.231853599999999,
    "raw_reasoning_cost_component_mean": 5.231853599999999,
    "raw_reasoning_cost_component_api_mean": 0.50224237,
    "raw_reasoning_cost_component_token_mean": 5.231853599999999,
    "raw_path_cost_component_mean": 0.069802,
    "algorithm_cumulative_total_cost": 41.78230416415208,
    "raw_algorithm_cumulative_total_cost": 1384.66556,
    "oracle_stationary_total_cost": 6.449607724803863,
    "raw_oracle_stationary_total_cost": 213.74,
    "raw_outcome_penalty_cumulative": 738.5,
    "raw_policy_penalty_cumulative": 116.0,
    "raw_terminal_penalty_cumulative": 854.5,
    "raw_path_cost_component_cumulative": 6.980200000000001,
    "raw_reasoning_cost_component_cumulative": 523.18536,
    "mean_llm_call_count": 12.56,
    "mean_prompt_tokens": 48052.56,
    "mean_completion_tokens": 1376.53,
    "mean_total_tokens": 49429.09,
    "cumulative_total_tokens": 4942909.0,
    "mean_api_cost_usd_raw": 0.0051263459999999995,
    "cumulative_api_cost_usd_raw": 0.5126345999999999,
    "mean_generation_time_seconds": 37.9602001234889,
    "p50_generation_time_seconds": 37.520702506415546,
    "p90_generation_time_seconds": 44.572040951438254,
    "mean_llm_round_trip_seconds": 38.002826511692255,
    "mean_episode_wall_clock_seconds": 41.38777032757178,
    "p50_episode_wall_clock_seconds": 40.91311267018318,
    "p90_episode_wall_clock_seconds": 47.936600694991654,
    "mean_tool_wall_clock_seconds": 0.011461298782378435,
    "policy_action_violation_rate": 0.58,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.58,
    "subset_mismatch_count": 77,
    "episodes_with_stage5_verification_tools": 100,
    "shared_path_fraction": 0.9,
    "unshared_path_fraction": 0.1,
    "mean_barrier_stop_depth": 2.5526315789473686,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 52,
        "fraction": 0.52
      },
      "stage1": {
        "count": 19,
        "fraction": 0.19
      },
      "stage2": {
        "count": 4,
        "fraction": 0.04
      },
      "stage3": {
        "count": 16,
        "fraction": 0.16
      },
      "stage4": {
        "count": 4,
        "fraction": 0.04
      },
      "stage5": {
        "count": 5,
        "fraction": 0.05
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.5,
      3.05,
      2.78,
      2.9
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.5,
      3.05,
      2.78,
      2.9
    ],
    "specialist_task_count": 68,
    "specialist_task_unshared_fraction": 0.10294117647058823,
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
    "specialist_episode_count": 68,
    "specialist_shared_path_fraction": 0.8970588235294118,
    "specialist_unshared_path_fraction": 0.10294117647058823,
    "specialist_exact_match_mean": 0.19117647058823528,
    "specialist_total_cost_mean": 0.5189481943803472,
    "specialist_raw_outcome_penalty_mean": 10.323529411764707,
    "specialist_raw_policy_penalty_mean": 1.5,
    "specialist_raw_terminal_penalty_mean": 11.823529411764707,
    "specialist_raw_path_cost_component_mean": 0.06940882352941177,
    "specialist_raw_reasoning_cost_component_mean": 5.3050049264705885,
    "specialist_raw_reasoning_cost_component_api_mean": 0.5004173382352941,
    "specialist_raw_reasoning_cost_component_token_mean": 5.3050049264705885,
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