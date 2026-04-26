{
  "summary": {
    "test_name": "risky_ps_repeated_trap_switch_x8_shared_basin_strong_prefix_dedup_profile_switch_full_llm",
    "dataset": "data\\derived\\telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch\\tasks.json",
    "dataset_indices": [
      0,
      1,
      8,
      9,
      10,
      11,
      14,
      17,
      18,
      20,
      25,
      26,
      27,
      29,
      35,
      36,
      37,
      38,
      44,
      51,
      53,
      54,
      61,
      68,
      69,
      70,
      77,
      84,
      85,
      87,
      92,
      98
    ],
    "repeats": 8,
    "episodes": 128,
    "method": "risky_ps",
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
      "stage4_n5__from__n0073__c04",
      "stage5_n5__from__n0215__c03"
    ],
    "exact_match_mean": 0.4296875,
    "terminal_penalty_mean": 7.7734375,
    "raw_outcome_penalty_mean": 6.1640625,
    "raw_policy_penalty_mean": 1.609375,
    "raw_terminal_penalty_mean": 7.7734375,
    "total_cost_mean": 0.37562882609761616,
    "raw_total_cost_mean": 12.448339296875,
    "raw_total_cost_api_mean": 8.441447625,
    "raw_total_cost_token_mean": 12.448339296875,
    "reasoning_cost_mean": 4.604176796875,
    "raw_reasoning_cost_component_mean": 4.604176796875,
    "raw_reasoning_cost_component_api_mean": 0.597285125,
    "raw_reasoning_cost_component_token_mean": 4.604176796875,
    "raw_path_cost_component_mean": 0.07072500000000001,
    "algorithm_cumulative_total_cost": 48.08048974049487,
    "raw_algorithm_cumulative_total_cost": 1593.38743,
    "oracle_stationary_total_cost": 14.762776101388052,
    "raw_oracle_stationary_total_cost": 489.23840000000007,
    "raw_outcome_penalty_cumulative": 789.0,
    "raw_policy_penalty_cumulative": 206.0,
    "raw_terminal_penalty_cumulative": 995.0,
    "raw_path_cost_component_cumulative": 9.052800000000001,
    "raw_reasoning_cost_component_cumulative": 589.33463,
    "mean_llm_call_count": 11.1328125,
    "mean_prompt_tokens": 41885.484375,
    "mean_completion_tokens": 1273.2265625,
    "mean_total_tokens": 43158.7109375,
    "cumulative_total_tokens": 5524315.0,
    "mean_api_cost_usd_raw": 0.0061059765625,
    "cumulative_api_cost_usd_raw": 0.781565,
    "mean_generation_time_seconds": 32.14001113050108,
    "p50_generation_time_seconds": 32.345321849687025,
    "p90_generation_time_seconds": 36.2422002193518,
    "mean_llm_round_trip_seconds": 32.16364965629691,
    "mean_episode_wall_clock_seconds": 34.80484212492229,
    "p50_episode_wall_clock_seconds": 34.94024684932083,
    "p90_episode_wall_clock_seconds": 38.794435660308224,
    "mean_tool_wall_clock_seconds": 0.005095879721920937,
    "policy_action_violation_rate": 0.8046875,
    "policy_communication_violation_rate": 0.0,
    "policy_nl_assertion_failure_rate": 0.0,
    "mean_policy_violation_count": 0.8046875,
    "subset_mismatch_count": 73,
    "episodes_with_stage5_verification_tools": 128,
    "shared_path_fraction": 0.8984375,
    "unshared_path_fraction": 0.1015625,
    "mean_barrier_stop_depth": 2.3958333333333335,
    "first_private_barrier_stage_distribution": {
      "none": {
        "count": 67,
        "fraction": 0.5234375
      },
      "stage1": {
        "count": 28,
        "fraction": 0.21875
      },
      "stage2": {
        "count": 7,
        "fraction": 0.0546875
      },
      "stage3": {
        "count": 21,
        "fraction": 0.1640625
      },
      "stage4": {
        "count": 4,
        "fraction": 0.03125
      },
      "stage5": {
        "count": 1,
        "fraction": 0.0078125
      }
    },
    "mean_candidate_count_per_stage": [
      5.0,
      3.328125,
      3.1171875,
      2.7109375,
      2.984375
    ],
    "mean_legal_child_count_per_stage": [
      5.0,
      3.328125,
      3.1171875,
      2.7109375,
      2.984375
    ],
    "specialist_task_count": 91,
    "specialist_task_unshared_fraction": 0.08791208791208792,
    "stage_source_summary": {
      "stage1": {
        "llm_bench": 128
      },
      "stage2": {
        "llm_bench": 128
      },
      "stage3": {
        "llm_bench": 128
      },
      "stage4": {
        "llm_bench": 128
      },
      "stage5": {
        "llm_bench": 128
      }
    },
    "reasoning_cost_mode_default": "token"
  },
  "specialist_summary": {
    "specialist_episode_count": 91,
    "specialist_shared_path_fraction": 0.9120879120879121,
    "specialist_unshared_path_fraction": 0.08791208791208792,
    "specialist_exact_match_mean": 0.4175824175824176,
    "specialist_total_cost_mean": 0.4403075546963598,
    "specialist_raw_outcome_penalty_mean": 7.967032967032967,
    "specialist_raw_policy_penalty_mean": 1.7802197802197801,
    "specialist_raw_terminal_penalty_mean": 9.747252747252746,
    "specialist_raw_path_cost_component_mean": 0.07084505494505494,
    "specialist_raw_reasoning_cost_component_mean": 4.77369456043956,
    "specialist_raw_reasoning_cost_component_api_mean": 0.6249973956043956,
    "specialist_raw_reasoning_cost_component_token_mean": 4.77369456043956,
    "specialist_task_ids": [
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|data_usage_exceeded|user_abroad_roaming_disabled_on[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:None]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_off[PERSONA:Easy]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_disabled_on[PERSONA:Easy]",
      "[mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Easy]"
    ]
  }
}