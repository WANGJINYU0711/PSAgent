# Seed1 old risky_ps vs probfloor0002 targeted diagnostic

## Scope

- old: `tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/risky_ps`

- probfloor: `tmp/llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_probfloor0002/risky_ps`

- 对齐键：`episode_index`，并断言 `dataset_index` 一致。

- 注意：这两个 repeated smoke 产物没有保存完整 executor `stage_trace` 或 LLM Stage 4/5 raw JSON。下面的“Stage 4/5 reasons”因此分两类：task oracle 中的 Stage 4/5 blocker/terminal oracle reasons，以及 episode 扁平记录里的 `terminal_adjustment_reasons`、Stage 5 replay/executed tools、path/mode/interface 字段。


## Headline Post Split

| run | n | total | terminal | reasoning | modecost_report | clear | aux | strict | fast_on_deep_n | deep_on_deep_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old | 75 | 12.436 | 7.980 | 4.387 | 1.700 | 0.533 | 0.600 | 0.493 | 14 | 61 |
| probfloor0002 | 75 | 12.059 | 7.640 | 4.347 | 1.647 | 0.587 | 0.653 | 0.547 | 12 | 63 |


## Probfloor Improved Fast-On-Deep Episodes

| episode_index | dataset_index | oracle_action | old_terminal | pf_terminal | delta_terminal_pf_minus_old | old_pattern | pf_pattern | old_final | pf_final | old_cause | pf_cause | pf_replay_tools |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 49 | 16 | repair_all | 14.000 | 0.000 | -14.000 | fdffd on fdddd | fffdd on fdddd | repair_subset | repair_all | subset_mismatch_with_missing_oracle_repair_tools | no_terminal_failure_or_low_penalty | toggle_airplane_mode\|reseat_sim_card\|toggle_data\|toggle_roaming\|set_network_mode_preference\|toggle_wifi_calling\|reset_apn_settings\|reboot_device\|grant_app_permission |


## All Probfloor Fast-On-Deep Episodes

| episode_index | dataset_index | oracle_action | old_terminal | pf_terminal | delta_terminal_pf_minus_old | old_pattern | pf_pattern | old_final | pf_final | old_cause | pf_cause |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 40 | 32 | repair_all | 0.000 | 0.000 | 0.000 | ffdfd on fdddd | dffdf on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 41 | 33 | repair_all | 0.000 | 0.000 | 0.000 | fdfdd on fdddd | fdffd on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 42 | 34 | repair_all | 0.000 | 0.000 | 0.000 | fdddd on fdddd | fffdd on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 49 | 16 | repair_all | 14.000 | 0.000 | -14.000 | fdffd on fdddd | fffdd on fdddd | repair_subset | repair_all | subset_mismatch_with_missing_oracle_repair_tools | no_terminal_failure_or_low_penalty |
| 51 | 33 | repair_all | 0.000 | 0.000 | 0.000 | fdddd on fdddd | ffddf on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 61 | 33 | repair_all | 0.000 | 0.000 | 0.000 | fdddd on fdddd | fdffd on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 63 | 35 | repair_all | 0.000 | 0.000 | 0.000 | fdfff on fdddd | fddff on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 64 | 36 | repair_all | 0.000 | 0.000 | 0.000 | fdddd on fdddd | dffdf on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 94 | 36 | repair_all | 0.000 | 0.000 | 0.000 | fdddd on fdddd | fdfdf on fdddd | repair_all | repair_all | no_terminal_failure_or_low_penalty | no_terminal_failure_or_low_penalty |
| 66 | 9 | repair_subset | 14.000 | 14.000 | 0.000 | dfdff on fdddd | fffdd on fdddd | repair_subset | repair_subset | subset_mismatch_with_missing_oracle_repair_tools | subset_mismatch_terminal_decision |
| 89 | 16 | repair_all | 15.000 | 15.000 | 0.000 | fffff on fdddd | ffddf on fdddd | repair_subset | repair_subset | subset_mismatch_with_missing_oracle_repair_tools | subset_mismatch_with_missing_oracle_repair_tools |
| 46 | 9 | repair_subset | 12.000 | 23.500 | 11.500 | dfddf on fdddd | fffdd on fdddd | repair_subset | transfer | subset_mismatch_terminal_decision | over_transfer_on_local_oracle |


## Probfloor Deep-On-Deep Terminal Worse Episodes

| episode_index | dataset_index | oracle_action | old_terminal | pf_terminal | delta_terminal_pf_minus_old | old_pattern | pf_pattern | old_final | pf_final | old_cause | pf_cause | pf_missing_blockers |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 74 | 36 | repair_all | 0.000 | 18.500 | 18.500 | fdddd on fdddd | dfdfd on fdddd | repair_all | transfer | no_terminal_failure_or_low_penalty | over_transfer_on_local_oracle | bad_wifi_calling\|break_apn_mms_setting\|break_app_storage_permission\|data_mode_off\|unseat_sim_card |
| 56 | 9 | repair_subset | 6.000 | 23.500 | 17.500 | fdddf on fdddd | dfddf on fdddd | repair_subset | transfer | policy_action_violation | over_transfer_on_local_oracle | airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|data_mode_off\|unseat_sim_card |
| 27 | 10 | repair_subset | 6.000 | 22.500 | 16.500 | fdddd on fdddd | fdddd on fdddd | repair_subset | transfer | policy_action_violation | over_transfer_on_local_oracle | airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|unseat_sim_card |
| 69 | 16 | repair_all | 0.000 | 15.000 | 15.000 | ddddd on fdddd | ffddd on fdddd | repair_all | repair_subset | no_terminal_failure_or_low_penalty | subset_mismatch_with_missing_oracle_repair_tools | airplane_mode_on\|bad_network_preference\|data_mode_off\|unseat_sim_card\|user_abroad_roaming_enabled_off |
| 58 | 13 | repair_subset | 6.000 | 19.000 | 13.000 | fdfdd on fdddd | dffdd on fdddd | repair_subset | repair_subset | policy_action_violation | subset_mismatch_terminal_decision |  |
| 26 | 9 | repair_subset | 12.000 | 23.500 | 11.500 | dfddd on fdddd | dfddd on fdddd | repair_subset | transfer | subset_mismatch_terminal_decision | over_transfer_on_local_oracle | airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions\|data_mode_off\|unseat_sim_card |
| 97 | 10 | repair_subset | 12.000 | 23.500 | 11.500 | ffddd on fdddd | ffddd on fdddd | repair_subset | transfer | subset_mismatch_terminal_decision | over_transfer_on_local_oracle | airplane_mode_on\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission\|data_mode_off\|unseat_sim_card |
| 83 | 35 | repair_all | 0.000 | 10.000 | 10.000 | fdddd on fdddd | ddddd on fdddd | repair_all | repair_subset | no_terminal_failure_or_low_penalty | subset_mismatch_with_missing_oracle_repair_tools | airplane_mode_on\|user_abroad_roaming_enabled_off |
| 35 | 2 | repair_all | 10.000 | 18.500 | 8.500 | ddddd on fdddd | fdddd on fdddd | repair_subset | transfer | subset_mismatch_with_missing_oracle_repair_tools | over_transfer_on_local_oracle | airplane_mode_on\|unseat_sim_card |
| 48 | 13 | repair_subset | 15.000 | 17.000 | 2.000 | ffdff on fdddd | ffddd on fdddd | repair_subset | repair_subset | subset_mismatch_with_missing_oracle_repair_tools | subset_mismatch_with_missing_oracle_repair_tools | airplane_mode_on\|bad_network_preference\|data_mode_off\|unseat_sim_card\|user_abroad_roaming_enabled_off |


## Focus Datasets 16/13/10/2



### Stage 4/5 Oracle Reasons Available From Tasks

| dataset_index | oracle_action | repairability | selected | deferred | stage5 | stage4_repair_reasons | stage4_defer_reasons |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | repair_all | repairable | airplane_mode_on\|unseat_sim_card\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_storage_permission |  | final=repair_all; success=can_send_mms_true; postchecks=can_send_mms; transfer=None | airplane_mode_on:repair_service_blocker deps=; unseat_sim_card:repair_service_blocker deps=airplane_mode_on; bad_network_preference:repair_data_blocker deps=airplane_mode_on,unseat_sim_card; bad_wifi_calling:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,bad_network_preference; break_apn_mms_setting:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,bad_network_preference; break_app_storage_permission:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,bad_network_preference |  |
| 10 | repair_subset | partially_repairable | airplane_mode_on\|unseat_sim_card\|data_mode_off\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission | user_abroad_roaming_disabled_on\|data_usage_exceeded | final=repair_subset; success=partial_resolution_only; postchecks=; transfer=None | airplane_mode_on:repair_service_blocker deps=; unseat_sim_card:repair_service_blocker deps=airplane_mode_on; data_mode_off:repair_data_blocker deps=airplane_mode_on,unseat_sim_card; bad_network_preference:repair_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off; bad_wifi_calling:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_disabled_on,data_usage_exceeded,bad_network_preference; break_apn_mms_setting:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_disabled_on,data_usage_exceeded,bad_network_preference; break_app_sms_permission:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_disabled_on,data_usage_exceeded,bad_network_preference | user_abroad_roaming_disabled_on:defer_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off; data_usage_exceeded:defer_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off |
| 13 | repair_subset | partially_repairable | airplane_mode_on\|unseat_sim_card\|data_mode_off\|user_abroad_roaming_enabled_off\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_both_permissions | data_usage_exceeded | final=repair_subset; success=partial_resolution_only; postchecks=; transfer=None | airplane_mode_on:repair_service_blocker deps=; unseat_sim_card:repair_service_blocker deps=airplane_mode_on; data_mode_off:repair_data_blocker deps=airplane_mode_on,unseat_sim_card; user_abroad_roaming_enabled_off:repair_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off; bad_network_preference:repair_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off; bad_wifi_calling:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_enabled_off,data_usage_exceeded,bad_network_preference; break_apn_mms_setting:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_enabled_off,data_usage_exceeded,bad_network_preference; break_app_both_permissions:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_enabled_off,data_usage_exceeded,bad_network_preference | data_usage_exceeded:defer_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off |
| 16 | repair_all | repairable | airplane_mode_on\|unseat_sim_card\|data_mode_off\|user_abroad_roaming_enabled_off\|bad_network_preference\|bad_wifi_calling\|break_apn_mms_setting\|break_app_sms_permission |  | final=repair_all; success=can_send_mms_true; postchecks=can_send_mms; transfer=None | airplane_mode_on:repair_service_blocker deps=; unseat_sim_card:repair_service_blocker deps=airplane_mode_on; data_mode_off:repair_data_blocker deps=airplane_mode_on,unseat_sim_card; user_abroad_roaming_enabled_off:repair_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off; bad_network_preference:repair_data_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off; bad_wifi_calling:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_enabled_off,bad_network_preference; break_apn_mms_setting:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_enabled_off,bad_network_preference; break_app_sms_permission:repair_mms_app_blocker deps=airplane_mode_on,unseat_sim_card,data_mode_off,user_abroad_roaming_enabled_off,bad_network_preference |  |


### Episode Outcome Summary

| dataset_index | n | oracle_action | repairability | old_terminal_mean | pf_terminal_mean | delta_terminal | old_final_counts | pf_final_counts | old_cause_counts | pf_cause_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2 | 8 | repair_all | repairable | 11.000 | 7.562 | -3.438 | {'repair_subset': 8} | {'repair_subset': 4, 'transfer': 1, 'repair_all': 3} | {'subset_mismatch_with_missing_oracle_repair_tools': 8} | {'subset_mismatch_with_missing_oracle_repair_tools': 4, 'over_transfer_on_local_oracle': 1, 'no_terminal_failure_or_low_penalty': 3} |
| 10 | 8 | repair_subset | partially_repairable | 19.500 | 22.875 | 3.375 | {'repair_subset': 2, 'transfer': 6} | {'transfer': 8} | {'policy_action_violation': 1, 'over_transfer_on_local_oracle': 6, 'subset_mismatch_terminal_decision': 1} | {'over_transfer_on_local_oracle': 8} |
| 13 | 8 | repair_subset | partially_repairable | 17.500 | 16.750 | -0.750 | {'repair_subset': 6, 'transfer': 2} | {'repair_subset': 8} | {'subset_mismatch_with_missing_oracle_repair_tools': 3, 'subset_mismatch_terminal_decision': 2, 'policy_action_violation': 1, 'over_transfer_on_local_oracle': 2} | {'subset_mismatch_with_missing_oracle_repair_tools': 5, 'subset_mismatch_terminal_decision': 3} |
| 16 | 8 | repair_all | repairable | 10.938 | 6.562 | -4.375 | {'repair_all': 3, 'repair_subset': 4, 'transfer': 1} | {'repair_all': 5, 'repair_subset': 2, 'transfer': 1} | {'no_terminal_failure_or_low_penalty': 3, 'subset_mismatch_with_missing_oracle_repair_tools': 4, 'over_transfer_on_local_oracle': 1} | {'no_terminal_failure_or_low_penalty': 5, 'subset_mismatch_with_missing_oracle_repair_tools': 2, 'over_transfer_on_local_oracle': 1} |


## Post Stage Pattern Details

| run | pattern | n | terminal | legacy_terminal | reasoning | modecost_report | total | clear | aux | strict | transfer_final_n | rtp_ge_10_n | rtp_ge_14_n | examples |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old_seed1_risky_ps | fdddd on fdddd | 18 | 6.639 | 5.028 | 3.664 | 0.000 | 10.376 | 0.667 | 0.778 | 0.611 | 3 | 6 | 5 | 27,30,39,42,51,52,54,57 |
| old_seed1_risky_ps | ffddd on fdddd | 14 | 2.286 | 0.929 | 4.016 | 1.500 | 6.371 | 0.786 | 0.929 | 0.786 | 0 | 3 | 0 | 31,34,43,44,55,71,72,80 |
| old_seed1_risky_ps | ddddd on fdddd | 8 | 9.938 | 7.812 | 4.655 | 0.500 | 14.667 | 0.500 | 0.625 | 0.500 | 3 | 4 | 3 | 33,35,47,69,76,81,84,88 |
| old_seed1_risky_ps | dfddd on fdddd | 6 | 10.750 | 6.417 | 4.574 | 2.000 | 15.394 | 0.333 | 0.333 | 0.333 | 1 | 4 | 2 | 26,32,70,77,96,98 |
| old_seed1_risky_ps | fdfdd on fdddd | 5 | 7.700 | 5.100 | 4.358 | 1.500 | 12.129 | 0.600 | 0.600 | 0.400 | 1 | 2 | 1 | 29,41,58,67,75 |
| old_seed1_risky_ps | dffdd on fdddd | 4 | 8.875 | 5.625 | 4.840 | 3.500 | 13.781 | 0.500 | 0.500 | 0.500 | 1 | 2 | 1 | 37,60,62,86 |
| old_seed1_risky_ps | ffddf on fdddd | 3 | 6.333 | 5.000 | 4.635 | 3.000 | 11.037 | 0.667 | 0.667 | 0.667 | 0 | 1 | 1 | 38,90,93 |
| old_seed1_risky_ps | fdddf on fdddd | 2 | 14.250 | 10.250 | 4.885 | 1.500 | 19.202 | 0.500 | 0.000 | 0.000 | 1 | 1 | 1 | 36,56 |
| old_seed1_risky_ps | fddfd on fdddd | 2 | 13.500 | 8.000 | 4.572 | 1.500 | 18.147 | 0.000 | 0.000 | 0.000 | 0 | 2 | 1 | 28,85 |
| old_seed1_risky_ps | fffdd on fdddd | 2 | 18.750 | 12.250 | 4.408 | 3.000 | 23.224 | 0.000 | 0.500 | 0.000 | 1 | 2 | 2 | 53,87 |
| old_seed1_risky_ps | fffff on fdddd | 2 | 14.500 | 9.000 | 6.018 | 6.000 | 20.576 | 0.000 | 0.000 | 0.000 | 0 | 2 | 2 | 45,89 |
| old_seed1_risky_ps | dfddf on fdddd | 1 | 12.000 | 3.000 | 6.107 | 3.500 | 18.172 | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 46 |
| old_seed1_risky_ps | dfdfd on fdddd | 1 | 12.000 | 7.000 | 5.067 | 3.500 | 17.137 | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 25 |
| old_seed1_risky_ps | dfdff on fdddd | 1 | 14.000 | 5.000 | 6.721 | 5.000 | 20.783 | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 66 |
| old_seed1_risky_ps | fdffd on fdddd | 1 | 14.000 | 5.000 | 5.671 | 3.000 | 19.738 | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 49 |
| old_seed1_risky_ps | fdfff on fdddd | 1 | 0.000 | 0.000 | 5.560 | 4.500 | 5.616 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 63 |
| old_seed1_risky_ps | ffdfd on fdddd | 1 | 0.000 | 0.000 | 4.061 | 3.000 | 4.125 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 40 |
| old_seed1_risky_ps | ffdff on fdddd | 1 | 15.000 | 11.000 | 5.708 | 4.500 | 20.770 | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 48 |
| old_seed1_risky_ps | fffdf on fdddd | 1 | 0.000 | 0.000 | 4.404 | 4.500 | 4.464 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 50 |
| old_seed1_risky_ps | ffffd on fdddd | 1 | 21.000 | 17.000 | 4.022 | 4.500 | 25.078 | 0.000 | 0.000 | 0.000 | 0 | 1 | 1 | 59 |
| seed1_probfloor0002 | ffddd on fdddd | 18 | 8.250 | 6.194 | 4.106 | 1.500 | 12.429 | 0.556 | 0.611 | 0.556 | 3 | 8 | 7 | 31,34,44,48,50,55,62,68 |
| seed1_probfloor0002 | fdddd on fdddd | 14 | 9.679 | 7.607 | 3.722 | 0.000 | 13.476 | 0.500 | 0.714 | 0.500 | 5 | 7 | 6 | 27,30,35,38,39,45,57,67 |
| seed1_probfloor0002 | ddddd on fdddd | 8 | 6.812 | 4.062 | 4.708 | 0.500 | 11.595 | 0.625 | 0.625 | 0.375 | 1 | 3 | 1 | 33,37,53,59,75,76,83,86 |
| seed1_probfloor0002 | dfddd on fdddd | 8 | 4.188 | 3.062 | 4.246 | 2.000 | 8.503 | 0.750 | 0.875 | 0.750 | 1 | 2 | 1 | 26,32,65,70,84,85,91,95 |
| seed1_probfloor0002 | fdfdd on fdddd | 5 | 4.500 | 3.700 | 4.197 | 1.500 | 8.770 | 0.800 | 0.800 | 0.800 | 1 | 1 | 1 | 29,43,47,52,60 |
| seed1_probfloor0002 | fffdd on fdddd | 4 | 9.375 | 5.625 | 4.392 | 3.000 | 13.830 | 0.500 | 0.500 | 0.500 | 1 | 2 | 2 | 42,46,49,66 |
| seed1_probfloor0002 | dfddf on fdddd | 3 | 19.167 | 15.167 | 5.661 | 3.500 | 24.894 | 0.000 | 0.000 | 0.000 | 1 | 3 | 3 | 56,88,98 |
| seed1_probfloor0002 | dfdfd on fdddd | 2 | 15.250 | 10.750 | 5.006 | 3.500 | 20.326 | 0.000 | 0.000 | 0.000 | 1 | 2 | 1 | 25,74 |
| seed1_probfloor0002 | dffdd on fdddd | 2 | 9.500 | 7.500 | 4.915 | 3.500 | 14.480 | 0.500 | 0.500 | 0.500 | 0 | 1 | 1 | 54,58 |
| seed1_probfloor0002 | dffdf on fdddd | 2 | 0.000 | 0.000 | 5.289 | 5.000 | 5.350 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 40,64 |
| seed1_probfloor0002 | fdddf on fdddd | 2 | 3.000 | 1.000 | 4.446 | 1.500 | 7.519 | 1.000 | 0.500 | 0.500 | 0 | 0 | 0 | 36,92 |
| seed1_probfloor0002 | fdffd on fdddd | 2 | 0.000 | 0.000 | 4.614 | 3.000 | 4.686 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 41,61 |
| seed1_probfloor0002 | ffddf on fdddd | 2 | 7.500 | 5.500 | 4.728 | 3.000 | 12.289 | 0.500 | 1.000 | 0.500 | 0 | 1 | 1 | 51,89 |
| seed1_probfloor0002 | fddfd on fdddd | 1 | 13.000 | 9.000 | 4.968 | 1.500 | 18.043 | 0.000 | 0.000 | 0.000 | 0 | 1 | 0 | 28 |
| seed1_probfloor0002 | fddff on fdddd | 1 | 0.000 | 0.000 | 5.322 | 3.000 | 5.394 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 63 |
| seed1_probfloor0002 | fdfdf on fdddd | 1 | 0.000 | 0.000 | 4.575 | 3.000 | 4.647 | 1.000 | 1.000 | 1.000 | 0 | 0 | 0 | 94 |


## Terminal Transition Causes

| run | transition | n | terminal | main_causes |
| --- | --- | --- | --- | --- |
| old | matched_terminal_action | 51 | 3.451 | {'no_terminal_failure_or_low_penalty': 37, 'subset_mismatch_terminal_decision': 7, 'subset_mismatch_with_missing_oracle_repair_tools': 4, 'policy_action_violation': 3} |
| old | repair_all_to_repair_subset | 13 | 12.846 | {'subset_mismatch_with_missing_oracle_repair_tools': 13} |
| old | local_to_transfer | 11 | 23.227 | {'over_transfer_on_local_oracle': 11} |
| probfloor0002 | matched_terminal_action | 54 | 3.296 | {'no_terminal_failure_or_low_penalty': 41, 'subset_mismatch_with_missing_oracle_repair_tools': 5, 'subset_mismatch_terminal_decision': 5, 'policy_action_violation': 3} |
| probfloor0002 | local_to_transfer | 14 | 22.357 | {'over_transfer_on_local_oracle': 14} |
| probfloor0002 | repair_all_to_repair_subset | 7 | 11.714 | {'subset_mismatch_with_missing_oracle_repair_tools': 7} |


### Targeted Transition Dataset/Pattern Counts

| run | transition | n | terminal | dataset_counts | pattern_counts |
| --- | --- | --- | --- | --- | --- |
| old | repair_all_to_repair_subset | 13 | 12.846 | {2: 8, 16: 4, 35: 1} | {'fdddd on fdddd': 2, 'fffff on fdddd': 2, 'ffddd on fdddd': 2, 'dfdfd on fdddd': 1, 'ddddd on fdddd': 1, 'fdffd on fdddd': 1, 'fffdd on fdddd': 1, 'ffffd on fdddd': 1} |
| old | local_to_transfer | 11 | 23.227 | {10: 6, 9: 2, 13: 2, 16: 1} | {'ddddd on fdddd': 3, 'fdddd on fdddd': 3, 'fdddf on fdddd': 1, 'dffdd on fdddd': 1, 'fdfdd on fdddd': 1, 'dfddd on fdddd': 1, 'fffdd on fdddd': 1} |
| probfloor0002 | repair_all_to_repair_subset | 7 | 11.714 | {2: 4, 16: 2, 35: 1} | {'ddddd on fdddd': 2, 'dfdfd on fdddd': 1, 'fdddd on fdddd': 1, 'ffddd on fdddd': 1, 'dfddd on fdddd': 1, 'ffddf on fdddd': 1} |
| probfloor0002 | local_to_transfer | 14 | 22.357 | {10: 8, 9: 3, 2: 1, 36: 1, 16: 1} | {'fdddd on fdddd': 5, 'ffddd on fdddd': 3, 'dfddd on fdddd': 1, 'ddddd on fdddd': 1, 'fffdd on fdddd': 1, 'fdfdd on fdddd': 1, 'dfddf on fdddd': 1, 'dfdfd on fdddd': 1} |


## High-Terminal Suffix Concentration

| run | pattern | suffix | shared | n_high_terminal | terminal_mean | datasets | episodes | final_counts | cause_counts |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| old | fdddd on fdddd | stage4_n1__from__n0024__c01 > stage5_n2__from__n0075__c02 | True | 2 | 16.500 | 13,16 | 39,78 | {'repair_subset': 2} | {'subset_mismatch_with_missing_oracle_repair_tools': 1, 'subset_mismatch_terminal_decision': 1} |
| old | ddddd on fdddd | stage4_n1__from__n0032__c01 > stage5_n2__from__n0097__c02 | True | 1 | 22.500 | 9 | 76 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | ddddd on fdddd | stage4_n1__from__n0033__c01 > stage5_n1__from__n0098__c01 | True | 1 | 24.500 | 13 | 88 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | ddddd on fdddd | stage4_n1__from__n0033__c01 > stage5_n2__from__n0098__c02 | True | 1 | 10.000 | 2 | 35 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | ddddd on fdddd | stage4_n1__from__n0041__c01 > stage5_n2__from__n0120__c02 | True | 1 | 22.500 | 10 | 47 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | dfddd on fdddd | stage4_n1__from__n0043__c01 > stage5_n1__from__n0122__c01 | True | 1 | 17.000 | 13 | 98 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | dfddd on fdddd | stage4_n3__from__n0035__c02 > stage5_n2__from__n0101__c02 | True | 1 | 12.000 | 9 | 26 | {'repair_subset': 1} | {'subset_mismatch_terminal_decision': 1} |
| old | dfddd on fdddd | stage4_n3__from__n0047__c02 > stage5_n3__from__n0134__c03 | True | 1 | 12.000 | 9 | 96 | {'repair_subset': 1} | {'subset_mismatch_terminal_decision': 1} |
| old | dfddd on fdddd | stage4_n5__from__n0049__c04 > stage5_n5__from__n0142__c03 | False | 1 | 23.500 | 10 | 77 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | dfddf on fdddd | stage4_n3__from__n0044__c02 > stage5_n4__from__n0124__c04 | True | 1 | 12.000 | 9 | 46 | {'repair_subset': 1} | {'subset_mismatch_terminal_decision': 1} |
| old | dfdfd on fdddd | stage4_n4__from__n0037__c03 > stage5_n1__from__n0108__c01 | True | 1 | 12.000 | 2 | 25 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | dfdff on fdddd | stage4_n4__from__n0049__c03 > stage5_n4__from__n0141__c03 | True | 1 | 14.000 | 9 | 66 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | dffdd on fdddd | stage4_n2__from__n0039__c01 > stage5_n1__from__n0112__c01 | True | 1 | 23.500 | 10 | 37 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | dffdd on fdddd | stage4_n2__from__n0048__c01 > stage5_n1__from__n0135__c01 | True | 1 | 12.000 | 9 | 86 | {'repair_subset': 1} | {'subset_mismatch_terminal_decision': 1} |
| old | fdddd on fdddd | stage4_n1__from__n0023__c01 > stage5_n1__from__n0074__c01 | True | 1 | 22.500 | 16 | 79 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | fdddd on fdddd | stage4_n1__from__n0023__c01 > stage5_n2__from__n0074__c02 | True | 1 | 25.500 | 13 | 68 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | fdddd on fdddd | stage4_n1__from__n0051__c01 > stage5_n1__from__n0144__c01 | True | 1 | 10.000 | 2 | 65 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | fdddd on fdddd | stage4_n1__from__n0051__c01 > stage5_n2__from__n0144__c02 | True | 1 | 22.500 | 10 | 57 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | fdddf on fdddd | stage4_n3__from__n0068__c02 > stage5_n4__from__n0197__c04 | True | 1 | 22.500 | 9 | 36 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | fddfd on fdddd | stage4_n4__from__n0070__c03 > stage5_n1__from__n0204__c01 | True | 1 | 15.000 | 13 | 28 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | fddfd on fdddd | stage4_n4__from__n0073__c03 > stage5_n1__from__n0214__c01 | True | 1 | 12.000 | 2 | 85 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | fdfdd on fdddd | stage4_n2__from__n0060__c01 > stage5_n2__from__n0168__c02 | True | 1 | 22.500 | 10 | 67 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |
| old | fdfdd on fdddd | stage4_n2__from__n0063__c01 > stage5_n2__from__n0178__c02 | True | 1 | 10.000 | 2 | 75 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | fdffd on fdddd | stage4_n4__from__n0069__c03 > stage5_n1__from__n0200__c01 | True | 1 | 14.000 | 16 | 49 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | ffddd on fdddd | stage4_n1__from__n0052__c01 > stage5_n1__from__n0145__c01 | True | 1 | 10.000 | 2 | 55 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | ffddd on fdddd | stage4_n3__from__n0029__c02 > stage5_n2__from__n0088__c02 | True | 1 | 12.000 | 10 | 97 | {'repair_subset': 1} | {'subset_mismatch_terminal_decision': 1} |
| old | ffddd on fdddd | stage4_n3__from__n0065__c02 > stage5_n3__from__n0187__c03 | True | 1 | 10.000 | 2 | 95 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | ffddf on fdddd | stage4_n5__from__n0028__c04 > stage5_n4__from__n0086__c02 | True | 1 | 19.000 | 13 | 38 | {'repair_subset': 1} | {'subset_mismatch_terminal_decision': 1} |
| old | ffdff on fdddd | stage4_n4__from__n0055__c03 > stage5_n4__from__n0154__c03 | True | 1 | 15.000 | 13 | 48 | {'repair_subset': 1} | {'subset_mismatch_with_missing_oracle_repair_tools': 1} |
| old | fffdd on fdddd | stage4_n2__from__n0066__c01 > stage5_n1__from__n0188__c01 | True | 1 | 23.500 | 10 | 87 | {'transfer': 1} | {'over_transfer_on_local_oracle': 1} |


## Interpretation

- `probfloor0002` 的 post terminal 改善主要不是来自 deep-on-deep 质量提升；相反，deep-on-deep 桶里出现了多条 terminal 变差 episode。

- fast-on-deep 改善 episode 需要谨慎解读：如果它们从 old 的 `repair_all -> repair_subset` / clear failure 变成 probfloor 的低罚，且 Stage 5 replay tools 覆盖 oracle tools，说明 probfloor 只是把采样推到某些更好执行的接口/路径；这不是“fast-on-deep 普遍更好”。

- 高 terminal 如果集中在同一个 shared suffix 且跨 dataset 重复，才支持 shareable suffix failure。若同一 dataset 在不同 pattern/suffix 间好坏切换，或者 high terminal 同时伴随 missing oracle tools、subset mismatch、local clear floor，则更像 path/interface-specific execution failure。

- 当前扁平产物无法直接恢复 LLM Stage 4/5 raw rationale；若要定位“为什么 Stage 4 漏 blocker / Stage 5 转 transfer”，下一轮 runner 需要持久化 executor `stage_trace` 中的 `stage4_output`、`stage4_execution_summary`、`stage5_output`、`stage5_verification_summary` 和 raw LLM JSON。
