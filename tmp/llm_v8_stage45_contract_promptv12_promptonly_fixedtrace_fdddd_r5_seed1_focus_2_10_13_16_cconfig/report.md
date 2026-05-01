# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | fdddd | 5 | 3.700 | [0.0, 0.0, 0.0, 18.5, 0.0] | {"repair_all": 4, "transfer": 1} | {"": 4, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|unseat_sim_card": 1} | {"": 5} | 0.000 | 0.200 |
| 10 | canonical_pattern | fdddd | 5 | 13.900 | [6.0, 12.0, 21.0, 20.5, 10.0] | {"repair_subset": 3, "transfer": 1, "repair_all": 1} | {"": 2, "bad_wifi_calling": 1, "airplane_mode_on|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card": 1, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card": 1} | {"": 3, "data_usage_exceeded|user_abroad_roaming_disabled_on": 2} | 0.000 | 0.800 |
| 13 | canonical_pattern | fdddd | 5 | 15.900 | [22.5, 6.0, 22.5, 6.0, 22.5] | {"transfer": 3, "repair_subset": 2} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 3, "": 2} | {"": 5} | 0.000 | 1.000 |
| 16 | canonical_pattern | fdddd | 5 | 12.000 | [22.5, 22.5, 0.0, 15.0, 0.0] | {"transfer": 2, "repair_all": 2, "repair_subset": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 2, "": 2, "airplane_mode_on|bad_network_preference|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"": 5} | 0.000 | 0.400 |
