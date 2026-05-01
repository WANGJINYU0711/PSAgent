# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | fdddd | 5 | 0.000 | [0.0, 0.0, 0.0, 0.0, 0.0] | {"repair_all": 5} | {"": 5} | {"": 5} | 0.000 | 0.000 |
| 10 | canonical_pattern | fdddd | 5 | 10.200 | [6.0, 23.0, 6.0, 10.0, 6.0] | {"repair_subset": 4, "repair_all": 1} | {"": 4, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card": 1} | {"": 3, "data_usage_exceeded|user_abroad_roaming_disabled_on": 2} | 0.000 | 0.400 |
| 13 | canonical_pattern | fdddd | 5 | 18.000 | [21.0, 21.0, 21.0, 6.0, 21.0] | {"repair_subset": 5} | {"airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 4, "": 1} | {"data_usage_exceeded": 4, "": 1} | 0.000 | 0.200 |
| 16 | canonical_pattern | fdddd | 5 | 9.000 | [0.0, 22.5, 0.0, 22.5, 0.0] | {"repair_all": 3, "transfer": 2} | {"": 3, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 2} | {"": 5} | 0.000 | 0.400 |
