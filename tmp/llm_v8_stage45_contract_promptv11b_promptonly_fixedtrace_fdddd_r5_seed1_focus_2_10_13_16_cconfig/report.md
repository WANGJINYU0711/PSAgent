# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | fdddd | 5 | 0.000 | [0.0, 0.0, 0.0, 0.0, 0.0] | {"repair_all": 5} | {"": 5} | {"": 5} | 0.000 | 0.000 |
| 10 | canonical_pattern | fdddd | 5 | 6.000 | [6.0, 6.0, 6.0, 6.0, 6.0] | {"repair_subset": 5} | {"": 5} | {"": 5} | 0.000 | 0.000 |
| 13 | canonical_pattern | fdddd | 5 | 12.000 | [21.0, 21.0, 6.0, 6.0, 6.0] | {"repair_subset": 5} | {"airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 2, "": 3} | {"data_usage_exceeded": 2, "": 3} | 0.000 | 0.600 |
| 16 | canonical_pattern | fdddd | 5 | 4.500 | [0.0, 22.5, 0.0, 0.0, 0.0] | {"repair_all": 4, "transfer": 1} | {"": 4, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"": 5} | 0.000 | 0.200 |
