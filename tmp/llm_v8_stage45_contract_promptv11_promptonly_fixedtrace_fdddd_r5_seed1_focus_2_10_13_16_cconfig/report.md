# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | fdddd | 5 | 0.000 | [0.0, 0.0, 0.0, 0.0, 0.0] | {"repair_all": 5} | {"": 5} | {"": 5} | 0.000 | 0.000 |
| 10 | canonical_pattern | fdddd | 5 | 12.600 | [6.0, 22.5, 6.0, 6.0, 22.5] | {"repair_subset": 3, "transfer": 2} | {"": 3, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card": 2} | {"": 5} | 0.000 | 0.400 |
| 13 | canonical_pattern | fdddd | 5 | 9.000 | [21.0, 6.0, 6.0, 6.0, 6.0] | {"repair_subset": 5} | {"airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1, "": 4} | {"data_usage_exceeded": 1, "": 4} | 0.000 | 0.400 |
| 16 | canonical_pattern | fdddd | 5 | 0.000 | [0.0, 0.0, 0.0, 0.0, 0.0] | {"repair_all": 5} | {"": 5} | {"": 5} | 0.000 | 0.000 |
