# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 5 | canonical_pattern | ddddd | 3 | 8.667 | [10.0, 10.0, 6.0] | {"repair_all": 2, "repair_subset": 1} | {"": 3} | {"data_usage_exceeded": 2, "": 1} | 0.000 | 1.000 |
| 8 | canonical_pattern | ddddd | 3 | 16.000 | [19.0, 19.0, 10.0] | {"repair_subset": 2, "repair_all": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|data_mode_off|user_abroad_roaming_enabled_off": 2, "": 1} | {"data_usage_exceeded": 3} | 0.000 | 0.333 |
| 11 | canonical_pattern | ddddd | 3 | 13.333 | [17.0, 6.0, 17.0] | {"repair_subset": 3} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|data_mode_off": 2, "": 1} | {"data_usage_exceeded": 2, "": 1} | 0.000 | 0.000 |
