# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 5 | canonical_pattern | ddddd | 3 | 6.000 | [12.0, 0.0, 6.0] | {"repair_all": 1, "repair_subset": 2} | {"": 3} | {"data_usage_exceeded": 1, "": 2} | 0.000 | 0.000 |
| 5 | canonical_pattern | fffff | 3 | 14.333 | [14.0, 15.0, 14.0] | {"repair_subset": 2, "repair_all": 1} | {"bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions": 1, "bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions": 1, "": 1} | {"data_usage_exceeded": 3} | 0.000 | 1.000 |
| 11 | canonical_pattern | ddddd | 3 | 8.000 | [12.0, 0.0, 12.0] | {"repair_all": 2, "repair_subset": 1} | {"": 3} | {"data_usage_exceeded": 2, "": 1} | 0.000 | 0.000 |
| 11 | canonical_pattern | fffff | 3 | 14.000 | [14.0, 14.0, 14.0] | {"repair_all": 1, "repair_subset": 2} | {"": 1, "break_apn_mms_setting": 1, "break_apn_mms_setting|break_app_sms_permission": 1} | {"data_usage_exceeded": 3} | 0.000 | 1.000 |
