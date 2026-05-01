# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 18 | canonical_pattern | fffff | 3 | 12.500 | [12.5, 12.5, 12.5] | {"repair_subset": 3} | {"bad_wifi_calling|break_app_storage_permission": 3} | {"": 3} | 0.000 | 1.000 |
| 19 | canonical_pattern | fffff | 3 | 0.000 | [0.0, 0.0, 0.0] | {"repair_all": 3} | {"": 3} | {"": 3} | 0.000 | 1.000 |
| 20 | canonical_pattern | fffff | 3 | 8.333 | [0.0, 12.5, 12.5] | {"repair_all": 1, "repair_subset": 2} | {"": 1, "bad_wifi_calling|break_app_sms_permission": 2} | {"": 3} | 0.000 | 1.000 |
| 21 | canonical_pattern | fffff | 3 | 0.000 | [0.0, 0.0, 0.0] | {"repair_all": 3} | {"": 3} | {"": 3} | 0.000 | 0.333 |
| 29 | canonical_pattern | fffff | 3 | 4.167 | [0.0, 12.5, 0.0] | {"repair_all": 2, "repair_subset": 1} | {"": 2, "bad_wifi_calling|break_app_both_permissions": 1} | {"": 3} | 0.000 | 0.333 |
