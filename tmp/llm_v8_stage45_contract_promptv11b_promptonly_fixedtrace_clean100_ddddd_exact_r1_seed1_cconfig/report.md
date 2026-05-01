# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 5 | canonical_pattern | ddddd | 1 | 12.000 | [12.0] | {"repair_subset": 1} | {"bad_wifi_calling": 1} | {"": 1} | 0.000 | 1.000 |
| 8 | canonical_pattern | ddddd | 1 | 17.000 | [17.0] | {"repair_subset": 1} | {"airplane_mode_on|bad_network_preference|break_app_sms_permission|data_mode_off|user_abroad_roaming_enabled_off": 1} | {"data_usage_exceeded": 1} | 0.000 | 0.000 |
| 11 | canonical_pattern | ddddd | 1 | 12.000 | [12.0] | {"repair_subset": 1} | {"airplane_mode_on|bad_network_preference|data_mode_off": 1} | {"": 1} | 0.000 | 0.000 |
