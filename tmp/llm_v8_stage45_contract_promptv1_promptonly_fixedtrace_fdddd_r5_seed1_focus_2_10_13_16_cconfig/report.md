# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | fdddd | 5 | 4.800 | [0.0, 0.0, 12.0, 12.0, 0.0] | {"repair_all": 3, "repair_subset": 2} | {"": 3, "break_app_storage_permission": 2} | {"": 5} | 0.000 | 0.400 |
| 10 | canonical_pattern | fdddd | 5 | 11.000 | [6.0, 10.0, 10.0, 19.0, 10.0] | {"repair_subset": 2, "repair_all": 3} | {"": 4, "airplane_mode_on|bad_network_preference|bad_wifi_calling|data_mode_off|unseat_sim_card": 1} | {"": 1, "data_usage_exceeded|user_abroad_roaming_disabled_on": 4} | 0.000 | 0.800 |
| 13 | canonical_pattern | fdddd | 5 | 6.000 | [6.0, 6.0, 6.0, 6.0, 6.0] | {"repair_subset": 5} | {"": 5} | {"": 5} | 0.000 | 0.800 |
| 16 | canonical_pattern | fdddd | 5 | 0.000 | [0.0, 0.0, 0.0, 0.0, 0.0] | {"repair_all": 5} | {"": 5} | {"": 5} | 0.000 | 0.000 |
