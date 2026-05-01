# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | ddddd | 3 | 4.000 | [0.0, 12.0, 0.0] | {"repair_all": 2, "repair_subset": 1} | {"": 2, "break_app_storage_permission": 1} | {"": 3} | 0.000 | 0.000 |
| 2 | canonical_pattern | fdddd | 3 | 14.167 | [18.5, 12.0, 12.0] | {"transfer": 1, "repair_subset": 2} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|unseat_sim_card": 1, "break_app_storage_permission": 2} | {"": 3} | 0.000 | 1.000 |
| 2 | canonical_pattern | ffddd | 3 | 8.000 | [12.0, 12.0, 0.0] | {"repair_subset": 2, "repair_all": 1} | {"break_app_storage_permission": 2, "": 1} | {"": 3} | 0.000 | 0.667 |
| 10 | canonical_pattern | ddddd | 3 | 8.667 | [6.0, 10.0, 10.0] | {"repair_subset": 1, "repair_all": 2} | {"": 3} | {"": 1, "data_usage_exceeded|user_abroad_roaming_disabled_on": 2} | 0.000 | 0.667 |
| 10 | canonical_pattern | fdddd | 3 | 8.667 | [10.0, 6.0, 10.0] | {"repair_all": 2, "repair_subset": 1} | {"": 3} | {"data_usage_exceeded|user_abroad_roaming_disabled_on": 2, "": 1} | 0.000 | 0.667 |
| 10 | canonical_pattern | ffddd | 3 | 12.000 | [12.0, 12.0, 12.0] | {"repair_subset": 3} | {"": 3} | {"data_usage_exceeded": 3} | 0.000 | 0.000 |
| 13 | canonical_pattern | ddddd | 3 | 6.000 | [6.0, 6.0, 6.0] | {"repair_subset": 3} | {"": 3} | {"": 3} | 0.000 | 0.667 |
| 13 | canonical_pattern | fdddd | 3 | 6.000 | [6.0, 6.0, 6.0] | {"repair_subset": 3} | {"": 3} | {"": 3} | 0.000 | 0.333 |
| 13 | canonical_pattern | ffddd | 3 | 19.667 | [19.0, 21.0, 19.0] | {"repair_subset": 3} | {"airplane_mode_on|bad_network_preference|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 2, "airplane_mode_on|bad_network_preference|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"data_usage_exceeded": 3} | 0.000 | 0.000 |
| 16 | canonical_pattern | ddddd | 3 | 0.000 | [0.0, 0.0, 0.0] | {"repair_all": 3} | {"": 3} | {"": 3} | 0.000 | 0.000 |
| 16 | canonical_pattern | fdddd | 3 | 0.000 | [0.0, 0.0, 0.0] | {"repair_all": 3} | {"": 3} | {"": 3} | 0.000 | 0.000 |
| 16 | canonical_pattern | ffddd | 3 | 0.000 | [0.0, 0.0, 0.0] | {"repair_all": 3} | {"": 3} | {"": 3} | 0.000 | 0.000 |
