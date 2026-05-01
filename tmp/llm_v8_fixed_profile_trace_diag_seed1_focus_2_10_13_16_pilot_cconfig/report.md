# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 2 | canonical_pattern | ddddd | 1 | 0.000 | [0.0] | {"repair_all": 1} | {"": 1} | {"": 1} | 0.000 | 0.000 |
| 2 | canonical_pattern | fdddd | 1 | 18.500 | [18.5] | {"transfer": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|unseat_sim_card": 1} | {"": 1} | 0.000 | 1.000 |
| 2 | canonical_pattern | ffddd | 1 | 18.500 | [18.5] | {"transfer": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|unseat_sim_card": 1} | {"": 1} | 0.000 | 1.000 |
| 2 | observed_high_terminal_exact_match | fdddd | 3 | 13.833 | [10.0, 13.0, 18.5] | {"repair_subset": 2, "transfer": 1} | {"airplane_mode_on|unseat_sim_card": 1, "airplane_mode_on|bad_network_preference|bad_wifi_calling|unseat_sim_card": 1, "airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_storage_permission|unseat_sim_card": 1} | {"": 3} | 0.000 | 1.000 |
| 10 | canonical_pattern | ddddd | 1 | 20.500 | [20.5] | {"transfer": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card": 1} | {"": 1} | 0.000 | 1.000 |
| 10 | canonical_pattern | fdddd | 1 | 12.000 | [12.0] | {"repair_subset": 1} | {"bad_wifi_calling": 1} | {"": 1} | 0.000 | 1.000 |
| 10 | canonical_pattern | ffddd | 1 | 23.500 | [23.5] | {"transfer": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card": 1} | {"data_usage_exceeded": 1} | 0.000 | 1.000 |
| 10 | observed_high_terminal_exact_match | fdddd | 3 | 13.333 | [6.0, 15.0, 19.0] | {"repair_subset": 3} | {"": 1, "airplane_mode_on|data_mode_off|unseat_sim_card": 1, "airplane_mode_on|bad_network_preference|bad_wifi_calling|data_mode_off|unseat_sim_card": 1} | {"": 1, "data_usage_exceeded|user_abroad_roaming_disabled_on": 2} | 0.000 | 1.000 |
| 13 | canonical_pattern | ddddd | 1 | 6.000 | [6.0] | {"repair_subset": 1} | {"": 1} | {"": 1} | 0.000 | 1.000 |
| 13 | canonical_pattern | fdddd | 1 | 6.000 | [6.0] | {"repair_subset": 1} | {"": 1} | {"": 1} | 0.000 | 1.000 |
| 13 | canonical_pattern | ffddd | 1 | 17.000 | [17.0] | {"repair_subset": 1} | {"airplane_mode_on|bad_network_preference|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"data_usage_exceeded": 1} | 0.000 | 1.000 |
| 13 | observed_high_terminal_exact_match | fdddd | 1 | 24.500 | [24.5] | {"transfer": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"": 1} | 0.000 | 1.000 |
| 16 | canonical_pattern | ddddd | 1 | 0.000 | [0.0] | {"repair_all": 1} | {"": 1} | {"": 1} | 0.000 | 0.000 |
| 16 | canonical_pattern | fdddd | 1 | 15.000 | [15.0] | {"repair_subset": 1} | {"airplane_mode_on|bad_network_preference|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"": 1} | 0.000 | 1.000 |
| 16 | canonical_pattern | ffddd | 1 | 22.500 | [22.5] | {"transfer": 1} | {"airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_sms_permission|data_mode_off|unseat_sim_card|user_abroad_roaming_enabled_off": 1} | {"": 1} | 0.000 | 1.000 |
| 16 | observed_high_terminal_exact_match | fdddd | 1 | 0.000 | [0.0] | {"repair_all": 1} | {"": 1} | {"": 1} | 0.000 | 0.000 |
