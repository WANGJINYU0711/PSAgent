# Fixed Profile Trace Diagnostic

This run persists Stage 4/5 raw/normalized trace fields in `records.json`.

| dataset | group | pattern | n | terminal_mean | terminal_values | final_counts | selected_missing_counts | deferred_missing_counts | completion_pass_rate | normalizer_changed_rate |
|---:|---|---|---:|---:|---|---|---|---|---:|---:|
| 5 | canonical_pattern | ddddd | 3 | 6.000 | [6.0, 6.0, 6.0] | {"repair_subset": 3} | {"": 3} | {"": 3} | 0.000 | 0.000 |
| 8 | canonical_pattern | ddddd | 3 | 3.333 | [0.0, 10.0, 0.0] | {"repair_subset": 2, "repair_all": 1} | {"": 3} | {"": 2, "data_usage_exceeded": 1} | 0.000 | 0.333 |
| 11 | canonical_pattern | ddddd | 3 | 0.000 | [0.0, 0.0, 0.0] | {"repair_subset": 3} | {"": 3} | {"": 3} | 0.000 | 0.000 |
