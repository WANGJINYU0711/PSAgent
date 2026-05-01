# llm_v8 ps update stability seed1 probfloor0002

Date: 2026-04-30

Experiment name: `llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_probfloor0002`

Output directory: `tmp/llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_probfloor0002`

## Setting

| field | value |
| --- | --- |
| model | gpt-4o-mini |
| executor | llm_bench |
| family | shared_basin_strong_prefix_dedup_profile_switch |
| schedule | trap_switch |
| d / switch denominator | 4.000 |
| eta | 0.300 |
| epsilon | 0.010 |
| repeats | 10.000 |
| horizon per method | 100.000 |
| switch episode | 25.000 |
| methods | risky_ps |
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | legacy term | reasoning | mode cost | path | total | exact | mode exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | all | 100.000 | 5.730 | 4.200 | 4.492 | 1.625 | 0.071 | 10.293 | 0.690 | 0.140 | 0.690 | 0.740 | 0.660 | 0.740 | 1.030 |
| risky_ps | pre | 25.000 | 0.000 | 0.000 | 4.925 | 1.560 | 0.069 | 4.995 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.120 |
| risky_ps | post | 75.000 | 7.640 | 5.600 | 4.347 | 1.647 | 0.071 | 12.059 | 0.587 | 0.187 | 0.587 | 0.653 | 0.547 | 0.987 | 0.333 |
| risky_ps | post_local_nontransfer | 75.000 | 7.640 | 5.600 | 4.347 | 1.647 | 0.071 | 12.059 | 0.587 | 0.187 | 0.587 | 0.653 | 0.547 | 0.987 | 0.333 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | risky_ps | 10.293 | 5.730 | 4.492 | 1.625 | 0.071 | 0.690 | 0.140 | 0.660 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | all_stage_modes_match | 14.000 | 9.679 | 3.722 | 0.000 | 13.476 | 0.500 | 0.714 | 0.500 | 0.000 | 0.000 |
| risky_ps | both_mismatch_types | 17.000 | 8.265 | 4.787 | 2.971 | 13.119 | 0.529 | 0.588 | 0.529 | 1.647 | 1.000 |
| risky_ps | deep_on_fast_required | 33.000 | 1.652 | 4.873 | 1.303 | 6.595 | 0.909 | 0.909 | 0.848 | 0.000 | 2.606 |
| risky_ps | fast_on_deep_required | 36.000 | 6.736 | 4.303 | 1.917 | 11.110 | 0.639 | 0.667 | 0.611 | 1.278 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | mostly_deep_vs_mostly_deep_required | 63.000 | 8.262 | 4.275 | 1.325 | 12.609 | 0.556 | 0.619 | 0.508 |
| risky_ps | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.747 | 1.938 | 5.820 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_fast_vs_mostly_deep_required | 12.000 | 4.375 | 4.727 | 3.333 | 9.168 | 0.750 | 0.833 | 0.750 |
| risky_ps | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.465 | 0.889 | 3.528 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | target_post_switch | mostly_deep_vs_mostly_deep_required | 63.000 | 8.262 | 4.275 | 1.325 | 12.609 | 0.508 |
| risky_ps | target_post_switch | mostly_fast_vs_mostly_deep_required | 12.000 | 4.375 | 4.727 | 3.333 | 9.168 | 0.750 |
| risky_ps | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.747 | 1.938 | 5.820 | 1.000 |
| risky_ps | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.465 | 0.889 | 3.528 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | deep | deep | 226.000 | 75.000 | 7.741 | 4.265 | 1.327 | 12.078 | 0.584 | 0.659 | 0.535 |
| risky_ps | deep | fast | 74.000 | 53.000 | 7.331 | 4.600 | 2.622 | 12.000 | 0.595 | 0.635 | 0.581 |
| risky_ps | fast | deep | 103.000 | 50.000 | 1.893 | 5.193 | 1.879 | 7.157 | 0.893 | 0.903 | 0.874 |
| risky_ps | fast | fast | 97.000 | 72.000 | 3.897 | 4.194 | 1.289 | 8.160 | 0.794 | 0.835 | 0.784 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.748 | 1.000 | 3.818 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.432 | 2.500 | 5.507 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.090 | 1.500 | 9.168 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.511 | 1.500 | 4.588 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.340 | 1.000 | 3.404 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.079 | 0.500 | 3.135 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/fast/fast | False | deep_on_fast_required |
| risky_ps | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.129 | 1.500 | 4.195 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.860 | 1.000 | 3.923 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | False | deep_on_fast_required |
| risky_ps | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.710 | 1.000 | 3.775 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.363 | 1.000 | 3.424 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | False | deep_on_fast_required |
| risky_ps | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.110 | 1.500 | 4.188 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.738 | 1.000 | 3.804 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.106 | 2.000 | 9.180 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.328 | 2.000 | 5.401 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.994 | 2.000 | 5.069 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.008 | 2.500 | 6.082 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.668 | 2.000 | 4.747 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.465 | 2.000 | 5.532 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.811 | 0.500 | 2.869 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| risky_ps | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.049 | 1.500 | 4.111 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.714 | 2.000 | 4.790 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.537 | 1.000 | 3.598 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.101 | 2.000 | 9.175 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.051 | 2.500 | 6.125 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.191 | 2.000 | 5.263 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 25.000 | target_post_switch | repair_all | repair_subset | 12.000 | 5.068 | 3.500 | 17.137 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/deep | False | both_mismatch_types |
| risky_ps | 26.000 | target_post_switch | repair_subset | transfer | 23.500 | 4.701 | 2.000 | 28.271 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 27.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.064 | 0.000 | 26.637 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 28.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 4.968 | 1.500 | 18.043 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| risky_ps | 29.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.417 | 1.500 | 4.498 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.062 | 0.000 | 3.140 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.024 | 1.500 | 3.098 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.792 | 2.000 | 3.857 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 33.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.688 | 0.500 | 4.763 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.187 | 1.500 | 3.259 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 35.000 | target_post_switch | repair_all | transfer | 18.500 | 4.108 | 0.000 | 22.683 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 36.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.020 | 1.500 | 11.090 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 37.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.809 | 0.500 | 27.382 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 38.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.413 | 0.000 | 21.491 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 39.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.995 | 0.000 | 4.070 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.134 | 5.000 | 5.193 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/fast | False | both_mismatch_types |
| risky_ps | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.607 | 3.000 | 4.683 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| risky_ps | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.866 | 3.000 | 3.931 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 43.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.975 | 1.500 | 4.048 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.567 | 1.500 | 3.633 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 45.000 | target_post_switch | repair_all | repair_subset | 10.000 | 3.798 | 0.000 | 13.871 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 46.000 | target_post_switch | repair_subset | transfer | 23.500 | 4.483 | 3.000 | 28.046 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 47.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.628 | 1.500 | 27.194 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 48.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.526 | 1.500 | 21.594 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 49.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.658 | 3.000 | 4.721 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.030 | 1.500 | 3.104 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.341 | 3.000 | 4.402 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.921 | 1.500 | 3.996 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 53.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.669 | 0.500 | 4.743 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.344 | 3.500 | 4.410 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 55.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.677 | 1.500 | 3.743 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 56.000 | target_post_switch | repair_subset | transfer | 23.500 | 5.282 | 3.500 | 28.848 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | False | both_mismatch_types |
| risky_ps | 57.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.091 | 0.000 | 26.666 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 58.000 | target_post_switch | repair_subset | repair_subset | 19.000 | 5.486 | 3.500 | 24.550 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 59.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.762 | 0.500 | 4.840 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.043 | 1.500 | 4.114 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.620 | 3.000 | 4.688 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| risky_ps | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 10.660 | 1.500 | 10.737 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.322 | 3.000 | 5.394 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/fast/fast | False | fast_on_deep_required |
| risky_ps | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.443 | 5.000 | 5.508 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/fast | False | both_mismatch_types |
| risky_ps | 65.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.075 | 2.000 | 4.140 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 66.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 4.560 | 3.000 | 18.622 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 67.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.076 | 0.000 | 26.651 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 68.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.611 | 1.500 | 21.685 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 69.000 | target_post_switch | repair_all | repair_subset | 15.000 | 4.463 | 1.500 | 19.533 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.220 | 2.000 | 4.294 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.000 | 1.500 | 3.074 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.059 | 1.500 | 3.134 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.798 | 1.500 | 3.870 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 74.000 | target_post_switch | repair_all | transfer | 18.500 | 4.944 | 3.500 | 23.515 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/deep | False | both_mismatch_types |
| risky_ps | 75.000 | target_post_switch | repair_all | repair_subset | 10.000 | 4.541 | 0.500 | 14.615 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 76.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.499 | 0.500 | 10.574 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 77.000 | target_post_switch | repair_subset | transfer | 23.500 | 3.912 | 1.500 | 27.491 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 78.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.578 | 1.500 | 21.653 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 79.000 | target_post_switch | repair_all | transfer | 22.500 | 3.365 | 0.000 | 25.940 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
