# llm_v8 ps update stability seed1 eta_shared002

Date: 2026-04-30

Experiment name: `llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_eta_shared002`

Output directory: `tmp/llm_v8_ps_update_stability_cconfig_d4_eta03_eps001_10x10_seed1_eta_shared002`

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
| risky_ps | all | 100.000 | 5.800 | 3.725 | 4.593 | 1.690 | 0.070 | 10.464 | 0.670 | 0.210 | 0.670 | 0.720 | 0.620 | 0.820 | 0.920 |
| risky_ps | pre | 25.000 | 0.000 | 0.000 | 4.881 | 1.560 | 0.069 | 4.951 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.120 |
| risky_ps | post | 75.000 | 7.733 | 4.967 | 4.497 | 1.733 | 0.071 | 12.301 | 0.560 | 0.280 | 0.560 | 0.627 | 0.493 | 1.093 | 0.187 |
| risky_ps | post_local_nontransfer | 75.000 | 7.733 | 4.967 | 4.497 | 1.733 | 0.071 | 12.301 | 0.560 | 0.280 | 0.560 | 0.627 | 0.493 | 1.093 | 0.187 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | risky_ps | 10.464 | 5.800 | 4.593 | 1.690 | 0.070 | 0.670 | 0.210 | 0.620 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | all_stage_modes_match | 21.000 | 6.714 | 3.915 | 0.000 | 10.704 | 0.667 | 0.762 | 0.571 | 0.000 | 0.000 |
| risky_ps | both_mismatch_types | 9.000 | 11.278 | 5.882 | 3.167 | 17.227 | 0.333 | 0.556 | 0.333 | 1.778 | 1.000 |
| risky_ps | deep_on_fast_required | 30.000 | 0.800 | 4.799 | 1.383 | 5.669 | 0.967 | 0.967 | 0.933 | 0.000 | 2.767 |
| risky_ps | fast_on_deep_required | 40.000 | 7.838 | 4.506 | 2.475 | 12.412 | 0.525 | 0.550 | 0.475 | 1.650 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | mostly_deep_vs_mostly_deep_required | 57.000 | 7.526 | 4.376 | 1.061 | 11.975 | 0.596 | 0.684 | 0.509 |
| risky_ps | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.674 | 1.938 | 5.748 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_fast_vs_mostly_deep_required | 18.000 | 8.389 | 4.883 | 3.861 | 13.337 | 0.444 | 0.444 | 0.444 |
| risky_ps | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.471 | 0.889 | 3.534 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | target_post_switch | mostly_deep_vs_mostly_deep_required | 57.000 | 7.526 | 4.376 | 1.061 | 11.975 | 0.509 |
| risky_ps | target_post_switch | mostly_fast_vs_mostly_deep_required | 18.000 | 8.389 | 4.883 | 3.861 | 13.337 | 0.444 |
| risky_ps | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.674 | 1.938 | 5.748 | 1.000 |
| risky_ps | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.471 | 0.889 | 3.534 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | deep | deep | 218.000 | 73.000 | 7.273 | 4.352 | 1.165 | 11.697 | 0.606 | 0.674 | 0.523 |
| risky_ps | deep | fast | 82.000 | 49.000 | 8.957 | 4.884 | 3.244 | 13.908 | 0.439 | 0.500 | 0.415 |
| risky_ps | fast | deep | 92.000 | 39.000 | 1.364 | 5.293 | 1.848 | 6.728 | 0.924 | 0.946 | 0.913 |
| risky_ps | fast | fast | 108.000 | 83.000 | 4.208 | 4.264 | 1.435 | 8.541 | 0.759 | 0.787 | 0.722 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.719 | 1.000 | 3.789 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.420 | 2.500 | 5.495 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.688 | 1.500 | 7.765 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.466 | 1.500 | 4.543 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.335 | 1.000 | 3.399 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.159 | 0.500 | 3.215 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/fast/fast | False | deep_on_fast_required |
| risky_ps | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.119 | 1.500 | 4.185 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.860 | 1.000 | 3.923 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | False | deep_on_fast_required |
| risky_ps | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.710 | 1.000 | 3.774 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.363 | 1.000 | 3.424 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | False | deep_on_fast_required |
| risky_ps | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.414 | 1.500 | 4.493 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.736 | 1.000 | 3.802 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.105 | 2.000 | 9.179 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.376 | 2.000 | 5.449 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.994 | 2.000 | 5.069 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.008 | 2.500 | 6.082 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.666 | 2.000 | 4.745 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.464 | 2.000 | 5.532 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.811 | 0.500 | 2.868 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| risky_ps | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.047 | 1.500 | 4.109 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.004 | 2.000 | 5.081 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.548 | 1.000 | 3.609 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.108 | 2.000 | 9.181 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.726 | 2.500 | 5.800 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.181 | 2.000 | 5.253 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 25.000 | target_post_switch | repair_all | repair_subset | 12.000 | 5.438 | 3.500 | 17.508 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/deep | False | both_mismatch_types |
| risky_ps | 26.000 | target_post_switch | repair_subset | transfer | 23.500 | 4.697 | 2.000 | 28.267 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 27.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.080 | 0.000 | 26.653 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 28.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 4.717 | 1.500 | 16.791 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| risky_ps | 29.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.553 | 1.500 | 4.634 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.064 | 0.000 | 3.142 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.462 | 1.500 | 3.536 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 11.397 | 2.000 | 11.464 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 33.000 | target_post_switch | repair_all | transfer | 18.000 | 4.829 | 0.500 | 22.904 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.033 | 1.500 | 4.105 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 35.000 | target_post_switch | repair_all | transfer | 18.500 | 4.188 | 0.000 | 22.763 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 36.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.891 | 1.500 | 10.961 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 37.000 | target_post_switch | repair_subset | transfer | 23.500 | 4.657 | 2.000 | 28.227 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 38.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.414 | 0.000 | 21.488 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 39.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.018 | 0.000 | 4.092 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.886 | 3.000 | 4.954 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/fast | False | fast_on_deep_required |
| risky_ps | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.614 | 1.500 | 3.689 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.713 | 3.500 | 4.783 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | False | both_mismatch_types |
| risky_ps | 43.000 | target_post_switch | repair_all | repair_subset | 14.000 | 5.482 | 4.500 | 19.544 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/fast/fast | False | fast_on_deep_required |
| risky_ps | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.725 | 0.000 | 3.803 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 45.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.728 | 0.000 | 3.804 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 46.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 3.608 | 1.500 | 15.675 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 47.000 | target_post_switch | repair_subset | transfer | 23.500 | 3.911 | 1.500 | 27.481 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 48.000 | target_post_switch | repair_subset | repair_subset | 19.000 | 4.192 | 1.500 | 23.260 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 49.000 | target_post_switch | repair_all | repair_all | 10.000 | 4.055 | 1.500 | 14.120 | False | True | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.526 | 3.000 | 3.588 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.520 | 0.000 | 3.601 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.056 | 1.500 | 3.131 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 53.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.504 | 3.000 | 4.569 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.619 | 0.000 | 3.695 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 55.000 | target_post_switch | repair_all | transfer | 18.500 | 3.638 | 1.500 | 22.212 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 56.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.043 | 1.500 | 11.108 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 57.000 | target_post_switch | repair_subset | transfer | 23.500 | 3.799 | 0.000 | 27.374 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 58.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.946 | 1.500 | 22.021 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 59.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.737 | 3.000 | 4.811 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.084 | 0.000 | 3.163 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.470 | 1.500 | 3.551 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.663 | 0.500 | 3.740 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 63.000 | target_post_switch | repair_all | repair_subset | 14.000 | 5.901 | 5.000 | 19.964 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/fast | False | both_mismatch_types |
| risky_ps | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.413 | 3.500 | 5.483 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | False | both_mismatch_types |
| risky_ps | 65.000 | target_post_switch | repair_all | transfer | 18.500 | 5.243 | 3.500 | 23.809 | False | True | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 66.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.897 | 0.500 | 10.972 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 67.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 3.968 | 1.500 | 21.034 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 68.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.529 | 1.500 | 21.603 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 69.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.204 | 0.000 | 4.271 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.523 | 3.000 | 3.594 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.393 | 4.500 | 4.453 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/fast | False | fast_on_deep_required |
| risky_ps | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 10.671 | 1.500 | 10.748 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.673 | 1.500 | 3.739 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.542 | 1.500 | 3.611 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 75.000 | target_post_switch | repair_all | transfer | 18.500 | 4.144 | 1.500 | 22.711 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 76.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 3.755 | 0.000 | 9.834 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 77.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 6.102 | 0.000 | 21.177 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 78.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 5.097 | 3.000 | 22.164 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 79.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.204 | 0.000 | 4.282 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
