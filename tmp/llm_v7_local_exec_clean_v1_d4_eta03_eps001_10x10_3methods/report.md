# LLM v7 Local Exec Clean 10x4 3-Method Smoke

Date: 2026-04-29

Experiment name: `llm_v7_local_exec_clean_v1_d4_eta03_eps001_10x10_3methods`

Output directory: `tmp/llm_v7_local_exec_clean_v1_d4_eta03_eps001_10x10_3methods`

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
| horizon per method | 40.000 |
| switch episode | 10.000 |
| methods | direct_multistage_exp3, epsilon_exp3, risky_ps_linear |
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v1/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v1_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | reasoning | path | total | exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 40.000 | 6.550 | 5.428 | 0.070 | 12.048 | 0.475 | 0.475 | 0.425 | 0.325 | 0.875 | 1.200 |
| epsilon_exp3 | all | 40.000 | 7.188 | 5.316 | 0.071 | 12.574 | 0.400 | 0.400 | 0.400 | 0.350 | 0.750 | 1.250 |
| risky_ps_linear | all | 40.000 | 7.787 | 5.235 | 0.070 | 13.093 | 0.350 | 0.350 | 0.425 | 0.325 | 0.775 | 1.075 |
| direct_multistage_exp3 | pre | 10.000 | 0.000 | 5.145 | 0.070 | 5.215 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.100 |
| epsilon_exp3 | pre | 10.000 | 0.000 | 5.330 | 0.070 | 5.400 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.800 |
| risky_ps_linear | pre | 10.000 | 0.000 | 4.651 | 0.069 | 4.720 | 1.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.000 |
| direct_multistage_exp3 | post | 30.000 | 8.733 | 5.523 | 0.070 | 14.326 | 0.300 | 0.300 | 0.233 | 0.100 | 1.167 | 0.567 |
| epsilon_exp3 | post | 30.000 | 9.583 | 5.311 | 0.071 | 14.966 | 0.200 | 0.200 | 0.200 | 0.133 | 1.000 | 0.400 |
| risky_ps_linear | post | 30.000 | 10.383 | 5.430 | 0.070 | 15.883 | 0.133 | 0.133 | 0.233 | 0.100 | 1.033 | 0.433 |
| direct_multistage_exp3 | post_local_nontransfer | 30.000 | 8.733 | 5.523 | 0.070 | 14.326 | 0.300 | 0.300 | 0.233 | 0.100 | 1.167 | 0.567 |
| epsilon_exp3 | post_local_nontransfer | 30.000 | 9.583 | 5.311 | 0.071 | 14.966 | 0.200 | 0.200 | 0.200 | 0.133 | 1.000 | 0.400 |
| risky_ps_linear | post_local_nontransfer | 30.000 | 10.383 | 5.430 | 0.070 | 15.883 | 0.133 | 0.133 | 0.233 | 0.100 | 1.033 | 0.433 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | path | exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | direct_multistage_exp3 | 12.048 | 6.550 | 5.428 | 0.070 | 0.475 | 0.325 |
| 2.000 | epsilon_exp3 | 12.574 | 7.188 | 5.316 | 0.071 | 0.400 | 0.350 |
| 3.000 | risky_ps_linear | 13.093 | 7.787 | 5.235 | 0.070 | 0.350 | 0.325 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 3.000 | 8.833 | 5.376 | 14.282 | 0.333 | 0.333 | 0.000 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 9.000 | 8.389 | 5.829 | 14.285 | 0.111 | 0.333 | 0.111 | 2.111 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 18.000 | 3.944 | 5.346 | 9.364 | 0.778 | 0.667 | 0.611 | 0.000 | 2.167 |
| direct_multistage_exp3 | fast_on_deep_required | 10.000 | 8.900 | 5.231 | 14.197 | 0.300 | 0.100 | 0.100 | 1.600 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 6.000 | 9.917 | 4.956 | 14.949 | 0.333 | 0.333 | 0.333 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 8.000 | 8.500 | 5.616 | 14.184 | 0.125 | 0.125 | 0.125 | 1.750 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 14.000 | 3.714 | 5.481 | 9.267 | 0.786 | 0.786 | 0.714 | 0.000 | 3.000 |
| epsilon_exp3 | fast_on_deep_required | 12.000 | 9.000 | 5.103 | 14.172 | 0.167 | 0.167 | 0.083 | 1.333 | 0.000 |
| risky_ps_linear | all_stage_modes_match | 6.000 | 13.917 | 5.122 | 19.115 | 0.167 | 0.333 | 0.000 | 0.000 | 0.000 |
| risky_ps_linear | both_mismatch_types | 8.000 | 5.188 | 5.659 | 10.914 | 0.250 | 0.250 | 0.250 | 1.500 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 15.000 | 4.567 | 5.048 | 9.686 | 0.733 | 0.733 | 0.733 | 0.000 | 2.333 |
| risky_ps_linear | fast_on_deep_required | 11.000 | 10.727 | 5.243 | 16.037 | 0.000 | 0.182 | 0.000 | 1.727 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 22.000 | 7.659 | 5.484 | 13.215 | 0.409 | 0.273 | 0.136 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 7.000 | 0.000 | 5.720 | 5.790 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 8.000 | 11.688 | 5.631 | 17.380 | 0.000 | 0.125 | 0.000 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 3.000 | 0.000 | 3.802 | 3.873 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 24.000 | 10.146 | 5.225 | 15.443 | 0.250 | 0.250 | 0.167 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 10.000 | 0.000 | 5.330 | 5.400 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 6.000 | 7.333 | 5.657 | 13.055 | 0.000 | 0.000 | 0.000 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 24.000 | 11.583 | 5.357 | 17.013 | 0.167 | 0.292 | 0.125 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 8.000 | 0.000 | 4.879 | 4.950 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 6.000 | 5.583 | 5.721 | 11.367 | 0.000 | 0.000 | 0.000 |
| risky_ps_linear | mostly_fast_vs_mostly_fast_required | 2.000 | 0.000 | 3.738 | 3.800 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 22.000 | 7.659 | 5.484 | 13.215 | 0.136 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 8.000 | 11.688 | 5.631 | 17.380 | 0.000 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 7.000 | 0.000 | 5.720 | 5.790 | 1.000 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 3.000 | 0.000 | 3.802 | 3.873 | 1.000 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 24.000 | 10.146 | 5.225 | 15.443 | 0.167 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 6.000 | 7.333 | 5.657 | 13.055 | 0.000 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 10.000 | 0.000 | 5.330 | 5.400 | 1.000 |
| risky_ps_linear | target_post_switch | mostly_deep_vs_mostly_deep_required | 24.000 | 11.583 | 5.357 | 17.013 | 0.125 |
| risky_ps_linear | target_post_switch | mostly_fast_vs_mostly_deep_required | 6.000 | 5.583 | 5.721 | 11.367 | 0.000 |
| risky_ps_linear | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 8.000 | 0.000 | 4.879 | 4.950 | 1.000 |
| risky_ps_linear | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 2.000 | 0.000 | 3.738 | 3.800 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 85.000 | 30.000 | 8.494 | 5.496 | 14.062 | 0.376 | 0.259 | 0.118 |
| direct_multistage_exp3 | deep | fast | 35.000 | 19.000 | 9.314 | 5.588 | 14.967 | 0.114 | 0.171 | 0.057 |
| direct_multistage_exp3 | fast | deep | 48.000 | 27.000 | 3.052 | 5.576 | 8.699 | 0.750 | 0.750 | 0.688 |
| direct_multistage_exp3 | fast | fast | 32.000 | 23.000 | 3.609 | 4.851 | 8.530 | 0.719 | 0.656 | 0.625 |
| epsilon_exp3 | deep | deep | 90.000 | 29.000 | 10.133 | 5.261 | 15.467 | 0.233 | 0.233 | 0.156 |
| epsilon_exp3 | deep | fast | 30.000 | 20.000 | 7.933 | 5.462 | 13.462 | 0.100 | 0.100 | 0.067 |
| epsilon_exp3 | fast | deep | 50.000 | 22.000 | 2.400 | 5.431 | 7.902 | 0.800 | 0.800 | 0.780 |
| epsilon_exp3 | fast | fast | 30.000 | 26.000 | 5.583 | 5.144 | 10.797 | 0.533 | 0.533 | 0.500 |
| risky_ps_linear | deep | deep | 89.000 | 29.000 | 11.326 | 5.387 | 16.785 | 0.157 | 0.270 | 0.112 |
| risky_ps_linear | deep | fast | 31.000 | 19.000 | 7.677 | 5.552 | 13.294 | 0.065 | 0.129 | 0.065 |
| risky_ps_linear | fast | deep | 43.000 | 23.000 | 2.558 | 5.024 | 7.652 | 0.767 | 0.767 | 0.767 |
| risky_ps_linear | fast | fast | 37.000 | 27.000 | 5.446 | 4.850 | 10.364 | 0.568 | 0.649 | 0.541 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | total | clear | aux | strict | required | actual | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.431 | 4.502 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.599 | 4.666 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.929 | 8.999 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.848 | 4.927 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.158 | 4.224 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/fast/deep | deep_on_fast_required |
| direct_multistage_exp3 | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.706 | 3.780 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.112 | 9.178 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.010 | 4.081 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.964 | 4.036 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.689 | 3.757 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 10.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 4.679 | 7.741 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 11.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.709 | 18.773 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/fast | fast_on_deep_required |
| direct_multistage_exp3 | 12.000 | target_post_switch | repair_subset | transfer | 18.500 | 4.547 | 23.107 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/fast | fast_on_deep_required |
| direct_multistage_exp3 | 13.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.158 | 6.227 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 14.000 | target_post_switch | repair_subset | repair_all | 6.000 | 6.004 | 12.080 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 15.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.293 | 18.367 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 16.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 6.065 | 8.129 | True | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 17.000 | target_post_switch | repair_all | repair_subset | 11.000 | 6.243 | 17.309 | False | True | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/fast | both_mismatch_types |
| direct_multistage_exp3 | 18.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.298 | 7.370 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | fast_on_deep_required |
| direct_multistage_exp3 | 19.000 | target_post_switch | repair_subset | repair_subset | 11.000 | 5.445 | 16.500 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | fast_on_deep_required |
| direct_multistage_exp3 | 20.000 | target_post_switch | repair_subset | repair_subset | 5.000 | 6.083 | 11.144 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | both_mismatch_types |
| direct_multistage_exp3 | 21.000 | target_post_switch | repair_all | transfer | 18.500 | 5.402 | 23.978 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 22.000 | target_post_switch | repair_subset | repair_subset | 13.500 | 4.689 | 18.263 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 23.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.907 | 18.983 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 24.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 6.372 | 8.448 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 25.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.810 | 4.876 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 26.000 | target_post_switch | repair_subset | transfer | 18.500 | 5.026 | 23.595 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 27.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 4.853 | 19.931 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 28.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 6.425 | 8.501 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 29.000 | target_post_switch | repair_all | transfer | 18.500 | 5.795 | 24.370 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 30.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 5.574 | 8.647 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 31.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.188 | 7.264 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 32.000 | target_post_switch | repair_subset | transfer | 18.500 | 5.582 | 24.160 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 33.000 | target_post_switch | repair_all | repair_subset | 17.000 | 4.498 | 21.561 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | both_mismatch_types |
| direct_multistage_exp3 | 34.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.098 | 7.171 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 35.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.774 | 18.839 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/fast | fast_on_deep_required |
| direct_multistage_exp3 | 36.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 5.455 | 8.520 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 37.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.000 | 5.077 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 38.000 | target_post_switch | repair_subset | repair_subset | 5.000 | 6.746 | 11.808 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | both_mismatch_types |
| direct_multistage_exp3 | 39.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.967 | 19.042 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.963 | 4.029 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.231 | 4.294 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/fast | deep_on_fast_required |
| epsilon_exp3 | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.841 | 7.900 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/fast | deep_on_fast_required |
| epsilon_exp3 | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.690 | 4.768 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.986 | 5.062 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.320 | 5.395 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.018 | 9.088 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.297 | 4.371 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.411 | 4.481 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.545 | 4.615 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 10.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.870 | 6.940 | True | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 11.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.602 | 18.661 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 12.000 | target_post_switch | repair_subset | transfer | 19.500 | 5.156 | 24.722 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| epsilon_exp3 | 13.000 | target_post_switch | repair_all | transfer | 18.500 | 5.234 | 23.806 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 14.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 4.981 | 8.049 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 15.000 | target_post_switch | repair_subset | repair_subset | 5.000 | 5.763 | 10.833 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | fast_on_deep_required |
| epsilon_exp3 | 16.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 4.589 | 7.659 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 17.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.105 | 5.181 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| epsilon_exp3 | 18.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.720 | 7.797 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 19.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.602 | 18.669 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 20.000 | target_post_switch | repair_subset | transfer | 18.500 | 4.973 | 23.551 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| epsilon_exp3 | 21.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.401 | 4.477 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| epsilon_exp3 | 22.000 | target_post_switch | repair_subset | transfer | 18.500 | 5.637 | 24.215 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 23.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 5.264 | 20.331 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 24.000 | target_post_switch | repair_subset | repair_subset | 4.000 | 5.240 | 9.314 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| epsilon_exp3 | 25.000 | target_post_switch | repair_all | repair_subset | 5.000 | 6.257 | 11.313 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/fast | both_mismatch_types |
| epsilon_exp3 | 26.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 5.490 | 8.558 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | both_mismatch_types |
| epsilon_exp3 | 27.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.965 | 19.040 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 28.000 | target_post_switch | repair_subset | transfer | 18.500 | 4.992 | 23.571 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| epsilon_exp3 | 29.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.932 | 6.001 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| epsilon_exp3 | 30.000 | target_post_switch | repair_subset | transfer | 18.500 | 5.027 | 23.601 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| epsilon_exp3 | 31.000 | target_post_switch | repair_subset | repair_subset | 5.000 | 5.740 | 10.806 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | both_mismatch_types |
| epsilon_exp3 | 32.000 | target_post_switch | repair_subset | transfer | 19.500 | 5.150 | 24.726 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| epsilon_exp3 | 33.000 | target_post_switch | repair_all | transfer | 18.500 | 6.109 | 24.686 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| epsilon_exp3 | 34.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 4.690 | 7.764 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 35.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.903 | 18.975 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| epsilon_exp3 | 36.000 | target_post_switch | repair_subset | transfer | 19.500 | 4.540 | 24.113 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 37.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.808 | 4.881 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| epsilon_exp3 | 38.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 5.298 | 8.373 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| epsilon_exp3 | 39.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.294 | 18.360 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 7.000 |
| target_post_switch | repair_subset | 23.000 |
| trap_pre_switch | repair_all | 10.000 |
