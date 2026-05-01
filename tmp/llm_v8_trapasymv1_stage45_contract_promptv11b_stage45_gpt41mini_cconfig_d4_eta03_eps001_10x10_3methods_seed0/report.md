# llm_v8_trapasymv1_seed0_3methods

Date: 2026-05-01

Experiment name: `llm_v8_trapasymv1_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0`

Output directory: `tmp/llm_v8_trapasymv1_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0`

## Setting

| field | value |
| --- | --- |
| model | gpt-4o-mini |
| executor | llm_bench |
| family | shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v1 |
| schedule | trap_switch |
| d / switch denominator | 4.000 |
| eta | 0.300 |
| epsilon | 0.010 |
| repeats | 10.000 |
| horizon per method | 100.000 |
| switch episode | 25.000 |
| methods | risky_ps, direct_multistage_exp3, epsilon_exp3 |
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | legacy term | reasoning | mode cost | path | total | exact | mode exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 100.000 | 3.640 | 1.695 | 5.236 | 1.930 | 0.066 | 8.941 | 0.790 | 0.240 | 0.790 | 0.690 | 0.690 | 0.990 | 0.890 |
| epsilon_exp3 | all | 100.000 | 4.000 | 2.030 | 5.466 | 2.245 | 0.065 | 9.531 | 0.740 | 0.160 | 0.740 | 0.680 | 0.680 | 1.180 | 0.950 |
| risky_ps | all | 100.000 | 2.520 | 0.900 | 5.518 | 1.395 | 0.070 | 8.108 | 0.830 | 0.140 | 0.830 | 0.770 | 0.770 | 0.440 | 1.470 |
| direct_multistage_exp3 | pre | 25.000 | 0.720 | 0.260 | 4.201 | 1.080 | 0.062 | 4.983 | 0.960 | 0.240 | 0.960 | 0.960 | 0.960 | 0.000 | 2.160 |
| epsilon_exp3 | pre | 25.000 | 0.000 | 0.000 | 4.608 | 1.280 | 0.064 | 4.671 | 1.000 | 0.160 | 1.000 | 1.000 | 1.000 | 0.000 | 2.560 |
| risky_ps | pre | 25.000 | 0.000 | 0.000 | 5.659 | 1.960 | 0.070 | 5.729 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.920 |
| direct_multistage_exp3 | post | 75.000 | 4.613 | 2.173 | 5.581 | 2.213 | 0.067 | 10.261 | 0.733 | 0.240 | 0.733 | 0.600 | 0.600 | 1.320 | 0.467 |
| epsilon_exp3 | post | 75.000 | 5.333 | 2.707 | 5.752 | 2.567 | 0.066 | 11.151 | 0.653 | 0.160 | 0.653 | 0.573 | 0.573 | 1.573 | 0.413 |
| risky_ps | post | 75.000 | 3.360 | 1.200 | 5.471 | 1.207 | 0.070 | 8.901 | 0.773 | 0.187 | 0.773 | 0.693 | 0.693 | 0.587 | 0.653 |
| direct_multistage_exp3 | post_local_nontransfer | 75.000 | 4.613 | 2.173 | 5.581 | 2.213 | 0.067 | 10.261 | 0.733 | 0.240 | 0.733 | 0.600 | 0.600 | 1.320 | 0.467 |
| epsilon_exp3 | post_local_nontransfer | 75.000 | 5.333 | 2.707 | 5.752 | 2.567 | 0.066 | 11.151 | 0.653 | 0.160 | 0.653 | 0.573 | 0.573 | 1.573 | 0.413 |
| risky_ps | post_local_nontransfer | 75.000 | 3.360 | 1.200 | 5.471 | 1.207 | 0.070 | 8.901 | 0.773 | 0.187 | 0.773 | 0.693 | 0.693 | 0.587 | 0.653 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | risky_ps | 8.108 | 2.520 | 5.518 | 1.395 | 0.070 | 0.830 | 0.140 | 0.770 |
| 2.000 | direct_multistage_exp3 | 8.941 | 3.640 | 5.236 | 1.930 | 0.066 | 0.790 | 0.240 | 0.690 |
| 3.000 | epsilon_exp3 | 9.531 | 4.000 | 5.466 | 2.245 | 0.065 | 0.740 | 0.160 | 0.680 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 24.000 | 2.000 | 4.527 | 0.000 | 6.594 | 0.958 | 0.708 | 0.708 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 26.000 | 6.115 | 6.040 | 3.500 | 12.220 | 0.577 | 0.577 | 0.577 | 2.000 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 28.000 | 1.286 | 4.948 | 1.125 | 6.301 | 0.964 | 0.857 | 0.857 | 0.000 | 2.250 |
| direct_multistage_exp3 | fast_on_deep_required | 22.000 | 5.500 | 5.426 | 3.205 | 10.987 | 0.636 | 0.591 | 0.591 | 2.136 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 16.000 | 0.750 | 4.177 | 0.000 | 4.996 | 1.000 | 0.875 | 0.875 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 20.000 | 6.950 | 6.425 | 3.800 | 13.439 | 0.500 | 0.500 | 0.500 | 2.200 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 32.000 | 1.125 | 5.446 | 1.172 | 6.639 | 0.969 | 0.844 | 0.844 | 0.000 | 2.344 |
| epsilon_exp3 | fast_on_deep_required | 32.000 | 6.656 | 5.531 | 3.469 | 12.248 | 0.531 | 0.531 | 0.531 | 2.312 | 0.000 |
| risky_ps | all_stage_modes_match | 14.000 | 0.857 | 4.859 | 0.000 | 5.790 | 1.000 | 0.857 | 0.857 | 0.000 | 0.000 |
| risky_ps | both_mismatch_types | 20.000 | 7.500 | 6.195 | 2.675 | 13.762 | 0.450 | 0.400 | 0.400 | 1.450 | 1.000 |
| risky_ps | deep_on_fast_required | 54.000 | 0.556 | 5.513 | 1.176 | 6.139 | 0.981 | 0.926 | 0.926 | 0.000 | 2.352 |
| risky_ps | fast_on_deep_required | 12.000 | 5.000 | 5.184 | 1.875 | 10.250 | 0.583 | 0.583 | 0.583 | 1.250 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 53.000 | 2.698 | 5.356 | 1.255 | 8.124 | 0.887 | 0.698 | 0.698 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 9.000 | 2.000 | 5.603 | 2.111 | 7.674 | 0.889 | 0.889 | 0.889 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 22.000 | 9.227 | 6.123 | 4.523 | 15.408 | 0.364 | 0.364 | 0.364 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 16.000 | 0.000 | 3.413 | 0.500 | 3.470 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 43.000 | 2.791 | 5.548 | 1.105 | 8.410 | 0.837 | 0.698 | 0.698 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 12.000 | 0.000 | 5.744 | 2.042 | 5.814 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 32.000 | 8.750 | 6.026 | 4.531 | 14.835 | 0.406 | 0.406 | 0.406 |
| epsilon_exp3 | mostly_fast_vs_mostly_fast_required | 13.000 | 0.000 | 3.558 | 0.577 | 3.616 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_deep_vs_mostly_deep_required | 70.000 | 3.157 | 5.438 | 1.000 | 8.666 | 0.786 | 0.700 | 0.700 |
| risky_ps | mostly_deep_vs_mostly_fast_required | 21.000 | 0.000 | 6.031 | 2.190 | 6.102 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_fast_vs_mostly_deep_required | 5.000 | 6.200 | 5.932 | 4.100 | 12.192 | 0.600 | 0.600 | 0.600 |
| risky_ps | mostly_fast_vs_mostly_fast_required | 4.000 | 0.000 | 3.709 | 0.750 | 3.772 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 53.000 | 2.698 | 5.356 | 1.255 | 8.124 | 0.698 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 22.000 | 9.227 | 6.123 | 4.523 | 15.408 | 0.364 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 9.000 | 2.000 | 5.603 | 2.111 | 7.674 | 0.889 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 16.000 | 0.000 | 3.413 | 0.500 | 3.470 | 1.000 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 43.000 | 2.791 | 5.548 | 1.105 | 8.410 | 0.698 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 32.000 | 8.750 | 6.026 | 4.531 | 14.835 | 0.406 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 12.000 | 0.000 | 5.744 | 2.042 | 5.814 | 1.000 |
| epsilon_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 13.000 | 0.000 | 3.558 | 0.577 | 3.616 | 1.000 |
| risky_ps | target_post_switch | mostly_deep_vs_mostly_deep_required | 70.000 | 3.157 | 5.438 | 1.000 | 8.666 | 0.700 |
| risky_ps | target_post_switch | mostly_fast_vs_mostly_deep_required | 5.000 | 6.200 | 5.932 | 4.100 | 12.192 | 0.600 |
| risky_ps | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 21.000 | 0.000 | 6.031 | 2.190 | 6.102 | 1.000 |
| risky_ps | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 4.000 | 0.000 | 3.709 | 0.750 | 3.772 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 201.000 | 67.000 | 2.811 | 5.330 | 1.256 | 8.211 | 0.881 | 0.687 | 0.687 |
| direct_multistage_exp3 | deep | fast | 99.000 | 48.000 | 8.273 | 6.089 | 4.157 | 14.423 | 0.434 | 0.424 | 0.424 |
| direct_multistage_exp3 | fast | deep | 89.000 | 54.000 | 2.596 | 5.475 | 2.163 | 8.138 | 0.843 | 0.809 | 0.809 |
| direct_multistage_exp3 | fast | fast | 111.000 | 61.000 | 1.847 | 4.112 | 0.977 | 6.020 | 0.901 | 0.838 | 0.838 |
| epsilon_exp3 | deep | deep | 182.000 | 63.000 | 3.038 | 5.468 | 1.357 | 8.576 | 0.824 | 0.692 | 0.692 |
| epsilon_exp3 | deep | fast | 118.000 | 52.000 | 8.873 | 6.191 | 4.432 | 15.124 | 0.390 | 0.390 | 0.390 |
| epsilon_exp3 | fast | deep | 95.000 | 52.000 | 1.842 | 5.755 | 2.089 | 7.664 | 0.884 | 0.842 | 0.842 |
| epsilon_exp3 | fast | fast | 105.000 | 65.000 | 2.143 | 4.387 | 1.467 | 6.592 | 0.857 | 0.838 | 0.838 |
| risky_ps | deep | deep | 256.000 | 74.000 | 2.660 | 5.388 | 0.912 | 8.119 | 0.828 | 0.738 | 0.738 |
| risky_ps | deep | fast | 44.000 | 32.000 | 7.432 | 5.955 | 2.920 | 13.452 | 0.455 | 0.432 | 0.432 |
| risky_ps | fast | deep | 147.000 | 74.000 | 1.224 | 5.880 | 1.898 | 7.175 | 0.918 | 0.891 | 0.891 |
| risky_ps | fast | fast | 53.000 | 41.000 | 1.358 | 4.781 | 1.066 | 6.208 | 0.906 | 0.868 | 0.868 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.363 | 2.500 | 6.437 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.334 | 2.500 | 6.407 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.817 | 2.000 | 5.887 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.525 | 0.500 | 3.584 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/fast | False | deep_on_fast_required |
| risky_ps | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.380 | 2.000 | 5.453 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.213 | 0.500 | 3.275 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| risky_ps | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.279 | 2.500 | 8.347 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.317 | 2.000 | 5.388 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.350 | 1.000 | 4.416 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.454 | 2.500 | 6.526 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.492 | 2.000 | 5.568 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.037 | 2.500 | 6.106 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.428 | 1.500 | 5.494 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.806 | 2.500 | 5.880 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.066 | 2.000 | 5.140 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.748 | 1.000 | 3.811 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/fast/fast | False | deep_on_fast_required |
| risky_ps | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.307 | 2.500 | 7.377 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.559 | 2.000 | 7.628 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.828 | 1.500 | 4.892 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.818 | 2.500 | 5.890 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.776 | 2.500 | 6.846 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.601 | 2.000 | 5.675 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.930 | 2.000 | 6.000 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.811 | 2.500 | 5.882 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.249 | 2.000 | 5.321 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 25.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.984 | 0.000 | 5.062 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 26.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.566 | 0.500 | 11.635 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 27.000 | target_post_switch | repair_subset | repair_subset | 0.000 | 7.333 | 0.000 | 7.404 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 28.000 | target_post_switch | repair_subset | repair_all | 12.000 | 5.455 | 2.000 | 17.526 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 29.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.470 | 0.000 | 4.542 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.174 | 0.500 | 4.242 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.445 | 0.000 | 3.523 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.167 | 3.500 | 5.232 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 33.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.993 | 0.000 | 4.067 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.728 | 0.500 | 4.802 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 35.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.188 | 0.500 | 6.258 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 36.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 7.292 | 2.000 | 19.359 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 37.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 6.026 | 0.500 | 12.098 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 38.000 | target_post_switch | repair_subset | repair_all | 12.000 | 6.347 | 3.500 | 18.412 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 39.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.379 | 0.500 | 5.454 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.676 | 2.000 | 4.749 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.506 | 2.000 | 4.572 | True | True | True | fast/deep/deep/deep/deep | deep/deep/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.779 | 0.500 | 4.850 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 43.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.163 | 3.000 | 5.221 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.499 | 0.500 | 5.572 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 45.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.431 | 0.000 | 5.506 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 46.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 6.639 | 0.500 | 12.713 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 47.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 8.384 | 2.000 | 20.456 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 48.000 | target_post_switch | repair_subset | repair_all | 12.000 | 5.753 | 0.500 | 17.823 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 49.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.635 | 0.000 | 4.713 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 50.000 | target_post_switch | repair_all | repair_subset | 14.000 | 6.839 | 6.500 | 20.897 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/fast | False | both_mismatch_types |
| risky_ps | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.446 | 0.000 | 3.515 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.773 | 0.500 | 4.846 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 53.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.862 | 2.000 | 4.932 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.895 | 0.500 | 4.970 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 55.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.145 | 0.500 | 6.219 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 56.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 7.978 | 2.000 | 20.050 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 57.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 5.001 | 1.500 | 17.074 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 58.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 7.196 | 5.000 | 24.257 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/fast | False | both_mismatch_types |
| risky_ps | 59.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.185 | 0.500 | 5.260 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.737 | 0.500 | 4.803 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.694 | 2.000 | 4.766 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.983 | 1.500 | 4.052 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.731 | 0.500 | 4.801 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.172 | 0.000 | 4.246 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 65.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.945 | 2.000 | 6.009 | True | True | True | fast/deep/deep/deep/deep | deep/deep/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 66.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 6.542 | 1.500 | 18.613 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 67.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 7.389 | 0.000 | 13.465 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 68.000 | target_post_switch | repair_subset | repair_all | 12.000 | 5.512 | 1.500 | 17.580 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 69.000 | target_post_switch | repair_all | repair_subset | 17.000 | 7.232 | 3.500 | 24.292 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/fast/fast | False | both_mismatch_types |
| risky_ps | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.754 | 0.500 | 4.825 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.458 | 1.500 | 4.524 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.328 | 1.500 | 4.394 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.942 | 0.500 | 6.012 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.128 | 0.500 | 6.203 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 75.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.517 | 0.500 | 5.589 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 76.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 5.889 | 2.000 | 17.955 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 77.000 | target_post_switch | repair_subset | repair_subset | 0.000 | 7.539 | 0.500 | 7.607 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 78.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 7.729 | 2.000 | 13.796 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/fast | False | both_mismatch_types |
| risky_ps | 79.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.511 | 2.000 | 6.579 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/fast | False | both_mismatch_types |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
