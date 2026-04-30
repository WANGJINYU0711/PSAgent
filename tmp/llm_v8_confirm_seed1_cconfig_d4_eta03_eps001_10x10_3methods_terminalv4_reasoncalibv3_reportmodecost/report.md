# Confirmatory seed1 C config d4: risky_ps vs direct vs epsilon

Date: 2026-04-30

Experiment name: `llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost`

Output directory: `tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost`

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
| methods | risky_ps, direct_multistage_exp3, epsilon_exp3 |
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | legacy term | reasoning | mode cost | path | total | exact | mode exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 100.000 | 5.115 | 3.180 | 4.712 | 1.825 | 0.070 | 9.898 | 0.710 | 0.200 | 0.710 | 0.760 | 0.630 | 0.840 | 1.130 |
| epsilon_exp3 | all | 100.000 | 5.030 | 3.145 | 4.512 | 1.490 | 0.071 | 9.612 | 0.720 | 0.310 | 0.720 | 0.750 | 0.640 | 0.660 | 1.000 |
| risky_ps | all | 100.000 | 5.985 | 3.945 | 4.449 | 1.665 | 0.070 | 10.504 | 0.650 | 0.180 | 0.650 | 0.700 | 0.620 | 0.780 | 0.990 |
| direct_multistage_exp3 | pre | 25.000 | 0.720 | 0.180 | 5.341 | 1.720 | 0.071 | 6.132 | 0.960 | 0.000 | 0.960 | 1.000 | 0.960 | 0.000 | 3.440 |
| epsilon_exp3 | pre | 25.000 | 0.000 | 0.000 | 5.038 | 1.620 | 0.070 | 5.108 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.240 |
| risky_ps | pre | 25.000 | 0.000 | 0.000 | 4.637 | 1.560 | 0.069 | 4.706 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.120 |
| direct_multistage_exp3 | post | 75.000 | 6.580 | 4.180 | 4.503 | 1.860 | 0.070 | 11.153 | 0.627 | 0.267 | 0.627 | 0.680 | 0.520 | 1.120 | 0.360 |
| epsilon_exp3 | post | 75.000 | 6.707 | 4.193 | 4.336 | 1.447 | 0.071 | 11.114 | 0.627 | 0.413 | 0.627 | 0.667 | 0.520 | 0.880 | 0.253 |
| risky_ps | post | 75.000 | 7.980 | 5.260 | 4.387 | 1.700 | 0.070 | 12.436 | 0.533 | 0.240 | 0.533 | 0.600 | 0.493 | 1.040 | 0.280 |
| direct_multistage_exp3 | post_local_nontransfer | 75.000 | 6.580 | 4.180 | 4.503 | 1.860 | 0.070 | 11.153 | 0.627 | 0.267 | 0.627 | 0.680 | 0.520 | 1.120 | 0.360 |
| epsilon_exp3 | post_local_nontransfer | 75.000 | 6.707 | 4.193 | 4.336 | 1.447 | 0.071 | 11.114 | 0.627 | 0.413 | 0.627 | 0.667 | 0.520 | 0.880 | 0.253 |
| risky_ps | post_local_nontransfer | 75.000 | 7.980 | 5.260 | 4.387 | 1.700 | 0.070 | 12.436 | 0.533 | 0.240 | 0.533 | 0.600 | 0.493 | 1.040 | 0.280 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | epsilon_exp3 | 9.612 | 5.030 | 4.512 | 1.490 | 0.071 | 0.720 | 0.310 | 0.640 |
| 2.000 | direct_multistage_exp3 | 9.898 | 5.115 | 4.712 | 1.825 | 0.070 | 0.710 | 0.200 | 0.630 |
| 3.000 | risky_ps | 10.504 | 5.985 | 4.449 | 1.665 | 0.070 | 0.650 | 0.180 | 0.620 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 20.000 | 5.200 | 3.726 | 0.000 | 9.001 | 0.750 | 0.700 | 0.600 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 22.000 | 6.659 | 4.845 | 2.886 | 11.573 | 0.636 | 0.727 | 0.636 | 1.591 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 30.000 | 1.900 | 5.255 | 1.517 | 7.226 | 0.900 | 0.967 | 0.867 | 0.000 | 3.033 |
| direct_multistage_exp3 | fast_on_deep_required | 28.000 | 7.286 | 4.732 | 2.625 | 12.084 | 0.536 | 0.607 | 0.393 | 1.750 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 31.000 | 4.435 | 3.803 | 0.000 | 8.314 | 0.774 | 0.742 | 0.548 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 15.000 | 8.200 | 5.098 | 3.200 | 13.364 | 0.467 | 0.600 | 0.467 | 1.800 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 29.000 | 1.897 | 5.013 | 1.466 | 6.981 | 0.897 | 0.931 | 0.897 | 0.000 | 2.931 |
| epsilon_exp3 | fast_on_deep_required | 25.000 | 7.500 | 4.457 | 2.340 | 12.024 | 0.600 | 0.640 | 0.560 | 1.560 | 0.000 |
| risky_ps | all_stage_modes_match | 18.000 | 6.639 | 3.664 | 0.000 | 10.376 | 0.667 | 0.778 | 0.611 | 0.000 | 0.000 |
| risky_ps | both_mismatch_types | 13.000 | 10.615 | 4.977 | 2.923 | 15.660 | 0.308 | 0.308 | 0.308 | 1.615 | 1.000 |
| risky_ps | deep_on_fast_required | 33.000 | 2.409 | 4.641 | 1.303 | 7.121 | 0.879 | 0.909 | 0.879 | 0.000 | 2.606 |
| risky_ps | fast_on_deep_required | 36.000 | 7.264 | 4.475 | 2.375 | 11.807 | 0.556 | 0.611 | 0.500 | 1.583 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 58.000 | 6.767 | 4.291 | 1.241 | 11.130 | 0.621 | 0.690 | 0.517 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 22.000 | 0.000 | 5.632 | 1.864 | 5.704 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 17.000 | 5.941 | 5.225 | 3.971 | 11.231 | 0.647 | 0.647 | 0.529 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 3.000 | 6.000 | 3.210 | 0.667 | 9.270 | 0.667 | 1.000 | 0.667 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 60.000 | 6.683 | 4.161 | 0.825 | 10.917 | 0.617 | 0.683 | 0.500 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 17.000 | 0.000 | 5.601 | 1.912 | 5.673 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 15.000 | 6.800 | 5.039 | 3.933 | 11.902 | 0.667 | 0.600 | 0.600 |
| epsilon_exp3 | mostly_fast_vs_mostly_fast_required | 8.000 | 0.000 | 3.840 | 1.000 | 3.907 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_deep_vs_mostly_deep_required | 61.000 | 7.361 | 4.231 | 1.172 | 11.663 | 0.574 | 0.639 | 0.525 |
| risky_ps | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.270 | 1.938 | 5.343 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_fast_vs_mostly_deep_required | 14.000 | 10.679 | 5.065 | 4.000 | 15.806 | 0.357 | 0.429 | 0.357 |
| risky_ps | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.511 | 0.889 | 3.573 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 58.000 | 6.767 | 4.291 | 1.241 | 11.130 | 0.517 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 17.000 | 5.941 | 5.225 | 3.971 | 11.231 | 0.529 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 22.000 | 0.000 | 5.632 | 1.864 | 5.704 | 1.000 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 3.000 | 6.000 | 3.210 | 0.667 | 9.270 | 0.667 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 60.000 | 6.683 | 4.161 | 0.825 | 10.917 | 0.500 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 15.000 | 6.800 | 5.039 | 3.933 | 11.902 | 0.600 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 17.000 | 0.000 | 5.601 | 1.912 | 5.673 | 1.000 |
| epsilon_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 8.000 | 0.000 | 3.840 | 1.000 | 3.907 | 1.000 |
| risky_ps | target_post_switch | mostly_deep_vs_mostly_deep_required | 61.000 | 7.361 | 4.231 | 1.172 | 11.663 | 0.525 |
| risky_ps | target_post_switch | mostly_fast_vs_mostly_deep_required | 14.000 | 10.679 | 5.065 | 4.000 | 15.806 | 0.357 |
| risky_ps | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.270 | 1.938 | 5.343 | 1.000 |
| risky_ps | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.511 | 0.889 | 3.573 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 216.000 | 74.000 | 6.394 | 4.316 | 1.294 | 10.781 | 0.644 | 0.694 | 0.523 |
| direct_multistage_exp3 | deep | fast | 84.000 | 50.000 | 7.060 | 4.983 | 3.315 | 12.109 | 0.583 | 0.643 | 0.512 |
| direct_multistage_exp3 | fast | deep | 113.000 | 52.000 | 1.801 | 5.404 | 2.018 | 7.277 | 0.903 | 0.938 | 0.894 |
| direct_multistage_exp3 | fast | fast | 87.000 | 69.000 | 4.368 | 4.536 | 1.454 | 8.974 | 0.747 | 0.805 | 0.667 |
| epsilon_exp3 | deep | deep | 234.000 | 73.000 | 6.553 | 4.163 | 0.912 | 10.789 | 0.641 | 0.684 | 0.513 |
| epsilon_exp3 | deep | fast | 66.000 | 40.000 | 7.250 | 4.951 | 3.341 | 12.266 | 0.576 | 0.606 | 0.545 |
| epsilon_exp3 | fast | deep | 100.000 | 44.000 | 1.780 | 5.262 | 1.955 | 7.112 | 0.890 | 0.920 | 0.890 |
| epsilon_exp3 | fast | fast | 100.000 | 77.000 | 3.250 | 4.287 | 1.155 | 7.608 | 0.830 | 0.830 | 0.750 |
| risky_ps | deep | deep | 222.000 | 73.000 | 7.556 | 4.226 | 1.182 | 11.854 | 0.572 | 0.649 | 0.527 |
| risky_ps | deep | fast | 78.000 | 49.000 | 9.186 | 4.843 | 3.173 | 14.094 | 0.423 | 0.462 | 0.397 |
| risky_ps | fast | deep | 99.000 | 46.000 | 2.197 | 4.943 | 1.828 | 7.211 | 0.869 | 0.879 | 0.869 |
| risky_ps | fast | fast | 101.000 | 76.000 | 3.772 | 4.151 | 1.401 | 7.991 | 0.782 | 0.822 | 0.752 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.748 | 1.000 | 3.818 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.432 | 2.500 | 5.507 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.688 | 1.500 | 7.766 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.444 | 1.500 | 4.521 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.335 | 1.000 | 3.399 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.079 | 0.500 | 3.135 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/fast/fast | False | deep_on_fast_required |
| risky_ps | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.122 | 1.500 | 4.188 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.886 | 1.000 | 3.949 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | False | deep_on_fast_required |
| risky_ps | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.754 | 1.000 | 3.819 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.704 | 1.000 | 3.764 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | False | deep_on_fast_required |
| risky_ps | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.382 | 1.500 | 4.461 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.750 | 1.000 | 3.816 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.285 | 2.000 | 8.359 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.999 | 2.000 | 5.073 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.030 | 2.000 | 5.104 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.381 | 2.500 | 6.454 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.670 | 2.000 | 4.749 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.860 | 2.000 | 4.928 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.805 | 0.500 | 2.863 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| risky_ps | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.049 | 1.500 | 4.111 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.725 | 2.000 | 4.801 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.537 | 1.000 | 3.598 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.334 | 2.000 | 4.407 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.724 | 2.500 | 5.798 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.192 | 2.000 | 5.264 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 25.000 | target_post_switch | repair_all | repair_subset | 12.000 | 5.067 | 3.500 | 17.137 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/deep | False | both_mismatch_types |
| risky_ps | 26.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 4.772 | 2.000 | 16.842 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 27.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.129 | 0.000 | 10.202 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 28.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 4.534 | 1.500 | 19.608 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| risky_ps | 29.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.623 | 1.500 | 4.704 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.536 | 0.000 | 3.614 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.490 | 1.500 | 3.564 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.798 | 2.000 | 3.864 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 33.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.673 | 0.500 | 4.748 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.670 | 1.500 | 3.742 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 35.000 | target_post_switch | repair_all | repair_subset | 10.000 | 4.964 | 0.500 | 15.040 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 36.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.739 | 1.500 | 27.309 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 37.000 | target_post_switch | repair_subset | transfer | 23.500 | 5.096 | 3.500 | 28.660 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 38.000 | target_post_switch | repair_subset | repair_subset | 19.000 | 4.707 | 3.000 | 23.780 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 39.000 | target_post_switch | repair_all | repair_subset | 15.000 | 4.422 | 0.000 | 19.497 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.061 | 3.000 | 4.125 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | False | fast_on_deep_required |
| risky_ps | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.603 | 1.500 | 3.668 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.111 | 0.000 | 3.185 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 43.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.578 | 1.500 | 3.647 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.550 | 1.500 | 3.624 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 45.000 | target_post_switch | repair_all | repair_subset | 14.000 | 5.850 | 6.000 | 19.908 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/fast/fast | False | fast_on_deep_required |
| risky_ps | 46.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 6.107 | 3.500 | 18.172 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | False | both_mismatch_types |
| risky_ps | 47.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.834 | 0.500 | 27.408 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 48.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 5.708 | 4.500 | 20.770 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/fast/fast | False | fast_on_deep_required |
| risky_ps | 49.000 | target_post_switch | repair_all | repair_subset | 14.000 | 5.671 | 3.000 | 19.738 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| risky_ps | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.404 | 4.500 | 4.464 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/fast | False | fast_on_deep_required |
| risky_ps | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.079 | 0.000 | 3.154 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.468 | 0.000 | 3.539 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 53.000 | target_post_switch | repair_all | repair_subset | 14.000 | 4.454 | 3.000 | 18.518 | False | True | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.620 | 0.000 | 3.696 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 55.000 | target_post_switch | repair_all | repair_subset | 10.000 | 3.775 | 1.500 | 13.849 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 56.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.030 | 1.500 | 11.095 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps | 57.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.095 | 0.000 | 26.670 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 58.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.302 | 1.500 | 10.377 | True | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 59.000 | target_post_switch | repair_all | repair_subset | 21.000 | 4.022 | 4.500 | 25.078 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | False | fast_on_deep_required |
| risky_ps | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.285 | 3.500 | 4.357 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.508 | 0.000 | 3.576 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.673 | 3.500 | 4.736 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.560 | 4.500 | 5.616 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/fast | False | fast_on_deep_required |
| risky_ps | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.423 | 0.000 | 3.491 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 65.000 | target_post_switch | repair_all | repair_subset | 10.000 | 3.638 | 0.000 | 13.713 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 66.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 6.721 | 5.000 | 20.783 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/fast | False | both_mismatch_types |
| risky_ps | 67.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.622 | 1.500 | 27.188 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 68.000 | target_post_switch | repair_subset | transfer | 25.500 | 3.751 | 0.000 | 29.330 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 69.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.947 | 0.500 | 5.023 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.189 | 2.000 | 4.265 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.003 | 1.500 | 3.069 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.058 | 1.500 | 3.120 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.547 | 0.000 | 3.616 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.723 | 0.000 | 3.799 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 75.000 | target_post_switch | repair_all | repair_subset | 10.000 | 4.639 | 1.500 | 14.708 | False | True | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps | 76.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.869 | 0.500 | 27.445 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps | 77.000 | target_post_switch | repair_subset | transfer | 23.500 | 4.659 | 2.000 | 28.235 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps | 78.000 | target_post_switch | repair_subset | repair_subset | 18.000 | 3.753 | 0.000 | 21.828 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps | 79.000 | target_post_switch | repair_all | transfer | 22.500 | 3.952 | 0.000 | 26.530 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
