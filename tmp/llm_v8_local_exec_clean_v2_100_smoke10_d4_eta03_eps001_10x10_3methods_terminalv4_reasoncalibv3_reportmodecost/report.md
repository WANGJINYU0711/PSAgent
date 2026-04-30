# llm_v8 C terminalv4 + reasoning calibration v3 + report-only modecost

Date: 2026-04-29

Experiment name: `llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost`

Output directory: `tmp/llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost`

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
| methods | direct_multistage_exp3, epsilon_exp3, risky_ps_linear |
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | legacy term | reasoning | mode cost | path | total | exact | mode exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 100.000 | 4.910 | 3.045 | 4.551 | 1.340 | 0.071 | 9.532 | 0.730 | 0.180 | 0.730 | 0.780 | 0.640 | 0.550 | 1.030 |
| epsilon_exp3 | all | 100.000 | 5.605 | 3.385 | 4.821 | 1.810 | 0.070 | 10.495 | 0.670 | 0.150 | 0.670 | 0.730 | 0.620 | 0.850 | 1.070 |
| risky_ps_linear | all | 100.000 | 5.160 | 3.430 | 4.472 | 1.300 | 0.071 | 9.703 | 0.700 | 0.170 | 0.700 | 0.780 | 0.650 | 0.500 | 1.100 |
| direct_multistage_exp3 | pre | 25.000 | 0.400 | 0.120 | 5.261 | 1.460 | 0.069 | 5.730 | 0.960 | 0.000 | 0.960 | 1.000 | 0.960 | 0.000 | 2.920 |
| epsilon_exp3 | pre | 25.000 | 0.000 | 0.000 | 6.402 | 1.740 | 0.070 | 6.473 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.480 |
| risky_ps_linear | pre | 25.000 | 0.400 | 0.120 | 5.151 | 1.540 | 0.069 | 5.620 | 0.960 | 0.000 | 0.960 | 1.000 | 0.960 | 0.000 | 3.080 |
| direct_multistage_exp3 | post | 75.000 | 6.413 | 4.020 | 4.314 | 1.300 | 0.072 | 10.800 | 0.653 | 0.240 | 0.653 | 0.707 | 0.533 | 0.733 | 0.400 |
| epsilon_exp3 | post | 75.000 | 7.473 | 4.513 | 4.293 | 1.833 | 0.069 | 11.836 | 0.560 | 0.200 | 0.560 | 0.640 | 0.493 | 1.133 | 0.267 |
| risky_ps_linear | post | 75.000 | 6.747 | 4.533 | 4.246 | 1.220 | 0.072 | 11.064 | 0.613 | 0.227 | 0.613 | 0.707 | 0.547 | 0.667 | 0.440 |
| direct_multistage_exp3 | post_local_nontransfer | 75.000 | 6.413 | 4.020 | 4.314 | 1.300 | 0.072 | 10.800 | 0.653 | 0.240 | 0.653 | 0.707 | 0.533 | 0.733 | 0.400 |
| epsilon_exp3 | post_local_nontransfer | 75.000 | 7.473 | 4.513 | 4.293 | 1.833 | 0.069 | 11.836 | 0.560 | 0.200 | 0.560 | 0.640 | 0.493 | 1.133 | 0.267 |
| risky_ps_linear | post_local_nontransfer | 75.000 | 6.747 | 4.533 | 4.246 | 1.220 | 0.072 | 11.064 | 0.613 | 0.227 | 0.613 | 0.707 | 0.547 | 0.667 | 0.440 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | direct_multistage_exp3 | 9.532 | 4.910 | 4.551 | 1.340 | 0.071 | 0.730 | 0.180 | 0.640 |
| 2.000 | risky_ps_linear | 9.703 | 5.160 | 4.472 | 1.300 | 0.071 | 0.700 | 0.170 | 0.650 |
| 3.000 | epsilon_exp3 | 10.495 | 5.605 | 4.821 | 1.810 | 0.070 | 0.670 | 0.150 | 0.620 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 18.000 | 5.722 | 3.759 | 0.000 | 9.555 | 0.722 | 0.667 | 0.444 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 12.000 | 5.417 | 4.798 | 2.750 | 10.285 | 0.667 | 0.833 | 0.667 | 1.500 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 43.000 | 2.372 | 4.954 | 1.058 | 7.398 | 0.860 | 0.930 | 0.814 | 0.000 | 2.116 |
| direct_multistage_exp3 | fast_on_deep_required | 27.000 | 8.185 | 4.327 | 2.056 | 12.581 | 0.556 | 0.593 | 0.481 | 1.370 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 15.000 | 10.133 | 3.742 | 0.000 | 13.950 | 0.533 | 0.667 | 0.400 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 14.000 | 5.000 | 4.677 | 2.857 | 9.744 | 0.571 | 0.714 | 0.571 | 1.571 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 31.000 | 1.306 | 5.991 | 1.500 | 7.369 | 0.935 | 0.935 | 0.871 | 0.000 | 3.000 |
| epsilon_exp3 | fast_on_deep_required | 40.000 | 7.450 | 4.368 | 2.362 | 11.885 | 0.550 | 0.600 | 0.525 | 1.575 | 0.000 |
| risky_ps_linear | all_stage_modes_match | 17.000 | 9.882 | 3.910 | 0.000 | 13.865 | 0.471 | 0.588 | 0.353 | 0.000 | 0.000 |
| risky_ps_linear | both_mismatch_types | 11.000 | 11.136 | 4.845 | 2.682 | 16.050 | 0.455 | 0.545 | 0.455 | 1.455 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 47.000 | 1.628 | 4.783 | 1.053 | 6.483 | 0.894 | 0.936 | 0.830 | 0.000 | 2.106 |
| risky_ps_linear | fast_on_deep_required | 25.000 | 5.960 | 4.106 | 2.040 | 10.135 | 0.600 | 0.720 | 0.600 | 1.360 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 66.000 | 5.962 | 4.196 | 0.992 | 10.231 | 0.682 | 0.742 | 0.561 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 17.000 | 0.588 | 5.837 | 1.765 | 6.495 | 0.941 | 1.000 | 0.941 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 9.000 | 9.722 | 5.185 | 3.556 | 14.971 | 0.444 | 0.444 | 0.333 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 8.000 | 0.000 | 4.036 | 0.812 | 4.103 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 56.000 | 6.527 | 4.020 | 1.205 | 10.619 | 0.625 | 0.714 | 0.536 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 20.000 | 0.000 | 6.602 | 1.925 | 6.673 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 19.000 | 10.263 | 5.097 | 3.684 | 15.424 | 0.368 | 0.421 | 0.368 |
| epsilon_exp3 | mostly_fast_vs_mostly_fast_required | 5.000 | 0.000 | 5.605 | 1.000 | 5.671 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 67.000 | 6.239 | 4.210 | 0.985 | 10.521 | 0.642 | 0.731 | 0.567 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 19.000 | 0.526 | 5.595 | 1.763 | 6.193 | 0.947 | 1.000 | 0.947 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 8.000 | 11.000 | 4.544 | 3.188 | 15.610 | 0.375 | 0.500 | 0.375 |
| risky_ps_linear | mostly_fast_vs_mostly_fast_required | 6.000 | 0.000 | 3.744 | 0.833 | 3.806 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 66.000 | 5.962 | 4.196 | 0.992 | 10.231 | 0.561 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 9.000 | 9.722 | 5.185 | 3.556 | 14.971 | 0.333 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 17.000 | 0.588 | 5.837 | 1.765 | 6.495 | 0.941 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 8.000 | 0.000 | 4.036 | 0.812 | 4.103 | 1.000 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 56.000 | 6.527 | 4.020 | 1.205 | 10.619 | 0.536 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 19.000 | 10.263 | 5.097 | 3.684 | 15.424 | 0.368 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 20.000 | 0.000 | 6.602 | 1.925 | 6.673 | 1.000 |
| epsilon_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 5.000 | 0.000 | 5.605 | 1.000 | 5.671 | 1.000 |
| risky_ps_linear | target_post_switch | mostly_deep_vs_mostly_deep_required | 67.000 | 6.239 | 4.210 | 0.985 | 10.521 | 0.567 |
| risky_ps_linear | target_post_switch | mostly_fast_vs_mostly_deep_required | 8.000 | 11.000 | 4.544 | 3.188 | 15.610 | 0.375 |
| risky_ps_linear | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 19.000 | 0.526 | 5.595 | 1.763 | 6.193 | 0.947 |
| risky_ps_linear | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 6.000 | 0.000 | 3.744 | 0.833 | 3.806 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 245.000 | 75.000 | 6.192 | 4.223 | 0.986 | 10.488 | 0.669 | 0.727 | 0.539 |
| direct_multistage_exp3 | deep | fast | 55.000 | 39.000 | 7.400 | 4.722 | 2.700 | 12.189 | 0.582 | 0.618 | 0.509 |
| direct_multistage_exp3 | fast | deep | 103.000 | 55.000 | 1.913 | 5.361 | 1.568 | 7.345 | 0.874 | 0.951 | 0.854 |
| direct_multistage_exp3 | fast | fast | 97.000 | 70.000 | 3.443 | 4.422 | 1.222 | 7.935 | 0.814 | 0.825 | 0.742 |
| epsilon_exp3 | deep | deep | 215.000 | 75.000 | 7.244 | 4.132 | 1.370 | 11.447 | 0.591 | 0.674 | 0.502 |
| epsilon_exp3 | deep | fast | 85.000 | 54.000 | 8.053 | 4.702 | 3.006 | 12.821 | 0.482 | 0.553 | 0.471 |
| epsilon_exp3 | fast | deep | 107.000 | 45.000 | 1.033 | 6.131 | 1.930 | 7.235 | 0.925 | 0.944 | 0.907 |
| epsilon_exp3 | fast | fast | 93.000 | 76.000 | 4.839 | 5.014 | 1.597 | 9.921 | 0.731 | 0.774 | 0.699 |
| risky_ps_linear | deep | deep | 250.000 | 75.000 | 6.324 | 4.201 | 0.952 | 10.597 | 0.636 | 0.724 | 0.556 |
| risky_ps_linear | deep | fast | 50.000 | 36.000 | 8.860 | 4.470 | 2.560 | 13.398 | 0.500 | 0.620 | 0.500 |
| risky_ps_linear | fast | deep | 110.000 | 58.000 | 1.991 | 5.140 | 1.555 | 7.203 | 0.882 | 0.927 | 0.855 |
| risky_ps_linear | fast | fast | 90.000 | 66.000 | 3.744 | 4.409 | 1.256 | 8.223 | 0.767 | 0.844 | 0.744 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.659 | 1.500 | 4.730 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.851 | 2.000 | 4.918 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.229 | 2.000 | 8.299 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 3.000 | trap_pre_switch | repair_all | repair_subset | 10.000 | 5.182 | 2.000 | 15.260 | False | True | False | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.542 | 1.500 | 4.608 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.702 | 1.000 | 3.776 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 10.653 | 2.000 | 10.719 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.704 | 1.000 | 3.775 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.134 | 1.500 | 4.206 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.732 | 1.000 | 8.800 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.150 | 1.500 | 4.212 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.845 | 0.500 | 2.910 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.599 | 0.500 | 2.657 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.230 | 2.000 | 5.299 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.323 | 2.000 | 5.398 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.746 | 1.500 | 4.821 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.983 | 0.500 | 3.048 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.859 | 1.000 | 3.925 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.863 | 1.000 | 3.930 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.521 | 2.000 | 9.591 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.118 | 2.000 | 5.183 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.137 | 1.500 | 4.205 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.069 | 1.500 | 9.143 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.186 | 2.000 | 5.262 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.497 | 1.500 | 4.564 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 25.000 | target_post_switch | repair_all | repair_subset | 10.000 | 3.775 | 1.500 | 13.841 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 26.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.112 | 0.000 | 10.181 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 27.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.841 | 0.500 | 27.419 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 28.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.511 | 0.500 | 10.588 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 29.000 | target_post_switch | repair_all | repair_subset | 15.000 | 5.127 | 2.000 | 20.203 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.086 | 0.000 | 3.151 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.072 | 0.500 | 4.147 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.904 | 4.500 | 4.964 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 33.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.408 | 2.000 | 4.480 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 34.000 | target_post_switch | repair_all | repair_subset | 10.000 | 3.037 | 0.000 | 13.110 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 35.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.430 | 1.500 | 5.500 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 36.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 4.050 | 3.000 | 18.109 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 37.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 4.737 | 0.000 | 16.809 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 38.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 4.858 | 3.000 | 18.924 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 39.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.947 | 0.500 | 5.023 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.276 | 0.500 | 4.353 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.753 | 2.000 | 3.824 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.835 | 0.500 | 3.910 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 43.000 | target_post_switch | repair_all | transfer | 18.000 | 4.100 | 1.500 | 22.172 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.278 | 0.500 | 4.356 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 45.000 | target_post_switch | repair_all | transfer | 18.500 | 4.001 | 0.000 | 22.578 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 46.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 5.211 | 3.500 | 17.282 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 47.000 | target_post_switch | repair_subset | transfer | 23.500 | 3.911 | 1.500 | 27.483 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 48.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 6.967 | 4.500 | 13.023 | True | False | False | fast/deep/deep/deep/deep | fast/deep/fast/fast/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 49.000 | target_post_switch | repair_all | transfer | 22.500 | 5.221 | 2.000 | 27.790 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.202 | 2.000 | 4.278 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.467 | 1.500 | 3.534 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.135 | 1.500 | 4.213 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 53.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.004 | 0.500 | 4.082 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.794 | 0.500 | 3.872 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 55.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.401 | 1.500 | 4.471 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 56.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 6.953 | 0.500 | 22.029 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 57.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.149 | 0.000 | 10.226 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 58.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.976 | 1.500 | 11.049 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 59.000 | target_post_switch | repair_all | transfer | 22.500 | 4.346 | 0.000 | 26.922 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.395 | 1.500 | 4.469 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.085 | 0.500 | 4.161 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.096 | 0.000 | 3.175 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.014 | 0.500 | 4.089 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.178 | 1.500 | 3.248 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 65.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.897 | 0.000 | 3.964 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 66.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.180 | 0.000 | 10.259 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 67.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.150 | 0.000 | 10.229 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 68.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.525 | 1.500 | 21.596 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 69.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.800 | 3.500 | 5.871 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.276 | 0.500 | 4.351 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.060 | 0.500 | 4.135 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 2.929 | 0.000 | 3.005 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.356 | 3.500 | 4.425 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.162 | 1.500 | 3.234 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 75.000 | target_post_switch | repair_all | repair_subset | 14.000 | 4.503 | 3.000 | 18.567 | False | True | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 76.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 6.103 | 3.000 | 20.174 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 77.000 | target_post_switch | repair_subset | repair_subset | 15.500 | 5.432 | 3.500 | 20.996 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | False | both_mismatch_types |
| direct_multistage_exp3 | 78.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 3.770 | 0.000 | 9.846 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 79.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.312 | 2.000 | 4.383 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
