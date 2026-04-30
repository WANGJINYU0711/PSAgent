# LLM v8 Local Exec Clean v2 100 Smoke10 TerminalV4 3-Method

Date: 2026-04-29

Experiment name: `llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4`

Output directory: `tmp/llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4`

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
| direct_multistage_exp3 | all | 100.000 | 7.140 | 5.185 | 4.864 | 0.000 | 0.071 | 12.075 | 0.630 | 0.240 | 0.630 | 0.740 | 0.590 | 0.610 | 1.020 |
| epsilon_exp3 | all | 100.000 | 6.390 | 4.585 | 4.927 | 0.000 | 0.070 | 11.387 | 0.670 | 0.180 | 0.670 | 0.740 | 0.640 | 0.800 | 0.990 |
| risky_ps_linear | all | 100.000 | 4.710 | 3.065 | 4.927 | 0.000 | 0.072 | 9.709 | 0.750 | 0.250 | 0.750 | 0.760 | 0.680 | 0.540 | 0.990 |
| direct_multistage_exp3 | pre | 25.000 | 0.000 | 0.000 | 4.836 | 0.000 | 0.069 | 4.906 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| epsilon_exp3 | pre | 25.000 | 0.000 | 0.000 | 5.473 | 0.000 | 0.070 | 5.543 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.440 |
| risky_ps_linear | pre | 25.000 | 0.400 | 0.060 | 4.889 | 0.000 | 0.070 | 5.359 | 0.960 | 0.000 | 0.960 | 1.000 | 0.960 | 0.000 | 3.200 |
| direct_multistage_exp3 | post | 75.000 | 9.520 | 6.913 | 4.874 | 0.000 | 0.072 | 14.465 | 0.507 | 0.320 | 0.507 | 0.653 | 0.453 | 0.813 | 0.373 |
| epsilon_exp3 | post | 75.000 | 8.520 | 6.113 | 4.745 | 0.000 | 0.070 | 13.335 | 0.560 | 0.240 | 0.560 | 0.653 | 0.520 | 1.067 | 0.173 |
| risky_ps_linear | post | 75.000 | 6.147 | 4.067 | 4.940 | 0.000 | 0.072 | 11.159 | 0.680 | 0.333 | 0.680 | 0.680 | 0.587 | 0.720 | 0.253 |
| direct_multistage_exp3 | post_local_nontransfer | 75.000 | 9.520 | 6.913 | 4.874 | 0.000 | 0.072 | 14.465 | 0.507 | 0.320 | 0.507 | 0.653 | 0.453 | 0.813 | 0.373 |
| epsilon_exp3 | post_local_nontransfer | 75.000 | 8.520 | 6.113 | 4.745 | 0.000 | 0.070 | 13.335 | 0.560 | 0.240 | 0.560 | 0.653 | 0.520 | 1.067 | 0.173 |
| risky_ps_linear | post_local_nontransfer | 75.000 | 6.147 | 4.067 | 4.940 | 0.000 | 0.072 | 11.159 | 0.680 | 0.333 | 0.680 | 0.680 | 0.587 | 0.720 | 0.253 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | risky_ps_linear | 9.709 | 4.710 | 4.927 | 0.000 | 0.072 | 0.750 | 0.250 | 0.680 |
| 2.000 | epsilon_exp3 | 11.387 | 6.390 | 4.927 | 0.000 | 0.070 | 0.670 | 0.180 | 0.640 |
| 3.000 | direct_multistage_exp3 | 12.075 | 7.140 | 4.864 | 0.000 | 0.071 | 0.630 | 0.240 | 0.590 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 24.000 | 10.021 | 4.614 | 0.000 | 14.711 | 0.542 | 0.667 | 0.458 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 14.000 | 10.429 | 5.309 | 0.000 | 15.805 | 0.429 | 0.714 | 0.429 | 1.857 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 39.000 | 2.359 | 4.906 | 0.000 | 7.337 | 0.897 | 0.872 | 0.846 | 0.000 | 2.256 |
| direct_multistage_exp3 | fast_on_deep_required | 23.000 | 10.239 | 4.785 | 0.000 | 15.091 | 0.391 | 0.609 | 0.391 | 1.522 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 18.000 | 10.361 | 4.614 | 0.000 | 15.048 | 0.556 | 0.500 | 0.444 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 10.000 | 7.850 | 5.262 | 0.000 | 13.179 | 0.600 | 0.900 | 0.600 | 1.600 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 28.000 | 1.446 | 5.449 | 0.000 | 6.966 | 0.929 | 0.964 | 0.929 | 0.000 | 3.179 |
| epsilon_exp3 | fast_on_deep_required | 44.000 | 7.580 | 4.647 | 0.000 | 12.296 | 0.568 | 0.659 | 0.545 | 1.455 | 0.000 |
| risky_ps_linear | all_stage_modes_match | 25.000 | 4.720 | 4.723 | 0.000 | 9.518 | 0.760 | 0.720 | 0.640 | 0.000 | 0.000 |
| risky_ps_linear | both_mismatch_types | 6.000 | 17.333 | 5.609 | 0.000 | 23.008 | 0.167 | 0.500 | 0.167 | 2.000 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 38.000 | 1.658 | 4.949 | 0.000 | 6.679 | 0.921 | 0.921 | 0.868 | 0.000 | 2.447 |
| risky_ps_linear | fast_on_deep_required | 31.000 | 6.000 | 4.934 | 0.000 | 11.003 | 0.645 | 0.645 | 0.581 | 1.355 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 62.000 | 9.871 | 4.872 | 0.000 | 14.816 | 0.516 | 0.694 | 0.452 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.529 | 0.000 | 5.601 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 13.000 | 7.846 | 4.882 | 0.000 | 12.792 | 0.462 | 0.462 | 0.462 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.604 | 0.000 | 3.670 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 58.000 | 8.655 | 4.676 | 0.000 | 13.403 | 0.569 | 0.672 | 0.517 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 21.000 | 0.000 | 5.779 | 0.000 | 5.849 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 17.000 | 8.059 | 4.982 | 0.000 | 13.106 | 0.529 | 0.588 | 0.529 |
| epsilon_exp3 | mostly_fast_vs_mostly_fast_required | 4.000 | 0.000 | 3.869 | 0.000 | 3.935 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 65.000 | 6.069 | 4.899 | 0.000 | 11.042 | 0.692 | 0.677 | 0.585 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 19.000 | 0.526 | 5.312 | 0.000 | 5.911 | 0.947 | 1.000 | 0.947 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 10.000 | 6.650 | 5.206 | 0.000 | 11.921 | 0.600 | 0.700 | 0.600 |
| risky_ps_linear | mostly_fast_vs_mostly_fast_required | 6.000 | 0.000 | 3.548 | 0.000 | 3.611 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 62.000 | 9.871 | 4.872 | 0.000 | 14.816 | 0.452 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 13.000 | 7.846 | 4.882 | 0.000 | 12.792 | 0.462 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 16.000 | 0.000 | 5.529 | 0.000 | 5.601 | 1.000 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 9.000 | 0.000 | 3.604 | 0.000 | 3.670 | 1.000 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 58.000 | 8.655 | 4.676 | 0.000 | 13.403 | 0.517 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 17.000 | 8.059 | 4.982 | 0.000 | 13.106 | 0.529 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 21.000 | 0.000 | 5.779 | 0.000 | 5.849 | 1.000 |
| epsilon_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 4.000 | 0.000 | 3.869 | 0.000 | 3.935 | 1.000 |
| risky_ps_linear | target_post_switch | mostly_deep_vs_mostly_deep_required | 65.000 | 6.069 | 4.899 | 0.000 | 11.042 | 0.585 |
| risky_ps_linear | target_post_switch | mostly_fast_vs_mostly_deep_required | 10.000 | 6.650 | 5.206 | 0.000 | 11.921 | 0.600 |
| risky_ps_linear | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 19.000 | 0.526 | 5.312 | 0.000 | 5.911 | 0.947 |
| risky_ps_linear | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 6.000 | 0.000 | 3.548 | 0.000 | 3.611 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 239.000 | 75.000 | 9.475 | 4.835 | 0.000 | 14.383 | 0.527 | 0.669 | 0.460 |
| direct_multistage_exp3 | deep | fast | 61.000 | 37.000 | 9.697 | 5.024 | 0.000 | 14.787 | 0.426 | 0.590 | 0.426 |
| direct_multistage_exp3 | fast | deep | 102.000 | 53.000 | 2.333 | 5.169 | 0.000 | 7.574 | 0.882 | 0.912 | 0.863 |
| direct_multistage_exp3 | fast | fast | 98.000 | 71.000 | 4.857 | 4.518 | 0.000 | 9.445 | 0.745 | 0.827 | 0.724 |
| epsilon_exp3 | deep | deep | 220.000 | 75.000 | 8.684 | 4.700 | 0.000 | 13.455 | 0.564 | 0.645 | 0.514 |
| epsilon_exp3 | deep | fast | 80.000 | 54.000 | 8.069 | 4.869 | 0.000 | 13.006 | 0.550 | 0.675 | 0.537 |
| epsilon_exp3 | fast | deep | 99.000 | 38.000 | 1.202 | 5.742 | 0.000 | 7.015 | 0.939 | 0.980 | 0.939 |
| epsilon_exp3 | fast | fast | 101.000 | 84.000 | 5.149 | 4.669 | 0.000 | 9.887 | 0.733 | 0.762 | 0.703 |
| risky_ps_linear | deep | deep | 246.000 | 75.000 | 5.663 | 4.894 | 0.000 | 10.629 | 0.711 | 0.691 | 0.606 |
| risky_ps_linear | deep | fast | 54.000 | 37.000 | 8.352 | 5.151 | 0.000 | 13.571 | 0.537 | 0.630 | 0.500 |
| risky_ps_linear | fast | deep | 99.000 | 44.000 | 1.990 | 5.140 | 0.000 | 7.201 | 0.889 | 0.939 | 0.869 |
| risky_ps_linear | fast | fast | 101.000 | 78.000 | 3.109 | 4.681 | 0.000 | 7.860 | 0.822 | 0.822 | 0.772 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.411 | 0.000 | 4.482 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.545 | 0.000 | 4.612 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.970 | 0.000 | 9.040 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.842 | 0.000 | 4.920 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.474 | 0.000 | 4.540 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.678 | 0.000 | 3.752 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.546 | 0.000 | 4.612 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.693 | 0.000 | 3.764 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.940 | 0.000 | 4.012 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.488 | 0.000 | 3.555 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.973 | 0.000 | 4.035 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.370 | 0.000 | 3.434 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.919 | 0.000 | 2.977 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.847 | 0.000 | 4.917 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.966 | 0.000 | 5.041 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.931 | 0.000 | 4.006 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.664 | 0.000 | 3.729 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.955 | 0.000 | 4.021 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.770 | 0.000 | 3.838 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.970 | 0.000 | 9.040 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.900 | 0.000 | 3.962 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.549 | 0.000 | 4.621 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 11.105 | 0.000 | 11.178 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.841 | 0.000 | 4.917 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.560 | 0.000 | 5.637 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 25.000 | target_post_switch | repair_all | repair_subset | 10.000 | 4.325 | 0.000 | 14.391 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 26.000 | target_post_switch | repair_subset | repair_all | 10.000 | 5.692 | 0.000 | 15.758 | False | True | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 27.000 | target_post_switch | repair_subset | transfer | 22.500 | 5.576 | 0.000 | 28.150 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 28.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 5.972 | 0.000 | 23.048 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 29.000 | target_post_switch | repair_all | transfer | 22.500 | 5.842 | 0.000 | 28.418 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.783 | 0.000 | 4.842 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.907 | 0.000 | 4.984 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.760 | 0.000 | 4.820 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 33.000 | target_post_switch | repair_all | repair_subset | 14.000 | 5.011 | 0.000 | 19.073 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.410 | 0.000 | 4.483 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 35.000 | target_post_switch | repair_all | transfer | 18.500 | 4.837 | 0.000 | 23.408 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 36.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 5.786 | 0.000 | 19.845 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 37.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.963 | 0.000 | 27.537 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 38.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 4.797 | 0.000 | 18.862 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 39.000 | target_post_switch | repair_all | repair_subset | 15.000 | 5.484 | 0.000 | 20.547 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.897 | 0.000 | 4.973 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.524 | 0.000 | 3.592 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.912 | 0.000 | 4.976 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 43.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.827 | 0.000 | 4.899 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.279 | 0.000 | 4.348 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 45.000 | target_post_switch | repair_all | transfer | 18.500 | 4.817 | 0.000 | 23.394 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 46.000 | target_post_switch | repair_subset | transfer | 22.500 | 5.336 | 0.000 | 27.904 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 47.000 | target_post_switch | repair_subset | transfer | 22.500 | 6.028 | 0.000 | 28.599 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 48.000 | target_post_switch | repair_subset | transfer | 24.500 | 5.006 | 0.000 | 29.584 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 49.000 | target_post_switch | repair_all | transfer | 22.500 | 5.191 | 0.000 | 27.760 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.631 | 0.000 | 4.707 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.289 | 0.000 | 4.365 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.585 | 0.000 | 4.663 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 53.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.301 | 0.000 | 4.378 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.091 | 0.000 | 4.156 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 55.000 | target_post_switch | repair_all | repair_subset | 14.000 | 4.309 | 0.000 | 18.378 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 56.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.687 | 0.000 | 11.764 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 57.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.970 | 0.000 | 27.548 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 58.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 5.495 | 0.000 | 22.557 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 59.000 | target_post_switch | repair_all | transfer | 22.500 | 5.843 | 0.000 | 28.408 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 60.000 | target_post_switch | repair_all | transfer | 18.000 | 4.295 | 0.000 | 22.362 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.897 | 0.000 | 4.973 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.229 | 0.000 | 4.305 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.819 | 0.000 | 4.882 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | False | both_mismatch_types |
| direct_multistage_exp3 | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.421 | 0.000 | 4.487 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 65.000 | target_post_switch | repair_all | transfer | 18.500 | 4.487 | 0.000 | 23.063 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 66.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.573 | 0.000 | 10.652 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 67.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.925 | 0.000 | 27.501 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 68.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.642 | 0.000 | 21.713 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 69.000 | target_post_switch | repair_all | transfer | 22.500 | 6.145 | 0.000 | 28.716 | False | True | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.693 | 0.000 | 4.770 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.066 | 0.000 | 4.142 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.602 | 0.000 | 4.681 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.760 | 0.000 | 4.838 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.863 | 0.000 | 3.929 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 75.000 | target_post_switch | repair_all | transfer | 18.500 | 4.439 | 0.000 | 23.013 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 76.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.109 | 0.000 | 11.185 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 77.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 6.016 | 0.000 | 20.079 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/fast/fast | False | both_mismatch_types |
| direct_multistage_exp3 | 78.000 | target_post_switch | repair_subset | repair_subset | 18.000 | 5.195 | 0.000 | 23.271 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 79.000 | target_post_switch | repair_all | transfer | 22.500 | 5.387 | 0.000 | 27.964 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
