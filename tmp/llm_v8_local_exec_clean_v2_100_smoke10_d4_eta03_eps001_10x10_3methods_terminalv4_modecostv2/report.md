# LLM v8 Local Exec Clean v2 100 Smoke10 TerminalV4 ModeCostV2 3-Method

Date: 2026-04-29

Experiment name: `llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_modecostv2`

Output directory: `tmp/llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_modecostv2`

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
| direct_multistage_exp3 | all | 100.000 | 5.810 | 3.865 | 6.223 | 1.380 | 0.071 | 12.104 | 0.700 | 0.230 | 0.700 | 0.720 | 0.610 | 0.580 | 1.020 |
| epsilon_exp3 | all | 100.000 | 5.720 | 3.780 | 6.596 | 1.530 | 0.071 | 12.387 | 0.670 | 0.150 | 0.670 | 0.770 | 0.650 | 0.660 | 1.080 |
| risky_ps_linear | all | 100.000 | 6.200 | 4.320 | 6.623 | 1.540 | 0.071 | 12.894 | 0.680 | 0.200 | 0.680 | 0.770 | 0.620 | 0.660 | 1.100 |
| direct_multistage_exp3 | pre | 25.000 | 0.000 | 0.000 | 6.466 | 1.580 | 0.070 | 6.536 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.160 |
| epsilon_exp3 | pre | 25.000 | 0.000 | 0.000 | 6.987 | 1.660 | 0.070 | 7.057 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.320 |
| risky_ps_linear | pre | 25.000 | 0.000 | 0.000 | 7.367 | 1.740 | 0.071 | 7.438 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.480 |
| direct_multistage_exp3 | post | 75.000 | 7.747 | 5.153 | 6.142 | 1.313 | 0.072 | 13.960 | 0.600 | 0.307 | 0.600 | 0.627 | 0.480 | 0.773 | 0.307 |
| epsilon_exp3 | post | 75.000 | 7.627 | 5.040 | 6.465 | 1.487 | 0.072 | 14.163 | 0.560 | 0.200 | 0.560 | 0.693 | 0.533 | 0.880 | 0.333 |
| risky_ps_linear | post | 75.000 | 8.267 | 5.760 | 6.374 | 1.473 | 0.071 | 14.712 | 0.573 | 0.267 | 0.573 | 0.693 | 0.493 | 0.880 | 0.307 |
| direct_multistage_exp3 | post_local_nontransfer | 75.000 | 7.747 | 5.153 | 6.142 | 1.313 | 0.072 | 13.960 | 0.600 | 0.307 | 0.600 | 0.627 | 0.480 | 0.773 | 0.307 |
| epsilon_exp3 | post_local_nontransfer | 75.000 | 7.627 | 5.040 | 6.465 | 1.487 | 0.072 | 14.163 | 0.560 | 0.200 | 0.560 | 0.693 | 0.533 | 0.880 | 0.333 |
| risky_ps_linear | post_local_nontransfer | 75.000 | 8.267 | 5.760 | 6.374 | 1.473 | 0.071 | 14.712 | 0.573 | 0.267 | 0.573 | 0.693 | 0.493 | 0.880 | 0.307 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | direct_multistage_exp3 | 12.104 | 5.810 | 6.223 | 1.380 | 0.071 | 0.700 | 0.230 | 0.610 |
| 2.000 | epsilon_exp3 | 12.387 | 5.720 | 6.596 | 1.530 | 0.071 | 0.670 | 0.150 | 0.650 |
| 3.000 | risky_ps_linear | 12.894 | 6.200 | 6.623 | 1.540 | 0.071 | 0.680 | 0.200 | 0.620 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 23.000 | 6.630 | 4.546 | 0.000 | 11.251 | 0.739 | 0.565 | 0.522 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 10.000 | 9.500 | 8.352 | 3.200 | 17.920 | 0.400 | 0.800 | 0.400 | 1.800 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 38.000 | 1.711 | 6.196 | 1.211 | 7.978 | 0.947 | 0.868 | 0.868 | 0.000 | 2.421 |
| direct_multistage_exp3 | fast_on_deep_required | 29.000 | 9.259 | 6.855 | 2.069 | 16.183 | 0.448 | 0.621 | 0.414 | 1.379 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 15.000 | 7.433 | 4.686 | 0.000 | 12.194 | 0.600 | 0.667 | 0.533 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 11.000 | 9.227 | 7.591 | 2.409 | 16.888 | 0.364 | 0.545 | 0.364 | 1.273 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 39.000 | 2.538 | 6.574 | 1.244 | 9.185 | 0.872 | 0.949 | 0.872 | 0.000 | 2.487 |
| epsilon_exp3 | fast_on_deep_required | 35.000 | 7.429 | 7.125 | 2.229 | 14.622 | 0.571 | 0.686 | 0.543 | 1.486 | 0.000 |
| risky_ps_linear | all_stage_modes_match | 20.000 | 8.625 | 4.633 | 0.000 | 13.331 | 0.600 | 0.700 | 0.450 | 0.000 | 0.000 |
| risky_ps_linear | both_mismatch_types | 14.000 | 10.036 | 7.958 | 2.750 | 18.063 | 0.500 | 0.571 | 0.500 | 1.500 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 34.000 | 1.191 | 7.029 | 1.412 | 8.292 | 0.941 | 0.941 | 0.882 | 0.000 | 2.824 |
| risky_ps_linear | fast_on_deep_required | 32.000 | 8.328 | 6.851 | 2.109 | 15.248 | 0.531 | 0.719 | 0.500 | 1.406 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 62.000 | 7.476 | 5.619 | 0.839 | 13.168 | 0.645 | 0.645 | 0.500 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 7.277 | 1.861 | 7.348 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 13.000 | 9.038 | 8.637 | 3.577 | 17.741 | 0.385 | 0.538 | 0.385 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 4.381 | 0.857 | 4.448 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 61.000 | 7.746 | 6.019 | 1.066 | 13.838 | 0.557 | 0.721 | 0.541 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 19.000 | 0.000 | 7.643 | 1.868 | 7.714 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 14.000 | 7.107 | 8.408 | 3.321 | 15.580 | 0.571 | 0.571 | 0.500 |
| epsilon_exp3 | mostly_fast_vs_mostly_fast_required | 6.000 | 0.000 | 4.910 | 1.000 | 4.977 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 61.000 | 8.459 | 5.964 | 1.090 | 14.495 | 0.574 | 0.656 | 0.475 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 22.000 | 0.000 | 7.699 | 1.841 | 7.771 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 14.000 | 7.429 | 8.162 | 3.143 | 15.656 | 0.571 | 0.857 | 0.571 |
| risky_ps_linear | mostly_fast_vs_mostly_fast_required | 3.000 | 0.000 | 4.937 | 1.000 | 5.001 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 62.000 | 7.476 | 5.619 | 0.839 | 13.168 | 0.500 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 13.000 | 9.038 | 8.637 | 3.577 | 17.741 | 0.385 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 7.277 | 1.861 | 7.348 | 1.000 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 4.381 | 0.857 | 4.448 | 1.000 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 61.000 | 7.746 | 6.019 | 1.066 | 13.838 | 0.541 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 14.000 | 7.107 | 8.408 | 3.321 | 15.580 | 0.500 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 19.000 | 0.000 | 7.643 | 1.868 | 7.714 | 1.000 |
| epsilon_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 6.000 | 0.000 | 4.910 | 1.000 | 4.977 | 1.000 |
| risky_ps_linear | target_post_switch | mostly_deep_vs_mostly_deep_required | 61.000 | 8.459 | 5.964 | 1.090 | 14.495 | 0.475 |
| risky_ps_linear | target_post_switch | mostly_fast_vs_mostly_deep_required | 14.000 | 7.429 | 8.162 | 3.143 | 15.656 | 0.571 |
| risky_ps_linear | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 22.000 | 0.000 | 7.699 | 1.841 | 7.771 | 1.000 |
| risky_ps_linear | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 3.000 | 0.000 | 4.937 | 1.000 | 5.001 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 242.000 | 75.000 | 7.362 | 5.741 | 0.946 | 13.175 | 0.645 | 0.628 | 0.500 |
| direct_multistage_exp3 | deep | fast | 58.000 | 39.000 | 9.353 | 7.816 | 2.845 | 17.237 | 0.414 | 0.621 | 0.397 |
| direct_multistage_exp3 | fast | deep | 102.000 | 48.000 | 1.569 | 6.997 | 1.735 | 8.637 | 0.922 | 0.931 | 0.892 |
| direct_multistage_exp3 | fast | fast | 98.000 | 76.000 | 4.296 | 5.665 | 1.214 | 10.031 | 0.776 | 0.786 | 0.714 |
| epsilon_exp3 | deep | deep | 234.000 | 75.000 | 7.485 | 6.120 | 1.158 | 13.678 | 0.577 | 0.722 | 0.551 |
| epsilon_exp3 | deep | fast | 66.000 | 46.000 | 8.129 | 7.687 | 2.652 | 15.884 | 0.500 | 0.591 | 0.470 |
| epsilon_exp3 | fast | deep | 108.000 | 50.000 | 1.856 | 7.231 | 1.685 | 9.159 | 0.889 | 0.935 | 0.889 |
| epsilon_exp3 | fast | fast | 92.000 | 73.000 | 4.038 | 6.276 | 1.489 | 10.383 | 0.772 | 0.826 | 0.750 |
| risky_ps_linear | deep | deep | 234.000 | 75.000 | 8.111 | 6.036 | 1.152 | 14.219 | 0.590 | 0.697 | 0.491 |
| risky_ps_linear | deep | fast | 66.000 | 46.000 | 8.818 | 7.575 | 2.614 | 16.461 | 0.515 | 0.682 | 0.500 |
| risky_ps_linear | fast | deep | 110.000 | 48.000 | 1.645 | 7.458 | 1.850 | 9.175 | 0.918 | 0.927 | 0.900 |
| risky_ps_linear | fast | fast | 90.000 | 74.000 | 4.878 | 6.429 | 1.383 | 11.376 | 0.744 | 0.833 | 0.700 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.680 | 1.500 | 5.751 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.563 | 2.000 | 6.630 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.948 | 2.000 | 10.018 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.840 | 2.000 | 6.918 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.964 | 1.500 | 6.030 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.676 | 1.000 | 4.750 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.543 | 2.000 | 6.609 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.671 | 1.000 | 4.742 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.428 | 1.500 | 5.500 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.498 | 1.000 | 4.566 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.484 | 1.500 | 5.545 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.870 | 0.500 | 3.934 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.946 | 0.500 | 3.004 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.845 | 2.000 | 6.914 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 14.757 | 2.000 | 14.832 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.005 | 1.500 | 6.079 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.353 | 2.000 | 6.429 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.954 | 1.000 | 5.020 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | False | deep_on_fast_required |
| direct_multistage_exp3 | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.053 | 1.000 | 5.121 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.435 | 2.000 | 6.505 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.699 | 2.000 | 6.764 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.575 | 2.000 | 6.647 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 12.271 | 2.500 | 12.348 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.854 | 2.000 | 6.930 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.747 | 1.500 | 5.814 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 25.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.780 | 1.500 | 5.847 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 26.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.980 | 0.000 | 11.049 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 27.000 | target_post_switch | repair_subset | transfer | 22.500 | 6.055 | 0.500 | 28.633 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 28.000 | target_post_switch | repair_subset | transfer | 24.500 | 6.125 | 0.500 | 30.701 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 29.000 | target_post_switch | repair_all | repair_subset | 15.000 | 7.833 | 2.000 | 22.909 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 7.578 | 3.000 | 7.643 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 31.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.604 | 0.500 | 4.679 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.006 | 0.000 | 4.081 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 33.000 | target_post_switch | repair_all | repair_subset | 14.000 | 10.010 | 5.000 | 24.072 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.692 | 0.000 | 3.765 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 35.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.780 | 1.500 | 5.851 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 36.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 6.962 | 1.500 | 19.033 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 37.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.958 | 0.000 | 27.532 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 38.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 6.817 | 1.500 | 23.885 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 39.000 | target_post_switch | repair_all | transfer | 22.500 | 6.695 | 1.500 | 29.269 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.403 | 0.500 | 5.479 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.507 | 0.000 | 3.575 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 7.864 | 3.500 | 7.932 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 43.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.034 | 1.500 | 6.106 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.855 | 0.500 | 5.930 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 45.000 | target_post_switch | repair_all | repair_subset | 14.000 | 7.619 | 3.000 | 21.683 | False | True | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 46.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.109 | 0.000 | 11.187 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 47.000 | target_post_switch | repair_subset | transfer | 23.500 | 6.043 | 1.500 | 29.616 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 48.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 10.800 | 4.500 | 24.856 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/fast/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 49.000 | target_post_switch | repair_all | transfer | 22.500 | 6.712 | 1.500 | 29.282 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 7.323 | 3.000 | 7.393 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.059 | 0.000 | 4.138 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 6.304 | 1.500 | 6.377 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 53.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.255 | 0.500 | 5.330 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.904 | 0.000 | 3.979 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 55.000 | target_post_switch | repair_all | repair_subset | 14.000 | 7.825 | 3.000 | 21.894 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 56.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 6.187 | 0.500 | 12.263 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 57.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.954 | 0.000 | 27.532 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 58.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 6.758 | 1.500 | 12.830 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 59.000 | target_post_switch | repair_all | transfer | 22.500 | 7.793 | 2.000 | 30.363 | False | True | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 7.592 | 3.000 | 7.666 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.516 | 0.000 | 3.589 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.560 | 0.000 | 3.639 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.547 | 0.500 | 5.622 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.934 | 0.000 | 4.012 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 65.000 | target_post_switch | repair_all | repair_subset | 14.000 | 8.264 | 3.000 | 22.324 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 66.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 5.109 | 0.000 | 11.188 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 67.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.946 | 0.000 | 27.525 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 68.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 6.794 | 1.500 | 23.865 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 69.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.637 | 0.000 | 5.701 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 8.427 | 3.500 | 8.493 | True | True | True | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| direct_multistage_exp3 | 71.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.397 | 0.500 | 5.472 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| direct_multistage_exp3 | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.008 | 0.000 | 4.087 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.156 | 1.500 | 5.221 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.737 | 1.500 | 5.809 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 75.000 | target_post_switch | repair_all | repair_subset | 10.000 | 6.388 | 1.500 | 16.459 | False | True | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 76.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 8.815 | 3.000 | 22.886 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/fast | False | fast_on_deep_required |
| direct_multistage_exp3 | 77.000 | target_post_switch | repair_subset | transfer | 23.500 | 6.039 | 1.500 | 29.609 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| direct_multistage_exp3 | 78.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.569 | 0.000 | 10.644 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| direct_multistage_exp3 | 79.000 | target_post_switch | repair_all | repair_subset | 15.000 | 11.165 | 5.000 | 26.225 | False | True | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/fast | False | both_mismatch_types |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
