# LLM v6 Local Clean 10x10 3-Method Smoke

Date: 2026-04-29

Experiment name: `llm_v6_local_clean_v1_d4_eta03_eps001_10x10_3methods`

Output directory: `tmp/llm_v6_local_clean_v1_d4_eta03_eps001_10x10_3methods`

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
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_clean_v1/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_clean_v1_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | reasoning | path | total | exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 100.000 | 8.620 | 5.205 | 0.071 | 13.896 | 0.260 | 0.260 | 0.470 | 0.150 | 0.550 | 0.910 |
| epsilon_exp3 | all | 100.000 | 8.985 | 5.221 | 0.070 | 14.277 | 0.250 | 0.250 | 0.410 | 0.150 | 0.770 | 1.180 |
| risky_ps_linear | all | 100.000 | 9.260 | 5.261 | 0.071 | 14.591 | 0.190 | 0.190 | 0.440 | 0.130 | 0.770 | 1.070 |
| direct_multistage_exp3 | pre | 25.000 | 0.900 | 4.911 | 0.070 | 5.881 | 0.400 | 0.400 | 1.000 | 0.400 | 0.000 | 3.080 |
| epsilon_exp3 | pre | 25.000 | 0.900 | 5.238 | 0.069 | 6.208 | 0.400 | 0.400 | 1.000 | 0.400 | 0.000 | 3.400 |
| risky_ps_linear | pre | 25.000 | 0.900 | 5.312 | 0.071 | 6.283 | 0.400 | 0.400 | 1.000 | 0.400 | 0.000 | 3.440 |
| direct_multistage_exp3 | post | 75.000 | 11.193 | 5.303 | 0.072 | 16.568 | 0.213 | 0.213 | 0.293 | 0.067 | 0.733 | 0.187 |
| epsilon_exp3 | post | 75.000 | 11.680 | 5.216 | 0.070 | 16.966 | 0.200 | 0.200 | 0.213 | 0.067 | 1.027 | 0.440 |
| risky_ps_linear | post | 75.000 | 12.047 | 5.243 | 0.071 | 17.361 | 0.120 | 0.120 | 0.253 | 0.040 | 1.027 | 0.280 |
| direct_multistage_exp3 | post_local_nontransfer | 75.000 | 11.193 | 5.303 | 0.072 | 16.568 | 0.213 | 0.213 | 0.293 | 0.067 | 0.733 | 0.187 |
| epsilon_exp3 | post_local_nontransfer | 75.000 | 11.680 | 5.216 | 0.070 | 16.966 | 0.200 | 0.200 | 0.213 | 0.067 | 1.027 | 0.440 |
| risky_ps_linear | post_local_nontransfer | 75.000 | 12.047 | 5.243 | 0.071 | 17.361 | 0.120 | 0.120 | 0.253 | 0.040 | 1.027 | 0.280 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | path | exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | direct_multistage_exp3 | 13.896 | 8.620 | 5.205 | 0.071 | 0.260 | 0.150 |
| 2.000 | epsilon_exp3 | 14.277 | 8.985 | 5.221 | 0.070 | 0.250 | 0.150 |
| 3.000 | risky_ps_linear | 14.591 | 9.260 | 5.261 | 0.071 | 0.190 | 0.130 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 33.000 | 10.712 | 5.008 | 15.795 | 0.364 | 0.485 | 0.121 | 0.000 | 0.000 |
| direct_multistage_exp3 | both_mismatch_types | 8.000 | 15.250 | 5.240 | 20.559 | 0.000 | 0.000 | 0.000 | 1.750 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 31.000 | 3.742 | 5.168 | 8.981 | 0.355 | 0.871 | 0.323 | 0.000 | 2.677 |
| direct_multistage_exp3 | fast_on_deep_required | 28.000 | 9.661 | 5.470 | 15.198 | 0.107 | 0.143 | 0.036 | 1.464 | 0.000 |
| epsilon_exp3 | all_stage_modes_match | 16.000 | 9.938 | 5.012 | 15.024 | 0.438 | 0.250 | 0.062 | 0.000 | 0.000 |
| epsilon_exp3 | both_mismatch_types | 23.000 | 10.870 | 5.500 | 16.439 | 0.087 | 0.130 | 0.087 | 1.609 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 35.000 | 3.771 | 5.340 | 9.183 | 0.400 | 0.800 | 0.314 | 0.000 | 2.714 |
| epsilon_exp3 | fast_on_deep_required | 26.000 | 13.750 | 4.944 | 18.761 | 0.077 | 0.231 | 0.038 | 1.538 | 0.000 |
| risky_ps_linear | all_stage_modes_match | 14.000 | 13.679 | 5.123 | 18.875 | 0.214 | 0.214 | 0.000 | 0.000 | 0.000 |
| risky_ps_linear | both_mismatch_types | 15.000 | 15.600 | 5.755 | 21.423 | 0.067 | 0.133 | 0.067 | 1.467 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 31.000 | 3.323 | 5.374 | 8.769 | 0.323 | 0.935 | 0.323 | 0.000 | 2.968 |
| risky_ps_linear | fast_on_deep_required | 40.000 | 9.938 | 5.036 | 15.043 | 0.125 | 0.250 | 0.050 | 1.375 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 62.000 | 11.734 | 5.160 | 16.967 | 0.242 | 0.339 | 0.065 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 17.000 | 0.971 | 5.449 | 6.491 | 0.353 | 1.000 | 0.353 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 13.000 | 8.615 | 5.986 | 14.665 | 0.077 | 0.077 | 0.077 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_fast_required | 8.000 | 0.750 | 3.769 | 4.585 | 0.500 | 1.000 | 0.500 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 59.000 | 11.703 | 5.271 | 17.047 | 0.254 | 0.237 | 0.085 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 21.000 | 0.857 | 5.450 | 6.377 | 0.429 | 1.000 | 0.429 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 16.000 | 11.594 | 5.011 | 16.668 | 0.000 | 0.125 | 0.000 |
| epsilon_exp3 | mostly_fast_vs_mostly_fast_required | 4.000 | 1.125 | 4.124 | 5.316 | 0.250 | 1.000 | 0.250 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 63.000 | 12.500 | 5.271 | 17.843 | 0.127 | 0.286 | 0.048 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 21.000 | 0.929 | 5.235 | 6.236 | 0.381 | 1.000 | 0.381 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 12.000 | 9.667 | 5.098 | 14.830 | 0.083 | 0.083 | 0.000 |
| risky_ps_linear | mostly_fast_vs_mostly_fast_required | 4.000 | 0.750 | 5.719 | 6.532 | 0.500 | 1.000 | 0.500 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 62.000 | 11.734 | 5.160 | 16.967 | 0.065 |
| direct_multistage_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 13.000 | 8.615 | 5.986 | 14.665 | 0.077 |
| direct_multistage_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 17.000 | 0.971 | 5.449 | 6.491 | 0.353 |
| direct_multistage_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 8.000 | 0.750 | 3.769 | 4.585 | 0.500 |
| epsilon_exp3 | target_post_switch | mostly_deep_vs_mostly_deep_required | 59.000 | 11.703 | 5.271 | 17.047 | 0.085 |
| epsilon_exp3 | target_post_switch | mostly_fast_vs_mostly_deep_required | 16.000 | 11.594 | 5.011 | 16.668 | 0.000 |
| epsilon_exp3 | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 21.000 | 0.857 | 5.450 | 6.377 | 0.429 |
| epsilon_exp3 | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 4.000 | 1.125 | 4.124 | 5.316 | 0.250 |
| risky_ps_linear | target_post_switch | mostly_deep_vs_mostly_deep_required | 63.000 | 12.500 | 5.271 | 17.843 | 0.048 |
| risky_ps_linear | target_post_switch | mostly_fast_vs_mostly_deep_required | 12.000 | 9.667 | 5.098 | 14.830 | 0.000 |
| risky_ps_linear | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 21.000 | 0.929 | 5.235 | 6.236 | 0.381 |
| risky_ps_linear | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 4.000 | 0.750 | 5.719 | 6.532 | 0.500 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 245.000 | 75.000 | 11.298 | 5.240 | 16.611 | 0.245 | 0.339 | 0.073 |
| direct_multistage_exp3 | deep | fast | 55.000 | 36.000 | 10.727 | 5.584 | 16.378 | 0.073 | 0.091 | 0.036 |
| direct_multistage_exp3 | fast | deep | 91.000 | 39.000 | 3.159 | 5.301 | 8.532 | 0.330 | 0.868 | 0.319 |
| direct_multistage_exp3 | fast | fast | 109.000 | 85.000 | 6.096 | 4.855 | 11.022 | 0.330 | 0.624 | 0.239 |
| epsilon_exp3 | deep | deep | 223.000 | 75.000 | 11.684 | 5.238 | 16.994 | 0.247 | 0.233 | 0.072 |
| epsilon_exp3 | deep | fast | 77.000 | 49.000 | 11.669 | 5.152 | 16.887 | 0.065 | 0.156 | 0.052 |
| epsilon_exp3 | fast | deep | 118.000 | 58.000 | 3.695 | 5.405 | 9.170 | 0.339 | 0.771 | 0.314 |
| epsilon_exp3 | fast | fast | 82.000 | 64.000 | 6.738 | 4.978 | 11.784 | 0.305 | 0.610 | 0.220 |
| risky_ps_linear | deep | deep | 223.000 | 75.000 | 12.314 | 5.246 | 17.631 | 0.126 | 0.278 | 0.036 |
| risky_ps_linear | deep | fast | 77.000 | 55.000 | 11.273 | 5.237 | 16.577 | 0.104 | 0.182 | 0.052 |
| risky_ps_linear | fast | deep | 107.000 | 46.000 | 3.668 | 5.475 | 9.215 | 0.327 | 0.860 | 0.327 |
| risky_ps_linear | fast | fast | 93.000 | 77.000 | 6.704 | 5.069 | 11.843 | 0.258 | 0.559 | 0.194 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | total | clear | aux | strict | required | actual | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 5.183 | 6.754 | False | True | False | fast/fast/fast/fast/fast | fast/deep/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.564 | 4.631 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.669 | 4.739 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 3.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 5.219 | 6.797 | False | True | False | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 4.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 3.887 | 5.453 | False | True | False | fast/fast/fast/fast/fast | deep/fast/deep/fast/deep | deep_on_fast_required |
| direct_multistage_exp3 | 5.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 3.735 | 5.309 | False | True | False | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.822 | 7.888 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.995 | 4.066 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 8.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 4.434 | 6.006 | False | True | False | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 9.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 3.987 | 5.555 | False | True | False | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 10.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 4.898 | 6.459 | False | True | False | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.224 | 3.289 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | deep_on_fast_required |
| direct_multistage_exp3 | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.225 | 3.283 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/fast/fast | deep_on_fast_required |
| direct_multistage_exp3 | 13.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 5.217 | 6.786 | False | True | False | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 14.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 4.383 | 5.958 | False | True | False | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 15.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 4.497 | 6.072 | False | True | False | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 8.549 | 8.625 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.679 | 3.745 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/fast | deep_on_fast_required |
| direct_multistage_exp3 | 18.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 4.187 | 5.755 | False | True | False | fast/fast/fast/fast/fast | fast/deep/fast/fast/deep | deep_on_fast_required |
| direct_multistage_exp3 | 19.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 9.937 | 11.507 | False | True | False | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 20.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 4.117 | 5.678 | False | True | False | fast/fast/fast/fast/fast | deep/fast/fast/fast/deep | deep_on_fast_required |
| direct_multistage_exp3 | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.585 | 4.657 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.968 | 4.042 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 23.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 5.238 | 6.814 | False | True | False | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 24.000 | trap_pre_switch | repair_all | repair_all | 1.500 | 5.583 | 7.160 | False | True | False | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 25.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 4.656 | 7.722 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 26.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.669 | 18.736 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 27.000 | target_post_switch | repair_all | transfer | 18.500 | 6.099 | 24.674 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 28.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.610 | 7.688 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 29.000 | target_post_switch | repair_subset | transfer | 21.500 | 5.132 | 26.707 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 30.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.923 | 6.982 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | fast_on_deep_required |
| direct_multistage_exp3 | 31.000 | target_post_switch | repair_all | transfer | 18.500 | 8.002 | 26.579 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 32.000 | target_post_switch | repair_subset | repair_subset | 11.500 | 7.958 | 19.518 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | fast_on_deep_required |
| direct_multistage_exp3 | 33.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.533 | 21.596 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/fast/deep | both_mismatch_types |
| direct_multistage_exp3 | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.942 | 5.016 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 35.000 | target_post_switch | repair_subset | repair_subset | 3.000 | 4.689 | 7.760 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 36.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.605 | 18.664 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 37.000 | target_post_switch | repair_all | repair_subset | 11.000 | 5.376 | 16.450 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 38.000 | target_post_switch | repair_subset | repair_subset | 5.000 | 5.194 | 10.259 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | fast_on_deep_required |
| direct_multistage_exp3 | 39.000 | target_post_switch | repair_subset | transfer | 20.500 | 4.943 | 25.521 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 40.000 | target_post_switch | repair_subset | transfer | 16.500 | 5.399 | 21.975 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.695 | 4.764 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 42.000 | target_post_switch | repair_subset | transfer | 19.500 | 4.992 | 24.568 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | both_mismatch_types |
| direct_multistage_exp3 | 43.000 | target_post_switch | repair_subset | transfer | 20.500 | 4.953 | 25.527 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 44.000 | target_post_switch | repair_all | repair_subset | 5.000 | 4.573 | 9.652 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 45.000 | target_post_switch | repair_subset | repair_subset | 11.000 | 4.914 | 15.979 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 46.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.855 | 6.923 | True | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 47.000 | target_post_switch | repair_all | repair_subset | 11.000 | 5.189 | 16.261 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 48.000 | target_post_switch | repair_subset | transfer | 19.500 | 4.550 | 24.125 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 49.000 | target_post_switch | repair_subset | repair_subset | 15.000 | 5.264 | 20.333 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 50.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.203 | 21.275 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 51.000 | target_post_switch | repair_all | transfer | 18.500 | 5.274 | 23.850 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 52.000 | target_post_switch | repair_subset | repair_subset | 14.000 | 3.996 | 18.069 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/fast/deep | fast_on_deep_required |
| direct_multistage_exp3 | 53.000 | target_post_switch | repair_subset | transfer | 21.500 | 6.915 | 28.492 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 54.000 | target_post_switch | repair_all | transfer | 14.500 | 4.265 | 18.839 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 55.000 | target_post_switch | repair_subset | repair_subset | 9.000 | 4.961 | 14.025 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | fast_on_deep_required |
| direct_multistage_exp3 | 56.000 | target_post_switch | repair_subset | transfer | 20.500 | 4.576 | 25.155 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 57.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.247 | 4.325 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 58.000 | target_post_switch | repair_subset | transfer | 19.500 | 4.539 | 24.106 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 59.000 | target_post_switch | repair_subset | transfer | 20.500 | 4.961 | 25.537 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 60.000 | target_post_switch | repair_subset | transfer | 16.500 | 5.111 | 21.670 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/fast | fast_on_deep_required |
| direct_multistage_exp3 | 61.000 | target_post_switch | repair_all | transfer | 18.500 | 4.574 | 23.152 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 62.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.502 | 6.577 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 63.000 | target_post_switch | repair_subset | repair_all | 4.000 | 12.307 | 16.371 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/fast | fast_on_deep_required |
| direct_multistage_exp3 | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.188 | 5.254 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 65.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.979 | 7.054 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 66.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.564 | 6.643 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 67.000 | target_post_switch | repair_all | repair_subset | 11.000 | 5.346 | 16.425 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 68.000 | target_post_switch | repair_subset | transfer | 19.500 | 4.538 | 24.109 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 69.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 4.531 | 6.596 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 70.000 | target_post_switch | repair_subset | transfer | 16.500 | 5.390 | 21.968 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | deep_on_fast_required |
| direct_multistage_exp3 | 71.000 | target_post_switch | repair_all | transfer | 18.500 | 6.692 | 25.267 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 72.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.768 | 7.847 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 73.000 | target_post_switch | repair_subset | transfer | 20.500 | 4.955 | 25.529 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 74.000 | target_post_switch | repair_all | repair_subset | 5.000 | 4.823 | 9.895 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 75.000 | target_post_switch | repair_subset | repair_subset | 2.000 | 5.095 | 7.167 | True | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 76.000 | target_post_switch | repair_subset | repair_subset | 13.000 | 5.298 | 18.370 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 77.000 | target_post_switch | repair_all | repair_subset | 11.000 | 5.309 | 16.379 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | fast_on_deep_required |
| direct_multistage_exp3 | 78.000 | target_post_switch | repair_subset | transfer | 18.500 | 4.974 | 23.549 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match |
| direct_multistage_exp3 | 79.000 | target_post_switch | repair_subset | transfer | 21.500 | 6.112 | 27.676 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/fast | both_mismatch_types |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 22.000 |
| target_post_switch | repair_subset | 53.000 |
| trap_pre_switch | repair_all | 25.000 |
