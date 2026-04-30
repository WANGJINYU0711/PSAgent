# llm_v8 PS family C config d4 eta03 eps001 pslinear eta_shared015

Date: 2026-04-29

Experiment name: `llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015`

Output directory: `tmp/llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015`

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
| methods | risky_ps_old, risky_ps, risky_ps_linear, risky_ps_ix, risky_ps_safe_conditional, risky_ps_safe_conditional_ix, risky_ps_direct_cost |
| dataset | data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json |
| schedule buckets | analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json |

## Main Cost And Success Summary

Definitions: `clear_success_proxy = exact_match && subset_clean`; `auxiliary_success_proxy = policy_violation_count == 0`; `strict_clean = clear_success_proxy && auxiliary_success_proxy`. The runner still does not export a native clean_success_no_fallback or auxiliary_success field, so these are auditable proxies.

| method | split | n | terminal | legacy term | reasoning | mode cost | path | total | exact | mode exact | clear | aux | strict | fast-on-deep | deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | all | 100.000 | 4.500 | 3.060 | 4.437 | 1.285 | 0.071 | 9.008 | 0.770 | 0.220 | 0.770 | 0.770 | 0.700 | 0.480 | 1.130 |
| risky_ps_direct_cost | all | 100.000 | 5.815 | 3.685 | 4.442 | 1.555 | 0.071 | 10.327 | 0.660 | 0.150 | 0.660 | 0.740 | 0.620 | 0.670 | 1.100 |
| risky_ps_ix | all | 100.000 | 5.410 | 3.895 | 4.646 | 1.500 | 0.071 | 10.127 | 0.720 | 0.170 | 0.720 | 0.780 | 0.680 | 0.590 | 1.230 |
| risky_ps_linear | all | 100.000 | 6.645 | 4.470 | 4.472 | 1.435 | 0.071 | 11.189 | 0.620 | 0.210 | 0.620 | 0.750 | 0.590 | 0.600 | 1.070 |
| risky_ps_old | all | 100.000 | 5.890 | 4.090 | 4.446 | 1.465 | 0.071 | 10.407 | 0.660 | 0.170 | 0.660 | 0.760 | 0.630 | 0.620 | 1.070 |
| risky_ps_safe_conditional | all | 100.000 | 5.020 | 3.395 | 4.490 | 1.455 | 0.071 | 9.581 | 0.730 | 0.180 | 0.730 | 0.730 | 0.670 | 0.630 | 1.020 |
| risky_ps_safe_conditional_ix | all | 100.000 | 5.085 | 3.350 | 4.391 | 1.480 | 0.071 | 9.547 | 0.710 | 0.150 | 0.710 | 0.750 | 0.670 | 0.650 | 1.010 |
| risky_ps | pre | 25.000 | 0.000 | 0.000 | 4.857 | 1.480 | 0.070 | 4.927 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| risky_ps_direct_cost | pre | 25.000 | 0.000 | 0.000 | 4.686 | 1.480 | 0.069 | 4.754 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| risky_ps_ix | pre | 25.000 | 0.000 | 0.000 | 5.282 | 1.660 | 0.070 | 5.352 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 3.320 |
| risky_ps_linear | pre | 25.000 | 0.000 | 0.000 | 4.651 | 1.480 | 0.069 | 4.720 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| risky_ps_old | pre | 25.000 | 0.000 | 0.000 | 4.931 | 1.480 | 0.069 | 5.000 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| risky_ps_safe_conditional | pre | 25.000 | 0.000 | 0.000 | 4.989 | 1.480 | 0.069 | 5.057 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| risky_ps_safe_conditional_ix | pre | 25.000 | 0.000 | 0.000 | 4.660 | 1.480 | 0.069 | 4.728 | 1.000 | 0.000 | 1.000 | 1.000 | 1.000 | 0.000 | 2.960 |
| risky_ps | post | 75.000 | 6.000 | 4.080 | 4.296 | 1.220 | 0.072 | 10.368 | 0.693 | 0.293 | 0.693 | 0.693 | 0.600 | 0.640 | 0.520 |
| risky_ps_direct_cost | post | 75.000 | 7.753 | 4.913 | 4.360 | 1.580 | 0.071 | 12.185 | 0.547 | 0.200 | 0.547 | 0.653 | 0.493 | 0.893 | 0.480 |
| risky_ps_ix | post | 75.000 | 7.213 | 5.193 | 4.434 | 1.447 | 0.071 | 11.718 | 0.627 | 0.227 | 0.627 | 0.707 | 0.573 | 0.787 | 0.533 |
| risky_ps_linear | post | 75.000 | 8.860 | 5.960 | 4.413 | 1.420 | 0.072 | 13.345 | 0.493 | 0.280 | 0.493 | 0.667 | 0.453 | 0.800 | 0.440 |
| risky_ps_old | post | 75.000 | 7.853 | 5.453 | 4.284 | 1.460 | 0.072 | 12.210 | 0.547 | 0.227 | 0.547 | 0.680 | 0.507 | 0.827 | 0.440 |
| risky_ps_safe_conditional | post | 75.000 | 6.693 | 4.527 | 4.324 | 1.447 | 0.071 | 11.088 | 0.640 | 0.240 | 0.640 | 0.640 | 0.560 | 0.840 | 0.373 |
| risky_ps_safe_conditional_ix | post | 75.000 | 6.780 | 4.467 | 4.301 | 1.480 | 0.071 | 11.153 | 0.613 | 0.200 | 0.613 | 0.667 | 0.560 | 0.867 | 0.360 |
| risky_ps | post_local_nontransfer | 75.000 | 6.000 | 4.080 | 4.296 | 1.220 | 0.072 | 10.368 | 0.693 | 0.293 | 0.693 | 0.693 | 0.600 | 0.640 | 0.520 |
| risky_ps_direct_cost | post_local_nontransfer | 75.000 | 7.753 | 4.913 | 4.360 | 1.580 | 0.071 | 12.185 | 0.547 | 0.200 | 0.547 | 0.653 | 0.493 | 0.893 | 0.480 |
| risky_ps_ix | post_local_nontransfer | 75.000 | 7.213 | 5.193 | 4.434 | 1.447 | 0.071 | 11.718 | 0.627 | 0.227 | 0.627 | 0.707 | 0.573 | 0.787 | 0.533 |
| risky_ps_linear | post_local_nontransfer | 75.000 | 8.860 | 5.960 | 4.413 | 1.420 | 0.072 | 13.345 | 0.493 | 0.280 | 0.493 | 0.667 | 0.453 | 0.800 | 0.440 |
| risky_ps_old | post_local_nontransfer | 75.000 | 7.853 | 5.453 | 4.284 | 1.460 | 0.072 | 12.210 | 0.547 | 0.227 | 0.547 | 0.680 | 0.507 | 0.827 | 0.440 |
| risky_ps_safe_conditional | post_local_nontransfer | 75.000 | 6.693 | 4.527 | 4.324 | 1.447 | 0.071 | 11.088 | 0.640 | 0.240 | 0.640 | 0.640 | 0.560 | 0.840 | 0.373 |
| risky_ps_safe_conditional_ix | post_local_nontransfer | 75.000 | 6.780 | 4.467 | 4.301 | 1.480 | 0.071 | 11.153 | 0.613 | 0.200 | 0.613 | 0.667 | 0.560 | 0.867 | 0.360 |

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | mode cost | path | exact | mode exact | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1.000 | risky_ps | 9.008 | 4.500 | 4.437 | 1.285 | 0.071 | 0.770 | 0.220 | 0.700 |
| 2.000 | risky_ps_safe_conditional_ix | 9.547 | 5.085 | 4.391 | 1.480 | 0.071 | 0.710 | 0.150 | 0.670 |
| 3.000 | risky_ps_safe_conditional | 9.581 | 5.020 | 4.490 | 1.455 | 0.071 | 0.730 | 0.180 | 0.670 |
| 4.000 | risky_ps_ix | 10.127 | 5.410 | 4.646 | 1.500 | 0.071 | 0.720 | 0.170 | 0.680 |
| 5.000 | risky_ps_direct_cost | 10.327 | 5.815 | 4.442 | 1.555 | 0.071 | 0.660 | 0.150 | 0.620 |
| 6.000 | risky_ps_old | 10.407 | 5.890 | 4.446 | 1.465 | 0.071 | 0.660 | 0.170 | 0.630 |
| 7.000 | risky_ps_linear | 11.189 | 6.645 | 4.472 | 1.435 | 0.071 | 0.620 | 0.210 | 0.590 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | mode cost | total | clear | aux | strict | avg fast-on-deep | avg deep-on-fast |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | all_stage_modes_match | 22.000 | 6.136 | 3.822 | 0.000 | 10.032 | 0.773 | 0.682 | 0.591 | 0.000 | 0.000 |
| risky_ps | both_mismatch_types | 22.000 | 3.182 | 4.664 | 2.477 | 7.915 | 0.773 | 0.818 | 0.773 | 1.318 | 1.000 |
| risky_ps | deep_on_fast_required | 42.000 | 3.214 | 4.668 | 1.083 | 7.955 | 0.857 | 0.857 | 0.810 | 0.000 | 2.167 |
| risky_ps | fast_on_deep_required | 14.000 | 7.857 | 4.350 | 2.036 | 12.275 | 0.500 | 0.571 | 0.429 | 1.357 | 0.000 |
| risky_ps_direct_cost | all_stage_modes_match | 15.000 | 7.500 | 3.893 | 0.000 | 11.468 | 0.600 | 0.800 | 0.467 | 0.000 | 0.000 |
| risky_ps_direct_cost | both_mismatch_types | 25.000 | 5.800 | 4.702 | 2.540 | 10.571 | 0.640 | 0.680 | 0.640 | 1.360 | 1.000 |
| risky_ps_direct_cost | deep_on_fast_required | 36.000 | 2.819 | 4.680 | 1.181 | 7.571 | 0.861 | 0.833 | 0.806 | 0.000 | 2.361 |
| risky_ps_direct_cost | fast_on_deep_required | 24.000 | 9.271 | 4.155 | 2.062 | 13.495 | 0.417 | 0.625 | 0.417 | 1.375 | 0.000 |
| risky_ps_ix | all_stage_modes_match | 17.000 | 9.206 | 4.057 | 0.000 | 13.337 | 0.588 | 0.647 | 0.471 | 0.000 | 0.000 |
| risky_ps_ix | both_mismatch_types | 24.000 | 5.958 | 4.798 | 2.562 | 10.825 | 0.667 | 0.708 | 0.667 | 1.375 | 1.000 |
| risky_ps_ix | deep_on_fast_required | 41.000 | 2.098 | 4.938 | 1.207 | 7.108 | 0.902 | 0.927 | 0.854 | 0.000 | 2.415 |
| risky_ps_ix | fast_on_deep_required | 18.000 | 8.639 | 4.333 | 2.167 | 13.039 | 0.500 | 0.667 | 0.500 | 1.444 | 0.000 |
| risky_ps_linear | all_stage_modes_match | 21.000 | 6.429 | 3.862 | 0.000 | 10.367 | 0.619 | 0.762 | 0.524 | 0.000 | 0.000 |
| risky_ps_linear | both_mismatch_types | 18.000 | 10.722 | 4.951 | 2.750 | 15.742 | 0.389 | 0.611 | 0.389 | 1.500 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 40.000 | 3.013 | 4.659 | 1.113 | 7.743 | 0.825 | 0.900 | 0.800 | 0.000 | 2.225 |
| risky_ps_linear | fast_on_deep_required | 21.000 | 10.286 | 4.318 | 2.357 | 14.671 | 0.429 | 0.571 | 0.429 | 1.571 | 0.000 |
| risky_ps_old | all_stage_modes_match | 17.000 | 7.412 | 3.770 | 0.000 | 11.257 | 0.529 | 0.882 | 0.529 | 0.000 | 0.000 |
| risky_ps_old | both_mismatch_types | 21.000 | 6.381 | 4.514 | 2.286 | 10.965 | 0.619 | 0.667 | 0.619 | 1.190 | 1.000 |
| risky_ps_old | deep_on_fast_required | 37.000 | 2.622 | 4.779 | 1.162 | 7.471 | 0.865 | 0.865 | 0.784 | 0.000 | 2.324 |
| risky_ps_old | fast_on_deep_required | 25.000 | 9.280 | 4.357 | 2.220 | 13.706 | 0.480 | 0.600 | 0.480 | 1.480 | 0.000 |
| risky_ps_safe_conditional | all_stage_modes_match | 18.000 | 4.000 | 3.471 | 0.000 | 7.544 | 0.833 | 0.722 | 0.667 | 0.000 | 0.000 |
| risky_ps_safe_conditional | both_mismatch_types | 14.000 | 10.964 | 4.909 | 2.643 | 15.942 | 0.429 | 0.429 | 0.429 | 1.429 | 1.000 |
| risky_ps_safe_conditional | deep_on_fast_required | 39.000 | 0.897 | 4.789 | 1.128 | 5.758 | 0.974 | 0.897 | 0.897 | 0.000 | 2.256 |
| risky_ps_safe_conditional | fast_on_deep_required | 29.000 | 8.328 | 4.518 | 2.224 | 12.914 | 0.483 | 0.655 | 0.483 | 1.483 | 0.000 |
| risky_ps_safe_conditional_ix | all_stage_modes_match | 15.000 | 5.300 | 3.912 | 0.000 | 9.286 | 0.667 | 0.733 | 0.533 | 0.000 | 0.000 |
| risky_ps_safe_conditional_ix | both_mismatch_types | 19.000 | 5.105 | 4.701 | 2.395 | 9.876 | 0.684 | 0.789 | 0.684 | 1.263 | 1.000 |
| risky_ps_safe_conditional_ix | deep_on_fast_required | 33.000 | 2.152 | 4.593 | 1.242 | 6.816 | 0.909 | 0.879 | 0.879 | 0.000 | 2.485 |
| risky_ps_safe_conditional_ix | fast_on_deep_required | 33.000 | 7.909 | 4.227 | 1.864 | 12.206 | 0.545 | 0.606 | 0.515 | 1.242 | 0.000 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | mostly_deep_vs_mostly_deep_required | 70.000 | 5.629 | 4.224 | 1.043 | 9.925 | 0.729 | 0.729 | 0.629 |
| risky_ps | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.414 | 1.750 | 5.487 | 1.000 | 1.000 | 1.000 |
| risky_ps | mostly_fast_vs_mostly_deep_required | 5.000 | 11.200 | 5.310 | 3.700 | 16.576 | 0.200 | 0.200 | 0.200 |
| risky_ps | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.423 | 0.786 | 3.486 | 1.000 | 1.000 | 1.000 |
| risky_ps_direct_cost | mostly_deep_vs_mostly_deep_required | 65.000 | 7.054 | 4.298 | 1.277 | 11.424 | 0.600 | 0.692 | 0.538 |
| risky_ps_direct_cost | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.120 | 1.750 | 5.191 | 1.000 | 1.000 | 1.000 |
| risky_ps_direct_cost | mostly_fast_vs_mostly_deep_required | 10.000 | 12.300 | 4.768 | 3.550 | 17.132 | 0.200 | 0.400 | 0.200 |
| risky_ps_direct_cost | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.569 | 0.786 | 3.632 | 1.000 | 1.000 | 1.000 |
| risky_ps_ix | mostly_deep_vs_mostly_deep_required | 67.000 | 6.799 | 4.343 | 1.209 | 11.213 | 0.657 | 0.746 | 0.597 |
| risky_ps_ix | mostly_deep_vs_mostly_fast_required | 20.000 | 0.000 | 5.751 | 1.875 | 5.823 | 1.000 | 1.000 | 1.000 |
| risky_ps_ix | mostly_fast_vs_mostly_deep_required | 8.000 | 10.688 | 5.196 | 3.438 | 15.949 | 0.375 | 0.375 | 0.375 |
| risky_ps_ix | mostly_fast_vs_mostly_fast_required | 5.000 | 0.000 | 3.406 | 0.800 | 3.468 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 64.000 | 8.773 | 4.311 | 0.992 | 13.158 | 0.500 | 0.703 | 0.453 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.127 | 1.750 | 5.198 | 1.000 | 1.000 | 1.000 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 11.000 | 9.364 | 5.008 | 3.909 | 14.435 | 0.455 | 0.455 | 0.455 |
| risky_ps_linear | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.427 | 0.786 | 3.489 | 1.000 | 1.000 | 1.000 |
| risky_ps_old | mostly_deep_vs_mostly_deep_required | 65.000 | 7.992 | 4.189 | 1.177 | 12.254 | 0.538 | 0.677 | 0.492 |
| risky_ps_old | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.550 | 1.750 | 5.621 | 1.000 | 1.000 | 1.000 |
| risky_ps_old | mostly_fast_vs_mostly_deep_required | 10.000 | 6.950 | 4.907 | 3.300 | 11.924 | 0.600 | 0.700 | 0.600 |
| risky_ps_old | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.342 | 0.786 | 3.404 | 1.000 | 1.000 | 1.000 |
| risky_ps_safe_conditional | mostly_deep_vs_mostly_deep_required | 63.000 | 5.929 | 4.149 | 1.048 | 10.150 | 0.698 | 0.683 | 0.603 |
| risky_ps_safe_conditional | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.632 | 1.750 | 5.703 | 1.000 | 1.000 | 1.000 |
| risky_ps_safe_conditional | mostly_fast_vs_mostly_deep_required | 12.000 | 10.708 | 5.242 | 3.542 | 16.016 | 0.333 | 0.417 | 0.333 |
| risky_ps_safe_conditional | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.334 | 0.786 | 3.397 | 1.000 | 1.000 | 1.000 |
| risky_ps_safe_conditional_ix | mostly_deep_vs_mostly_deep_required | 68.000 | 6.412 | 4.249 | 1.250 | 10.733 | 0.632 | 0.691 | 0.574 |
| risky_ps_safe_conditional_ix | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.107 | 1.750 | 5.178 | 1.000 | 1.000 | 1.000 |
| risky_ps_safe_conditional_ix | mostly_fast_vs_mostly_deep_required | 7.000 | 10.357 | 4.805 | 3.714 | 15.227 | 0.429 | 0.429 | 0.429 |
| risky_ps_safe_conditional_ix | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.510 | 0.786 | 3.573 | 1.000 | 1.000 | 1.000 |

## Phase + Majority Pair Summary

| method | phase | majority pair | n | terminal | reasoning | mode cost | total | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | target_post_switch | mostly_deep_vs_mostly_deep_required | 70.000 | 5.629 | 4.224 | 1.043 | 9.925 | 0.629 |
| risky_ps | target_post_switch | mostly_fast_vs_mostly_deep_required | 5.000 | 11.200 | 5.310 | 3.700 | 16.576 | 0.200 |
| risky_ps | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.414 | 1.750 | 5.487 | 1.000 |
| risky_ps | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.423 | 0.786 | 3.486 | 1.000 |
| risky_ps_direct_cost | target_post_switch | mostly_deep_vs_mostly_deep_required | 65.000 | 7.054 | 4.298 | 1.277 | 11.424 | 0.538 |
| risky_ps_direct_cost | target_post_switch | mostly_fast_vs_mostly_deep_required | 10.000 | 12.300 | 4.768 | 3.550 | 17.132 | 0.200 |
| risky_ps_direct_cost | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.120 | 1.750 | 5.191 | 1.000 |
| risky_ps_direct_cost | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.569 | 0.786 | 3.632 | 1.000 |
| risky_ps_ix | target_post_switch | mostly_deep_vs_mostly_deep_required | 67.000 | 6.799 | 4.343 | 1.209 | 11.213 | 0.597 |
| risky_ps_ix | target_post_switch | mostly_fast_vs_mostly_deep_required | 8.000 | 10.688 | 5.196 | 3.438 | 15.949 | 0.375 |
| risky_ps_ix | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 20.000 | 0.000 | 5.751 | 1.875 | 5.823 | 1.000 |
| risky_ps_ix | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 5.000 | 0.000 | 3.406 | 0.800 | 3.468 | 1.000 |
| risky_ps_linear | target_post_switch | mostly_deep_vs_mostly_deep_required | 64.000 | 8.773 | 4.311 | 0.992 | 13.158 | 0.453 |
| risky_ps_linear | target_post_switch | mostly_fast_vs_mostly_deep_required | 11.000 | 9.364 | 5.008 | 3.909 | 14.435 | 0.455 |
| risky_ps_linear | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.127 | 1.750 | 5.198 | 1.000 |
| risky_ps_linear | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.427 | 0.786 | 3.489 | 1.000 |
| risky_ps_old | target_post_switch | mostly_deep_vs_mostly_deep_required | 65.000 | 7.992 | 4.189 | 1.177 | 12.254 | 0.492 |
| risky_ps_old | target_post_switch | mostly_fast_vs_mostly_deep_required | 10.000 | 6.950 | 4.907 | 3.300 | 11.924 | 0.600 |
| risky_ps_old | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.550 | 1.750 | 5.621 | 1.000 |
| risky_ps_old | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.342 | 0.786 | 3.404 | 1.000 |
| risky_ps_safe_conditional | target_post_switch | mostly_deep_vs_mostly_deep_required | 63.000 | 5.929 | 4.149 | 1.048 | 10.150 | 0.603 |
| risky_ps_safe_conditional | target_post_switch | mostly_fast_vs_mostly_deep_required | 12.000 | 10.708 | 5.242 | 3.542 | 16.016 | 0.333 |
| risky_ps_safe_conditional | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.632 | 1.750 | 5.703 | 1.000 |
| risky_ps_safe_conditional | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.334 | 0.786 | 3.397 | 1.000 |
| risky_ps_safe_conditional_ix | target_post_switch | mostly_deep_vs_mostly_deep_required | 68.000 | 6.412 | 4.249 | 1.250 | 10.733 | 0.574 |
| risky_ps_safe_conditional_ix | target_post_switch | mostly_fast_vs_mostly_deep_required | 7.000 | 10.357 | 4.805 | 3.714 | 15.227 | 0.429 |
| risky_ps_safe_conditional_ix | trap_pre_switch | mostly_deep_vs_mostly_fast_required | 18.000 | 0.000 | 5.107 | 1.750 | 5.178 | 1.000 |
| risky_ps_safe_conditional_ix | trap_pre_switch | mostly_fast_vs_mostly_fast_required | 7.000 | 0.000 | 3.510 | 0.786 | 3.573 | 1.000 |

## Stage-Level Required/Actual Mode Pair Summary

| method | required | actual | n stage obs | episode n | terminal | reasoning | mode cost | total | clear | aux | strict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | deep | deep | 252.000 | 75.000 | 6.048 | 4.211 | 0.942 | 10.331 | 0.710 | 0.702 | 0.603 |
| risky_ps | deep | fast | 48.000 | 36.000 | 5.750 | 4.746 | 2.677 | 10.563 | 0.604 | 0.646 | 0.583 |
| risky_ps | fast | deep | 113.000 | 64.000 | 1.814 | 4.916 | 1.655 | 6.802 | 0.903 | 0.912 | 0.885 |
| risky_ps | fast | fast | 87.000 | 59.000 | 2.816 | 4.297 | 1.029 | 7.182 | 0.862 | 0.851 | 0.805 |
| risky_ps_direct_cost | deep | deep | 233.000 | 75.000 | 7.573 | 4.288 | 1.260 | 11.933 | 0.571 | 0.674 | 0.502 |
| risky_ps_direct_cost | deep | fast | 67.000 | 49.000 | 8.381 | 4.611 | 2.694 | 13.060 | 0.463 | 0.582 | 0.463 |
| risky_ps_direct_cost | fast | deep | 110.000 | 61.000 | 2.241 | 4.844 | 1.755 | 7.156 | 0.873 | 0.873 | 0.855 |
| risky_ps_direct_cost | fast | fast | 90.000 | 62.000 | 3.722 | 4.221 | 1.228 | 8.011 | 0.778 | 0.867 | 0.756 |
| risky_ps_ix | deep | deep | 241.000 | 75.000 | 7.081 | 4.338 | 1.129 | 11.491 | 0.647 | 0.726 | 0.581 |
| risky_ps_ix | deep | fast | 59.000 | 42.000 | 7.754 | 4.825 | 2.746 | 12.646 | 0.542 | 0.627 | 0.542 |
| risky_ps_ix | fast | deep | 123.000 | 65.000 | 1.862 | 5.348 | 1.813 | 7.282 | 0.902 | 0.919 | 0.886 |
| risky_ps_ix | fast | fast | 77.000 | 56.000 | 4.052 | 4.350 | 1.208 | 8.470 | 0.792 | 0.844 | 0.766 |
| risky_ps_linear | deep | deep | 240.000 | 73.000 | 8.573 | 4.295 | 0.981 | 12.941 | 0.512 | 0.696 | 0.463 |
| risky_ps_linear | deep | fast | 60.000 | 39.000 | 10.008 | 4.885 | 3.175 | 14.959 | 0.417 | 0.550 | 0.417 |
| risky_ps_linear | fast | deep | 107.000 | 58.000 | 2.930 | 4.891 | 1.692 | 7.892 | 0.832 | 0.897 | 0.822 |
| risky_ps_linear | fast | fast | 93.000 | 65.000 | 3.774 | 4.183 | 1.188 | 8.026 | 0.785 | 0.849 | 0.763 |
| risky_ps_old | deep | deep | 238.000 | 75.000 | 7.721 | 4.206 | 1.170 | 12.000 | 0.555 | 0.702 | 0.504 |
| risky_ps_old | deep | fast | 62.000 | 46.000 | 8.363 | 4.585 | 2.573 | 13.017 | 0.516 | 0.597 | 0.516 |
| risky_ps_old | fast | deep | 107.000 | 58.000 | 2.159 | 5.066 | 1.664 | 7.296 | 0.879 | 0.888 | 0.850 |
| risky_ps_old | fast | fast | 93.000 | 65.000 | 3.849 | 4.255 | 1.253 | 8.173 | 0.774 | 0.871 | 0.774 |
| risky_ps_safe_conditional | deep | deep | 237.000 | 75.000 | 6.006 | 4.178 | 1.086 | 10.256 | 0.692 | 0.667 | 0.591 |
| risky_ps_safe_conditional | deep | fast | 63.000 | 43.000 | 9.278 | 4.873 | 2.802 | 14.219 | 0.444 | 0.540 | 0.444 |
| risky_ps_safe_conditional | fast | deep | 102.000 | 53.000 | 1.848 | 5.105 | 1.647 | 7.024 | 0.912 | 0.882 | 0.882 |
| risky_ps_safe_conditional | fast | fast | 98.000 | 70.000 | 3.199 | 4.359 | 1.281 | 7.626 | 0.816 | 0.847 | 0.786 |
| risky_ps_safe_conditional_ix | deep | deep | 235.000 | 75.000 | 6.591 | 4.240 | 1.219 | 10.903 | 0.626 | 0.677 | 0.562 |
| risky_ps_safe_conditional_ix | deep | fast | 65.000 | 52.000 | 7.462 | 4.524 | 2.423 | 12.054 | 0.569 | 0.631 | 0.554 |
| risky_ps_safe_conditional_ix | fast | deep | 101.000 | 52.000 | 1.663 | 4.836 | 1.718 | 6.570 | 0.911 | 0.921 | 0.901 |
| risky_ps_safe_conditional_ix | fast | fast | 99.000 | 71.000 | 3.439 | 4.208 | 1.237 | 7.716 | 0.798 | 0.828 | 0.768 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact first 80 rows below.

| method | ep | phase | oracle | final | terminal | reasoning | mode cost | total | clear | aux | strict | required | actual | mode exact | mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps_old | 0.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.153 | 1.500 | 4.219 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 1.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.282 | 1.500 | 4.347 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/fast | False | deep_on_fast_required |
| risky_ps_old | 2.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 9.054 | 1.500 | 9.129 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 3.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.200 | 2.000 | 5.276 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 4.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.778 | 1.500 | 4.842 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 5.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.627 | 2.000 | 5.702 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 6.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.558 | 1.000 | 3.621 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/fast/fast | False | deep_on_fast_required |
| risky_ps_old | 7.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.698 | 1.000 | 3.758 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 8.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.118 | 1.500 | 4.191 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 9.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.230 | 1.500 | 4.303 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 10.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.180 | 0.500 | 3.236 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/fast/fast | False | deep_on_fast_required |
| risky_ps_old | 11.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.686 | 2.000 | 4.764 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 12.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 2.330 | 0.500 | 2.394 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/fast/deep | False | deep_on_fast_required |
| risky_ps_old | 13.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.450 | 1.500 | 4.520 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 14.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.444 | 1.500 | 4.512 | True | True | True | fast/fast/fast/fast/fast | fast/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 15.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 6.009 | 2.500 | 6.085 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 16.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.978 | 2.000 | 5.053 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 17.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.704 | 1.000 | 3.770 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 18.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 5.396 | 2.500 | 5.473 | True | True | True | fast/fast/fast/fast/fast | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 19.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 12.613 | 2.000 | 12.681 | True | True | True | fast/fast/fast/fast/fast | deep/fast/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 20.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.780 | 1.000 | 3.841 | True | True | True | fast/fast/fast/fast/fast | deep/fast/fast/fast/deep | False | deep_on_fast_required |
| risky_ps_old | 21.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.140 | 0.500 | 3.208 | True | True | True | fast/fast/fast/fast/fast | fast/fast/fast/deep/fast | False | deep_on_fast_required |
| risky_ps_old | 22.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 7.821 | 1.500 | 7.885 | True | True | True | fast/fast/fast/fast/fast | fast/deep/fast/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 23.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 4.181 | 1.500 | 4.245 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/fast | False | deep_on_fast_required |
| risky_ps_old | 24.000 | trap_pre_switch | repair_all | repair_all | 0.000 | 3.877 | 1.500 | 3.948 | True | True | True | fast/fast/fast/fast/fast | fast/deep/deep/deep/fast | False | deep_on_fast_required |
| risky_ps_old | 25.000 | target_post_switch | repair_all | repair_subset | 10.000 | 4.024 | 0.000 | 14.099 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 26.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 3.997 | 1.500 | 16.071 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 27.000 | target_post_switch | repair_subset | transfer | 22.500 | 4.830 | 0.500 | 27.405 | False | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 28.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.394 | 0.000 | 21.458 | False | False | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 29.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.035 | 0.000 | 4.103 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 30.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.632 | 3.000 | 4.701 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| risky_ps_old | 31.000 | target_post_switch | repair_all | transfer | 18.000 | 3.061 | 0.000 | 21.135 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 32.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.793 | 2.000 | 3.865 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 33.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.789 | 1.500 | 3.858 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 34.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.134 | 1.500 | 4.214 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 35.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.078 | 2.000 | 4.150 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 36.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 5.285 | 3.500 | 22.352 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps_old | 37.000 | target_post_switch | repair_subset | repair_subset | 12.000 | 4.669 | 2.000 | 16.734 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 38.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.495 | 0.500 | 10.572 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 39.000 | target_post_switch | repair_all | transfer | 22.500 | 4.478 | 0.000 | 27.057 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 40.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.368 | 1.500 | 4.440 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/fast | False | fast_on_deep_required |
| risky_ps_old | 41.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.642 | 3.000 | 4.710 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/fast/deep | False | fast_on_deep_required |
| risky_ps_old | 42.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.239 | 2.000 | 4.308 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 43.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.277 | 1.500 | 4.349 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 44.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.304 | 3.000 | 4.375 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/fast/deep | False | fast_on_deep_required |
| risky_ps_old | 45.000 | target_post_switch | repair_all | repair_subset | 10.000 | 3.661 | 0.000 | 13.744 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 46.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.001 | 1.500 | 21.063 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 47.000 | target_post_switch | repair_subset | transfer | 23.500 | 3.913 | 1.500 | 27.485 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 48.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.517 | 0.500 | 10.593 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 49.000 | target_post_switch | repair_all | repair_subset | 15.000 | 4.915 | 3.000 | 19.981 | False | True | False | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 50.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.767 | 2.000 | 3.841 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 51.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.518 | 0.000 | 3.594 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 52.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.889 | 3.000 | 4.960 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/fast/fast | False | fast_on_deep_required |
| risky_ps_old | 53.000 | target_post_switch | repair_all | repair_subset | 10.000 | 4.846 | 0.500 | 14.922 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 54.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.660 | 3.000 | 4.727 | True | True | True | fast/deep/deep/deep/deep | fast/fast/deep/deep/fast | False | fast_on_deep_required |
| risky_ps_old | 55.000 | target_post_switch | repair_all | repair_all | 0.000 | 5.137 | 3.000 | 5.206 | True | True | True | fast/deep/deep/deep/deep | fast/fast/fast/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 56.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.873 | 3.500 | 21.943 | False | False | False | fast/deep/deep/deep/deep | deep/fast/fast/deep/deep | False | both_mismatch_types |
| risky_ps_old | 57.000 | target_post_switch | repair_subset | repair_subset | 11.000 | 4.232 | 0.000 | 15.312 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 58.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 5.257 | 2.000 | 22.325 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 59.000 | target_post_switch | repair_all | repair_subset | 14.000 | 5.327 | 4.500 | 19.391 | False | False | False | fast/deep/deep/deep/deep | fast/fast/fast/fast/deep | False | fast_on_deep_required |
| risky_ps_old | 60.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.805 | 0.500 | 3.881 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 61.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.073 | 0.000 | 3.152 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 62.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.531 | 1.500 | 3.601 | True | True | True | fast/deep/deep/deep/deep | fast/deep/fast/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 63.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.441 | 0.000 | 3.517 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 64.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.953 | 0.500 | 4.030 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 65.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.058 | 2.000 | 4.134 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 66.000 | target_post_switch | repair_subset | transfer | 22.500 | 5.277 | 3.000 | 27.842 | False | False | False | fast/deep/deep/deep/deep | fast/deep/fast/deep/fast | False | fast_on_deep_required |
| risky_ps_old | 67.000 | target_post_switch | repair_subset | transfer | 23.500 | 4.608 | 2.000 | 28.175 | False | False | False | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 68.000 | target_post_switch | repair_subset | repair_subset | 17.000 | 4.526 | 1.500 | 21.593 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 69.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.315 | 2.000 | 4.383 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 70.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.324 | 0.000 | 3.399 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 71.000 | target_post_switch | repair_all | transfer | 18.000 | 3.000 | 1.500 | 21.073 | False | True | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 72.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.690 | 0.500 | 3.768 | True | True | True | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 73.000 | target_post_switch | repair_all | repair_all | 0.000 | 4.408 | 2.000 | 4.482 | True | True | True | fast/deep/deep/deep/deep | deep/fast/deep/deep/deep | False | both_mismatch_types |
| risky_ps_old | 74.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.614 | 0.000 | 3.686 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |
| risky_ps_old | 75.000 | target_post_switch | repair_all | transfer | 18.500 | 4.225 | 0.500 | 22.802 | False | True | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 76.000 | target_post_switch | repair_subset | repair_subset | 6.000 | 4.947 | 0.500 | 11.023 | True | False | False | fast/deep/deep/deep/deep | deep/deep/deep/deep/deep | False | deep_on_fast_required |
| risky_ps_old | 77.000 | target_post_switch | repair_subset | transfer | 23.500 | 3.922 | 1.500 | 27.488 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 78.000 | target_post_switch | repair_subset | repair_subset | 19.000 | 3.610 | 1.500 | 22.681 | False | False | False | fast/deep/deep/deep/deep | fast/fast/deep/deep/deep | False | fast_on_deep_required |
| risky_ps_old | 79.000 | target_post_switch | repair_all | repair_all | 0.000 | 3.907 | 0.000 | 3.986 | True | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | True | all_stage_modes_match |

## Schedule Composition

| phase | oracle | n |
| --- | --- | --- |
| target_post_switch | repair_all | 51.000 |
| target_post_switch | repair_subset | 24.000 |
| trap_pre_switch | repair_all | 25.000 |
