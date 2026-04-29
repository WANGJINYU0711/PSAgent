# LLM v5 No-Archetype-Intervention Layer 1 Sanity

Date: 2026-04-28

Experiment name: `llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods`

Output directory: `tmp/llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods`

## Setting

| field | value |
| --- | --- |
| model | `gpt-4o-mini` |
| executor | `llm_bench` |
| family | `shared_basin_strong_prefix_dedup_profile_switch` |
| schedule | `trap_switch` |
| d | `4` |
| eta | `0.3` |
| epsilon | `0.01` |
| repeats | `2` |
| horizon per method | `20` |
| switch episode | `5` |
| methods | `direct_multistage_exp3, epsilon_exp3, risky_ps_linear` |
| archetype intervention | `disabled; route buckets are logging only` |
| attribute guidance | `disabled by executor setting` |
| path cost weight | `default 0.1` |

## Main Cost And Success Summary

| method | split | n | terminal | reasoning | path | total | exact | policy clean | subset clean | strict clean | match/stages | fast-on-deep | deep-on-fast | llm calls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | all | 20 | 8.050 | 4.996 | 0.0688 | 13.115 | 0.150 | 0.350 | 0.150 | 0.100 | 2.900 | 1.000 | 1.100 | 12.100 |
| `direct_multistage_exp3` | post | 15 | 10.133 | 5.074 | 0.0682 | 15.275 | 0.067 | 0.200 | 0.067 | 0.000 | 3.400 | 1.333 | 0.267 | 12.000 |
| `epsilon_exp3` | all | 20 | 11.600 | 5.152 | 0.0690 | 16.821 | 0.200 | 0.500 | 0.200 | 0.100 | 3.050 | 0.750 | 1.200 | 12.650 |
| `epsilon_exp3` | post | 15 | 14.933 | 5.191 | 0.0692 | 20.194 | 0.133 | 0.333 | 0.133 | 0.000 | 3.600 | 1.000 | 0.400 | 12.600 |
| `risky_ps_linear` | all | 20 | 8.625 | 4.992 | 0.0702 | 13.687 | 0.200 | 0.450 | 0.200 | 0.100 | 3.200 | 0.850 | 0.950 | 12.050 |
| `risky_ps_linear` | post | 15 | 11.033 | 5.192 | 0.0704 | 16.296 | 0.133 | 0.333 | 0.133 | 0.000 | 3.667 | 1.133 | 0.200 | 12.267 |

Definitions: `exact` is final structured exact match; `policy clean` means policy_violation_count=0; `subset clean` means no subset_mismatch; `strict clean` means exact + policy clean + subset clean. The repeated-smoke export does not expose a separate clean_success_no_fallback flag, so these are the auditable proxies from exported fields.

## Ranking By Raw Total Cost

| rank | method | raw total | terminal | reasoning | path | exact | strict clean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `direct_multistage_exp3` | 13.115 | 8.050 | 4.996 | 0.0688 | 0.150 | 0.100 |
| 2 | `risky_ps_linear` | 13.687 | 8.625 | 4.992 | 0.0702 | 0.200 | 0.100 |
| 3 | `epsilon_exp3` | 16.821 | 11.600 | 5.152 | 0.0690 | 0.200 | 0.100 |

## Mode-Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | total | exact | strict clean | avg fast-on-deep | avg deep-on-fast | avg match/stages |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | `all_stage_modes_match` | 2 | 9.500 | 4.573 | 14.145 | 0.000 | 0.000 | 0.000 | 0.000 | 5.000 |
| `direct_multistage_exp3` | `both_mismatch_types` | 3 | 13.000 | 6.042 | 19.109 | 0.000 | 0.000 | 1.667 | 1.000 | 2.333 |
| `direct_multistage_exp3` | `deep_on_fast_required` | 6 | 1.667 | 4.796 | 6.534 | 0.333 | 0.333 | 0.000 | 3.167 | 1.833 |
| `direct_multistage_exp3` | `fast_on_deep_required` | 9 | 10.333 | 4.875 | 15.276 | 0.111 | 0.000 | 1.667 | 0.000 | 3.333 |
| `epsilon_exp3` | `all_stage_modes_match` | 3 | 12.500 | 4.635 | 17.208 | 0.333 | 0.000 | 0.000 | 0.000 | 5.000 |
| `epsilon_exp3` | `both_mismatch_types` | 5 | 17.400 | 5.531 | 23.000 | 0.000 | 0.000 | 1.200 | 1.000 | 2.800 |
| `epsilon_exp3` | `deep_on_fast_required` | 6 | 1.667 | 5.052 | 6.788 | 0.500 | 0.333 | 0.000 | 3.167 | 1.833 |
| `epsilon_exp3` | `fast_on_deep_required` | 6 | 16.250 | 5.194 | 21.511 | 0.000 | 0.000 | 1.500 | 0.000 | 3.500 |
| `risky_ps_linear` | `all_stage_modes_match` | 6 | 7.250 | 4.953 | 12.280 | 0.333 | 0.000 | 0.000 | 0.000 | 5.000 |
| `risky_ps_linear` | `both_mismatch_types` | 3 | 16.667 | 5.658 | 22.393 | 0.000 | 0.000 | 2.000 | 1.000 | 2.000 |
| `risky_ps_linear` | `deep_on_fast_required` | 5 | 1.400 | 4.391 | 5.861 | 0.400 | 0.400 | 0.000 | 3.200 | 1.800 |
| `risky_ps_linear` | `fast_on_deep_required` | 6 | 12.000 | 5.199 | 17.264 | 0.000 | 0.000 | 1.833 | 0.000 | 3.167 |

## Majority Fast/Deep Pair Summary

| method | majority pair | n | terminal | reasoning | total | exact | strict clean |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | `mostly_deep_vs_mostly_deep_required` | 8 | 9.062 | 4.924 | 14.056 | 0.125 | 0.000 |
| `direct_multistage_exp3` | `mostly_deep_vs_mostly_fast_required` | 5 | 1.800 | 4.764 | 6.634 | 0.400 | 0.400 |
| `direct_multistage_exp3` | `mostly_fast_vs_mostly_deep_required` | 7 | 11.357 | 5.245 | 16.669 | 0.000 | 0.000 |
| `epsilon_exp3` | `mostly_deep_vs_mostly_deep_required` | 12 | 15.500 | 5.148 | 20.719 | 0.167 | 0.000 |
| `epsilon_exp3` | `mostly_deep_vs_mostly_fast_required` | 5 | 1.600 | 5.034 | 6.702 | 0.400 | 0.400 |
| `epsilon_exp3` | `mostly_fast_vs_mostly_deep_required` | 3 | 12.667 | 5.363 | 18.094 | 0.000 | 0.000 |
| `risky_ps_linear` | `mostly_deep_vs_mostly_deep_required` | 10 | 11.500 | 5.150 | 16.724 | 0.200 | 0.000 |
| `risky_ps_linear` | `mostly_deep_vs_mostly_fast_required` | 5 | 1.400 | 4.391 | 5.861 | 0.400 | 0.400 |
| `risky_ps_linear` | `mostly_fast_vs_mostly_deep_required` | 5 | 10.100 | 5.277 | 15.440 | 0.000 | 0.000 |

## Stage-Level Required/Actual Mode Pair Summary

`ALL` aggregates stage observations across the five stages. Costs are episode-level costs averaged over episodes containing that stage-pair observation, so use this as an association diagnostic, not a causal per-stage decomposition.

| method | stage | required | actual | n stage obs | terminal | reasoning | total | exact | strict clean |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | ALL | deep | deep | 40 | 9.637 | 4.985 | 14.692 | 0.075 | 0.000 |
| `direct_multistage_exp3` | ALL | deep | fast | 20 | 11.125 | 5.251 | 16.443 | 0.050 | 0.000 |
| `direct_multistage_exp3` | ALL | fast | deep | 22 | 3.318 | 4.942 | 8.331 | 0.364 | 0.364 |
| `direct_multistage_exp3` | ALL | fast | fast | 18 | 6.889 | 4.804 | 11.762 | 0.167 | 0.111 |
| `epsilon_exp3` | ALL | deep | deep | 45 | 14.533 | 5.135 | 19.739 | 0.178 | 0.000 |
| `epsilon_exp3` | ALL | deep | fast | 15 | 16.133 | 5.358 | 21.559 | 0.000 | 0.000 |
| `epsilon_exp3` | ALL | fast | deep | 24 | 5.042 | 5.245 | 10.356 | 0.292 | 0.250 |
| `epsilon_exp3` | ALL | fast | fast | 16 | 8.938 | 4.865 | 13.870 | 0.312 | 0.250 |
| `risky_ps_linear` | ALL | deep | deep | 43 | 10.512 | 5.116 | 15.700 | 0.186 | 0.000 |
| `risky_ps_linear` | ALL | deep | fast | 17 | 12.353 | 5.387 | 17.805 | 0.000 | 0.000 |
| `risky_ps_linear` | ALL | fast | deep | 19 | 3.947 | 4.599 | 8.616 | 0.316 | 0.316 |
| `risky_ps_linear` | ALL | fast | fast | 21 | 5.976 | 4.775 | 10.822 | 0.286 | 0.190 |

## Episode-Level Table

Full CSV: `episode_mode_cost_analysis.csv`. Compact view below.

| method | ep | phase | oracle -> final | terminal | reasoning | total | exact | policy clean | strict clean | required modes | actual modes | mismatch bucket | route bucket |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| `direct_multistage_exp3` | 0 | trap_pre_switch | repair_all -> repair_all | 1.500 | 5.128 | 6.699 | False | True | False | `fast/fast/fast/fast/fast` | `fast/deep/fast/deep/deep` | `deep_on_fast_required` | `neutral` |
| `direct_multistage_exp3` | 1 | trap_pre_switch | repair_all -> repair_all | 0.000 | 4.908 | 4.975 | True | True | True | `fast/fast/fast/fast/fast` | `deep/fast/deep/deep/deep` | `deep_on_fast_required` | `neutral` |
| `direct_multistage_exp3` | 2 | trap_pre_switch | repair_all -> repair_all | 0.000 | 4.698 | 4.768 | True | True | True | `fast/fast/fast/fast/fast` | `fast/deep/deep/deep/deep` | `deep_on_fast_required` | `neutral` |
| `direct_multistage_exp3` | 3 | trap_pre_switch | repair_subset -> repair_subset | 6.000 | 4.579 | 10.657 | False | False | False | `fast/fast/fast/fast/fast` | `deep/fast/deep/deep/deep` | `deep_on_fast_required` | `neutral` |
| `direct_multistage_exp3` | 4 | trap_pre_switch | repair_all -> repair_all | 1.500 | 4.507 | 6.073 | False | True | False | `fast/fast/fast/fast/fast` | `deep/fast/deep/fast/deep` | `deep_on_fast_required` | `neutral` |
| `direct_multistage_exp3` | 5 | target_post_switch | repair_subset -> repair_subset | 3.000 | 4.486 | 7.560 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/deep/deep` | `fast_on_deep_required` | `target_decoy_medium` |
| `direct_multistage_exp3` | 6 | target_post_switch | repair_subset -> repair_subset | 13.000 | 5.904 | 18.970 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `neutral` |
| `direct_multistage_exp3` | 7 | target_post_switch | repair_all -> repair_subset | 11.000 | 5.642 | 16.714 | False | True | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/deep/deep` | `fast_on_deep_required` | `target_decoy_medium` |
| `direct_multistage_exp3` | 8 | target_post_switch | repair_subset -> repair_subset | 5.000 | 4.648 | 9.723 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `target_safe_specialist_good` |
| `direct_multistage_exp3` | 9 | target_post_switch | repair_subset -> transfer | 21.500 | 5.417 | 26.985 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/deep/deep` | `fast_on_deep_required` | `neutral` |
| `direct_multistage_exp3` | 10 | target_post_switch | repair_subset -> repair_subset | 9.500 | 3.931 | 13.493 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/deep` | `fast_on_deep_required` | `trap_like_bad` |
| `direct_multistage_exp3` | 11 | target_post_switch | repair_all -> repair_subset | 11.000 | 5.436 | 16.501 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/fast/fast` | `fast_on_deep_required` | `neutral` |
| `direct_multistage_exp3` | 12 | target_post_switch | repair_subset -> repair_subset | 19.000 | 4.614 | 23.674 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/fast/fast` | `fast_on_deep_required` | `trap_like_bad` |
| `direct_multistage_exp3` | 13 | target_post_switch | repair_subset -> repair_subset | 15.000 | 5.917 | 20.986 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `target_safe_majority_bad` |
| `direct_multistage_exp3` | 14 | target_post_switch | transfer -> transfer | 1.000 | 4.959 | 6.033 | False | True | False | `fast/deep/deep/deep/deep` | `deep/deep/deep/deep/deep` | `deep_on_fast_required` | `target_safe_majority_bad` |
| `direct_multistage_exp3` | 15 | target_post_switch | repair_subset -> repair_subset | 13.000 | 4.680 | 17.756 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/deep` | `fast_on_deep_required` | `target_decoy_medium` |
| `direct_multistage_exp3` | 16 | target_post_switch | repair_subset -> repair_subset | 2.000 | 4.855 | 6.919 | True | False | False | `fast/deep/deep/deep/deep` | `fast/deep/fast/deep/deep` | `fast_on_deep_required` | `trap_like_bad` |
| `direct_multistage_exp3` | 17 | target_post_switch | repair_all -> repair_subset | 11.000 | 6.305 | 17.371 | False | True | False | `fast/deep/deep/deep/deep` | `deep/fast/fast/deep/fast` | `both_mismatch_types` | `target_decoy_medium` |
| `direct_multistage_exp3` | 18 | target_post_switch | repair_subset -> repair_subset | 3.000 | 4.817 | 7.878 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/fast` | `fast_on_deep_required` | `trap_like_bad` |
| `direct_multistage_exp3` | 19 | target_post_switch | repair_subset -> repair_subset | 14.000 | 4.497 | 18.567 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `trap_like_bad` |
| `epsilon_exp3` | 0 | trap_pre_switch | repair_all -> repair_all | 1.500 | 4.959 | 6.524 | False | True | False | `fast/fast/fast/fast/fast` | `fast/fast/deep/deep/deep` | `deep_on_fast_required` | `trap_like_good` |
| `epsilon_exp3` | 1 | trap_pre_switch | repair_all -> repair_all | 0.000 | 4.225 | 4.288 | True | True | True | `fast/fast/fast/fast/fast` | `deep/fast/deep/deep/fast` | `deep_on_fast_required` | `neutral` |
| `epsilon_exp3` | 2 | trap_pre_switch | repair_all -> repair_all | 0.000 | 4.734 | 4.793 | True | True | True | `fast/fast/fast/fast/fast` | `fast/deep/deep/deep/fast` | `deep_on_fast_required` | `trap_like_good` |
| `epsilon_exp3` | 3 | trap_pre_switch | repair_subset -> repair_all | 5.000 | 4.936 | 10.014 | False | True | False | `fast/fast/fast/fast/fast` | `fast/deep/deep/deep/deep` | `deep_on_fast_required` | `neutral` |
| `epsilon_exp3` | 4 | trap_pre_switch | repair_all -> repair_all | 1.500 | 6.315 | 7.891 | False | True | False | `fast/fast/fast/fast/fast` | `deep/deep/deep/deep/deep` | `deep_on_fast_required` | `trap_safe_overcautious` |
| `epsilon_exp3` | 5 | target_post_switch | repair_subset -> repair_subset | 2.000 | 5.143 | 7.218 | True | False | False | `fast/deep/deep/deep/deep` | `deep/deep/deep/deep/deep` | `deep_on_fast_required` | `target_safe_specialist_good` |
| `epsilon_exp3` | 6 | target_post_switch | repair_subset -> transfer | 21.500 | 5.913 | 27.483 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `target_safe_majority_bad` |
| `epsilon_exp3` | 7 | target_post_switch | repair_all -> transfer | 18.500 | 5.290 | 23.864 | False | True | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/deep` | `fast_on_deep_required` | `target_safe_specialist_good` |
| `epsilon_exp3` | 8 | target_post_switch | repair_subset -> transfer | 19.500 | 4.550 | 24.121 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/deep` | `fast_on_deep_required` | `neutral` |
| `epsilon_exp3` | 9 | target_post_switch | repair_subset -> transfer | 21.500 | 5.141 | 26.711 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `neutral` |
| `epsilon_exp3` | 10 | target_post_switch | repair_subset -> repair_subset | 17.000 | 4.219 | 21.288 | False | True | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `trap_like_bad` |
| `epsilon_exp3` | 11 | target_post_switch | repair_all -> transfer | 18.500 | 5.538 | 24.096 | False | True | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/deep/deep` | `fast_on_deep_required` | `trap_like_bad` |
| `epsilon_exp3` | 12 | target_post_switch | repair_subset -> repair_subset | 3.000 | 5.231 | 8.297 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `neutral` |
| `epsilon_exp3` | 13 | target_post_switch | repair_subset -> repair_subset | 2.000 | 4.519 | 6.592 | True | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `neutral` |
| `epsilon_exp3` | 14 | target_post_switch | transfer -> transfer | 1.000 | 4.337 | 5.405 | False | True | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/deep/deep` | `fast_on_deep_required` | `target_decoy_medium` |
| `epsilon_exp3` | 15 | target_post_switch | repair_subset -> transfer | 18.500 | 6.213 | 24.779 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/fast/fast/deep` | `fast_on_deep_required` | `neutral` |
| `epsilon_exp3` | 16 | target_post_switch | repair_subset -> transfer | 21.500 | 5.904 | 27.473 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `target_safe_majority_bad` |
| `epsilon_exp3` | 17 | target_post_switch | repair_all -> transfer | 18.500 | 5.167 | 23.743 | False | True | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `target_safe_specialist_good` |
| `epsilon_exp3` | 18 | target_post_switch | repair_subset -> transfer | 19.500 | 5.466 | 25.033 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/fast/deep/deep` | `both_mismatch_types` | `neutral` |
| `epsilon_exp3` | 19 | target_post_switch | repair_subset -> transfer | 21.500 | 5.236 | 26.802 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/deep` | `fast_on_deep_required` | `neutral` |
| `risky_ps_linear` | 0 | trap_pre_switch | repair_all -> repair_all | 1.500 | 4.908 | 6.475 | False | True | False | `fast/fast/fast/fast/fast` | `fast/fast/deep/deep/deep` | `deep_on_fast_required` | `trap_like_good` |
| `risky_ps_linear` | 1 | trap_pre_switch | repair_all -> repair_all | 0.000 | 4.231 | 4.296 | True | True | True | `fast/fast/fast/fast/fast` | `deep/fast/deep/deep/fast` | `deep_on_fast_required` | `neutral` |
| `risky_ps_linear` | 2 | trap_pre_switch | repair_all -> repair_all | 0.000 | 3.961 | 4.036 | True | True | True | `fast/fast/fast/fast/fast` | `fast/fast/deep/deep/deep` | `deep_on_fast_required` | `neutral` |
| `risky_ps_linear` | 3 | trap_pre_switch | repair_subset -> repair_subset | 4.000 | 4.541 | 8.617 | False | False | False | `fast/fast/fast/fast/fast` | `deep/fast/deep/deep/deep` | `deep_on_fast_required` | `neutral` |
| `risky_ps_linear` | 4 | trap_pre_switch | repair_all -> repair_all | 1.500 | 4.315 | 5.880 | False | True | False | `fast/fast/fast/fast/fast` | `deep/fast/fast/deep/deep` | `deep_on_fast_required` | `neutral` |
| `risky_ps_linear` | 5 | target_post_switch | repair_subset -> repair_subset | 2.000 | 4.994 | 7.069 | True | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `target_safe_specialist_good` |
| `risky_ps_linear` | 6 | target_post_switch | repair_subset -> repair_subset | 11.000 | 5.855 | 16.918 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/fast/fast` | `both_mismatch_types` | `neutral` |
| `risky_ps_linear` | 7 | target_post_switch | repair_all -> repair_subset | 11.000 | 5.562 | 16.622 | False | True | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/deep/deep` | `fast_on_deep_required` | `trap_like_bad` |
| `risky_ps_linear` | 8 | target_post_switch | repair_subset -> repair_subset | 2.000 | 4.527 | 6.599 | True | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `trap_like_bad` |
| `risky_ps_linear` | 9 | target_post_switch | repair_subset -> transfer | 21.500 | 6.152 | 27.725 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/fast/deep/deep` | `both_mismatch_types` | `target_decoy_medium` |
| `risky_ps_linear` | 10 | target_post_switch | repair_subset -> transfer | 17.500 | 4.968 | 22.535 | False | False | False | `fast/deep/deep/deep/deep` | `deep/fast/deep/deep/deep` | `both_mismatch_types` | `neutral` |
| `risky_ps_linear` | 11 | target_post_switch | repair_all -> transfer | 18.500 | 5.278 | 23.852 | False | True | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `neutral` |
| `risky_ps_linear` | 12 | target_post_switch | repair_subset -> repair_subset | 3.000 | 4.701 | 7.768 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/fast` | `fast_on_deep_required` | `neutral` |
| `risky_ps_linear` | 13 | target_post_switch | repair_subset -> transfer | 20.500 | 5.255 | 25.817 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/fast/deep/fast` | `fast_on_deep_required` | `trap_like_bad` |
| `risky_ps_linear` | 14 | target_post_switch | transfer -> transfer | 1.000 | 4.379 | 5.458 | False | True | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `target_safe_majority_bad` |
| `risky_ps_linear` | 15 | target_post_switch | repair_subset -> repair_all | 6.000 | 6.005 | 12.084 | False | True | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `target_safe_specialist_good` |
| `risky_ps_linear` | 16 | target_post_switch | repair_subset -> transfer | 21.500 | 5.305 | 26.876 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/deep/deep/deep` | `fast_on_deep_required` | `neutral` |
| `risky_ps_linear` | 17 | target_post_switch | repair_all -> repair_subset | 11.000 | 5.354 | 16.425 | False | True | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/fast` | `fast_on_deep_required` | `neutral` |
| `risky_ps_linear` | 18 | target_post_switch | repair_subset -> repair_subset | 5.000 | 5.015 | 10.076 | False | False | False | `fast/deep/deep/deep/deep` | `fast/fast/fast/fast/deep` | `fast_on_deep_required` | `neutral` |
| `risky_ps_linear` | 19 | target_post_switch | repair_subset -> repair_subset | 14.000 | 4.538 | 18.618 | False | False | False | `fast/deep/deep/deep/deep` | `fast/deep/deep/deep/deep` | `all_stage_modes_match` | `neutral` |

## Interpretation

- Best raw total in this tiny sanity run is `direct_multistage_exp3` at 13.115.
- This is a 3-method, 20-episode sanity run, not a final ranking.
- Attribute guidance is disabled in the executor, so capability-fit summaries are not included in the prompt. Agent `attribute_skill` may still exist internally for legacy structures, but this setting does not expose capability guidance to the LLM.
- Route buckets/archetypes are logging-only in this run; they do not rewrite stage5 output and do not multiply reasoning/token costs.
- The mode mismatch diagnostics let us inspect whether fast-on-deep-required paths pay more. In this small sample, episode-level terminal variation is still large, so treat these as directional diagnostics.
