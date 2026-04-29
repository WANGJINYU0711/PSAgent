# LLM v5 Fixed Layer1 Probability Sanity Report

Experiment: `llm_v5_no_archetype_intervention_layer1_fixedprob_d4_eta03_eps001_r2_3methods`

Source baseline: `llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods`

## Schedule Clarification

- `repeats=2`, `cycle_length=10`, `horizon=20`: this is **20 total episodes computed as `cycle_length=10` × `repeats=2`**, not 20 tasks repeated 2 times. It is also not literally the same fixed 10 task IDs replayed twice: the trap-switch scheduler draws from 10 trap and 10 target bucket lists, switches after episode index 5, so this horizon consumes 5 trap episodes and 15 target episodes. The run config lists 15 unique dataset indices because those 20 scheduled episodes cover 5 trap tasks plus 10 target tasks, with some target tasks repeated after the target bucket wraps.

## Switch-Time Layer1 Probabilities

| method | snapshot_episode_1based | child_id | prob | distribution_kind |
| --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 6 | stage1_n1__from__root__c01 | 0.25598317584382285 | stagewise_marginal_mixture |
| direct_multistage_exp3 | 6 | stage1_n2__from__root__c02 | 0.09191012532375376 | stagewise_marginal_mixture |
| direct_multistage_exp3 | 6 | stage1_n3__from__root__c03 | 0.25598317584382285 | stagewise_marginal_mixture |
| direct_multistage_exp3 | 6 | stage1_n4__from__root__c04 | 0.25598317584382285 | stagewise_marginal_mixture |
| direct_multistage_exp3 | 6 | stage1_n5__from__root__c05 | 0.14014034714477783 | stagewise_marginal_mixture |
| epsilon_exp3 | 6 | stage1_n1__from__root__c01 | 0.2587923986159856 | stagewise_marginal_mixture |
| epsilon_exp3 | 6 | stage1_n2__from__root__c02 | 0.2587923986159856 | stagewise_marginal_mixture |
| epsilon_exp3 | 6 | stage1_n3__from__root__c03 | 0.15612340428614677 | stagewise_marginal_mixture |
| epsilon_exp3 | 6 | stage1_n4__from__root__c04 | 0.15229249309440304 | stagewise_marginal_mixture |
| epsilon_exp3 | 6 | stage1_n5__from__root__c05 | 0.1739993053874791 | stagewise_marginal_mixture |
| risky_ps_linear | 6 | stage1_n1__from__root__c01 | 0.2581366774788512 | ps_risky_marginal_mixture |
| risky_ps_linear | 6 | stage1_n2__from__root__c02 | 0.20325497777043017 | ps_risky_marginal_mixture |
| risky_ps_linear | 6 | stage1_n3__from__root__c03 | 0.127180784801162 | ps_risky_marginal_mixture |
| risky_ps_linear | 6 | stage1_n4__from__root__c04 | 0.19293049251089017 | ps_risky_marginal_mixture |
| risky_ps_linear | 6 | stage1_n5__from__root__c05 | 0.21849706743866645 | ps_risky_marginal_mixture |

## Fixed-vs-Dynamic Cost/Success Compare

| method | split | dynamic_total | fixed_total | delta_total_fixed_minus_dynamic | dynamic_terminal | fixed_terminal | delta_terminal | dynamic_reasoning | fixed_reasoning | delta_reasoning | dynamic_exact | fixed_exact | dynamic_strict_clean | fixed_strict_clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 13.115 | 12.645 | -0.470 | 8.050 | 7.475 | -0.575 | 4.996 | 5.100 | 0.104 | 0.150 | 0.200 | 0.100 | 0.150 |
| direct_multistage_exp3 | pre | 6.634 | 6.265 | -0.369 | 1.800 | 1.400 | -0.400 | 4.764 | 4.794 | 0.031 | 0.400 | 0.400 | 0.400 | 0.400 |
| direct_multistage_exp3 | post | 15.275 | 14.772 | -0.503 | 10.133 | 9.500 | -0.633 | 5.074 | 5.202 | 0.128 | 0.067 | 0.133 | 0.000 | 0.067 |
| epsilon_exp3 | all | 16.821 | 14.468 | -2.353 | 11.600 | 9.275 | -2.325 | 5.152 | 5.123 | -0.029 | 0.200 | 0.250 | 0.100 | 0.200 |
| epsilon_exp3 | pre | 6.702 | 6.472 | -0.230 | 1.600 | 1.600 | 0.000 | 5.034 | 4.804 | -0.230 | 0.400 | 0.400 | 0.400 | 0.400 |
| epsilon_exp3 | post | 20.194 | 17.133 | -3.060 | 14.933 | 11.833 | -3.100 | 5.191 | 5.230 | 0.038 | 0.133 | 0.200 | 0.000 | 0.133 |
| risky_ps_linear | all | 13.687 | 12.591 | -1.097 | 8.625 | 7.750 | -0.875 | 4.992 | 4.770 | -0.223 | 0.200 | 0.200 | 0.100 | 0.150 |
| risky_ps_linear | pre | 5.861 | 6.484 | 0.623 | 1.400 | 1.800 | 0.400 | 4.391 | 4.614 | 0.223 | 0.400 | 0.400 | 0.400 | 0.400 |
| risky_ps_linear | post | 16.296 | 14.626 | -1.670 | 11.033 | 9.733 | -1.300 | 5.192 | 4.821 | -0.371 | 0.133 | 0.133 | 0.000 | 0.067 |

## Fixed Run Summary

| method | split | n | terminal | reasoning | path | total | exact | policy_clean | subset_clean | strict_clean | assistant_required_n | assistant_success_proxy | fast_on_deep | deep_on_fast | match_count |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all | 20 | 7.475 | 5.100 | 0.070 | 12.645 | 0.200 | 0.450 | 0.200 | 0.150 | 13 | 0.154 | 0.900 | 1.050 | 3.050 |
| direct_multistage_exp3 | pre | 5 | 1.400 | 4.794 | 0.070 | 6.265 | 0.400 | 0.800 | 0.400 | 0.400 | 1 | 0.000 | 0.000 | 3.600 | 1.400 |
| direct_multistage_exp3 | post | 15 | 9.500 | 5.202 | 0.070 | 14.772 | 0.133 | 0.333 | 0.133 | 0.067 | 12 | 0.167 | 1.200 | 0.200 | 3.600 |
| epsilon_exp3 | all | 20 | 9.275 | 5.123 | 0.070 | 14.468 | 0.250 | 0.400 | 0.250 | 0.200 | 13 | 0.077 | 0.800 | 1.250 | 2.950 |
| epsilon_exp3 | pre | 5 | 1.600 | 4.804 | 0.068 | 6.472 | 0.400 | 1.000 | 0.400 | 0.400 | 1 | 1.000 | 0.000 | 3.600 | 1.400 |
| epsilon_exp3 | post | 15 | 11.833 | 5.230 | 0.070 | 17.133 | 0.200 | 0.200 | 0.200 | 0.133 | 12 | 0.000 | 1.067 | 0.467 | 3.467 |
| risky_ps_linear | all | 20 | 7.750 | 4.770 | 0.071 | 12.591 | 0.200 | 0.400 | 0.200 | 0.150 | 13 | 0.077 | 0.800 | 0.900 | 3.300 |
| risky_ps_linear | pre | 5 | 1.800 | 4.614 | 0.069 | 6.484 | 0.400 | 0.800 | 0.400 | 0.400 | 1 | 0.000 | 0.000 | 3.200 | 1.800 |
| risky_ps_linear | post | 15 | 9.733 | 4.821 | 0.072 | 14.626 | 0.133 | 0.267 | 0.133 | 0.067 | 12 | 0.083 | 1.067 | 0.133 | 3.800 |

## Mismatch Bucket Summary

| method | bucket | n | terminal | reasoning | total | exact | strict_clean | assistant_success_proxy | post_fraction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 4 | 6.875 | 4.524 | 11.476 | 0.250 | 0.000 | 0.250 | 1.000 |
| direct_multistage_exp3 | both_mismatch_types | 3 | 9.333 | 5.680 | 15.081 | 0.333 | 0.333 | 0.000 | 1.000 |
| direct_multistage_exp3 | deep_on_fast_required | 5 | 1.400 | 4.794 | 6.265 | 0.400 | 0.400 | 0.000 | 0.000 |
| direct_multistage_exp3 | fast_on_deep_required | 8 | 10.875 | 5.361 | 16.305 | 0.000 | 0.000 | 0.167 | 1.000 |
| epsilon_exp3 | all_stage_modes_match | 4 | 10.500 | 5.171 | 15.746 | 0.250 | 0.000 | 0.000 | 1.000 |
| epsilon_exp3 | both_mismatch_types | 5 | 8.500 | 5.494 | 14.059 | 0.200 | 0.200 | 0.000 | 1.000 |
| epsilon_exp3 | deep_on_fast_required | 7 | 6.714 | 5.063 | 11.848 | 0.286 | 0.286 | 0.500 | 0.286 |
| epsilon_exp3 | fast_on_deep_required | 4 | 13.500 | 4.718 | 18.287 | 0.250 | 0.250 | 0.000 | 1.000 |
| risky_ps_linear | all_stage_modes_match | 5 | 12.400 | 4.949 | 17.424 | 0.200 | 0.000 | 0.200 | 1.000 |
| risky_ps_linear | both_mismatch_types | 1 | 5.000 | 6.086 | 11.150 | 0.000 | 0.000 | 0.000 | 1.000 |
| risky_ps_linear | deep_on_fast_required | 6 | 1.500 | 4.651 | 6.221 | 0.500 | 0.500 | 0.000 | 0.167 |
| risky_ps_linear | fast_on_deep_required | 8 | 9.875 | 4.582 | 14.527 | 0.000 | 0.000 | 0.000 | 1.000 |

## Majority Fast/Deep Pair Summary

| method | majority_pair | n | terminal | reasoning | total | exact | strict_clean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_vs_mostly_deep_required | 9 | 7.500 | 4.814 | 12.386 | 0.111 | 0.000 |
| direct_multistage_exp3 | mostly_deep_vs_mostly_fast_required | 5 | 1.400 | 4.794 | 6.265 | 0.400 | 0.400 |
| direct_multistage_exp3 | mostly_fast_vs_mostly_deep_required | 6 | 12.500 | 5.784 | 18.351 | 0.167 | 0.167 |
| epsilon_exp3 | mostly_deep_vs_mostly_deep_required | 11 | 12.727 | 5.265 | 18.065 | 0.182 | 0.091 |
| epsilon_exp3 | mostly_deep_vs_mostly_fast_required | 5 | 1.600 | 4.804 | 6.472 | 0.400 | 0.400 |
| epsilon_exp3 | mostly_fast_vs_mostly_deep_required | 4 | 9.375 | 5.133 | 14.573 | 0.250 | 0.250 |
| risky_ps_linear | mostly_deep_vs_mostly_deep_required | 11 | 10.727 | 4.701 | 15.502 | 0.182 | 0.091 |
| risky_ps_linear | mostly_deep_vs_mostly_fast_required | 5 | 1.800 | 4.614 | 6.484 | 0.400 | 0.400 |
| risky_ps_linear | mostly_fast_vs_mostly_deep_required | 4 | 7.000 | 5.153 | 12.218 | 0.000 | 0.000 |

## Stage-Level Mode Pair Summary (ALL only)

| method | required | actual | n_stage_obs | terminal | reasoning | total | exact | strict_clean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | deep | deep | 42 | 9.048 | 5.026 | 14.145 | 0.119 | 0.024 |
| direct_multistage_exp3 | deep | fast | 18 | 10.556 | 5.613 | 16.236 | 0.167 | 0.167 |
| direct_multistage_exp3 | fast | deep | 21 | 2.524 | 4.925 | 7.519 | 0.429 | 0.429 |
| direct_multistage_exp3 | fast | fast | 19 | 6.553 | 4.972 | 11.595 | 0.158 | 0.105 |
| epsilon_exp3 | deep | deep | 44 | 12.636 | 5.221 | 17.930 | 0.182 | 0.091 |
| epsilon_exp3 | deep | fast | 16 | 9.625 | 5.253 | 14.943 | 0.250 | 0.250 |
| epsilon_exp3 | fast | deep | 25 | 4.540 | 5.087 | 9.696 | 0.280 | 0.280 |
| epsilon_exp3 | fast | fast | 15 | 6.933 | 4.757 | 11.759 | 0.400 | 0.333 |
| risky_ps_linear | deep | deep | 44 | 10.205 | 4.795 | 15.072 | 0.182 | 0.091 |
| risky_ps_linear | deep | fast | 16 | 8.438 | 4.895 | 13.400 | 0.000 | 0.000 |
| risky_ps_linear | fast | deep | 18 | 2.111 | 4.730 | 6.910 | 0.389 | 0.389 |
| risky_ps_linear | fast | fast | 22 | 6.955 | 4.661 | 11.686 | 0.227 | 0.182 |

## Episode Compact View

| method | episode_1based | schedule_phase | raw_terminal_penalty | raw_reasoning_cost_component | raw_total_cost | strict_clean_success | assistant_required | assistant_success_proxy | actual_modes | required_modes | mismatch_bucket | majority_pair | root_conditional_prob | final_action | oracle_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 1 | trap_pre_switch | 1.500 | 5.405 | 6.976 | False | False | None | fast/deep/fast/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.200 | repair_all | repair_all |
| direct_multistage_exp3 | 2 | trap_pre_switch | 0.000 | 4.575 | 4.642 | True | False | None | deep/fast/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.211 | repair_all | repair_all |
| direct_multistage_exp3 | 3 | trap_pre_switch | 0.000 | 5.007 | 5.077 | True | False | None | fast/deep/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.160 | repair_all | repair_all |
| direct_multistage_exp3 | 4 | trap_pre_switch | 4.000 | 4.878 | 8.957 | False | True | False | deep/fast/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.188 | repair_subset | repair_subset |
| direct_multistage_exp3 | 5 | trap_pre_switch | 1.500 | 4.106 | 5.672 | False | False | None | deep/fast/deep/fast/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.130 | repair_all | repair_all |
| direct_multistage_exp3 | 6 | target_post_switch | 13.000 | 7.953 | 21.027 | False | True | False | fast/fast/fast/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 7 | target_post_switch | 13.000 | 5.902 | 18.968 | False | True | False | deep/fast/deep/deep/deep | fast/deep/deep/deep/deep | both_mismatch_types | mostly_deep_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 8 | target_post_switch | 11.000 | 5.518 | 16.589 | False | False | None | fast/fast/fast/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.256 | repair_subset | repair_all |
| direct_multistage_exp3 | 9 | target_post_switch | 21.000 | 4.265 | 25.340 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 10 | target_post_switch | 15.000 | 5.858 | 20.926 | False | True | False | fast/fast/fast/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.140 | repair_subset | repair_subset |
| direct_multistage_exp3 | 11 | target_post_switch | 3.000 | 3.909 | 6.971 | False | True | False | fast/fast/deep/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 12 | target_post_switch | 17.000 | 4.413 | 21.479 | False | False | None | fast/deep/deep/fast/fast | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.140 | repair_subset | repair_all |
| direct_multistage_exp3 | 13 | target_post_switch | 19.000 | 4.559 | 23.619 | False | True | False | fast/deep/deep/fast/fast | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 14 | target_post_switch | 15.000 | 4.736 | 19.805 | False | True | False | deep/fast/deep/deep/deep | fast/deep/deep/deep/deep | both_mismatch_types | mostly_deep_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 15 | target_post_switch | 1.000 | 4.864 | 5.941 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.256 | transfer | transfer |
| direct_multistage_exp3 | 16 | target_post_switch | 3.000 | 4.653 | 7.728 | False | True | False | fast/fast/deep/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.256 | repair_subset | repair_subset |
| direct_multistage_exp3 | 17 | target_post_switch | 2.000 | 4.425 | 6.501 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.140 | repair_subset | repair_subset |
| direct_multistage_exp3 | 18 | target_post_switch | 0.000 | 6.403 | 6.469 | True | False | None | deep/fast/fast/deep/fast | fast/deep/deep/deep/deep | both_mismatch_types | mostly_fast_vs_mostly_deep_required | 0.092 | repair_all | repair_all |
| direct_multistage_exp3 | 19 | target_post_switch | 6.000 | 6.027 | 12.098 | False | True | True | fast/deep/deep/deep/fast | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.140 | repair_all | repair_subset |
| direct_multistage_exp3 | 20 | target_post_switch | 3.500 | 4.544 | 8.123 | False | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.140 | repair_all | repair_subset |
| epsilon_exp3 | 1 | trap_pre_switch | 1.500 | 4.897 | 6.462 | False | False | None | fast/fast/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.200 | repair_all | repair_all |
| epsilon_exp3 | 2 | trap_pre_switch | 0.000 | 4.215 | 4.278 | True | False | None | deep/fast/deep/deep/fast | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.211 | repair_all | repair_all |
| epsilon_exp3 | 3 | trap_pre_switch | 0.000 | 4.320 | 4.379 | True | False | None | fast/deep/deep/deep/fast | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.163 | repair_all | repair_all |
| epsilon_exp3 | 4 | trap_pre_switch | 5.000 | 4.945 | 10.022 | False | True | True | fast/deep/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.226 | repair_all | repair_subset |
| epsilon_exp3 | 5 | trap_pre_switch | 1.500 | 5.642 | 7.218 | False | False | None | deep/deep/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.200 | repair_all | repair_all |
| epsilon_exp3 | 6 | target_post_switch | 18.500 | 5.027 | 23.603 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.152 | transfer | repair_subset |
| epsilon_exp3 | 7 | target_post_switch | 15.000 | 5.807 | 20.873 | False | True | False | deep/fast/fast/deep/deep | fast/deep/deep/deep/deep | both_mismatch_types | mostly_deep_vs_mostly_deep_required | 0.156 | repair_subset | repair_subset |
| epsilon_exp3 | 8 | target_post_switch | 18.500 | 5.867 | 24.445 | False | False | None | deep/deep/deep/deep/deep | fast/deep/deep/deep/deep | deep_on_fast_required | mostly_deep_vs_mostly_deep_required | 0.156 | transfer | repair_all |
| epsilon_exp3 | 9 | target_post_switch | 2.000 | 5.733 | 7.808 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.174 | repair_subset | repair_subset |
| epsilon_exp3 | 10 | target_post_switch | 21.500 | 5.565 | 27.134 | False | True | False | fast/deep/deep/deep/fast | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.152 | transfer | repair_subset |
| epsilon_exp3 | 11 | target_post_switch | 17.500 | 4.679 | 22.252 | False | True | False | fast/fast/fast/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.174 | transfer | repair_subset |
| epsilon_exp3 | 12 | target_post_switch | 0.000 | 4.134 | 4.205 | True | False | None | fast/fast/deep/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.259 | repair_all | repair_all |
| epsilon_exp3 | 13 | target_post_switch | 19.500 | 5.151 | 24.719 | False | True | False | deep/fast/deep/deep/deep | fast/deep/deep/deep/deep | both_mismatch_types | mostly_deep_vs_mostly_deep_required | 0.156 | transfer | repair_subset |
| epsilon_exp3 | 14 | target_post_switch | 20.500 | 5.553 | 26.128 | False | True | False | deep/deep/deep/deep/deep | fast/deep/deep/deep/deep | deep_on_fast_required | mostly_deep_vs_mostly_deep_required | 0.259 | transfer | repair_subset |
| epsilon_exp3 | 15 | target_post_switch | 3.000 | 4.967 | 8.040 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.174 | transfer | transfer |
| epsilon_exp3 | 16 | target_post_switch | 3.000 | 5.150 | 8.221 | False | True | False | deep/fast/deep/deep/deep | fast/deep/deep/deep/deep | both_mismatch_types | mostly_deep_vs_mostly_deep_required | 0.259 | repair_subset | repair_subset |
| epsilon_exp3 | 17 | target_post_switch | 15.000 | 4.493 | 19.555 | False | True | False | fast/fast/fast/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.259 | repair_subset | repair_subset |
| epsilon_exp3 | 18 | target_post_switch | 0.000 | 5.670 | 5.730 | True | False | None | deep/fast/fast/deep/fast | fast/deep/deep/deep/deep | both_mismatch_types | mostly_fast_vs_mostly_deep_required | 0.259 | repair_all | repair_all |
| epsilon_exp3 | 19 | target_post_switch | 18.500 | 4.956 | 23.535 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.259 | transfer | repair_subset |
| epsilon_exp3 | 20 | target_post_switch | 5.000 | 5.692 | 10.753 | False | True | False | deep/fast/fast/fast/deep | fast/deep/deep/deep/deep | both_mismatch_types | mostly_fast_vs_mostly_deep_required | 0.259 | repair_subset | repair_subset |
| risky_ps_linear | 1 | trap_pre_switch | 1.500 | 4.924 | 6.491 | False | False | None | fast/fast/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.200 | repair_all | repair_all |
| risky_ps_linear | 2 | trap_pre_switch | 0.000 | 4.203 | 4.269 | True | False | None | deep/fast/deep/deep/fast | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.211 | repair_all | repair_all |
| risky_ps_linear | 3 | trap_pre_switch | 0.000 | 3.982 | 4.057 | True | False | None | fast/fast/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.218 | repair_all | repair_all |
| risky_ps_linear | 4 | trap_pre_switch | 6.000 | 4.997 | 11.073 | False | True | False | deep/fast/deep/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.188 | repair_subset | repair_subset |
| risky_ps_linear | 5 | trap_pre_switch | 1.500 | 4.966 | 6.530 | False | False | None | deep/fast/fast/deep/deep | fast/fast/fast/fast/fast | deep_on_fast_required | mostly_deep_vs_mostly_fast_required | 0.245 | repair_all | repair_all |
| risky_ps_linear | 6 | target_post_switch | 18.500 | 5.024 | 23.598 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.218 | transfer | repair_subset |
| risky_ps_linear | 7 | target_post_switch | 5.000 | 6.086 | 11.150 | False | True | False | deep/fast/deep/fast/fast | fast/deep/deep/deep/deep | both_mismatch_types | mostly_fast_vs_mostly_deep_required | 0.203 | repair_subset | repair_subset |
| risky_ps_linear | 8 | target_post_switch | 5.000 | 5.649 | 10.718 | False | False | None | fast/deep/fast/fast/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.218 | repair_subset | repair_all |
| risky_ps_linear | 9 | target_post_switch | 13.000 | 4.636 | 17.709 | False | True | False | fast/fast/deep/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.258 | repair_subset | repair_subset |
| risky_ps_linear | 10 | target_post_switch | 20.500 | 3.880 | 24.459 | False | True | False | fast/deep/deep/fast/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.218 | transfer | repair_subset |
| risky_ps_linear | 11 | target_post_switch | 13.000 | 3.793 | 16.858 | False | True | False | fast/fast/fast/fast/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.218 | repair_subset | repair_subset |
| risky_ps_linear | 12 | target_post_switch | 18.500 | 4.531 | 23.103 | False | False | None | fast/fast/deep/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.218 | transfer | repair_all |
| risky_ps_linear | 13 | target_post_switch | 18.500 | 4.950 | 23.526 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.193 | transfer | repair_subset |
| risky_ps_linear | 14 | target_post_switch | 5.000 | 5.082 | 10.146 | False | True | False | fast/fast/fast/fast/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_fast_vs_mostly_deep_required | 0.258 | repair_subset | repair_subset |
| risky_ps_linear | 15 | target_post_switch | 1.000 | 4.397 | 5.462 | False | True | False | fast/deep/fast/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.193 | transfer | transfer |
| risky_ps_linear | 16 | target_post_switch | 3.000 | 4.687 | 7.759 | False | True | False | fast/fast/deep/deep/deep | fast/deep/deep/deep/deep | fast_on_deep_required | mostly_deep_vs_mostly_deep_required | 0.218 | repair_subset | repair_subset |
| risky_ps_linear | 17 | target_post_switch | 2.000 | 4.581 | 6.656 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.258 | repair_subset | repair_subset |
| risky_ps_linear | 18 | target_post_switch | 0.000 | 4.831 | 4.908 | True | False | None | deep/deep/deep/deep/deep | fast/deep/deep/deep/deep | deep_on_fast_required | mostly_deep_vs_mostly_deep_required | 0.127 | repair_all | repair_all |
| risky_ps_linear | 19 | target_post_switch | 6.000 | 5.326 | 11.402 | False | True | True | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.193 | repair_all | repair_subset |
| risky_ps_linear | 20 | target_post_switch | 17.000 | 4.865 | 21.938 | False | True | False | fast/deep/deep/deep/deep | fast/deep/deep/deep/deep | all_stage_modes_match | mostly_deep_vs_mostly_deep_required | 0.218 | repair_subset | repair_subset |

## Definitions And Caveats

- `strict_clean_success = exact_match && policy_violation_count == 0 && subset_mismatch == false`.
- `assistant_success_proxy` is auditable but imperfect: for tasks whose metadata says `contains_assistant_side_action=true`, it requires at least one assistant-side mutating tool call, a repair final action, and no policy violation. The repeated-smoke export still does not expose a single canonical hidden `clean_success_no_fallback` flag.
- Mode-pair cost summaries are association diagnostics: an episode contributes to every stage-pair it contains, so stage-level rows are not causal decompositions.


## Interpretation

- Best fixed-layer1 post-switch raw total is `risky_ps_linear` at `14.626`.

- `direct_multistage_exp3` improved vs dynamic on post-switch total by `-0.503` raw cost/episode. Terminal delta `-0.633`, reasoning delta `0.128`.

- `epsilon_exp3` improved vs dynamic on post-switch total by `-3.060` raw cost/episode. Terminal delta `-3.100`, reasoning delta `0.038`.

- `risky_ps_linear` improved vs dynamic on post-switch total by `-1.670` raw cost/episode. Terminal delta `-1.300`, reasoning delta `-0.371`.

- In this LLM run, fixed layer1 probabilities reduce total cost for all three methods versus the prior dynamic v5 baseline, mostly through lower terminal penalties rather than lower reasoning cost. This differs from the earlier controlled Bernoulli sanity, so the real LLM executor appears sensitive to stabilizing the root route after switch. Caveat: the dynamic baseline is the previous v5 run, not a fresh same-wall-clock paired rerun; small pre-switch deltas show ordinary LLM/run stochasticity and should not be attributed to the post-switch intervention.

- The mode bucket table shows the cheapest episodes are generally those where task requirements are fast but the selected path is deep (`deep_on_fast_required`); this is because the pre-switch/easy tasks are intrinsically cheaper and tolerate extra reasoning. The expensive cases are mostly post-switch deep-required tasks where the path remains too fast in at least one required-deep stage (`fast_on_deep_required` or `both_mismatch_types`).

- The main issue is still terminal quality on post-switch deep-required tasks: strict clean remains low in post split even when total cost improves. Fixing root choice alone does not solve Stage 4/5 completeness and subset correctness.


## Suggested Next Fixes

- Keep this fixed-layer1 variant as a candidate diagnostic, but do not claim final superiority from r=2/h=20 alone. Rerun with at least r=10/h=100 on the low-transfer smoke, ideally with a fresh paired dynamic control launched from the same code revision, before comparing algorithms.
- Export canonical executor flags in repeated smoke: `clean_success_no_fallback`, `hard_transfer_guard_applied`, Stage 4 completion-pass fields, and assistant-side mutation success.
- For local repair quality, split post-switch metrics by `repair_all` and `repair_subset`, and inspect the high-cost `fast_on_deep_required` episodes first.
