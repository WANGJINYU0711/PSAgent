# PS-favored trap controlled simulation

- tree_spec_cost_mode: `ps_favored_trap_v11_cyclic_baited`
- trap_basin_definition: `{'b1': 'stage1_n4', 'b2': ['stage2_n4', 'stage2_n5']}`
- trap_basin_leaf_count: `64`
- trap_path_base_aliases: `['stage1_n4', 'stage2_n5', 'stage3_n5', 'stage4_n5', 'stage5_n5']`
- exact_trap_path_exists: `True`
- trap_switch_denominator: `3`
- trap_switch_episode: `20`
- safe_basin_definition: `{'b3': ['stage3_n1', 'stage3_n2'], 'b4': 'stage4_n1', 'b5': ['stage5_n1', 'stage5_n2'], 'suffix_g': [0, 0, 0]}`
- cost_landscape_design: `v11_cyclic_baited_fixed_safe_suffix`
- target_candidate_leaf_count: `16`
- target_good_leaf_count: `4`
- target_bad_leaf_count: `12`
- target_good_distribution_by_b3: `{'stage3_n1': 3, 'stage3_n2': 1}`
- target_good_distribution_by_b5: `{'stage5_n1': 3, 'stage5_n2': 1}`
- stage1_n3_stage2_n2_decoy_count: `17`
- stage1_n3_stage2_n3_decoy_count: `15`
- pre_calibration_stage1_n3_marginal: `{'stage2_n1': 0.4657811999804922, 'stage2_n2': 0.8519027937062562, 'stage2_n3': 0.838592444630292}`
- post_calibration_stage1_n3_marginal: `{'stage2_n1': 0.4719333783270411, 'stage2_n2': 0.5123916892863243, 'stage2_n3': 0.5231772721499174}`
- calibration_actions: `{'g2_decoy_count': 17, 'g3_decoy_count': 15, 'g1_target_bad_adjusted': True, 'g1_target_bad_adjusted_p_range': {'min': 0.9066924148854096, 'max': 0.9542547390354603}}`
- balancing_decoy_expected_p_range: `{'min': 0.514125736521828, 'max': 0.5489938134916168}`
- root_child_marginal_expected_cost: `{'stage1_n1': 0.6769348445031185, 'stage1_n2': 0.6432373972709208, 'stage1_n3': 0.6272356303626198, 'stage1_n4': 0.7079569831145549, 'stage1_n5': 0.7985968244337843}`
- stage2_marginal_expected_cost: `{'stage1_n1': {'stage2_n1': 0.5173507833585794, 'stage2_n2': 0.7151505220311883, 'stage2_n3': 0.6562786947726116}, 'stage1_n2': {'stage2_n1': 0.6006576450753996, 'stage2_n2': 0.6411658845151355, 'stage2_n3': 0.6507608485983828}, 'stage1_n3': {'stage2_n1': 0.4059230360078989, 'stage2_n2': 0.6369308815093174, 'stage2_n3': 0.6445985003135937}}`
- exact_best_path: `['stage1_n3__from__root__c03', 'stage2_n1__from__n0003__c01', 'stage3_n2__from__n0012__c02', 'stage4_n1__from__n0042__c01', 'stage5_n1__from__n0121__c01']`
- exact_best_base_aliases: `['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1']`
- exact_best_expected_probability: `0.021166148921981407`
- safe_basin_leaf_count: `24`
- safe_suffix_group_count: `4`
- oracle_best_leaf_type: `shared`
- oracle_best_is_shared: `True`
- target_ordering_met: `False`

## Method Results
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | trap_basin_fraction_mean | target_subtree_fraction_mean | target_good_fraction_mean | target_bad_fraction_mean | calibrated_decoy_fraction_mean | decoy_branch_fraction_mean | ordinary_safe_basin_fraction_mean | broad_safe_basin_fraction_mean | ps_favored_exact_best_hit_rate_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_mixed | 0.00577294923999484 | 0.016666666666666666 | 1.0 | 0.0 | 1.0 | 0.7333333333333333 | 0.26666666666666666 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 440 | 60 | 1 |
| naive_mixed_avg | 0.00577294923999484 | 0.016666666666666666 | 1.0 | 0.0 | 1.0 | 0.7333333333333333 | 0.26666666666666666 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 440 | 60 | 1 |
| direct_multistage_exp3 | 0.4224396159066615 | 0.43333333333333335 | 0.9166666666666666 | 0.016666666666666666 | 0.48333333333333334 | 0.35 | 0.13333333333333333 | 0.16666666666666666 | 0.21666666666666667 | 0.03333333333333333 | 0.5166666666666667 | 0.13333333333333333 | 440 | 60 | 1 |
| direct_multistage_exp3_local | 0.48910628257332817 | 0.5 | 0.9166666666666666 | 0.016666666666666666 | 0.43333333333333335 | 0.2833333333333333 | 0.15 | 0.13333333333333333 | 0.25 | 0.06666666666666667 | 0.5 | 0.13333333333333333 | 440 | 60 | 1 |
| risky_ps | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.05 | 0.13333333333333333 | 0.06666666666666667 | 0.38333333333333336 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| risky_ps_direct_cost | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.03333333333333333 | 0.15 | 0.05 | 0.36666666666666664 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| risky_ps_ix | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.05 | 0.13333333333333333 | 0.06666666666666667 | 0.38333333333333336 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| risky_ps_safe_conditional | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.03333333333333333 | 0.15 | 0.05 | 0.36666666666666664 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| risky_ps_safe_conditional_ix | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.03333333333333333 | 0.15 | 0.05 | 0.36666666666666664 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| risky_ps_linear | 0.5391062825733282 | 0.55 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.06666666666666667 | 0.11666666666666667 | 0.05 | 0.36666666666666664 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| risky_ps_old | 0.5391062825733282 | 0.55 | 0.9333333333333333 | 0.1 | 0.18333333333333332 | 0.06666666666666667 | 0.11666666666666667 | 0.05 | 0.36666666666666664 | 0.1 | 0.2833333333333333 | 0.0 | 440 | 60 | 1 |
| epsilon_exp3 | 0.6557729492399947 | 0.6666666666666666 | 0.95 | 0.11666666666666667 | 0.2833333333333333 | 0.05 | 0.23333333333333334 | 0.016666666666666666 | 0.21666666666666667 | 0.05 | 0.3333333333333333 | 0.0 | 440 | 60 | 1 |
| random_path | 0.7224396159066615 | 0.7333333333333333 | 0.9333333333333333 | 0.1 | 0.2 | 0.06666666666666667 | 0.13333333333333333 | 0.06666666666666667 | 0.2833333333333333 | 0.06666666666666667 | 0.26666666666666666 | 0.016666666666666666 | 440 | 60 | 1 |

## Ordering Checks
| risky_ps_better_than_epsilon_exp3 | epsilon_exp3_better_than_direct_multistage_exp3 | target_ordering_met |
| --- | --- | --- |
| True | False | False |

## Top-10 Leaf Expected Probabilities
| rank | mean_probability | leaf_type | family_label | base_aliases |
| --- | --- | --- | --- | --- |
| 1 | 0.010893717426671827 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n1', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n2'] |
| 2 | 0.021166148921981407 | shared | ps_favored_v11_exact_best_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1'] |
| 3 | 0.03157992302987191 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n2', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 4 | 0.03214097202988957 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 5 | 0.39314105266362165 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n2', 'stage5_n1'] |
| 6 | 0.39658459889958614 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n1', 'stage2_n3', 'stage3_n4', 'stage4_n4', 'stage5_n1'] |
| 7 | 0.4172416493084302 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n3', 'stage4_n3', 'stage5_n3'] |
| 8 | 0.4235895830750634 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |
| 9 | 0.4797945535927666 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n2'] |
| 10 | 0.514125736521828 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n3', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |

## Top Safe Suffix Signatures
| signature | leaf_count | mean_probability |
| --- | --- | --- |
| ['stage3_n2', 'stage4_n1', 'stage5_n2'] | 8 | 0.6794261688364202 |
| ['stage3_n2', 'stage4_n1', 'stage5_n1'] | 8 | 0.580085237909344 |
| ['stage3_n1', 'stage4_n1', 'stage5_n2'] | 4 | 0.5994293399519793 |
| ['stage3_n1', 'stage4_n1', 'stage5_n1'] | 4 | 0.3809368158919674 |
