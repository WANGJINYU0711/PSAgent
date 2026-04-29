# PS-favored trap controlled simulation

- tree_spec_cost_mode: `ps_favored_trap_v11_cyclic_baited`
- trap_basin_definition: `{'b1': 'stage1_n4', 'b2': ['stage2_n4', 'stage2_n5']}`
- trap_basin_leaf_count: `64`
- trap_path_base_aliases: `['stage1_n4', 'stage2_n5', 'stage3_n5', 'stage4_n5', 'stage5_n5']`
- exact_trap_path_exists: `True`
- trap_switch_denominator: `3`
- trap_switch_episode: `333`
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
- balancing_decoy_expected_p_range: `{'min': 0.6422030500013146, 'max': 0.6747206665917836}`
- root_child_marginal_expected_cost: `{'stage1_n1': 0.677746132496393, 'stage1_n2': 0.6550018028355593, 'stage1_n3': 0.6950413185363882, 'stage1_n4': 0.7687629497880041, 'stage1_n5': 0.8104232695651205}`
- stage2_marginal_expected_cost: `{'stage1_n1': {'stage2_n1': 0.5704173687859431, 'stage2_n2': 0.690070534659258, 'stage2_n3': 0.6780675506621552}, 'stage1_n2': {'stage2_n1': 0.6276367092012495, 'stage2_n2': 0.6642550837390296, 'stage2_n3': 0.6485908285799109}, 'stage1_n3': {'stage2_n1': 0.4230421171350281, 'stage2_n2': 0.7096754424446158, 'stage2_n3': 0.7134924620590664}}`
- exact_best_path: `['stage1_n3__from__root__c03', 'stage2_n1__from__n0003__c01', 'stage3_n2__from__n0012__c02', 'stage4_n1__from__n0042__c01', 'stage5_n1__from__n0121__c01']`
- exact_best_base_aliases: `['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1']`
- exact_best_expected_probability: `0.01640632288419565`
- safe_basin_leaf_count: `24`
- safe_suffix_group_count: `4`
- oracle_best_leaf_type: `shared`
- oracle_best_is_shared: `True`
- target_ordering_met: `False`

## Method Results
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | trap_basin_fraction_mean | target_subtree_fraction_mean | target_good_fraction_mean | target_bad_fraction_mean | calibrated_decoy_fraction_mean | decoy_branch_fraction_mean | ordinary_safe_basin_fraction_mean | broad_safe_basin_fraction_mean | ps_favored_exact_best_hit_rate_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_mixed_avg | 0.009431367288761953 | 0.022000000000000002 | 1.0 | 0.0 | 0.9733333333333333 | 0.7986666666666666 | 0.17466666666666666 | 0.016999999999999998 | 0.025333333333333333 | 0.013333333333333334 | 0.9866666666666667 | 0.20266666666666666 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.05343136728876196 | 0.066 | 0.9856666666666666 | 0.06366666666666666 | 0.8683333333333333 | 0.8069999999999999 | 0.06133333333333333 | 0.012333333333333333 | 0.042 | 0.005 | 0.8733333333333334 | 0.022333333333333334 | 440 | 1000 | 3 |
| risky_ps_old | 0.05709803395542862 | 0.06966666666666667 | 0.9913333333333334 | 0.042666666666666665 | 0.8776666666666667 | 0.8126666666666668 | 0.065 | 0.011666666666666667 | 0.048999999999999995 | 0.012333333333333333 | 0.89 | 0.006666666666666667 | 440 | 1000 | 3 |
| risky_ps_linear | 0.05743136728876195 | 0.07 | 0.9913333333333334 | 0.042 | 0.8746666666666667 | 0.8076666666666666 | 0.067 | 0.01 | 0.053 | 0.015 | 0.8896666666666667 | 0.004 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.06843136728876195 | 0.081 | 0.9883333333333333 | 0.052 | 0.8733333333333334 | 0.7920000000000001 | 0.08133333333333333 | 0.006333333333333333 | 0.04066666666666666 | 0.008333333333333333 | 0.8816666666666667 | 0.0006666666666666666 | 440 | 1000 | 3 |
| risky_ps | 0.08909803395542863 | 0.10166666666666667 | 0.9893333333333333 | 0.042 | 0.8353333333333334 | 0.7473333333333333 | 0.08800000000000001 | 0.011000000000000001 | 0.082 | 0.013666666666666667 | 0.8490000000000001 | 0.0036666666666666666 | 440 | 1000 | 3 |
| risky_ps_ix | 0.0900980339554286 | 0.10266666666666667 | 0.989 | 0.04133333333333333 | 0.8316666666666667 | 0.7423333333333334 | 0.08933333333333333 | 0.012333333333333335 | 0.08433333333333333 | 0.015333333333333332 | 0.847 | 0.005333333333333333 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.09143136728876194 | 0.104 | 0.9836666666666667 | 0.030333333333333334 | 0.835 | 0.7656666666666667 | 0.06933333333333333 | 0.013666666666666667 | 0.08766666666666667 | 0.013333333333333334 | 0.8483333333333333 | 0.0030000000000000005 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.10143136728876195 | 0.114 | 0.9923333333333333 | 0.05333333333333334 | 0.852 | 0.7616666666666667 | 0.09033333333333333 | 0.018 | 0.05833333333333333 | 0.016666666666666666 | 0.8686666666666666 | 0.127 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.10143136728876195 | 0.114 | 0.9923333333333333 | 0.05333333333333334 | 0.852 | 0.7616666666666667 | 0.09033333333333333 | 0.018 | 0.05833333333333333 | 0.016666666666666666 | 0.8686666666666666 | 0.127 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.323431367288762 | 0.336 | 0.9819999999999999 | 0.059000000000000004 | 0.6233333333333334 | 0.41933333333333334 | 0.204 | 0.14566666666666667 | 0.23433333333333337 | 0.035666666666666666 | 0.6589999999999999 | 0.19733333333333333 | 440 | 1000 | 3 |
| naive_mixed | 0.585431367288762 | 0.598 | 1.0 | 0.0 | 0.9966666666666667 | 0.17166666666666666 | 0.8250000000000001 | 0.0 | 0.002 | 0.001 | 0.9976666666666666 | 0.0 | 440 | 1000 | 3 |
| random_path | 0.6860980339554287 | 0.6986666666666667 | 0.9266666666666667 | 0.08433333333333333 | 0.23266666666666666 | 0.06533333333333334 | 0.16733333333333333 | 0.08966666666666667 | 0.26 | 0.06233333333333333 | 0.295 | 0.015333333333333332 | 440 | 1000 | 3 |

## Ordering Checks
| risky_ps_better_than_epsilon_exp3 | epsilon_exp3_better_than_direct_multistage_exp3 | target_ordering_met |
| --- | --- | --- |
| False | False | False |

## Top-10 Leaf Expected Probabilities
| rank | mean_probability | leaf_type | family_label | base_aliases |
| --- | --- | --- | --- | --- |
| 1 | 0.012568632711238047 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n1', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n2'] |
| 2 | 0.01640632288419565 | shared | ps_favored_v11_exact_best_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1'] |
| 3 | 0.027135958778855024 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n2', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 4 | 0.027138019679134068 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 5 | 0.44415041278167633 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n2', 'stage5_n1'] |
| 6 | 0.4447600862538346 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n1', 'stage2_n3', 'stage3_n4', 'stage4_n4', 'stage5_n1'] |
| 7 | 0.46825097398293314 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n3', 'stage4_n3', 'stage5_n3'] |
| 8 | 0.47033383103432663 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |
| 9 | 0.4767740027267289 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n2'] |
| 10 | 0.51990286827976 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |

## Top Safe Suffix Signatures
| signature | leaf_count | mean_probability |
| --- | --- | --- |
| ['stage3_n2', 'stage4_n1', 'stage5_n2'] | 8 | 0.7287009158654367 |
| ['stage3_n2', 'stage4_n1', 'stage5_n1'] | 8 | 0.6248690425835268 |
| ['stage3_n1', 'stage4_n1', 'stage5_n2'] | 4 | 0.6269623184838754 |
| ['stage3_n1', 'stage4_n1', 'stage5_n1'] | 4 | 0.4069893537935083 |
