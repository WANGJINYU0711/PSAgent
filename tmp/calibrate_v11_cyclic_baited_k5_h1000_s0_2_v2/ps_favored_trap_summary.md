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
- target_good_leaf_count: `2`
- target_bad_leaf_count: `14`
- target_good_distribution_by_b3: `{'stage3_n1': 3, 'stage3_n2': 1}`
- target_good_distribution_by_b5: `{'stage5_n1': 3, 'stage5_n2': 1}`
- stage1_n3_stage2_n2_decoy_count: `17`
- stage1_n3_stage2_n3_decoy_count: `15`
- pre_calibration_stage1_n3_marginal: `{'stage2_n1': 0.4657811999804922, 'stage2_n2': 0.8519027937062562, 'stage2_n3': 0.838592444630292}`
- post_calibration_stage1_n3_marginal: `{'stage2_n1': 0.4719333783270411, 'stage2_n2': 0.5123916892863243, 'stage2_n3': 0.5231772721499174}`
- calibration_actions: `{'g2_decoy_count': 17, 'g3_decoy_count': 15, 'g1_target_bad_adjusted': True, 'g1_target_bad_adjusted_p_range': {'min': 0.9066924148854096, 'max': 0.9542547390354603}}`
- balancing_decoy_expected_p_range: `{'min': 0.6422030500013146, 'max': 0.6747206665917836}`
- root_child_marginal_expected_cost: `{'stage1_n1': 0.6882688829009476, 'stage1_n2': 0.6663722454027882, 'stage1_n3': 0.6950413185363882, 'stage1_n4': 0.7687629497880041, 'stage1_n5': 0.8104232695651205}`
- stage2_marginal_expected_cost: `{'stage1_n1': {'stage2_n1': 0.754565500865648, 'stage2_n2': 0.690070534659258, 'stage2_n3': 0.6780675506621552}, 'stage1_n2': {'stage2_n1': 0.8266194541277541, 'stage2_n2': 0.6642550837390296, 'stage2_n3': 0.6485908285799109}, 'stage1_n3': {'stage2_n1': 0.4230421171350281, 'stage2_n2': 0.7096754424446158, 'stage2_n3': 0.7134924620590664}}`
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
| naive_mixed_avg | 0.011260343782471012 | 0.02766666666666667 | 1.0 | 0.0 | 0.9783333333333334 | 0.7956666666666666 | 0.18066666666666667 | 0.01833333333333333 | 0.02033333333333333 | 0.013666666666666666 | 0.992 | 0.47900000000000004 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.14692701044913767 | 0.16333333333333333 | 0.9813333333333333 | 0.043000000000000003 | 0.8006666666666667 | 0.6976666666666667 | 0.08700000000000001 | 0.024666666666666667 | 0.09466666666666668 | 0.023000000000000003 | 0.8236666666666667 | 0.5243333333333333 | 440 | 1000 | 3 |
| risky_ps | 0.159260343782471 | 0.17566666666666667 | 0.987 | 0.04533333333333334 | 0.647 | 0.5376666666666666 | 0.08533333333333333 | 0.08933333333333333 | 0.24933333333333332 | 0.016666666666666666 | 0.6636666666666667 | 0.31966666666666665 | 440 | 1000 | 3 |
| risky_ps_ix | 0.16192701044913768 | 0.17833333333333334 | 0.9819999999999999 | 0.04566666666666667 | 0.6306666666666666 | 0.5226666666666667 | 0.085 | 0.08600000000000001 | 0.24766666666666667 | 0.018333333333333333 | 0.649 | 0.299 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.1769270104491377 | 0.19333333333333336 | 0.9886666666666667 | 0.065 | 0.7206666666666667 | 0.607 | 0.09033333333333333 | 0.05866666666666667 | 0.15566666666666668 | 0.021666666666666667 | 0.7423333333333333 | 0.3156666666666667 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.1769270104491377 | 0.19333333333333336 | 0.9886666666666667 | 0.065 | 0.7206666666666667 | 0.607 | 0.09033333333333333 | 0.05866666666666667 | 0.15566666666666668 | 0.021666666666666667 | 0.7423333333333333 | 0.3153333333333333 | 440 | 1000 | 3 |
| risky_ps_linear | 0.18959367711580435 | 0.206 | 0.9866666666666667 | 0.04533333333333334 | 0.6456666666666666 | 0.5023333333333334 | 0.11966666666666666 | 0.07266666666666667 | 0.21966666666666668 | 0.03266666666666667 | 0.6783333333333333 | 0.32 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.196260343782471 | 0.21266666666666667 | 0.9793333333333333 | 0.07366666666666667 | 0.612 | 0.49666666666666665 | 0.09666666666666666 | 0.013666666666666667 | 0.19266666666666665 | 0.057 | 0.669 | 0.2823333333333333 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.20759367711580434 | 0.224 | 0.975 | 0.052 | 0.61 | 0.498 | 0.08866666666666667 | 0.01 | 0.22766666666666666 | 0.045000000000000005 | 0.6549999999999999 | 0.17166666666666666 | 440 | 1000 | 3 |
| risky_ps_old | 0.27959367711580435 | 0.296 | 0.9773333333333333 | 0.050333333333333334 | 0.5486666666666666 | 0.40833333333333327 | 0.108 | 0.014 | 0.261 | 0.026 | 0.5746666666666667 | 0.27499999999999997 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.3389270104491377 | 0.3553333333333333 | 0.9780000000000001 | 0.064 | 0.5723333333333334 | 0.375 | 0.16866666666666666 | 0.127 | 0.238 | 0.03 | 0.6023333333333333 | 0.19233333333333333 | 440 | 1000 | 3 |
| naive_mixed | 0.661260343782471 | 0.6776666666666666 | 1.0 | 0.0 | 0.9966666666666667 | 0.08233333333333333 | 0.9119999999999999 | 0.0 | 0.002 | 0.001 | 0.9976666666666666 | 0.0 | 440 | 1000 | 3 |
| random_path | 0.7105936771158042 | 0.727 | 0.9266666666666667 | 0.08433333333333333 | 0.23266666666666666 | 0.03133333333333333 | 0.16733333333333333 | 0.08966666666666667 | 0.26 | 0.06233333333333333 | 0.295 | 0.015333333333333332 | 440 | 1000 | 3 |

## Ordering Checks
| risky_ps_better_than_epsilon_exp3 | epsilon_exp3_better_than_direct_multistage_exp3 | target_ordering_met |
| --- | --- | --- |
| True | False | False |

## Top-10 Leaf Expected Probabilities
| rank | mean_probability | leaf_type | family_label | base_aliases |
| --- | --- | --- | --- | --- |
| 1 | 0.01640632288419565 | shared | ps_favored_v11_exact_best_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1'] |
| 2 | 0.027138019679134068 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 3 | 0.44415041278167633 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n2', 'stage5_n1'] |
| 4 | 0.4447600862538346 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n1', 'stage2_n3', 'stage3_n4', 'stage4_n4', 'stage5_n1'] |
| 5 | 0.46825097398293314 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n3', 'stage4_n3', 'stage5_n3'] |
| 6 | 0.47033383103432663 | shared | ps_favored_v11_local_decoy_cyclic_target_phase | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |
| 7 | 0.4767740027267289 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n2'] |
| 8 | 0.51990286827976 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |
| 9 | 0.5223039656104204 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n4', 'stage5_n4'] |
| 10 | 0.5260501618545654 | shared | ps_favored_v11_stale_rotating_bait_corridor | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n4', 'stage5_n1'] |

## Top Safe Suffix Signatures
| signature | leaf_count | mean_probability |
| --- | --- | --- |
| ['stage3_n2', 'stage4_n1', 'stage5_n2'] | 8 | 0.7287009158654367 |
| ['stage3_n2', 'stage4_n1', 'stage5_n1'] | 8 | 0.6248690425835268 |
| ['stage3_n1', 'stage4_n1', 'stage5_n2'] | 4 | 0.8111104505635803 |
| ['stage3_n1', 'stage4_n1', 'stage5_n1'] | 4 | 0.6059720987200129 |
