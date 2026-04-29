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
- root_child_marginal_expected_cost: `{'stage1_n1': 0.6840096191091647, 'stage1_n2': 0.6622521106265935, 'stage1_n3': 0.7035262957516033, 'stage1_n4': 0.7687629497880041, 'stage1_n5': 0.8104232695651205}`
- stage2_marginal_expected_cost: `{'stage1_n1': {'stage2_n1': 0.6800283845094478, 'stage2_n2': 0.690070534659258, 'stage2_n3': 0.6780675506621552}, 'stage1_n2': {'stage2_n1': 0.7545170955443488, 'stage2_n2': 0.6642550837390296, 'stage2_n3': 0.6485908285799109}, 'stage1_n3': {'stage2_n1': 0.5715292184012936, 'stage2_n2': 0.7096754424446158, 'stage2_n3': 0.7134924620590664}}`
- exact_best_path: `['stage1_n3__from__root__c03', 'stage2_n1__from__n0003__c01', 'stage3_n2__from__n0012__c02', 'stage4_n1__from__n0042__c01', 'stage5_n1__from__n0121__c01']`
- exact_best_base_aliases: `['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1']`
- exact_best_expected_probability: `0.3142977449706376`
- safe_basin_leaf_count: `24`
- safe_suffix_group_count: `4`
- oracle_best_leaf_type: `shared`
- oracle_best_is_shared: `True`
- target_ordering_met: `False`

## Method Results
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | trap_basin_fraction_mean | target_subtree_fraction_mean | target_good_fraction_mean | target_bad_fraction_mean | calibrated_decoy_fraction_mean | decoy_branch_fraction_mean | ordinary_safe_basin_fraction_mean | broad_safe_basin_fraction_mean | ps_favored_exact_best_hit_rate_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_mixed_avg | -0.10196441163730428 | 0.21233333333333335 | 1.0 | 0.0 | 0.9483333333333333 | 0.5670000000000001 | 0.3813333333333333 | 0.04766666666666667 | 0.049666666666666665 | 0.032 | 0.9803333333333333 | 0.20299999999999999 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.028702255029362387 | 0.34299999999999997 | 0.9703333333333334 | 0.07366666666666667 | 0.4506666666666666 | 0.27299999999999996 | 0.17766666666666667 | 0.10233333333333333 | 0.31866666666666665 | 0.014666666666666666 | 0.4653333333333333 | 0.15966666666666665 | 440 | 1000 | 3 |
| risky_ps_ix | 0.039702255029362386 | 0.35400000000000004 | 0.9673333333333334 | 0.056999999999999995 | 0.28933333333333333 | 0.1426666666666667 | 0.14666666666666667 | 0.08266666666666667 | 0.3946666666666667 | 0.09399999999999999 | 0.38333333333333336 | 0.09499999999999999 | 440 | 1000 | 3 |
| risky_ps | 0.05836892169602906 | 0.3726666666666667 | 0.9743333333333334 | 0.051333333333333335 | 0.30433333333333334 | 0.15366666666666667 | 0.15066666666666667 | 0.159 | 0.45966666666666667 | 0.14033333333333334 | 0.4446666666666667 | 0.085 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.07136892169602906 | 0.38566666666666666 | 0.968 | 0.05533333333333334 | 0.37933333333333336 | 0.19866666666666666 | 0.18066666666666667 | 0.08533333333333333 | 0.37600000000000006 | 0.083 | 0.4623333333333333 | 0.03166666666666667 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.07403558836269573 | 0.38833333333333336 | 0.979 | 0.07666666666666666 | 0.4676666666666667 | 0.278 | 0.18966666666666665 | 0.15533333333333332 | 0.314 | 0.03966666666666667 | 0.5073333333333333 | 0.15233333333333335 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.07436892169602906 | 0.38866666666666666 | 0.9780000000000001 | 0.07633333333333332 | 0.468 | 0.276 | 0.19199999999999998 | 0.155 | 0.31333333333333335 | 0.03966666666666667 | 0.5076666666666666 | 0.14866666666666667 | 440 | 1000 | 3 |
| risky_ps_linear | 0.09703558836269573 | 0.41133333333333333 | 0.9733333333333333 | 0.052333333333333336 | 0.3243333333333333 | 0.10933333333333334 | 0.215 | 0.13999999999999999 | 0.43966666666666665 | 0.06899999999999999 | 0.39333333333333337 | 0.05633333333333334 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.10870225502936239 | 0.42300000000000004 | 0.9673333333333334 | 0.05266666666666667 | 0.39766666666666667 | 0.20533333333333334 | 0.19233333333333333 | 0.18666666666666668 | 0.39066666666666666 | 0.09733333333333334 | 0.49499999999999994 | 0.12733333333333333 | 440 | 1000 | 3 |
| risky_ps_old | 0.11670225502936239 | 0.431 | 0.9686666666666666 | 0.052 | 0.38233333333333336 | 0.208 | 0.17433333333333334 | 0.04033333333333333 | 0.33899999999999997 | 0.07633333333333332 | 0.4586666666666666 | 0.06233333333333333 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.17736892169602905 | 0.4916666666666667 | 0.9700000000000001 | 0.08 | 0.2843333333333333 | 0.09833333333333333 | 0.18600000000000003 | 0.14433333333333334 | 0.33966666666666673 | 0.08566666666666667 | 0.36999999999999994 | 0.04633333333333333 | 440 | 1000 | 3 |
| naive_mixed | 0.21303558836269573 | 0.5273333333333333 | 1.0 | 0.0 | 0.9966666666666667 | 0.24833333333333332 | 0.7483333333333334 | 0.0 | 0.002 | 0.001 | 0.9976666666666666 | 0.0 | 440 | 1000 | 3 |
| random_path | 0.41170225502936236 | 0.726 | 0.9266666666666667 | 0.08433333333333333 | 0.23266666666666666 | 0.03166666666666667 | 0.20099999999999998 | 0.08966666666666667 | 0.26 | 0.06233333333333333 | 0.295 | 0.017333333333333333 | 440 | 1000 | 3 |

## Ordering Checks
| risky_ps_better_than_epsilon_exp3 | epsilon_exp3_better_than_direct_multistage_exp3 | target_ordering_met |
| --- | --- | --- |
| True | False | False |

## Top-10 Leaf Expected Probabilities
| rank | mean_probability | leaf_type | family_label | base_aliases |
| --- | --- | --- | --- | --- |
| 1 | 0.3142977449706376 | shared | ps_favored_v11_exact_best_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1'] |
| 2 | 0.32319500265775397 | shared | ps_favored_v11_target_good_cyclic_target_phase | ['stage1_n3', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
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
| ['stage3_n2', 'stage4_n1', 'stage5_n1'] | 8 | 0.6258791402473456 |
| ['stage3_n1', 'stage4_n1', 'stage5_n2'] | 4 | 0.8111104505635803 |
| ['stage3_n1', 'stage4_n1', 'stage5_n1'] | 4 | 0.6057995297190354 |
