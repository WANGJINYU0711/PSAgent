# PS-favored trap controlled simulation

- tree_spec_cost_mode: `ps_favored_trap_v12_gap_compressed_baited`
- trap_basin_definition: `{'b1': 'stage1_n4', 'b2': ['stage2_n4', 'stage2_n5']}`
- trap_basin_leaf_count: `64`
- trap_path_base_aliases: `['stage1_n4', 'stage2_n5', 'stage3_n5', 'stage4_n5', 'stage5_n5']`
- exact_trap_path_exists: `True`
- trap_switch_denominator: `7`
- trap_switch_episode: `11`
- safe_basin_definition: `{'b3': ['stage3_n1', 'stage3_n2'], 'b4': 'stage4_n1', 'b5': ['stage5_n1', 'stage5_n2'], 'suffix_g': [0, 0, 0]}`
- cost_landscape_design: `v12_gap_compressed_baited_trap_late`
- target_candidate_leaf_count: `16`
- target_good_leaf_count: `4`
- target_bad_leaf_count: `12`
- target_good_distribution_by_b3: `{'stage3_n1': 3, 'stage3_n2': 1}`
- target_good_distribution_by_b5: `{'stage5_n1': 3, 'stage5_n2': 1}`
- stage1_n3_stage2_n2_decoy_count: `17`
- stage1_n3_stage2_n3_decoy_count: `15`
- pre_calibration_stage1_n3_marginal: `{'stage2_n1': 0.025527720110727372, 'stage2_n2': 0.03755364600807393, 'stage2_n3': 0.037943164739763384}`
- post_calibration_stage1_n3_marginal: `{'stage2_n1': 0.4721791759619722, 'stage2_n2': 0.14188680227626696, 'stage2_n3': 0.13292589204286773}`
- calibration_actions: `{'g2_decoy_count': 17, 'g3_decoy_count': 15, 'g1_target_bad_adjusted': True, 'g1_target_bad_adjusted_p_range': {'min': 0.9066924148854096, 'max': 0.9542547390354603}}`
- balancing_decoy_expected_p_range: `{'min': 0.05289113789581383, 'max': 0.06561911479242685}`
- root_child_marginal_expected_cost: `{'stage1_n1': 0.3074968327901743, 'stage1_n2': 0.09452397912268314, 'stage1_n3': 0.0891981350490497, 'stage1_n4': 0.4647216572340304, 'stage1_n5': 0.13292714644348771}`
- stage2_marginal_expected_cost: `{'stage1_n1': {'stage2_n1': 0.7691551820256206, 'stage2_n2': 0.4498076453810534, 'stage2_n3': 0.09858430075793452}, 'stage1_n2': {'stage2_n1': 0.07855020237490913, 'stage2_n2': 0.0936304029228966, 'stage2_n3': 0.09747012592842806}, 'stage1_n3': {'stage2_n1': 0.05488608440342842, 'stage2_n2': 0.09019521034083412, 'stage2_n3': 0.09242774888223142}}`
- exact_best_path: `['stage1_n3__from__root__c03', 'stage2_n1__from__n0003__c01', 'stage3_n2__from__n0012__c02', 'stage4_n1__from__n0042__c01', 'stage5_n1__from__n0121__c01']`
- exact_best_base_aliases: `['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1']`
- exact_best_expected_probability: `0.017378881907089774`
- safe_basin_leaf_count: `24`
- safe_suffix_group_count: `4`
- oracle_best_leaf_type: `shared`
- oracle_best_is_shared: `True`
- target_ordering_met: `False`

## Method Results
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | trap_basin_fraction_mean | target_subtree_fraction_mean | target_good_fraction_mean | target_bad_fraction_mean | calibrated_decoy_fraction_mean | decoy_branch_fraction_mean | ordinary_safe_basin_fraction_mean | broad_safe_basin_fraction_mean | ps_favored_exact_best_hit_rate_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| naive_mixed | -0.004878881907089775 | 0.0125 | 1.0 | 0.0 | 1.0 | 0.85 | 0.15 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 440 | 80 | 1 |
| naive_mixed_avg | -0.004878881907089775 | 0.0125 | 1.0 | 0.0 | 1.0 | 0.85 | 0.15 | 0.0 | 0.0 | 0.0 | 1.0 | 0.0 | 440 | 80 | 1 |
| epsilon_exp3 | 0.10762111809291022 | 0.125 | 0.95 | 0.05 | 0.325 | 0.0875 | 0.2375 | 0.075 | 0.35 | 0.075 | 0.4 | 0.0 | 440 | 80 | 1 |
| risky_ps_direct_cost | 0.17012111809291022 | 0.1875 | 0.9125 | 0.0375 | 0.225 | 0.1125 | 0.1125 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.0125 | 440 | 80 | 1 |
| risky_ps_safe_conditional | 0.17012111809291022 | 0.1875 | 0.9125 | 0.0375 | 0.225 | 0.1125 | 0.1125 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.0125 | 440 | 80 | 1 |
| risky_ps_safe_conditional_ix | 0.17012111809291022 | 0.1875 | 0.9125 | 0.0375 | 0.225 | 0.1125 | 0.1125 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.0125 | 440 | 80 | 1 |
| direct_multistage_exp3 | 0.1826211180929102 | 0.2 | 0.9375 | 0.025 | 0.3125 | 0.125 | 0.1875 | 0.2125 | 0.3 | 0.075 | 0.3875 | 0.05 | 440 | 80 | 1 |
| risky_ps | 0.1826211180929102 | 0.2 | 0.9125 | 0.0375 | 0.225 | 0.125 | 0.1 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.025 | 440 | 80 | 1 |
| risky_ps_ix | 0.1826211180929102 | 0.2 | 0.9125 | 0.0375 | 0.225 | 0.125 | 0.1 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.025 | 440 | 80 | 1 |
| risky_ps_linear | 0.1826211180929102 | 0.2 | 0.9125 | 0.0375 | 0.225 | 0.125 | 0.1 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.025 | 440 | 80 | 1 |
| risky_ps_old | 0.1826211180929102 | 0.2 | 0.9125 | 0.0375 | 0.225 | 0.125 | 0.1 | 0.1625 | 0.4 | 0.0375 | 0.2625 | 0.025 | 440 | 80 | 1 |
| direct_multistage_exp3_local | 0.2576211180929102 | 0.275 | 0.9125 | 0.0625 | 0.2125 | 0.1125 | 0.1 | 0.0625 | 0.3125 | 0.025 | 0.2375 | 0.025 | 440 | 80 | 1 |
| random_path | 0.2951211180929102 | 0.3125 | 0.925 | 0.1 | 0.225 | 0.05 | 0.175 | 0.05 | 0.2875 | 0.075 | 0.3 | 0.0125 | 440 | 80 | 1 |

## Ordering Checks
| risky_ps_better_than_epsilon_exp3 | epsilon_exp3_better_than_direct_multistage_exp3 | target_ordering_met |
| --- | --- | --- |
| False | True | False |

## Top-10 Leaf Expected Probabilities
| rank | mean_probability | leaf_type | family_label | base_aliases |
| --- | --- | --- | --- | --- |
| 1 | 0.017378881907089774 | shared | ps_favored_v12_exact_best_post_switch_gap_compressed | ['stage1_n3', 'stage2_n1', 'stage3_n2', 'stage4_n1', 'stage5_n1'] |
| 2 | 0.023587273931707023 | shared | ps_favored_v12_safe_decoy_post_switch_gap_compressed | ['stage1_n2', 'stage2_n2', 'stage3_n3', 'stage4_n3', 'stage5_n3'] |
| 3 | 0.023884644572200964 | shared | ps_favored_v12_safe_decoy_post_switch_gap_compressed | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |
| 4 | 0.02421044407930758 | shared | ps_favored_v12_safe_decoy_post_switch_gap_compressed | ['stage1_n2', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n2'] |
| 5 | 0.025167517041377808 | shared | ps_favored_v12_target_good_post_switch_gap_compressed | ['stage1_n3', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 6 | 0.02531514166276107 | shared | ps_favored_v12_safe_decoy_post_switch_gap_compressed | ['stage1_n2', 'stage2_n2', 'stage3_n4', 'stage4_n2', 'stage5_n1'] |
| 7 | 0.02552912909479077 | shared | ps_favored_v12_target_good_post_switch_gap_compressed | ['stage1_n2', 'stage2_n1', 'stage3_n1', 'stage4_n1', 'stage5_n1'] |
| 8 | 0.026411402103814558 | shared | ps_favored_v12_safe_decoy_post_switch_gap_compressed | ['stage1_n1', 'stage2_n3', 'stage3_n4', 'stage4_n4', 'stage5_n1'] |
| 9 | 0.05289113789581383 | shared | ps_favored_v12_balancing_decoy_post_switch_gap_compressed | ['stage1_n3', 'stage2_n3', 'stage3_n4', 'stage4_n3', 'stage5_n1'] |
| 10 | 0.0530742650956457 | shared | ps_favored_v12_balancing_decoy_post_switch_gap_compressed | ['stage1_n3', 'stage2_n2', 'stage3_n3', 'stage4_n2', 'stage5_n1'] |

## Top Safe Suffix Signatures
| signature | leaf_count | mean_probability |
| --- | --- | --- |
| ['stage3_n2', 'stage4_n1', 'stage5_n2'] | 8 | 0.25465057086860504 |
| ['stage3_n2', 'stage4_n1', 'stage5_n1'] | 8 | 0.25143020138642214 |
| ['stage3_n1', 'stage4_n1', 'stage5_n2'] | 4 | 0.26819632712741986 |
| ['stage3_n1', 'stage4_n1', 'stage5_n1'] | 4 | 0.2280341160586436 |
