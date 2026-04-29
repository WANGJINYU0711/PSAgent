# Unique-agent full-branching unbound controlled simulation

- tree_spec_role_mode: `spec_or_agent_id`
- num_paths: `440`
- duplicate_agent_count: `0`
- cross_prefix_duplicate_count: `0`

## Current Tree
| method | regret_per_t_mean | average_cost_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| naive_mixed_avg | 0.009431367288761953 | 0.022000000000000002 | 1.0 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.05343136728876196 | 0.066 | 0.9856666666666666 | 440 | 1000 | 3 |
| risky_ps_old | 0.05709803395542862 | 0.06966666666666667 | 0.9913333333333334 | 440 | 1000 | 3 |
| risky_ps_linear | 0.05743136728876195 | 0.07 | 0.9913333333333334 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.06843136728876195 | 0.081 | 0.9883333333333333 | 440 | 1000 | 3 |
| risky_ps | 0.08909803395542863 | 0.10166666666666667 | 0.9893333333333333 | 440 | 1000 | 3 |
| risky_ps_ix | 0.0900980339554286 | 0.10266666666666667 | 0.989 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.09143136728876194 | 0.104 | 0.9836666666666667 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.10143136728876195 | 0.114 | 0.9923333333333333 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.10143136728876195 | 0.114 | 0.9923333333333333 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.323431367288762 | 0.336 | 0.9819999999999999 | 440 | 1000 | 3 |
| naive_mixed | 0.585431367288762 | 0.598 | 1.0 | 440 | 1000 | 3 |
| random_path | 0.6860980339554287 | 0.6986666666666667 | 0.9266666666666667 | 440 | 1000 | 3 |

## Delta vs Old Unique-Agent Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| direct_multistage_exp3 | 0.008185573917715794 | -0.028168400718569825 | -0.008533333333333393 |
| epsilon_exp3 | -0.03161660437685346 | -0.06797057901313906 | 0.00033333333333340764 |
| risky_ps | -0.014013599793962314 | -0.05036757443024793 | 0.0013333333333334085 |
| naive_mixed | 0.5645780956588475 | 0.5282241210225618 | 0.0 |
| random_path | 0.21185853078491584 | 0.17550455614863014 | -0.0003333333333331856 |

## Delta vs Old Theory-Aligned Partial 4/5 Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| direct_multistage_exp3 | -0.020154340647309636 | -0.05453624211417821 | 0.004946666666666544 |
| epsilon_exp3 | -0.1969378496422252 | -0.23131975110909375 | 0.08037333333333341 |
| risky_ps | -0.1762711829755585 | -0.21065308444242709 | 0.08137333333333341 |
| naive_mixed | 0.23620676743087676 | 0.20182486596400812 | 0.97936 |
| random_path | 0.056548735609281864 | 0.022166834142413272 | 0.12836666666666674 |

## Direct Answers
- Current risky_ps vs epsilon_exp3 gap: 0.020667. Old unique-agent gap: 0.003064. Old theory-aligned gap: 0.000000.
- Current risky_ps vs direct_multistage_exp3 gap: 0.035667. Old unique-agent gap: 0.057866. Old theory-aligned gap: 0.191784.
- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is smaller: current=0.028167, old_theory=0.095892.
- Relative to the older bound unique-agent tree, the average risky_ps gap to epsilon/direct is smaller: current=0.028167, old_unique=0.030465.
- PS-family ranking on this truly-unbound tree: 1. risky_ps (0.089098), 2. risky_ps_ix (0.090098), 3. risky_ps_safe_conditional (0.101431), 4. risky_ps_safe_conditional_ix (0.101431), 5. risky_ps_direct_cost (0.323431).
- naive_mixed rank is 12 with regret/T=0.585431.
- In `--tree-spec` mode, the synthetic cost role is now resolved in this order: explicit `cost_role`/`synthetic_role`/`latent_role` from the spec, then `agent_id` when role mode is unbound, and only `base_alias` in explicit compatibility mode.
- On this run the external-tree suffix family is keyed by the unique cost-role sequence plus gate pattern, so repeated `base_alias` strings no longer create cross-prefix latent-family reuse.
- This is still a structure ablation, not an apples-to-apples replacement for the old theory-aligned tree. The new tree keeps full branching and `num_paths=3125`, so any comparison to the old compact DAG must be read with that caveat.
- If the PS-family gap still does not shrink enough here, that points more toward aggregation/update behavior than toward tree reuse confounding alone.

## Old Unique-Agent Reference
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| naive_mixed | 0.0208532716299145 | 0.06977587897743817 | 1.0 | 440 | 1000 | 5 |
| direct_multistage_exp3 | 0.045245793371046165 | 0.09416840071856983 | 0.9942 | 440 | 1000 | 5 |
| epsilon_exp3 | 0.10004797166561541 | 0.14897057901313907 | 0.9879999999999999 | 440 | 1000 | 5 |
| risky_ps | 0.10311163374939095 | 0.1520342410969146 | 0.9879999999999999 | 440 | 1000 | 5 |
| random_path | 0.47423950317051283 | 0.5231621105180365 | 0.9269999999999999 | 440 | 1000 | 5 |

## Old Theory-Aligned Reference
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.0735857079360716 | 0.12053624211417821 | 0.98072 | 3125 | 5000 | 10 |
| risky_ps | 0.26536921693098714 | 0.31231975110909377 | 0.9079599999999999 | 3125 | 5000 | 10 |
| epsilon_exp3 | 0.26536921693098714 | 0.31231975110909377 | 0.9079599999999999 | 3125 | 5000 | 10 |
| naive_mixed | 0.34922459985788523 | 0.39617513403599186 | 0.02064 | 3125 | 5000 | 10 |
| random_path | 0.6295492983461468 | 0.6764998325242534 | 0.7983 | 3125 | 5000 | 10 |
