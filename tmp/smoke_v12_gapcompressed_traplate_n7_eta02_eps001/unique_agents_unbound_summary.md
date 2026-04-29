# Unique-agent full-branching unbound controlled simulation

- tree_spec_role_mode: `spec_or_agent_id`
- num_paths: `440`
- duplicate_agent_count: `0`
- cross_prefix_duplicate_count: `0`

## Current Tree
| method | regret_per_t_mean | average_cost_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| naive_mixed | -0.004878881907089775 | 0.0125 | 1.0 | 440 | 80 | 1 |
| naive_mixed_avg | -0.004878881907089775 | 0.0125 | 1.0 | 440 | 80 | 1 |
| epsilon_exp3 | 0.10762111809291022 | 0.125 | 0.95 | 440 | 80 | 1 |
| risky_ps_direct_cost | 0.17012111809291022 | 0.1875 | 0.9125 | 440 | 80 | 1 |
| risky_ps_safe_conditional | 0.17012111809291022 | 0.1875 | 0.9125 | 440 | 80 | 1 |
| risky_ps_safe_conditional_ix | 0.17012111809291022 | 0.1875 | 0.9125 | 440 | 80 | 1 |
| direct_multistage_exp3 | 0.1826211180929102 | 0.2 | 0.9375 | 440 | 80 | 1 |
| risky_ps | 0.1826211180929102 | 0.2 | 0.9125 | 440 | 80 | 1 |
| risky_ps_ix | 0.1826211180929102 | 0.2 | 0.9125 | 440 | 80 | 1 |
| risky_ps_linear | 0.1826211180929102 | 0.2 | 0.9125 | 440 | 80 | 1 |
| risky_ps_old | 0.1826211180929102 | 0.2 | 0.9125 | 440 | 80 | 1 |
| direct_multistage_exp3_local | 0.2576211180929102 | 0.275 | 0.9125 | 440 | 80 | 1 |
| random_path | 0.2951211180929102 | 0.3125 | 0.925 | 440 | 80 | 1 |

## Delta vs Old Unique-Agent Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| naive_mixed | -0.025732153537004275 | -0.057275878977438174 | 0.0 |
| epsilon_exp3 | 0.0075731464272948135 | -0.023970579013139065 | -0.03799999999999992 |
| direct_multistage_exp3 | 0.13737532472186403 | 0.10583159928143018 | -0.05669999999999997 |
| risky_ps | 0.07950948434351926 | 0.04796575890308541 | -0.0754999999999999 |
| random_path | -0.17911838507760264 | -0.21066211051803652 | -0.0019999999999998908 |

## Delta vs Old Theory-Aligned Partial 4/5 Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| naive_mixed | -0.354103481764975 | -0.38367513403599185 | 0.97936 |
| epsilon_exp3 | -0.15774809883807692 | -0.18731975110909377 | 0.04204000000000008 |
| direct_multistage_exp3 | 0.10903541015683861 | 0.0794637578858218 | -0.043220000000000036 |
| risky_ps | -0.08274809883807693 | -0.11231975110909376 | 0.0045400000000000995 |
| random_path | -0.3344281802532366 | -0.3639998325242534 | 0.12670000000000003 |

## Direct Answers
- Current risky_ps vs epsilon_exp3 gap: 0.075000. Old unique-agent gap: 0.003064. Old theory-aligned gap: 0.000000.
- Current risky_ps vs direct_multistage_exp3 gap: 0.000000. Old unique-agent gap: 0.057866. Old theory-aligned gap: 0.191784.
- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is smaller: current=0.037500, old_theory=0.095892.
- Relative to the older bound unique-agent tree, the average risky_ps gap to epsilon/direct is not smaller: current=0.037500, old_unique=0.030465.
- PS-family ranking on this truly-unbound tree: 1. risky_ps_direct_cost (0.170121), 2. risky_ps_safe_conditional (0.170121), 3. risky_ps_safe_conditional_ix (0.170121), 4. risky_ps (0.182621), 5. risky_ps_ix (0.182621).
- naive_mixed rank is 1 with regret/T=-0.004879.
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
