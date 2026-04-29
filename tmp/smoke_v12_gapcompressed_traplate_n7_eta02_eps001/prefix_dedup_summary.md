# Prefix-dedup controlled simulation

- tree_spec_role_mode: `spec_or_agent_id`
- depth: `5`
- num_paths: `440`
- total_agent_ids: `655`
- duplicate_agent_count: `0`
- cross_prefix_duplicate_count: `0`

## Current Tree
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
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

## Delta vs Old Theory-Aligned Partial 4/5 Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| naive_mixed | -0.354103481764975 | -0.38367513403599185 | 0.97936 |
| epsilon_exp3 | -0.15774809883807692 | -0.18731975110909377 | 0.04204000000000008 |
| direct_multistage_exp3 | 0.10903541015683861 | 0.0794637578858218 | -0.043220000000000036 |
| risky_ps | -0.08274809883807693 | -0.11231975110909376 | 0.0045400000000000995 |
| random_path | -0.3344281802532366 | -0.3639998325242534 | 0.12670000000000003 |

## Direct Answers
- risky_ps_old vs risky_ps gap: 0.000000 (old=0.182621, new=0.182621).
- risky_ps vs epsilon_exp3 gap: 0.075000.
- risky_ps vs direct_multistage_exp3 gap: 0.000000.
- PS-family ranking on the prefix-dedup tree: 1. risky_ps_direct_cost (0.170121), 2. risky_ps_safe_conditional (0.170121), 3. risky_ps_safe_conditional_ix (0.170121), 4. risky_ps (0.182621), 5. risky_ps_ix (0.182621), 6. risky_ps_old (0.182621).
- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is smaller: current=0.037500, old_theory=0.095892.
- This tree preserves the original shared_basin_strong 4/5 minimal DAG connectivity and all original g values.
- The only structural change is parent-specific cloning of reused child aliases, so cross-prefix repeated agent identity is removed while local continuation patterns are unchanged.
- tree_spec_role_mode for this run: `spec_or_agent_id`.

## Old Theory-Aligned Reference
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.0735857079360716 | 0.12053624211417821 | 0.98072 | 3125 | 5000 | 10 |
| risky_ps | 0.26536921693098714 | 0.31231975110909377 | 0.9079599999999999 | 3125 | 5000 | 10 |
| epsilon_exp3 | 0.26536921693098714 | 0.31231975110909377 | 0.9079599999999999 | 3125 | 5000 | 10 |
| naive_mixed | 0.34922459985788523 | 0.39617513403599186 | 0.02064 | 3125 | 5000 | 10 |
| random_path | 0.6295492983461468 | 0.6764998325242534 | 0.7983 | 3125 | 5000 | 10 |
