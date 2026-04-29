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
| naive_mixed_avg | -0.10196441163730428 | 0.21233333333333335 | 1.0 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.028702255029362387 | 0.34299999999999997 | 0.9703333333333334 | 440 | 1000 | 3 |
| risky_ps_ix | 0.039702255029362386 | 0.35400000000000004 | 0.9673333333333334 | 440 | 1000 | 3 |
| risky_ps | 0.05836892169602906 | 0.3726666666666667 | 0.9743333333333334 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.07136892169602906 | 0.38566666666666666 | 0.968 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.07403558836269573 | 0.38833333333333336 | 0.979 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.07436892169602906 | 0.38866666666666666 | 0.9780000000000001 | 440 | 1000 | 3 |
| risky_ps_linear | 0.09703558836269573 | 0.41133333333333333 | 0.9733333333333333 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.10870225502936239 | 0.42300000000000004 | 0.9673333333333334 | 440 | 1000 | 3 |
| risky_ps_old | 0.11670225502936239 | 0.431 | 0.9686666666666666 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.17736892169602905 | 0.4916666666666667 | 0.9700000000000001 | 440 | 1000 | 3 |
| naive_mixed | 0.21303558836269573 | 0.5273333333333333 | 1.0 | 440 | 1000 | 3 |
| random_path | 0.41170225502936236 | 0.726 | 0.9266666666666667 | 440 | 1000 | 3 |

## Delta vs Old Theory-Aligned Partial 4/5 Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| direct_multistage_exp3 | -0.04488345290670921 | 0.22246375788582176 | -0.010386666666666655 |
| risky_ps | -0.20700029523495808 | 0.06034691555757293 | 0.0663733333333335 |
| epsilon_exp3 | -0.1940002952349581 | 0.07334691555757289 | 0.06004000000000009 |
| naive_mixed | -0.1361890114951895 | 0.13115819929734146 | 0.97936 |
| random_path | -0.21784704331678445 | 0.04950016747574659 | 0.12836666666666674 |

## Direct Answers
- risky_ps_old vs risky_ps gap: 0.058333 (old=0.116702, new=0.058369).
- risky_ps vs epsilon_exp3 gap: -0.013000.
- risky_ps vs direct_multistage_exp3 gap: 0.029667.
- PS-family ranking on the prefix-dedup tree: 1. risky_ps_ix (0.039702), 2. risky_ps (0.058369), 3. risky_ps_safe_conditional_ix (0.074036), 4. risky_ps_safe_conditional (0.074369), 5. risky_ps_old (0.116702), 6. risky_ps_direct_cost (0.177369).
- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is smaller: current=0.008333, old_theory=0.095892.
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
