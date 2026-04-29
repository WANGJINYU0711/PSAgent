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
| naive_mixed | 0.00577294923999484 | 0.016666666666666666 | 1.0 | 440 | 60 | 1 |
| naive_mixed_avg | 0.00577294923999484 | 0.016666666666666666 | 1.0 | 440 | 60 | 1 |
| direct_multistage_exp3 | 0.4224396159066615 | 0.43333333333333335 | 0.9166666666666666 | 440 | 60 | 1 |
| direct_multistage_exp3_local | 0.48910628257332817 | 0.5 | 0.9166666666666666 | 440 | 60 | 1 |
| risky_ps | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 440 | 60 | 1 |
| risky_ps_direct_cost | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 440 | 60 | 1 |
| risky_ps_ix | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 440 | 60 | 1 |
| risky_ps_safe_conditional | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 440 | 60 | 1 |
| risky_ps_safe_conditional_ix | 0.5224396159066615 | 0.5333333333333333 | 0.9333333333333333 | 440 | 60 | 1 |
| risky_ps_linear | 0.5391062825733282 | 0.55 | 0.9333333333333333 | 440 | 60 | 1 |
| risky_ps_old | 0.5391062825733282 | 0.55 | 0.9333333333333333 | 440 | 60 | 1 |
| epsilon_exp3 | 0.6557729492399947 | 0.6666666666666666 | 0.95 | 440 | 60 | 1 |
| random_path | 0.7224396159066615 | 0.7333333333333333 | 0.9333333333333333 | 440 | 60 | 1 |

## Delta vs Old Theory-Aligned Partial 4/5 Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| naive_mixed | -0.3434516506178904 | -0.3795084673693252 | 0.97936 |
| direct_multistage_exp3 | 0.3488539079705899 | 0.3127970912191551 | -0.0640533333333334 |
| risky_ps | 0.2570703989756744 | 0.22101358222423956 | 0.02537333333333347 |
| epsilon_exp3 | 0.3904037323090076 | 0.35434691555757286 | 0.04204000000000008 |
| random_path | 0.09289031756051469 | 0.056833500809079895 | 0.13503333333333334 |

## Direct Answers
- risky_ps_old vs risky_ps gap: 0.016667 (old=0.539106, new=0.522440).
- risky_ps vs epsilon_exp3 gap: -0.133333.
- risky_ps vs direct_multistage_exp3 gap: 0.100000.
- PS-family ranking on the prefix-dedup tree: 1. risky_ps (0.522440), 2. risky_ps_direct_cost (0.522440), 3. risky_ps_ix (0.522440), 4. risky_ps_safe_conditional (0.522440), 5. risky_ps_safe_conditional_ix (0.522440), 6. risky_ps_old (0.539106).
- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is smaller: current=-0.016667, old_theory=0.095892.
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
