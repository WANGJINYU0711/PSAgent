# Unique-agent full-branching unbound controlled simulation

- tree_spec_role_mode: `spec_or_agent_id`
- num_paths: `440`
- duplicate_agent_count: `0`
- cross_prefix_duplicate_count: `0`

## Current Tree
| method | regret_per_t_mean | average_cost_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
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

## Delta vs Old Unique-Agent Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| naive_mixed | -0.01508032238991966 | -0.05310921231077151 | 0.0 |
| direct_multistage_exp3 | 0.37719382253561534 | 0.33916493261476355 | -0.07753333333333334 |
| risky_ps | 0.41932798215727063 | 0.3812990922364187 | -0.05466666666666653 |
| epsilon_exp3 | 0.5557249775743793 | 0.5176960876535276 | -0.03799999999999992 |
| random_path | 0.24820011273614867 | 0.21017122281529677 | 0.006333333333333413 |

## Delta vs Old Theory-Aligned Partial 4/5 Reference
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| naive_mixed | -0.3434516506178904 | -0.3795084673693252 | 0.97936 |
| direct_multistage_exp3 | 0.3488539079705899 | 0.3127970912191551 | -0.0640533333333334 |
| risky_ps | 0.2570703989756744 | 0.22101358222423956 | 0.02537333333333347 |
| epsilon_exp3 | 0.3904037323090076 | 0.35434691555757286 | 0.04204000000000008 |
| random_path | 0.09289031756051469 | 0.056833500809079895 | 0.13503333333333334 |

## Direct Answers
- Current risky_ps vs epsilon_exp3 gap: -0.133333. Old unique-agent gap: 0.003064. Old theory-aligned gap: 0.000000.
- Current risky_ps vs direct_multistage_exp3 gap: 0.100000. Old unique-agent gap: 0.057866. Old theory-aligned gap: 0.191784.
- Relative to old theory-aligned partial_4of5, the average risky_ps gap to epsilon/direct is smaller: current=-0.016667, old_theory=0.095892.
- Relative to the older bound unique-agent tree, the average risky_ps gap to epsilon/direct is smaller: current=-0.016667, old_unique=0.030465.
- PS-family ranking on this truly-unbound tree: 1. risky_ps (0.522440), 2. risky_ps_direct_cost (0.522440), 3. risky_ps_ix (0.522440), 4. risky_ps_safe_conditional (0.522440), 5. risky_ps_safe_conditional_ix (0.522440).
- naive_mixed rank is 1 with regret/T=0.005773.
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
