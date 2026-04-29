# Fixed Layer-1 Post-Switch Probability Sanity

Experiment name: `llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods_fixed_layer1_prob_sanity`

## Config

| field | value |
| --- | --- |
| source_llm_experiment | llm_v5_no_archetype_intervention_layer1_d4_eta03_eps001_r2_3methods |
| tree_spec | /home/ubuntu/data/PSAgent/analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch.json |
| tree_spec_cost_mode | ps_favored_trap_v10_avg_baited |
| horizon | 20 |
| seeds | [0, 1] |
| switch_denominator | 4 |
| switch_episode_1based | 6 |
| methods | {'eps': 'epsilon_exp3', 'exp': 'direct_multistage_exp3', 'ps': 'risky_ps_linear'} |
| eta | 0.3000 |
| epsilon | 0.0100 |

## Cost Summary

| method_short | regime | seeds | pre_cumulative_cost_avg_mean | post_cost_avg_mean | total_cost_avg_mean | post_regret_avg_mean | post_target_good_fraction_mean | post_target_bad_fraction_mean | post_trap_basin_fraction_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| eps | dynamic | 2 | 0.2000 | 0.7000 | 0.5750 | 0.6900 | 0.0333 | 0.1667 | 0.0667 |
| eps | freeze_layer1_post_switch | 2 | 0.2000 | 0.7333 | 0.6000 | 0.7233 | 0.0333 | 0.2333 | 0.0667 |
| exp | dynamic | 2 | 0.7000 | 0.7333 | 0.7250 | 0.7233 | 0.1333 | 0.2667 | 0.0333 |
| exp | freeze_layer1_post_switch | 2 | 0.7000 | 0.8333 | 0.8000 | 0.8233 | 0.0667 | 0.2000 | 0.1000 |
| ps | dynamic | 2 | 0.7000 | 0.6333 | 0.6500 | 0.6233 | 0.1667 | 0.1000 | 0.1333 |
| ps | freeze_layer1_post_switch | 2 | 0.7000 | 0.7333 | 0.7250 | 0.7233 | 0.1000 | 0.1667 | 0.0667 |

## Freeze Minus Dynamic

| method_short | post_cost_avg_delta | total_cost_avg_delta | post_target_good_fraction_delta | post_target_bad_fraction_delta | post_trap_basin_fraction_delta |
| --- | --- | --- | --- | --- | --- |
| eps | 0.0333 | 0.0250 | 0.0000 | 0.0667 | 0.0000 |
| exp | 0.1000 | 0.0750 | -0.0667 | -0.0667 | 0.0667 |
| ps | 0.1000 | 0.0750 | -0.0667 | 0.0667 | -0.0667 |

## Switch-Time Layer-1 Probabilities

| method_short | regime | seed | base_alias | prob | distribution_kind |
| --- | --- | --- | --- | --- | --- |
| eps | freeze_layer1_post_switch | 0 | stage1_n1 | 0.2364 | stagewise_marginal_mixture |
| eps | freeze_layer1_post_switch | 0 | stage1_n2 | 0.2364 | stagewise_marginal_mixture |
| eps | freeze_layer1_post_switch | 0 | stage1_n3 | 0.2364 | stagewise_marginal_mixture |
| eps | freeze_layer1_post_switch | 0 | stage1_n4 | 0.2364 | stagewise_marginal_mixture |
| eps | freeze_layer1_post_switch | 0 | stage1_n5 | 0.0543 | stagewise_marginal_mixture |
| exp | freeze_layer1_post_switch | 0 | stage1_n1 | 0.1224 | stagewise_marginal_mixture |
| exp | freeze_layer1_post_switch | 0 | stage1_n2 | 0.3504 | stagewise_marginal_mixture |
| exp | freeze_layer1_post_switch | 0 | stage1_n3 | 0.3504 | stagewise_marginal_mixture |
| exp | freeze_layer1_post_switch | 0 | stage1_n4 | 0.0987 | stagewise_marginal_mixture |
| exp | freeze_layer1_post_switch | 0 | stage1_n5 | 0.0782 | stagewise_marginal_mixture |
| ps | freeze_layer1_post_switch | 0 | stage1_n1 | 0.3489 | ps_risky_marginal_mixture |
| ps | freeze_layer1_post_switch | 0 | stage1_n2 | 0.1232 | ps_risky_marginal_mixture |
| ps | freeze_layer1_post_switch | 0 | stage1_n3 | 0.3489 | ps_risky_marginal_mixture |
| ps | freeze_layer1_post_switch | 0 | stage1_n4 | 0.0794 | ps_risky_marginal_mixture |
| ps | freeze_layer1_post_switch | 0 | stage1_n5 | 0.0997 | ps_risky_marginal_mixture |

## Interpretation

- `dynamic` is the unmodified policy behavior in the same controlled simulator.
- `freeze_layer1_post_switch` snapshots the root/direct-child marginal distribution at the switch boundary and reuses that distribution for every post-switch root choice.
- Downstream stages and all policy updates remain active, so this isolates whether continued layer-1 adaptation after the switch is helping or hurting.
- A negative `post_cost_avg_delta` means freezing the layer-1 distribution improved post-switch cost; a positive delta means ongoing layer-1 adaptation helped.

## Findings

- Freezing the layer-1 distribution did not improve this sanity run. `freeze_minus_dynamic` is positive for all three algorithms on post-switch cost: `eps +0.0333`, `exp +0.1000`, `ps +0.1000`.
- The pre-switch cumulative average is identical between dynamic and frozen regimes for each method, which is expected because the intervention starts only at the switch boundary.
- `ps` is the best dynamic method in this controlled run by post-switch cost (`0.6333`), but falls back to `0.7333` when layer-1 probabilities are frozen. That suggests post-switch layer-1 adaptation is currently useful rather than harmful for PS in this toy landscape.
- The switch-time layer-1 probabilities are already non-uniform. For seed 0, `ps` places most mass on `stage1_n1` and `stage1_n3` (`0.3489` each), while `exp` places most mass on `stage1_n2` and `stage1_n3` (`0.3504` each), and `eps` mostly suppresses `stage1_n5`.

## Caveats And Fixes

- This is intentionally a tiny sanity check matching the source v5 `r2/h20` shape. With only 5 pre-switch and 15 post-switch episodes per seed, Bernoulli terminal noise is large; treat signs as diagnostic, not final evidence.
- The run uses the controlled profile-switch tree with `ps_favored_trap_v10_avg_baited` costs and does not call the LLM executor. It reuses the v5 policy settings, horizon, seeds, switch denominator, and three methods, but it is not a replacement for a full LLM repeated smoke.
- If we want a cleaner causal read, rerun this exact script with more seeds and horizon, for example `--horizon 100 --seeds 0 1 2 3 4 5 6 7 8 9`, or add an expected-cost/no-Bernoulli mode so selection effects are not buried by binary observation noise.
- If we want to test the real LLM track, add the same root-probability freeze hook to `run_shared_basin_repeated_smoke.py` and export per-episode root distribution, selected root child, clean success, hard-transfer guard, and completion diagnostics.
