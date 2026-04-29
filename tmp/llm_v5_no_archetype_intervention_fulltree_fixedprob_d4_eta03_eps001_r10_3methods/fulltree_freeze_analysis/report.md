# Full-tree Fixed Probability LLM Sanity Report

- Run dir: `tmp/llm_v5_no_archetype_intervention_fulltree_fixedprob_d4_eta03_eps001_r10_3methods`
- Experiment name: `llm_v5_no_archetype_intervention_fulltree_fixedprob_d4_eta03_eps001_r10_3methods`
- Freeze mode: `full_tree_child_marginal_at_switch`
- Model: `gpt-4o-mini`
- Methods: `direct_multistage_exp3, epsilon_exp3, risky_ps_linear`
- Repeats/horizon: `10` / `100`
- Switch episode index: `25` (1-based episode `26`)
- Schedule: 25 pre-switch episodes from the trap bucket, 75 post-switch episodes from the target bucket for the 10x10 run.

## Tree-freeze validation

| method | snapshot_rows | parent_prefix_count | prefix_depths | freeze_mode |
| --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 655 | 216 | 0,1,2,3,4 | full_tree_child_marginal_at_switch |
| epsilon_exp3 | 655 | 216 | 0,1,2,3,4 | full_tree_child_marginal_at_switch |
| risky_ps_linear | 655 | 216 | 0,1,2,3,4 | full_tree_child_marginal_at_switch |

## Total cost by method

| method | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | exact_match_rate | clear_execution_success_rate | assistant_required_count | assistant_execution_success_rate_applicable |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps_linear | 100 | 8.1400 | 5.1666 | 13.3770 | 0.2400 | 0.1100 | 63 | 0.0000 |
| epsilon_exp3 | 100 | 8.5600 | 5.1070 | 13.7388 | 0.2400 | 0.1500 | 63 | 0.0000 |
| direct_multistage_exp3 | 100 | 8.8850 | 5.2336 | 14.1897 | 0.1900 | 0.1300 | 63 | 0.0000 |

## Pre/post switch cost

| method | phase | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | clear_execution_success_rate | assistant_required_count | assistant_execution_success_rate_applicable | actual_majority_distribution | required_majority_distribution |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | post_switch | 75 | 11.3800 | 5.2243 | 16.6758 | 0.0400 | 60 | 0.0000 | {"mostly_deep": 64, "mostly_fast": 11} | {"mostly_deep": 75} |
| direct_multistage_exp3 | pre_switch | 25 | 1.4000 | 5.2615 | 6.7312 | 0.4000 | 3 | 0.0000 | {"mostly_deep": 19, "mostly_fast": 6} | {"mostly_fast": 25} |
| epsilon_exp3 | post_switch | 75 | 10.9733 | 5.0901 | 16.1360 | 0.0667 | 60 | 0.0000 | {"mostly_deep": 66, "mostly_fast": 9} | {"mostly_deep": 75} |
| epsilon_exp3 | pre_switch | 25 | 1.3200 | 5.1577 | 6.5472 | 0.4000 | 3 | 0.0000 | {"mostly_deep": 21, "mostly_fast": 4} | {"mostly_fast": 25} |
| risky_ps_linear | post_switch | 75 | 10.3333 | 5.1604 | 15.5638 | 0.0133 | 60 | 0.0000 | {"mostly_deep": 53, "mostly_fast": 22} | {"mostly_deep": 75} |
| risky_ps_linear | pre_switch | 25 | 1.5600 | 5.1853 | 6.8166 | 0.4000 | 3 | 0.0000 | {"mostly_deep": 22, "mostly_fast": 3} | {"mostly_fast": 25} |

## Path majority vs task requirement

| method | majority_pair | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | clear_execution_success_rate |
| --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | mostly_deep_path__mostly_deep_task | 64 | 11.3281 | 5.2160 | 16.6171 | 0.0469 |
| direct_multistage_exp3 | mostly_deep_path__mostly_fast_task | 19 | 1.6842 | 5.5990 | 7.3535 | 0.3158 |
| direct_multistage_exp3 | mostly_fast_path__mostly_deep_task | 11 | 11.6818 | 5.2723 | 17.0174 | 0.0000 |
| direct_multistage_exp3 | mostly_fast_path__mostly_fast_task | 6 | 0.5000 | 4.1930 | 4.7603 | 0.6667 |
| epsilon_exp3 | mostly_deep_path__mostly_deep_task | 66 | 10.8182 | 5.1419 | 16.0335 | 0.0758 |
| epsilon_exp3 | mostly_deep_path__mostly_fast_task | 21 | 1.3571 | 5.3631 | 6.7904 | 0.4286 |
| epsilon_exp3 | mostly_fast_path__mostly_deep_task | 9 | 12.1111 | 4.7100 | 16.8873 | 0.0000 |
| epsilon_exp3 | mostly_fast_path__mostly_fast_task | 4 | 1.1250 | 4.0794 | 5.2707 | 0.2500 |
| risky_ps_linear | mostly_deep_path__mostly_deep_task | 53 | 11.1981 | 5.1554 | 16.4253 | 0.0000 |
| risky_ps_linear | mostly_deep_path__mostly_fast_task | 22 | 1.3636 | 5.2153 | 6.6513 | 0.4091 |
| risky_ps_linear | mostly_fast_path__mostly_deep_task | 22 | 8.2500 | 5.1723 | 13.4882 | 0.0455 |
| risky_ps_linear | mostly_fast_path__mostly_fast_task | 3 | 3.0000 | 4.9650 | 8.0289 | 0.3333 |

## Stage mismatch buckets

| method | mismatch_bucket | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | clear_execution_success_rate |
| --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | all_stage_modes_match | 20 | 9.7500 | 4.9652 | 14.7886 | 0.0000 |
| direct_multistage_exp3 | both_mismatch_types | 16 | 11.2500 | 5.5577 | 16.8769 | 0.0625 |
| direct_multistage_exp3 | deep_on_fast_required | 39 | 5.3718 | 5.3479 | 10.7920 | 0.3077 |
| direct_multistage_exp3 | fast_on_deep_required | 25 | 12.1600 | 5.0626 | 17.2910 | 0.0000 |
| epsilon_exp3 | all_stage_modes_match | 20 | 9.4250 | 4.8904 | 14.3925 | 0.1000 |
| epsilon_exp3 | both_mismatch_types | 19 | 11.1579 | 5.5053 | 16.7339 | 0.0526 |
| epsilon_exp3 | deep_on_fast_required | 33 | 3.5909 | 5.2327 | 8.8949 | 0.3333 |
| epsilon_exp3 | fast_on_deep_required | 28 | 12.0357 | 4.8432 | 16.9483 | 0.0357 |
| risky_ps_linear | all_stage_modes_match | 20 | 10.6250 | 4.8489 | 15.5468 | 0.0000 |
| risky_ps_linear | both_mismatch_types | 6 | 9.1667 | 5.4833 | 14.7179 | 0.0000 |
| risky_ps_linear | deep_on_fast_required | 28 | 3.4464 | 5.2641 | 8.7824 | 0.3571 |
| risky_ps_linear | fast_on_deep_required | 46 | 9.7826 | 5.2041 | 15.0554 | 0.0217 |

## Terminal action split

| method | phase | expected_terminal_action | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | clear_execution_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | post_switch | repair_all | 15 | 11.3000 | 5.4449 | 16.8189 | 0.2000 |
| direct_multistage_exp3 | post_switch | repair_subset | 53 | 12.6981 | 5.2199 | 17.9888 | 0.0000 |
| direct_multistage_exp3 | post_switch | transfer | 7 | 1.5714 | 4.7851 | 6.4283 | 0.0000 |
| direct_multistage_exp3 | pre_switch | repair_all | 22 | 0.8182 | 5.2855 | 6.1733 | 0.4545 |
| direct_multistage_exp3 | pre_switch | repair_subset | 3 | 5.6667 | 5.0854 | 10.8223 | 0.0000 |
| epsilon_exp3 | post_switch | repair_all | 15 | 8.8333 | 5.2995 | 14.2059 | 0.3333 |
| epsilon_exp3 | post_switch | repair_subset | 53 | 12.8585 | 5.0815 | 18.0123 | 0.0000 |
| epsilon_exp3 | post_switch | transfer | 7 | 1.2857 | 4.7062 | 6.0654 | 0.0000 |
| epsilon_exp3 | pre_switch | repair_all | 22 | 0.8182 | 5.1691 | 6.0560 | 0.4545 |
| epsilon_exp3 | pre_switch | repair_subset | 3 | 5.0000 | 5.0740 | 10.1493 | 0.0000 |
| risky_ps_linear | post_switch | repair_all | 15 | 12.0667 | 5.3893 | 17.5251 | 0.0667 |
| risky_ps_linear | post_switch | repair_subset | 53 | 11.0377 | 5.1365 | 16.2445 | 0.0000 |
| risky_ps_linear | post_switch | transfer | 7 | 1.2857 | 4.8507 | 6.2068 | 0.0000 |
| risky_ps_linear | pre_switch | repair_all | 22 | 0.8182 | 5.2396 | 6.1288 | 0.4545 |
| risky_ps_linear | pre_switch | repair_subset | 3 | 7.0000 | 4.7867 | 11.8606 | 0.0000 |

## Assistant execution status

| method | assistant_execution_status | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | clear_execution_success_rate |
| --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | assistant_required_but_no_tool_or_not_clean | 55 | 10.8909 | 5.1576 | 16.1198 | 0.0000 |
| direct_multistage_exp3 | assistant_tool_called_but_not_clean | 8 | 12.7500 | 5.2172 | 18.0349 | 0.0000 |
| direct_multistage_exp3 | not_required | 37 | 5.0676 | 5.3501 | 10.4890 | 0.3514 |
| epsilon_exp3 | assistant_required_but_no_tool_or_not_clean | 56 | 11.3214 | 5.0290 | 16.4229 | 0.0000 |
| epsilon_exp3 | assistant_tool_called_but_not_clean | 7 | 10.2143 | 5.1231 | 15.4105 | 0.0000 |
| epsilon_exp3 | not_required | 37 | 4.0676 | 5.2220 | 9.3600 | 0.4054 |
| risky_ps_linear | assistant_required_but_no_tool_or_not_clean | 55 | 9.8182 | 5.0143 | 14.9030 | 0.0000 |
| risky_ps_linear | assistant_tool_called_but_not_clean | 8 | 9.3750 | 5.5954 | 15.0402 | 0.0000 |
| risky_ps_linear | not_required | 37 | 5.3784 | 5.3003 | 10.7489 | 0.2973 |

## Interpretation

The full-tree freeze behaves like the sanity check expected. For all three methods,
post-switch total cost is much worse than pre-switch total cost:

| method | pre_total | post_total | delta | post/pre | pre_terminal | post_terminal |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| direct_multistage_exp3 | 6.7312 | 16.6758 | +9.9447 | 2.48x | 1.4000 | 11.3800 |
| epsilon_exp3 | 6.5472 | 16.1360 | +9.5887 | 2.46x | 1.3200 | 10.9733 |
| risky_ps_linear | 6.8166 | 15.5638 | +8.7472 | 2.28x | 1.5600 | 10.3333 |

The degradation is almost entirely terminal-quality driven. Reasoning cost stays
near 5.1 before and after switch, while terminal cost rises by roughly 8.8 to
10.0 points. That supports the intended diagnosis: freezing the pre-switch tree
probabilities prevents the policy from adapting to the post-switch target/deep
task distribution.

Aggregating all methods, path/task mode alignment looks like this:

| majority_pair | episode_count | terminal_cost_avg | reasoning_cost_avg | total_cost_avg | clear_execution_success_rate |
| --- | ---: | ---: | ---: | ---: | ---: |
| mostly_fast_path__mostly_fast_task | 13 | 1.269 | 4.336 | 5.672 | 0.462 |
| mostly_deep_path__mostly_fast_task | 62 | 1.460 | 5.383 | 6.914 | 0.387 |
| mostly_fast_path__mostly_deep_task | 42 | 9.976 | 5.099 | 15.141 | 0.024 |
| mostly_deep_path__mostly_deep_task | 183 | 11.107 | 5.172 | 16.351 | 0.044 |

So the expected "fast path on fast task is cheaper" signal is present. The
"fast path on deep task is expensive" signal is also present. The more subtle
result is that "deep path on fast task" is not very expensive here: terminal
cost remains low because pre-switch fast tasks are easy, and the extra cost is
mostly a small reasoning premium. Conversely, "deep path on deep task" is still
expensive because these post-switch target tasks contain many repair_subset /
assistant-side cases that the execution layer often does not complete cleanly.

The main issue is assistant-side execution quality. Among the 63 assistant-side
episodes per method, the strict assistant execution success rate is 0.0 under the
definition used here. Most assistant-side cases either required assistant-side
action but made no assistant mutating tool call, or called the tool but still did
not finish cleanly. This also explains why post-switch repair_subset remains the
dominant failure pocket.

Recommended next steps:

1. Keep this run as the full-tree freeze sanity check; it shows the intended
   post-switch degradation.
2. For a fair freeze-vs-dynamic claim, run a matching non-frozen 10x10 v5 run.
   The earlier dynamic v5 run is only repeats=2/horizon=20, so it is useful as a
   smoke reference but not an apples-to-apples baseline.
3. Fix or instrument assistant-side repair execution before using clean success
   as the primary quality claim. In particular, export stage-level fallback and
   assistant-side diagnostics in repeated smoke, and inspect why repair_subset
   cases reach exact action match with policy_violation_count=1 and no assistant
   mutating tool call.

## Output files

- `episode_cost_success_mode_analysis.csv`: full per-episode table
- `episode_compact_view.csv`: compact per-episode table
- `summary_cost_success_mode.csv`: total and pre/post summaries
- `majority_pair_cost_summary.csv`: mostly-fast/deep path vs mostly-fast/deep task table
- `mismatch_bucket_cost_summary.csv`: fast-on-deep/deep-on-fast mismatch table
- `stage_mode_pair_cost_summary.csv`: stage-level mode-pair table
- `terminal_action_cost_summary.csv`: repair_all/repair_subset/transfer split
- `assistant_execution_status_summary.csv`: assistant-side status split
