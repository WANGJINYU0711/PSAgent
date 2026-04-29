# Layer 1 Profile-Switch Clean Target-Route Diagnostic

Date: 2026-04-28

Experiment name:

`llm_v4_profilefix_layer1_clean_targetroute_d4_eta03_eps001_r2_3methods`

Output directory:

`tmp/llm_v4_profilefix_layer1_clean_targetroute_d4_eta03_eps001_r2_3methods`

## Code change

File:

`envs/fixed_tree_env.py`

Change:

- Keep post-switch `trap_like_bad` recognition unchanged.
- For `shared_basin_strong_prefix_dedup_profile_switch`, recognize clean
  profile-switch target routes using current labels instead of old
  `public_stage*` labels.
- `target_safe_specialist_good` now requires:
  - stage1 in `general_stage1_intake`, `general_stage1_verify`, or
    `target_stage1_handoff`;
  - no `trap_*` labels;
  - no `barrier_*` labels;
  - stage4 is `target_stage4_repair`;
  - stage5 is `target_stage5_verify` or `target_stage5_decision`;
  - target phase and specialist task.
- `target_safe_majority_bad` covers clean general stage4 repair/verify plus a
  target stage5 suffix, but is kept separate from exact target-good.

Verification:

`python -m py_compile envs/fixed_tree_env.py scripts/run_shared_basin_repeated_smoke.py`
passed.

## Run setting

| field | value |
| --- | --- |
| dataset | `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json` |
| buckets | `analysis/shared_basin_prefix_dedup_profile_switch_lowtransfer_smoke_schedule_buckets.json` |
| model | `gpt-4o-mini` |
| executor | `llm_bench` |
| family | `shared_basin_strong_prefix_dedup_profile_switch` |
| schedule | `trap_switch` |
| switch denominator | `d=4` |
| eta | `0.3` |
| epsilon | `0.01` |
| repeats | `2` |
| horizon per method | `20` |
| switch episode | `5` |
| pre/post episodes | `5 / 15` |
| methods | `direct_multistage_exp3`, `epsilon_exp3`, `risky_ps_linear` |

## Main sanity table

| method | n | pre | post | all archetypes | post archetypes | pre trap stage1 | post trap stage1 | raw total | post raw total | terminal | post terminal | exact | post exact |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | 20 | 5 | 15 | neutral 9, decoy 4, target_good 1, trap_bad 4, target_majority_bad 2 | decoy 4, neutral 4, target_good 1, trap_bad 4, target_majority_bad 2 | 0.000 | 0.267 | 12.758 | 14.773 | 7.200 | 9.000 | 0.250 | 0.200 |
| `epsilon_exp3` | 20 | 5 | 15 | trap_good 2, neutral 9, trap_safe 1, target_good 3, target_majority_bad 2, trap_bad 2, decoy 1 | target_good 3, target_majority_bad 2, neutral 7, trap_bad 2, decoy 1 | 0.400 | 0.133 | 14.150 | 16.756 | 8.800 | 11.200 | 0.350 | 0.267 |
| `risky_ps_linear` | 20 | 5 | 15 | trap_good 1, neutral 11, target_good 4, trap_bad 2, decoy 1, target_majority_bad 1 | target_good 4, neutral 7, trap_bad 2, decoy 1, target_majority_bad 1 | 0.200 | 0.133 | 13.441 | 15.695 | 8.150 | 10.200 | 0.350 | 0.267 |

## Archetype cost table

| method | archetype | n | raw total | terminal | exact | final actions |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `direct_multistage_exp3` | neutral | 9 | 9.955 | 4.667 | 0.333 | repair_all 4, repair_subset 5 |
| `direct_multistage_exp3` | target_decoy_medium | 4 | 9.107 | 3.500 | 0.000 | repair_subset 4 |
| `direct_multistage_exp3` | target_safe_majority_bad | 2 | 11.153 | 5.000 | 0.500 | repair_subset 1, transfer 1 |
| `direct_multistage_exp3` | target_safe_specialist_good | 1 | 6.267 | 2.000 | 1.000 | repair_subset 1 |
| `direct_multistage_exp3` | trap_like_bad | 4 | 25.138 | 19.000 | 0.000 | transfer 4 |
| `epsilon_exp3` | neutral | 9 | 17.922 | 12.667 | 0.111 | repair_all 2, transfer 4, repair_subset 3 |
| `epsilon_exp3` | target_decoy_medium | 1 | 7.187 | 2.000 | 1.000 | transfer 1 |
| `epsilon_exp3` | target_safe_majority_bad | 2 | 16.222 | 10.000 | 0.000 | repair_subset 2 |
| `epsilon_exp3` | target_safe_specialist_good | 3 | 5.730 | 0.667 | 1.000 | repair_subset 1, repair_all 2 |
| `epsilon_exp3` | trap_like_bad | 2 | 23.775 | 17.500 | 0.000 | transfer 2 |
| `epsilon_exp3` | trap_like_good | 2 | 4.363 | 0.000 | 1.000 | repair_all 2 |
| `epsilon_exp3` | trap_safe_overcautious | 1 | 8.613 | 3.000 | 0.000 | repair_subset 1 |
| `risky_ps_linear` | neutral | 11 | 14.518 | 9.455 | 0.182 | repair_all 3, repair_subset 3, transfer 5 |
| `risky_ps_linear` | target_decoy_medium | 1 | 11.270 | 4.000 | 0.000 | repair_subset 1 |
| `risky_ps_linear` | target_safe_majority_bad | 1 | 15.389 | 10.000 | 0.000 | repair_subset 1 |
| `risky_ps_linear` | target_safe_specialist_good | 4 | 7.187 | 2.000 | 1.000 | repair_subset 4 |
| `risky_ps_linear` | trap_like_bad | 2 | 24.532 | 18.500 | 0.000 | transfer 2 |
| `risky_ps_linear` | trap_like_good | 1 | 4.646 | 0.000 | 1.000 | repair_all 1 |

## Sanity check conclusion

Passed.

The two required checks now both hold:

1. Post-switch `trap_like_bad` appears and is expensive.
   - `direct_multistage_exp3`: raw total `25.138`, terminal `19.000`.
   - `epsilon_exp3`: raw total `23.775`, terminal `17.500`.
   - `risky_ps_linear`: raw total `24.532`, terminal `18.500`.

2. Post-switch `target_safe_specialist_good` appears and is cheap.
   - `direct_multistage_exp3`: raw total `6.267`, terminal `2.000`, exact `1.000`.
   - `epsilon_exp3`: raw total `5.730`, terminal `0.667`, exact `1.000`.
   - `risky_ps_linear`: raw total `7.187`, terminal `2.000`, exact `1.000`.

This fixes the previous unreasonable issue where `target_safe_specialist_good`
was always zero.

## Interpretation

The profile-switch environment is now qualitatively aligned with the intended
mechanism:

- trap can be good before the switch;
- trap becomes bad after the switch;
- clean target-specialist routes are recognized after the switch;
- clean target-specialist routes are much cheaper than post-switch trap routes.

This run should not be used as the final algorithm ranking because it is only a
3-method, 2-repeat tiny diagnostic. In this tiny sample, `direct_multistage_exp3`
has the lowest post raw total (`14.773`), while `risky_ps_linear` has more
target-good hits (`4` vs direct's `1`) and fewer trap-bad hits (`2` vs direct's
`4`). That is a useful mechanism signal, not yet a final performance claim.

## Remaining issues

1. `target_decoy_medium` can sometimes be relatively cheap.
   This is especially visible for `epsilon_exp3`, where one decoy episode has
   raw total `7.187` and exact match `1.000`. That episode is a transfer-oracle
   case, so it is not necessarily a bug, but decoy costs should be split by
   oracle action in the next report.

2. `target_safe_specialist_good` exact match is `1.000`, but terminal penalty is
   not always zero.
   This suggests some residual terminal/policy components remain even when the
   final action matches. This is acceptable for the first-layer sanity check,
   but the next layer should report terminal breakdowns.

3. The tiny diagnostic is too small for ranking.
   It verifies the landscape. It does not prove PS wins.

## Recommended next step

Proceed to Layer 2 flat-cost diagnostic using the same cleaned target-route
recognition:

- same 3 methods;
- same `d=4`, `eta=0.3`, `eps=0.01`;
- add `PSAGENT_PROFILE_SWITCH_FLAT_PATH_COST=1`;
- verify that target-good remains cheap after removing path base-cost
  differences.

If Layer 2 also passes, then run the Layer 3 pilot/full sweep.

