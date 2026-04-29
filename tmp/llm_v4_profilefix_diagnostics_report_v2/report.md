# LLM profile-switch diagnostic report v2

Date: 2026-04-28

## Question: PS pre/post trap selection ratio

Source run:

`tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini`

This run did not export family route labels at the time, so the ratios below map
`selected_path` node ids back through the current
`shared_basin_strong_prefix_dedup_profile_switch` family spec and use
`episode_index < 25` as pre-switch for `d=4`.

| method | family | pre trap stage1 | post trap stage1 | delta |
| --- | --- | ---: | ---: | ---: |
| `risky_ps_direct_cost` | PS | 0.160 | 0.133 | -0.027 |
| `risky_ps_linear` | PS | 0.160 | 0.187 | +0.027 |
| `risky_ps` | PS | 0.160 | 0.200 | +0.040 |
| `risky_ps_safe_conditional` | PS | 0.160 | 0.240 | +0.080 |
| `risky_ps_safe_conditional_ix` | PS | 0.160 | 0.280 | +0.120 |
| `risky_ps_old` | PS | 0.160 | 0.293 | +0.133 |
| `risky_ps_ix` | PS | 0.160 | 0.373 | +0.213 |
| `naive_mixed` | baseline | 0.040 | 0.040 | +0.000 |
| `naive_mixed_avg` | baseline | 0.040 | 0.067 | +0.027 |
| `direct_multistage_exp3_local` | baseline | 0.160 | 0.187 | +0.027 |
| `random_path` | baseline | 0.200 | 0.227 | +0.027 |
| `epsilon_exp3` | baseline | 0.280 | 0.280 | +0.000 |
| `direct_multistage_exp3` | baseline | 0.160 | 0.307 | +0.147 |

Interpretation: yes, several PS variants also choose trap stage1 more often
after the switch. That is a red flag for the intended profile-switch mechanism:
if trap is supposed to become bad after the switch, a good PS setting should
reduce trap traffic or compensate by moving onto a target/shared suffix quickly.

## Code instrumentation added

Files changed:

- `envs/fixed_tree_env.py`
- `scripts/run_shared_basin_repeated_smoke.py`

Changes:

- Export family route labels, node semantics, deliberation modes, and behavior
  archetype fields into repeated-smoke `episodes.json`.
- Allow the profile-switch family to enter `_prefix_dedup_behavior_context`.
- Add an env-gated diagnostic path-cost ablation:
  `PSAGENT_PROFILE_SWITCH_FLAT_PATH_COST=1`.
- Make flat path cost propagate into `raw_path_cost_component` for LLM executor
  runs.

Verification:

`python -m py_compile envs/fixed_tree_env.py scripts/run_shared_basin_repeated_smoke.py`
passed.

## Layer 1: archetype diagnostic

Run:

`tmp/llm_v4_profilefix_archetype_diagnostic_d4_eta03_eps001`

Setting:

- dataset: low-transfer smoke
- executor: `llm_bench`
- family: `shared_basin_strong_prefix_dedup_profile_switch`
- schedule: `trap_switch`
- `d=4`, `eta=0.3`, `eps=0.01`
- repeats: 2
- methods: `epsilon_exp3`, `risky_ps_linear`, `direct_multistage_exp3`

| method | n | pre | post | all archetypes | post archetypes | pre trap stage1 | post trap stage1 | raw total | post raw total | terminal |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | 20 | 5 | 15 | neutral 16, trap_bad 4 | neutral 11, trap_bad 4 | 0.000 | 0.267 | 17.371 | 21.138 | 12.075 |
| `epsilon_exp3` | 20 | 5 | 15 | neutral 17, trap_good 2, trap_bad 1 | neutral 14, trap_bad 1 | 0.400 | 0.067 | 15.083 | 18.046 | 9.700 |
| `risky_ps_linear` | 20 | 5 | 15 | neutral 17, trap_good 1, trap_bad 2 | neutral 13, trap_bad 2 | 0.200 | 0.133 | 13.813 | 16.522 | 8.675 |

Layer 1 sanity result:

- Good: `trap_like_bad` is active post-switch and has high terminal cost.
- Bad: `target_safe_specialist_good` appears 0 times for all methods.

Observed route labels include many current profile-switch target labels:

- stage4: `target_stage4_repair`
- stage5: `target_stage5_verify`, `target_stage5_decision`

But the current behavior-context recognition still checks older/public labels,
such as `public_stage4_core`, `public_stage4_verify`,
`public_stage5_verify`, and `public_stage5_decision`. It also defines
`safe_prefix` as `node_semantics[0] == "safe_core"`, while the profile-switch
tree uses first-stage semantics such as `general_shared`, `private_barrier`,
and `trap_lane`.

So the target-good path exists in the tree labels, but the diagnostic/cost
archetype logic is not recognizing it.

## Layer 2: flat path cost diagnostic

Invalid first attempt:

`tmp/llm_v4_profilefix_flatcost_diagnostic_d4_eta03_eps001`

This first attempt is not used as evidence because flat costs were applied to
the runtime catalog but not propagated into the LLM executor's
`raw_path_cost_component`.

Valid v2 run:

`tmp/llm_v4_profilefix_flatcost_diagnostic_d4_eta03_eps001_v2`

Setting:

- same as Layer 1
- plus `PSAGENT_PROFILE_SWITCH_FLAT_PATH_COST=1`

| method | n | pre | post | all archetypes | post archetypes | pre trap stage1 | post trap stage1 | raw total | post raw total | terminal | unique path cost |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `direct_multistage_exp3` | 20 | 5 | 15 | neutral 16, trap_bad 4 | neutral 11, trap_bad 4 | 0.000 | 0.267 | 14.700 | 17.350 | 9.400 | 1 |
| `epsilon_exp3` | 20 | 5 | 15 | neutral 16, trap_good 2, trap_bad 2 | neutral 13, trap_bad 2 | 0.400 | 0.133 | 15.202 | 18.214 | 9.925 | 1 |
| `risky_ps_linear` | 20 | 5 | 15 | neutral 16, trap_good 1, trap_bad 3 | neutral 12, trap_bad 3 | 0.200 | 0.200 | 14.775 | 17.662 | 9.350 | 1 |

Flat-cost sanity result:

- `raw_path_cost_component` has exactly one unique value for every method:
  `0.069566`.
- So path base-cost differences are no longer a confound in Layer 2 v2.
- Despite that, `target_safe_specialist_good` still appears 0 times.

Layer 2 conclusion:

Flattening path cost does not solve the problem. The remaining unreasonable
issue is target-good recognition/reward activation, not private/share base-cost
differences.

## Layer 3 status

Layer 3 was not run.

Reason: Layer 1 and Layer 2 both failed the required sanity check that
target-good paths should appear under the post-switch target phase. Running the
larger pilot or 13-method smoke now would waste budget and produce a ranking
that mixes algorithm behavior with a broken target-good mechanism.

## Diagnosis

The current tree labels are profile-switch labels:

- `trap_stage1_intake`, `trap_stage2_router`, `trap_stage3_network`,
  `trap_stage4_execute`, `trap_stage5_terminal`
- `target_stage1_handoff`, `target_stage2_router`, `target_stage3_apn`,
  `target_stage3_roaming`, `target_stage4_repair`,
  `target_stage5_verify`, `target_stage5_decision`
- `general_stage*`
- `barrier_stage*`

But `_prefix_dedup_behavior_context` still recognizes target-good using older
labels from the non-profile-switch tree. As a result:

- post-switch trap-bad can be detected and penalized;
- post-switch target-good is never detected;
- PS cannot receive the intended advantage for reaching the target/shared good
  subtree;
- ranking can favor direct/epsilon methods even when trap is active, because the
  positive target signal is missing.

## Next fix

Update `_prefix_dedup_behavior_context` for
`shared_basin_strong_prefix_dedup_profile_switch`:

- Treat profile-switch target/general shared prefixes as safe enough for
  target-good recognition.
- Recognize `target_stage4_repair` plus
  `target_stage5_verify`/`target_stage5_decision` as target-good.
- Optionally treat `general_stage4_repair` or `general_stage4_verify` combined
  with target stage5 as target-safe, but keep exact target-good stricter.
- Keep `trap_like_bad` unchanged so post-switch trap penalty remains active.

Recommended rerun sequence after that fix:

1. Rerun Layer 1 tiny diagnostic.
2. Continue only if both conditions hold:
   - post-switch `trap_like_bad` appears and is costly;
   - post-switch `target_safe_specialist_good` appears and is cheaper.
3. Rerun Layer 2 flat-cost diagnostic.
4. Only then run Layer 3 pilot / full smoke.

