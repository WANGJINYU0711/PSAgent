# Telecom MMS C-Config Seed Confirmatory Status (2026-04-30)

This note is the current handoff summary for the telecom MMS profile-switch /
shared-basin LLM smoke line.

It captures the main conclusions from the recent C-config runs, the seed0 vs
seed1 confirmatory check, the PS-family scan, and the early-stop diagnostic.

## 1. Current status in one screen

- The current best objective candidate is still the C config:
  - `terminalv4`
  - `reasoning calibration v3`
  - `report-only modecost`
  - `switch_denominator=4`
  - `eta=0.3`
  - `epsilon=0.01`
- This config looks behaviorally sane:
  - post/deep `fast-on-deep` is clearly more expensive than `deep-on-deep`
  - terminal share stays around the desired `6:4` neighborhood instead of being
    swallowed by reasoning penalties
- But "PS must be first" is **not** stable enough yet:
  - seed0 can make `risky_ps` first
  - seed1 confirmatory can make `risky_ps` third
- Therefore:
  - **do not start the full formal LLM run yet**
  - keep C as the main candidate config
  - first diagnose why `risky_ps` is seed-sensitive in post/deep terminal
    quality

## 2. C config definition

Primary dataset and schedule used in this line:

- dataset:
  `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- buckets:
  `analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- executor:
  `llm_bench`
- model:
  `gpt-4o-mini`
- horizon / repeats:
  `100 / 10`
- schedule:
  `trap_switch`, `switch_denominator=4`, switch at episode `25`
- methods in the main head-to-head comparisons:
  `risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`

C cost setup:

- `terminalv4`
- reasoning weight calibration v3:
  - mode match multiplier reduced
  - `deep-required + actual fast` amplified
  - `fast-required + actual deep` moderately amplified
- report-only modecost:
  - mismatch cost is exported for diagnosis
  - mismatch cost does **not** enter `raw_total_cost`

Important invariant:

- `raw_total_cost = terminal + calibrated reasoning + path`
- report-modecost is only a reporting field

## 3. Main runs and artifacts

### Seed0 baseline C 3-method run

- dir:
  `tmp/llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/`
- result:
  - all split winner: `direct_multistage_exp3`
  - post split winner: `direct_multistage_exp3`

### Seed0 PS-family run under the same C config

- dir:
  `tmp/llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015/`
- methods:
  - `risky_ps_old`
  - `risky_ps`
  - `risky_ps_linear`
  - `risky_ps_ix`
  - `risky_ps_safe_conditional`
  - `risky_ps_safe_conditional_ix`
  - `risky_ps_direct_cost`
- result:
  - winner: `risky_ps`
  - `risky_ps_linear` with `eta_shared=0.15` was clearly worse

### Seed1 confirmatory run

- dir:
  `tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/`
- same config as C + `d=4`, only seed changed
- methods:
  - `risky_ps`
  - `direct_multistage_exp3`
  - `epsilon_exp3`
- result:
  - all split winner: `epsilon_exp3`
  - post split winner: `epsilon_exp3`
  - `risky_ps` fell to third

### Seed0 vs seed1 comparison report

- compare dir:
  `tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_compare_with_c_seed0/`
- cost-curve / early-stop dir:
  `tmp/llm_v8_seed0_seed1_cost_curves_cconfig_d4_eta03_eps001/`

## 4. Key scoreboard

### 4.1 Seed0 / seed1 headline comparison

| run | seed | method | all total | post total | note |
|---|---:|---|---:|---:|---|
| PS-family C | 0 | `risky_ps` | 9.01 | 10.37 | seed0 winner |
| C 3-method | 0 | `direct_multistage_exp3` | 9.53 | 10.80 | seed0 direct winner in the original 3-method C run |
| confirmatory | 1 | `epsilon_exp3` | 9.61 | 11.11 | seed1 winner |
| confirmatory | 1 | `direct_multistage_exp3` | 9.90 | 11.15 | seed1 second |
| confirmatory | 1 | `risky_ps` | 10.50 | 12.44 | seed1 third |

Main conclusion:

- the same C configuration can support a PS win in one seed
- but that win is not robust enough yet

### 4.2 Why `risky_ps_linear eta_shared=0.15` is not the right mainline

In the PS-family run:

- `risky_ps` all total: `9.01`
- `risky_ps_linear` all total: `11.19`
- `risky_ps` post total: `10.37`
- `risky_ps_linear` post total: `13.34`

The degradation is mostly terminal:

- `risky_ps_linear` post terminal: `8.86`
- `risky_ps` post terminal: `6.00`

Interpretation:

- that linear setting did not make PS better at exploiting shared suffix
- it mainly hurt target/deep local terminal quality

## 5. What the seed1 failure actually says

The important point is that seed1 does **not** mainly look like a route-choice
collapse.

`risky_ps` still chose many deep post-switch routes:

- seed0 `risky_ps` post `deep-on-deep`: `70 / 75`
- seed1 `risky_ps` post `deep-on-deep`: `61 / 75`

But seed1 `risky_ps` had much worse deep-on-deep terminal quality:

- seed0 `risky_ps` post `deep-on-deep` terminal: `5.63`
- seed1 `risky_ps` post `deep-on-deep` terminal: `7.36`

And its bad mismatch cases were very bad:

- seed1 `risky_ps` post `fast-on-deep` terminal: `10.68`
- seed1 `risky_ps` post `fast-on-deep` total: `15.81`

Interpretation:

- the main remaining problem is likely PS update / terminal instability
- not just "PS failed because it did not go deep enough"

This matters because the next fix should not start by heavily changing the C
cost structure again.

## 6. Why C is still the best cost candidate

C remains the most balanced objective so far.

Relative to earlier versions:

- A (`terminalv4`) fixed the obvious under-penalty issue for failing local
  repair cases
- B (modecost in main total) pushed mismatches harder but made reasoning/mode
  penalty too dominant
- C kept the good terminal behavior while making mismatch cases visibly more
  expensive, without making reasoning dominate

Operationally, C achieved the intended diagnostic pattern:

- `deep-on-deep` on deep-required tasks is cheaper
- `fast-on-deep` is clearly more expensive
- mismatch cost remains a report field instead of silently redefining the main
  objective

So the current blocker is not "find another cost setup first"; it is "make
PS-first stable enough under the current sane setup."

## 7. Shared-basin structure check

There was a concern that the current tree or target basin might no longer have
enough shared structure for PS to benefit.

The available evidence says the tree is still strongly share-biased.

### Static family evidence

From `analysis/shared_basin_strong_static_analysis.json`:

- `shared_leaf_fraction = 0.8818`
- `top1_shared_fraction = 0.7`
- `top5_shared_majority_fraction = 0.99`
- `mean_top5_shared_fraction = 0.724`

Interpretation:

- most leaves are shared leaves
- for almost all tasks, the top-5 candidate region is majority shared

### LLM smoke evidence

From current specialist/target-heavy summaries:

- seed0 C direct specialist shared-path fraction: `1.0`
- seed0 C `risky_ps` specialist shared-path fraction: `0.9375`

Interpretation:

- current target/deep-heavy slices are still overwhelmingly using shared paths
- PS underperformance is unlikely to be explained simply by "there is no shared
  suffix to exploit anymore"

## 8. Early-stop diagnostic

Artifacts:

- `tmp/llm_v8_seed0_seed1_cost_curves_cconfig_d4_eta03_eps001/`

Headline conclusion:

- the algorithm can recover after being behind
- an early leader can also collapse later
- so hard early stopping is not justified in this LLM smoke before enough
  post-switch evidence accumulates

Checkpoint summary:

| seed | t=25 | t=50 | t=75 | t=90 | t=100 |
|---|---|---|---|---|---|
| seed0 | `risky_ps` leads and stays final winner | still correct | correct | correct | correct |
| seed1 | `risky_ps` leads but is wrong | `risky_ps` still leads but is wrong | first reliable checkpoint | `direct` briefly leads but is wrong | `epsilon` final winner |

Practical interpretation:

- `t=25` cannot be used for stopping
- `t=50` is still unreliable
- `t=75` was the first checkpoint that matched the final winner for both tested
  seeds
- `t=90` can still be noisy if margins are tiny

Suggested operational rule:

- treat early stop as an early warning only
- avoid hard stop unless all of the following hold:
  - `t >= 75`
  - leader stable for at least 10-15 episodes
  - margin clearly above noise, for example `> 0.3` or `> 0.5`
  - rolling trend does not reverse

## 9. Legacy `specialist` naming

The user questioned whether there is still a meaningful separate `specialist`
agent concept.

Current state:

- legacy `specialist_*` fields and summaries still exist in analysis and export
  code
- examples still appear in:
  - `scripts/analyze_llm_repeated_smoke_modes.py`
  - `scripts/run_shared_basin_mechanism_repeated_smoke.py`
  - older notes and many `tmp/` exports
- this was **not** cleaned up in this session

Practical interpretation for future Codex sessions:

- do not assume the current agent semantics still center on a separate
  specialist axis
- the meaningful current task/profile axes are closer to:
  - share vs unshared
  - fast vs deep
- `specialist` is now mostly a legacy diagnostic label unless/until the schema
  is explicitly cleaned

## 10. Seed control change

`scripts/run_shared_basin_repeated_smoke.py` now supports environment-level seed
override:

- `PSAGENT_REPEATED_SMOKE_SEED`

Implementation pattern:

```bash
PSAGENT_REPEATED_SMOKE_SEED=1 python scripts/run_shared_basin_repeated_smoke.py ...
```

This was added specifically to make confirmatory smoke reruns easier without
duplicating the runner.

## 11. Recommended next steps

Do these before any full formal run:

1. Diagnose seed1 `risky_ps` failure episode-wise.
   - focus on post-switch episodes
   - isolate high-terminal episodes
   - compare seed0 vs seed1 for the same task or the same scheduled slot

2. Check whether failures concentrate in a narrow subtype.
   - `repair_subset`
   - transfer fallback
   - a few repeated local blocker families
   - exact/aux failure despite deep-on-deep routing

3. If the problem looks like PS update variance, try stabilization before
   changing the objective again.
   Candidate directions:
   - smaller shared update aggressiveness
   - shared loss clipping
   - probability floor stabilization

4. Keep C + `d=4` fixed while diagnosing.
   - do not simultaneously change schedule ratio and objective
   - do not interpret a schedule change as proof that PS is finally stable

## 12. Bottom line

The session-level bottom line is:

- C is the right cost/objective candidate to keep
- the tree still has strong shared structure
- `risky_ps` can win
- but `risky_ps` is not yet robust enough across seeds
- early-stop before late post-switch evidence is not trustworthy
- the next job is targeted PS instability diagnosis, not another broad cost
  redesign

## 13. Update: fixed-trace Stage 4/5 diagnostic changed the immediate next step

After the seed1 confirmatory diagnosis, we ran a more targeted fixed-profile /
fixed-path trace study instead of going straight into another confirmatory
smoke.

Why this mattered:

- exact mode-match cases such as `fdddd on fdddd` were still sometimes getting
  high terminal penalty
- that means the main problem is not simply deep/fast ratio or PS update shape
- PS learns from terminal outcome, not from the mode label; if an exact-match
  deep path is noisy, PS will be pushed away from the correct profile

Main targeted diagnostic artifacts:

- seed1 old vs probfloor targeted compare:
  `tmp/llm_v8_seed1_old_vs_probfloor_targeted_diagnostic/report.md`
- fixed-profile / fixed-path trace pilot v2:
  `tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2/`
- pilot v2 write-up:
  `tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2/analysis_report_zh.md`

Focus datasets and patterns:

- datasets: `2, 10, 13, 16`
- patterns: `fdddd`, `ffddd`, `ddddd`

The pilot conclusion was important:

- do **not** delete tasks yet
- do **not** enter smoke yet
- the main exposed issue is Stage 4/5 contract instability on local repair
  chains
- no task was proven to be an obvious benchmark/oracle contradiction

Pilot classification:

- `dataset 10`: keep-main + targeted diagnostic
- `dataset 16`: keep-main + targeted diagnostic
- `dataset 2`: diagnostic-only for the current exact-path analysis
- `dataset 13`: diagnostic-only until repeats are cleaner
- `exclude-from-smoke`: empty for now

Observed failure patterns from the pilot:

- `dataset 10` is **not** "always transfer"; the real issue is
  selected/deferred boundary instability in `repair_subset`
- `dataset 2/16` high penalty usually comes from missing upstream local
  prerequisites such as `airplane_mode_on` / `unseat_sim_card`
- completion pass was usually not firing, so the bottleneck looked more like
  Stage 4 raw planning/contract than late Stage 5 cleanup

## 14. User constraint for the fix direction

The user explicitly narrowed the allowed change surface for this line:

- prefer LLM-layer / prompt changes whenever possible
- normalizer may inspect, expose, and validate, but should **not** directly
  rewrite `selected` / `deferred` / `final_action` as a new behavior
- diagnostic fields are allowed only as **report-only**
- do not add decision-changing retry / fix-up logic in this iteration
- if a future change would alter terminal decision behavior more directly, bring
  it back for review first

This constraint matters for future Codex sessions:

- do not "quietly fix" Stage 4/5 by adding normalizer-side auto-correction
- treat prompt design as the first-line intervention

## 15. Prompt-only Stage 4/5 contract v1

We implemented a prompt-only contract pass under the same C config rather than
changing PS, terminal semantics, or dataset labels.

Config label:

- `llm_v8_stage45_contract_promptv1_promptonly_cconfig`

Code / export changes:

- executor env flag:
  `PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1=1`
- modified:
  - `envs/executors/telecom_llm_bench_executor.py`
  - `scripts/run_llm_path_sweep_diagnostic.py`
- helper script already present in this line:
  - `scripts/run_llm_fixed_profile_trace_diagnostic.py`

Prompt-only v1 added:

- Stage 4 prompt:
  - selected/deferred contract
  - prerequisite closure wording
  - `repair_subset != transfer`
  - local / ordinary defer / hard transfer framing
  - abstract few-shot style examples
- Stage 5 prompt:
  - preserve the Stage 4 plan by default
  - treat `repair_subset` as `partial_resolution_only`
  - do not over-transfer without explicit hard reason
- report-only fields:
  - `stage4_contract_prompt_version`
  - `stage4_contract_self_check`
  - `stage5_contract_prompt_version`
  - `stage5_contract_self_check`

Important:

- self-check fields are for diagnosis only
- they do not influence executor decisions
- there is still **no** normalizer-side auto-correction of the LLM answer

Compile sanity check passed:

```bash
python -m py_compile \
  envs/executors/telecom_llm_bench_executor.py \
  scripts/run_llm_path_sweep_diagnostic.py \
  scripts/run_llm_fixed_profile_trace_diagnostic.py
```

## 16. Prompt-only v1 regression runs

Artifacts:

- comparison report:
  `tmp/llm_v8_stage45_contract_promptv1_promptonly_compare/report_zh.md`
- fixed trace `fdddd`, repeat `5`:
  `tmp/llm_v8_stage45_contract_promptv1_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig/`
- fixed trace `fdddd/ffddd/ddddd`, repeat `3`:
  `tmp/llm_v8_stage45_contract_promptv1_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig/`

Headline outcome:

- prompt-only v1 was directionally successful
- it improved several exact-path failures without adding hidden behavior changes
- but it did **not** fully solve dataset `10`, dataset `2` permission closure,
  or fast-heavy `ffddd` failures

Per-dataset readout:

- `dataset 16` improved the most:
  - `fdddd` repeat-5 became `5/5` terminal `0.0`, all `repair_all`
  - expanded `fdddd/ffddd/ddddd` all reached terminal `0.0`
  - interpretation: prompt contract was enough to stabilize the full local
    prerequisite chain here
- `dataset 13` improved strongly on `fdddd` and `ddddd`:
  - `fdddd` repeat-5 became stable `repair_subset` with terminal `6.0`
  - `ddddd` also stabilized at terminal `6.0`
  - but `ffddd` remained bad, usually from missing upstream local blockers
- `dataset 2` improved partially:
  - `fdddd` repeat-5 mean terminal dropped to `4.8`
  - transfer mostly disappeared in repeat-5
  - remaining issue: `break_app_storage_permission` still gets deferred too
    often, causing replay to miss `grant_app_permission`
- `dataset 10` improved partially and remains the main blocker:
  - no longer "always transfer"
  - correct cases now end in `repair_subset` with deferred
    `data_usage_exceeded` / roaming-policy blockers and terminal `6`
  - remaining failure:
    - ordinary-defer blockers are sometimes incorrectly selected into
      `repair_all`, giving terminal around `10`
    - selected/deferred can still partially flip, giving terminal around `19`

Practical conclusion from prompt-only v1:

- this is a real improvement signal, not noise
- the right next step is still prompt refinement, not PS retuning and not task
  deletion

## 17. Current recommendation after this session

This supersedes the earlier "go stabilize PS first" instinct for this exact
thread.

Recommended order:

1. Do **not** delete tasks from the dataset yet.
2. Do **not** run the next smoke yet.
3. Continue with prompt-only `v1.1` on fixed trace first.

Current prompt-only `v1.1` target changes:

- make `contract_self_check` a required diagnostic key, but still report-only
- strengthen the Stage 4 wording that `can_be_deferred=true` account / usage /
  policy blockers should stay ordinary-deferred unless there is an actual local
  canonical repair for them
- add a stronger dataset-10-style abstract example:
  - local MMS chain selected
  - `data_usage_exceeded` and account roaming policy deferred
  - final action should still be `repair_subset`
- add an explicit app-permission rule:
  - if app permission blocker is active and there is a canonical local
    permission repair, do not defer it while repairing APN / Wi-Fi / MMS
    downstream blockers

Only after prompt-only `v1.1` shows clear improvement on fixed trace should we
consider:

- running the latest clean 100-task dataset through a fixed-trace style check
- and only later returning to smoke / PS comparisons

Short version for handoff:

- the current blocker is Stage 4/5 contract stability on local repair chains
- exact-match `fdddd` high penalty is a real execution-layer cleanliness issue
- prompt-only v1 helped materially
- dataset `10` remains the main repair-subset contract case
- dataset `16` now looks healthy
- do not bypass this with task deletion or by immediately tuning PS
