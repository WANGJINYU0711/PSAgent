
# Telecom MMS Formal Experiment Protocol

This note defines the formal experiment protocol for the `telecom_mms_recovery` track.

The central principle is:

**We evaluate how algorithms choose stage/path/agent under one fixed agent background.**

We do **not** center the formal comparison on:

- which executor is engineered most perfectly
- which LLM fallback is most complete
- which agent system is most realistic as a customer-service product

Once the track is frozen, those implementation details should not keep changing during the formal comparison.

---

## 1. Formal Goal

The formal goal is:

**Compare algorithms on the same fixed telecom workflow family, under the same path space, the same evaluator, and the same decision space.**

The primary object of comparison is:

- how well each method selects stage/path/agent
- and how much regret it incurs relative to stronger references

This is not an executor benchmark.

---

## 2. Frozen Track Definition

### 2.1 Dataset

- Formal dataset:
  - `data/derived/telecom_mms_fixed_tree_base/tasks.json`
- Sanity-only dataset:
  - `data/derived/telecom_mms_fixed_tree_smoke10/tasks.json`

#### Difference Between `base` and `smoke10`

`base`

- the formal experiment dataset
- used for all reported main results
- contains the full derived telecom MMS benchmark currently used for comparison
- includes all three terminal actions:
  - `repair_all`
  - `repair_subset`
  - `transfer`

`smoke10`

- a tiny sanity-only subset
- used for:
  - smoke tests
  - regression checks
  - quick debugging
  - executor sanity validation
- not used for formal headline results

Rule:

- **Formal reporting must use `base`.**
- `smoke10` is only for sanity and regression.

### 2.2 Family and Schema

- Family:
  - `telecom_mms_recovery`
- Schema:
  - fixed `5-stage`

Stage names:

1. `stage1_user_grounding`
2. `stage2_customer_line_resolution`
3. `stage3_observed_feature_extraction`
4. `stage4_blocker_adjudication_and_repair_plan`
5. `stage5_terminal_execution_decision`

### 2.3 Executors

- Primary formal experiment executor:
  - `simulated`
- Supporting sanity executors only:
  - `bench_backed`
  - `llm_bench`

Rule:

- Main algorithm comparison should use `simulated`.
- `bench_backed` and `llm_bench` are for sanity/fairness/supporting checks only, not the main scoreboard.

### 2.4 Evaluator

- Evaluator:
  - `envs/telecom_mms_evaluator.py`

Supported terminal actions:

- `repair_all`
- `repair_subset`
- `transfer`

### 2.5 Terminal Decision Freeze

The current first-pass terminal-decision policy is frozen during formal comparison:

- `repair_all`
  - all blockers are safe for automatic repair
- `repair_subset`
  - assistant-side deferable blockers are deferred
- `transfer`
  - hybrid blockers trigger transfer

This policy should not change during the formal comparison run.

---

## 3. Algorithms To Run

### 3.1 Required Formal Methods

The formal comparison should include all of the following:

1. `risky_ps`
2. `direct_multistage_exp3`
3. `epsilon_exp3`
4. `full_share`
5. `full_unshare`
6. `naive_mixed`
7. `random_path`
8. `oracle`

### 3.2 Implementation Note

At the moment, the repository clearly contains:

- `direct_multistage_exp3`
- `epsilon_exp3`
- `full_share`
- `full_unshare`
- `naive_mixed`
- `random_path`

I do **not** currently see an explicit baseline file for:

- `risky_ps`
- `oracle`

So for the formal run, these two must be treated as:

- required experiment methods
- and, if not already implemented elsewhere, must be connected before the final run starts

The protocol assumes they will be available.

---

## 4. Interaction Mechanisms

Each method must be evaluated under **three interaction mechanisms**.

### Mechanism A

**Algorithm directly selects the agent/path**

Meaning:

- the algorithm chooses the stage/path/agent directly
- the agent does not reinterpret a separate algorithm score

### Mechanism B

**Algorithm outputs theta/score; the agent sees it and then selects**

Meaning:

- the algorithm provides score guidance
- the agent uses the score as reference
- the final choice is still made in the agent background

### Mechanism C

**Agent selects by itself, without algorithm guidance**

Meaning:

- no algorithm signal is used
- the agent chooses on its own

### Formal Rule

Every method should be evaluated under all three mechanisms:

1. direct algorithm selection
2. theta-guided agent selection
3. agent-only selection

This is required for a complete formal comparison.

---

## 5. Seeds and Repeats

### 5.1 Minimal First Formal Pass

You said this can start with one run first, so the protocol is:

- first formal pass:
  - `1` seed

### 5.2 Recommended Final Report Setting

After the first full run works, recommended final setting:

- `3` seeds

Optional stronger setting:

- `5` seeds

### Seed Policy

- All methods must use the same seed set
- All three interaction mechanisms must use the same seed set
- Seeds must be recorded in config and results

Recommended progression:

1. first full pass:
   - seed `0`
2. stable comparison pass:
   - seeds `0, 1, 2`
3. stronger final robustness check:
   - seeds `0, 1, 2, 3, 4`

---

## 6. Family-Kind Policy

Formal runs should include:

- `neutral`
- `moderate`
- `strong`

Rule:

- Every compared method must run on the same family kinds
- All three interaction mechanisms must run on the same family kinds

---

## 7. Metrics

### 7.1 Primary Metric

The primary metric should include **regret**.

Regret is the main metric because your core research question is algorithmic decision quality under the same path-selection environment.

At minimum, formal reporting should include:

- `regret`

Formal regret must be defined against a **stationary oracle**:

- one fixed path is chosen for the whole run
- that same path is reused for every episode in the horizon
- the stationary oracle is the fixed path with minimum cumulative total cost over the whole run

Formally:

- `cumulative_regret`
  - `algorithm_cumulative_total_cost - stationary_oracle_cumulative_total_cost`
- `mean_regret`
  - `cumulative_regret / num_episodes`

Important distinction:

- per-instance oracle:
  - best path for one instance only
  - useful for sanity/supporting analysis
  - **not** the formal regret baseline
- stationary oracle:
  - best fixed path over the whole run
  - **this is the formal regret baseline**

### 7.2 Required Main Metrics

For each method × mechanism × family kind:

- `regret`
- `exact_match_mean`
- `terminal_penalty_mean`
- `total_cost_mean`
- `final_action_accuracy`

### 7.3 Required Terminal-Action Metrics

Report action-stratified results for:

- `repair_all`
- `repair_subset`
- `transfer`

Acceptable summary forms include:

- per-action exact match
- per-action terminal penalty
- per-action final-action accuracy

### 7.4 Recommended Supporting Metrics

- `path_cost_mean`
- `leaf_type` distribution
- `persona`-stratified performance
- `blocker complexity`-stratified performance

Suggested blocker-complexity views:

- by `num_blockers`
- by `blocker_layers_present`

### 7.5 Stage-Level Metrics

Only include stage-level success metrics if they are already stably available and consistent across methods.

They are optional, not mandatory.

The formal comparison should not be delayed waiting for more detailed stage-level instrumentation.

---

## 8. Aggregation and Statistics

### 8.1 Required Aggregation

For each reported metric, provide:

- `mean`
- `std`

### 8.2 Optional Aggregation

If cheap to compute, also include:

- confidence intervals
- per-seed raw summaries

Rule:

- Mean and standard deviation are the minimum standard for the formal comparison.

---

## 9. Logging Requirements

Every formal run should preserve:

1. experiment config
2. method/mechanism list
3. seed list
4. episode-level logs
5. overall summary
6. by-action summary
7. by-family-kind summary

Recommended output directory structure:

- `outputs/telecom_mms_main_experiment/experiment_config.json`
- `outputs/telecom_mms_main_experiment/episode_logs.jsonl`
- `outputs/telecom_mms_main_experiment/overall_summary.json`
- `outputs/telecom_mms_main_experiment/overall_summary.csv`
- `outputs/telecom_mms_main_experiment/by_action_summary.json`
- `outputs/telecom_mms_main_experiment/by_action_summary.csv`
- `outputs/telecom_mms_main_experiment/by_family_summary.json`
- `outputs/telecom_mms_main_experiment/by_family_summary.csv`
- optionally:
  - `outputs/telecom_mms_main_experiment/by_method_seed_summary.json`

Each episode log should at least contain:

- method
- mechanism
- seed
- family_kind
- instance_id
- selected_path
- leaf_type
- final_action
- oracle_action
- exact_match
- terminal_penalty
- total_cost
- regret contribution, if logged at episode level

If episode-level regret is included, it should be interpreted as:

- `episode_regret = episode_total_cost - stationary_oracle_episode_cost`

where the reference episode cost comes from the same run-level stationary oracle path.

---

## 10. Formal Run Plan

### 10.1 First Full Formal Pass

Run:

- dataset:
  - full `telecom_mms_fixed_tree_base`
- methods:
  - all required methods
- mechanisms:
  - all 3 mechanisms
- family kinds:
  - `neutral`
  - `moderate`
  - `strong`
- seeds:
  - first pass with `1` seed
- executor:
  - `simulated`

Purpose:

- verify the full formal pipeline runs end-to-end
- ensure all methods and all mechanisms are compatible
- collect the first full comparison table

### 10.2 Main Formal Comparison

After the first full pass is stable, run:

- the same dataset
- the same method set
- the same 3 mechanisms
- the same family kinds
- `3` seeds minimum

### 10.3 Optional Stronger Final Version

If runtime allows:

- extend to `5` seeds
- include extra stratified summaries by persona and blocker complexity

---

## 11. Completion Standard

The formal protocol is satisfied when:

1. every selected method can be run under the same frozen telecom track
2. all methods are evaluated under the same:
   - dataset
   - evaluator
   - executor
   - family kinds
   - seed set
   - mechanism set
3. results are exported in a unified format
4. the main comparison table includes regret and terminal-decision metrics
5. the protocol can be rerun without redefining the track mid-experiment

---

## 12. Non-Goals

The following are explicitly not the center of the formal comparison phase:

- improving executor engineering details
- polishing LLM fallback completeness
- making customer interaction more realistic
- upgrading stage4/5 to real execution
- redesigning the telecom family
- chasing perfect oracle-level diagnosis recall before the main run

These can be explored later, but they are not prerequisites for the formal algorithm comparison.
