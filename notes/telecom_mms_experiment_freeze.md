# Telecom MMS Experiment Freeze

This note freezes the first-pass telecom MMS fixed-tree setup for pilot and formal algorithm comparison.

## Dataset

- Primary dataset:
  - `data/derived/telecom_mms_fixed_tree_base/tasks.json`
- Sanity-only dataset:
  - `data/derived/telecom_mms_fixed_tree_smoke10/tasks.json`

`smoke10` is for quick sanity and regression checks. Formal reporting should use `base`.

## Environment

- Family:
  - `telecom_mms_recovery`
- Schema:
  - fixed `5-stage`
- Executors:
  - `simulated`:
    - primary experimental executor for algorithm comparison
  - `bench_backed`:
    - supporting sanity / fairness validation only
  - `llm_bench`:
    - supporting sanity / fairness validation only

## Evaluator

- Telecom evaluator:
  - `envs/telecom_mms_evaluator.py`
- Supported terminal actions:
  - `repair_all`
  - `repair_subset`
  - `transfer`

## Terminal Decision Freeze

The current first-pass terminal decision policy is intentionally simple:

- `repair_all`:
  - all blockers are safe for automatic repair
- `repair_subset`:
  - assistant-side deferable blockers are deferred
- `transfer`:
  - hybrid blockers trigger transfer

This policy is frozen for pilot and should not be changed during formal comparison runs.

## Pilot Intent

Pilot runs are meant to verify:

- algorithms run end-to-end on the frozen telecom track
- evaluator outputs are stable
- subset / transfer instances are included in scoring
- no method crashes on the frozen environment
