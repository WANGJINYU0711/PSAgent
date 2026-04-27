# PSAgent Working Notes

This repository is being actively edited by multiple Codex sessions. Treat the
worktree as shared and dirty: do not revert changes you did not make, and do not
delete generated `tmp/` outputs unless the user explicitly asks.

## Current Goal

The active experiment is a telecom MMS profile-switch / shared-basin study. The
user is trying to make post-switch `deep` / `target` / shared paths genuinely
cheaper on local repair-heavy tasks, without letting fast/trap paths win by
conservatively transferring hard cases.

Important high-level objective:

- Show that `deep`/`target` paths improve terminal quality on local non-transfer
  post-switch tasks.
- Keep hard hybrid/nonlocal transfer safety intact.
- Avoid letting `transfer`-expected tasks dominate smoke metrics.
- Compare PS/baselines only after execution-layer terminal signals are sane.

## Files And Areas

Key executor and diagnostic files:

- `envs/executors/telecom_llm_bench_executor.py`
  - Telecom LLM executor.
  - Contains fast token soft penalty constants.
  - Contains Stage 4 hard hybrid/nonlocal guard.
  - Contains Stage 4 deep/target-only local downstream completion logic.
  - Contains Stage 4 raw diagnostics fields.
- `envs/executors/_telecom_llm_bench_bridge.py`
  - LLM bridge for stage execution.
  - Tracks per-call token usage, JSON retry count, valid JSON first try, and
    estimated token totals.
- `scripts/run_llm_path_sweep_diagnostic.py`
  - Fixed-path diagnostic sweep.
  - Writes records/CSV/summary with stage resource diagnostics, fast token
    over-budget fields, Stage 4 raw output, completion diagnostics, and
    `raw_total_cost_with_token_penalty`.
- `scripts/run_shared_basin_repeated_smoke.py`
  - Main repeated smoke runner used for PS/baseline runs.
- `scripts/build_shared_basin_profile_switch_assets.py`
  - Builds profile-switch task/bucket assets.
- `scripts/build_telecom_mms_profile_switch_low_transfer_benchmark.py`
  - New/experimental script for low-transfer profile-switch dataset creation.

Important task/bucket assets:

- Original 100-task profile-switch dataset:
  `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch/tasks.json`
- Original profile-switch buckets:
  `analysis/shared_basin_prefix_dedup_profile_switch_schedule_buckets.json`
- Non-transfer smoke dataset:
  `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_nontransfer_smoke/tasks.json`
- Non-transfer smoke buckets:
  `analysis/shared_basin_prefix_dedup_profile_switch_nontransfer_smoke_schedule_buckets.json`
- Low-transfer 100-task candidate dataset:
  `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_low_transfer/tasks.json`
- Low-transfer buckets:
  `analysis/shared_basin_prefix_dedup_profile_switch_low_transfer_schedule_buckets.json`

## Important Experiment State

The original 100-task profile-switch dataset has roughly balanced terminal
actions:

- `repair_all`: 33
- `repair_subset`: 34
- `transfer`: 33

There is no explicit `transfer-easy` field. The current proxy is
`metadata.expected_terminal_action == "transfer"`. Those transfer tasks are
mostly hard hybrid/nonlocal cases:

- `repairability = transfer_required`
- `contains_hybrid_action = true`
- `contains_assistant_side_action = true`

Because about one third of the original 100-task set is transfer-expected, smoke
metrics over the full set can make fast/trap paths look artificially strong by
conservative transfer. For smoke that evaluates deep local repair, prefer the
non-transfer or low-transfer datasets above.

## Recent Diagnostic Results

Early diagnostics found that Stage 4 was collapsing many post-switch paths into
`transfer_required` via `shallow_subset_requires_downstream_unlock_v1`. That was
narrowed to hard hybrid/nonlocal cases, now using
`shallow_subset_defers_nonlocal_blocker_v2`.

Stage 5 previously flipped `repair_subset` selected/deferred semantics. That was
fixed: Stage 5 now defaults to the Stage 4 `should_repair=true` subset for
`repair_subset`.

Hybrid/nonlocal hard guard was added so LLM output cannot localize hard transfer
blockers such as `user_abroad_roaming_disabled_off` into `repair_all`.

Recent local-only diagnostic output:

- `tmp/llm_path_sweep_diagnostic_local_nontransfer_v2/`
  - `pure_target` began winning on local repair tasks.
  - Stage 4 completion pass applied on all `pure_target` paths.
  - One positive local repair chain reached `repair_all`, `exact_match=true`,
    `raw_terminal_penalty=0`.
  - Remaining issue: in a storage-permission local task, Stage 4 raw output
    selected only `airplane_mode_on` and missed `unseat_sim_card`, so downstream
    completion could not safely add APN / Wi-Fi / permission blockers.
- `tmp/llm_path_sweep_diagnostic_local_nontransfer_v3/`
  - Exists and should be checked before making further changes; it may include
    the follow-up prerequisite completion experiment.

The current suspected remaining bottleneck is Stage 4 deep/target upstream
completeness on local repair chains, not Stage 5 closure.

## Fast Cost And Token Penalty

Fast token diagnostics were added because fast paths may exceed their intended
budget via retries, fallback, or large prompts. The current soft penalty is a
diagnostic cost view, not a replacement for `raw_terminal_penalty` or the
original `raw_total_cost`.

Current constants:

- `FAST_TOKEN_BUDGET_PER_STAGE = 1200`
- `FAST_TOKEN_PENALTY_BLOCK_SIZE = 200`
- `FAST_TOKEN_OVER_BUDGET_PENALTY_PER_BLOCK = 0.25`

Path-level field:

- `raw_total_cost_with_token_penalty = raw_total_cost + fast_token_over_budget_penalty_total`

Use this for analysis, but be cautious before making it the main smoke objective.

## Data Strategy

For smoke, avoid using the original 100-task dataset as the only metric because
33% transfer cases can hide the deep/target local repair signal.

Recommended smoke progression:

1. Run non-transfer smoke on:
   `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_nontransfer_smoke/tasks.json`
2. Optionally run low-transfer smoke on:
   `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_low_transfer/tasks.json`
3. Report hard-transfer control separately from local non-transfer repair
   metrics.

When analyzing results, split metrics by:

- `repair_all`
- `repair_subset`
- `transfer`
- local non-transfer
- hard hybrid/nonlocal transfer control

Do not mix hard-transfer cases into the main claim that deep/target local repair
is cheaper.

## Current Worktree Notes

As of this note, the worktree has tracked modifications in:

- profile-switch generated assets under `analysis/` and `data/derived/`
- `envs/executors/_telecom_llm_bench_bridge.py`
- `envs/executors/telecom_llm_bench_executor.py`
- `scripts/build_shared_basin_profile_switch_assets.py`
- `scripts/run_llm_path_sweep_diagnostic.py`

There are untracked generated datasets, bucket files, scripts, and `tmp/`
diagnostic/smoke outputs. Treat these as intentional unless proven otherwise.

## Safety Boundaries

Do not change unless explicitly asked:

- PS algorithm behavior.
- share/unshare mechanics.
- baseline definitions.
- oracle labels.
- `raw_terminal_penalty` semantics.
- original 100-task dataset in place.

Prefer adding new dataset directories and new bucket files rather than
overwriting existing profile-switch assets.

## Useful Commands

Inspect current relevant outputs:

```bash
python - <<'PY'
import json
from pathlib import Path
p = Path('tmp/llm_path_sweep_diagnostic_local_nontransfer_v3/summary.json')
print(p, p.exists())
if p.exists():
    d = json.loads(p.read_text())
    print(json.dumps(d.get('extended_summary', d), indent=2)[:4000])
PY
```

Find smoke runner options:

```bash
python scripts/run_shared_basin_repeated_smoke.py --help
```

Search fast/token or Stage 4 completion logic:

```bash
rg -n "FAST_TOKEN|stage4_completion|completion_pass|hard_nonlocal|raw_total_cost_with_token_penalty" \
  envs/executors scripts
```

