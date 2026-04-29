# Telecom MMS v4 Low-Transfer Smoke, 2026-04-27

This note records the latest completed telecom MMS profile-switch smoke run so
future Codex sessions do not need to reconstruct the context from `tmp/`.

## Run Identity

- Experiment label:
  `telecom_mms_lowtransfer_v4_10x10_n7_eta02_eps001_gpt4omini_13methods`
- Output directory:
  `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_n7_eta02_eps001_gpt4omini/`
- Runner:
  `scripts/run_shared_basin_repeated_smoke.py orchestrate`
- Executor:
  `llm_bench`
- Executor setting:
  `telecom_mms_agent_profile_only_clean_v4_hard_transfer_contract`
- Model:
  `gpt-4o-mini`
- Dataset:
  `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json`
- Schedule buckets:
  `analysis/shared_basin_prefix_dedup_profile_switch_lowtransfer_smoke_schedule_buckets.json`
- Family:
  `shared_basin_strong_prefix_dedup_profile_switch`
- Schedule mode:
  `trap_switch`
- Switch denominator:
  `7`
- Repeats:
  `10`
- Horizon:
  `100`
- Eta:
  `0.2`
- Epsilon:
  `0.01`
- Parallelism:
  method-level parallelism, 13 methods launched together.

## Dataset And Schedule

Unique low-transfer smoke dataset composition:

| expected terminal action | count |
| --- | ---: |
| `repair_all` | 33 |
| `repair_subset` | 34 |
| `transfer` | 1 |

Transfer rate is about `1.47%` in the 68-task dataset.

Actual repeated schedule composition:

| schedule category | count |
| --- | ---: |
| `trap_pre_switch` | 14 |
| `target_post_switch` | 86 |
| specialist episodes | 78 |

The schedule still repeats the single transfer task enough that there are
8 transfer-oracle episodes in the 100-episode smoke.

## Overall Ranking

Sorted by `raw_total_cost_mean`:

| rank | method | raw_total | terminal | outcome | policy | reasoning | exact | calls | tokens |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `direct_multistage_exp3` | 14.188 | 9.075 | 7.755 | 1.320 | 5.042 | 0.25 | 12.57 | 49510 |
| 2 | `risky_ps_safe_conditional` | 14.673 | 9.390 | 8.130 | 1.260 | 5.212 | 0.20 | 12.88 | 51333 |
| 3 | `risky_ps_ix` | 14.810 | 9.610 | 8.370 | 1.240 | 5.129 | 0.24 | 12.62 | 50500 |
| 4 | `risky_ps_old` | 14.887 | 9.525 | 8.265 | 1.260 | 5.292 | 0.20 | 12.88 | 51950 |
| 5 | `epsilon_exp3` | 14.973 | 9.695 | 8.535 | 1.160 | 5.208 | 0.18 | 12.65 | 50968 |
| 6 | `risky_ps_direct_cost` | 15.103 | 9.890 | 8.630 | 1.260 | 5.143 | 0.19 | 12.65 | 50360 |
| 7 | `direct_multistage_exp3_local` | 15.311 | 10.120 | 8.880 | 1.240 | 5.122 | 0.13 | 12.65 | 50099 |
| 8 | `risky_ps_safe_conditional_ix` | 15.366 | 10.215 | 8.935 | 1.280 | 5.079 | 0.23 | 12.74 | 50853 |
| 9 | `random_path` | 15.539 | 10.430 | 9.130 | 1.300 | 5.039 | 0.21 | 12.41 | 49355 |
| 10 | `naive_mixed_avg` | 15.577 | 10.045 | 8.985 | 1.060 | 5.459 | 0.20 | 13.90 | 53835 |
| 11 | `naive_mixed` | 15.639 | 10.205 | 8.965 | 1.240 | 5.363 | 0.16 | 13.15 | 52181 |
| 12 | `risky_ps` | 15.886 | 10.675 | 9.435 | 1.240 | 5.140 | 0.20 | 12.76 | 51179 |
| 13 | `risky_ps_linear` | 15.887 | 10.590 | 9.290 | 1.300 | 5.227 | 0.14 | 12.87 | 51234 |

Main takeaway: `direct_multistage_exp3` is best on this run. PS is close but
does not win; the best PS variants are `risky_ps_safe_conditional` and
`risky_ps_ix`.

## Post-Switch And Specialist Ranking

Sorted by `post_raw`:

| rank | method | post_raw | post_term | post_exact | specialist_raw | specialist_term | specialist_exact | specialist_shared |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `direct_multistage_exp3` | 15.485 | 10.308 | 0.22 | 16.395 | 11.186 | 0.24 | 0.90 |
| 2 | `risky_ps_safe_conditional` | 16.070 | 10.674 | 0.16 | 17.065 | 11.615 | 0.18 | 0.94 |
| 3 | `risky_ps_ix` | 16.171 | 10.930 | 0.21 | 17.191 | 11.897 | 0.23 | 0.92 |
| 4 | `risky_ps_old` | 16.274 | 10.831 | 0.16 | 17.258 | 11.763 | 0.18 | 0.96 |
| 5 | `epsilon_exp3` | 16.314 | 10.988 | 0.15 | 17.346 | 11.962 | 0.17 | 0.90 |
| 6 | `risky_ps_direct_cost` | 16.515 | 11.262 | 0.15 | 17.526 | 12.237 | 0.17 | 0.97 |
| 7 | `direct_multistage_exp3_local` | 16.818 | 11.523 | 0.08 | 17.914 | 12.577 | 0.09 | 0.91 |
| 8 | `risky_ps_safe_conditional_ix` | 16.825 | 11.634 | 0.20 | 17.856 | 12.647 | 0.22 | 0.92 |
| 9 | `naive_mixed_avg` | 16.962 | 11.459 | 0.16 | 18.084 | 12.532 | 0.18 | 0.99 |
| 10 | `random_path` | 17.013 | 11.919 | 0.17 | 18.131 | 13.013 | 0.19 | 0.91 |
| 11 | `naive_mixed` | 17.064 | 11.640 | 0.13 | 18.147 | 12.679 | 0.14 | 0.97 |
| 12 | `risky_ps` | 17.430 | 12.169 | 0.16 | 18.636 | 13.314 | 0.18 | 0.97 |
| 13 | `risky_ps_linear` | 17.446 | 12.093 | 0.09 | 18.632 | 13.231 | 0.10 | 0.94 |

Post-switch/specialist is the key claim area. `direct_multistage_exp3` still
wins there. The best PS methods are again `risky_ps_safe_conditional` and
`risky_ps_ix`.

## Transfer Safety

All 13 methods output `final_action=transfer` for all 8 transfer-oracle repeated
episodes.

This is evidence that v4 removed the major failure mode where local paths could
turn hard hybrid/nonlocal blockers into local `repair_all`. However, this smoke
export does not include per-episode fields such as:

- `clean_success_no_fallback`
- `hard_transfer_guard_applied`
- executor completion diagnostics
- Stage 4 raw decision diagnostics

Do not claim those rates from this repeated smoke output alone. Use the fixed
path diagnostics or extend the smoke export if those fields are needed.

## Interpretation

The v4 executor/safety setup is good enough to run smoke on low-transfer data:
hard-transfer cases no longer dominate the metric, and transfer safety is not
obviously broken.

The smoke does not yet show PS as the best method. The likely bottleneck is not
the transfer guard, but noisy local repair execution after the switch. Specialist
exact match remains low even for the best method (`0.24` for
`direct_multistage_exp3`), so many target/local repair tasks still end as
`repair_subset` or otherwise incur high terminal penalty.

This means the next investigation should focus on whether PS fails because:

- it does not select the target/shared/deep path quickly enough; or
- it selects a good path, but the LLM execution still fails Stage 4 local repair
  completeness.

## Recommended Next Diagnostic

Run a focused comparison on the same low-transfer schedule for:

- `direct_multistage_exp3`
- `risky_ps_safe_conditional`
- `risky_ps_ix`

For each post-switch specialist episode, export:

- selected path and stage profiles
- whether the path is shared/unshared
- final action and terminal penalty
- Stage 3 blocker list
- Stage 4 raw decision
- Stage 4 normalized decision
- hard-transfer guard flag
- fallback flag
- executor completion flag
- Stage 5 replay tool names and verification tools

This would tell us whether the remaining gap is algorithmic path selection or
LLM execution quality on local repair chains.

## Reproduction Command

```bash
PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini python scripts/run_shared_basin_repeated_smoke.py orchestrate \
  --data data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json \
  --output-dir tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_n7_eta02_eps001_gpt4omini \
  --repeats 10 \
  --family-kind shared_basin_strong_prefix_dedup_profile_switch \
  --schedule-mode trap_switch \
  --switch-denominator 7 \
  --schedule-buckets analysis/shared_basin_prefix_dedup_profile_switch_lowtransfer_smoke_schedule_buckets.json \
  --common-eta-override 0.2 \
  --common-epsilon-override 0.01 \
  --executor-name llm_bench \
  --methods naive_mixed_avg direct_multistage_exp3 risky_ps_safe_conditional risky_ps_linear risky_ps_safe_conditional_ix direct_multistage_exp3_local risky_ps risky_ps_direct_cost random_path naive_mixed risky_ps_old risky_ps_ix epsilon_exp3
```
