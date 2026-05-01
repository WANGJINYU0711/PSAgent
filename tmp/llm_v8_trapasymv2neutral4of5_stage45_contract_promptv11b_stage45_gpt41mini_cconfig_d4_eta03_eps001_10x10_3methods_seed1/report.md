# TrapAsym v2 Neutral 4/5 Seed1 LLM Smoke Report

## Run

- Experiment: `llm_v8_trapasymv2neutral4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed1`
- Output dir: `tmp/llm_v8_trapasymv2neutral4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed1`
- Tree/family: `shared_basin_strong_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5`
- Dataset: `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- Buckets: `analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- Seed: `PSAGENT_REPEATED_SMOKE_SEED=1`
- Methods: `risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`
- Schedule: `trap_switch`, `switch_denominator=4`, `repeats=10`, `horizon=100`
- Config: C config, terminalv4, reasoning calibration v3, report-only modecost, stage45 prompt v1.1b, stage45 model `gpt-4.1-mini`, base LLM `gpt-4o-mini`
- W&B: live partial uploader only, no final post-run upload.

## W&B

- Project: `psagent-llm-smoke`
- Entity: `wangjinyu0711-microsoft`
- Group: `llm_v8_trapasymv2neutral4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed1`
- Live uploader state: all methods uploaded through episode index 99.

Known live run URLs from the uploader log:

- `risky_ps`: https://wandb.ai/wangjinyu0711-microsoft/psagent-llm-smoke/runs/llm_v8_trapasymv2neutral4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_-c0f6fd24f34b
- `direct_multistage_exp3`: https://wandb.ai/wangjinyu0711-microsoft/psagent-llm-smoke/runs/llm_v8_trapasymv2neutral4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_-06b322eb9bc9
- `epsilon_exp3`: https://wandb.ai/wangjinyu0711-microsoft/psagent-llm-smoke/runs/llm_v8_trapasymv2neutral4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_-faeb0c21737f

## Early Stop

At the 75-episode checkpoint, early stop did not trigger.

```json
{
  "ps_raw_mean": 8.0739368,
  "best_baseline_method": "direct_multistage_exp3",
  "best_baseline_raw_mean": 8.947931333333333,
  "gap_vs_best_baseline": -0.8739945333333328,
  "raw_gap_threshold": 0.5,
  "early_stop": false
}
```

Interpretation: the requested early-stop condition was "stop if PS is clearly behind exp/eps at 75." The gap was negative, so PS was ahead of the best baseline by 0.874 raw cost at the checkpoint.

## Final Ranking

| rank | method | raw_total_cost_mean | total_cost_mean | raw_terminal_penalty_mean | exact_match_mean | raw_reasoning_cost_component_mean | raw_path_cost_component_mean | shared_path_fraction |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `risky_ps` | 8.846800 | 0.238201 | 3.360000 | 0.790000 | 5.420330 | 0.066470 | 0.860000 |
| 2 | `direct_multistage_exp3` | 9.179600 | 0.247162 | 3.710000 | 0.750000 | 5.403417 | 0.066183 | 0.850000 |
| 3 | `epsilon_exp3` | 9.233381 | 0.248610 | 3.750000 | 0.760000 | 5.416453 | 0.066928 | 0.840000 |

Main result: `risky_ps` wins on the final 100-episode run.

Margins:

- PS vs direct: `-0.332800` raw total cost, `-0.350000` raw terminal penalty, `+0.040000` exact match.
- PS vs epsilon: `-0.386582` raw total cost, `-0.390000` raw terminal penalty, `+0.030000` exact match.

The PS win is mostly terminal-quality driven. Reasoning cost and path cost are close across methods.

## Phase Split

### Pre-switch, episodes 0-24

| method | n | raw_total | terminal | reasoning | path | exact | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| `risky_ps` | 25 | 4.855898 | 0.000000 | 4.790694 | 0.065204 | 1.000000 | 0.840000 |
| `direct_multistage_exp3` | 25 | 4.923302 | 0.000000 | 4.858610 | 0.064692 | 1.000000 | 0.840000 |
| `epsilon_exp3` | 25 | 4.565601 | 0.000000 | 4.501525 | 0.064076 | 1.000000 | 0.800000 |

Pre-switch is easy for all methods; terminal penalty is zero across the board. Epsilon is cheapest in raw cost pre-switch, mainly by lower reasoning cost.

### Post-switch, episodes 25-99

| method | n | raw_total | terminal | reasoning | path | exact | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| `risky_ps` | 75 | 10.177100 | 4.480000 | 5.630208 | 0.066892 | 0.720000 | 0.866667 |
| `direct_multistage_exp3` | 75 | 10.598365 | 4.946667 | 5.585019 | 0.066680 | 0.666667 | 0.853333 |
| `epsilon_exp3` | 75 | 10.789308 | 5.000000 | 5.721430 | 0.067879 | 0.680000 | 0.853333 |

Post-switch is where PS wins. It has lower raw total cost than direct by 0.421 and lower raw terminal penalty by 0.467. It beats epsilon by 0.612 raw total and 0.520 terminal penalty.

### Last 25 Episodes

| method | n | raw_total | terminal | reasoning | path | exact | shared |
|---|---:|---:|---:|---:|---:|---:|---:|
| `risky_ps` | 25 | 11.165388 | 5.720000 | 5.377940 | 0.067448 | 0.600000 | 0.880000 |
| `direct_multistage_exp3` | 25 | 9.874604 | 4.320000 | 5.487500 | 0.067104 | 0.720000 | 0.960000 |
| `epsilon_exp3` | 25 | 9.879166 | 4.200000 | 5.610302 | 0.068864 | 0.800000 | 0.800000 |

Caveat: PS leads overall and at the 75 checkpoint, but its last-25 window is worse than both baselines. This suggests PS gained enough in the earlier post-switch segment to win the full horizon, then degraded late on this seed/tree. Do not overclaim monotonic PS stabilization from this run alone.

## Specialist Slice

The legacy "specialist" slice has 32 episodes.

| method | specialist_raw_terminal_penalty_mean | specialist_total_cost_mean | specialist_shared_path_fraction | specialist_unshared_path_fraction |
|---|---:|---:|---:|---:|
| `risky_ps` | 9.625000 | 0.435814 | 0.875000 | 0.125000 |
| `direct_multistage_exp3` | 9.437500 | 0.429881 | 0.843750 | 0.156250 |
| `epsilon_exp3` | 7.406250 | 0.369884 | 0.812500 | 0.187500 |

Specialist-slice caveat: PS does not win the legacy specialist slice here. Epsilon is best on that slice, with substantially lower terminal penalty. The overall PS win is not because PS dominates every target-heavy subpopulation; it is an aggregate 100-episode win driven by better non-specialist/post-switch behavior.

## Run Integrity

- The run was executed with tmux and parallel method windows.
- W&B uploading was live via `scripts/live_wandb_partial_uploader.py`; final merge did not perform an additional W&B upload.
- Transient OpenAI/litellm connection errors occurred during the run. Checkpoints were used to resume the affected methods in the same run directory.
- A resume guard was added for this run after the first transient failures; completed episode files and final summaries contain 100 episodes for all three methods.
- Final local merge artifacts were generated after method completion.

## Interpretation

This seed1 run supports the new v2 neutral 4/5 tree as a paper-facing PS-positive but less hand-protected tree:

- The tree is not the earlier all-share or fully PS-protected v1 layout.
- Despite target/private noise in the v2 neutral tree, `risky_ps` ranks first overall.
- The win is modest but clean in final raw total cost: `8.8468` vs `9.1796` and `9.2334`.
- The 75 checkpoint strongly favored PS and correctly avoided early stop.
- However, the last-25 degradation and specialist-slice weakness mean this should be presented as evidence of PS advantage on this run, not as proof that PS dominates every post-switch subset.

