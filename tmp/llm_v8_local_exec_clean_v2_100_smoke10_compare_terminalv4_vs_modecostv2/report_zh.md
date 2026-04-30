# llm_v8 local_exec_clean_v2_100 10x10 3-method smoke 对比报告

## 实验设置

- 数据集：`data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- 10x10 bucket：`analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- 方法：`direct_multistage_exp3`, `epsilon_exp3`, `risky_ps_linear`
- LLM：`gpt-4o-mini`
- executor：`llm_bench`
- repeats/horizon：`10 / 100`
- schedule：`trap_switch`, `switch_denominator=4`, switch episode = 25
- policy 参数：`eta=0.3`, `epsilon=0.01`
- schedule 组成：25 个 `trap_pre_switch`，75 个 `target_post_switch`
- 任务要求：pre-switch 全部 `mostly_fast_required`；post-switch 全部 `mostly_deep_required`
- oracle：pre-switch 25 个 `repair_all`；post-switch 51 个 `repair_all` + 24 个 `repair_subset`

## 两个版本

1. `terminalv4`：启用 clean terminal floor / subset mismatch base / 显式 local-transfer floor；不改 reasoning 权重。
2. `terminalv4_modecostv2`：在 `terminalv4` 基础上，启用 mode mismatch reasoning penalty：fast-on-deep-required 每 stage +1.5，deep-on-fast-required 每 stage +0.5。

## 总表

见：`summary_compare.csv`、`summary_compare.svg`。

| run | split | method | total | terminal | legacy | reasoning | modecost | terminal share | clear | aux | strict | mode exact |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminalv4 | all | direct_multistage_exp3 | 12.08 | 7.14 | 5.18 | 4.86 | 0.00 | 59% | 63% | 74% | 59% | 24% |
| terminalv4 | all | epsilon_exp3 | 11.39 | 6.39 | 4.58 | 4.93 | 0.00 | 56% | 67% | 74% | 64% | 18% |
| terminalv4 | all | risky_ps_linear | 9.71 | 4.71 | 3.06 | 4.93 | 0.00 | 49% | 75% | 76% | 68% | 25% |
| terminalv4 | post | direct_multistage_exp3 | 14.47 | 9.52 | 6.91 | 4.87 | 0.00 | 66% | 51% | 65% | 45% | 32% |
| terminalv4 | post | epsilon_exp3 | 13.34 | 8.52 | 6.11 | 4.74 | 0.00 | 64% | 56% | 65% | 52% | 24% |
| terminalv4 | post | risky_ps_linear | 11.16 | 6.15 | 4.07 | 4.94 | 0.00 | 55% | 68% | 68% | 59% | 33% |
| terminalv4_modecostv2 | all | direct_multistage_exp3 | 12.10 | 5.81 | 3.87 | 6.22 | 1.38 | 48% | 70% | 72% | 61% | 23% |
| terminalv4_modecostv2 | all | epsilon_exp3 | 12.39 | 5.72 | 3.78 | 6.60 | 1.53 | 46% | 67% | 77% | 65% | 15% |
| terminalv4_modecostv2 | all | risky_ps_linear | 12.89 | 6.20 | 4.32 | 6.62 | 1.54 | 48% | 68% | 77% | 62% | 20% |
| terminalv4_modecostv2 | post | direct_multistage_exp3 | 13.96 | 7.75 | 5.15 | 6.14 | 1.31 | 56% | 60% | 63% | 48% | 31% |
| terminalv4_modecostv2 | post | epsilon_exp3 | 14.16 | 7.63 | 5.04 | 6.46 | 1.49 | 54% | 56% | 69% | 53% | 20% |
| terminalv4_modecostv2 | post | risky_ps_linear | 14.71 | 8.27 | 5.76 | 6.37 | 1.47 | 56% | 57% | 69% | 49% | 27% |

## 关键 mode 路线对比：post-switch deep-required

### terminalv4

| method | pair | n | terminal | legacy | reasoning | total | clear | aux | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | deep-on-deep | 62 | 9.87 | 7.57 | 4.87 | 14.82 | 52% | 69% | 45% |
| direct | fast-on-deep | 13 | 7.85 | 3.77 | 4.88 | 12.79 | 46% | 46% | 46% |
| epsilon | deep-on-deep | 58 | 8.66 | 6.41 | 4.68 | 13.40 | 57% | 67% | 52% |
| epsilon | fast-on-deep | 17 | 8.06 | 5.12 | 4.98 | 13.11 | 53% | 59% | 53% |
| risky | deep-on-deep | 65 | 6.07 | 4.10 | 4.90 | 11.04 | 69% | 68% | 58% |
| risky | fast-on-deep | 10 | 6.65 | 3.85 | 5.21 | 11.92 | 60% | 70% | 60% |

### terminalv4 + modecostv2

| method | pair | n | terminal | legacy | reasoning | modecost | total | clear | aux | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | deep-on-deep | 62 | 7.48 | 5.29 | 5.62 | 0.84 | 13.17 | 65% | 65% | 50% |
| direct | fast-on-deep | 13 | 9.04 | 4.50 | 8.64 | 3.58 | 17.74 | 38% | 54% | 38% |
| epsilon | deep-on-deep | 61 | 7.75 | 5.48 | 6.02 | 1.07 | 13.84 | 56% | 72% | 54% |
| epsilon | fast-on-deep | 14 | 7.11 | 3.11 | 8.41 | 3.32 | 15.58 | 57% | 57% | 50% |
| risky | deep-on-deep | 61 | 8.46 | 6.07 | 5.96 | 1.09 | 14.50 | 57% | 66% | 48% |
| risky | fast-on-deep | 14 | 7.43 | 4.39 | 8.16 | 3.14 | 15.66 | 57% | 86% | 57% |

## 解释

- terminalv4 确实把失败 episode 的 terminal 拉高了：例如 clear=false 或 aux=false 的 episode 不再只吃 2-4 分的小罚，而会命中 10/12/14/18 这类 floor。legacy terminal 和 v4 terminal 的差距能直接看到。
- 但 terminalv4 不会机械地把所有 episode 都拉到 terminal 占比 60-70%。pre-switch 里很多成功 repair_all 的 terminal=0，所以 all split 的 terminal share 会被拉低。post split 更接近你要的比例：direct/epsilon 是 66%/64%，risky 是 55%。
- modecostv2 把 fast-on-deep 的 total 明显推高，尤其 direct fast-on-deep 从 12.79 上到 17.74；这解决了“fast-on-deep 失败但便宜”的显示问题。
- 但是 modecostv2 作为训练目标有点太重：all split terminal share 反而降到 46-48%，post 也只有 54-56%。如果目标是 terminal:reasoning 接近 6:4 或 7:3，这版不适合作为最终主 objective，更适合作为诊断或需要降权。
- terminalv4-only 下 risky_ps_linear 仍然最好，不是因为失败低罚，而是因为它这批样本上的 clear/strict 更高。modecostv2 下 direct 变第一，但主要是 objective 被 mode mismatch 成本重塑，不能直接说明 terminal 质量最好。

## 产物

- A run：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4/`
- B run：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_modecostv2/`
- 每组都有：
  - `report.md`
  - `episode_cost_success_mode_compact.csv`
  - `majority_pair_cost_matrix.svg`
  - `phase_majority_pair_cost_table.svg`
- 合并对比：
  - `summary_compare.csv`
  - `summary_compare.md`
  - `summary_compare.svg`
  - `report_zh.md`

## 建议

1. 主线先采用 terminalv4，不建议直接把 modecostv2 当最终 objective。
2. 如果继续试 reasoning 权重，下一版建议把 modecost 降到 fast-on-deep +0.75、deep-on-fast +0.25，或只在报告中作为 diagnostic，不进入 policy update。
3. 对“64开/73开”建议用 post/local repair split 来看，而不是 all split；all split 会被 pre-switch 成功任务的 terminal=0 稀释。
4. 下一轮可以固定 terminalv4，跑 seed/repeat 扩展，确认 risky_ps_linear 是否稳定优于 direct；如果稳定，问题就转向 PS 共享/迁移策略，而不是 terminal cost 低罚。
