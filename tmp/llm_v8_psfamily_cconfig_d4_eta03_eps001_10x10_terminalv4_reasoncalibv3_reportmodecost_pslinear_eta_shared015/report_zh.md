# PS-family under C config: terminalv4 + reasoning calibration v3 + report-only modecost

## 实验名

`llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015`

## 实验设置

- 数据集：`data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- 10x10 bucket：`analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- 方法：原 13 方法中的全部 PS-family 方法：
  - `risky_ps_old`
  - `risky_ps`
  - `risky_ps_linear`
  - `risky_ps_ix`
  - `risky_ps_safe_conditional`
  - `risky_ps_safe_conditional_ix`
  - `risky_ps_direct_cost`
- LLM：`gpt-4o-mini`
- executor：`llm_bench`
- repeats/horizon：`10 / 100`
- schedule：`trap_switch`, `switch_denominator=4`, switch episode = 25
- policy 参数：`eta=0.3`, `epsilon=0.01`
- 特殊参数：`risky_ps_linear` 使用新参数 `eta_shared=0.15`
- C 配置：
  - `terminalv4`
  - `reasoning weight calibration v3`
  - `report-only modecost`
- 并行方式：tmux 外层守护 + 7 个 PS-family method 并行
- tmux session：`llm_v8_psfamily_c_d4`，已自然完成退出

`report-only modecost` 仍然只进报告字段 `raw_mode_mismatch_cost_component`，不加入 `raw_total_cost`。

## 运行状态

7 个 PS-family 方法全部完成 `100/100` episode，merge 和 mode analysis 已完成。当前没有残留 tmux session。

## PS-family 总体排名

| rank | method | total | terminal | legacy terminal | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `risky_ps` | 9.01 | 4.50 | 3.06 | 4.44 | 1.28 | 50% | 77% | 77% | 70% | 22% |
| 2 | `risky_ps_safe_conditional_ix` | 9.55 | 5.08 | 3.35 | 4.39 | 1.48 | 54% | 71% | 75% | 67% | 15% |
| 3 | `risky_ps_safe_conditional` | 9.58 | 5.02 | 3.40 | 4.49 | 1.46 | 53% | 73% | 73% | 67% | 18% |
| 4 | `risky_ps_ix` | 10.13 | 5.41 | 3.90 | 4.65 | 1.50 | 54% | 72% | 78% | 68% | 17% |
| 5 | `risky_ps_direct_cost` | 10.33 | 5.82 | 3.69 | 4.44 | 1.55 | 57% | 66% | 74% | 62% | 15% |
| 6 | `risky_ps_old` | 10.41 | 5.89 | 4.09 | 4.45 | 1.47 | 57% | 66% | 76% | 63% | 17% |
| 7 | `risky_ps_linear` | 11.19 | 6.64 | 4.47 | 4.47 | 1.44 | 60% | 62% | 75% | 59% | 21% |

核心结论：这次 PS-family 中第一名是 `risky_ps`，不是 `risky_ps_linear eta_shared=0.15`。`eta_shared=0.15` 这版 linear 明显变差，主要坏在 terminal quality，而不是 reasoning cost。

## PS-family post/deep-required 结果

| rank | method | total | terminal | legacy terminal | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `risky_ps` | 10.37 | 6.00 | 4.08 | 4.30 | 1.22 | 58% | 69% | 69% | 60% | 29% |
| 2 | `risky_ps_safe_conditional` | 11.09 | 6.69 | 4.53 | 4.32 | 1.45 | 61% | 64% | 64% | 56% | 24% |
| 3 | `risky_ps_safe_conditional_ix` | 11.15 | 6.78 | 4.47 | 4.30 | 1.48 | 61% | 61% | 67% | 56% | 20% |
| 4 | `risky_ps_ix` | 11.72 | 7.21 | 5.19 | 4.43 | 1.45 | 62% | 63% | 71% | 57% | 23% |
| 5 | `risky_ps_direct_cost` | 12.18 | 7.75 | 4.91 | 4.36 | 1.58 | 64% | 55% | 65% | 49% | 20% |
| 6 | `risky_ps_old` | 12.21 | 7.85 | 5.45 | 4.28 | 1.46 | 65% | 55% | 68% | 51% | 23% |
| 7 | `risky_ps_linear` | 13.34 | 8.86 | 5.96 | 4.41 | 1.42 | 67% | 49% | 67% | 45% | 28% |

post split 中 `risky_ps` 依旧第一，terminal share 是 `58%`，仍然接近你想要的 `6:4`。这说明 C 版 cost 结构没有被 reasoning 吞掉，而且 post/deep 部分的胜负主要还是 terminal quality。

## post deep-required 的 path/task 分布

| method | path/task pair | n | terminal | reasoning | modecost report | total | clear | aux | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `risky_ps` | deep-on-deep | 70 | 5.63 | 4.22 | 1.04 | 9.92 | 73% | 73% | 63% |
| `risky_ps` | fast-on-deep | 5 | 11.20 | 5.31 | 3.70 | 16.58 | 20% | 20% | 20% |
| `risky_ps_safe_conditional` | deep-on-deep | 63 | 5.93 | 4.15 | 1.05 | 10.15 | 70% | 68% | 60% |
| `risky_ps_safe_conditional` | fast-on-deep | 12 | 10.71 | 5.24 | 3.54 | 16.02 | 33% | 42% | 33% |
| `risky_ps_safe_conditional_ix` | deep-on-deep | 68 | 6.41 | 4.25 | 1.25 | 10.73 | 63% | 69% | 57% |
| `risky_ps_safe_conditional_ix` | fast-on-deep | 7 | 10.36 | 4.80 | 3.71 | 15.23 | 43% | 43% | 43% |
| `risky_ps_linear` | deep-on-deep | 64 | 8.77 | 4.31 | 0.99 | 13.16 | 50% | 70% | 45% |
| `risky_ps_linear` | fast-on-deep | 11 | 9.36 | 5.01 | 3.91 | 14.43 | 45% | 45% | 45% |

最重要的是 `risky_ps`：post/deep-required 里 deep-on-deep 有 `70/75`，fast-on-deep 只有 `5/75`；deep-on-deep total `9.92`，fast-on-deep total `16.58`。这正是你想验证的方向：path 内 agent 大多 deep 且任务要求 deep 时 cost 明显更低，fast path 去做 deep 任务时 terminal/reasoning/total 都更高，clear/aux/strict 也明显更差。

`risky_ps_linear eta_shared=0.15` 的异常点也很清楚：它虽然仍多数 deep-on-deep，但 deep-on-deep terminal 高到 `8.77`，导致 post total `13.34`。因此这版不是“更会利用 shared suffix”，而是把 target/deep local repair 的 terminal quality 弄差了。

## 问题分析

1. `risky_ps` 已经达到目标：在 C 配置、d=4、公平同 schedule 下 PS 第一。

   all split：`risky_ps 9.01`，优于旧 C direct 的 `9.53` 和 epsilon 的 `10.50`。post split：`risky_ps 10.37`，优于旧 C direct 的 `10.80` 和 epsilon 的 `11.84`。

2. `risky_ps_linear eta_shared=0.15` 不建议继续作为主线。

   它不是 reasoning 更高，而是 terminal 明显更差：all terminal `6.64`，post terminal `8.86`，post strict 只有 `45%`。如果下一步正式实验要压缩候选，应该保留 `risky_ps`，备选保留 `risky_ps_safe_conditional` 和 `risky_ps_safe_conditional_ix`，不要用这版 linear 作为主 PS。

3. 个别旧 PS 变体的 pair 表不适合当主结论。

   例如 `risky_ps_old` 的 fast-on-deep 小样本表现不稳定，甚至 total 低于 deep-on-deep。这不是 C objective 的主问题，而是旧变体和小样本选择偏差叠在一起的结果。主报告应以 winner `risky_ps` 和 C direct/epsilon 的同配置对比为准。

4. D2 暂时不建议执行。

   你的判断是对的：`switch_denominator=4` 可能更利于 PS 相对 direct/epsilon 胜出，因为 pre-switch trap 足够形成误导，但 post-switch 仍有 75 个 target episode 让 shared suffix 传播。把 post 拉到 80 或 90 会改变实验变量，而且也可能给 direct/epsilon 更多 post feedback 来恢复。现在 d=4 已经跑出 PS 第一，下一步应先固定 d=4 做 confirmatory smoke，而不是马上改 D2。

## 建议下一步

1. 固定 C 配置和 `switch_denominator=4`，用 `risky_ps` 作为主 PS。
2. 做一个 confirmatory smoke：
   - 方法：`risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`
   - 可选加：`risky_ps_safe_conditional`, `risky_ps_safe_conditional_ix`
   - 配置不变，只换 seed 或增大 repeat。
3. 如果 confirmatory 仍然是 `risky_ps` 第一，再进入全量实验。
4. D2 只作为 robustness check，不作为当前主线。它回答的是“post target 占比变化是否稳定”，不是“C 配置下 PS 是否能第一”。

## 产物

- PS-family run：`tmp/llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015/`
- PS-family 英文报告：`report.md`
- PS-family 中文报告：`report_zh.md`
- episode compact：`episode_cost_success_mode_compact.csv`
- majority SVG：`majority_pair_cost_matrix.svg`
- phase SVG：`phase_majority_pair_cost_table.svg`
- 合并 C direct/epsilon 对比目录：`../llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_compare_with_c_exp_eps/`
