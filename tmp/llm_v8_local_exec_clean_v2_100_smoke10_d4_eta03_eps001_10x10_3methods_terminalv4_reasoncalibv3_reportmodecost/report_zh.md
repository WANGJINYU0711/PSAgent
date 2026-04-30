# C: terminalv4 + reasoning calibration v3 + report-only modecost

## 实验名

`llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost`

## 实验设置

- 数据集：`data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- 10x10 bucket：`analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- 方法：`direct_multistage_exp3`, `epsilon_exp3`, `risky_ps_linear`
- LLM：`gpt-4o-mini`
- executor：`llm_bench`
- repeats/horizon：`10 / 100`
- schedule：`trap_switch`, `switch_denominator=4`, switch episode = 25
- policy 参数：`eta=0.3`, `epsilon=0.01`
- 并行方式：tmux 外层守护 + runner 内部 method-level 并行
- tmux session：`llm_v8_c_reasoncalib`，已自然完成退出

## C 版改动

- 保留 A 版 `terminalv4`。
- 开启 reasoning weight calibration v3：
  - mode 匹配倍率：`0.85 -> 0.70`
  - deep-required 但 actual fast：`1.35 -> 1.55`
  - fast-required 但 actual deep：`1.15 -> 1.25`
- 开启 report-only modecost：
  - fast-on-deep report cost：`+1.5/stage`
  - deep-on-fast report cost：`+0.5/stage`
  - 不加入 `raw_total_cost`，只进入报告字段 `raw_mode_mismatch_cost_component`

已抽查 episode：`raw_total_cost = terminal + calibrated reasoning + path`，`mode_mismatch_cost_enabled=False`，`mode_mismatch_report_only_enabled=True`。

## Schedule 组成

| phase | n | requirement |
|---|---:|---|
| `trap_pre_switch` | 25 | 全部 `mostly_fast_required` |
| `target_post_switch` | 75 | 全部 `mostly_deep_required` |

| phase | oracle | n |
|---|---|---:|
| `trap_pre_switch` | `repair_all` | 25 |
| `target_post_switch` | `repair_all` | 51 |
| `target_post_switch` | `repair_subset` | 24 |

## C 总体结果

| method | total | terminal | legacy terminal | reasoning | modecost report | terminal share | clear | aux | strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_multistage_exp3` | 9.53 | 4.91 | 3.04 | 4.55 | 1.34 | 52% | 73% | 78% | 64% |
| `risky_ps_linear` | 9.70 | 5.16 | 3.43 | 4.47 | 1.30 | 54% | 70% | 78% | 65% |
| `epsilon_exp3` | 10.50 | 5.61 | 3.38 | 4.82 | 1.81 | 54% | 67% | 73% | 62% |

C 版总体上 `direct_multistage_exp3` 第一，`risky_ps_linear` 很接近，差距只有约 `0.17`。

## C post/deep-required 结果

| method | total | terminal | legacy terminal | reasoning | modecost report | terminal share | clear | aux | strict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `direct_multistage_exp3` | 10.80 | 6.41 | 4.02 | 4.31 | 1.30 | 60% | 65% | 71% | 53% |
| `risky_ps_linear` | 11.06 | 6.75 | 4.53 | 4.25 | 1.22 | 61% | 61% | 71% | 55% |
| `epsilon_exp3` | 11.84 | 7.47 | 4.51 | 4.29 | 1.83 | 64% | 56% | 64% | 49% |

这个 split 正好落在你想要的 6:4 附近：terminal share 是 60%-64%。相比 B 版，C 没有让 reasoning/mode penalty 吞掉 terminal；相比 A 版，C 对 mode mismatch 的路线选择有更强的间接压力。

## C: post deep-required 的 path/task 分布

| method | path/task pair | n | terminal | legacy | reasoning | modecost report | total | clear | aux | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| direct | deep-on-deep | 66 | 5.96 | 3.95 | 4.20 | 0.99 | 10.23 | 68% | 74% | 56% |
| direct | fast-on-deep | 9 | 9.72 | 4.50 | 5.19 | 3.56 | 14.97 | 44% | 44% | 33% |
| epsilon | deep-on-deep | 56 | 6.53 | 4.04 | 4.02 | 1.21 | 10.62 | 62% | 71% | 54% |
| epsilon | fast-on-deep | 19 | 10.26 | 5.89 | 5.10 | 3.68 | 15.42 | 37% | 42% | 37% |
| risky | deep-on-deep | 67 | 6.24 | 4.28 | 4.21 | 0.99 | 10.52 | 64% | 73% | 57% |
| risky | fast-on-deep | 8 | 11.00 | 6.62 | 4.54 | 3.19 | 15.61 | 38% | 50% | 38% |

这张表最关键：C 版里 fast-on-deep 已经明显比 deep-on-deep 贵，且 clear/aux/strict 都更差。这个方向符合你的目标，而且 modecost 只是报告字段，不是额外罚项。

## A/B/C 对比结论

| run | split | winner | 观察 |
|---|---|---|---|
| A `terminalv4` | all/post | risky | terminal floor 修正了低罚，但 risky 成功率最高 |
| B `terminalv4 + modecostv2` | all/post | direct | fast-on-deep 被强力推贵，但 reasoning 占比过高 |
| C `terminalv4 + reasoncalib + report modecost` | all/post | direct | terminal/reasoning 比例回到约 6:4，同时 fast-on-deep 仍明显贵 |

C 是目前三版里最平衡的一版：

- 不像 A 那样仍可能让部分 fast-on-deep 看起来偏便宜。
- 不像 B 那样把 reasoning/mode penalty 加得太重。
- post split 的 terminal share 是 60%-64%，最贴近 6:4。
- direct 在 all 和 post 都第一，但 risky 非常接近，需要更大 repeat/seed 验证稳定性。

## 问题与解决建议

1. `direct` 和 `risky` 差距很小。

   C 版 all split：direct `9.53`，risky `9.70`；post split：direct `10.80`，risky `11.06`。这个差距不够大，可能受 LLM stochasticity 和 10x10 sample 影响。

   建议：下一步不要继续大改 cost，先用 C 配置跑更大的 confirmatory run，例如 20x10 或换 seed 的 10x10。

2. epsilon 的 fast-on-deep 比例仍偏高。

   epsilon post deep-required 中 fast-on-deep 是 `19/75`，direct 是 `9/75`，risky 是 `8/75`。C 的校准已经让 fast-on-deep 变贵，但 epsilon 仍探索到较多 fast-on-deep。

   建议：如果要专门压 epsilon，可以降低 epsilon 或在 post-switch 后做 epsilon decay；但这会改变算法行为，建议先不要动，先确认 C objective 稳定。

3. C 的 mode exact 仍不高。

   A/B/C 的 exact stage mode match 都不高，C all split 大约 15%-18%。不过 majority-level 已经有效：post deep-required 中 deep-on-deep 占主流，fast-on-deep 明显更少。

   建议：报告主指标用 majority pair，不要强行追逐 per-stage exact mode match；per-stage exact 更像诊断，不适合作为主胜负标准。

## 产物

- C run：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/`
- C analysis：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/report.md`
- C episode compact：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/episode_cost_success_mode_compact.csv`
- C majority SVG：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/majority_pair_cost_matrix.svg`
- C phase SVG：`../llm_v8_local_exec_clean_v2_100_smoke10_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/phase_majority_pair_cost_table.svg`
- A/B/C compare：`../llm_v8_local_exec_clean_v2_100_smoke10_compare_abc/summary_compare_abc.md`
- A/B/C compare SVG：`../llm_v8_local_exec_clean_v2_100_smoke10_compare_abc/summary_compare_abc.svg`
