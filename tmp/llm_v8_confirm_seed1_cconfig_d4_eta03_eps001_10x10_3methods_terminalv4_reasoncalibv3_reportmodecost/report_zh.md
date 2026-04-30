# Confirmatory seed1: C config d4, risky_ps vs direct vs epsilon

## 实验名

`llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost`

## 实验设置

- 数据集：`data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- 10x10 bucket：`analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- 方法：`risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`
- LLM：`gpt-4o-mini`
- executor：`llm_bench`
- repeats/horizon：`10 / 100`
- schedule：`trap_switch`, `switch_denominator=4`, switch episode = 25
- policy 参数：`eta=0.3`, `epsilon=0.01`
- seed：`1`
- C 配置：
  - `terminalv4`
  - `reasoning weight calibration v3`
  - `report-only modecost`
- 并行方式：tmux 外层守护 + runner method-level 并行
- tmux session：`llm_v8_confirm_c_seed1`，已自然完成退出

为了只换 seed，我给 `scripts/run_shared_basin_repeated_smoke.py` 加了一个向后兼容的 seed 环境变量覆盖：`PSAGENT_REPEATED_SMOKE_SEED`。默认仍是 `0`，本次 run 使用 `PSAGENT_REPEATED_SMOKE_SEED=1`，run_config 已记录 `seed=1`。

运行中第一次启动在真正跑 episode 前失败，原因是 shell 里缺少 `PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini`。补齐该环境变量后重新启动；实际 seed1 run 三个方法全部完成 `100/100`，exit code 全部为 0。

## Schedule 组成

| phase | oracle | n |
|---|---|---:|
| `trap_pre_switch` | `repair_all` | 25 |
| `target_post_switch` | `repair_all` | 51 |
| `target_post_switch` | `repair_subset` | 24 |

## seed1 总体结果

| rank | method | total | terminal | legacy terminal | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `epsilon_exp3` | 9.61 | 5.03 | 3.15 | 4.51 | 1.49 | 52% | 72% | 75% | 64% | 31% |
| 2 | `direct_multistage_exp3` | 9.90 | 5.12 | 3.18 | 4.71 | 1.82 | 52% | 71% | 76% | 63% | 20% |
| 3 | `risky_ps` | 10.50 | 5.99 | 3.94 | 4.45 | 1.67 | 57% | 65% | 70% | 62% | 18% |

seed1 confirmatory 没有复现 seed0 的 `risky_ps` 第一。seed1 中 `epsilon_exp3` 第一，`direct_multistage_exp3` 第二，`risky_ps` 第三。

## seed1 post/deep-required 结果

| rank | method | total | terminal | legacy terminal | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `epsilon_exp3` | 11.11 | 6.71 | 4.19 | 4.34 | 1.45 | 60% | 63% | 67% | 52% | 41% |
| 2 | `direct_multistage_exp3` | 11.15 | 6.58 | 4.18 | 4.50 | 1.86 | 59% | 63% | 68% | 52% | 27% |
| 3 | `risky_ps` | 12.44 | 7.98 | 5.26 | 4.39 | 1.70 | 64% | 53% | 60% | 49% | 24% |

post split 中 `risky_ps` 的 reasoning 并不高，甚至低于 direct；真正拉开差距的是 terminal。`risky_ps` 比 epsilon 的 post terminal 高 `+1.27`，比 direct 高 `+1.40`。

## seed1 post deep-required 的 path/task 分布

| method | path/task pair | n | terminal | reasoning | modecost report | total | clear | aux | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `epsilon_exp3` | deep-on-deep | 60 | 6.68 | 4.16 | 0.82 | 10.92 | 62% | 68% | 50% |
| `epsilon_exp3` | fast-on-deep | 15 | 6.80 | 5.04 | 3.93 | 11.90 | 67% | 60% | 60% |
| `direct_multistage_exp3` | deep-on-deep | 58 | 6.77 | 4.29 | 1.24 | 11.13 | 62% | 69% | 52% |
| `direct_multistage_exp3` | fast-on-deep | 17 | 5.94 | 5.23 | 3.97 | 11.23 | 65% | 65% | 53% |
| `risky_ps` | deep-on-deep | 61 | 7.36 | 4.23 | 1.17 | 11.66 | 57% | 64% | 52% |
| `risky_ps` | fast-on-deep | 14 | 10.68 | 5.06 | 4.00 | 15.81 | 36% | 43% | 36% |

这张表说明 seed1 的问题不是 `risky_ps` 没走 deep。`risky_ps` 的 deep-on-deep 是 `61/75`，和 epsilon 的 `60/75`、direct 的 `58/75` 差不多，甚至略高。问题是 `risky_ps` 在 deep-on-deep 上 terminal 明显更高，clear/aux 更低；fast-on-deep 的少量错误路线也非常贵。

## seed0 vs seed1 对比结论

| run | seed | winner all | winner post | 观察 |
|---|---:|---|---|---|
| PS-family C smoke | 0 | `risky_ps` | `risky_ps` | `risky_ps` all `9.01`，post `10.37`，PS 明确第一 |
| confirmatory C smoke | 1 | `epsilon_exp3` | `epsilon_exp3` | `risky_ps` all `10.50`，post `12.44`，PS 第三 |

因此目前不能上全量实验。C 配置本身仍然合理，但 `risky_ps` 的 PS-first 结论在 10x10 LLM smoke 上对 seed 敏感。

## 问题分析

1. PS-first 不稳定。

   seed0 中 `risky_ps` 第一；seed1 中 `risky_ps` 第三。这说明当前证据还不能支持正式全量。全量前至少需要一个小型 multi-seed confirmatory，或者先解决 seed1 暴露出的 terminal instability。

2. seed1 的主要失败点是 terminal，不是 reasoning。

   post split 中 `risky_ps` reasoning 是 `4.39`，epsilon 是 `4.34`，direct 是 `4.50`，三者接近；但 `risky_ps` terminal 是 `7.98`，epsilon 是 `6.71`，direct 是 `6.58`。所以不要继续只调 reasoning weight，应该检查 PS update 导致的 path/leaf feedback 是否把一些高 terminal repair_subset/transfer 错误放大了。

3. route majority 不是主问题。

   `risky_ps` 在 post/deep-required 中 deep-on-deep 有 `61/75`，不比 baseline 少。它输在 deep-on-deep 的 terminal quality：`7.36` 高于 epsilon 的 `6.68` 和 direct 的 `6.77`。

4. fast-on-deep 仍然被 C 配置正确拉贵。

   seed1 中 `risky_ps` fast-on-deep total `15.81`，strict `36%`；deep-on-deep total `11.66`，strict `52%`。方向仍然对，但 PS 的 deep-on-deep 质量不够稳。

## 解决建议

1. 暂停全量，不要把 seed0 的 `risky_ps` 第一当作稳定结论。

2. 下一步优先做 seed1 diagnostic，而不是大改 C cost：
   - 对 `risky_ps` seed1 的 post episodes 找出 terminal >= 14 或 final_action=transfer 的 local repair cases。
   - 看这些高罚 episode 是否集中在某些 repair_subset task 或某些 shared suffix。
   - 对比 seed0 `risky_ps` 同 dataset_index/position 的 selected path 和 final_action。

3. 如果确认是 shared update 被少数高 loss episode 推歪，可以试两个小改动之一：
   - 降低 `eta_shared`，例如 `0.03` 或 `0.02`，只跑 `risky_ps` vs direct/epsilon smoke。
   - 给 shared estimated loss 加 clip/prob_floor，降低 full-path importance weighting 的极端方差。

4. 如果确认是 repair_subset terminal 本身不稳定，则先修 executor/terminal diagnosis，而不是调算法。

5. `switch_denominator=4` 暂时仍保留。

   seed1 失败不是因为 post 不够长，也不是因为 route 不够 deep；先不要同时引入 D2，否则会把 seed sensitivity 和 schedule change 混在一起。

## 产物

- confirmatory run：`tmp/llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_3methods_terminalv4_reasoncalibv3_reportmodecost/`
- 英文报告：`report.md`
- 中文报告：`report_zh.md`
- episode compact：`episode_cost_success_mode_compact.csv`
- majority SVG：`majority_pair_cost_matrix.svg`
- phase SVG：`phase_majority_pair_cost_table.svg`
- seed0/seed1 compare：`../llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_compare_with_c_seed0/`
