# Confirmatory seed1 vs C seed0 对比报告

## 实验名

`llm_v8_confirm_seed1_cconfig_d4_eta03_eps001_10x10_compare_with_c_seed0`

本报告比较：

- seed1 confirmatory：`risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`
- seed0 C 版 3-method：`direct_multistage_exp3`, `epsilon_exp3`, `risky_ps_linear`
- seed0 PS-family C 版中的 `risky_ps`

三者使用同一个 clean v2 数据集、同一个 10x10 bucket、同一个 C cost 配置、同一个 `switch_denominator=4`。主要区别是 seed 和方法集合。

## all split 总表

| rank | run | seed | method | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PSfamily_seed0 | 0 | `risky_ps` | 9.01 | 4.50 | 3.06 | 4.44 | 1.28 | 50% | 77% | 77% | 70% | 22% |
| 2 | C_seed0_3methods | 0 | `direct_multistage_exp3` | 9.53 | 4.91 | 3.04 | 4.55 | 1.34 | 52% | 73% | 78% | 64% | 18% |
| 3 | confirm_seed1 | 1 | `epsilon_exp3` | 9.61 | 5.03 | 3.15 | 4.51 | 1.49 | 52% | 72% | 75% | 64% | 31% |
| 4 | C_seed0_3methods | 0 | `risky_ps_linear` | 9.70 | 5.16 | 3.43 | 4.47 | 1.30 | 53% | 70% | 78% | 65% | 17% |
| 5 | confirm_seed1 | 1 | `direct_multistage_exp3` | 9.90 | 5.12 | 3.18 | 4.71 | 1.82 | 52% | 71% | 76% | 63% | 20% |
| 6 | C_seed0_3methods | 0 | `epsilon_exp3` | 10.50 | 5.61 | 3.38 | 4.82 | 1.81 | 53% | 67% | 73% | 62% | 15% |
| 7 | confirm_seed1 | 1 | `risky_ps` | 10.50 | 5.99 | 3.94 | 4.45 | 1.67 | 57% | 65% | 70% | 62% | 18% |

## post split 总表

| rank | run | seed | method | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PSfamily_seed0 | 0 | `risky_ps` | 10.37 | 6.00 | 4.08 | 4.30 | 1.22 | 58% | 69% | 69% | 60% | 29% |
| 2 | C_seed0_3methods | 0 | `direct_multistage_exp3` | 10.80 | 6.41 | 4.02 | 4.31 | 1.30 | 59% | 65% | 71% | 53% | 24% |
| 3 | C_seed0_3methods | 0 | `risky_ps_linear` | 11.06 | 6.75 | 4.53 | 4.25 | 1.22 | 61% | 61% | 71% | 55% | 23% |
| 4 | confirm_seed1 | 1 | `epsilon_exp3` | 11.11 | 6.71 | 4.19 | 4.34 | 1.45 | 60% | 63% | 67% | 52% | 41% |
| 5 | confirm_seed1 | 1 | `direct_multistage_exp3` | 11.15 | 6.58 | 4.18 | 4.50 | 1.86 | 59% | 63% | 68% | 52% | 27% |
| 6 | C_seed0_3methods | 0 | `epsilon_exp3` | 11.84 | 7.47 | 4.51 | 4.29 | 1.83 | 63% | 56% | 64% | 49% | 20% |
| 7 | confirm_seed1 | 1 | `risky_ps` | 12.44 | 7.98 | 5.26 | 4.39 | 1.70 | 64% | 53% | 60% | 49% | 24% |

## post deep-required path/task 对比

| run | seed | method | path/task pair | n | terminal | reasoning | modecost report | total | clear | aux | strict |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| confirm_seed1 | 1 | `epsilon_exp3` | deep-on-deep | 60 | 6.68 | 4.16 | 0.82 | 10.92 | 62% | 68% | 50% |
| confirm_seed1 | 1 | `epsilon_exp3` | fast-on-deep | 15 | 6.80 | 5.04 | 3.93 | 11.90 | 67% | 60% | 60% |
| confirm_seed1 | 1 | `direct_multistage_exp3` | deep-on-deep | 58 | 6.77 | 4.29 | 1.24 | 11.13 | 62% | 69% | 52% |
| confirm_seed1 | 1 | `direct_multistage_exp3` | fast-on-deep | 17 | 5.94 | 5.23 | 3.97 | 11.23 | 65% | 65% | 53% |
| confirm_seed1 | 1 | `risky_ps` | deep-on-deep | 61 | 7.36 | 4.23 | 1.17 | 11.66 | 57% | 64% | 52% |
| confirm_seed1 | 1 | `risky_ps` | fast-on-deep | 14 | 10.68 | 5.06 | 4.00 | 15.81 | 36% | 43% | 36% |
| PS_seed0 | 0 | `risky_ps` | deep-on-deep | 70 | 5.63 | 4.22 | 1.04 | 9.92 | 73% | 73% | 63% |
| PS_seed0 | 0 | `risky_ps` | fast-on-deep | 5 | 11.20 | 5.31 | 3.70 | 16.58 | 20% | 20% | 20% |
| C_seed0 | 0 | `direct_multistage_exp3` | deep-on-deep | 66 | 5.96 | 4.20 | 0.99 | 10.23 | 68% | 74% | 56% |
| C_seed0 | 0 | `direct_multistage_exp3` | fast-on-deep | 9 | 9.72 | 5.19 | 3.56 | 14.97 | 44% | 44% | 33% |
| C_seed0 | 0 | `epsilon_exp3` | deep-on-deep | 56 | 6.53 | 4.02 | 1.21 | 10.62 | 62% | 71% | 54% |
| C_seed0 | 0 | `epsilon_exp3` | fast-on-deep | 19 | 10.26 | 5.10 | 3.68 | 15.42 | 37% | 42% | 37% |
| C_seed0 | 0 | `risky_ps_linear` | deep-on-deep | 67 | 6.24 | 4.21 | 0.99 | 10.52 | 64% | 73% | 57% |
| C_seed0 | 0 | `risky_ps_linear` | fast-on-deep | 8 | 11.00 | 4.54 | 3.19 | 15.61 | 38% | 50% | 38% |

## 结论

1. 这次 confirmatory seed1 没有支持 PS 稳定第一。

   seed0 的 `risky_ps` 是 all/post 第一；seed1 中 `risky_ps` 是 all/post 第三。这个结果足够说明目前不能上全量。

2. C 配置仍然可用。

   terminal/reasoning 比例仍在合理范围；错误路线 fast-on-deep 仍然被 report modecost 和 reasoning calibration 诊断出来。问题不是 C cost 配置突然失真，而是 PS 的 seed robustness 不够。

3. seed1 暴露出的核心问题是 `risky_ps` 的 post/deep terminal instability。

   `risky_ps` 并没有少走 deep-on-deep，它的 deep-on-deep 数量是 `61/75`；但是 deep-on-deep terminal `7.36`，明显高于 seed0 的 `5.63`，也高于 seed1 epsilon/direct。

4. 下一步应该做 targeted diagnostic。

   不建议马上调 D2，也不建议马上改 reasoning weight。先比较 seed0/seed1 `risky_ps` 在相同 post task 上的 selected path、final_action、terminal floor reasons 和 shared update dynamics。

## 建议方案

1. 保留 C + d=4 作为主实验配置候选，但暂缓全量。
2. 做 `risky_ps` seed1 failure diagnostic：
   - 找出 post 中 `raw_terminal_penalty >= 14` 的 episodes。
   - 对比 seed0/seed1 同 `episode_index` 或同 `dataset_index` 的 selected path 与 final action。
   - 检查是否是 repair_subset 的 transfer/partial repair 错误集中爆发。
3. 如果是 PS shared update 方差问题，优先试：
   - `eta_shared=0.03` 或 `0.02`
   - 或 shared estimated loss clip/prob_floor
4. 如果是 executor terminal 的 repair_subset 不稳定，先修 terminal/executor diagnosis，再重跑 confirmatory。

## 产物

- 总表 CSV：`summary_compare_seed1_vs_c_seed0.csv`
- 总表 Markdown：`summary_compare_seed1_vs_c_seed0.md`
- 总表 SVG：`summary_compare_seed1_vs_c_seed0.svg`
- post pair CSV：`post_deep_required_pair_compare_seed1_vs_seed0.csv`
- post pair SVG：`post_deep_required_pair_compare_seed1_vs_seed0.svg`
