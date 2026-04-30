# C direct/epsilon vs PS-family C-config 对比报告

## 实验名

`llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_compare_with_c_exp_eps`

本报告把本次 PS-family run 和之前已完成的 C 版 `direct_multistage_exp3` / `epsilon_exp3` 放在同一张表里比较。两边使用同一个 clean v2 数据集、同一个 10x10 bucket、同一个 C cost 配置、同一个 `switch_denominator=4`。

## 总体总表

| rank | run | method | split | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PS_family_C | `risky_ps` | all | 9.01 | 4.50 | 3.06 | 4.44 | 1.28 | 50% | 77% | 77% | 70% | 22% |
| 2 | C_baseline | `direct_multistage_exp3` | all | 9.53 | 4.91 | 3.04 | 4.55 | 1.34 | 52% | 73% | 78% | 64% | 18% |
| 3 | PS_family_C | `risky_ps_safe_conditional_ix` | all | 9.55 | 5.08 | 3.35 | 4.39 | 1.48 | 54% | 71% | 75% | 67% | 15% |
| 4 | PS_family_C | `risky_ps_safe_conditional` | all | 9.58 | 5.02 | 3.40 | 4.49 | 1.46 | 53% | 73% | 73% | 67% | 18% |
| 5 | PS_family_C | `risky_ps_ix` | all | 10.13 | 5.41 | 3.90 | 4.65 | 1.50 | 54% | 72% | 78% | 68% | 17% |
| 6 | PS_family_C | `risky_ps_direct_cost` | all | 10.33 | 5.82 | 3.69 | 4.44 | 1.55 | 57% | 66% | 74% | 62% | 15% |
| 7 | PS_family_C | `risky_ps_old` | all | 10.41 | 5.89 | 4.09 | 4.45 | 1.47 | 57% | 66% | 76% | 63% | 17% |
| 8 | C_baseline | `epsilon_exp3` | all | 10.50 | 5.61 | 3.38 | 4.82 | 1.81 | 54% | 67% | 73% | 62% | 15% |
| 9 | PS_family_C | `risky_ps_linear` | all | 11.19 | 6.64 | 4.47 | 4.47 | 1.44 | 60% | 62% | 75% | 59% | 21% |

结论：`risky_ps` 是 all split 第一，领先 C direct `0.52`，领先 epsilon `1.49`。

## post/deep-required 总表

| rank | run | method | split | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PS_family_C | `risky_ps` | post | 10.37 | 6.00 | 4.08 | 4.30 | 1.22 | 58% | 69% | 69% | 60% | 29% |
| 2 | C_baseline | `direct_multistage_exp3` | post | 10.80 | 6.41 | 4.02 | 4.31 | 1.30 | 60% | 65% | 71% | 53% | 24% |
| 3 | PS_family_C | `risky_ps_safe_conditional` | post | 11.09 | 6.69 | 4.53 | 4.32 | 1.45 | 61% | 64% | 64% | 56% | 24% |
| 4 | PS_family_C | `risky_ps_safe_conditional_ix` | post | 11.15 | 6.78 | 4.47 | 4.30 | 1.48 | 61% | 61% | 67% | 56% | 20% |
| 5 | PS_family_C | `risky_ps_ix` | post | 11.72 | 7.21 | 5.19 | 4.43 | 1.45 | 62% | 63% | 71% | 57% | 23% |
| 6 | C_baseline | `epsilon_exp3` | post | 11.84 | 7.47 | 4.51 | 4.29 | 1.83 | 64% | 56% | 64% | 49% | 20% |
| 7 | PS_family_C | `risky_ps_direct_cost` | post | 12.18 | 7.75 | 4.91 | 4.36 | 1.58 | 64% | 55% | 65% | 49% | 20% |
| 8 | PS_family_C | `risky_ps_old` | post | 12.21 | 7.85 | 5.45 | 4.28 | 1.46 | 65% | 55% | 68% | 51% | 23% |
| 9 | PS_family_C | `risky_ps_linear` | post | 13.34 | 8.86 | 5.96 | 4.41 | 1.42 | 67% | 49% | 67% | 45% | 28% |

结论：`risky_ps` 是 post/deep-required 第一，领先 C direct `0.43`，领先 epsilon `1.47`。这正好满足“在公平 C 配置正确时，PS 至少在 smoke 上第一”的门槛。

## 三个核心方法对比

| split | method | total | terminal | reasoning | modecost report | terminal share | clear | aux | strict |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| all | `risky_ps` | 9.01 | 4.50 | 4.44 | 1.28 | 50% | 77% | 77% | 70% |
| all | `direct_multistage_exp3` | 9.53 | 4.91 | 4.55 | 1.34 | 52% | 73% | 78% | 64% |
| all | `epsilon_exp3` | 10.50 | 5.61 | 4.82 | 1.81 | 54% | 67% | 73% | 62% |
| post | `risky_ps` | 10.37 | 6.00 | 4.30 | 1.22 | 58% | 69% | 69% | 60% |
| post | `direct_multistage_exp3` | 10.80 | 6.41 | 4.31 | 1.30 | 60% | 65% | 71% | 53% |
| post | `epsilon_exp3` | 11.84 | 7.47 | 4.29 | 1.83 | 64% | 56% | 64% | 49% |

`risky_ps` 的优势不是靠 reasoning 少很多，而是 terminal 更低、clear/strict 更高。post split 中三者 reasoning 都在 `4.29-4.31` 左右，差异几乎全来自 terminal quality。

## post deep-required 的 path/task 分布

| run | method | path/task pair | n | terminal | reasoning | modecost report | total | clear | aux | strict |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PS_family_C | `risky_ps` | deep-on-deep | 70 | 5.63 | 4.22 | 1.04 | 9.92 | 73% | 73% | 63% |
| PS_family_C | `risky_ps` | fast-on-deep | 5 | 11.20 | 5.31 | 3.70 | 16.58 | 20% | 20% | 20% |
| C_baseline | `direct_multistage_exp3` | deep-on-deep | 66 | 5.96 | 4.20 | 0.99 | 10.23 | 68% | 74% | 56% |
| C_baseline | `direct_multistage_exp3` | fast-on-deep | 9 | 9.72 | 5.19 | 3.56 | 14.97 | 44% | 44% | 33% |
| C_baseline | `epsilon_exp3` | deep-on-deep | 56 | 6.53 | 4.02 | 1.21 | 10.62 | 62% | 71% | 54% |
| C_baseline | `epsilon_exp3` | fast-on-deep | 19 | 10.26 | 5.10 | 3.68 | 15.42 | 37% | 42% | 37% |

这张表支持你的分析目标：

- deep-on-deep 比 fast-on-deep 明显便宜。
- `risky_ps` 在 post/deep-required 中选择 deep-on-deep 的比例最高：`70/75 = 93.3%`。
- `risky_ps` 的 deep-on-deep terminal `5.63` 低于 C direct 的 `5.96` 和 epsilon 的 `6.53`。
- `risky_ps` 的 fast-on-deep 只有 `5/75`，但 cost 极高、成功率极低；这说明 C 版 cost 对错误路线有足够惩罚。

## 对 D2 的判断

你说的 `switch_denominator=4` 更适合 PS 胜过 exp/eps 是合理的，当前结果也支持这个判断。d=4 的结构是 `25 pre + 75 post`，pre 足够形成 trap，但 post 又足够长，让 PS 利用 target/shared suffix 做传播。

D2 增加 post 占比当然公平，因为所有方法看到同样 schedule；但它会改变实验问题。`switch_denominator=5` 或 `10` 不只是“更多 target”，也会减少 pre trap 压力，并给 direct/epsilon 更多 post 反馈恢复机会。既然本次 d=4 已经让 `risky_ps` 第一，D2 现在不应该和 D1 混在一起执行。建议把 D2 放到后面的 robustness section。

## 目前最合理的新方案

1. 主配置固定为 C：
   - `terminalv4`
   - `reasoning calibration v3`
   - `report-only modecost`
   - `switch_denominator=4`
   - `eta=0.3`, `epsilon=0.01`

2. 主 PS 方法用 `risky_ps`。

   这是本次唯一同时在 all 和 post split 明确第一的 PS 方法。`risky_ps_linear eta_shared=0.15` 不作为主线。

3. 下一步先做 confirmatory smoke，不直接上全量。

   建议方法集：`risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`，可加 `risky_ps_safe_conditional` 和 `risky_ps_safe_conditional_ix` 作 PS robustness。配置保持 C/d=4，只换 seed 或加 repeats。

4. 如果 confirmatory 仍是 `risky_ps` 第一，再上全量正式实验。

5. D2 只作为附录鲁棒性实验。

   它可以回答“post target 更长时 PS 是否仍稳定”，但不应替代当前主线。

## 产物

- 合并总表：`summary_compare_c_exp_eps_vs_psfamily.md`
- 合并 SVG：`summary_compare_c_exp_eps_vs_psfamily.svg`
- post deep-required pair CSV：`post_deep_required_pair_compare.csv`
- post deep-required pair SVG：`post_deep_required_pair_compare.svg`
- PS-family run 目录：`../llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015/`
- PS-family episode compact：`../llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015/episode_cost_success_mode_compact.csv`
- PS-family majority SVG：`../llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015/majority_pair_cost_matrix.svg`
- PS-family phase SVG：`../llm_v8_psfamily_cconfig_d4_eta03_eps001_10x10_terminalv4_reasoncalibv3_reportmodecost_pslinear_eta_shared015/phase_majority_pair_cost_table.svg`
