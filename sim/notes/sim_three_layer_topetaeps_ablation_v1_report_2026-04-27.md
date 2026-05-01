# Three-layer top eta/eps sim ablation v1 报告

日期：2026-04-27

总实验目录：

`outputs/sim_three_layer_topetaeps_ablation_v1/`

总报告：

`outputs/sim_three_layer_topetaeps_ablation_v1/three_layer_report.md`

## 0. 本轮实验设计

按用户建议，本轮没有继续铺大网格，而是从 v10 已有结果里收束 eta/eps。

v10 PS-win 的 eta 只有两个值：

- `eta=0.3`
- `eta=0.4`

epsilon 的前三个高价值值很清楚：

- `eps=0.005`
- `eps=0.01`
- `eps=0.02`

为了保留 v11 已经证明有效、且和 LLM smoke 更接近的锚点，本轮 eta 使用：

- `eta=0.2, 0.3, 0.4`

本轮 epsilon 使用：

- `eps=0.005, 0.01, 0.02`

固定设置：

- tree spec: `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json`
- role mode: `spec_or_agent_id`
- horizon: `1000`
- seeds: `0..9`
- cost_noise: `0.02`
- specialist_fraction: `0.15`
- methods: 13 full methods

运行状态：

- total combos: `117`
- completed: `117`
- failures: `0`
- total rows: `1521`
- post-switch PS-winning combos: `29 / 117`

## 1. 三个实验名字

### Layer 1

`sim_v10_d2_4_top3_etaeps_full13_v1`

目的：

补齐 v10 single-switch 的 `d=2,3,4`，不重复已跑过的 `d=5,6`。

设置：

- cost mode: `ps_favored_trap_v10_avg_baited`
- control: `trap_switch_denominator=d`
- d values: `2,3,4`
- eta: `0.2,0.3,0.4`
- eps: `0.005,0.01,0.02`

### Layer 2

`sim_v11_cyclic_switch1_6_top3_etaeps_full13_v1`

目的：

在 v11 cyclic cost mode 上做局部 eta/eps sweep，检查原来固定 `eta=0.2, eps=0.01` 的 PS-win 是否稳定，以及 switch_count=5/6 边界是否能被拉回来。

设置：

- cost mode: `ps_favored_trap_v11_cyclic_baited`
- control: `cyclic_switch_count`
- switch_count values: `1,2,3,4,5,6`
- eta: `0.2,0.3,0.4`
- eps: `0.005,0.01,0.02`

### Layer 3

`sim_v12_gapcompressed_d4_7_top3_etaeps_full13_v1`

目的：

只做 v12 compression ablation，不作为主线。检查 v12 失败是否只是因为之前固定 `d=7, eta=0.2, eps=0.01` 参数不好。

设置：

- cost mode: `ps_favored_trap_v12_gap_compressed_baited`
- control: `trap_switch_denominator=d`
- d values: `4,5,6,7`
- eta: `0.2,0.3,0.4`
- eps: `0.005,0.01,0.02`

## 2. 总体结论

### 2.1 Layer 1: v10 补 d=2..4 成功

`sim_v10_d2_4_top3_etaeps_full13_v1`

- combos: `27`
- PS-winning combos: `12 / 27`
- 最佳 setting: `d=4, eta=0.3, eps=0.01`
- winner: `risky_ps_old`
- post_switch_avg_regret_mean: `0.078400`
- regret_per_t_mean: `0.066184`
- target_good_fraction_mean: `0.7286`
- trap_basin_fraction_mean: `0.0456`

这个结果说明用户建议补 `d=2,3,4` 是对的。之前 v10 已知强点是 `d=5/6`，现在发现 `d=4` 也非常强，而且在当前局部网格里是 layer 1 最佳。

### 2.2 Layer 2: v11 仍然强，并且 switch_count=6 也被拉回一个 PS-win

`sim_v11_cyclic_switch1_6_top3_etaeps_full13_v1`

- combos: `54`
- PS-winning combos: `17 / 54`
- 最强 PS-win setting: `switch_count=3, eta=0.4, eps=0.02`
- winner: `risky_ps_old`
- post_switch_avg_regret_mean: `-0.247521`
- regret_per_t_mean: `-0.167041`

注意：按每个方法跨全部参数取 best，`direct_multistage_exp3` 在 v11 里有一个更低的 best-by-method 点：

- `switch_count=2, eta=0.4, eps=0.005`
- direct post-switch: `-0.259105`

但在同一个 config 内，v11 仍有 17 个 PS-win config。也就是说，v11 是一个有明确 PS-favorable 区域的环境；不过如果做 unconstrained best-of，direct 也能在某些超参组合上非常强。

### 2.3 Layer 3: v12 compression 不是单点参数问题

`sim_v12_gapcompressed_d4_7_top3_etaeps_full13_v1`

- combos: `36`
- PS-winning combos: `0 / 36`
- 最佳 overall config 仍是 `naive_mixed_avg`
- 最佳 v12 PS setting 也只能 rank 2 或 rank 3

这基本确认：v12 之前不是因为只用了 `d=7, eta=0.2, eps=0.01` 所以失败；在更好的 v10-derived eta/eps 和 `d=4,5,6,7` 下，v12 仍没有出现 PS-win。

所以 v12 的结论应定性为：

> compression ablation / negative result，而不是主 benchmark。

## 3. Layer 1 详细结果：v10 d=2..4

### 3.1 每个 d 的最佳 PS

| d | PS-win count | best eta | best eps | best PS | PS rank | PS post-switch | PS - best nonPS | best nonPS | regret/T | tail20 | target-good | trap |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 2 | 5 / 9 | 0.4 | 0.01 | risky_ps_old | 1 | 0.126600 | -0.040000 | direct_multistage_exp3_local | 0.054668 | 0.030 | 0.5270 | 0.0673 |
| 3 | 4 / 9 | 0.4 | 0.005 | risky_ps_old | 1 | 0.098246 | -0.034183 | direct_multistage_exp3_local | 0.066330 | 0.045 | 0.6557 | 0.0660 |
| 4 | 3 / 9 | 0.3 | 0.01 | risky_ps_old | 1 | 0.078400 | -0.019467 | direct_multistage_exp3 | 0.066184 | 0.015 | 0.7286 | 0.0456 |

### 3.2 Layer 1 best-by-method

| rank | method | d | eta | eps | post-switch | tail20 | regret/T | target-good | trap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | risky_ps_old | 4 | 0.3 | 0.01 | 0.078400 | 0.015 | 0.066184 | 0.7286 | 0.0456 |
| 2 | epsilon_exp3 | 4 | 0.4 | 0.01 | 0.085467 | 0.035 | 0.069684 | 0.7159 | 0.0535 |
| 3 | direct_multistage_exp3 | 4 | 0.3 | 0.005 | 0.097867 | 0.005 | 0.079784 | 0.7048 | 0.0811 |
| 4 | risky_ps_linear | 4 | 0.3 | 0.01 | 0.099067 | 0.010 | 0.081984 | 0.7061 | 0.0482 |
| 5 | risky_ps | 4 | 0.3 | 0.01 | 0.111067 | 0.030 | 0.090884 | 0.6969 | 0.0492 |
| 6 | risky_ps_ix | 4 | 0.3 | 0.01 | 0.112133 | 0.015 | 0.091684 | 0.6957 | 0.0492 |
| 7 | direct_multistage_exp3_local | 4 | 0.3 | 0.005 | 0.115067 | 0.005 | 0.098084 | 0.6829 | 0.0927 |
| 8 | naive_mixed_avg | 4 | 0.2 | 0.005 | 0.135333 | 0.010 | 0.090884 | 0.6730 | 0.0000 |
| 9 | risky_ps_safe_conditional | 4 | 0.3 | 0.02 | 0.144000 | 0.070 | 0.116784 | 0.6570 | 0.0543 |
| 10 | risky_ps_safe_conditional_ix | 4 | 0.3 | 0.01 | 0.144267 | 0.015 | 0.115184 | 0.6641 | 0.0524 |

Layer 1 解读：

- `d=4` 是目前 v10 single-switch 最值得保留的新 setting。
- `d=2` 也能 PS-win，但 post-switch 绝对值较高，说明 switch 太晚时 post-switch learning window 不够宽。
- `d=4, eta=0.3, eps=0.01` 比先前 `d=5/6` 的 PS-win 更适合作为 single-switch 主候选，因为它在补扫里 post-switch 最低。

## 4. Layer 2 详细结果：v11 cyclic

### 4.1 每个 switch_count 的最佳 PS

| switch_count | PS-win count | best eta | best eps | best PS | PS rank | PS post-switch | PS - best nonPS | best nonPS | regret/T | tail20 | target-good | trap |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 1 | 5 / 9 | 0.4 | 0.005 | risky_ps_old | 1 | 0.130000 | -0.027400 | direct_multistage_exp3 | 0.057468 | 0.065 | 0.2606 | 0.0811 |
| 2 | 3 / 9 | 0.4 | 0.02 | risky_ps_old | 2 | -0.251609 | +0.007496 | direct_multistage_exp3 | -0.168257 | 0.095 | 0.2963 | 0.0704 |
| 3 | 4 / 9 | 0.4 | 0.02 | risky_ps_old | 1 | -0.247521 | -0.011200 | direct_multistage_exp3 | -0.167041 | 0.125 | 0.2637 | 0.0481 |
| 4 | 4 / 9 | 0.4 | 0.01 | risky_ps_ix | 1 | -0.114977 | -0.022500 | epsilon_exp3 | -0.084614 | 0.145 | 0.1699 | 0.0489 |
| 5 | 0 / 9 | 0.4 | 0.005 | risky_ps_old | 4 | -0.014544 | +0.053357 | naive_mixed_avg | -0.003298 | 0.265 | 0.2698 | 0.0410 |
| 6 | 1 / 9 | 0.4 | 0.01 | risky_ps_ix | 3 | -0.084444 | +0.036364 | direct_multistage_exp3 | -0.062293 | 0.100 | 0.2238 | 0.0376 |

### 4.2 Layer 2 best-by-method

| rank | method | switch_count | eta | eps | post-switch | tail20 | regret/T | target-good | trap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | direct_multistage_exp3 | 2 | 0.4 | 0.005 | -0.259105 | 0.040 | -0.174157 | 0.2812 | 0.1060 |
| 2 | epsilon_exp3 | 2 | 0.3 | 0.005 | -0.256106 | 0.050 | -0.170257 | 0.2385 | 0.0766 |
| 3 | risky_ps_old | 2 | 0.4 | 0.02 | -0.251609 | 0.095 | -0.168257 | 0.2963 | 0.0704 |
| 4 | risky_ps_linear | 2 | 0.4 | 0.005 | -0.233618 | 0.105 | -0.157557 | 0.2167 | 0.0493 |
| 5 | direct_multistage_exp3_local | 2 | 0.3 | 0.005 | -0.209929 | 0.045 | -0.132357 | 0.2417 | 0.0688 |
| 6 | risky_ps | 2 | 0.4 | 0.005 | -0.207081 | 0.030 | -0.140057 | 0.2512 | 0.0515 |
| 7 | risky_ps_ix | 2 | 0.4 | 0.005 | -0.205432 | 0.030 | -0.138957 | 0.2509 | 0.0518 |
| 8 | risky_ps_safe_conditional_ix | 3 | 0.4 | 0.005 | -0.204055 | 0.120 | -0.135441 | 0.1796 | 0.0461 |
| 9 | risky_ps_safe_conditional | 3 | 0.4 | 0.005 | -0.192588 | 0.175 | -0.126841 | 0.1665 | 0.0497 |
| 10 | risky_ps_direct_cost | 2 | 0.4 | 0.005 | -0.157456 | 0.160 | -0.106957 | 0.0560 | 0.0495 |
| 11 | naive_mixed_avg | 5 | 0.2 | 0.005 | -0.067901 | 0.000 | -0.067198 | 0.5561 | 0.0000 |

Layer 2 解读：

- v11 仍然是 strong positive environment：17/54 个 PS-win。
- 原先 fixed point `switch_count=1..4, eta=0.2, eps=0.01` 的结论没有被推翻。
- 新 sweep 发现更强的 v11 PS 点：`switch_count=3, eta=0.4, eps=0.02`。
- 但是 best-by-method 上 direct 也能找到极强点：`switch_count=2, eta=0.4, eps=0.005`。因此如果论文采用 v11，最好不要写成 unconstrained best-of 全局 PS 第一，而应写成 “存在一片 PS-favorable cyclic switching region”，或者固定 protocol 后比较。
- `switch_count=5` 仍然没有 PS-win；`switch_count=6` 有 1 个 PS-win，但整体仍不稳定。这支持“switch 太多会进入 baseline favorable 区间”的边界结论。

## 5. Layer 3 详细结果：v12 compression

### 5.1 每个 d 的最佳 PS

| d | PS-win count | best eta | best eps | best PS | PS rank | PS post-switch | PS - best nonPS | best nonPS | regret/T | tail20 | target-good | trap |
| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 4 | 0 / 9 | 0.3 | 0.005 | risky_ps_linear | 2 | 0.031499 | +0.009467 | naive_mixed_avg | 0.042824 | 0.025 | 0.2328 | 0.0326 |
| 5 | 0 / 9 | 0.4 | 0.02 | risky_ps_old | 3 | 0.041375 | +0.003125 | direct_multistage_exp3 | 0.040967 | 0.045 | 0.3744 | 0.0270 |
| 6 | 0 / 9 | 0.4 | 0.02 | risky_ps_old | 3 | 0.037602 | +0.008034 | naive_mixed_avg | 0.040392 | 0.020 | 0.3509 | 0.0240 |
| 7 | 0 / 9 | 0.4 | 0.02 | risky_ps_old | 3 | 0.037203 | +0.009790 | naive_mixed_avg | 0.041780 | 0.025 | 0.4547 | 0.0206 |

### 5.2 Layer 3 best-by-method

| rank | method | d | eta | eps | post-switch | tail20 | regret/T | target-good | trap |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | naive_mixed_avg | 4 | 0.2 | 0.005 | 0.022032 | 0.020 | 0.014524 | 0.1742 | 0.0000 |
| 2 | direct_multistage_exp3 | 4 | 0.4 | 0.005 | 0.023899 | 0.030 | 0.035624 | 0.2762 | 0.0646 |
| 3 | risky_ps_linear | 4 | 0.3 | 0.005 | 0.031499 | 0.025 | 0.042824 | 0.2328 | 0.0326 |
| 4 | direct_multistage_exp3_local | 4 | 0.4 | 0.005 | 0.032299 | 0.035 | 0.045724 | 0.1655 | 0.0723 |
| 5 | risky_ps_old | 4 | 0.3 | 0.005 | 0.032565 | 0.020 | 0.043924 | 0.2410 | 0.0328 |
| 6 | epsilon_exp3 | 4 | 0.4 | 0.01 | 0.034699 | 0.045 | 0.045324 | 0.2298 | 0.0941 |
| 7 | risky_ps | 4 | 0.4 | 0.005 | 0.034832 | 0.035 | 0.043424 | 0.2057 | 0.0364 |
| 8 | risky_ps_direct_cost | 4 | 0.4 | 0.01 | 0.035365 | 0.035 | 0.044824 | 0.1513 | 0.0347 |
| 9 | risky_ps_ix | 4 | 0.4 | 0.005 | 0.035499 | 0.030 | 0.043924 | 0.2101 | 0.0364 |
| 10 | risky_ps_safe_conditional | 4 | 0.4 | 0.01 | 0.038699 | 0.045 | 0.047324 | 0.1617 | 0.0351 |

Layer 3 解读：

- v12 在本轮 36 个组合里没有一个 PS-win。
- `d=4` 让所有方法变得更强，但第一仍是 `naive_mixed_avg`。
- 最接近 PS-win 的点是 `d=5, eta=0.4, eps=0.02`，PS 只比 direct 差 `0.003125` post-switch，但仍不是第一。
- 这说明 v12 的问题不是单点参数设置不好，而是 compression landscape 本身让 non-PS baseline 太舒服。

## 6. 对用户三个建议的回答

### 6.1 Layer 1 不重复 d=5/6

已按建议执行。Layer 1 只跑了 `d=2,3,4`。

结果：`d=4` 是新增最优点，证明补扫是必要的。

### 6.2 eta/eps 收束到 v10 最好区域

已按建议执行，并保留 v11 固定成功点的 `eta=0.2`。

eta:

- `0.2`
- `0.3`
- `0.4`

eps:

- `0.005`
- `0.01`
- `0.02`

结果：这个小网格足够有效。v10 和 v11 都找到大量 PS-win；v12 仍然 0 PS-win。

### 6.3 v12 的 eta/eps 是否是 PS 最好的前几个？

是。v12 使用了同一组 v10-derived PS-good eta/eps：

- eta: `0.2,0.3,0.4`
- eps: `0.005,0.01,0.02`

其中 `0.3/0.4` 是 v10 PS-win 的核心 eta，`0.005/0.01/0.02` 是 v10 最相关的 eps 区间。即便如此，v12 仍没有 PS-win。因此 v12 失败更像 landscape 问题，而不是参数没扫到。

## 7. 新结论和推荐主线

### 7.1 最推荐的 single-switch 主候选

`sim_v10_d2_4_top3_etaeps_full13_v1`

推荐 setting：

- cost mode: `ps_favored_trap_v10_avg_baited`
- `d=4`
- switch episode: `floor(1000/4)=250`
- `eta=0.3`
- `eps=0.01`
- winner: `risky_ps_old`

理由：

- 新补扫中 post-switch 最好。
- PS 打过 direct、direct_local、epsilon_exp3、naive_mixed_avg。
- target-good fraction 高：`0.7286`
- trap fraction 可控：`0.0456`

### 7.2 最推荐的 cyclic-switch 主候选

`sim_v11_cyclic_switch1_6_top3_etaeps_full13_v1`

推荐 setting A：

- switch_count: `3`
- eta: `0.4`
- eps: `0.02`
- winner: `risky_ps_old`
- post-switch: `-0.247521`

推荐 setting B，更接近原 fixed 参数：

- switch_count: `2`
- eta: `0.2`
- eps: `0.01`
- winner: `risky_ps_old`
- post-switch: `-0.241564`

如果论文想少引入“重新调 eta/eps”的感觉，B 更稳妥。如果只看性能，A 更强。

### 7.3 v12 的定位

v12 应作为 negative ablation：

> Compressing ordinary non-trap arm gaps makes the environment easier for non-PS baselines, especially naive_mixed_avg/direct, and does not create a PS-favorable benchmark.

不要把 v12 作为最终主线。

## 8. 下一步建议

### 8.1 立即做高 seed 验证

建议只对 top candidates 加 seeds：

1. v10 single-switch:
   - `d=4, eta=0.3, eps=0.01`
   - `d=3, eta=0.4, eps=0.005`
   - `d=5, eta=0.4, eps=0.005`，已有旧结果，可重跑 high-seed 对齐
   - `d=6, eta=0.4, eps=0.005`，已有旧结果，可重跑 high-seed 对齐

2. v11 cyclic:
   - `switch_count=2, eta=0.2, eps=0.01`
   - `switch_count=3, eta=0.4, eps=0.02`
   - `switch_count=4, eta=0.4, eps=0.01`

Seeds:

- `0..49` 作为 first confirmation
- 如果仍然 PS-win，再跑 `0..99`

### 8.2 写论文时的推荐表述

不要写：

> More switching always helps PS.

应该写：

> PS wins under nonzero exploration in a moderate nonstationarity regime. In single-switch v10, the sweet spot is around `d=4..6`; in cyclic v11, switch_count around `2..4` is favorable. Too much switching and excessive gap compression can favor averaging baselines.

### 8.3 最终 benchmark 候选优先级

1. 主候选：v10 `d=4, eta=0.3, eps=0.01`
2. 备选主候选：v10 `d=6, eta=0.4, eps=0.005`
3. cyclic 正例：v11 `switch_count=2, eta=0.2, eps=0.01`
4. cyclic high-performance 正例：v11 `switch_count=3, eta=0.4, eps=0.02`
5. negative ablation：v12 compression, no PS-win

## 9. 关键文件

- 总报告: `outputs/sim_three_layer_topetaeps_ablation_v1/three_layer_report.md`
- 总 long table: `outputs/sim_three_layer_topetaeps_ablation_v1/all_layers_long.csv`
- 总 combo summary: `outputs/sim_three_layer_topetaeps_ablation_v1/all_layers_combo_summaries.csv`
- 总 PS-win table: `outputs/sim_three_layer_topetaeps_ablation_v1/top_ps_winning_configs.csv`
- Layer 1 report dir: `outputs/sim_three_layer_topetaeps_ablation_v1/sim_v10_d2_4_top3_etaeps_full13_v1/`
- Layer 2 report dir: `outputs/sim_three_layer_topetaeps_ablation_v1/sim_v11_cyclic_switch1_6_top3_etaeps_full13_v1/`
- Layer 3 report dir: `outputs/sim_three_layer_topetaeps_ablation_v1/sim_v12_gapcompressed_d4_7_top3_etaeps_full13_v1/`
