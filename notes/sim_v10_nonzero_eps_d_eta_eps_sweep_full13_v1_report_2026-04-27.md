# sim_v10 nonzero-eps d/eta/eps sweep full13 v1 报告

日期：2026-04-27

## 结论摘要

这版实验命名为：

`sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1`

简称建议：

`nonzero-eps d-eta-eps v1`

这里的 `d` 就是此前 LLM run_config 里你说的 `n`，也就是 `switch_denominator`。在 horizon `T=1000` 时，switch episode 是 `floor(T / d)`。

本轮没有使用 `eps=0`。epsilon grid 全部为非零值：

`0.005, 0.01, 0.02, 0.05, 0.1`

主结论：

1. 非零 epsilon 下确实存在 PS-family 第一的 controlled-sim setting。
2. PS 第一主要出现在 `d=5/6`，也就是 switch episode `200/166`，而不是 LLM smoke 对齐的 `d=7`。
3. 最稳定的 winner 是 `risky_ps_old`。
4. 最佳 overall PS-win setting 是 `d=6, eta=0.4, eps=0.005`。
5. `d=7` 时，最佳 PS 仍能打过 `direct_multistage_exp3`、`direct_multistage_exp3_local` 和 `epsilon_exp3`，但没有打过 `naive_mixed_avg`，所以不是全方法第一。
6. 如果目标是“eps 不为 0 且 PS 在 13 方法中第一”，下一轮应优先锁定 `d=5/6, eta=0.3/0.4, eps=0.005/0.01`，再做更多 seeds 或局部细扫。

## 实验设置

输出目录：

`outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/`

运行脚本：

`tmp/run_sim_v10_nonzero_eps_d_eta_eps_sweep.py`

底层 runner：

`scripts/run_barriershare_controlled_sim.py`

固定设置：

- tree spec: `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json`
- tree_spec_role_mode: `spec_or_agent_id`
- tree_spec_cost_mode: `ps_favored_trap_v10_avg_baited`
- horizon: `1000`
- seeds: `0 1 2 3 4 5 6 7 8 9`
- cost_noise: `0.02`
- specialist_fraction: `0.15`
- methods: 13

Sweep grid：

- `d`: `5, 6, 7, 8, 10, 12, 16`
- `eta`: `0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5`
- `eps`: `0.005, 0.01, 0.02, 0.05, 0.1`

Coverage：

- combos: `245 / 245`
- rows: `3185`
- failures: `0`
- post-switch PS-winning combos: `6 / 245`
- overall PS-winning combos: `2 / 245`

主 ranking key：

`post_switch_avg_regret_mean`, then `tail20_avg_total_cost_mean`, then `regret_per_t_mean`

secondary ranking：

`regret_per_t_mean`, then `overall_avg_total_cost_mean`, then `post_switch_avg_regret_mean`

## 13 方法 best setting: post-switch primary

| rank | method | d | switch | eta | eps | post_switch | tail20 | regret/T | target_good | trap |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | naive_mixed_avg | 16 | 62 | 0.05 | 0.005 | 0.022196 | 0.010 | 0.018573 | 0.9263 | 0.0000 |
| 2 | direct_multistage_exp3 | 16 | 62 | 0.5 | 0.005 | 0.040426 | 0.005 | 0.047273 | 0.9045 | 0.0188 |
| 3 | epsilon_exp3 | 16 | 62 | 0.5 | 0.005 | 0.051834 | 0.015 | 0.059773 | 0.8888 | 0.0138 |
| 4 | risky_ps_old | 16 | 62 | 0.5 | 0.005 | 0.054179 | 0.020 | 0.060473 | 0.8839 | 0.0122 |
| 5 | direct_multistage_exp3_local | 12 | 83 | 0.5 | 0.005 | 0.058157 | 0.015 | 0.065946 | 0.8617 | 0.0319 |
| 6 | risky_ps_linear | 16 | 62 | 0.3 | 0.005 | 0.058337 | 0.000 | 0.066073 | 0.8818 | 0.0126 |
| 7 | risky_ps | 10 | 100 | 0.3 | 0.005 | 0.076778 | 0.005 | 0.080634 | 0.8262 | 0.0200 |
| 8 | risky_ps_ix | 8 | 125 | 0.3 | 0.005 | 0.080514 | 0.010 | 0.082392 | 0.8122 | 0.0259 |
| 9 | risky_ps_safe_conditional | 16 | 62 | 0.5 | 0.005 | 0.096183 | 0.020 | 0.099873 | 0.8373 | 0.0146 |
| 10 | risky_ps_safe_conditional_ix | 16 | 62 | 0.5 | 0.005 | 0.099701 | 0.025 | 0.103173 | 0.8315 | 0.0147 |
| 11 | risky_ps_direct_cost | 16 | 62 | 0.3 | 0.01 | 0.280299 | 0.065 | 0.274473 | 0.6272 | 0.0192 |
| 12 | random_path | 7 | 142 | 0.05 | 0.005 | 0.751538 | 0.755 | 0.689980 | 0.0720 | 0.0792 |
| 13 | naive_mixed | 16 | 62 | 0.05 | 0.005 | 0.814520 | 0.905 | 0.761273 | 0.0812 | 0.0000 |

这个表说明：如果允许跨所有 d/eta/eps 选每个方法的 best setting，`naive_mixed_avg` 在 very-early switch 的 `d=16` 上仍非常强，整体和 post-switch 都第一。但这不否定 PS-win setting 的存在，因为 PS-win 是在同一个 `(d, eta, eps)` 内 13 方法公平比较。

## PS 在同一 config 内第一的 setting

| d | switch | eta | eps | best PS | post_switch | PS - direct | PS - direct_local | PS - epsilon_exp3 | PS - naive_avg | regret/T |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 6 | 166 | 0.4 | 0.005 | risky_ps_old | 0.057746 | -0.093885 | -0.030815 | -0.018225 | -0.030935 | 0.056792 |
| 5 | 200 | 0.4 | 0.005 | risky_ps_old | 0.063875 | -0.057750 | -0.035250 | -0.041875 | -0.025500 | 0.058667 |
| 6 | 166 | 0.3 | 0.01 | risky_ps_old | 0.069496 | -0.000480 | -0.058034 | -0.038129 | -0.019185 | 0.068492 |
| 5 | 200 | 0.4 | 0.01 | risky_ps_old | 0.076750 | -0.044875 | -0.022375 | -0.019000 | -0.012625 | 0.068867 |
| 5 | 200 | 0.3 | 0.01 | risky_ps_old | 0.077250 | -0.006625 | -0.036625 | -0.022000 | -0.012125 | 0.070967 |
| 6 | 166 | 0.4 | 0.01 | risky_ps_old | 0.079568 | -0.072062 | -0.008993 | -0.011151 | -0.009113 | 0.075092 |

解读：

- 所有 PS-win setting 都集中在 `d=5/6`。
- 所有 PS-win setting 的 epsilon 都是非零：`0.005` 或 `0.01`。
- `eta=0.4` 最稳，`eta=0.3` 也有两个 win。
- `eps=0.005` 的两个 setting 也在 overall regret/T 上让 PS 第一。

## Overall PS-win setting

| d | switch | eta | eps | winner | regret/T | PS - direct | PS - direct_local | PS - epsilon_exp3 | PS - naive_avg |
| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 6 | 166 | 0.4 | 0.005 | risky_ps_old | 0.056792 | -0.077600 | -0.028500 | -0.015200 | -0.010400 |
| 5 | 200 | 0.4 | 0.005 | risky_ps_old | 0.058667 | -0.045300 | -0.031000 | -0.033300 | -0.004900 |

这两个 setting 是最干净的证据：不只是 post-switch，连 overall regret/T 也是 PS-family 第一。

## d=7 专项

`d=7` 对应 `switch_episode=floor(1000/7)=142`，和 LLM smoke 的 `n=7` 语义对齐。

`d=7` 最佳 PS setting：

| d | switch | eta | eps | best overall method in config | best PS | PS rank | PS post_switch | PS - direct | PS - direct_local | PS - epsilon_exp3 | PS - naive_avg | PS regret/T |
| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 7 | 142 | 0.4 | 0.005 | naive_mixed_avg | risky_ps_old | 2 | 0.063427 | -0.009907 | -0.022727 | -0.041142 | 0.011305 | 0.063980 |

解读：

- `d=7, eta=0.4, eps=0.005` 下，PS 比 direct、direct_local、epsilon_exp3 都好。
- 但是 `naive_mixed_avg` 更好，所以 PS 是 rank 2，不是 13 方法第一。
- 这和 LLM smoke 里“PS 接近但没有第一”的现象方向一致。

## 按 d 看最佳 PS

| d | switch | best PS eta | best PS eps | PS rank in config | PS post_switch | PS - direct | PS - direct_local | PS - epsilon_exp3 | PS - naive_avg |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 5 | 200 | 0.4 | 0.005 | 1 | 0.063875 | -0.057750 | -0.035250 | -0.041875 | -0.025500 |
| 6 | 166 | 0.4 | 0.005 | 1 | 0.057746 | -0.093885 | -0.030815 | -0.018225 | -0.030935 |
| 7 | 142 | 0.4 | 0.005 | 2 | 0.063427 | -0.009907 | -0.022727 | -0.041142 | 0.011305 |
| 8 | 125 | 0.4 | 0.005 | 2 | 0.060743 | -0.001714 | -0.004800 | -0.016571 | 0.016686 |
| 10 | 100 | 0.3 | 0.005 | 3 | 0.056333 | 0.006333 | -0.023111 | -0.016889 | 0.023889 |
| 12 | 83 | 0.3 | 0.005 | 3 | 0.061101 | 0.004035 | -0.004144 | -0.035224 | 0.027590 |
| 16 | 62 | 0.5 | 0.005 | 4 | 0.054179 | 0.013753 | -0.010661 | 0.002345 | 0.031983 |

这个表最关键：

- `d=5/6`：PS rank 1。
- `d=7/8`：PS rank 2，主要输给 `naive_mixed_avg`。
- `d=10/12/16`：PS 的 absolute post-switch cost 还可以，但 direct/naive 的优势增强，PS 不再第一。

## Cost landscape：低 cost 是否接近？

当前 v10 cost 不是 `1,2,3,4,5,6` 这种线性展开；它是明显分簇的。

pre-switch：

| group | count | min | mean | max |
| --- | ---: | ---: | ---: | ---: |
| bait corridor | 21 | 0.001099 | 0.002790 | 0.005000 |
| local decoy | 5 | 0.012000 | 0.012000 | 0.012000 |
| trap basin | 64 | 0.012234 | 0.016962 | 0.021887 |
| exact best | 1 | 0.063665 | 0.063665 | 0.063665 |
| target good | 2 | 0.071573 | 0.074225 | 0.076877 |
| balancing candidate | 32 | 0.180288 | 0.218584 | 0.256817 |
| decoy branch | 42 | 0.280629 | 0.313220 | 0.345085 |
| one barrier | 126 | 0.520031 | 0.569029 | 0.619888 |
| multi barrier | 106 | 0.700750 | 0.747604 | 0.799160 |

post-switch：

| group | count | min | mean | max |
| --- | ---: | ---: | ---: | ---: |
| exact best | 1 | 0.010000 | 0.010000 | 0.010000 |
| target good | 2 | 0.018996 | 0.019437 | 0.019878 |
| balancing candidate | 32 | 0.520091 | 0.571373 | 0.618275 |
| non-safe all-shared | 30 | 0.582515 | 0.615684 | 0.679790 |
| decoy branch | 42 | 0.601683 | 0.658830 | 0.699723 |
| local decoy | 5 | 0.620000 | 0.620000 | 0.620000 |
| one barrier | 126 | 0.700807 | 0.746231 | 0.797832 |
| bait corridor | 21 | 0.867735 | 0.895306 | 0.927018 |
| target bad | 9 | 0.888645 | 0.924096 | 0.948558 |
| trap basin | 64 | 0.992000 | 0.992000 | 0.992000 |

所以回答“低 cost 会很接近吗？”：

- pre-switch 的最低三簇很接近：bait corridor `0.001-0.005`，local decoy `0.012`，trap basin `0.012-0.022`。
- post-switch 的低 cost 非常尖锐：exact best `0.010`，target good `0.019` 左右；下一簇直接跳到约 `0.52+`。
- 因此非零 epsilon 的探索税很真实：post-switch 只要随机探索到 wrong branch，经常从 `0.01-0.02` 跳到 `0.52-0.99`。

这也解释了为什么 `eps=0.005` 明显比 `eps=0.02/0.05/0.1` 更适合 PS-win：非零 exploration 必须小到不会频繁踩进高 cost 分簇。

## 下一步建议

如果目标是获得一版 “eps != 0 且 PS 明确最佳” 的主 sim 图表，建议下一轮不要继续扩大全局网格，而是做局部稳定性验证：

1. 锁定 `d=6, eta=0.4, eps=0.005`，增加 seeds 到 `0..49` 或 `0..99`。
2. 同时跑 nearby settings：
   - `d=5,6`
   - `eta=0.3,0.35,0.4,0.45`
   - `eps=0.0025,0.005,0.0075,0.01`
3. 保留 13 methods。
4. 主表同时报告：
   - post_switch_avg_regret_mean/std
   - regret_per_t_mean/std
   - tail20_avg_total_cost_mean
   - target_good_fraction_mean
   - trap_basin_fraction_mean
   - shared_path_fraction_mean
5. 把 `d=7` 作为 LLM-aligned secondary result，而不是主 result，因为当前 `d=7` 下 PS 不是 13 方法第一。

## 关键文件

- Full report: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/sweep_report.md`
- Long table: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/sweep_results_long.csv`
- Compact long table: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/sweep_results_long_primary_metrics.csv`
- Combo summary: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/combo_summaries.csv`
- PS-winning post-switch configs: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/top_ps_winning_post_switch_configs.csv`
- Best by method, post-switch: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/best_by_method_post_switch.csv`
- Best by method, overall: `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/best_by_method_overall.csv`
