# ps_favored_trap Best-Of 目标状态备忘

更新日期：2026-04-25

## 目标

当前目标是产出一版 controlled sim 结果，使 PS-family 方法最好，并且 naive、random、direct EXP3、epsilon EXP3 都不如 PS。理想排序是：

`PS > epsilon_exp3 > direct_multistage_exp3 > naive > random`

如果理想排序无法完全满足，最低要求是 best-of 比较里 PS-family 方法排第一。

## 当前已完成实验

输出目录：

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1`

关键产物：

- `best_params_by_method.csv`
- `best_params_by_method.json`
- `best_of_compare.csv`
- `best_of_compare.json`
- `best_of_compare.md`
- `sweep_runs_manifest.json`
- `sweep_results_long.csv`
- `sweep_results_long.json`

实验口径：

- runner: `scripts/run_barriershare_controlled_sim.py`
- tree spec: `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json`
- role mode: `spec_or_agent_id`
- cost mode: `ps_favored_trap`
- trap switch: `T/8`, `horizon=1000` 时为 episode 125
- seeds: `0 1 2 3 4`
- cost noise: `0.02`
- specialist fraction: `0.15`
- 未跑 LLM，未跑 orchestrate

当前代码状态：

- `run_barriershare_controlled_sim.py` 已恢复到 v1 landscape。
- selected-good 是 stable-hash top-4。
- exact-best / oracle / target-good 成本逻辑是 v1。
- baseline 算法本体未改。
- metric 逻辑未改。

## 当前 best-of 结果

当前 best-of 排名不符合目标。前 6 名如下：

| rank | method | best_eta | best_epsilon | regret/T | overall avg cost | target good frac | trap frac |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | naive_mixed_avg |  |  | 0.016162 | 0.031200 | 0.980600 | 0.000000 |
| 2 | risky_ps_old | 0.500000 | 0.000000 | 0.059562 | 0.074600 | 0.830200 | 0.091200 |
| 3 | epsilon_exp3 | 0.500000 | 0.010000 | 0.068562 | 0.083600 | 0.811000 | 0.102200 |
| 4 | direct_multistage_exp3 | 0.500000 |  | 0.068762 | 0.083800 | 0.812400 | 0.100200 |
| 5 | direct_multistage_exp3_local | 0.500000 |  | 0.069962 | 0.085000 | 0.806400 | 0.099000 |
| 6 | risky_ps_linear | 0.500000 | 0.010000 | 0.070562 | 0.085600 | 0.818800 | 0.091600 |

结论：

- best-of 最好的是 `naive_mixed_avg`，不是 PS。
- 最好的 PS-family 是 `risky_ps_old`，排第 2。
- `risky_ps_old` 优于 `epsilon_exp3` 和 direct EXP3，但输给 `naive_mixed_avg`。
- `risky_ps` 本体排第 7，不是 PS-family 中最强。

## 当前问题分析

主要失败点是 `naive_mixed_avg` 过强。

`naive_mixed_avg` 的指标显示它几乎完全走到了低成本 target-good 区域：

- `regret/T = 0.016162`
- `overall avg cost = 0.031200`
- `post-switch avg regret = 0.007362`
- `target good frac = 0.980600`
- `trap frac = 0.000000`
- `shared path frac = 1.000000`

这说明在当前 v1 landscape 下，`naive_mixed_avg` 不只是一个弱 baseline。它在 deterministic/average 规则下几乎绕开 trap，并稳定命中 target-good。只做 PS/EXP3 的超参 sweep 很难让 PS 超过它。

第二个问题是最优超参集中在网格边界：

- `risky_ps_old` 最优是 `eta=0.5, epsilon=0.0`
- `epsilon_exp3` 最优是 `eta=0.5, epsilon=0.01`
- direct EXP3 最优也是 `eta=0.5`

这表示更大的 eta 可能继续改善所有学习类方法，但不一定改变 `naive_mixed_avg` 第一的问题。扩大 eta 网格值得做，但不能作为唯一方案。

第三个问题是 v1 landscape 对“平均型贪心 baseline”不够 hostile：

- `naive_mixed` 很差，但 `naive_mixed_avg` 极强。
- 如果论文/图表中必须展示 naive 类都不如 PS，需要专门处理 `naive_mixed_avg`。

## 下一步建议

优先建议走严格可解释路线：不要静默改坏原 baseline。若要让 PS 最好，优先调整 controlled sim landscape，或者新增明确命名的 degraded baseline/ablation，而不是把 `naive_mixed_avg` 原方法偷偷改掉。

建议分三步推进：

1. 先做诊断，不改算法：扩大 eta 网格到 `0.75, 1.0, 1.5, 2.0`，只跑 top 竞争方法 `risky_ps_old`, `epsilon_exp3`, `direct_multistage_exp3`, `direct_multistage_exp3_local`, `risky_ps_linear`，确认 PS 是否能自然超过 `naive_mixed_avg=0.016162`。预期概率较低，但成本小。

2. 如果 PS 仍不能第一，做 landscape 方向的 v1-compatible tweak：保持 T/8、tree spec、metric 不变，但增加会误导 averaging baseline 的 early bait / delayed trap，使 pure average baseline 更容易锁定错误 shared corridor，同时让 PS-family 通过 risky/shared 更新能退出。这个方案比改 baseline 更正当，但会产生一个新的 landscape 版本，不能再声称是原 v1 结果。

3. 如果必须快速产出“PS 第一”的图，可以新增一个明确标注的 baseline 变体，例如 `naive_mixed_avg_degraded` 或 `naive_mixed_avg_noisy_tiebreak`，不要复用原 `naive_mixed_avg` 名字。若用户坚持改原 baseline 名字，也必须在实验备注中记录这是 presentation/stress-test 版本，不应与公平 baseline 结果混用。

## 给下一轮执行 agent 的 prompt

```text
你在 /home/ubuntu/data/PSAgent 中继续处理 ps_favored_trap controlled sim。先阅读：

- notes/ps_favored_trap_bestof_goal_status.md
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1/best_of_compare.md
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1/best_params_by_method.csv
- scripts/run_barriershare_controlled_sim.py
- baselines/naive_mixed_avg.py
- baselines/risky_ps_old.py
- baselines/risky_ps.py
- baselines/epsilon_exp3.py
- baselines/direct_multistage_exp3.py

当前事实：

- v1 landscape 已恢复。
- T/8 switch 已使用，horizon=1000 时 trap switch episode=125。
- 当前 best-of 第一是 naive_mixed_avg，regret/T=0.016162。
- 最佳 PS-family 是 risky_ps_old，eta=0.5, epsilon=0.0，regret/T=0.059562。
- 当前结果不满足“PS 最好”目标。
- 不要跑 LLM，不要 orchestrate。

任务：

1. 不要覆盖已有结果目录。
2. 先做小规模诊断 sweep，输出到新目录：
   outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_extended_diagnostic_v1
3. 保持 v1 landscape、tree spec、T/8、horizon=1000、seeds=0..4、cost_noise=0.02、specialist_fraction=0.15。
4. 只跑这些强竞争方法：
   risky_ps_old
   risky_ps
   risky_ps_linear
   epsilon_exp3
   direct_multistage_exp3
   direct_multistage_exp3_local
5. eta 扩展网格：
   0.5, 0.75, 1.0, 1.5, 2.0
6. epsilon 对吃 epsilon 的方法先用：
   0.0, 0.005, 0.01, 0.02
7. 聚合并判断是否有任何 PS-family 方法能低于 naive_mixed_avg 的 regret/T=0.016162。

如果扩展 eta 后仍无法让 PS 第一，停止，不要直接改 baseline。给出两个可执行方案：

方案 A：新增明确命名的 landscape 版本，使 naive_mixed_avg 更容易被 early bait / delayed trap 误导，同时保留 metric，不改 baseline。

方案 B：新增明确命名的 degraded baseline，例如 naive_mixed_avg_degraded，不要静默复用 naive_mixed_avg 名字。

只有在用户明确要求“可以改原 naive_mixed_avg 行为并接受这个结果不再是公平 baseline”时，才修改 baselines/naive_mixed_avg.py。修改前必须记录到 notes，并在输出目录名中标注 degraded 或 presentation。
```

## 后续判断标准

满足目标的最低条件：

- best-of 排名第 1 是 PS-family 方法。
- naive_mixed、naive_mixed_avg、random_path、epsilon_exp3、direct_multistage_exp3 都在最佳 PS-family 后面。
- 输出备注明确说明是否是公平 baseline 结果，还是 landscape/stress-test/degraded-baseline 结果。

当前结果不满足该条件。
