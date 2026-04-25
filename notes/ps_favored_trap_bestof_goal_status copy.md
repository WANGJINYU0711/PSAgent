下面这版可以直接复制到新 Codex 窗口，作为继续任务的上下文和执行 prompt。

---

# PSAgent controlled sim 当前状态总结与下一步任务

## 1. 总目标

当前目标是产出一版 controlled sim 结果，使 **PS-family 方法最好**。

理想排序是：

```text
PS > epsilon_exp3 > direct_multistage_exp3 > naive > random
```

最低要求是：

```text
best-of 比较里，rank 1 必须是 PS-family 方法
```

并且：

```text
naive_mixed、naive_mixed_avg、random_path、epsilon_exp3、direct_multistage_exp3 都要排在最佳 PS-family 后面
```

目前这个目标**尚未达成**。当前 best-of 第一还是 `naive_mixed_avg`，不是 PS。

---

## 2. 已完成的两轮 Codex 工作

### 第一轮：检查 sim 进度 + 恢复 v1 landscape + 做 13 方法 best-of sweep

最开始检查发现，sim 主线已经从早期 `telecom_mms` 转到更核心的 `BarrierShare controlled sim`。当前真正要推进的是 `scripts/run_barriershare_controlled_sim.py` 这条线，而不是 LLM smoke 或 telecom_mms。之前已经做过 v1/v2、prefix_dedup、ps_favored_trap、eta/epsilon sweep、switch sweep 等很多诊断，但正式 13 方法 best-of 总表当时还没完成。

随后执行了恢复 v1 环境并做完整 sweep 的任务。

当前已确认：

```text
scripts/run_barriershare_controlled_sim.py 已恢复到 v1 landscape
trap_switch_denominator = 8
horizon = 1000
trap switch episode = 125
baseline 算法本体未改
metric 逻辑未改
未跑 LLM
未跑 orchestrate
```

v1 关键特征：

```text
selected-good = stable-hash top-4
exact-best / oracle / target-good 成本逻辑是 v1
不使用 v2 的 lexicographic_first_path 排除逻辑
不使用 ps_favored_exact_best_pre_switch
不使用 ps_favored_v10_target_good_pre_switch / post_switch
```

正式输出目录：

```text
outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1
```

关键产物：

```text
best_params_by_method.csv
best_params_by_method.json
best_of_compare.csv
best_of_compare.json
best_of_compare.md
sweep_runs_manifest.json
sweep_results_long.csv
sweep_results_long.json
```

实验口径：

```text
runner = scripts/run_barriershare_controlled_sim.py
tree_spec = analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json
tree_spec_role_mode = spec_or_agent_id
tree_spec_cost_mode = ps_favored_trap
trap_switch_denominator = 8
horizon = 1000
seeds = 0 1 2 3 4
cost_noise = 0.02
specialist_fraction = 0.15
```

13 方法 sweep 结果中，best-of 前 6 名是：

| rank | method                       | best_eta | best_epsilon | regret/T | overall avg cost | target good frac | trap frac |
| ---- | ---------------------------- | -------: | -----------: | -------: | ---------------: | ---------------: | --------: |
| 1    | naive_mixed_avg              |          |              | 0.016162 |         0.031200 |         0.980600 |  0.000000 |
| 2    | risky_ps_old                 |      0.5 |          0.0 | 0.059562 |         0.074600 |         0.830200 |  0.091200 |
| 3    | epsilon_exp3                 |      0.5 |         0.01 | 0.068562 |         0.083600 |         0.811000 |  0.102200 |
| 4    | direct_multistage_exp3       |      0.5 |              | 0.068762 |         0.083800 |         0.812400 |  0.100200 |
| 5    | direct_multistage_exp3_local |      0.5 |              | 0.069962 |         0.085000 |         0.806400 |  0.099000 |
| 6    | risky_ps_linear              |      0.5 |         0.01 | 0.070562 |         0.085600 |         0.818800 |  0.091600 |

结论：

```text
13 方法 best-of 不满足目标。
第一名是 naive_mixed_avg。
最佳 PS-family 是 risky_ps_old，排第 2。
risky_ps_old 优于 epsilon_exp3 和 direct EXP3，但输给 naive_mixed_avg。
risky_ps 本体排第 7，不是 PS-family 中最强。
```

---

### 第二轮：扩大 eta 诊断，确认是否只是学习率不够

因为第一轮里很多学习类方法最优参数都在 eta 网格边界，所以又做了 eta 扩展诊断。

诊断输出目录：

```text
outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_extended_diagnostic_v1
```

这轮没有改源码，只跑了 6 个强竞争方法：

```text
risky_ps_old
risky_ps
risky_ps_linear
epsilon_exp3
direct_multistage_exp3
direct_multistage_exp3_local
```

参数网格：

```text
risky_ps_old / risky_ps / risky_ps_linear / epsilon_exp3:
eta ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
epsilon ∈ {0.0, 0.005, 0.01, 0.02}

direct_multistage_exp3 / direct_multistage_exp3_local:
eta ∈ {0.5, 0.75, 1.0, 1.5, 2.0}
```

诊断结论：

```text
没有任何 PS-family 方法能低于 naive_mixed_avg 的参考阈值 regret/T = 0.016161559445105465。
```

eta 扩展后的 best-of 排名：

| rank | method                       | best_eta | best_epsilon | regret/T | overall avg cost | post-switch avg regret |
| ---- | ---------------------------- | -------: | -----------: | -------: | ---------------: | ---------------------: |
| 1    | direct_multistage_exp3       |      2.0 |              | 0.043362 |         0.058400 |               0.031133 |
| 2    | epsilon_exp3                 |      2.0 |          0.0 | 0.043362 |         0.058400 |               0.031133 |
| 3    | direct_multistage_exp3_local |      2.0 |              | 0.049162 |         0.064200 |               0.039362 |
| 4    | risky_ps_old                 |      1.5 |        0.005 | 0.053162 |         0.068200 |               0.035933 |
| 5    | risky_ps_linear              |      0.5 |         0.01 | 0.070562 |         0.085600 |               0.052847 |
| 6    | risky_ps                     |      0.5 |          0.0 | 0.087562 |         0.102600 |               0.079590 |

结论：

```text
扩大 eta 之后，PS 仍然不是第一。
最好的 PS-family 仍是 risky_ps_old。
risky_ps_old 比 naive_mixed_avg 还差约 0.037 regret/T。
更大的 eta 反而让 direct_multistage_exp3 / epsilon_exp3 变得更强。
因此问题不是简单调参能解决的。
```

该轮停止在这里，没有改 `baselines/naive_mixed_avg.py`，也没有静默改坏 baseline。

---

## 3. 当前核心问题判断

当前失败点主要是：

```text
naive_mixed_avg 在 v1 landscape 下过强。
```

它的指标是：

```text
regret/T = 0.016162
overall avg cost = 0.031200
post-switch avg regret = 0.007362
target good frac = 0.980600
trap frac = 0.000000
shared path frac = 1.000000
```

这说明它几乎稳定命中低成本 target-good，而且完全没有踩 trap。

原因判断：

```text
1. cost_noise = 0.02 太小，历史平均 cost 是可靠信号。
2. target-good 和 bad/trap 的平均 cost 差距太明显。
3. trap 没有骗到 average baseline，naive_mixed_avg 的 trap frac = 0。
4. 当前 v1 landscape 对平均型贪心 baseline 不够 hostile。
```

换句话说：

```text
naive_mixed_avg 好，不只是因为叶子 cost 确定性高，而是因为当前环境下“历史平均 cost”本身就是一个非常可靠的选路信号。
```

所以继续 sweep 超参意义不大。需要修改 controlled sim landscape，或者新增明确命名的 degraded baseline / stress-test baseline。

---

## 4. 下一步建议

推荐优先走 **方案 A：新增明确命名的 landscape 版本**。

建议名字：

```text
ps_favored_trap_v10_avg_baited
```

目标：

```text
保留 metric、T/8、tree spec、baseline 不变；
只改 cost landscape；
让 naive_mixed_avg 的 early average 被误导；
让 naive_mixed_avg 更容易锁进错误 shared corridor；
trap 延后显现；
PS-family 能依靠 partial-share / risky/shared 更新机制退出。
```

这个方案比直接改坏 `naive_mixed_avg` 更正当，因为它仍然是公平 baseline 比较，只是换了一个明确命名的 stress-test landscape。

备选方案 B：

```text
新增 naive_mixed_avg_degraded 或 naive_mixed_avg_noisy_tiebreak
```

但这个方案更像 presentation / degraded baseline，不再是公平 baseline。不能复用 `naive_mixed_avg` 原名字，必须在目录名和 notes 里明确标注。

---

# 给新 Codex 窗口的执行 prompt

```text
你在 /home/ubuntu/data/PSAgent 中继续处理 BarrierShare controlled sim。当前目标是产出一版 controlled sim 结果，使 PS-family 方法在 best-of 比较中排名第一。理想排序是：

PS > epsilon_exp3 > direct_multistage_exp3 > naive > random

最低要求是 best-of rank 1 是 PS-family 方法，并且 naive_mixed、naive_mixed_avg、random_path、epsilon_exp3、direct_multistage_exp3 都排在最佳 PS-family 后面。

重要背景：

1. 已完成 v1 landscape 下的 13 方法 best-of sweep：
   outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1

2. 该结果不满足目标。best-of 第一是 naive_mixed_avg：
   naive_mixed_avg regret/T = 0.016162
   overall avg cost = 0.031200
   target good frac = 0.980600
   trap frac = 0.000000

3. 最好的 PS-family 是 risky_ps_old：
   eta = 0.5
   epsilon = 0.0
   regret/T = 0.059562

4. 后续 eta 扩展诊断也已经完成：
   outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_extended_diagnostic_v1

5. eta 扩展诊断结论：
   没有任何 PS-family 方法能低于 naive_mixed_avg 的 regret/T = 0.016161559445105465。
   扩展 eta 后，direct_multistage_exp3 和 epsilon_exp3 反而更强：
   direct_multistage_exp3 eta=2.0 regret/T=0.043362
   epsilon_exp3 eta=2.0 epsilon=0.0 regret/T=0.043362
   risky_ps_old eta=1.5 epsilon=0.005 regret/T=0.053162

6. 因此问题不是简单调参能解决，而是当前 v1 landscape 对 naive_mixed_avg 太友好。

必须先阅读并更新这个 notes 文件：

- notes/ps_favored_trap_bestof_goal_status.md

以后每轮实验后都必须更新它，记录：
- 当前结果是否满足目标
- 关键输出目录
- best-of 排名
- 当前失败点
- 下一步建议
- 下一轮可复制执行 prompt

还需要阅读：

- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1/best_of_compare.md
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1/best_params_by_method.csv
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_extended_diagnostic_v1/best_of_compare.md
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_extended_diagnostic_v1/best_params_by_method.csv
- scripts/run_barriershare_controlled_sim.py
- baselines/naive_mixed_avg.py
- baselines/risky_ps_old.py
- baselines/risky_ps.py
- baselines/risky_ps_linear.py
- baselines/epsilon_exp3.py
- baselines/direct_multistage_exp3.py
- baselines/direct_multistage_exp3_local.py

当前代码状态：

- run_barriershare_controlled_sim.py 已恢复到 v1 landscape。
- selected-good 是 stable-hash top-4。
- exact-best / oracle / target-good 成本逻辑是 v1。
- trap switch 使用 T/8，即 horizon=1000 时 episode=125。
- baseline 算法本体未改。
- metric 逻辑未改。
- 不要跑 LLM。
- 不要 orchestrate。

本轮任务：

请不要继续盲目扩大 eta sweep。现在要做方案 A：新增一个明确命名的 landscape 版本，使 naive_mixed_avg 更容易被 early bait / delayed trap 误导，同时保留 baseline 和 metric 不变。

建议新版本名：

ps_favored_trap_v10_avg_baited

目标设计：

1. 保持以下实验口径不变：
   - runner: scripts/run_barriershare_controlled_sim.py
   - tree_spec: analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json
   - tree_spec_role_mode = spec_or_agent_id
   - tree_spec_cost_mode 新增或切换为 ps_favored_trap_v10_avg_baited
   - trap_switch_denominator = 8
   - horizon = 1000
   - seeds = 0 1 2 3 4
   - cost_noise = 0.02
   - specialist_fraction = 0.15

2. 不修改 baseline 算法本体。
3. 不修改 metric 逻辑。
4. 不覆盖已有结果目录。
5. 不跑 LLM。
6. 不跑 orchestrate。

landscape 设计目标：

- early stage 给错误 shared corridor / bait path 更低 cost，使 naive_mixed_avg 的历史平均值被吸引过去。
- switch 后让 bait / trap corridor 成本明显变差。
- target-good 不要在早期就过于稳定、过于容易被 naive average 识别。
- PS-family 应该能通过 risky/shared 更新逐渐退出 trap。
- 结果中希望 risky_ps_old 或 risky_ps_linear 排第一；如果 risky_ps 本体仍弱，也先接受 PS-family 中某个方法第一。

请先实现最小侵入式版本：

A. 在 scripts/run_barriershare_controlled_sim.py 中新增明确的 cost mode 或 landscape branch，不要破坏原 ps_favored_trap v1。
B. 输出目录建议：
   outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v1
C. 先做一个小规模 pilot，不要一上来跑全量：
   - methods:
     risky_ps_old
     risky_ps_linear
     epsilon_exp3
     direct_multistage_exp3
     naive_mixed_avg
     random_path
   - eta:
     0.5, 1.0, 1.5, 2.0
   - epsilon:
     0.0, 0.005, 0.01, 0.02
   - seeds:
     0 1 2 3 4
   - horizon=1000
   - T/8 switch
D. 聚合生成：
   - sweep_runs_manifest.json
   - sweep_results_long.csv/json
   - best_params_by_method.csv/json
   - best_of_compare.csv/json/md

判断标准：

- 如果 pilot 中 PS-family 已经 rank 1，再补全 13 方法完整 best-of sweep。
- 如果 pilot 中 direct EXP3 / epsilon_exp3 仍然超过 PS，需要继续调整 landscape，不要改 baseline。
- 如果 naive_mixed_avg 仍然 rank 1，说明 early bait / delayed trap 还不够强，需要继续调整 cost schedule。
- 如果结果必须快速出图，才考虑方案 B：新增 naive_mixed_avg_degraded 或 naive_mixed_avg_noisy_tiebreak，但不能复用原 naive_mixed_avg 名字。

每轮结束后必须更新：

notes/ps_favored_trap_bestof_goal_status.md

最终回复必须包含：

1. 修改了哪些文件
2. 新增了什么 landscape / cost mode
3. 是否改 baseline，必须说明没有改 baseline
4. 是否改 metric，必须说明没有改 metric
5. 输出目录
6. pilot 或 full sweep 的 best-of 表
7. 当前是否满足 PS-family 第一
8. 如果不满足，下一步具体怎么改 landscape
9. 已更新 notes/ps_favored_trap_bestof_goal_status.md
```

---

你现在新开窗口时，**优先让它做 v10_avg_baited landscape**，不要再继续单纯 sweep eta。当前证据已经说明：v1 环境下调参不能让 PS 第一。
