# ps_favored_trap Best-Of 目标状态备忘
更新日期：2026-04-25

## 1. 总目标

当前目标是产出一版 controlled sim 结果，使 PS-family 方法在 best-of 比较中排第一。

理想排序：

`PS > epsilon_exp3 > direct_multistage_exp3 > naive > random`

最低要求：

- best-of rank 1 必须是 PS-family 方法
- `naive_mixed`
- `naive_mixed_avg`
- `random_path`
- `epsilon_exp3`
- `direct_multistage_exp3`
- `direct_multistage_exp3_local`

都必须排在最佳 PS-family 后面。

## 2. 已完成实验

### 2.1 v1 landscape 下的 13 方法 best-of sweep

输出目录：

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_sweep_bestof_v1`

结论：

- 第一名是 `naive_mixed_avg`
- 最佳 PS-family 是 `risky_ps_old`
- 不满足 “PS-family 第一”

### 2.2 v1 landscape 下的 eta 扩展诊断

输出目录：

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_extended_diagnostic_v1`

结论：

- 扩大 eta 后，PS-family 仍然无法压过 `naive_mixed_avg`
- `direct_multistage_exp3` 和 `epsilon_exp3` 反而更强
- 这说明 v1 下问题不是简单调参

### 2.3 v10_avg_baited 6 方法 pilot

输出目录：

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v1`

pilot 结论：

- rank 1 是 `risky_ps_old`
- `direct_multistage_exp3`
- `epsilon_exp3`
- `naive_mixed_avg`
- `random_path`

都排在最佳 PS-family 后面

说明：

- 这是 `ps_favored_trap_v10_avg_baited` 新 landscape 的 pilot
- baseline 未改
- metric 未改

### 2.4 v10_avg_baited full 13 方法 best-of sweep

输出目录：

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v2_full13`

关键产物：

- `sweep_runs_manifest.json`
- `sweep_results_long.csv`
- `sweep_results_long.json`
- `best_params_by_method.csv`
- `best_params_by_method.json`
- `best_of_compare.csv`
- `best_of_compare.json`
- `best_of_compare.md`

实验口径：

- runner: `scripts/run_barriershare_controlled_sim.py`
- sweep script reference: `scripts/run_barriershare_controlled_bestof_sweep.py`
- tree spec: `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json`
- tree_spec_role_mode: `spec_or_agent_id`
- tree_spec_cost_mode: `ps_favored_trap_v10_avg_baited`
- trap_switch_denominator: `8`
- horizon: `1000`
- seeds: `0 1 2 3 4`
- cost_noise: `0.02`
- specialist_fraction: `0.15`

明确未改内容：

- baseline 未改
- metric 未改
- 没有新增 landscape
- 没有改 `scripts/run_barriershare_controlled_sim.py`
- 没有改 `scripts/run_barriershare_controlled_bestof_sweep.py`

## 3. full13 best-of 完整排名

| rank | method | best_eta | best_epsilon | regret/T | overall avg cost | tail20 avg cost | post-switch avg regret | target good frac | trap frac | shared path frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | direct_multistage_exp3_local | 2.0 | 0.0 | 0.041092 | 0.057800 | 0.020000 | 0.040286 | 0.843200 | 0.074000 | 0.986800 |
| 2 | risky_ps_old | 1.5 | 0.0 | 0.042292 | 0.059000 | 0.150000 | 0.041200 | 0.844600 | 0.080000 | 0.986600 |
| 3 | direct_multistage_exp3 | 2.0 | 0.0 | 0.046292 | 0.063000 | 0.010000 | 0.046000 | 0.842600 | 0.074600 | 0.986800 |
| 4 | epsilon_exp3 | 2.0 | 0.0 | 0.046292 | 0.063000 | 0.010000 | 0.046000 | 0.842600 | 0.074600 | 0.986800 |
| 5 | naive_mixed_avg |  |  | 0.047092 | 0.063800 | 0.020000 | 0.061543 | 0.824200 | 0.000000 | 1.000000 |
| 6 | risky_ps_linear | 1.5 | 0.0 | 0.051092 | 0.067800 | 0.010000 | 0.051257 | 0.831800 | 0.080800 | 0.985400 |
| 7 | risky_ps | 1.0 | 0.005 | 0.085492 | 0.102200 | 0.010000 | 0.090114 | 0.797800 | 0.032400 | 0.990200 |
| 8 | risky_ps_ix | 1.5 | 0.0 | 0.088892 | 0.105600 | 0.030000 | 0.094457 | 0.782600 | 0.082400 | 0.984000 |
| 9 | risky_ps_safe_conditional_ix | 2.0 | 0.01 | 0.104892 | 0.121600 | 0.000000 | 0.113657 | 0.772400 | 0.035200 | 0.987200 |
| 10 | risky_ps_safe_conditional | 2.0 | 0.01 | 0.106092 | 0.122800 | 0.000000 | 0.115029 | 0.771600 | 0.035000 | 0.987000 |
| 11 | risky_ps_direct_cost | 0.5 | 0.0 | 0.339292 | 0.356000 | 0.150000 | 0.376971 | 0.477200 | 0.083000 | 0.972200 |
| 12 | random_path |  |  | 0.708892 | 0.725600 | 0.750000 | 0.760514 | 0.065400 | 0.081800 | 0.927000 |
| 13 | naive_mixed |  |  | 0.723092 | 0.739800 | 0.910000 | 0.834114 | 0.054600 | 0.000000 | 1.000000 |

## 4. full13 结论

最佳 PS-family：

- `risky_ps_old`
- best params: `eta=1.5, epsilon=0.0`
- regret/T = `0.042291925539009304`
- overall avg cost = `0.0590`
- tail20 avg cost = `0.1500`
- post-switch avg regret = `0.0412`
- target good frac = `0.8446`
- trap frac = `0.0800`
- shared path frac = `0.9866`

full13 判断：

1. full 13 方法下，best-of rank 1 不是 PS-family。
2. rank 1 是 `direct_multistage_exp3_local`，best params 为 `eta=2.0`。
3. `direct_multistage_exp3_local` 以 `0.0012` regret/T 优势反超 `risky_ps_old`。
4. `naive_mixed_avg` 仍然被最佳 PS-family 压住。
5. `direct_multistage_exp3` 和 `epsilon_exp3` 仍然排在最佳 PS-family 后面。
6. 其他 PS 变体没有超过 `risky_ps_old`。
7. 新的非 PS-family 意外反超是 `direct_multistage_exp3_local`。

因此：

- `ps_favored_trap_v10_avg_baited` 的 pilot 虽然满足最低目标
- 但 full13 不满足 “PS-family 第一”
- 所以这版结果现在还不能标记为“公平 baseline 下的新 landscape / stress-test controlled sim 候选主结果”

## 5. 当前状态判断

当前应明确区分：

- v1 结果：原始 landscape，不满足目标
- v10_avg_baited pilot：6 方法子集满足最低目标
- v10_avg_baited full13：不满足目标，因为 `direct_multistage_exp3_local` 反超

本轮之后的结论是：

- 这是 `v10_avg_baited` 新 landscape 的结果，不是 v1
- baseline 未改
- metric 未改
- full13 结果不能作为更正式的 controlled sim 候选主结果

## 6. 下一步约束

如果继续推进，必须注意：

- 不要因为这次 full13 失败就去静默改 baseline
- 不要静默改 metric
- 不要把这次 full13 失败说成“已经正式达标”

按当前用户指令，本轮 full13 失败后应直接停止，不追加新实验。

## 7. 给下一轮执行 agent 的 prompt

```text
你在 /home/ubuntu/data/PSAgent 中继续处理 BarrierShare controlled sim。

先阅读：
- notes/ps_favored_trap_bestof_goal_status.md
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v2_full13/best_of_compare.md
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v2_full13/best_params_by_method.csv
- outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v2_full13/sweep_results_long.csv

当前事实：
- 本轮 full13 已经跑完
- 输出目录是 outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_v2_full13
- full13 rank 1 是 direct_multistage_exp3_local，不是 PS-family
- 最佳 PS-family 仍是 risky_ps_old，best params 为 eta=1.5, epsilon=0.0
- baseline 未改
- metric 未改
- 这是 v10_avg_baited，不是 v1

如果用户没有新的明确指令：
- 不要追加新实验
- 不要改 baseline
- 不要改 metric
- 不要改 environment
- 只基于现有结果做汇报或整理
```

## 8. 2026-04-25 补充：v10_avg_baited 在 eta <= 0.5 约束下的 full13 重跑

### 本轮为何重跑

上一轮 `v10_avg_baited` full13 不达标，直接原因是：

- `direct_multistage_exp3_local` 用 `eta=2.0` 拿到 rank 1
- 它以 `regret/T = 0.041092` 反超 `risky_ps_old`

因此本轮专门做一个受限 sweep：

- 保持 `ps_favored_trap_v10_avg_baited` environment 不变
- baseline 不变
- metric 不变
- 只允许所有带学习率的方法在 `eta <= 0.5` 的范围内选最优参数

### 输出目录

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_eta_leq_05_full13_v1`

### 本轮参数网格

- eta grid: `0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5`
- epsilon grid: `0.0, 0.005, 0.01, 0.02, 0.05, 0.1`

eta + epsilon 方法：

- `risky_ps_old`
- `risky_ps`
- `risky_ps_ix`
- `risky_ps_safe_conditional`
- `risky_ps_safe_conditional_ix`
- `risky_ps_direct_cost`
- `epsilon_exp3`
- `risky_ps_linear`

eta-only 方法：

- `direct_multistage_exp3`
- `direct_multistage_exp3_local`

固定单次方法：

- `naive_mixed`
- `naive_mixed_avg`
- `random_path`

### 本轮 full13 best-of 表

| rank | method | best_eta | best_epsilon | regret/T | overall avg cost | tail20 avg cost | post-switch avg regret | target good frac | trap frac | shared path frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | naive_mixed_avg |  |  | 0.047092 | 0.063800 | 0.020000 | 0.061543 | 0.824200 | 0.000000 | 1.000000 |
| 2 | risky_ps_old | 0.4 | 0.0 | 0.054292 | 0.071000 | 0.000000 | 0.049429 | 0.836600 | 0.061800 | 0.986600 |
| 3 | direct_multistage_exp3 | 0.4 | 0.0 | 0.056092 | 0.072800 | 0.010000 | 0.052857 | 0.832200 | 0.061800 | 0.987400 |
| 4 | epsilon_exp3 | 0.4 | 0.0 | 0.056092 | 0.072800 | 0.010000 | 0.052857 | 0.832200 | 0.061800 | 0.987400 |
| 5 | risky_ps_linear | 0.2 | 0.0 | 0.057492 | 0.074200 | 0.000000 | 0.050114 | 0.838800 | 0.045000 | 0.987000 |
| 6 | risky_ps | 0.3 | 0.0 | 0.062292 | 0.079000 | 0.000000 | 0.057429 | 0.834800 | 0.048200 | 0.988000 |
| 7 | risky_ps_ix | 0.3 | 0.0 | 0.065692 | 0.082400 | 0.000000 | 0.061314 | 0.831200 | 0.048200 | 0.988000 |
| 8 | direct_multistage_exp3_local | 0.4 | 0.0 | 0.067892 | 0.084600 | 0.010000 | 0.063829 | 0.825600 | 0.061600 | 0.986200 |
| 9 | risky_ps_safe_conditional | 0.3 | 0.0 | 0.084092 | 0.100800 | 0.000000 | 0.082114 | 0.811000 | 0.049000 | 0.987600 |
| 10 | risky_ps_safe_conditional_ix | 0.3 | 0.0 | 0.084292 | 0.101000 | 0.000000 | 0.082343 | 0.810800 | 0.049000 | 0.987600 |
| 11 | risky_ps_direct_cost | 0.3 | 0.0 | 0.249492 | 0.266200 | 0.090000 | 0.271143 | 0.609400 | 0.052600 | 0.980400 |
| 12 | random_path |  |  | 0.708892 | 0.725600 | 0.750000 | 0.760514 | 0.065400 | 0.081800 | 0.927000 |
| 13 | naive_mixed |  |  | 0.723092 | 0.739800 | 0.910000 | 0.834114 | 0.054600 | 0.000000 | 1.000000 |

### 本轮结论

第一名是谁：

- `naive_mixed_avg`

最佳 PS-family 是谁：

- `risky_ps_old`
- best params: `eta=0.4, epsilon=0.0`

是否满足 “PS-family 第一”：

- 不满足

`direct_multistage_exp3_local` 是否仍反超：

- 不再反超
- 它在本轮只排第 8

`naive_mixed_avg` 是否仍被 PS 压住：

- 没有
- 它重新回到 rank 1，并压过了最佳 PS-family `risky_ps_old`

额外核对：

- `best_of_compare` 已覆盖 13 个方法
- 没有任何 `best_eta > 0.5`
- baseline 未改
- metric 未改
- environment 未改
- 输出目录未覆盖旧结果

## 9. 2026-04-26 补充：PS-family shared constant-init top4 变种实验

### 本轮新增的 4 个算法

本轮在不改 environment、不改 v10 landscape、不改 metric、也不改原 baseline 默认行为的前提下，新增了 4 个 shared constant-init 变种：

- `risky_ps_old_const_init`
- `risky_ps_const_init`
- `risky_ps_linear_const_init`
- `risky_ps_ix_const_init`

它们对应的原算法分别是：

- `risky_ps_old`
- `risky_ps`
- `risky_ps_linear`
- `risky_ps_ix`

唯一改动点：

- shared 初始化不再由“子树里可上传的 shared 叶子数”决定
- 改成结构无关的 constant shared init

保持不变的内容：

- update rule 不变
- sampling rule 不变
- 分母与 importance weighting 不变
- metric 不变
- environment 不变
- `ps_favored_trap_v10_avg_baited` landscape 不变

### 本轮实验设置

本轮复用了最新版 `v10_avg_baited` 且 `eta <= 0.5` 的 full13 设定，并扩展成 17 方法对比。

输出目录：

`outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v10_avg_baited_sweep_bestof_eta_leq_05_full13_plus_const_init_top4ps_v1`

参数网格：

- eta grid: `0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5`
- epsilon grid: `0.0, 0.005, 0.01, 0.02, 0.05, 0.1`

### 表 A：按 regret/T 排序的主表结论

关键结论：

- rank 1 仍然是 `naive_mixed_avg`
- 最佳 PS-family 仍然是原版 `risky_ps_old`
- 4 个 const-init 变种没有一个超过对应原版

前 12 名如下：

| rank | method | best_eta | best_epsilon | regret/T | overall avg cost | tail20 avg cost | post-switch avg regret | target good frac | trap frac | shared path frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | naive_mixed_avg |  |  | 0.047092 | 0.063800 | 0.020000 | 0.061543 | 0.824200 | 0.000000 | 1.000000 |
| 2 | risky_ps_old | 0.4 | 0.0 | 0.054292 | 0.071000 | 0.000000 | 0.049429 | 0.836600 | 0.061800 | 0.986600 |
| 3 | direct_multistage_exp3 | 0.4 | 0.0 | 0.056092 | 0.072800 | 0.010000 | 0.052857 | 0.832200 | 0.061800 | 0.987400 |
| 4 | epsilon_exp3 | 0.4 | 0.0 | 0.056092 | 0.072800 | 0.010000 | 0.052857 | 0.832200 | 0.061800 | 0.987400 |
| 5 | risky_ps_linear | 0.2 | 0.0 | 0.057492 | 0.074200 | 0.000000 | 0.050114 | 0.838800 | 0.045000 | 0.987000 |
| 6 | risky_ps_old_const_init | 0.4 | 0.0 | 0.061692 | 0.078400 | 0.000000 | 0.059029 | 0.825200 | 0.063600 | 0.986600 |
| 7 | risky_ps | 0.3 | 0.0 | 0.062292 | 0.079000 | 0.000000 | 0.057429 | 0.834800 | 0.048200 | 0.988000 |
| 8 | risky_ps_ix | 0.3 | 0.0 | 0.065692 | 0.082400 | 0.000000 | 0.061314 | 0.831200 | 0.048200 | 0.988000 |
| 9 | direct_multistage_exp3_local | 0.4 | 0.0 | 0.067892 | 0.084600 | 0.010000 | 0.063829 | 0.825600 | 0.061600 | 0.986200 |
| 10 | risky_ps_linear_const_init | 0.2 | 0.0 | 0.076892 | 0.093600 | 0.130000 | 0.072286 | 0.819600 | 0.038800 | 0.989400 |
| 11 | risky_ps_ix_const_init | 0.2 | 0.0 | 0.079692 | 0.096400 | 0.000000 | 0.075257 | 0.815600 | 0.038800 | 0.988200 |
| 12 | risky_ps_const_init | 0.2 | 0.0 | 0.081292 | 0.098000 | 0.000000 | 0.077086 | 0.814800 | 0.038000 | 0.989000 |

### 表 B：按 post-switch avg regret 排序的主表结论

关键结论：

- 按 post-switch avg regret 看，第一名是 `risky_ps_old`
- `risky_ps_linear` 也仍然明显优于对应的 const-init 版本
- 4 个 const-init 变种在 post-switch 指标上同样全部变差

前 12 名如下：

| rank | method | best_eta | best_epsilon | regret/T | overall avg cost | tail20 avg cost | post-switch avg regret | target good frac | trap frac | shared path frac |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | risky_ps_old | 0.4 | 0.0 | 0.054292 | 0.071000 | 0.000000 | 0.049429 | 0.836600 | 0.061800 | 0.986600 |
| 2 | risky_ps_linear | 0.2 | 0.0 | 0.057492 | 0.074200 | 0.000000 | 0.050114 | 0.838800 | 0.045000 | 0.987000 |
| 3 | direct_multistage_exp3 | 0.4 | 0.0 | 0.056092 | 0.072800 | 0.010000 | 0.052857 | 0.832200 | 0.061800 | 0.987400 |
| 4 | epsilon_exp3 | 0.4 | 0.0 | 0.056092 | 0.072800 | 0.010000 | 0.052857 | 0.832200 | 0.061800 | 0.987400 |
| 5 | risky_ps | 0.3 | 0.0 | 0.062292 | 0.079000 | 0.000000 | 0.057429 | 0.834800 | 0.048200 | 0.988000 |
| 6 | risky_ps_old_const_init | 0.4 | 0.0 | 0.061692 | 0.078400 | 0.000000 | 0.059029 | 0.825200 | 0.063600 | 0.986600 |
| 7 | risky_ps_ix | 0.3 | 0.0 | 0.065692 | 0.082400 | 0.000000 | 0.061314 | 0.831200 | 0.048200 | 0.988000 |
| 8 | naive_mixed_avg |  |  | 0.047092 | 0.063800 | 0.020000 | 0.061543 | 0.824200 | 0.000000 | 1.000000 |
| 9 | direct_multistage_exp3_local | 0.4 | 0.0 | 0.067892 | 0.084600 | 0.010000 | 0.063829 | 0.825600 | 0.061600 | 0.986200 |
| 10 | risky_ps_linear_const_init | 0.2 | 0.0 | 0.076892 | 0.093600 | 0.130000 | 0.072286 | 0.819600 | 0.038800 | 0.989400 |
| 11 | risky_ps_ix_const_init | 0.2 | 0.0 | 0.079692 | 0.096400 | 0.000000 | 0.075257 | 0.815600 | 0.038800 | 0.988200 |
| 12 | risky_ps_const_init | 0.2 | 0.0 | 0.081292 | 0.098000 | 0.000000 | 0.077086 | 0.814800 | 0.038000 | 0.989000 |

### 原版 vs const-init 配对对比

#### 1. risky_ps_old vs risky_ps_old_const_init

- regret/T 变差：`+0.0074`
- post-switch avg regret 变差：`+0.0096`
- target good frac 下降：`-0.0114`
- trap frac 略升：`+0.0018`
- shared path frac 基本不变
- 说明：树结构先验对 `risky_ps_old` 有实际帮助，但影响幅度还算中等

#### 2. risky_ps vs risky_ps_const_init

- regret/T 变差：`+0.0190`
- post-switch avg regret 变差：`+0.019657`
- target good frac 下降：`-0.0200`
- trap frac 下降：`-0.0102`
- shared path frac 略升：`+0.0010`
- 说明：const-init 没有让它更会避 trap，反而 overall 与 post-switch 都更差，说明结构先验对有限样本性能很重要

#### 3. risky_ps_linear vs risky_ps_linear_const_init

- regret/T 变差：`+0.0194`
- post-switch avg regret 变差：`+0.022171`
- target good frac 下降：`-0.0192`
- trap frac 下降：`-0.0062`
- shared path frac 略升：`+0.0024`
- 说明：线性 shared mass 版本同样依赖结构初始化，const-init 后明显退化

#### 4. risky_ps_ix vs risky_ps_ix_const_init

- regret/T 变差：`+0.0140`
- post-switch avg regret 变差：`+0.013943`
- target good frac 下降：`-0.0156`
- trap frac 下降：`-0.0094`
- shared path frac 近乎不变：`+0.0002`
- 说明：IX 版本的下降比 `risky_ps` / `risky_ps_linear` 略小，但仍然明显变差

### 对 regret 上界是否同阶的保守判断

保守判断：

- 从更新形式看，推测 asymptotic regret order 大概率不变
- 但 initialization-dependent constant / prior term 很可能发生变化

依据：

1. 本轮只改了 shared 初始化，不改 update rule
2. 不改 sampling rule
3. 不改重要性加权分母 / normalization 结构
4. 不改 loss estimator 形式

因此更像是：

- 初始势函数
- 初始先验质量分配
- 有限时间内的探索偏置

发生了变化，而不是算法主干的渐近更新机制发生变化。

从实验现象看：

- 4 个 const-init 版本都比原版差
- 尤其是 `risky_ps`, `risky_ps_linear`, `risky_ps_ix` 的 finite-sample 指标恶化明显

这说明：

- 即使 asymptotic order 可能不变
- 树结构初始化在当前 horizon=1000 的有限样本 regime 里非常重要
- 它对 target-good 命中率和 post-switch 恢复速度都有明显帮助
