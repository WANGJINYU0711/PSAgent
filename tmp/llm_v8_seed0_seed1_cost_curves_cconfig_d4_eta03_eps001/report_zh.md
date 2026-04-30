# Seed0/Seed1 cost curves and early-stop diagnostic

## 数据来源

- seed0 direct/epsilon：原 C 版 3-method run。
- seed0 risky_ps：PS-family C run 中的 `risky_ps`。
- seed1 三方法：confirmatory seed1 run。
- cost：`raw_total_cost = terminal + calibrated reasoning + path`；modecost 是 report-only，不进入 total。
- switch episode：t=25。

## Checkpoint cumulative average ranks

| seed | t=10 | t=25 | t=50 | t=75 | t=90 | t=100 |
|---|---|---|---|---|---|---|
| seed0 | 1. risky_ps=5.24, 2. epsilon=6.59, 3. direct=6.91 | 1. risky_ps=4.93, 2. direct=5.73, 3. epsilon=6.47 | 1. risky_ps=7.45, 2. epsilon=8.83, 3. direct=9.33 | 1. risky_ps=8.24, 2. direct=8.68, 3. epsilon=9.38 | 1. risky_ps=8.72, 2. direct=9.22, 3. epsilon=10.60 | 1. risky_ps=9.01, 2. direct=9.53, 3. epsilon=10.50 |
| seed1 | 1. risky_ps=4.39, 2. epsilon=5.21, 3. direct=6.00 | 1. risky_ps=4.71, 2. epsilon=5.11, 3. direct=6.13 | 1. risky_ps=8.88, 2. epsilon=9.29, 3. direct=9.50 | 1. epsilon=9.10, 2. direct=9.18, 3. risky_ps=9.34 | 1. direct=9.60, 2. epsilon=9.62, 3. risky_ps=10.64 | 1. epsilon=9.61, 2. direct=9.90, 3. risky_ps=10.50 |

## Checkpoint leader vs final leader

| seed | final leader | checkpoint leader correctness |
|---|---|---|
| seed0 | `risky_ps` | t=10 risky_ps✓, t=25 risky_ps✓, t=50 risky_ps✓, t=75 risky_ps✓, t=90 risky_ps✓, t=100 risky_ps✓ |
| seed1 | `epsilon` | t=10 risky_ps✗, t=25 risky_ps✗, t=50 risky_ps✗, t=75 epsilon✓, t=90 direct✗, t=100 epsilon✓ |

## Leader changes

### seed0
- t=1: leader=epsilon_exp3 avg=4.1897, runner_up=risky_ps, margin=0.0332
- t=3: leader=risky_ps avg=4.0819, runner_up=epsilon_exp3, margin=1.4900

### seed1
- t=1: leader=risky_ps avg=3.8178, runner_up=epsilon_exp3, margin=0.0070
- t=2: leader=direct_multistage_exp3 avg=4.5925, runner_up=epsilon_exp3, margin=0.0662
- t=3: leader=risky_ps avg=5.6967, runner_up=direct_multistage_exp3, margin=0.2409
- t=56: leader=epsilon_exp3 avg=8.7174, runner_up=risky_ps, margin=0.0581
- t=57: leader=risky_ps avg=8.8162, runner_up=epsilon_exp3, margin=0.0457
- t=58: leader=direct_multistage_exp3 avg=9.0989, runner_up=risky_ps, margin=0.0251
- t=59: leader=risky_ps avg=9.1452, runner_up=epsilon_exp3, margin=0.0429
- t=60: leader=epsilon_exp3 avg=9.1158, runner_up=direct_multistage_exp3, margin=0.1025
- t=64: leader=direct_multistage_exp3 avg=8.9489, runner_up=epsilon_exp3, margin=0.1588
- t=74: leader=epsilon_exp3 avg=9.1709, runner_up=direct_multistage_exp3, margin=0.0706
- t=78: leader=direct_multistage_exp3 avg=9.3561, runner_up=epsilon_exp3, margin=0.1411
- t=80: leader=epsilon_exp3 avg=9.5567, runner_up=direct_multistage_exp3, margin=0.0924
- t=90: leader=direct_multistage_exp3 avg=9.5980, runner_up=epsilon_exp3, margin=0.0193
- t=94: leader=epsilon_exp3 avg=9.3606, runner_up=direct_multistage_exp3, margin=0.0839

## 早停结论

1. t=25 绝对不能早停：这是 switch 前，全是 pre/trap fast-required，不能代表 post/deep target。
2. t=50 也不可靠：seed1 在 t=50 的累计第一是 `risky_ps`，最终第一却是 `epsilon_exp3`。这正好说明“前半段领先/落后”不一定决定最终。
3. t=75 在这两次 seed 上都预测对了最终 winner：seed0 是 `risky_ps`，seed1 是 `epsilon_exp3`。但这只能说明 t=75 比 t=50 稳，不代表可以无条件 hard stop。
4. t=90 仍可能发生短暂误判：seed1 在 t=90 的累计第一是 `direct`，但最后 t=100 是 `epsilon`。原因是 t=90 时 direct/epsilon margin 只有约 `0.02`，属于噪声级差距。
5. 因此早停可以作为 early-warning，但不建议作为 hard stop，除非同时满足：已经过 t>=75；累计第一连续稳定至少 10-15 个 episode；领先 margin 足够大，比如 >0.3 或 >0.5；rolling-5/rolling-10 没有反向趋势。

## 对“开始差以后还能追回吗”的回答

- 会。seed1 里 `epsilon_exp3` 在 t=50 不是第一，但最终追回并成为第一。
- 也会反过来：seed1 里 `risky_ps` t=50 第一，但后面被反超，最后第三。
- 所以在这个 LLM smoke 里，前 50 episode 的排序不能作为最终排序的可靠判断，尤其 switch 后 post/deep 只跑了 25 个 episode 时。

## 产物

- `cumulative_avg_total_cost_by_seed.svg`
- `episode_total_cost_by_seed.svg`
- `cumulative_avg_terminal_cost_by_seed.svg`
- `cumulative_avg_reasoning_cost_by_seed.svg`
- `cost_timeseries_long.csv`
- `early_stop_rank_checkpoints.csv`
- `early_stop_prediction_check.csv`
- `leader_by_t.csv`