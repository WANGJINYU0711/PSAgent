# Stage Match/Mismatch Cost Probe v1 报告

## 实验配置名

`llm_v8_stage_mismatch_cost_probe_v1_stage123_4omini_stage45_41mini_promptv11b_cconfig`

最终合并输出：

- `tmp/llm_v8_stage_mismatch_cost_probe_v1_stage123_4omini_stage45_41mini_promptv11b_combined_r3_cconfig/records.json`
- `tmp/llm_v8_stage_mismatch_cost_probe_v1_stage123_4omini_stage45_41mini_promptv11b_combined_r3_cconfig/summary.json`
- `tmp/llm_v8_stage_mismatch_cost_probe_v1_stage123_4omini_stage45_41mini_promptv11b_combined_r3_cconfig/records_summary.csv`

原始 run：

- `tmp/llm_v8_stage_mismatch_cost_probe_v1_stage123_4omini_stage45_41mini_promptv11b_r3_cconfig/`
- 该 run 在 17/24 后遇到 OpenAI connection error，`exit_code=1`，但 dataset 1/3 的 12 条完整可用。

recovery run：

- `tmp/llm_v8_stage_mismatch_cost_probe_v1_stage123_4omini_stage45_41mini_promptv11b_dtasks_recovery_r3_cconfig/`
- 该 run 用于补齐 dataset 5/11，`exit_code=0`。

冗余 dataset 11 recovery 已按用户要求 kill，不纳入最终合并。

当前无遗留 tmux session。

## 和上一版/之前实验的区别

这不是新的 prompt 版本，不改 Stage 4/5 prompt 内容；本轮使用 v1.1b prompt-only 作为执行层基线：

- `PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B=1`
- `PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1`
- `PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1`
- `PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1`

本轮新变化是：

1. 固定路径 mismatch probe，而不是 prompt 回归。
2. Stage 1/2/3 模型为 `gpt-4o-mini`。
3. Stage 4/5 模型为 `gpt-4.1-mini`，通过 `PSAGENT_TELECOM_STAGE45_MODEL=gpt-4.1-mini` 覆盖。
4. 抽取 2 个 mostly-f 任务和 2 个 mostly-d 任务。
5. 每个任务分别跑 `fffff` 和 `ddddd` fixed path，各 3 次。

stage model 已在 records 的 `stage_resource_summary` 中确认：

- Stage 1/2/3: `gpt-4o-mini`
- Stage 4/5: `gpt-4.1-mini`

## 任务抽样

| dataset | task type | required pattern | expected | task 简述 |
|---:|---|---|---|---|
| 1 | mostly_f | `fffff` | repair_all | airplane + Wi-Fi calling + storage permission |
| 3 | mostly_f | `fffff` | repair_all | airplane + Wi-Fi calling + SMS permission |
| 5 | mostly_d | `ddddd` | repair_subset | local MMS chain + `data_usage_exceeded` ordinary defer |
| 11 | mostly_d | `ddddd` | repair_subset | local MMS chain + `data_usage_exceeded` ordinary defer |

匹配定义：

- mostly_f + `fffff` = matched。
- mostly_f + `ddddd` = mismatched。
- mostly_d + `ddddd` = matched。
- mostly_d + `fffff` = mismatched。

## 总体结论

如果主指标用 `raw_total_cost`，实验支持你的假设：mismatched total cost 高于 matched。

| group | n | raw_total_cost mean | terminal mean | raw_total_cost_with_token_penalty mean | clear | aux | exact_match |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched | 12 | 8.078 | 3.500 | 27.848 | 9/12 | 8/12 | 9/12 |
| mismatched | 12 | 13.953 | 7.083 | 37.266 | 6/12 | 6/12 | 6/12 |

差值：

- mismatched `raw_total_cost` 比 matched 高 `+5.875`。
- mismatched terminal mean 比 matched 高 `+3.583`。
- mismatched `raw_total_cost_with_token_penalty` 比 matched 高 `+9.418`。
- mismatched clear/aux/exact 都更低。

但有一个重要 caveat：

- 对 mostly_f 任务，matched `fffff` 的 `raw_total_cost` 很低，但 `raw_total_cost_with_token_penalty` 很高，因为 all-fast path 触发 fast token over-budget soft penalty。
- 所以如果你的“total cost”指原始 `raw_total_cost`，结论很干净。
- 如果你的“total cost”指 `raw_total_cost_with_token_penalty`，总体仍支持 mismatched 更高，但 mostly_f 子集会反向，因为 fast-token penalty 把 matched all-fast path 抬高。

## 按任务类型和路径

| task type | path | match? | n | raw_total_cost mean | terminal mean | raw_total_cost_with_token_penalty mean | clear | aux | exact_match |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| mostly_f | `fffff` | matched | 6 | 2.821 | 0.000 | 42.362 | 6/6 | 6/6 | 6/6 |
| mostly_f | `ddddd` | mismatched | 6 | 6.567 | 0.000 | 6.567 | 6/6 | 6/6 | 6/6 |
| mostly_d | `ddddd` | matched | 6 | 13.334 | 7.000 | 13.334 | 3/6 | 2/6 | 3/6 |
| mostly_d | `fffff` | mismatched | 6 | 21.340 | 14.167 | 67.965 | 0/6 | 0/6 | 0/6 |

解读：

- mostly_f：all-d 不匹配不会破坏 terminal，仍 6/6 clear，但 raw_total_cost 从 2.821 升到 6.567，主要是 deep path reasoning cost 更高。
- mostly_d：all-f 不匹配显著破坏执行质量，terminal 从 7.000 升到 14.167，clear/aux 从部分成功变成 0/6。
- 这组样本中，主要证明力来自 mostly_d tasks：deep-required 任务用 all-fast path 执行明显更差、更贵。

## 按 dataset 明细

| dataset | required | path | match? | terminal values | raw_total_cost values | raw_total_cost mean | clear | aux | final counts |
|---:|---|---|---|---|---|---:|---:|---:|---|
| 1 | `fffff` | `fffff` | matched | `[0,0,0]` | `[2.842,2.813,2.816]` | 2.824 | 3/3 | 3/3 | repair_all:3 |
| 1 | `fffff` | `ddddd` | mismatched | `[0,0,0]` | `[6.832,6.444,6.870]` | 6.715 | 3/3 | 3/3 | repair_all:3 |
| 3 | `fffff` | `fffff` | matched | `[0,0,0]` | `[2.815,2.822,2.815]` | 2.817 | 3/3 | 3/3 | repair_all:3 |
| 3 | `fffff` | `ddddd` | mismatched | `[0,0,0]` | `[6.409,6.410,6.436]` | 6.418 | 3/3 | 3/3 | repair_all:3 |
| 5 | `ddddd` | `ddddd` | matched | `[12,0,6]` | `[19.244,8.293,13.654]` | 13.730 | 2/3 | 1/3 | repair_all:1, repair_subset:2 |
| 5 | `ddddd` | `fffff` | mismatched | `[14,15,14]` | `[21.096,22.192,21.640]` | 21.643 | 0/3 | 0/3 | repair_all:1, repair_subset:2 |
| 11 | `ddddd` | `ddddd` | matched | `[12,0,12]` | `[17.163,4.655,16.998]` | 12.939 | 1/3 | 1/3 | repair_all:2, repair_subset:1 |
| 11 | `ddddd` | `fffff` | mismatched | `[14,14,14]` | `[21.204,20.926,20.982]` | 21.037 | 0/3 | 0/3 | repair_all:1, repair_subset:2 |

## 几种 cost 口径

| group | raw_total_cost | raw_terminal_penalty | raw_total_cost_with_token_penalty | path cost | reasoning cost | API USD raw | total tokens |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched | 8.078 | 3.500 | 27.848 | 0.068 | 4.509 | 0.013712 | 60,998 |
| mismatched | 13.953 | 7.083 | 37.266 | 0.068 | 6.802 | 0.010004 | 45,604 |

按 task type：

| task type/path | raw_total_cost | terminal | token-penalty total | reasoning | API USD raw | total tokens |
|---|---:|---:|---:|---:|---:|---:|
| mostly_f / matched `fffff` | 2.821 | 0.000 | 42.362 | 2.762 | 0.008017 | 37,137 |
| mostly_f / mismatched `ddddd` | 6.567 | 0.000 | 6.567 | 6.489 | 0.008782 | 48,205 |
| mostly_d / matched `ddddd` | 13.334 | 7.000 | 13.334 | 6.257 | 0.019408 | 84,859 |
| mostly_d / mismatched `fffff` | 21.340 | 14.167 | 67.965 | 7.115 | 0.011226 | 43,003 |

注意：

- API USD raw 不适合作为“性能成本”主指标，因为模型价格和 token shape 会让它与 terminal quality 不完全同向。
- `raw_total_cost_with_token_penalty` 对 all-fast path 非常严厉，尤其 mostly_f matched `fffff`，这会掩盖“matched path 语义上更便宜”的现象。
- 对当前要证明的“路径 profile 不匹配 total cost 更高”，最清楚的指标是 `raw_total_cost`。

## 真实输出因果链

### mostly_f 任务：不匹配 all-d 更贵但仍成功

dataset 1/3 都是小型 repair_all 任务：

- all-f matched：terminal 0，clear/aux/exact 全部 3/3。
- all-d mismatched：terminal 0，clear/aux/exact 也全部 3/3。
- 但 all-d raw_total_cost 约为 all-f 的 2.3 倍。

原因：

- deep path 对简单 task 使用更多 reasoning budget。
- Stage 4/5 仍能完整 selected 所有 blocker，所以 terminal 不受伤。
- 这类任务证明的是 “不匹配 deep overkill 增加 raw_total_cost”，不是 terminal failure。

### mostly_d 任务：不匹配 all-f 显著失败

dataset 5/11 是 partial-repair / ordinary-defer 类型，应该修完整 local MMS chain，并 defer `data_usage_exceeded`。

all-d matched 的常见结果：

- 有时正确输出 `repair_subset`，terminal 0 或 6。
- 有时误把 `data_usage_exceeded` 当作 selected / repair_all，terminal 12。
- clear 3/6，aux 2/6，说明 matched deep path 也有 execution noise。

all-f mismatched 的常见结果：

- Stage 4 输出 `partially_repairable`，但 selected 子集偏窄，漏 APN / Wi-Fi calling / app permission 等 downstream local blockers。
- terminal reasons 包含 `fast_path_on_deep_required_clear_failure_floor_14`。
- dataset 5: terminal `[14,15,14]`。
- dataset 11: terminal `[14,14,14]`。
- clear/aux/exact 全部 0/6。

这就是本轮最强因果链：deep-required partial-repair task 用 all-fast fixed path 时，Stage 4 selected/deferred contract 不完整，导致 terminal 和 total cost 同时变差。

## 是否符合预期

符合，但需要限定指标口径。

符合：

- 在 `raw_total_cost` 上，mismatched 明显高于 matched：13.953 vs 8.078。
- mostly_f 子集里，不匹配 all-d 比匹配 all-f 贵：6.567 vs 2.821。
- mostly_d 子集里，不匹配 all-f 比匹配 all-d 贵：21.340 vs 13.334。
- clear/aux/exact 也支持匹配更好：matched clear 9/12、aux 8/12、exact 9/12；mismatched clear/aux/exact 都是 6/12。

需要小心：

- `raw_total_cost_with_token_penalty` 在 mostly_f 子集上反向，因为 all-fast matched path 触发 fast token over-budget soft penalty。
- mostly_d matched 本身仍有 execution noise，尤其 dataset 5/11 的 `data_usage_exceeded` ordinary-defer contract 仍不稳。
- 原始 run 有 connection error，因此最终报告采用 “原始 dataset 1/3 + recovery dataset 5/11” 的合并结果；这不影响语义比较，但需要在记录中注明。

## 结论

这组 probe 可以支持你想证明的主张：

**按 `raw_total_cost`，fixed path agent profile 不匹配会比匹配更贵。**

更细一点：

- 对 mostly-f 简单任务，不匹配 all-d 是“成功但过度推理”，terminal 不变，raw_total_cost 上升。
- 对 mostly-d 复杂 partial-repair 任务，不匹配 all-f 是“执行质量下降 + terminal 高罚”，raw_total_cost 大幅上升。

如果后续要把这个做成更强证据，建议扩大到 10 个 mostly-f + 10 个 mostly-d，每个 matched/mismatched r3，并且主表同时报告 `raw_total_cost` 与 `raw_total_cost_with_token_penalty`，但结论主指标应明确使用 `raw_total_cost`。

