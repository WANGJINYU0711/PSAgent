# Stage 4/5 contract prompt v1.2 prompt-only 回归报告

## 配置名

`llm_v8_stage45_contract_promptv12_promptonly_cconfig`

两个 fixed trace run：

- `tmp/llm_v8_stage45_contract_promptv12_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv12_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig/`

两个 run 均 `exit_code=0`。当前无 tmux session。

## 结论

v1.2 明显退化，不触发 clean-100 fixed trace。

v1.2 的设计目标是加入三类防过修正：

- Stage 5 不要过度继承 Stage 4。
- ordinary_defer 不要吞掉真正 hard transfer。
- `can_send_mms=false` 不要被过度弱化。

实际结果是：这些规则让 Stage 4 变得更保守、更混乱，出现大量空 selected、错误 transfer、错误 repair_all，并破坏了 v1/v1.1 已经稳定的 dataset 16。

## Aggregate 对比

| run | n | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high >=10 | transfer |
|---|---:|---:|---:|---:|---:|---:|
| v1 fdddd r5 | 20 | 5.450 | 9.675 | 12.800 | 6 | 0 |
| v1.1 fdddd r5 | 20 | 5.400 | 9.575 | 12.700 | 3 | 2 |
| v1.2 fdddd r5 | 20 | 11.375 | 16.065 | 19.190 | 11 | 7 |
| v1 3patterns r3 | 36 | 7.264 | 11.675 | 14.758 | 16 | 1 |
| v1.1 3patterns r3 | 36 | 6.236 | 10.799 | 13.882 | 10 | 1 |
| v1.2 3patterns r3 | 36 | 11.833 | 16.938 | 20.021 | 23 | 11 |

## fdddd r5 明细

| dataset | v1.1 terminal | v1.1 mean | v1.2 terminal | v1.2 mean | v1.2 final counts | 判断 |
|---:|---|---:|---|---:|---|---|
| 2 | `[0,0,0,0,0]` | 0.0 | `[0,0,0,18.5,0]` | 3.7 | `repair_all:4, transfer:1` | 退化，permission 链回潮 transfer |
| 10 | `[6,22.5,6,6,22.5]` | 12.6 | `[6,12,21,20.5,10]` | 13.9 | `repair_subset:3, transfer:1, repair_all:1` | 小幅退化，形态更乱 |
| 13 | `[21,6,6,6,6]` | 9.0 | `[22.5,6,22.5,6,22.5]` | 15.9 | `transfer:3, repair_subset:2` | 明显退化 |
| 16 | `[0,0,0,0,0]` | 0.0 | `[22.5,22.5,0,15,0]` | 12.0 | `transfer:2, repair_all:2, repair_subset:1` | 严重退化，破坏稳定任务 |

## 3patterns r3 明细

| dataset | pattern | v1.1 mean | v1.2 terminal | v1.2 mean | 判断 |
|---:|---|---:|---|---:|---|
| 2 | ddddd | 0.0 | `[0,0,0]` | 0.0 | 持平 |
| 2 | fdddd | 0.0 | `[0,18.5,0]` | 6.167 | 退化 |
| 2 | ffddd | 0.0 | `[0,15,0]` | 5.0 | 退化 |
| 10 | ddddd | 6.0 | `[10,22.5,20.5]` | 17.667 | 明显退化 |
| 10 | fdddd | 11.5 | `[21,10,10]` | 13.667 | 退化 |
| 10 | ffddd | 12.0 | `[23.5,23.5,12]` | 19.667 | 明显退化 |
| 13 | ddddd | 11.0 | `[22.5,14.5,22.5]` | 19.833 | 明显退化 |
| 13 | fdddd | 14.667 | `[6,25.5,22.5]` | 18.0 | 退化 |
| 13 | ffddd | 19.667 | `[17,17,17]` | 17.0 | 小幅改善但仍差 |
| 16 | ddddd | 0.0 | `[15,0,22.5]` | 12.5 | 严重退化 |
| 16 | fdddd | 0.0 | `[0,0,15]` | 5.0 | 退化 |
| 16 | ffddd | 0.0 | `[0,0,22.5]` | 7.5 | 退化 |

## 真实因果链

### 1. Stage 4 出现“空 selected + repairability=repairable”自相矛盾

代表样本：dataset 2 / fdddd / terminal 18.5。

Stage 4 raw：

- 所有 blocker `should_repair=false`。
- `repairability=repairable`。
- `contract_self_check.local_repair_now_blocker_ids=[]`。
- `hard_transfer_blocker_ids=[]`。

Stage 5：

- 因 Stage 4 没有 selected repair，最终 transfer。
- terminal reasons 包括 `invalid_local_transfer_floor_18`。

判断：v1.2 的 self-check 变长后，LLM 会填出“格式正确但语义空心”的检查对象。它没有帮助执行，反而挤压/扰乱了 should_repair 决策。

### 2. Stage 4 把本地修复链错判成 hard/不可修，导致保守 transfer

代表样本：dataset 13、16 多个 transfer。

模式：

- Stage 4 把本来应 selected 的 local blockers 全部 `should_repair=false`。
- 有时 `hard_transfer_blocker_ids=[]`，却仍然走 transfer。
- 有时错误把 downstream/local blockers 标成 `requires hybrid handling`。
- Stage 5 继承空 repair plan，输出 transfer。

判断：为了防止 ordinary_defer 吞 hard transfer，v1.2 加了 hard-transfer 可审计字段，但 LLM 反而过度使用“hard/transfer”词汇，或者输出空 hard list 仍 transfer。prompt-only self-check 不能约束一致性。

### 3. dataset 10 仍有两类旧问题，并新增混乱

三种失败同时存在：

- 错误 repair_all：把 `user_abroad_roaming_disabled_on`、`data_usage_exceeded` 也 selected，terminal 10。
- 错误 transfer：local chain 未 selected，terminal 20.5/22.5/23.5。
- 错误 repair_subset：selected 只剩 downstream 或 account/usage，漏 upstream local chain，terminal 21。

判断：v1.2 没有解决 ordinary_defer contract，反而让模型在 selected/ordinary/hard 三类间摆动更大。

### 4. Stage 5 self-check 导出变完整，但 correctness 仍不可信

v1.2 records 里 Stage 4/5 self-check 基本全量出现，这是唯一诊断改善。

但 self-check 经常自相矛盾：

- `selected_deferred_partition_ok=true`，实际 selected/deferred 与分类列表不一致。
- `transfer_has_concrete_hard_blocker_ok=true`，实际 hard list 为空。
- `active_local_prerequisites_selected_ok=true`，实际 upstream prerequisites 被 deferred。

判断：self-check 可以作为 prompt-adherence artifact，但不能当 correctness signal。

## 为什么没有跑 clean-100

没有触发 gating。

v1.2 不仅没有明显进步，还破坏了 v1.1 中已经稳定的 dataset 2/16：

- dataset 16 从 v1.1 全部 0 变成 fdddd mean 12.0，3patterns 多个 transfer/subset。
- v1.2 transfer count 从 v1.1 的 2/20、1/36 上升到 7/20、11/36。
- high terminal 从 v1.1 的 3/20、10/36 上升到 11/20、23/36。

因此不应扩到 clean-100。

## 下一步建议

不要沿着 v1.2 继续加更长 prompt。v1.2 说明“复杂自检 + 三分类审计”对 gpt-4o-mini 会诱发保守/空心输出。

建议回退到 v1.1 作为基线，再做极小 v1.1b：

1. 保留 v1.1 的 dataset 2 permission closure，因为它有效。
2. 删除 v1.2 的长 self-check 分类数组和 hard_transfer_reason_by_blocker 复杂 schema。
3. Stage 4 只加一条轻规则：`transfer_required` 不能在 `hard_transfer_blocker_ids` 为空时输出；如果无法列出 hard blocker，就选 local repair subset 或 ordinary defer。
4. Stage 5 只加一条轻规则：selected/deferred 只能包含 input blocker ids，`can_send_mms` 不能作为 blocker id。
5. 不再要求 LLM 写大段 self-check；最多 required 一个短字段：`contract_self_check: {"has_concrete_transfer_blocker": bool, "ids_are_input_blockers_only": bool}`。

