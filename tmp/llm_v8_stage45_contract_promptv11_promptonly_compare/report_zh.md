# Stage 4/5 contract prompt v1.1 prompt-only 回归报告

## 实验配置名

本轮配置名：

`llm_v8_stage45_contract_promptv11_promptonly_cconfig`

两个 fixed-trace run：

- `llm_v8_stage45_contract_promptv11_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig`
- `llm_v8_stage45_contract_promptv11_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig`

输出目录：

- `tmp/llm_v8_stage45_contract_promptv11_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig/`

两个 tmux run 均已结束，`exit_code.txt` 均为 `0`，当前没有遗留 tmux session。

## 和上一版 v1 的区别

上一版：

- `PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1=1`
- Stage 4/5 prompt 增加 selected/deferred contract、prerequisite closure、ordinary defer、repair_subset 语义。
- `contract_self_check` 是 optional report-only 字段；多数输出没有主动填。

本版 v1.1：

- 新增独立开关：`PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1=1`，保留旧 `V1` 兼容。
- prompt version 记录为 `stage45_contract_prompt_v1_1`。
- `contract_self_check` 在 prompt output contract 中改为 required diagnostic key，但仍然 report-only，不参与任何 terminal decision 或 normalizer 纠正。
- Stage 4 更明确：`can_be_deferred=true` 的 account / usage / subscription / quota / billing / policy / roaming-policy blocker 默认是 `ordinary_defer`，不能为了 `repair_all` 放进 selected。
- Stage 4 直接点名 dataset 10 形态：local MMS chain selected，`data_usage_exceeded` 和 `user_abroad_roaming_disabled_on` deferred，case-level `partially_repairable`，Stage 5 通常应输出 `repair_subset`。
- Stage 4 增加 permission closure：active app permission blocker 如果有 canonical local permission repair，且正在修 APN / Wi-Fi calling / network preference / MMS app downstream repairs，不应 defer 或漏掉该 permission blocker。
- Stage 5 增加 preservation rule：不要把 Stage 4 deferred 的 account/usage/policy ordinary-defer blocker 移入 selected 来制造 `repair_all`。
- 诊断脚本补了一个 report-only fallback：如果 legacy normalized Stage 5 output 丢掉 `contract_self_check`，未来导出会从 raw final JSON 里恢复该字段。这个补丁不影响本轮已完成 run 的行为或 cost。

仍然没有做：

- 没有改 PS。
- 没有改 terminal penalty。
- 没有加 retry。
- 没有让 normalizer 新增自动替 LLM 修 selected/deferred/final_action 的行为。
- 没有删数据。
- 没有跑 smoke。

验证：

`python -m py_compile envs/executors/telecom_llm_bench_executor.py scripts/run_llm_path_sweep_diagnostic.py scripts/run_llm_fixed_profile_trace_diagnostic.py`

## 总体结论

v1.1 方向是混合的，不满足“明显进步后进入 clean-100 fixed trace”的门槛。

最好的变化：

- dataset 2 被明显修好。`fdddd r5` 从 v1 的 `[0, 0, 12, 12, 0]` 变成 `[0, 0, 0, 0, 0]`；`3patterns r3` 里 `fdddd/ffddd/ddddd` 全部 terminal 0、全部 `repair_all`。这说明 app permission closure prompt 有效。
- dataset 16 继续稳定。两条 run 里所有 pattern 全部 terminal 0、全部 `repair_all`。
- `3patterns r3` 全局 terminal mean 从 `7.264` 降到 `6.236`，高罚 `>=10` 从 `16/36` 降到 `10/36`。
- Stage 4 raw `contract_self_check` 从几乎缺失变成全量出现，说明 required diagnostic key 对 Stage 4 有效。

主要坏消息：

- dataset 10 `fdddd` 仍不稳，而且从 v1 的“误选 ordinary defer 导致高罚 10/19”变成了 v1.1 的“偶发整案 transfer 22.5”。这不是想要的方向。
- dataset 13 在 v1 的 `fdddd` 原本稳定 `[6, 6, 6, 6, 6]`，v1.1 出现 `[21, 6, 6, 6, 6]`；`3patterns` 里 `fdddd/ddddd` 也回潮高罚。
- Stage 5 raw self-check 虽然出现，但会自信地声称 preservation ok，同时实际 selected/deferred 已经没有保留 Stage 4 plan。这个 self-check 目前只能作为 adherence 观测，不能作为 correctness 信号。
- Stage 5 仍会把 verification signal 当成 blocker id，例如输出 `selected_blocker_ids=["bad_wifi_calling","can_send_mms"]`，normalizer 去掉非法 id 后只剩 `bad_wifi_calling`，导致 selected 子集大幅缩水。

因此本轮没有继续启动 clean-100 fixed trace。原因不是预算保守，而是核心 gating 失败：v1.1 虽然修好了 dataset 2，但 dataset 10/13 的局部 contract 还没有变稳，直接扩到 100 只会把同一个 failure mode 放大。

## fdddd r5 对比 v1

| dataset | v1 terminal | v1 mean | v1 final counts | v1.1 terminal | v1.1 mean | v1.1 final counts | 判断 |
|---:|---|---:|---|---|---:|---|---|
| 2 | `[0, 0, 12, 12, 0]` | 4.80 | `repair_all:3, repair_subset:2` | `[0, 0, 0, 0, 0]` | 0.00 | `repair_all:5` | 明显改善 |
| 10 | `[6, 10, 10, 19, 10]` | 11.00 | `repair_subset:2, repair_all:3` | `[6, 22.5, 6, 6, 22.5]` | 12.60 | `repair_subset:3, transfer:2` | 退化，偶发 transfer |
| 13 | `[6, 6, 6, 6, 6]` | 6.00 | `repair_subset:5` | `[21, 6, 6, 6, 6]` | 9.00 | `repair_subset:5` | 退化，Stage 5 缩 selected |
| 16 | `[0, 0, 0, 0, 0]` | 0.00 | `repair_all:5` | `[0, 0, 0, 0, 0]` | 0.00 | `repair_all:5` | 持平稳定 |

Aggregate:

| run | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high terminal >=10 | transfer count |
|---|---:|---:|---:|---:|---:|
| v1 fdddd r5 | 5.450 | 9.675 | 12.800 | 6/20 | 0/20 |
| v1.1 fdddd r5 | 5.400 | 9.575 | 12.700 | 3/20 | 2/20 |

表面 mean 轻微下降，但 transfer 回潮不可接受；不能称为稳定进步。

## 3patterns r3 对比 v1

| dataset | pattern | v1 terminal | v1 mean | v1.1 terminal | v1.1 mean | 判断 |
|---:|---|---|---:|---|---:|---|
| 2 | ddddd | `[0, 12, 0]` | 4.00 | `[0, 0, 0]` | 0.00 | 改善 |
| 2 | fdddd | `[18.5, 12, 12]` | 14.17 | `[0, 0, 0]` | 0.00 | 明显改善 |
| 2 | ffddd | `[12, 12, 0]` | 8.00 | `[0, 0, 0]` | 0.00 | 明显改善 |
| 10 | ddddd | `[6, 10, 10]` | 8.67 | `[6, 6, 6]` | 6.00 | 改善 |
| 10 | fdddd | `[10, 6, 10]` | 8.67 | `[6, 6, 22.5]` | 11.50 | 退化，transfer |
| 10 | ffddd | `[12, 12, 12]` | 12.00 | `[12, 12, 12]` | 12.00 | 持平差 |
| 13 | ddddd | `[6, 6, 6]` | 6.00 | `[6, 6, 21]` | 11.00 | 退化 |
| 13 | fdddd | `[6, 6, 6]` | 6.00 | `[21, 17, 6]` | 14.67 | 明显退化 |
| 13 | ffddd | `[19, 21, 19]` | 19.67 | `[21, 17, 21]` | 19.67 | 持平差 |
| 16 | ddddd | `[0, 0, 0]` | 0.00 | `[0, 0, 0]` | 0.00 | 持平稳定 |
| 16 | fdddd | `[0, 0, 0]` | 0.00 | `[0, 0, 0]` | 0.00 | 持平稳定 |
| 16 | ffddd | `[0, 0, 0]` | 0.00 | `[0, 0, 0]` | 0.00 | 持平稳定 |

Aggregate:

| run | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high terminal >=10 | transfer count |
|---|---:|---:|---:|---:|---:|
| v1 3patterns r3 | 7.264 | 11.675 | 14.758 | 16/36 | 1/36 |
| v1.1 3patterns r3 | 6.236 | 10.799 | 13.882 | 10/36 | 1/36 |

这里有整体改善，主要来自 dataset 2 全部归零；但 dataset 10/13 的核心 contract 仍不稳定，所以不进入 clean-100。

## 真实因果链

### dataset 2: app permission closure 被修好

v1 常见失败：

- Stage 4 会修 APN / Wi-Fi / network 等下游，但漏 `break_app_storage_permission`。
- Stage 5 replay 缺 `grant_app_permission`，terminal 12。

v1.1 结果：

- `fdddd r5`: 5/5 terminal 0。
- `3patterns r3`: 9/9 terminal 0。
- selected/deferred 与 oracle 无 missing。

判断：

- permission closure prompt 起效。
- 这部分不需要 normalizer 自动补 selected。

### dataset 10: ordinary defer 修了一半，但偶发整案 transfer

目标形态：

- selected: local chain，包括 `airplane_mode_on`, `unseat_sim_card`, `data_mode_off`, `bad_network_preference`, `bad_wifi_calling`, `break_apn_mms_setting`, `break_app_sms_permission`
- deferred: `user_abroad_roaming_disabled_on`, `data_usage_exceeded`
- final: `repair_subset`
- expected terminal: 6 左右

成功样本：

- Stage 4 raw correctly selected local chain and deferred usage/account policy blockers.
- Stage 5 final `repair_subset`。
- terminal 6。

失败样本：

- Stage 4 first executed local repair tool calls, but final JSON then set every blocker `should_repair=false`。
- Stage 4 final JSON marked `repairability=transfer_required` with reason `active hard_transfer_required blockers remain unresolved`。
- Stage 4 `contract_self_check.transfer_required_has_hard_blocker_ok=true`，但这看起来是自信错误：该 case 的 intended handling 是 ordinary defer + repair_subset，不是 hard transfer。
- Stage 5 preserved Stage 4 transfer and final became `transfer`。
- terminal 22.5，terminal reasons include `invalid_local_transfer_floor_18`。

判断：

- v1.1 prompt 把 “don't select account/usage blockers” 强化后，部分样本从 false repair_all 拉回 repair_subset。
- 但另一些样本过度保守，把 ordinary defer 误读成 hard_transfer_required，导致整案 transfer。
- 这是 Stage 4 classification instability，不是 Stage 5 closure failure。

### dataset 13: Stage 4 经常正确，Stage 5 把 selected 缩坏

高罚样本的模式：

- Stage 4 raw selected all active local blockers，`repairability=repairable`。
- Stage 4 executed all canonical local tools，包括 airplane/SIM/data/roaming/network/APN/permission。
- Stage 5 verification sees `can_send_mms=false` or misreads post-repair Wi-Fi calling/APN/app permission evidence.
- Stage 5 raw outputs `repair_subset` with selected only `bad_wifi_calling` or only downstream blockers。
- Some raw Stage 5 selected list even includes invalid id `can_send_mms`。
- Normalizer removes invalid id but preserves the too-small selected subset, producing terminal 17/21。

判断：

- 这不是 Stage 4 upstream completeness 的典型失败；Stage 4 已经完成 local chain。
- 主要是 Stage 5 verification-to-terminal mapping 不稳：它把 verification signal / residual `can_send_mms=false` 当成 blocker selection basis，未继承 Stage 4 repairable plan。
- `contract_self_check` 在这些样本中仍声称 preservation ok，所以 self-check 不能直接作为 correctness validator。

### dataset 16: 继续稳定

- `fdddd r5`: 5/5 terminal 0。
- `3patterns r3`: 9/9 terminal 0。
- final 全部 `repair_all`。

判断：

- v1.1 没破坏 dataset 16。
- 但 dataset 16 已经在 v1 稳了，所以它不能证明 v1.1 足以进 clean-100。

## 是否符合预期

部分符合：

- `contract_self_check` required 后，Stage 4 基本开始输出 self-check。
- app permission rule 对 dataset 2 非常有效。
- ordinary defer prompt 能让 dataset 10 的部分样本稳定在 `repair_subset=6`。
- 没有引入 retry、PS 改动、terminal penalty 改动或 normalizer 自动修答案。

不符合：

- dataset 10 仍有 `transfer_required` 回潮，且 terminal 22.5。
- dataset 13 被 v1.1 的 Stage 5 preservation 目标反向打破：Stage 4 repairable，但 Stage 5 缩 selected。
- self-check 是 report-only 且会自信错误，不能当 validator。
- Stage 5 normalized export 在本轮 records 里仍没有完整保留 raw self-check；已补未来 fallback，但本轮报告中的 Stage 5 self-check 主要来自 raw JSON 分析。

## 当前建议

不要跑 smoke，也不要跑 clean-100，先做一个更窄的 prompt-only v1.2 或诊断-only验证。

建议 v1.2 仍保持 prompt-only：

- Stage 4: 明确 “ordinary_defer != hard_transfer_required”，尤其 `can_be_deferred=true` + assistant/account/usage/policy blocker 在 local execution clean track 中应默认 ordinary_defer，除非 spec 显示 non-deferable/hybrid/manual。
- Stage 4: self-check 里的 `transfer_required_has_hard_blocker_ok` 必须列出具体 hard blocker ids 和 rule source；仍 report-only，不用于决策。
- Stage 5: 如果 Stage 4 `repairability=repairable` 且 Stage 4 selected all blockers and executed tools, Stage 5 不应输出 `repair_subset`，除非 verification proves a specific selected blocker failed and names that blocker id。
- Stage 5: `selected_blocker_ids` 只能是 Stage 4 selected/deferred blocker ids，不能放 `can_send_mms`、verification tool names、or generic verification signal。
- Stage 5: `can_send_mms=false` alone is not a blocker id and not a reason to shrink selected; it must be mapped to a concrete active blocker id or preserve Stage 4 plan。
- Stage 5: required self-check should include `selected_ids_are_blockers_only_ok` and `stage4_repairable_plan_preserved_or_named_failure_ok`。

如果还坚持不动 normalizer，那么下一轮仍只跑同样 fixed trace，不跑 clean-100；只有 dataset 2/10/13/16 同时不回潮，再扩到 100。
