# Stage 4/5 contract prompt v1.1c prompt-only 复测报告

## 配置名

`llm_v8_stage45_contract_promptv11c_promptonly_cconfig`

本轮新增开关：

`PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1C=1`

本轮仍然是 prompt-only：

- 不改 PS。
- 不改 terminal penalty。
- 不加 retry。
- 不让 normalizer 自动替 LLM 修 selected/deferred/final_action。
- 不删数据。
- 不跑 smoke。

## 本轮和 v1.1b 的区别

v1.1b 已有：

- 短 `contract_self_check`：`has_concrete_transfer_blocker` 和 `ids_are_input_blockers_only`。
- permission closure。
- `transfer_required` 需要 concrete hard input blocker 的轻规则。
- Stage 5 selected/deferred 只能包含 input blocker ids。

v1.1c 新增三条 prompt-only 约束：

1. Stage 4 connected local-chain closure：
   如果 service/SIM/data 上游 blocker 被 selected，且同一 case 中 downstream MMS/APN/Wi-Fi/app-permission input blockers 也 active、可用 canonical local Stage 4 repair、且不是 ordinary_defer/hard_transfer_required，不要只修上游后 defer downstream 本地链。

2. Stage 4 transfer consistency：
   `repairability=transfer_required` 与 `contract_self_check.has_concrete_transfer_blocker=false` 是 inconsistent；`transfer_reason` 必须能点名 concrete hard input blocker id。

3. Stage 5 edit-evidence rule：
   如果 Stage 5 改 Stage 4 selected/deferred/final_action，必须绑定 concrete input blocker id 和 verification evidence；否则保留 Stage 4 blocker plan。

这个 chain-closure 是刻意写窄的：不适用于 account/usage/policy/subscription/quota/billing blocker，不适用于 `can_be_deferred` ordinary defer，不适用于 hybrid/external/nonlocal blocker，也不适用于没有 Stage 4 canonical local repair tool 的 blocker。

## 运行产物

focus fixed trace：

- `tmp/llm_v8_stage45_contract_promptv11c_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11c_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig/`

clean-100 abnormal exact fixed trace：

- `tmp/llm_v8_stage45_contract_promptv11c_promptonly_fixedtrace_clean100_abnormal_fdddd_exact_r3_seed1_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11c_promptonly_fixedtrace_clean100_abnormal_fffff_exact_r3_seed1_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11c_promptonly_fixedtrace_clean100_abnormal_ddddd_exact_r3_seed1_cconfig/`

所有 tmux run 均 `exit_code=0`，当前没有遗留 tmux session。

验证：

`python -m py_compile envs/executors/telecom_llm_bench_executor.py scripts/run_llm_path_sweep_diagnostic.py scripts/run_llm_fixed_profile_trace_diagnostic.py`

## 是否会“修一类、伤另一类”

结论：会。

我原先的判断是：如果 chain-closure 被严格限定在 active input blocker + canonical local repair + 非 ordinary_defer/hard_transfer 的 connected local chain，它理论上不应该伤 ordinary defer 或 hard transfer。

实际 fixed trace 显示：这个规则确实大幅修好了 upstream-only 类异常，但也让 LLM 更容易过度关注“local chain completeness”，从而在 dataset 10 / partial-repair contract 上回潮：

- 有时把 `data_usage_exceeded` 和 `user_abroad_roaming_disabled_on` 错放进 selected，导致 `repair_all` 高罚 10。
- 有时 selected/deferred 反转，只 selected account/usage blockers，defer 整条 local MMS chain，导致 terminal 23。
- dataset 16 仍出现无 concrete hard blocker 的 `transfer_required`。

所以 v1.1c 不适合作为替代 v1.1b 的新基线。它更像一个诊断探针：证明 chain-closure 能修 upstream-only failure，但当前 prompt 形态会干扰 ordinary_defer 边界。

## focus fdddd r5

| dataset | v1.1b terminal | v1.1b mean | v1.1c terminal | v1.1c mean | 判断 |
|---:|---|---:|---|---:|---|
| 2 | `[0,0,0,0,0]` | 0.0 | `[0,0,0,0,0]` | 0.0 | 持平稳定 |
| 10 | `[6,6,6,6,6]` | 6.0 | `[6,23,6,10,6]` | 10.2 | 退化，ordinary_defer 被破坏 |
| 13 | `[21,21,6,6,6]` | 12.0 | `[21,21,21,6,21]` | 18.0 | 退化，Stage 5 缩 selected 更频繁 |
| 16 | `[0,22.5,0,0,0]` | 4.5 | `[0,22.5,0,22.5,0]` | 9.0 | 退化，transfer 回潮 |

Aggregate：

| run | n | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high >=10 | transfer |
|---|---:|---:|---:|---:|---:|---:|
| v1.1b fdddd r5 | 20 | 5.625 | 9.403 | 12.528 | 3 | 1 |
| v1.1c fdddd r5 | 20 | 9.300 | 13.524 | 16.649 | 8 | 2 |

## focus 3patterns r3

| dataset | pattern | v1.1b terminal | v1.1b mean | v1.1c terminal | v1.1c mean | 判断 |
|---:|---|---|---:|---|---:|---|
| 2 | ddddd | `[0,0,0]` | 0.0 | `[0,0,0]` | 0.0 | 持平 |
| 2 | fdddd | `[0,0,0]` | 0.0 | `[0,0,0]` | 0.0 | 持平 |
| 2 | ffddd | `[0,0,0]` | 0.0 | `[0,0,0]` | 0.0 | 持平 |
| 10 | ddddd | `[10,6,6]` | 7.33 | `[6,22.5,22.5]` | 17.0 | 明显退化，transfer/ordinary_defer 回潮 |
| 10 | fdddd | `[6,6,6]` | 6.0 | `[10,6,10]` | 8.67 | 退化，误 repair_all |
| 10 | ffddd | `[12,12,12]` | 12.0 | `[12,12,12]` | 12.0 | 持平差 |
| 13 | ddddd | `[21,6,21]` | 16.0 | `[21,21,6]` | 16.0 | 持平差 |
| 13 | fdddd | `[21,6,6]` | 11.0 | `[6,21,21]` | 16.0 | 退化 |
| 13 | ffddd | `[21,21,21]` | 21.0 | `[21,21,21]` | 21.0 | 持平差 |
| 16 | ddddd | `[0,0,22.5]` | 7.5 | `[22.5,0,0]` | 7.5 | 持平但仍有 transfer |
| 16 | fdddd | `[0,0,22.5]` | 7.5 | `[0,22.5,0]` | 7.5 | 持平但仍有 transfer |
| 16 | ffddd | `[0,0,22.5]` | 7.5 | `[0,0,0]` | 0.0 | 改善 |

Aggregate：

| run | n | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high >=10 | transfer |
|---|---:|---:|---:|---:|---:|---:|
| v1.1b 3patterns r3 | 36 | 7.986 | 12.111 | 15.195 | 13 | 3 |
| v1.1c 3patterns r3 | 36 | 8.806 | 13.207 | 16.291 | 16 | 4 |

## clean-100 abnormal exact 复测

本轮只复测 v1.1b clean-100 stage-exact 中的 21 个异常任务，并按 required pattern 分组，避免 stage mismatch。

### combined abnormal r3

| metric | value |
|---|---:|
| n | 63 |
| terminal mean | 2.333 |
| terminal == 0 | 51/63 |
| terminal <10 | 55/63 |
| terminal >=10 | 8/63 |
| transfer | 0/63 |

### fffff abnormal r3

| dataset | v1.1b one-shot | v1.1c r3 | 判断 |
|---:|---|---|---|
| 18 | 12 | `[0,0,0]` | 修好 |
| 19 | 12 | `[0,0,0]` | 修好 |
| 20 | 12 | `[0,0,0]` | 修好 |
| 21 | 12 | `[0,0,0]` | 修好 |
| 29 | 12 | `[0,0,0]` | 修好 |

这组是 v1.1c 最强改善。v1.1b 里 fast exact path 常只修 `data_mode_off` 或 `bad_network_preference`，把 `bad_wifi_calling` 和 app permission defer；v1.1c 把 connected local chain 拉进 selected，15/15 terminal 0。

### fdddd abnormal r3

| dataset | v1.1b one-shot | v1.1c r3 | 判断 |
|---:|---|---|---|
| 13 | 21 | `[6,21,6]` | 部分改善但仍不稳 |
| 32 | 13 | `[0,0,0]` | 修好 |
| 41 | 15 | `[0,0,0]` | 修好 |
| 50 | 18.5 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 51 | 18.5 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 57 | 18 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 64 | 12 | `[0,0,0]` | 修好 |
| 77 | 18 transfer | `[0,0,0]` | 修好，Stage 5 不再改成 transfer |
| 78 | 12 | `[0,0,0]` | 修好 |
| 79 | 18.5 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 86 | 15 | `[0,0,0]` | 修好 |
| 90 | 13 | `[0,0,0]` | 修好 |
| 96 | 17 | `[0,0,0]` | 修好 |

fdddd abnormal 是 v1.1c 的核心成功：39 条里只有 dataset 13 的一次 21，其余全部 terminal 0。上一版的 upstream-only selected、sim-only selected、无 hard blocker transfer 基本被清掉。

### ddddd abnormal r3

| dataset | v1.1b one-shot | v1.1c r3 | 判断 |
|---:|---|---|---|
| 5 | 12 | `[10,10,6]` | 小幅改善但仍误 repair_all |
| 8 | 17 | `[19,19,10]` | 退化/仍差 |
| 11 | 12 | `[17,6,17]` | 不稳/仍差 |

ddddd partial-repair 类没有被 chain-closure 解决。失败链主要变成两种：

- Stage 4 把 `data_usage_exceeded` 选进 selected，错误 `repair_all`，terminal 10。
- Stage 4 selected 完整 local chain 后，Stage 5 又把 selected 缩到单个 app permission blocker，terminal 17/19。

## 真实因果链

### 1. chain-closure 修好了 upstream-only local failures

v1.1b 的 clean-100 异常里有很多这种形态：

- selected 只有 `unseat_sim_card`、`data_mode_off` 或 `bad_network_preference`。
- `bad_wifi_calling`、`break_apn_mms_setting`、app permission 被 deferred。
- expected 是 `repair_all`，terminal 12/13/15/17。

v1.1c 对这类非常有效：

- `fffff` abnormal 15/15 terminal 0。
- `fdddd` abnormal 里 dataset 32/41/64/78/86/90/96 全部 0。
- dataset 50/51/57/79 的错误 transfer 也全部消失。

这说明 connected local-chain closure 的目标方向是对的。

### 2. 但它确实伤到了 ordinary_defer 边界

dataset 10 是最清楚的反例。

正确形态：

- selected local chain：`airplane_mode_on`, `unseat_sim_card`, `data_mode_off`, `bad_network_preference`, `bad_wifi_calling`, `break_apn_mms_setting`, app permission。
- deferred ordinary blockers：`data_usage_exceeded`, `user_abroad_roaming_disabled_on`。
- final `repair_subset`，terminal 6。

v1.1c 的失败出现两类：

- repair_all 误选 ordinary defers：Stage 4 把 `data_usage_exceeded` 和 `user_abroad_roaming_disabled_on` 也放进 selected，final `repair_all`，terminal 10。
- selected/deferred 反转：Stage 4 selected 只有 `data_usage_exceeded` 和 `user_abroad_roaming_disabled_on`，deferred 整条 local MMS chain，terminal 23。

这说明虽然 prompt 文字排除了 account/usage/policy ordinary defers，但 LLM 仍被 chain-completion 的注意力牵引，partial-repair contract 变差。

### 3. dataset 13 主要仍是 Stage 5 缩 selected

v1.1c 中 dataset 13 常见 raw chain：

- Stage 4 selected 完整 local chain。
- Stage 4 `repairability=repairable`。
- Stage 5 final `repair_subset`，selected 缩成只剩 `bad_wifi_calling`。
- terminal 21。

这说明 Stage 5 edit-evidence prompt 没能稳定阻止“无证据缩 selected”。这不是 Stage 4 chain closure 能解决的问题。

### 4. dataset 16 仍有无 hard blocker transfer

v1.1c 的 dataset 16 仍出现：

- Stage 4 selected 为空。
- deferred 全部 input blockers。
- `repairability=transfer_required`。
- `transfer_reason=no_safe_local_repair_subset_v2`。
- `contract_self_check.has_concrete_transfer_blocker=false`。
- final transfer，terminal 22.5。

这说明 prompt-only consistency rule 仍挡不住自相矛盾输出；如果未来要彻底消除这类，需要 validator-only fail-fast / re-ask，或者 normalizer/guard，但本轮按约束没有做。

## 是否符合预期

符合：

- v1.1c 精准修复了 v1.1b clean-100 中大量 upstream-only local-chain 异常。
- v1.1c 大幅降低 clean-100 abnormal exact 复测的 terminal：63 条 repeat 中 51 条为 0，55 条 <10，transfer 0。
- dataset 2 继续稳定。
- 没有新增 retry、normalizer 自动纠正、PS 改动、terminal penalty 改动。

不符合：

- v1.1c 伤到了 dataset 10 的 ordinary_defer / repair_subset contract。
- focus fdddd r5 从 v1.1b mean 5.625 退化到 9.300。
- focus 3patterns r3 从 v1.1b mean 7.986 退化到 8.806。
- dataset 13 仍大量 Stage 5 shrink selected。
- dataset 16 仍有无 concrete hard blocker transfer。

## 结论

v1.1c 不是合格的新基线。

它证明了一个局部事实：connected local-chain closure 是修 clean-100 upstream-only failures 的强信号。但当前 prompt 写法会让模型在 partial-repair case 里过度关注“修完整 local chain/complete case”，从而破坏 ordinary_defer，尤其 dataset 10。

建议不要进 smoke，也不要用 v1.1c 替代 v1.1b。

下一步更稳的方向不是继续加长 prompt，而是拆成更窄的 v1.1d：

1. 回到 v1.1b 为基线。
2. 只保留 chain-closure 的“反 upstream-only”部分，不再强化 “repairable/complete chain” 语言。
3. 把 rule 写成 precedence order：ordinary_defer classification 先于 connected local-chain closure；chain-closure 只能补 active local downstream，不能改变 account/usage/policy selected/deferred。
4. 对 dataset 10 类型加一个对照 negative example：即使 local chain closure 生效，`data_usage_exceeded` 和 account roaming policy 仍必须 deferred。
5. Stage 5 shrink selected 另行处理，不和 Stage 4 chain closure 混在同一版里。

