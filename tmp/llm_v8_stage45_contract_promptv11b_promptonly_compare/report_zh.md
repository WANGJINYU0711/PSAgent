# Stage 4/5 contract prompt v1.1b prompt-only 报告

## 实验配置名

本轮配置名：

`llm_v8_stage45_contract_promptv11b_promptonly_cconfig`

fixed trace run：

- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig/`

clean-100 stage-exact fixed trace 合并输出：

- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_clean100_stageexact_r1_seed1_cconfig/`
- 合并 records: `records.json`
- 异常摘要: `abnormal_records_summary.json`

clean-100 stage-exact 来源 run：

- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_clean100_fdddd_r1_seed1_cconfig/`，只取其中 `actual_pattern == required_pattern == fdddd` 的 73 条。
- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_clean100_fffff_exact_r1_seed1_cconfig/`，19 条。
- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_clean100_dfddd_exact_r1_seed1_cconfig/`，5 条。
- `tmp/llm_v8_stage45_contract_promptv11b_promptonly_fixedtrace_clean100_ddddd_exact_r1_seed1_cconfig/`，3 条。

四个 clean-100 tmux run 均 `exit_code=0`，当前无遗留 tmux session。注意：最早启动的 clean-100 `fdddd` 包含非 `fdddd` required 任务，最终 stage-exact 报告已严格过滤；没有把 fast-agent-on-deep/错配路径混入结论。

## 和上一版的区别

相对 v1.2：

- 删除 v1.2 的长 self-check 分类数组。
- 删除 `hard_transfer_reason_by_blocker` 复杂 schema。
- 回退到 v1.1 的主体 contract 和 permission closure。
- 保留 dataset 2 有效的 app permission closure：active app permission blocker 如果有 canonical local permission repair，且正在修 APN / Wi-Fi / MMS downstream，不应被 defer 或漏掉。
- Stage 4 只加轻规则：`transfer_required` 不能在没有 concrete hard input blocker id 时输出；如果无法列出 hard blocker，就选 local repair subset 或 ordinary defer。
- Stage 5 只加轻规则：`selected_blocker_ids` / `deferred_blocker_ids` 只能包含 input blocker ids，不能包含 `can_send_mms`、tool names、observed_state keys 或 generic symptoms。
- self-check 改短，只要求 report-only 对象：`has_concrete_transfer_blocker` 和 `ids_are_input_blockers_only`。

仍然没有做：

- 没有改 PS。
- 没有改 terminal penalty。
- 没有加 retry。
- 没有让 normalizer 自动替 LLM 修 selected/deferred/final_action。
- 没有删数据。
- 没有跑 smoke。

验证：

`python -m py_compile envs/executors/telecom_llm_bench_executor.py scripts/run_llm_path_sweep_diagnostic.py scripts/run_llm_fixed_profile_trace_diagnostic.py`

## 4-task fixed trace 结果

### fdddd r5

| dataset | terminal values | mean | final counts | 判断 |
|---:|---|---:|---|---|
| 2 | `[0,0,0,0,0]` | 0.0 | `repair_all:5` | 稳定，permission closure 保住 |
| 10 | `[6,6,6,6,6]` | 6.0 | `repair_subset:5` | 明显好转，ordinary defer 目标形态稳定 |
| 13 | `[21,21,6,6,6]` | 12.0 | `repair_subset:5` | 仍不稳，Stage 5/selected 收缩 |
| 16 | `[0,22.5,0,0,0]` | 4.5 | `repair_all:4, transfer:1` | 大多稳定，但有错误 transfer 回潮 |

Aggregate：

| run | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high >=10 | transfer |
|---|---:|---:|---:|---:|---:|
| v1.1 fdddd r5 | 5.400 | 9.575 | 12.700 | 3/20 | 2/20 |
| v1.2 fdddd r5 | 11.375 | 16.065 | 19.190 | 11/20 | 7/20 |
| v1.1b fdddd r5 | 5.625 | 9.403 | 12.528 | 3/20 | 1/20 |

结论：v1.1b 从 v1.2 明显恢复，dataset 10 比 v1.1 更好，但整体不严格优于 v1.1，因为 dataset 13/16 仍有高罚。

### 3patterns r3

| dataset | pattern | terminal values | mean | 判断 |
|---:|---|---|---:|---|
| 2 | ddddd/fdddd/ffddd | 全部 0 | 0.0 | 稳定修好 |
| 10 | fdddd | `[6,6,6]` | 6.0 | 目标形态稳定 |
| 10 | ddddd | `[10,6,6]` | 7.33 | 仍有一次误 `repair_all`/ordinary defer 不稳 |
| 10 | ffddd | `[12,12,12]` | 12.0 | fast-heavy 下仍差 |
| 13 | ddddd | `[21,6,21]` | 16.0 | 不稳 |
| 13 | fdddd | `[21,6,6]` | 11.0 | 不稳 |
| 13 | ffddd | `[21,21,21]` | 21.0 | fast-heavy 仍差 |
| 16 | ddddd/fdddd/ffddd | 每组 `[0,0,22.5]` | 7.5 | 有错误 transfer 回潮 |

Aggregate：

| run | terminal mean | raw_total_cost mean | raw_total_cost_with_token_penalty mean | high >=10 | transfer |
|---|---:|---:|---:|---:|---:|
| v1.1 3patterns r3 | 6.236 | 10.799 | 13.882 | 10/36 | 1/36 |
| v1.2 3patterns r3 | 11.833 | 16.938 | 20.021 | 23/36 | 11/36 |
| v1.1b 3patterns r3 | 7.986 | 12.111 | 15.195 | 13/36 | 3/36 |

结论：v1.1b 明显好于 v1.2，但 3patterns 上弱于 v1.1；主要退化来自 dataset 13 和 dataset 16 的偶发 transfer。

## clean-100 stage-exact 结果

clean-100 合并结果严格满足：`actual_pattern == required_pattern`。

总体：

| metric | value |
|---|---:|
| n | 100 |
| exact stage-match records | 100 |
| terminal mean | 3.225 |
| raw_total_cost mean | 6.590 |
| raw_total_cost_with_token_penalty mean | 15.752 |
| terminal == 0 | 77/100 |
| terminal <= 6 | 79/100 |
| terminal < 10 | 79/100 |
| terminal >= 10 | 21/100 |
| final_action=transfer | 5/100 |
| exact_match | 79/100 |

final action：

| final_action | count |
|---|---:|
| repair_all | 77 |
| repair_subset | 18 |
| transfer | 5 |

按 expected_terminal_action：

| expected | n | terminal mean | terminal 0 | terminal <10 | terminal >=10 | transfer | final counts |
|---|---:|---:|---:|---:|---:|---:|---|
| repair_all | 94 | 2.644 | 77 | 77 | 17 | 5 | `{'repair_all': 77, 'repair_subset': 12, 'transfer': 5}` |
| repair_subset | 6 | 12.333 | 0 | 2 | 4 | 0 | `{'repair_subset': 6}` |

按 required pattern：

| required_pattern | n | terminal mean | terminal 0 | terminal <10 | terminal >=10 | transfer | final counts |
|---|---:|---:|---:|---:|---:|---:|---|
| ddddd | 3 | 13.667 | 0 | 0 | 3 | 0 | `{'repair_subset': 3}` |
| dfddd | 5 | 0.000 | 5 | 5 | 0 | 0 | `{'repair_all': 5}` |
| fdddd | 73 | 3.034 | 58 | 60 | 13 | 5 | `{'repair_all': 58, 'repair_subset': 10, 'transfer': 5}` |
| fffff | 19 | 3.158 | 14 | 14 | 5 | 0 | `{'repair_all': 14, 'repair_subset': 5}` |

解读：

- 如果把 terminal `<10` 视为“正常/低 cost”，clean-100 stage-exact 当前是 79%。
- 如果只看完全干净 terminal `0` / exact match，是 77%-79% 左右。
- 这说明 v1.1b 已经恢复到“多数 clean local task 可低 cost”的水平，但还没有达到可以放心进入 formal smoke 的执行稳定度。
- `fffff` 的 `raw_total_cost_with_token_penalty` 很高，主要是 fast token over-budget soft penalty；这不影响 terminal 判断，但说明 fast exact path 的 token budget 问题依旧明显。

## clean-100 异常任务

异常定义：`raw_terminal_penalty >= 10` 或 `final_action=transfer`。

| dataset | required | expected | terminal | final | selected_missing_vs_oracle | replay_missing_tools | 主因判断 |
|---:|---|---|---:|---|---|---|---|
| 5 | ddddd | repair_subset | 12.0 | repair_subset | `bad_wifi_calling` | `toggle_wifi_calling` | Stage4 漏 local MMS downstream |
| 8 | ddddd | repair_subset | 17.0 | repair_subset | `airplane_mode_on,bad_network_preference,break_app_sms_permission,data_mode_off,user_abroad_roaming_enabled_off` | `` | Stage5 缩 selected / Stage4 defer 不完整 |
| 11 | ddddd | repair_subset | 12.0 | repair_subset | `airplane_mode_on,bad_network_preference,data_mode_off` | `` | Stage5 缩 selected / Stage4 defer 不完整 |
| 13 | fdddd | repair_subset | 21.0 | repair_subset | `airplane_mode_on,bad_network_preference,break_apn_mms_setting,break_app_both_permissions,data_mode_off,unseat_sim_card,user_abroad_roaming_enabled_off` | `` | Stage5 缩 selected / Stage4 defer 不完整 |
| 18 | fffff | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_app_storage_permission` | `grant_app_permission,toggle_wifi_calling` | Stage4 fast 只修一个上游，defer MMS/app |
| 19 | fffff | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_app_sms_permission` | `grant_app_permission,toggle_wifi_calling` | Stage4 fast 只修一个上游，defer MMS/app |
| 20 | fffff | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_app_sms_permission` | `grant_app_permission,toggle_wifi_calling` | Stage4 fast 只修一个上游，defer MMS/app |
| 21 | fffff | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_app_storage_permission` | `grant_app_permission,toggle_wifi_calling` | Stage4 fast 只修一个上游，defer MMS/app |
| 29 | fffff | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_app_both_permissions` | `grant_app_permission,toggle_wifi_calling` | Stage4 fast 只修一个上游，defer MMS/app |
| 32 | fdddd | repair_all | 13.0 | repair_subset | `bad_wifi_calling,break_apn_mms_setting,break_app_storage_permission` | `grant_app_permission,reboot_device,reset_apn_settings,toggle_wifi_calling` | Stage4 sim-only，defer downstream chain |
| 41 | fdddd | repair_all | 15.0 | repair_subset | `bad_network_preference,bad_wifi_calling,break_apn_mms_setting,break_app_sms_permission` | `grant_app_permission,reboot_device,reset_apn_settings,set_network_mode_preference,toggle_wifi_calling` | Stage4 sim-only，defer downstream chain |
| 50 | fdddd | repair_all | 18.5 | transfer | `bad_network_preference,bad_wifi_calling,break_apn_mms_setting,break_app_sms_permission,data_mode_off,unseat_sim_card` | `grant_app_permission,reboot_device,reseat_sim_card,reset_apn_settings,set_network_mode_preference,toggle_data,toggle_wifi_calling` | Stage4 无 hard id 错误 transfer |
| 51 | fdddd | repair_all | 18.5 | transfer | `bad_network_preference,bad_wifi_calling,break_apn_mms_setting,break_app_sms_permission,data_mode_off,user_abroad_roaming_enabled_off` | `grant_app_permission,reboot_device,reset_apn_settings,set_network_mode_preference,toggle_data,toggle_roaming,toggle_wifi_calling` | Stage4 无 hard id 错误 transfer |
| 57 | fdddd | repair_all | 18.0 | transfer | `bad_wifi_calling,break_apn_mms_setting,break_app_sms_permission,data_mode_off,unseat_sim_card` | `grant_app_permission,reboot_device,reseat_sim_card,reset_apn_settings,toggle_data,toggle_wifi_calling` | Stage4 无 hard id 错误 transfer |
| 64 | fdddd | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_apn_mms_setting` | `reboot_device,reset_apn_settings,toggle_wifi_calling` | Stage4 漏 local MMS downstream |
| 77 | fdddd | repair_all | 18.0 | transfer | `bad_wifi_calling,break_apn_mms_setting,data_mode_off,unseat_sim_card,user_abroad_roaming_enabled_off` | `` | Stage5 将 good Stage4 改成 transfer |
| 78 | fdddd | repair_all | 12.0 | repair_subset | `bad_wifi_calling,break_apn_mms_setting` | `reboot_device,reset_apn_settings,toggle_wifi_calling` | Stage4 漏 local MMS downstream |
| 79 | fdddd | repair_all | 18.5 | transfer | `airplane_mode_on,bad_network_preference,bad_wifi_calling,break_apn_mms_setting,unseat_sim_card,user_abroad_roaming_enabled_off` | `reboot_device,reseat_sim_card,reset_apn_settings,set_network_mode_preference,toggle_airplane_mode,toggle_roaming,toggle_wifi_calling` | Stage4 无 hard id 错误 transfer |
| 86 | fdddd | repair_all | 15.0 | repair_subset | `bad_network_preference,bad_wifi_calling,break_apn_mms_setting,break_app_both_permissions` | `grant_app_permission,reboot_device,reset_apn_settings,set_network_mode_preference,toggle_wifi_calling` | Stage4 sim-only，defer downstream chain |
| 90 | fdddd | repair_all | 13.0 | repair_subset | `bad_wifi_calling,break_apn_mms_setting,break_app_both_permissions` | `grant_app_permission,reboot_device,reset_apn_settings,toggle_wifi_calling` | Stage4 sim-only，defer downstream chain |
| 96 | fdddd | repair_all | 17.0 | repair_subset | `bad_wifi_calling,break_apn_mms_setting,break_app_both_permissions,data_mode_off,user_abroad_roaming_enabled_off` | `grant_app_permission,reboot_device,reset_apn_settings,toggle_data,toggle_roaming,toggle_wifi_calling` | Stage4 sim-only，defer downstream chain |

## 真实因果链

### 1. v1.1b 确实修复了 v1.2 的复杂 schema 退化

v1.2 的主要问题是长 self-check 和 hard-transfer schema 诱发保守/空心输出。v1.1b 删除这部分后：

- 4-task fdddd terminal mean 从 11.375 回到 5.625。
- 4-task 3patterns terminal mean 从 11.833 回到 7.986。
- dataset 2 保持全 0。
- dataset 10 fdddd 从 v1.1/v1.2 的不稳变成 5/5 terminal 6。

这说明“短规则 + 保留 permission closure”是比 v1.2 复杂审计更健康的方向。

### 2. dataset 2 permission closure 保留成功

dataset 2 在 fdddd r5 和 3patterns r3 全部 terminal 0。app permission blocker 没再被系统性漏掉，说明 v1.1 的 permission closure 是有效 prompt 约束，应该继续保留。

### 3. dataset 10 的目标形态在 fdddd 下稳定

dataset 10 的理想输出是 local MMS chain selected，`data_usage_exceeded` / account roaming policy deferred，final `repair_subset`。v1.1b 的 fdddd r5 是 `[6,6,6,6,6]`，3patterns 里的 fdddd 也是 `[6,6,6]`。

但 ddddd/ffddd 仍不完全稳，说明 ordinary_defer contract 已经改善，但不是跨所有 path profile 全稳。

### 4. clean-100 最大残留问题是 Stage 4 过窄 selected，不是 schema 长度

21 个异常里，最多的是 Stage 4 只修一个上游或一个 SIM/service blocker，然后把 downstream MMS chain defer：

- `fffff` 的 5 个失败：Stage 4 只 selected `bad_network_preference` 或 `data_mode_off`，defer `bad_wifi_calling` 和 app permission，导致本应 `repair_all` 的任务变成 terminal 12。
- `fdddd` 的多个失败：Stage 4 只 selected `unseat_sim_card` 或 service/data 上游，defer `bad_wifi_calling`、APN、app permission，导致 terminal 12/13/15/17。

这说明 v1.1b 的 permission closure 还不够泛化到“如果本轮已经修 local chain 上游，就不要把可本地修的 MMS downstream 链整体 defer”。它不是 normalizer 应该自动修的问题，而是 Stage 4 selected/deferred contract 仍弱。

### 5. 仍有没有 concrete hard blocker 的错误 transfer

clean-100 stage-exact 里有 5 个 transfer，其中 4 个是 Stage 4 直接 `transfer_required`：dataset 50/51/57/79。

共同点：

- expected 都是 `repair_all`。
- Stage 4 selected 为空，所有 blocker 进 deferred。
- `stage4_repairability=transfer_required`，`transfer_reason=no_safe_local_repair_subset_v2`。
- `contract_self_check.has_concrete_transfer_blocker=false` 的样本仍然输出了 transfer。

这证明 v1.1b 的轻规则能减少 v1.2 的 transfer 泛滥，但 prompt-only 不能完全阻止自相矛盾输出。由于我们没有 validator/normalizer 自动纠正，它仍会进入 terminal 高罚。

### 6. Stage 5 仍会偶发破坏好的 Stage 4 plan

dataset 77 是最清楚的例子：

- Stage 4 selected 了完整 local repair chain，`repairability=repairable`。
- Stage 5 最终输出 transfer，selected 变空。
- terminal 18。

这说明“Stage 5 不要过度继承 Stage 4”的担忧是对的，但当前相反方向也存在：Stage 5 有时会无证据地推翻一个正确 Stage 4 plan。下一版 prompt 不能只强调继承，也要要求任何改动都必须绑定 concrete input blocker id 和验证证据。

## 是否符合预期

符合的部分：

- v1.1b 明显摆脱 v1.2 的复杂 self-check 退化。
- dataset 2 继续被 permission closure 修好。
- dataset 10 的 fdddd 目标形态已经稳定。
- clean-100 stage-exact 多数任务 terminal 低：79/100 `<10`，77/100 为 0。
- 没有新增 retry、normalizer 自动纠正、PS 改动或 penalty 改动。

不符合的部分：

- v1.1b 不严格优于 v1.1，尤其 3patterns 上更差。
- dataset 13 仍然不稳，fast-heavy 更差。
- dataset 16 出现少量 transfer 回潮。
- clean-100 仍有 21% 高罚/transfer 异常，主要集中在 Stage 4 过窄 selected 和无 hard blocker transfer。
- `contract_self_check` 虽然短了，但仍只是 report-only；当它与 repairability 冲突时，当前系统不会拦截。

## 当前建议

不要进 smoke，也不要删数据。

下一步建议继续 prompt-only，但不要再加长 schema：

1. 保留 v1.1b 的短 self-check 和 permission closure。
2. Stage 4 增加一个通用 chain-closure 规则：如果 active blocker 有 canonical local repair 且不是 ordinary_defer/hard_transfer_required，不能只修 service/SIM/data 上游后 defer APN/Wi-Fi/app-permission downstream；selected 必须覆盖本地可修的 connected local chain，除非能给出 concrete ordinary/hard blocker id。
3. Stage 4 对 `transfer_required` 的轻规则还需要更直接：`has_concrete_transfer_blocker=false` 与 `repairability=transfer_required` 是 invalid output；prompt-only 下仍 report-only，但要写成 JSON consistency rule。
4. Stage 5 增加通用 edit-evidence rule：如果要改变 Stage 4 selected/deferred 或 final_action，必须列出 changed input blocker ids 和 verification evidence；不能把完整 repairable plan 改成 transfer 或空 selected。
5. 仍只跑 fixed trace，不跑 smoke；优先复测 clean-100 中这 21 个异常任务和 focus 2/10/13/16。
