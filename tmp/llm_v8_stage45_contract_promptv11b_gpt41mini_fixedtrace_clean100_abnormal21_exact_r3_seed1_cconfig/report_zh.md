# Stage 4/5 contract prompt v1.1b + gpt-4.1-mini abnormal21 fixed trace 报告

## 配置名

`llm_v8_stage45_contract_promptv11b_gpt41mini_abnormal21_exact_r3_cconfig`

本轮目的：

- 不改 prompt 逻辑，仍使用 v1.1b。
- 只把模型从 `gpt-4o-mini` 换成 `gpt-4.1-mini`。
- 复测 v1.1b clean-100 stage-exact 中出错的 21 条任务：
  `[5, 8, 11, 13, 18, 19, 20, 21, 29, 32, 41, 50, 51, 57, 64, 77, 78, 79, 86, 90, 96]`
- 按 required pattern 分组跑 stage-exact fixed trace，每条 `r3`。

本轮仍然没有：

- 没有改 PS。
- 没有改 terminal penalty。
- 没有加 retry。
- 没有让 normalizer 自动替 LLM 修 selected/deferred/final_action。
- 没有删数据。
- 没有跑 smoke。
- 没有使用 v1.1c 的 chain-closure prompt；环境变量只开了 `PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B=1`。

## 运行产物

分组 run：

- `tmp/llm_v8_stage45_contract_promptv11b_gpt41mini_fixedtrace_clean100_abnormal21_fdddd_exact_r3_seed1_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11b_gpt41mini_fixedtrace_clean100_abnormal21_fffff_exact_r3_seed1_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11b_gpt41mini_fixedtrace_clean100_abnormal21_ddddd_exact_r3_seed1_cconfig/`

合并输出：

- `tmp/llm_v8_stage45_contract_promptv11b_gpt41mini_fixedtrace_clean100_abnormal21_exact_r3_seed1_cconfig/records.json`
- `tmp/llm_v8_stage45_contract_promptv11b_gpt41mini_fixedtrace_clean100_abnormal21_exact_r3_seed1_cconfig/summary.json`

三个本轮 tmux run 均 `exit_code=0`。

备注：当前环境里还有一个其它 session `psagent_stage_mismatch_cost_probe_v1_r3` 在跑；它不是本报告使用的 run，我没有干预它。本报告只使用上面三个 abnormal21 exact r3 输出。

## 和上一版/基线的区别

和 v1.1b + `gpt-4o-mini` clean-100 one-shot 相比：

- prompt 相同：都是 v1.1b。
- dataset 相同：同一批 21 个异常任务。
- stage path 口径相同：都要求 `actual_pattern == required_pattern`。
- 主要差异：模型从 `gpt-4o-mini` 换成 `gpt-4.1-mini`；本轮每条跑 r3，上一版 clean-100 是每条 one-shot。

和刚才 v1.1c 相比：

- 本轮删除 v1.1c 的 connected local-chain closure / Stage 5 edit-evidence 新 prompt。
- 本轮回到 v1.1b prompt，但换更强模型。
- 因此这轮是“模型能力/遵循度”诊断，不是 prompt 规则诊断。

## 总体结果

### v1.1b + gpt-4o-mini clean-100 abnormal one-shot 基线

这 21 条在 v1.1b + `gpt-4o-mini` 的 clean-100 stage-exact one-shot 中全部异常：

| metric | value |
|---|---:|
| n | 21 |
| terminal mean | 14.786 |
| terminal == 0 | 0/21 |
| terminal < 10 | 0/21 |
| terminal >= 10 | 21/21 |
| transfer | 5/21 |
| final counts | `repair_subset:16, transfer:5` |

### v1.1b + gpt-4.1-mini abnormal21 exact r3

| metric | value |
|---|---:|
| n | 63 |
| exact stage-match | 63/63 |
| terminal mean | 2.254 |
| raw_total_cost mean | 7.056 |
| raw_total_cost_with_token_penalty mean | 18.524 |
| terminal == 0 | 49/63 |
| terminal <= 6 | 54/63 |
| terminal < 10 | 54/63 |
| terminal >= 10 | 9/63 |
| transfer | 0/63 |
| final counts | `repair_all:44, repair_subset:19` |

结论非常清楚：同一版 v1.1b prompt 下，`gpt-4.1-mini` 把这批异常从 21/21 高罚，降到 9/63 高罚，而且完全清掉了 transfer。模型能力/指令遵循度确实解释了一大块失败。

## 按 pattern 汇总

| pattern | n | terminal mean | terminal 0 | terminal <10 | terminal >=10 | transfer | final counts |
|---|---:|---:|---:|---:|---:|---:|---|
| fdddd | 39 | 1.000 | 35 | 37 | 2 | 0 | `repair_all:34, repair_subset:5` |
| fffff | 15 | 5.000 | 9 | 9 | 6 | 0 | `repair_all:9, repair_subset:6` |
| ddddd | 9 | 3.111 | 5 | 8 | 1 | 0 | `repair_subset:8, repair_all:1` |

## 按 dataset 明细

| dataset | pattern | v1.1b+4o-mini one-shot | v1.1b+4.1-mini r3 | 判断 |
|---:|---|---:|---|---|
| 5 | ddddd | 12 | `[6,6,6]` | 明显改善，稳定低罚 |
| 8 | ddddd | 17 | `[0,10,0]` | 明显改善，但有一次误 repair_all |
| 11 | ddddd | 12 | `[0,0,0]` | 修好 |
| 13 | fdddd | 21 | `[6,6,0]` | 明显改善 |
| 18 | fffff | 12 | `[12.5,12.5,12.5]` | 没修好，稳定漏 downstream/app permission |
| 19 | fffff | 12 | `[0,0,0]` | 修好 |
| 20 | fffff | 12 | `[0,12.5,12.5]` | 部分改善但不稳 |
| 21 | fffff | 12 | `[0,0,0]` | 修好 |
| 29 | fffff | 12 | `[0,12.5,0]` | 部分改善但不稳 |
| 32 | fdddd | 13 | `[0,0,0]` | 修好 |
| 41 | fdddd | 15 | `[0,0,0]` | 修好 |
| 50 | fdddd | 18.5 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 51 | fdddd | 18.5 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 57 | fdddd | 18 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 64 | fdddd | 12 | `[0,0,0]` | 修好 |
| 77 | fdddd | 18 transfer | `[0,0,0]` | 修好，transfer 消失 |
| 78 | fdddd | 12 | `[0,0,0]` | 修好 |
| 79 | fdddd | 18.5 transfer | `[15,0,12]` | transfer 消失，但 selected 仍不稳 |
| 86 | fdddd | 15 | `[0,0,0]` | 修好 |
| 90 | fdddd | 13 | `[0,0,0]` | 修好 |
| 96 | fdddd | 17 | `[0,0,0]` | 修好 |

## 真实因果链

### 1. 4.1-mini 大幅修掉了 fdddd 的 Stage 4 过窄 selected 和错误 transfer

v1.1b + 4o-mini 中，fdddd 异常常见两类：

- Stage 4 只 selected 一个上游 blocker，例如 `unseat_sim_card`，defer downstream MMS chain。
- Stage 4 直接 `transfer_required`，但没有 concrete hard blocker。

4.1-mini 下，fdddd 组 39 条里：

- 35 条 terminal 0。
- 37 条 terminal <10。
- 0 条 transfer。
- dataset 50/51/57/77 的错误 transfer 全部变成 0。
- dataset 32/41/64/78/86/90/96 这些 upstream-only / sim-only 类异常全部变成 0。

这说明这些任务本身不是坏任务；4o-mini 的 Stage 4 contract adherence / planning completeness 不够，是主要失败来源之一。

### 2. dataset 13 从 Stage 5 shrink 高罚变成基本可接受

v1.1b + 4o-mini 下 dataset 13 one-shot 是 terminal 21。

4.1-mini r3 是 `[6,6,0]`。

这说明 dataset 13 并不是任务不可解。4o-mini 常见的 Stage 5 把 selected 缩成只剩 `bad_wifi_calling` 的行为，在 4.1-mini 下显著减少。

### 3. ddddd partial-repair 也明显改善，但仍有 ordinary_defer 边界问题

ddddd 组：

- dataset 5 从 12 变 `[6,6,6]`。
- dataset 8 从 17 变 `[0,10,0]`。
- dataset 11 从 12 变 `[0,0,0]`。

唯一高罚是 dataset 8 的一次 terminal 10。raw chain 显示：

- Stage 4 把 `data_usage_exceeded` 也放进 selected。
- final `repair_all`。
- oracle deferred 应该包含 `data_usage_exceeded`。

这说明 4.1-mini 仍会偶发误选 ordinary_defer blocker，但频率远低于 4o-mini / v1.1c 的回潮。

### 4. fffff fast-exact path 仍是残留难点

4.1-mini 没完全修好的主要集中在 fffff：

- dataset 18: `[12.5,12.5,12.5]`
- dataset 20: `[0,12.5,12.5]`
- dataset 29: `[0,12.5,0]`

raw chain 很一致：

- expected 是 `repair_all`。
- oracle selected 包含 `bad_network_preference`, `bad_wifi_calling`, app permission。
- Stage 4 输出 `partially_repairable`。
- Stage 4 selected 只有 `data_mode_off` + `bad_network_preference`。
- Stage 4 deferred `bad_wifi_calling` 和 app permission。
- replay 缺 `toggle_wifi_calling` 和/或 `grant_app_permission`。

这说明 fast exact path 下，downstream MMS/app-permission closure 仍有结构性难点。它不是完全靠换更强 mini 模型就能 100% 消掉。

### 5. dataset 79 从 transfer 变成 partial-repair selected 不完整

v1.1b + 4o-mini 下 dataset 79 是 transfer 18.5。

4.1-mini r3 是 `[15,0,12]`。

raw chain：

- 不再 transfer。
- 失败样本中 Stage 4 selected 了 `airplane_mode_on`, `unseat_sim_card` 等上游。
- 但漏 `user_abroad_roaming_enabled_off` 或 downstream `bad_wifi_calling` / APN。

所以 4.1-mini 修掉了“错误 transfer”，但 selected completeness 仍偶发不足。

## 是否符合预期

符合，而且很有信息量。

符合点：

- 如果怀疑“是不是任务太难，4o-mini 本身无法完成”，这轮强烈支持这个怀疑：同样 v1.1b prompt，4.1-mini 把异常 21 条中的大多数修掉了。
- 错误 transfer 从 5/21 one-shot 基线降到 0/63。
- fdddd 组从大面积高罚变成 37/39 terminal <10、35/39 terminal 0。
- ddddd partial-repair 也明显改善。

不完全符合点：

- fffff fast-exact path 仍有 6/15 高罚，集中在 downstream Wi-Fi/app-permission closure。
- ordinary_defer 仍有一次误选 `data_usage_exceeded`，说明 partial-repair contract 不是完全靠模型解决。
- dataset 79 仍不稳，虽然 transfer 消失，但 selected completeness 仍有 2/3 高罚。

## 结论

这批异常不是“任务坏了”。更准确的判断是：

- 很大一部分是 `gpt-4o-mini` 在 Stage 4/5 selected-deferred contract 上的能力/遵循度不足。
- `gpt-4.1-mini` 在同一 v1.1b prompt 下显著更稳，尤其修掉了 fdddd 的错误 transfer 和 upstream-only local-chain failures。
- 但还有少量结构性 prompt/path 问题，主要是 `fffff` fast exact path 下的 downstream MMS/app-permission closure，以及 partial-repair ordinary_defer 边界。

如果目标是证明 clean local execution 本身可行，`gpt-4.1-mini` 结果很支持这一点。

如果目标是继续用 `gpt-4o-mini` 做正式 smoke，则还需要 prompt/validator 层补强；否则 4o-mini 会继续把一部分本地链任务打成高罚。

如果目标是尽快得到更干净的 execution-layer terminal signal，我建议下一步用 `gpt-4.1-mini + v1.1b` 跑更大的 stage-exact clean subset 或 clean-100，但仍不要直接进 PS smoke；先确认 full clean-100 的 abnormal rate 是否也能从 21% 明显降下去。
