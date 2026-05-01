# Stage 4/5 contract prompt v1 prompt-only 回归报告

## 实验配置名

本轮改动配置名：

`llm_v8_stage45_contract_promptv1_promptonly_cconfig`

两个 fixed-trace run：

- `llm_v8_stage45_contract_promptv1_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig`
- `llm_v8_stage45_contract_promptv1_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig`

## 和上一版的区别

上一版：

- `terminalv4 + reasoncalibv3 + reportmodecost`
- fixed trace 只持久化 Stage 4/5 raw/normalized trace。
- 不改变 prompt contract。

本版：

- 仍然保留 `terminalv4 + reasoncalibv3 + reportmodecost`。
- 新增 env flag：`PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1=1`。
- 只改 Stage 4/5 prompt，不新增 retry，不让 normalizer 替 LLM 修改 selected/deferred/final_action。
- Stage 4 prompt 增强：
  - selected/deferred contract。
  - prerequisite closure contract。
  - `repair_subset != transfer`。
  - local / ordinary defer / hard transfer 三分类。
  - few-shot contract examples。
- Stage 5 prompt 增强：
  - Stage 5 是 verification/terminal closure，不是重新 planning。
  - 默认继承 Stage 4 selected/deferred。
  - `repair_subset` 的成功条件是 `partial_resolution_only`，不要求 `can_send_mms=true`。
  - incomplete evidence 不应直接 transfer，除非有 explicit hard transfer reason。
- report-only 字段：
  - `stage4_contract_prompt_version`
  - `stage4_contract_self_check`
  - `stage5_contract_prompt_version`
  - `stage5_contract_self_check`

注意：`contract_self_check` 不参与任何 decision。当前 LLM 大多数输出仍没有主动填这个字段，所以本轮结论主要来自 raw Stage 4/5 trace、selected/deferred diff、terminal reasons 和 cost，而不是 self-check 字段。

## 代码改动范围

- `envs/executors/telecom_llm_bench_executor.py`
  - 新增 prompt-only contract v1。
  - 新增 report-only self-check 字段透传。
- `scripts/run_llm_path_sweep_diagnostic.py`
  - 导出 contract prompt/self-check 字段到 records/CSV。
- 没有修改 PS。
- 没有修改 terminal penalty 语义。
- 没有新增 retry。
- 没有让 normalizer 新增替 LLM 改 selected/deferred 的行为。

验证：

- `python -m py_compile envs/executors/telecom_llm_bench_executor.py scripts/run_llm_path_sweep_diagnostic.py scripts/run_llm_fixed_profile_trace_diagnostic.py` 通过。

## Run 1: fdddd repeat=5

输出目录：

`tmp/llm_v8_stage45_contract_promptv1_promptonly_fixedtrace_fdddd_r5_seed1_focus_2_10_13_16_cconfig`

| dataset | pattern | n | total | terminal | reasoning | path | token penalty | final counts | high terminal >=10 | selected missing | deferred missing | replay tool missing |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| 2 | `fdddd` | 5 | 9.23 | 4.80 | 4.35 | 0.08 | 3.00 | `repair_all:3, repair_subset:2` | 2 | `none:3, break_app_storage_permission:2` | `none:5` | `none:3, grant_app_permission:2` |
| 10 | `fdddd` | 5 | 15.53 | 11.00 | 4.45 | 0.08 | 3.25 | `repair_subset:2, repair_all:3` | 4 | `none:4, upstream-local-missing:1` | `none:1, data_usage_exceeded+roaming_disabled_on:4` | `none:4, upstream-tools-missing:1` |
| 13 | `fdddd` | 5 | 10.03 | 6.00 | 3.95 | 0.08 | 3.25 | `repair_subset:5` | 0 | `none:5` | `none:5` | `none:5` |
| 16 | `fdddd` | 5 | 3.91 | 0.00 | 3.83 | 0.08 | 3.00 | `repair_all:5` | 0 | `none:5` | `none:5` | `none:5` |

和上一版 canonical `fdddd` 对比：

| dataset | old terminal values | old mean | old high>=10 | new terminal values | new mean | new high>=10 | 主要变化 |
|---:|---|---:|---:|---|---:|---:|---|
| 2 | `[18.5, 10.0]` | 14.25 | 2/2 | `[0.0, 0.0, 12.0, 12.0, 0.0]` | 4.80 | 2/5 | 明显改善；transfer 消失，但仍有 storage permission 漏 repair |
| 10 | `[12.0, 15.0]` | 13.50 | 2/2 | `[6.0, 10.0, 10.0, 19.0, 10.0]` | 11.00 | 4/5 | 小幅改善但仍不达标；主要剩余问题是 ordinary defer 被 selected 成 repair_all，或 selected/deferred 反转 |
| 13 | `[6.0, 24.5]` | 15.25 | 1/2 | `[6.0, 6.0, 6.0, 6.0, 6.0]` | 6.00 | 0/5 | 明显稳定；transfer 消失 |
| 16 | `[15.0, 0.0]` | 7.50 | 1/2 | `[0.0, 0.0, 0.0, 0.0, 0.0]` | 0.00 | 0/5 | 明显稳定；repair_all 全部成功 |

## Run 2: fdddd / ffddd / ddddd repeat=3

输出目录：

`tmp/llm_v8_stage45_contract_promptv1_promptonly_fixedtrace_3patterns_r3_seed1_focus_2_10_13_16_cconfig`

| dataset | pattern | n | total | terminal | reasoning | path | token penalty | final counts | high terminal >=10 | selected missing | deferred missing | replay tool missing |
|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|
| 2 | `ddddd` | 3 | 9.24 | 4.00 | 5.16 | 0.08 | 0.00 | `repair_all:2, repair_subset:1` | 1 | `none:2, break_app_storage_permission:1` | `none:3` | `none:2, grant_app_permission:1` |
| 2 | `fdddd` | 3 | 18.37 | 14.17 | 4.12 | 0.08 | 3.00 | `transfer:1, repair_subset:2` | 3 | `all-local-missing:1, break_app_storage_permission:2` | `none:3` | `all-tools-missing:1, grant_app_permission:2` |
| 2 | `ffddd` | 3 | 12.33 | 8.00 | 4.25 | 0.08 | 6.00 | `repair_subset:2, repair_all:1` | 2 | `break_app_storage_permission:2, none:1` | `none:3` | `grant_app_permission:2, none:1` |
| 10 | `ddddd` | 3 | 14.13 | 8.67 | 5.38 | 0.08 | 0.00 | `repair_subset:1, repair_all:2` | 2 | `none:3` | `none:1, data_usage_exceeded+roaming_disabled_on:2` | `none:3` |
| 10 | `fdddd` | 3 | 13.35 | 8.67 | 4.61 | 0.08 | 3.25 | `repair_all:2, repair_subset:1` | 2 | `none:3` | `data_usage_exceeded+roaming_disabled_on:2, none:1` | `none:3` |
| 10 | `ffddd` | 3 | 15.84 | 12.00 | 3.76 | 0.08 | 6.25 | `repair_subset:3` | 3 | `none:3` | `data_usage_exceeded:3` | `none:3` |
| 13 | `ddddd` | 3 | 10.79 | 6.00 | 4.71 | 0.08 | 0.00 | `repair_subset:3` | 0 | `none:3` | `none:3` | `none:3` |
| 13 | `fdddd` | 3 | 10.02 | 6.00 | 3.95 | 0.08 | 3.25 | `repair_subset:3` | 0 | `none:3` | `none:3` | `none:3` |
| 13 | `ffddd` | 3 | 23.78 | 19.67 | 4.04 | 0.08 | 6.25 | `repair_subset:3` | 3 | `upstream-local-missing:3` | `data_usage_exceeded:3` | `none:3` |
| 16 | `ddddd` | 3 | 4.44 | 0.00 | 4.36 | 0.08 | 0.00 | `repair_all:3` | 0 | `none:3` | `none:3` | `none:3` |
| 16 | `fdddd` | 3 | 3.98 | 0.00 | 3.90 | 0.08 | 3.00 | `repair_all:3` | 0 | `none:3` | `none:3` | `none:3` |
| 16 | `ffddd` | 3 | 3.82 | 0.00 | 3.74 | 0.08 | 6.00 | `repair_all:3` | 0 | `none:3` | `none:3` | `none:3` |

## 真实输出因果链

### 改善链路 1：dataset 16

上一版：`fdddd` 有时 `repair_subset` terminal 15，原因是漏上游 service/data blockers。

本版：

- `fdddd r5`: 5/5 `repair_all`, terminal 0。
- expanded 中 `fdddd/ffddd/ddddd`: 全部 terminal 0。

解释：Stage 4 prompt 的 prerequisite closure / repair_all contract 对 dataset 16 起效。LLM 更稳定地选择完整 local repair chain，Stage 5 replay tools 覆盖 oracle tools。

### 改善链路 2：dataset 13 的 fdddd/ddddd

上一版：`fdddd/ddddd` 在 v1/v2 之间从 terminal 6 跳到 24.5 transfer。

本版：

- `fdddd r5`: 5/5 terminal 6，全部 `repair_subset`。
- expanded `fdddd`: 3/3 terminal 6。
- expanded `ddddd`: 3/3 terminal 6。

解释：prompt-only contract 明显减少了无硬原因 transfer。dataset 13 对 exact `fdddd` 已经稳定很多。

### 剩余失败链路 1：dataset 10

本版没有再出现大量 transfer，这是好事；但 selected/deferred contract 仍不稳定。

三类失败：

1. 正确 partial repair：
   - selected local chain 完整。
   - deferred 是 `user_abroad_roaming_disabled_on`, `data_usage_exceeded`。
   - final `repair_subset`。
   - terminal 6。

2. ordinary defer 被 selected 成 `repair_all`：
   - selected 包含 `user_abroad_roaming_disabled_on`, `data_usage_exceeded`。
   - deferred 为空。
   - final `repair_all`。
   - terminal 10，原因是 oracle expected `repair_subset`，但 LLM 把 ordinary deferred blockers 当成可修 selected。

3. selected/deferred 反转：
   - selected 只含下游 APN/app/ordinary defer。
   - deferred 反而包含 `airplane_mode_on`, `unseat_sim_card`, `data_mode_off`, `bad_network_preference`, `bad_wifi_calling`。
   - terminal 19。

结论：prompt v1 已经把 transfer 问题压下来了，但 dataset 10 的 ordinary defer contract 还没完全被 LLM 学稳。

### 剩余失败链路 2：dataset 2

本版 dataset 2 比上一版明显改善，但还剩两种失败：

1. `break_app_storage_permission` 被 deferred：
   - terminal 12。
   - replay missing `grant_app_permission`。
   - 说明 prerequisite closure 改善了 service/SIM/data 链，但 app permission 仍会被 LLM 当成可 defer。

2. expanded `fdddd` 中仍有一次 deferred-all/transfer：
   - terminal 18.5。
   - selected 全空，all local blockers missing。
   - 说明 prompt-only 不是完全稳定，尤其 fast-heavy path 仍可能退回老问题。

### 剩余失败链路 3：dataset 13 的 ffddd

`ffddd` 仍然很差：

- 3/3 terminal 高：19, 21, 19。
- final 都是 `repair_subset`，不再 transfer。
- 但 selected missing 包含大量上游 local blockers。

解释：prompt v1 能减少 over-transfer，但 fast-heavy `ffddd` 对 dataset 13 仍会漏 upstream local repair chain。这个不是 terminal rule 问题，而是 Stage 4 fast-on-deep / fast-heavy prompt adherence 不够。

## 是否符合预期

部分符合。

符合预期的部分：

- 没有新增 normalizer 改答案行为。
- 没有新增 retry。
- dataset 16 完全稳定。
- dataset 13 的 `fdddd/ddddd` 明显稳定。
- dataset 2 明显减少 full transfer / all-missing。
- dataset 10 不再表现为“必然 transfer”。

不符合预期的部分：

- dataset 10 selected/deferred contract 仍不稳，尤其 ordinary defer 被 selected 成 `repair_all`。
- dataset 2 仍会漏 `break_app_storage_permission`。
- dataset 13 `ffddd` 仍大量漏上游 local blockers。
- `contract_self_check` report-only 字段 LLM 大多数没有输出；字段已经透传，但 prompt v1 对 self-check 的约束还不够强。它没有影响本轮决策，也没有进入 cost。

## 当前判断

prompt-only v1 是有效方向，但不够完全。

它解决或显著缓解了：

- dataset 16 的 exact path 高罚。
- dataset 13 `fdddd/ddddd` 的 transfer/high variance。
- dataset 2 的一部分 all-missing/transfer。

它没有完全解决：

- dataset 10 的 ordinary defer vs selected local repair contract。
- fast-heavy pattern 下 Stage 4 upstream completeness。
- report-only self-check 的输出稳定性。

## 下一步建议

仍然不要删任务，也不要改 PS。

建议做 prompt-only v1.1，而不是 normalizer 行为改动：

1. 把 `contract_self_check` 从 optional 改成 required diagnostic key，但继续不用于决策。
2. 在 Stage 4 prompt 中更明确地写：
   - if `can_be_deferred=true` and no local canonical repair is appropriate, keep it deferred。
   - do not mark account/usage blockers selected in a repair_subset case unless you actually executed a local canonical repair for them。
3. 给 dataset 10 类型加更强 abstract example：
   - `data_usage_exceeded` 和 account roaming policy 是 ordinary defer。
   - local MMS chain selected 后，final 应是 `repair_subset`。
4. 给 app permission blocker 加一句：
   - if app permission blocker is active and canonical local permission repair exists, do not defer it while repairing APN/Wi-Fi/MMS downstream blockers。
5. 再跑同样 fixed trace，不进入 smoke。

如果 prompt-only v1.1 仍不能稳定 dataset 10，那么再 review 是否允许“validator-only fail-fast / no auto-correction”或更强的 LLM re-ask；但这一步需要单独确认。
