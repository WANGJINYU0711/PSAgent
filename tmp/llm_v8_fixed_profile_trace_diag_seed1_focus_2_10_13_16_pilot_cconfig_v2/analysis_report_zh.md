# fixed-profile / fixed-path Stage 4/5 trace diagnostic

## 目的

本轮只做诊断，不修改 executor、terminal rule、PS 更新逻辑或任务集。

目标是验证：`fdddd on fdddd` 这种 exact mode match 仍高 terminal 时，到底是任务应该删除，还是 Stage 4/5 的接口、归一化、工具回放、terminal rule 在某些 path 上不稳定。

## 实验名

`llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2`

## 配置

- 数据集：`data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- bucket：`analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- focus datasets：`2, 10, 13, 16`
- fixed patterns：`fdddd`, `ffddd`, `ddddd`
- observed high-terminal exact paths：来自 `tmp/llm_v8_seed1_old_vs_probfloor_targeted_diagnostic/aligned_old_vs_probfloor.csv`
- repeats：pilot v1 + v2 各 1 次；v2 输出完整 trace 字段
- model：`gpt-4o-mini`
- C config：
  - `PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1`
  - `PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1`
  - `PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1`

## 产物

- v2 完整记录：`tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2/records.json`
- v2 CSV：`tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2/records.csv`
- v2 自动摘要：`tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2/report.md`
- v2 dataset/pattern 汇总：`tmp/llm_v8_fixed_profile_trace_diag_seed1_focus_2_10_13_16_pilot_cconfig_v2/summary_by_dataset_pattern.json`

## Step 1: trace persistence 已完成

本轮没有改执行行为，只增强诊断导出。

新增脚本：

- `scripts/run_llm_fixed_profile_trace_diagnostic.py`

增强已有 fixed-path diagnostic 导出：

- `scripts/run_llm_path_sweep_diagnostic.py`

现在每条 record 持久化这些关键字段：

- Stage 4 raw：`stage4_llm_raw_output`, `stage4_raw_json_extracted`
- Stage 4 normalized：`stage4_output`, `stage4_selected_before_normalization`, `stage4_selected_after_normalization`, `stage4_deferred_before_normalization`, `stage4_deferred_after_normalization`
- Stage 4 completion：`stage4_completion_pass_applied`, `stage4_completion_added_blockers`, `stage4_completion_blocked_by_hard_transfer_guard`
- Stage 5 raw/normalized：`stage5_llm_raw_output`, `stage5_output`, `stage5_raw_action_hint`
- replay/tools：`stage5_replay_tool_names`, `stage5_executed_tool_names`, `stage4_executed_tool_names`
- oracle diff：`selected_missing_vs_oracle`, `deferred_missing_vs_oracle`, `oracle_tools_missing_from_stage5_replay`, `stage5_replay_tools_extra_vs_oracle`
- terminal rule：`terminal_adjustment_reasons`, `terminal_clear_success_proxy`, `terminal_auxiliary_success_proxy`, `raw_terminal_penalty_exec_clean_v4`

验证：

- `python -m py_compile scripts/run_llm_path_sweep_diagnostic.py scripts/run_llm_fixed_profile_trace_diagnostic.py` 通过。
- v2 tmux run `EXIT_CODE=0`。

## Step 2: fixed-path 诊断结果

下面这张表合并了 pilot v1 和 v2。每个 cell 的样本仍小，但足够判断“是否稳定 artifact”。

| dataset | group | pattern | n | terminal values | final counts | low/high | main issue |
|---:|---|---|---:|---|---|---|---|
| 2 | canonical | `ddddd` | 2 | `[0.0, 10.0]` | `repair_all:1, repair_subset:1` | `<=6:1, >=10:1` | 有一次漏 `airplane_mode_on/unseat_sim_card` |
| 2 | canonical | `fdddd` | 2 | `[18.5, 10.0]` | `transfer:1, repair_subset:1` | `>=10:2` | 一次 deferred-all/transfer，一次漏 service prerequisites |
| 2 | canonical | `ffddd` | 2 | `[18.5, 18.5]` | `transfer:2` | `>=10:2` | 稳定 deferred-all/transfer |
| 2 | observed exact | `fdddd` | 6 | `[10.0, 13.0, 18.5, 10.0, 18.5, 18.5]` | `repair_subset:3, transfer:3` | `>=10:6` | service prerequisites 被 Stage 4 判成 defer 或全 defer |
| 10 | canonical | `ddddd` | 2 | `[20.5, 23.0]` | `transfer:1, repair_subset:1` | `>=10:2` | repair/defer 语义反转或漏大多数 selected blockers |
| 10 | canonical | `fdddd` | 2 | `[12.0, 15.0]` | `repair_subset:2` | `>=10:2` | 漏 prerequisites 或把 deferred blockers 放进 selected |
| 10 | canonical | `ffddd` | 2 | `[23.5, 23.5]` | `transfer:2` | `>=10:2` | 稳定 deferred-all/transfer |
| 10 | observed exact | `fdddd` | 6 | `[6.0, 15.0, 19.0, 6.0, 15.0, 6.0]` | `repair_subset:6` | `<=6:3, >=10:3` | 不是必然 transfer；高罚来自 selected/deferred 边界不稳 |
| 13 | canonical | `ddddd` | 2 | `[6.0, 24.5]` | `repair_subset:1, transfer:1` | `<=6:1, >=10:1` | 同 pattern 跨 run 从低罚变 transfer |
| 13 | canonical | `fdddd` | 2 | `[6.0, 24.5]` | `repair_subset:1, transfer:1` | `<=6:1, >=10:1` | 同 pattern 跨 run 从低罚变 transfer |
| 13 | canonical | `ffddd` | 2 | `[17.0, 25.5]` | `repair_subset:1, transfer:1` | `>=10:2` | prerequisites/deferred miss 或 transfer |
| 13 | observed exact | `fdddd` | 2 | `[24.5, 17.0]` | `transfer:1, repair_subset:1` | `>=10:2` | high terminal 但 action 不稳定 |
| 16 | canonical | `ddddd` | 2 | `[0.0, 22.5]` | `repair_all:1, transfer:1` | `<=6:1, >=10:1` | 同 pattern 一次全修，一次漏上游后 transfer |
| 16 | canonical | `fdddd` | 2 | `[15.0, 0.0]` | `repair_subset:1, repair_all:1` | `<=6:1, >=10:1` | exact `fdddd` 可 0 罚，也可漏 prerequisites |
| 16 | canonical | `ffddd` | 2 | `[22.5, 15.0]` | `transfer:1, repair_subset:1` | `>=10:2` | 漏上游 service/data blockers |
| 16 | observed exact | `fdddd` | 2 | `[0.0, 0.0]` | `repair_all:2` | `<=6:2` | observed high exact path 重跑后稳定 0 罚 |

## Step 3: 分类

| dataset | oracle | 当前分类 | 原因 | 是否建议现在删除 |
|---:|---|---|---|---|
| 2 | `repair_all` | `diagnostic-only` for current smoke-sensitive exact-path analysis; task itself not exclude | `ffddd` 和 observed exact `fdddd` 高罚比较稳定，原因是 Stage 4 把 `airplane_mode_on/unseat_sim_card` 等上游 service prerequisites 判成 defer，甚至 deferred-all 后 transfer；但 `ddddd` 在 v1 可 0 罚，说明任务 oracle 不是必坏 | 否 |
| 10 | `repair_subset` | `keep-main + targeted diagnostic` | 这不是“该删”的任务。observed exact `fdddd` 6 次里 3 次只有 6 罚且全是 `repair_subset`，不是总 transfer。主要问题是 Stage 4/5 把 `user_abroad_roaming_disabled_on/data_usage_exceeded` 和 local selected blockers 的边界表达不稳，有时还把应 defer 的 blocker 放进 selected | 否 |
| 13 | `repair_subset` | `diagnostic-only` until trace repeats are cleaner | v1 的 `fdddd/ddddd` 可 6 罚，v2 同 pattern 变 24.5 transfer，说明 LLM/interface stochasticity 很强；不是 oracle 明显坏，而是同 fixed pattern 不稳定 | 否 |
| 16 | `repair_all` | `keep-main + targeted diagnostic` | canonical `fdddd` v2 为 0 罚，observed exact `fdddd` 两次重跑都是 0 罚；之前 high terminal 不能复现为稳定任务缺陷。`ffddd/ddddd` 的高罚来自漏上游 blockers 或转 transfer | 否 |

没有发现应立刻 `exclude-from-smoke` 的 dataset。当前更合理的处理是：

- `keep-main`：dataset 10、16。
- `diagnostic-only`：dataset 2 的 exact-path 高罚簇、dataset 13 的 high-variance 簇。
- `exclude-from-smoke`：暂时为空；除非后续证明 oracle/terminal rule 自相矛盾。

## 关键归因

### 1. `fdddd on fdddd` 高 terminal 不是单纯 deep/fast ratio 问题

fixed path 已经把学习关掉，仍能复现 high terminal。因此 PS seed sensitivity 的根因不是 PS 更新本身，而是 cost feedback 里有一批 path 的执行结果不 clean。

PS 学的是 terminal outcome，不是 mode label。如果 exact-match `fdddd` 有时高罚，它就会被推离正确 profile，换 seed 后 winner 改变就很自然。

### 2. dataset 10 不是总 transfer

之前 probfloor seed1 repeated smoke 里 dataset 10 是 `8/8 transfer`，但 fixed-path 诊断不支持“dataset 10 必然 transfer”。

在 observed exact `fdddd` 重跑中：

- 6 次全是 `repair_subset`，没有 final `transfer`。
- 3 次 terminal 是 `6.0`。
- 3 次 terminal 是 `15.0/19.0`，主要因为 Stage 4 selected/deferred 边界错：上游 `airplane_mode_on/unseat_sim_card/data_mode_off` 被 deferred，或者 `user_abroad_roaming_disabled_on/data_usage_exceeded` 被放入 selected。

所以 dataset 10 更像 Stage 5 输出规范和 partial repair interface 问题，不像任务坏。

### 3. dataset 2/16 的高罚大多是漏上游 prerequisites

dataset 2 的典型失败：

- Stage 4 raw JSON 将 `airplane_mode_on`、`unseat_sim_card` 标成 `should_repair=false`。
- Stage 5 replay 因此缺 `toggle_airplane_mode`、`reseat_sim_card`。
- terminal reasons 通常包含 `subset_mismatch_base_plus_linear` 和 `local_clear_failure_floor_10`；transfer 时还会触发 `invalid_local_transfer_floor_18`。

dataset 16 的 exact `fdddd` 能稳定 0 罚复现，说明它不该删。失败 run 里问题类似：Stage 4 只选下游 MMS blockers，漏 service/data prerequisites。

### 4. completion pass 没有救这些 case

v1/v2 里 `stage4_completion_pass_applied` 基本都是 `0.0`。也就是说当前失败不是 completion pass 已经尝试但做错，而是 Stage 4 raw/normalization 把 selected/deferred 边界定错后，没有触发可补救路径。

这点非常关键：下一步如果要修，应优先看 Stage 4 local upstream completion / partial-repair selected-deferred contract，而不是继续调 PS 的 `eta_shared/prob_floor`。

## 当前判断

现在不应该删除这些任务后直接重跑 confirmatory smoke。

更准确的结论是：

1. `fdddd on fdddd` 高 terminal 是真实信号，但不是“deep profile 不好”，而是 Stage 4/5 interface 对 local repair chain 不稳定。
2. dataset 10 是最值得保留和修的 case：它暴露了 `repair_subset` 的 selected/deferred contract 问题。
3. dataset 2/13 可以暂时从主结论里单独标成 diagnostic-only，但不应从数据集中物理删除。
4. dataset 16 应保留；observed high exact path 在 fixed rerun 中变成 0 罚，说明之前更多是 stochastic/path-interface failure。

## 建议下一步

仍然先不改 PS。

建议下一步做一个更小但重复更多的 trace run：

- datasets：`2, 10, 13, 16`
- patterns：`fdddd`
- repeats：`5`
- parallelism：`4`
- 保留当前 v2 trace 字段

如果重复后：

- dataset 10 的 `repair_subset` 仍经常把 defer blocker 放入 selected，则修 Stage 5/terminal partial-repair contract。
- dataset 2/16 仍经常漏 service prerequisites，则修 Stage 4 local upstream completion。
- 只有个别 run 高罚，且 raw JSON 本身随机，则考虑把这些作为 diagnostic-only split 单独报告，而不是从 smoke 删除。
