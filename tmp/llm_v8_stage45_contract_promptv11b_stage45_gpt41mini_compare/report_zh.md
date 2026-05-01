# Stage 4/5 gpt-4.1-mini + v1.1b prompt-only abn21 报告

## 配置名

`llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig`

本轮目的：

- 不换 prompt，仍用 v1.1b。
- 只把 Stage 4 / Stage 5 的 LLM 改为 `gpt-4.1-mini`。
- Stage 1 / Stage 2 / Stage 3 仍用 `gpt-4o-mini`。
- 只跑 v1.1b clean-100 stage-exact 中那 21 条异常任务。
- 不 repeat。
- 不跑 smoke。

新增 opt-in 环境变量：

`PSAGENT_TELECOM_STAGE45_MODEL=gpt-4.1-mini`

这个开关只在 `stage_name in {"stage4", "stage5"}` 时覆盖 bridge payload model。默认不设置时行为不变，因此不会影响正在跑的 all-stage gpt-4.1-mini 实验。

本轮实际模型核验：

| stage | model | count |
|---|---|---:|
| Stage 1 | `gpt-4o-mini` | 21 |
| Stage 2 | `gpt-4o-mini` | 21 |
| Stage 3 | `gpt-4o-mini` | 21 |
| Stage 4 | `gpt-4.1-mini` | 21 |
| Stage 5 | `gpt-4.1-mini` | 21 |

## 代码改动

改动是 opt-in 诊断支持，不改变默认行为：

- `envs/executors/telecom_llm_bench_executor.py`
  - 新增 `_model_for_stage(stage_name)`。
  - `_run_llm_stage_bridge()` 在 Stage 4/5 时读取 `PSAGENT_TELECOM_STAGE45_MODEL`。
  - `stage_resource_summary` 增加 `model` 字段，便于报告核对。
- `envs/executors/_telecom_llm_bench_bridge.py`
  - bridge result 透传实际 `model`。
- `scripts/run_llm_path_sweep_diagnostic.py`
  - `flatten_stage_resource_summary()` 保留 stage model 字段。

仍然没有做：

- 没有改 PS。
- 没有改 terminal penalty。
- 没有加 retry。
- 没有 normalizer 自动纠正。
- 没有改 v1.1b prompt。
- 没有删除数据。

验证：

`python -m py_compile envs/executors/telecom_llm_bench_executor.py envs/executors/_telecom_llm_bench_bridge.py scripts/run_llm_path_sweep_diagnostic.py scripts/run_llm_fixed_profile_trace_diagnostic.py`

## 运行产物

分 required pattern 跑，全部 stage-exact：

- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_fdddd_exact_r1_seed1_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_fffff_exact_r1_seed1_cconfig/`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_ddddd_exact_r1_seed1_cconfig/`

合并输出：

- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_stageexact_r1_seed1_cconfig/records.json`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_stageexact_r1_seed1_cconfig/summary.json`

这三个本轮 stage45-only r1 tmux run 均 `exit_code=0`。

注意：当前另有一个 `psagent_promptv11b_gpt41mini_abn21_fdddd_r3` tmux session 是 all-stage `gpt-4.1-mini`、`repeats=3` 的独立实验，不属于本报告，我没有干预它。

## 总体结果

对比对象是 v1.1b + gpt-4o-mini Stage 4/5 的 clean-100 stage-exact one-shot 异常 21 条。

| run | n | terminal mean | terminal 0 | terminal <10 | terminal >=10 | transfer | final counts |
|---|---:|---:|---:|---:|---:|---:|---|
| v1.1b + Stage4/5 gpt-4o-mini | 21 | 14.786 | 0 | 0 | 21 | 5 | `repair_subset:16, transfer:5` |
| v1.1b + Stage4/5 gpt-4.1-mini | 21 | 1.714 | 17 | 19 | 2 | 0 | `repair_all:17, repair_subset:4` |

结论非常明显：只换 Stage 4/5 模型，不改 prompt，就把 21 条异常中的 19 条拉到 terminal `<10`，17 条变成 terminal 0，并且 transfer 从 5 条降到 0。

这强烈支持一个判断：这些错误里很大一部分不是任务本身不可解，也不是必须靠 normalizer/retry/PS 修；主要瓶颈确实在 `gpt-4o-mini` 对 Stage 4/5 contract 的执行稳定性。

## 明细

| dataset | required | expected | v1.1b 4o-mini terminal/final | Stage4/5 gpt-4.1-mini terminal/final | 判断 |
|---:|---|---|---|---|---|
| 5 | ddddd | repair_subset | 12 / repair_subset | 6 / repair_subset | 改善，selected/deferred 正确，剩 auxiliary floor |
| 8 | ddddd | repair_subset | 17 / repair_subset | 12 / repair_all | 仍失败，Stage 3 漏 `data_usage_exceeded`，Stage 4/5 无法 defer |
| 11 | ddddd | repair_subset | 12 / repair_subset | 0 / repair_subset | 修好 |
| 13 | fdddd | repair_subset | 21 / repair_subset | 6 / repair_subset | 明显改善，Stage 5 不再缩 selected |
| 18 | fffff | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 19 | fffff | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 20 | fffff | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 21 | fffff | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 29 | fffff | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 32 | fdddd | repair_all | 13 / repair_subset | 0 / repair_all | 修好 |
| 41 | fdddd | repair_all | 15 / repair_subset | 0 / repair_all | 修好 |
| 50 | fdddd | repair_all | 18.5 / transfer | 0 / repair_all | 修好，错误 transfer 消失 |
| 51 | fdddd | repair_all | 18.5 / transfer | 0 / repair_all | 修好，错误 transfer 消失 |
| 57 | fdddd | repair_all | 18 / transfer | 0 / repair_all | 修好，错误 transfer 消失 |
| 64 | fdddd | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 77 | fdddd | repair_all | 18 / transfer | 0 / repair_all | 修好，Stage 5 不再改 transfer |
| 78 | fdddd | repair_all | 12 / repair_subset | 0 / repair_all | 修好 |
| 79 | fdddd | repair_all | 18.5 / transfer | 12 / repair_subset | 部分改善，transfer 消失但仍漏 roaming repair |
| 86 | fdddd | repair_all | 15 / repair_subset | 0 / repair_all | 修好 |
| 90 | fdddd | repair_all | 13 / repair_subset | 0 / repair_all | 修好 |
| 96 | fdddd | repair_all | 17 / repair_subset | 0 / repair_all | 修好 |

## 按 pattern

| required_pattern | n | terminal values | mean | terminal <10 | transfer |
|---|---:|---|---:|---:|---:|
| ddddd | 3 | `[6,12,0]` | 6.000 | 2/3 | 0 |
| fdddd | 13 | `[6,0,0,0,0,0,0,0,0,12,0,0,0]` | 1.385 | 12/13 | 0 |
| fffff | 5 | `[0,0,0,0,0]` | 0.000 | 5/5 | 0 |

## 真实因果链

### 1. upstream-only / sim-only / downstream漏修基本被修掉

v1.1b + 4o-mini 的常见失败：

- Stage 4 只 selected `unseat_sim_card`、`data_mode_off` 或 `bad_network_preference`。
- `bad_wifi_calling`、APN、app permission downstream 被 deferred。
- Stage 5 replay 缺少 `toggle_wifi_calling`、`reset_apn_settings`、`grant_app_permission`。
- terminal 12/13/15/17。

Stage 4/5 换成 `gpt-4.1-mini` 后：

- `fffff` 的 18/19/20/21/29 全部 terminal 0。
- `fdddd` 的 32/41/64/78/86/90/96 全部 terminal 0。
- selected/deferred 与 oracle 对齐。
- replay missing tools 清空。

这说明 upstream/downstream chain comprehension 主要是模型执行能力问题。

### 2. 无 concrete hard blocker 的错误 transfer 基本消失

v1.1b + 4o-mini 中：

- dataset 50/51/57/77/79 都出现 transfer。
- 这些大多是 local repair-all expected case，属于错误保守 transfer。

Stage 4/5 `gpt-4.1-mini` 后：

- 21 条里 transfer count = 0。
- 50/51/57/77 全部 terminal 0。
- 79 从 transfer 18.5 降到 repair_subset 12。

这说明 `gpt-4.1-mini` 明显更能遵守 v1.1b 的 “no concrete hard blocker -> do not transfer_required” 约束。

### 3. dataset 13 的 Stage 5 shrink selected 被修掉

v1.1b + 4o-mini 中，dataset 13 的典型失败：

- Stage 4 selected 完整 local chain。
- Stage 5 把 selected 缩成只剩 `bad_wifi_calling`。
- terminal 21。

Stage 4/5 `gpt-4.1-mini` 后：

- Stage 4 selected 完整 local chain。
- `data_usage_exceeded` deferred。
- Stage 5 保留 selected/deferred。
- terminal 6。

这说明 dataset 13 不是任务不可解，而是 4o-mini Stage 5 preservation / repair_subset contract 不稳。

### 4. dataset 8 的残留失败主要来自 Stage 3 输入漏 blocker

dataset 8 仍 terminal 12，final `repair_all`。

关键 raw chain：

- raw task blockers 包含 `data_usage_exceeded`。
- Stage 3 `raw_task_blocker_ids` 里也有 `data_usage_exceeded`。
- 但 Stage 3 `inferred_blocker_ids` 和 `per_blocker` 漏掉了 `data_usage_exceeded`。
- Stage 4/5 只看见 7 个 local blockers，因此输出 `repairable / repair_all`。
- oracle 需要把 `data_usage_exceeded` 放 deferred，所以 final 缺 deferred，terminal 12。

判断：

- 这条不是 Stage 4/5 换模型能完全解决的，因为 Stage 4 的 input blocker set 已经缺 `data_usage_exceeded`。
- 如果要修 dataset 8，需要 Stage 3 blocker inference 更稳，或者允许 Stage 4 看到 raw_task_blocker_ids 并要求补回 missing active/deferred blockers。但这已经不是本轮 “只换 Stage 4/5 model” 的范围。

### 5. dataset 79 的残留失败仍是 Stage 4 classification 错误

dataset 79 从 transfer 18.5 降到 repair_subset 12，但仍没到 0。

关键 raw chain：

- expected `repair_all`。
- oracle selected 包含 `user_abroad_roaming_enabled_off`。
- Stage 4 selected：`airplane_mode_on`, `unseat_sim_card`, `bad_network_preference`, `bad_wifi_calling`, `break_apn_mms_setting`。
- Stage 4 deferred：`user_abroad_roaming_enabled_off`。
- Stage 5 保留这个 plan，final `repair_subset`。
- replay missing `toggle_roaming`。

判断：

- `gpt-4.1-mini` 修掉了错误 transfer，但仍把一个本地可修 roaming blocker 当成 deferred。
- 这说明更强模型也不是 100% 修复；仍存在少量 Stage 4 local-vs-defer 分类边界问题。

## 是否符合预期

符合，而且比预期更强：

- 如果任务本身太难或 oracle/terminal 坏，单纯替换 Stage 4/5 模型不应该把 21 条里 19 条拉到 `<10`。
- 现在结果显示 17/21 terminal 0、19/21 `<10`、0 transfer，说明大部分失败确实是 4o-mini 的 Stage 4/5 contract adherence 问题。
- dataset 13、50/51/57/77 这些原本看起来像 prompt/contract 歧义的 case，在 gpt-4.1-mini 下直接稳定得多。

不完全符合：

- dataset 8 仍失败，但原因更像 Stage 3 漏 `data_usage_exceeded`，不是 Stage 4/5 无法理解。
- dataset 79 仍失败，说明 Stage 4 local/defer 分类仍有少量模型错误。

## 结论

本轮强烈支持你的怀疑：`gpt-4o-mini` 本身可能不足以稳定完成这些 Stage 4/5 contract-heavy 任务。

更精确地说：

- 不是 21 条任务整体太难。
- 不是必须靠 v1.1c chain-closure 这种更强 prompt 才能修。
- 在保持 v1.1b prompt 不变、不加 retry、不改 normalizer 的情况下，Stage 4/5 换成 `gpt-4.1-mini` 就能修掉绝大多数异常。

建议下一步不要立刻用 v1.1c；如果预算允许，优先把 Stage 4/5-only `gpt-4.1-mini` 作为新的执行层候选，先跑 focus 2/10/13/16 fixed trace，再跑 clean-100 stage-exact r1，看是否同时保住 dataset 10 的 ordinary_defer 和 dataset 16 的 no-transfer 稳定性。

