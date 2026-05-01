# Telecom MMS stage4/5 gpt-4.1-mini tmux smoke 配置审计

生成时间：2026-04-30  
审计对象：当前正在 tmux 中运行的两个 seed 的 3-method repeated smoke。  
结论先行：这批 smoke 可以继续跑完，用于确认 stage4/5 换 `gpt-4.1-mini` 后的 C-config 短跑趋势；但**不建议直接据此开全量正式 LLM run**。原因是当前 run 仍是 10x10 smoke，且截至审计时尚未全部完成；同时 `risky_ps` 相对 `direct_multistage_exp3` 仍没有明显稳健领先，尤其 seed0 partial 中 direct 暂时更低成本。

## 1. exact command

当前有两个 tmux session：

- `psagent_v11b_stage45gpt41_smoke_seed0_3m`
- `psagent_v11b_stage45gpt41_smoke_seed1_3m`

seed0 的实际启动命令：

```bash
tmux new-session -d -s psagent_v11b_stage45gpt41_smoke_seed0_3m \
  cd /home/ubuntu/data/PSAgent && \
  export PYTHONUNBUFFERED=1 \
    PSAGENT_REPEATED_SMOKE_SEED=0 \
    PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini \
    PSAGENT_TELECOM_STAGE45_MODEL=gpt-4.1-mini \
    PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B=1 \
    PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1 \
    PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1 \
    PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1 && \
  python scripts/run_shared_basin_repeated_smoke.py orchestrate \
    --data data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json \
    --schedule-buckets analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json \
    --output-dir tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0 \
    --repeats 10 \
    --family-kind shared_basin_strong_prefix_dedup_profile_switch \
    --schedule-mode trap_switch \
    --switch-denominator 4 \
    --common-eta-override 0.3 \
    --common-epsilon-override 0.01 \
    --executor-name llm_bench \
    --methods risky_ps direct_multistage_exp3 epsilon_exp3 \
    > tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_orchestrate.log 2>&1; \
  echo $? > tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0/orchestrate_exit_code.txt
```

seed1 完全相同，只有：

- `PSAGENT_REPEATED_SMOKE_SEED=1`
- `--output-dir tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed1`
- 日志和 exit code 文件名对应 seed1

orchestrator 会为三个 method 各自 fork 一个 `run-method` 子进程：

```bash
python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir <output-dir> --method risky_ps
python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir <output-dir> --method direct_multistage_exp3
python scripts/run_shared_basin_repeated_smoke.py run-method --run-dir <output-dir> --method epsilon_exp3
```

## 2. git commit hash

当前 HEAD：

```text
7e61b3e7fa9cfe242ea31adb9598c8f9c64235da
```

注意：worktree 是 dirty。相关修改包括 executor、bridge、diagnostic runner、repeated smoke runner、notes，以及多个 `tmp/` checkpoint/partial 输出。这个 hash 只能说明基线提交，不能单独完整复现实验；复现还需要当前未提交改动。

## 3. env vars

本 run 中显式设置：

```text
PYTHONUNBUFFERED=1
PSAGENT_REPEATED_SMOKE_SEED=0 或 1
PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini
PSAGENT_TELECOM_STAGE45_MODEL=gpt-4.1-mini
PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B=1
PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1
PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1
PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1
```

关键含义：

- `PSAGENT_LLM_BENCH_MODEL=gpt-4o-mini` 是 executor 默认模型，也是 stage1-3 的模型。
- `PSAGENT_TELECOM_STAGE45_MODEL=gpt-4.1-mini` 只覆盖 stage4、stage5。
- `PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B=1` 使用 stage4/5 contract prompt v1.1b。
- `PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1` 开启 terminal v4 cost adjustment。
- `PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1` 开启 reasoning weight calibration v3。
- `PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1` 只记录 mode mismatch cost，不把 mismatch cost 加进 raw reasoning cost。

未设置但相关的 env：

- `PSAGENT_TELECOM_MODE_MISMATCH_COST_V2` 未设置，所以 mismatch 不实际惩罚。
- `PSAGENT_ATTRIBUTE_WEAKENING_LEVEL` 未设置时默认为 0，但当前 clean profile-only run 中 attribute guidance disabled，因此不把 capability-fit summary 暴露给 LLM prompt。
- `PSAGENT_BRIDGE_DEBUG_DIR` 未设置时 bridge parse failure dump 到 `/tmp/psagent_bridge_failures`。

## 4. tmux session name

当前两个 session：

```text
psagent_v11b_stage45gpt41_smoke_seed0_3m
psagent_v11b_stage45gpt41_smoke_seed1_3m
```

tmux pane 输出为空是正常的，因为主命令把 stdout/stderr 重定向到 `*_orchestrate.log`；method 子进程输出写到各自 `<method>/runner.log`。

## 5. output dir

seed0：

```text
tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0
```

seed1：

```text
tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed1
```

每个 output dir 内的结构：

- `run_config.json`：固定运行配置。
- `schedule.json`：100 episode 的 materialized schedule。
- `stationary_oracle_summary.json`：stationary oracle。
- `specialist_task_ids.json`：当前 schedule/bucket 中标记为 specialist 的 task ids。
- `<method>/checkpoint.pkl`：method 级 checkpoint，包含 policy 状态和已完成 episodes。
- `<method>/episodes.partial.jsonl`：已完成 episode 的扁平记录。
- `<method>/summary_partial.json`：partial summary。
- `<method>/progress.json`：进度。
- run 完整后才会生成 `<method>/episodes.json`、`summary.json`、`summary_with_oracle.json`、`specialist_summary.json`，以及根目录 compare 文件。

截至本审计时，两个 run 仍在 running：

| seed | method | completed / scheduled | partial raw_total_cost_mean | partial EM | partial raw_terminal_penalty_mean | partial SharedFrac |
|---|---:|---:|---:|---:|---:|---:|
| seed0 | risky_ps | 88 / 100 | 8.003 | 0.830 | 2.784 | 0.898 |
| seed0 | direct_multistage_exp3 | 86 / 100 | 7.554 | 0.884 | 2.302 | 0.942 |
| seed0 | epsilon_exp3 | 88 / 100 | 8.418 | 0.784 | 3.125 | 0.943 |
| seed1 | risky_ps | 87 / 100 | 8.213 | 0.805 | 2.977 | 0.966 |
| seed1 | direct_multistage_exp3 | 90 / 100 | 8.023 | 0.833 | 2.633 | 0.933 |
| seed1 | epsilon_exp3 | 87 / 100 | 8.162 | 0.839 | 2.839 | 0.920 |

这些是 partial，不是 final。

## 6. tree config

family：

```text
shared_basin_strong_prefix_dedup_profile_switch
```

核心设置：

- 5 个 stage：`stage1` 到 `stage5`。
- 使用 prefix-dedup topology，拓扑 spec 来自：
  `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch.json`
- profile preset 定义在 `envs/tree_family/presets.py` 的 `build_shared_basin_strong_prefix_dedup_profile_switch_family_spec()`。
- 生成方式为 `capability_shared_basin_prefix_dedup`。
- seed 为 family seed，当前 repeated smoke runner 构建 env 时使用默认 family seed 0；policy seed 由 `PSAGENT_REPEATED_SMOKE_SEED` 控制。

本地用 seed0 introspection 得到：

- agent 数：655
- root children 数：5
- semantic 计数：
  - `target_specialist`: 260
  - `general_shared`: 165
  - `trap_lane`: 136
  - `private_barrier`: 94
- deliberation mode 计数：
  - `deep`: 513
  - `fast`: 142

stage profile 大意：

- stage1：
  - `general_stage1_intake` fast
  - `general_stage1_verify` deep
  - `target_stage1_handoff` deep
  - `trap_stage1_intake` fast
  - `barrier_stage1_gate` fast/private barrier
- stage2：
  - `target_stage2_router` deep/target
  - `general_stage2_core` fast/shared
  - `trap_stage2_router` fast/trap
  - `general_stage2_roaming` deep/shared
  - `barrier_stage2_gate` deep/private barrier
- stage3：
  - `target_stage3_apn` deep/target
  - `target_stage3_roaming` deep/target
  - `general_stage3_network` deep/shared
  - `trap_stage3_network` fast/trap
  - `barrier_stage3_gate` deep/private barrier
- stage4：
  - `target_stage4_repair` deep/target
  - `general_stage4_repair` deep/shared
  - `general_stage4_verify` deep/shared
  - `trap_stage4_execute` fast/trap
  - `barrier_stage4_gate` deep/private barrier
- stage5：
  - `target_stage5_verify` deep/target
  - `target_stage5_decision` deep/target
  - `general_stage5_verify` deep/shared
  - `trap_stage5_terminal` fast/trap
  - `barrier_stage5_transfer` deep/private barrier

AgentSpec 字段：

- `agent_id`
- `g`
- `base_cost`
- `competence_level`
- `scope_level`
- `stability_level`
- `attribute_skill`
- `deliberation_mode`
- `node_semantic`
- `route_label`

当前 clean profile-only executor 对 LLM 可见的主要 agent 信息不是完整 capability scores，而是：

- competence/scope/stability level，当前 preset 中这些字段值来自 `"capability"` 字符串。
- deliberation mode：`fast` 或 `deep`。
- style guidance / execution contract / round budget。
- route/profile switch labels 不直接作为 oracle label 给 LLM。

## 7. agent profile config

executor：

```text
llm_bench
```

executor internal experiment setting：

```text
telecom_mms_agent_profile_only_clean_v4_hard_transfer_contract
```

含义：

- strict error propagation enabled。
- hard transfer contract enabled。
- attribute guidance disabled：LLM 不拿到 task capability requirements / fit table。
- Stage 4 deep local completion disabled：当前 clean run 不再让 executor 用 oracle-like route metadata 自动补齐 deep/target repair chain。
- Stage 4/5 contract prompt v1.1b enabled。

fast/deep round budget：

| stage | fast base/max | deep base/max |
|---|---:|---:|
| stage1 | 2 | 4 |
| stage2 | 2 | 5 |
| stage3 | 2 | 6 |
| stage4 | 2 | 8 |
| stage5 | 2 | 7 |

max rounds 被 clamp 在 2..8。

## 8. model config by stage

当前 run 是混合模型：

| stage | model |
|---|---|
| stage1 | `gpt-4o-mini` |
| stage2 | `gpt-4o-mini` |
| stage3 | `gpt-4o-mini` |
| stage4 | `gpt-4.1-mini` |
| stage5 | `gpt-4.1-mini` |

证据：

- executor `_model_for_stage()` 如果 `PSAGENT_TELECOM_STAGE45_MODEL` 存在且 stage 是 stage4/stage5，则返回该模型。
- fixed trace summary 也显示 21 条中 stage1-3 全是 `gpt-4o-mini`，stage4-5 全是 `gpt-4.1-mini`。

LLM args：

```python
{"temperature": 0.0}
```

LiteLLM generate 默认 `num_retries=3`，除非调用方覆盖；当前没有覆盖。

## 9. switch schedule

schedule mode：

```text
trap_switch
```

核心逻辑：

- 从 bucket file 读取：
  - `trap_favoring_task_ids`
  - `target_favoring_task_ids`
  - `specialist_task_ids`
- 当前 smoke bucket 中 trap 和 target 都是 10 个 task。
- `cycle_length = len(trap_ids) = 10`
- `total_episodes = repeats * cycle_length = 10 * 10 = 100`
- `switch_episode = total_episodes // switch_denominator = 100 // 4 = 25`
- episode index `<25` 时，只从 trap bucket 循环抽。
- episode index `>=25` 时，只从 target bucket 循环抽。

所以是：**switch 之前在 fast/trap favoring bucket 里抽，switch 之后在 target/deep favoring bucket 里��**。不是从全 dataset 随机抽，也不是混合抽。

实际 schedule composition：

- `trap_pre_switch`: 25 episodes
- `target_post_switch`: 75 episodes
- `trap_favoring`: 25 episodes
- `target_favoring`: 75 episodes

注意 switch 发生在第 3 个 10-task cycle 的中间：

- episode 0-9：trap bucket 第 1 轮
- episode 10-19：trap bucket 第 2 轮
- episode 20-24：trap bucket 第 3 轮前 5 个
- episode 25-99：target bucket 从 target_ids[0] 开始独立循环

## 10. task dataset and sampling

dataset：

```text
data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json
```

bucket file：

```text
analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json
```

run_config 中实际用到 20 个 dataset index：

```text
1, 2, 3, 6, 9, 10, 13, 15, 16, 17, 18, 19, 20, 21, 22, 32, 33, 34, 35, 36
```

bucket file 的 clean dataset coverage：

- clean task count: 100
- expected terminal action:
  - `repair_all`: 94
  - `repair_subset`: 6
  - `transfer`: 0
- clean requirement counts:
  - `deep/fast/deep/deep/deep`: 5
  - `fast/fast/fast/fast/fast`: 19
  - `fast/deep/deep/deep/deep`: 73
  - `deep/deep/deep/deep/deep`: 3

smoke bucket：

- trap bucket size: 10
- target bucket size: 10
- specialist task ids: 4
- schedule 中 `is_specialist_task=True` 的 episode 是 32 条，因为 4 个 specialist task 在 post-switch target 75 条中重复出现。

repeat 实现：

- repeated smoke 不是“每个 task 随机 repeat N 次”。
- 它先构造长度为 100 的 schedule。
- 每个 method 是一个 stateful T=100 sequence。
- policy 状态跨 episode 更新，不在 repeat boundary reset。
- `repeat_index = episode_index // cycle_length` 只是 schedule 注释字段；算法不会因为 repeat_index 变化而重置。

## 11. per-stage prompt templates

当前 repeated smoke 的 `episodes.partial.jsonl` **没有保存完整 system/user prompt 原文**。executor 在每个 stage 构造 prompt 并发送给 bridge，但扁平记录只保留：

- `prompt_summary`
- LLM assistant raw messages / tool calls 在 full stage trace 中存在，但 repeated smoke flatten 后没有完整 stage_trace。
- 当前 repeated smoke flat record 只保留每 stage token/cost/latency 数组。

所以，下面是根据代码模板还原的 prompt 内容结构；不是从 current partial record 中逐字取出的 prompt dump。若要逐字保存，需要给 tau2 `llm_utils.set_llm_log_dir()` 接入日志目录，或修改 executor 把 bridge payload 的 `system_prompt` / `user_prompt` 写入 trace。

通用 system header：

- “You are the Stage N execution agent, not the user.”
- 工具 requestor=user 只表示模拟用户设备动作，不代表 assistant 成为用户。
- 当前 agent profile 是 FAST/DEEP。
- FAST/DEEP mode rules。
- 本 stage 的 goal sentence。
- “You must follow the agent execution contract exactly.”
- search_policy、stop_policy、profile_policy、stage_specific_hard_constraints 都是 binding。
- No attribute routing rules are active in this run。

stage1 system：

- Stage 1: user grounding。
- 只做用户、phone、line grounding。
- 可以用 allowed tools 做最小身份/line grounding。
- 不做 diagnosis，不 infer blockers，不做 terminal decision。
- 输出 JSON keys：
  - `domain`
  - `problem_family`
  - `customer_lookup`
  - `line_selector`
  - `symptom_report`
  - `context_flags`
  - `conversation_risk_flags`

stage1 user JSON 包含：

- `task_id`
- `agent_profile`
- `agent_deliberation_profile`
- `agent_execution_contract`
- `stage_goal`
- `policy_mode=grounding_only_minimal_lookup`
- `user_context`
- `task_metadata`
- `output_contract`
- `normalization_rules`

stage2 system：

- Stage 2: customer and line resolution。
- 只 resolve customer/line 和 minimal account snapshot。
- 不做 diagnosis，不谈 blockers。
- 输出 keys：
  - `candidate_customers`
  - `resolved_customer_id`
  - `candidate_line_ids`
  - `resolved_line_id`
  - `target_phone_number`
  - `assistant_account_snapshot`
  - `resolution_status`

stage2 user JSON 包含：

- `stage1_output`
- `user_context`
- `task_metadata`
- agent profile / execution contract。

stage3 system：

- Stage 3: observed-state extraction。
- 只收集 factual observed state，不做 terminal action。
- MMS diagnosis 中 service/SIM/permission/APN/network-mode/Wi-Fi-calling 是高收益起点。
- fast agents 只 compactly validate top 1-2 blocker families。
- deep agents 必须 cross-check decisive evidence。
- 输出 `observed_state`，包含固定 keys：
  - `can_send_mms`
  - `service_status`
  - `mobile_data_working`
  - `internet_speed_desc`
  - `is_abroad`
  - `roaming_enabled_on_device`
  - `roaming_enabled_on_account`
  - `airplane_mode`
  - `sim_status`
  - `network_mode_preference`
  - `wifi_calling_enabled`
  - `apn_mms_ok`
  - `messaging_sms_permission`
  - `messaging_storage_permission`
  - `data_usage_exceeded`

stage3 user JSON 包含：

- `stage1_output`
- `stage2_output`
- `tool_use_checklist`
- `stage3_blocker_decision_rules`
- `user_context`
- `task_metadata`

stage4 system：

- Stage 4: blocker adjudication and repair execution。
- 先逐 blocker 判断 `should_repair` / defer / transfer。
- 对 `should_repair=true` 执行 canonical repair steps。
- 工具调用后仍必须返回 final JSON；只有 tool calls 被视为 incomplete Stage 4 decision。
- 输出 keys：
  - `per_blocker`
  - `repairability`
  - `transfer_reason`
  - `decision_policy_version`
  - `contract_self_check`
- repairability values:
  - `repairable`
  - `partially_repairable`
  - `transfer_required`
- hard transfer contract：
  - active hard hybrid/nonlocal blocker unresolved 时 case-level repairability 必须是 `transfer_required`。
  - 不能因为一些 local repairs 成功就把 hard transfer case 降到 `partially_repairable`。
- v1.1b extra rules：
  - 若 `repairability=transfer_required`，`contract_self_check.has_concrete_transfer_blocker` 必须 true，且 `transfer_reason` 必须指向 concrete hard input blocker id。
  - 如果没有 concrete hard blocker id，不要用 `transfer_required`，而选 `repairable` 或 `partially_repairable`。

stage4 user JSON 包含：

- `stage2_context`
- `stage3_output`
- `blocker_specs`
- `repair_metadata`
- `stage4_repair_precondition_rules`
- `stage4_local_repair_decision_table`
- `stage4_hard_transfer_contract`
- `output_contract`
- `normalization_rules`
- `stage4_contract_prompt_v1`

stage5 system：

- Stage 5: post-repair verification and terminal decision。
- replay 已在 verification 前应用。
- 不执行 repair tools。
- 必须先 verify，再返回 JSON。
- final_action values：
  - `repair_all`
  - `repair_subset`
  - `transfer`
- hard terminal rules：
  - 若 stage4 repairability 是 `transfer_required`，final_action 必须是 `transfer`。
  - 不要因为 local tools 成功就 downgrade hard transfer case。
  - `repair_subset` 只在 deferred blockers 是 ordinary defers 时有效。
- 输出 keys：
  - `final_action`
  - `selected_blocker_ids`
  - `deferred_blocker_ids`
  - `response_mode`
  - `verification_plan`
  - `transfer_reason`
  - `cancelled_reservation_ids`
  - `refused_reservation_ids`
  - `contract_self_check`

stage5 user JSON 包含：

- `stage2_context`
- `stage4_output`
- `verification_checklist`
- `stage5_terminal_decision_rules`
- `stage5_hard_transfer_contract`
- `output_contract`
- `normalization_rules`
- `stage5_contract_prompt_v1`

## 12. per-stage output schema

stage1 normalized output：

```json
{
  "domain": "telecom",
  "problem_family": "mms_issue",
  "customer_lookup": {
    "full_name": "...",
    "phone_number": "...",
    "lookup_confidence": "high|medium|low"
  },
  "line_selector": {
    "type": "phone_number",
    "value": "..."
  },
  "symptom_report": {
    "cannot_send_mms": true,
    "wants_resolution": true,
    "target_success_signal": "..."
  },
  "context_flags": {
    "is_abroad_claimed": false,
    "refuel_allowed": true,
    "max_refuel_gb": 2.0,
    "plan_change_allowed": false
  },
  "conversation_risk_flags": []
}
```

stage2 normalized output：

```json
{
  "candidate_customers": [],
  "resolved_customer_id": "...",
  "candidate_line_ids": [],
  "resolved_line_id": "...",
  "target_phone_number": "...",
  "assistant_account_snapshot": {
    "line_status": "...",
    "roaming_enabled_on_account": true,
    "plan_id": "...",
    "data_used_gb": 0,
    "data_limit_gb": 0
  },
  "resolution_status": "resolved|unresolved"
}
```

stage3 normalized output：

```json
{
  "observed_state": {
    "can_send_mms": false,
    "service_status": "...",
    "mobile_data_working": false,
    "internet_speed_desc": "...",
    "is_abroad": false,
    "roaming_enabled_on_device": null,
    "roaming_enabled_on_account": true,
    "airplane_mode": false,
    "sim_status": "...",
    "network_mode_preference": "...",
    "wifi_calling_enabled": true,
    "apn_mms_ok": false,
    "messaging_sms_permission": true,
    "messaging_storage_permission": false,
    "data_usage_exceeded": false
  },
  "per_blocker": [],
  "per_blocker_mode": "inferred_from_observed_state_v2",
  "raw_task_blocker_ids": [],
  "inferred_blocker_ids": []
}
```

stage4 normalized output：

```json
{
  "per_blocker": [
    {
      "blocker_id": "...",
      "should_repair": true,
      "repair_order": 1,
      "canonical_repair_steps": [],
      "execution_attempted": true,
      "execution_succeeded": true,
      "executed_step_count": 1
    }
  ],
  "repairability": "repairable|partially_repairable|transfer_required",
  "transfer_reason": null,
  "decision_policy_version": "first_pass_v1",
  "executed_repair_steps": [],
  "failed_repair_steps": [],
  "skipped_repair_steps": [],
  "executed_blocker_ids": [],
  "deferred_blocker_ids": [],
  "post_execution_status": {},
  "stage4_raw_json_extracted": {},
  "stage4_selected_before_normalization": [],
  "stage4_selected_after_normalization": [],
  "stage4_normalizer_changed_output": false,
  "hard_transfer_guard_applied": false,
  "stage4_completion_pass_applied": false,
  "stage4_decision_valid": true
}
```

stage5 normalized output：

```json
{
  "final_action": "repair_all|repair_subset|transfer",
  "selected_blocker_ids": [],
  "deferred_blocker_ids": [],
  "stage5_contract_prompt_version": "stage45_contract_prompt_v1_1b",
  "contract_self_check": {},
  "response_mode": "telecom_structured_execution",
  "verification_plan": {},
  "transfer_reason": null,
  "cancelled_reservation_ids": [],
  "refused_reservation_ids": [],
  "verification_observed_state": {},
  "verification_evidence": [],
  "verification_summary": {},
  "post_repair_can_send_mms": true,
  "post_repair_blocker_ids": []
}
```

## 13. cost formula

核心 raw formula：

```text
raw_total_cost = raw_terminal_penalty + raw_path_cost_component + raw_reasoning_cost_component
```

path cost：

```text
raw_path_cost_component = path_agent_cost_weight * sum(selected agents' base_cost)
```

LLM bench reasoning cost：

- 聚合每个 stage 的 prompt tokens、completion tokens、api cost。
- 每个 stage 先算 base reasoning components：
  - API view：由 `api_cost_total_usd_stage` 乘 alpha。
  - token view：由 input/output tokens 乘 alpha。
- 再乘 reasoning match multiplier。
- 默认 reasoning cost mode 当前记录为 `api`。
- `PSAGENT_TELECOM_REASONING_WEIGHT_CALIBRATION_V3=1` 使 calibration 标记进入记录。

mode mismatch：

- `PSAGENT_TELECOM_MODE_MISMATCH_REPORT_ONLY_V2=1` 时只报告：
  - `raw_mode_mismatch_cost_component`
  - `mode_mismatch_fast_on_deep_cost`
  - `mode_mismatch_deep_on_fast_cost`
- 但不加入 `raw_reasoning_cost_component`。
- 如果设置 `PSAGENT_TELECOM_MODE_MISMATCH_COST_V2=1` 才会实际加到 reasoning cost。

terminal v4：

当 `PSAGENT_TELECOM_EXEC_CLEAN_TERMINAL_V4=1` 且 family 是 telecom MMS recovery：

- local repair task 出现 subset mismatch 会加 base penalty。
- local clear failure 有 floor 10。
- clear + auxiliary failure 有 floor 12。
- mostly fast path on mostly deep required 且 clear failure 有 floor 14。
- local repair task 却 predicted transfer 有 floor 18。
- clear success 但 auxiliary failure 有 floor 6。

normalized total：

```text
normalized_total_cost = min(raw_total_cost / total_cost_upper_bound, 1.0)
```

当前 repeated smoke compare 排名通常看 `raw_total_cost_mean` 更直观；`total_cost` 是 normalized cost。

额外 diagnostic formula：

```text
raw_total_cost_with_token_penalty =
  raw_total_cost + fast_token_over_budget_penalty_total
```

这个字段在 path diagnostic 中有；当前 repeated smoke flat record 没有直接保存该字段，但每 stage 的 token 信息可用于重算类似 penalty。

## 14. logged fields

当前 repeated smoke 每条 episode flat record 包含的关键字段：

- schedule/task：
  - `episode_index`
  - `repeat_index`
  - `position_in_cycle`
  - `dataset_index`
  - `instance_id`
  - `original_task_id`
  - `schedule_phase`
  - `task_bucket`
  - `is_specialist_task`
- path/tree：
  - `selected_path`
  - `leaf_type`
  - `selected_shared_path`
  - `selected_unshared_path`
  - `family_route_labels`
  - `family_deliberation_modes`
  - `family_node_semantics`
  - `first_private_barrier_stage`
  - `candidate_count_per_stage`
  - `legal_child_count_per_stage`
- terminal/cost：
  - `oracle_action`
  - `final_action`
  - `exact_match`
  - `subset_mismatch`
  - `terminal_penalty`
  - `raw_terminal_penalty`
  - `raw_terminal_penalty_exec_clean_v4`
  - `raw_outcome_penalty`
  - `raw_policy_penalty`
  - `raw_path_cost_component`
  - `raw_reasoning_cost_component`
  - `raw_total_cost`
  - `total_cost`
  - `terminal_adjustment_reasons`
- LLM/resource：
  - `llm_call_count`
  - `prompt_tokens_total`
  - `completion_tokens_total`
  - `total_tokens_total`
  - `api_cost_total_usd_raw`
  - `generation_time_total_seconds`
  - `llm_round_trip_total_seconds`
  - `tool_wall_clock_total_seconds`
  - `episode_wall_clock_seconds`
  - `stage_prompt_tokens`
  - `stage_completion_tokens`
  - `stage_total_tokens`
  - `stage_api_cost_usd`
  - `stage_generation_time_seconds`
  - `stage_llm_round_trip_seconds`
  - `stage_tool_wall_clock_seconds`
  - `stage_wall_clock_seconds`
- tools:
  - `tool_calls_made`
  - `mutating_tool_calls_made`
  - `assistant_side_mutating_tool_calls_made`
  - `stage5_replay_tool_names`
  - `stage5_executed_tool_names`
- policy selection/update：
  - `selection_info`
  - `update_info`
  - `selection_path_prob`
  - `selection_stage_probs`
  - `root_child_id`
  - `root_selection_mode`
  - `shared_branch_triggered`
  - `unshared_branch_triggered`
  - `shared_update_count`
  - `unshared_edge_update_count`
  - `cumulative_shared_path_ratio`
  - `cumulative_unshared_path_ratio`
  - `rolling_shared_path_ratio_last10`
  - `cumulative_shared_update_count`
  - `cumulative_unshared_edge_update_count`

用户列出的 dashboard 指标是否支持：

| 指标 | 当前 repeated smoke 是否可得 | 说明 |
|---|---|---|
| Avg Total Cost ↓ | 可得 | `raw_total_cost_mean` 或 `total_cost_mean` |
| Success / EM ↑ | 可得 | `exact_match_mean` |
| Terminal Penalty ↓ | 可得 | `raw_terminal_penalty_mean` |
| Path Cost ↓ | 可得 | `raw_path_cost_component_mean` |
| LLM Calls ↓ | 可得 | `mean_llm_call_count` |
| Total Tokens ↓ | 可得 | `mean_total_tokens` / `cumulative_total_tokens` |
| Est. LLM Cost ↓ | 可得 | `mean_api_cost_usd_raw` |
| Avg Latency ↓ | 可得 | `mean_episode_wall_clock_seconds` 或 generation/roundtrip |
| Input Tokens ↓ | 可得 | `mean_prompt_tokens` |
| Output Tokens ↓ | 可得 | `mean_completion_tokens` |
| P95 Latency ↓ | 可算 | episode records 有 `episode_wall_clock_seconds`，summary 当前只有 p50/p90，需要额外算 p95 |
| Tokens / Success ↓ | 可算 | `sum(total_tokens_total) / sum(exact_match)` |
| Cost / Success ↓ | 可算 | `sum(raw_total_cost) / sum(exact_match)` |
| SharedFrac | 可得 | `shared_path_fraction` |
| UnsharedFrac | 可得 | `unshared_path_fraction` |
| SharedUpdFrac | 可算 | `mean(shared_branch_triggered)` 或 `mean(shared_update_count > 0)` |
| SharedUpdCnt | 可得 | `shared_update_count` / cumulative sum |
| CumSharedUpd | 可得 | `cumulative_shared_update_count` |

当前 repeated smoke **没有**直接保留完整 `stage_trace`、stage1-5 full prompt、Stage 4 raw JSON、Stage 5 raw JSON。要分析这些，要跑或复用 fixed trace diagnostic。

## 15. one matched trace

来自当前 seed1 / `risky_ps` partial 的 episode 0：

```text
episode_index: 0
dataset_index: 1
task: [mms_issue]airplane_mode_on|bad_wifi_calling|break_app_storage_permission[PERSONA:Easy]
schedule_phase: trap_pre_switch
task_bucket: trap_favoring
oracle_action: repair_all
final_action: repair_all
exact_match: true
raw_terminal_penalty: 0
raw_total_cost: 4.42754
llm_call_count: 10
```

selected path：

```text
stage1_n5__from__root__c05
stage2_n3__from__n0005__c01
stage3_n4__from__n0020__c02
stage4_n5__from__n0066__c04
stage5_n2__from__n0191__c01
```

route/mode：

```text
route_labels:
barrier_stage1_gate -> trap_stage2_router -> trap_stage3_network -> barrier_stage4_gate -> target_stage5_decision

deliberation_modes:
fast -> fast -> fast -> deep -> deep
```

per-stage resource：

| stage | prompt tokens | total tokens | api cost usd | wall seconds |
|---|---:|---:|---:|---:|
| stage1 | 3502 | 3544 | 0.0004353 | 5.12 |
| stage2 | 3358 | 3452 | 0.0004449 | 4.97 |
| stage3 | 5222 | 5346 | 0.0007809 | 5.98 |
| stage4 | 15029 | 15195 | 0.0062772 | 7.12 |
| stage5 | 10079 | 10514 | 0.0032684 | 19.77 |

raw trace 过程解释：

1. Stage 1 用 fast profile 做最小 grounding。输出客户/手机号/line selector，不诊断 blocker。
2. Stage 2 用 fast profile resolve customer/line/account snapshot。输出 resolved line、plan、roaming/data account snapshot。
3. Stage 3 用 fast profile 进行 compact observed-state extraction。根据工具 evidence 推导 `per_blocker`，但 fast 可能只覆盖最高收益 blocker families。
4. Stage 4 用 deep profile + `gpt-4.1-mini`，在 hard transfer contract 下 adjudicate blocker，并执行 canonical local repair steps。当前 repeated flat record 不保留 Stage 4 raw JSON；如果要逐字看 raw output，需要 fixed trace diagnostic。
5. Stage 5 用 deep profile + `gpt-4.1-mini`，replay Stage 4 repair tool calls，然后运行 verification tools，输出 terminal decision。该 episode 最终 `repair_all` 且 exact match。

补充：可复用的 fixed trace matched raw 例子来自：

```text
tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_fdddd_exact_r1_seed1_cconfig/records.json
```

其中 dataset 13 / pattern `fdddd`：

```text
oracle_action: repair_subset
final_action: repair_subset
exact_match: true
raw_terminal_penalty: 6.0
raw_total_cost: 11.4333
stage4_normalizer_changed_output: false
stage4_completion_pass_applied: false
```

Stage 4 raw JSON 摘要：

```json
{
  "per_blocker": [
    {"blocker_id": "airplane_mode_on", "should_repair": true},
    {"blocker_id": "unseat_sim_card", "should_repair": true},
    {"blocker_id": "data_mode_off", "should_repair": true},
    {"blocker_id": "user_abroad_roaming_enabled_off", "should_repair": true},
    {"blocker_id": "data_usage_exceeded", "should_repair": false},
    {"blocker_id": "bad_network_preference", "should_repair": true},
    {"blocker_id": "bad_wifi_calling", "should_repair": true},
    {"blocker_id": "break_apn_mms_setting", "should_repair": true},
    {"blocker_id": "break_app_both_permissions", "should_repair": true}
  ],
  "repairability": "partially_repairable",
  "transfer_reason": null,
  "decision_policy_version": "stage45_contract_prompt_v1_1b",
  "contract_self_check": {
    "has_concrete_transfer_blocker": false,
    "ids_are_input_blockers_only": true
  }
}
```

Stage 4 tool calls included local repairs such as `toggle_airplane_mode`, `reseat_sim_card`, `toggle_data`, `toggle_roaming`, `set_network_mode_preference`, `toggle_wifi_calling`, `reset_apn_settings`, `reboot_device`, and app permission grants. Stage 5 replayed those repairs and verified with network/SIM/mode/APN/Wi-Fi/permissions/MMS checks.

## 16. one mismatched trace

来自当前 seed1 / `risky_ps` partial 的 worst mismatch：

```text
episode_index: 28
dataset_index: 13
task: [mms_issue]airplane_mode_on|bad_network_preference|bad_wifi_calling|break_apn_mms_setting|break_app_both_permissions|data_mode_off|data_usage_exceeded|unseat_sim_card|user_abroad_roaming_enabled_off[PERSONA:Hard]
schedule_phase: target_post_switch
task_bucket: target_favoring
is_specialist_task: true
oracle_action: repair_subset
final_action: repair_subset
exact_match: false
raw_terminal_penalty: 17.0
```

selected path：

```text
stage1_n3__from__root__c03
stage2_n2__from__n0003__c02
stage3_n5__from__n0013__c04
stage4_n4__from__n0046__c03
stage5_n1__from__n0131__c01
```

route/mode：

```text
target_stage1_handoff -> general_stage2_core -> barrier_stage3_gate -> trap_stage4_execute -> target_stage5_verify
deep -> fast -> deep -> fast -> deep
```

解释：

- 这是 post-switch target bucket 的 hard local-repair-heavy task。
- oracle 和 final action 都是 `repair_subset`，但 exact mismatch 说明 selected/deferred blocker set 或 policy/evaluation details 不一致。
- 路径里 stage4 是 `trap_stage4_execute` fast，这对多 blocker repair chain 是高风险组合。
- raw_terminal_penalty 17 表明 terminal v4 adjustment 认为这是严重 local execution mismatch，而不是单纯 reasoning cost 高。
- 当前 repeated flat record 没保留 stage4 raw per_blocker，因此不能直接在这个 episode 上看“模型 raw Stage4 选了哪些 blocker”。要对这个 mismatched episode 做完整 raw 复盘，应使用 fixed exact-path diagnostic 重跑同一个 `dataset_index=13` 和同一 `selected_path`，并在 record 中保存 `stage4_raw_json_extracted`、Stage 4/5 raw messages。

fixed trace 中可参考的 mismatched raw 例子：

```text
tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_fixedtrace_clean100_abnormal21_ddddd_exact_r1_seed1_cconfig/records.json
dataset_index: 8
pattern: ddddd
oracle_action: repair_subset
final_action: repair_all
exact_match: false
raw_terminal_penalty: 12.0
stage4_raw_json_extracted: present
stage4_normalizer_changed_output: false
```

这类 mismatch 更接近“模型/Stage5 terminal decision 本身过度升级为 repair_all”，不是 normalizer 强行改写。

## 17. known risks before full run

1. 当前 run 尚未结束，partial 不能替代 final。

2. seed0 partial 下 `direct_multistage_exp3` 暂时优于 `risky_ps`：
   - seed0 direct raw total 7.554 vs risky 8.003。
   - seed1 direct 8.023 vs risky 8.213。
   这说明“PS first”仍未显著稳健。

3. stage4/5 换 `gpt-4.1-mini` 有改善信号，但当前不是 formal run，只是 10x10 smoke。

4. repeated smoke flat records 不保存 full prompt / full raw stage trace。若 full run 后才发现异常，追 raw prompt 会需要重跑 trace diagnostic。

5. current smoke 中 `specialist` 是 legacy analysis label，不是独立 runtime agent。不要把 specialist summary 解读成“另一个 agent 参与了执行”。

6. target_post_switch 的 specialist task 重复 32 次，会显著影响 post-switch metric。它是刻意压力测试 target/deep heavy task，但解读时要分 bucket 和 task type。

7. API error 如果耗尽 LiteLLM retry 会导致当前 episode 不落 record、method 进程失败。当前 runner 不写 per-episode error record；只靠 checkpoint 恢复继续。

8. stage4/5 contract prompt v1.1b 会减少无 concrete blocker 的 transfer，但也可能使 repair_subset/repair_all 边界更依赖 Stage 4 blocker selection。

9. normalizer 会在安全/contract 层改写模型输出。它能保护 hard transfer，但也可能掩盖模型 raw 意图；需要用 fixed trace diagnostic 监控 `stage4_normalizer_changed_output`。

10. mode mismatch 当前 report-only，不进入 objective。若 full run 改成 actual cost，结论会变。

11. `raw_total_cost_with_token_penalty` 当前 repeated smoke 没直接输出，不能作为主 objective 直接比较，除非补导出或后处理重算。

12. worktree dirty。正式 run 前建议 commit 或至少保存 diff，否则配置 provenance 不够干净。

## 18. recommendation: run full / do not run full

建议：**当前不要直接启动 full formal LLM run**。

可以先让这两个 tmux smoke 跑完，然后做三个确认：

1. 比较 seed0/seed1 final `repeated_smoke_compare.json`，看 stage4/5 `gpt-4.1-mini` 是否让 `risky_ps` 稳定接近或超过 direct。
2. 对 post-switch high terminal episodes，尤其 dataset 13 / 34 / 36 这类 repeated high penalty task，跑 fixed exact-path trace diagnostic，确认 Stage 4 raw blocker selection 是否改善。
3. 如要 full run，先决定是否把 prompt raw logging 打开，否则 full run 后定位 prompt/normalizer 问题会麻烦。

更稳妥的下一步：

- 等 seed0/seed1 当前 smoke 完成。
- 用 final summaries 生成一个 compare note。
- 如果 `risky_ps` 仍低于 direct，则不要 full run；优先诊断 PS update variance 或 stage4 fast-on-target mismatch。
- 如果 `risky_ps` 两 seed 都领先或接近领先，再跑一个小规模 seed2 confirmatory smoke，而不是直接 full formal。

## normalizer 是否改变了模型原意，它改变了什么

短答：会改变，但主要是 contract/safety normalization，而不是任意重写。

Stage 1 normalizer：

- 从 LLM JSON、工具结果、user context 合成 canonical grounding。
- 可能用工具结果覆盖或补齐 phone/full_name/lookup confidence。
- 会固定 domain/problem family。
- 这会改变模型的自由表达，但目的是把输出规整成下游可消费结构。

Stage 2 normalizer：

- 用 `get_customer_by_phone` / `get_details_by_id` 工具结果解析 customer、line、plan。
- LLM 如果给了 resolved_line_id，但工具结果有匹配 phone 的 line，会优先工具 evidence。
- 会补齐 account snapshot。

Stage 3 normalizer：

- tool-first 合并 observed_state。
- strict mode 下，没有调用某类工具时，会把对应 observed field 置为 null，而不是信任 LLM 猜测。
- 根据 observed_state 推导 `per_blocker`，不是直接信任 LLM 的 blocker list。
- 这会明显限制模型“猜 blocker”的自由度。

Stage 4 normalizer：

- 只接受输入 blocker ids，不允许 invent blockers。
- 每个 blocker 必须出现一次。
- 根据 LLM `should_repair` 生成 selected/deferred。
- 对 hard hybrid/nonlocal blocker，若模型试图 repair，会被 guard 改回 defer/transfer。
- 若 active hard transfer blocker 存在，会把 case-level `repairability` 强制为 `transfer_required`。
- 若 selected 是 shallow subset 但 deferred 有 hard nonlocal blocker，会转成 `transfer_required`。
- 如果 JSON invalid，在 strict mode 下只承认模型实际执行过的 repair tool calls；如果没有有效 repair subset，则 transfer/invalid。
- 当前 clean run 中 deep local completion disabled，所以不会再额外替 deep/target 自动加 downstream blockers。

Stage 5 normalizer：

- final_action 只能是 `repair_all` / `repair_subset` / `transfer`。
- 若 Stage 4 是 `transfer_required`，Stage 5 必须 transfer。
- 若 Stage 4 是 partially repairable 且模型无 hard transfer reason 却想 transfer，会被拉回 Stage 4 subset semantics。
- 若模型 selected/deferred 翻转 Stage 4 subset，会默认回 Stage 4 selected subset。
- 如果 shallow subset verification floor 不满足，可能 forced transfer，reason 为 `shallow_subset_verification_floor_not_met_v1`。

因此 normalizer 改变的是：

- 非 schema 输出。
- 不可信或缺 evidence 的 guessed fields。
- unsafe hard-transfer/local-repair 决策。
- selected/deferred blocker partition。
- invalid JSON / missing final JSON 的后果。

它不应该改变的是：

- valid、evidence-supported、contract-consistent 的 Stage 4/5 raw decision。

现有 fixed trace 中一个 matched 例子 `dataset_index=13 fdddd` 的 `stage4_normalizer_changed_output=false`，说明该例模型 raw Stage4 与 normalized Stage4 一致。另一个 `fffff` 例子出现 `stage4_raw_json_extracted=false` 且 `stage4_normalizer_changed_output=true`，属于 invalid/missing Stage4 JSON 后 strict fallback，不是普通语义微调。

## API error / retry / partial / recovery 审计

### 1. 单条 episode API error 是否自动 retry

单次 LLM call 使用 tau2 的 `llm_utils.generate()`，其 LiteLLM 调用默认设置：

```text
num_retries = DEFAULT_MAX_RETRIES = 3
```

所以 provider/API 层的失败会先由 LiteLLM 重试。这里的 retry 是每次 LLM generate call 级别，不是整个 episode 级别。

此外还有 JSON repair prompt 级别的“retry”：

- 如果 assistant 不返回工具调用，也无法解析 JSON，bridge 会追加一条 user message：
  `Return only a valid JSON object matching the required schema. Do not include prose.`
- 继续下一 round，直到 `max_rounds` 用完。
- `json_retry_count` 记录的是这种 JSON retry 轮数，不是 API retry 次数。

### 2. retry 次数是多少

API call retry：

```text
DEFAULT_MAX_RETRIES = 3
```

JSON/schema retry：

- 受每个 stage 的 `max_rounds` 限制。
- fast stage 通常 max 2。
- deep stage stage4 最多 8，stage5 最多 7。
- `json_retry_count = json_attempt_count - 1`，仅统计 assistant content JSON 解析失败后追加 JSON-only prompt 的次数。

### 3. retry 后是否写入 error record

如果 API retry 最终成功：

- 不会写单独 error record。
- 成功 call 的 usage/cost/latency 会进入 `llm_raw_output` 和 resource summary。
- API retry 发生过几次通常不在 episode record 中显式记录。

如果 API retry 最终失败：

- bridge 子进程非零或抛异常。
- executor 抛 `RuntimeError`。
- 当前 episode 不 append 到 `episodes.partial.jsonl`。
- method 子进程失败。
- orchestrator 最后写 `orchestrator_failures.json` 并以失败退出。
- 不会把失败 episode 写成一条完整 result row。

如果是 bridge stdout 被污染但最后一行是 JSON：

- executor 会 recover，返回 payload，并标记：
  - `_bridge_stdout_recovered=true`
  - `_bridge_stdout_extra_line_count`
  - `_bridge_stdout_recovery_mode=last_json_line`
- 但 repeated smoke flatten 当前未显式保留这些 bridge recovery flags。

### 4. partial records 是否会被误当成完整结果

不会在正式 merge 中被误当成完整。

机制：

- 每个 method 持续写 `episodes.partial.jsonl`、`summary_partial.json`、`progress.json`。
- 完成 100/100 时才写：
  - `episodes.json`
  - `summary.json`
  - `summary_with_oracle.json`
- `merge_method_results()` 强制检查：
  - `progress["completed_episodes"] == total_episodes`
  - episode indices 必须等于 `0..total_episodes-1`
- 不满足会 raise，不会生成 final compare。

当前 partial rows 的 episode_index 是连续的；例如截至审计时 seed1 direct 是 0..89。

### 5. recovery run 是否会重复统计同一个 dataset/seed/path

正常不会。

机制：

- `checkpoint.pkl` 保存：
  - `completed_count`
  - `episodes`
  - `policy`
  - `model`
- recovery/restart 时 load checkpoint，然后：

```text
for local_offset in range(completed_count, total_episodes):
    ...
```

- 已完成的 episode 不会重跑，不会重复 append。
- `episodes.partial.jsonl` 是从 checkpoint 中的 `episodes` 全量原子重写，不是盲目 append 历史 partial。
- run dir 兼容性检查会比较 dataset、repeats、methods、family_kind、schedule、eta/epsilon、executor 等，配置不匹配会拒绝复用目录。

潜在风险：

- 如果手工删除或篡改 `checkpoint.pkl`，但保留 partial jsonl，runner 以 checkpoint 为准，可能从 0 重跑并覆盖 partial。
- 如果用不同代码版本恢复同一个 checkpoint，policy object pickle 可能语义不完全一致。
- 如果同一个 run dir 同时启动两个同 method 的 `run-method`，可能竞争写 checkpoint；当前 orchestrator 只为每个 method 启动一个进程。

## specialist 现在还有吗？怎么影响结果？

有，但它现在主要是 legacy diagnostic label，不是独立 runtime specialist agent。

当前 run 中的 specialist 出现位置：

- bucket file 有 `specialist_task_ids`。
- `schedule.json` 每条 episode 有 `is_specialist_task`。
- flat record 有 `is_specialist_task`。
- final merge 会写 `specialist_summary.json` 和 `specialist_unshared_hit_analysis.json`。

它不会：

- 创建单独的 runtime specialist process。
- 让某个外部 specialist agent 接管。
- 直接改变 executor prompt 或 policy update。

它会：

- 影响分析切片：post-switch target-heavy/specialist task 的 shared/unshared fraction、exact match、terminal penalty 会单独统计。
- 因为 specialist task 在 target bucket 中占 4/10，post-switch 75 条中约 32 条是 specialist 标记，所以对 post-switch metrics 有较大权重。
- tree 中确实存在 `target_specialist` semantic nodes，但这是 path/agent profile 的一类 node，不等同于 schedule 的 `is_specialist_task` 标签。

解读时建议：

- 使用 `is_specialist_task` 作为 “target/deep/share-heavy hard slice” 的诊断标签。
- 不要说“specialist agent 影响了结果”；更准确是“target-specialist task slice 重复较多，且 policy 选择了某些 target_specialist / shared / trap nodes，影响 terminal quality 和 shared fraction”。

## 当前是否可以从记录中分析出用户列出的所有指标

可以分析绝大部分，且 repeated smoke flat records 已足够支持 dashboard：

```text
Avg Total Cost       -> raw_total_cost_mean 或 total_cost_mean
Success / EM         -> exact_match_mean
Terminal Penalty     -> raw_terminal_penalty_mean
Path Cost            -> raw_path_cost_component_mean
LLM Calls            -> mean_llm_call_count
Total Tokens         -> mean_total_tokens
Est. LLM Cost        -> mean_api_cost_usd_raw
Avg Latency          -> mean_episode_wall_clock_seconds
Input Tokens         -> mean_prompt_tokens
Output Tokens        -> mean_completion_tokens
P95 Latency          -> percentile(episode_wall_clock_seconds, .95)，需后处理
Tokens / Success     -> sum(total_tokens_total) / sum(exact_match)
Cost / Success       -> sum(raw_total_cost) / sum(exact_match)
SharedFrac           -> shared_path_fraction
UnsharedFrac         -> unshared_path_fraction
SharedUpdFrac        -> mean(shared_update_count > 0) 或 mean(shared_branch_triggered)
SharedUpdCnt         -> sum/mean(shared_update_count)
CumSharedUpd         -> final cumulative_shared_update_count
```

不足：

- `raw_total_cost_with_token_penalty` 当前 repeated smoke 未直接导出。
- full prompt 原文当前 repeated smoke 未保存。
- Stage 4 raw JSON / normalizer before-after 当前 repeated smoke 未保存；fixed trace diagnostic 保存。

## 建议的正式 run 前 checklist

1. 等当前两个 tmux run 结束，确认 exit code 都是 0。
2. 看两个 output dir 根目录是否生成：
   - `repeated_smoke_compare.json`
   - `repeated_smoke_compare.csv`
   - `repeated_smoke_compare.md`
3. 分 seed 比较 `raw_total_cost_mean`、`raw_terminal_penalty_mean`、post-switch target metrics。
4. 对 high terminal mismatches 跑 fixed exact-path trace，至少覆盖：
   - seed1 risky_ps episode 28 dataset 13 selected path
   - seed0 risky_ps episode 42 dataset 34 selected path
   - seed0 epsilon episode 34 dataset 36 selected path
5. 正式 full run 前 commit 或 stash 当前代码 diff，避免只剩 dirty worktree provenance。
6. 如要保留 full prompt 证据，先加 prompt payload logging，不要等 full run 后再补。
