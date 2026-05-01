# llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1 报告

## 结论

- 两个 seed 均完整跑完，两个 `orchestrate_exit_code.txt` 都是 `0`，所有 6 个 method run 都是 `100/100 complete`。
- `risky_ps` 没有成为第一：aggregate raw_total_cost 第一是 `direct_multistage_exp3`，并且 post-switch raw_total_cost 也是 `direct_multistage_exp3` 最低。
- agent/profile mismatch 证据在这个 smoke 中是“方向支持但不纯”：按 exact pattern match 聚合，matched 的 raw_total_cost 低于 mismatched；但不同算法采样分布不同，不能只用 smoke 表证明因果，最好和 fixed-path mismatch probe 一起引用。
- 主要问题仍是 terminal quality 而不是 reasoning cost：`risky_ps` 的 reasoning component 不高，但 terminal penalty 和 exact_match 明显差于 direct。

## 配置

- 实验名：`llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1`
- seed 输出目录：
- seed0: `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0`
- seed1: `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed1`
- 数据集：`data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_local_exec_clean_v2_100/tasks.json`
- buckets：`analysis/shared_basin_prefix_dedup_profile_switch_local_exec_clean_v2_100_smoke10_schedule_buckets.json`
- methods：`risky_ps`, `direct_multistage_exp3`, `epsilon_exp3`
- repeats/horizon：`10x10`, 每个 method 100 episodes；`switch_denominator=4`, `eta=0.3`, `epsilon=0.01`
- 模型：Stage 1/2/3 使用 `gpt-4o-mini`，Stage 4/5 使用 `gpt-4.1-mini`
- prompt：`PSAGENT_TELECOM_STAGE45_CONTRACT_PROMPT_V1_1B=1`
- 其他开关：terminalv4, reasoning calibration v3, mode mismatch report-only v2

## 总表

| method | n | raw_total_cost_mean | post_raw_total_cost_mean | raw_terminal_penalty_mean | post_raw_terminal_penalty_mean | raw_reasoning_cost_component_mean | raw_mode_mismatch_cost_component_mean | exact_match_rate | clear_success_rate | aux_success_rate | subset_mismatch_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| risky_ps | 200 | 8.209 | 9.207 | 2.970 | 3.893 | 5.168 | 1.405 | 0.820 | 0.820 | 0.715 | 0.180 |
| direct_multistage_exp3 | 200 | 7.774 | 8.677 | 2.455 | 3.273 | 5.248 | 1.442 | 0.870 | 0.870 | 0.750 | 0.130 |
| epsilon_exp3 | 200 | 8.412 | 9.370 | 3.135 | 4.113 | 5.207 | 1.587 | 0.800 | 0.800 | 0.715 | 0.200 |

## 分 Seed 表

| seed | method | n | raw_total_cost_mean | post_raw_total_cost_mean | raw_terminal_penalty_mean | raw_reasoning_cost_component_mean | exact_match_rate | clear_success_rate | aux_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | direct_multistage_exp3 | 100 | 7.554 | 8.345 | 2.280 | 5.202 | 0.900 | 0.900 | 0.740 |
| 0 | risky_ps | 100 | 8.260 | 9.361 | 2.990 | 5.198 | 0.820 | 0.820 | 0.710 |
| 0 | epsilon_exp3 | 100 | 8.487 | 9.391 | 3.220 | 5.197 | 0.780 | 0.780 | 0.720 |
| 1 | direct_multistage_exp3 | 100 | 7.994 | 9.009 | 2.630 | 5.293 | 0.840 | 0.840 | 0.760 |
| 1 | risky_ps | 100 | 8.159 | 9.053 | 2.950 | 5.137 | 0.820 | 0.820 | 0.720 |
| 1 | epsilon_exp3 | 100 | 8.337 | 9.350 | 3.050 | 5.217 | 0.820 | 0.820 | 0.710 |

## Match/Mismatch 总表

| method | matched | n | post_n | raw_total_cost_mean | raw_terminal_penalty_mean | raw_reasoning_cost_component_mean | raw_mode_mismatch_cost_component_mean | exact_match_rate | clear_success_rate | aux_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | no | 154 | 104 | 7.875 | 2.487 | 5.318 | 1.873 | 0.838 | 0.838 | 0.786 |
| direct_multistage_exp3 | yes | 46 | 46 | 7.435 | 2.348 | 5.012 | 0.000 | 0.978 | 0.978 | 0.630 |
| epsilon_exp3 | no | 151 | 101 | 8.827 | 3.397 | 5.361 | 2.103 | 0.755 | 0.755 | 0.728 |
| epsilon_exp3 | yes | 49 | 49 | 7.134 | 2.327 | 4.732 | 0.000 | 0.939 | 0.939 | 0.673 |
| risky_ps | no | 158 | 108 | 8.583 | 3.228 | 5.285 | 1.778 | 0.772 | 0.772 | 0.728 |
| risky_ps | yes | 42 | 42 | 6.802 | 2.000 | 4.726 | 0.000 | 1.000 | 1.000 | 0.667 |

## Agent Pattern × Task Pattern

| method | actual_agent_pattern | required_task_pattern | matched | n | post_n | raw_total_cost_mean | raw_terminal_penalty_mean | raw_reasoning_cost_component_mean | raw_mode_mismatch_cost_component_mean | exact_match_rate | clear_success_rate | aux_success_rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | fdddd | fdddd | yes | 46 | 46 | 7.435 | 2.348 | 5.012 | 0.000 | 0.978 | 0.978 | 0.630 |
| direct_multistage_exp3 | ddddd | fdddd | no | 23 | 23 | 7.688 | 2.087 | 5.525 | 0.500 | 0.957 | 0.957 | 0.696 |
| direct_multistage_exp3 | ffddd | fdddd | no | 20 | 20 | 7.279 | 2.400 | 4.809 | 1.500 | 0.800 | 0.800 | 0.800 |
| direct_multistage_exp3 | dfddd | fdddd | no | 15 | 15 | 6.705 | 1.600 | 5.034 | 2.000 | 0.867 | 0.867 | 0.867 |
| direct_multistage_exp3 | dfddd | fffff | no | 10 | 0 | 5.716 | 0.000 | 5.644 | 2.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | fdddd | fffff | no | 9 | 0 | 5.526 | 0.000 | 5.452 | 2.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | ffddd | fffff | no | 9 | 0 | 5.030 | 0.000 | 4.959 | 1.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | fdddf | fdddd | no | 7 | 7 | 9.818 | 3.857 | 5.893 | 1.500 | 0.714 | 0.714 | 0.714 |
| direct_multistage_exp3 | fdfdd | fdddd | no | 6 | 6 | 5.937 | 1.000 | 4.866 | 1.500 | 1.000 | 1.000 | 0.833 |
| direct_multistage_exp3 | fffdd | fdddd | no | 5 | 5 | 11.355 | 5.600 | 5.691 | 3.000 | 0.600 | 0.600 | 0.600 |
| direct_multistage_exp3 | ddddd | fffff | no | 4 | 0 | 6.546 | 0.000 | 6.471 | 2.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dfdfd | fdddd | no | 3 | 3 | 13.627 | 8.000 | 5.558 | 3.500 | 0.333 | 0.333 | 0.333 |
| direct_multistage_exp3 | dffdd | fdddd | no | 3 | 3 | 9.611 | 4.000 | 5.545 | 3.500 | 0.667 | 0.667 | 0.667 |
| direct_multistage_exp3 | dfffd | fdddd | no | 3 | 3 | 16.301 | 9.667 | 6.574 | 5.000 | 0.333 | 0.333 | 0.333 |
| direct_multistage_exp3 | fddfd | fdddd | no | 3 | 3 | 10.041 | 4.667 | 5.302 | 1.500 | 0.667 | 0.667 | 0.667 |
| direct_multistage_exp3 | fdfdd | fffff | no | 3 | 0 | 4.993 | 0.000 | 4.922 | 1.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | ffddf | fdddd | no | 3 | 3 | 10.496 | 4.667 | 5.765 | 3.000 | 0.667 | 0.667 | 0.667 |
| direct_multistage_exp3 | fffdd | fffff | no | 3 | 0 | 4.343 | 0.000 | 4.272 | 1.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dfddf | fdddd | no | 2 | 2 | 12.867 | 6.000 | 6.805 | 3.500 | 0.500 | 0.500 | 0.500 |
| direct_multistage_exp3 | dffdf | fdddd | no | 2 | 2 | 7.088 | 0.000 | 7.026 | 5.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | fdffd | fffff | no | 2 | 0 | 3.995 | 0.000 | 3.933 | 1.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | ffdfd | fdddd | no | 2 | 2 | 20.344 | 14.500 | 5.777 | 3.000 | 0.000 | 0.000 | 0.000 |
| direct_multistage_exp3 | fffdf | fdddd | no | 2 | 2 | 13.171 | 7.000 | 6.107 | 4.500 | 0.500 | 0.500 | 0.500 |
| direct_multistage_exp3 | ffffd | fffff | no | 2 | 0 | 3.331 | 0.000 | 3.271 | 0.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dfddf | fffff | no | 1 | 0 | 4.896 | 0.000 | 4.826 | 1.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dfdfd | fffff | no | 1 | 0 | 4.466 | 0.000 | 4.400 | 1.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dffdf | fffff | no | 1 | 0 | 4.478 | 0.000 | 4.412 | 1.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dfffd | fffff | no | 1 | 0 | 4.072 | 0.000 | 4.009 | 1.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | dffff | fdddd | no | 1 | 1 | 21.125 | 14.000 | 7.067 | 6.500 | 0.000 | 0.000 | 0.000 |
| direct_multistage_exp3 | dffff | fffff | no | 1 | 0 | 3.543 | 0.000 | 3.488 | 0.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | fdfdf | fdddd | no | 1 | 1 | 14.699 | 6.000 | 8.627 | 3.000 | 1.000 | 1.000 | 0.000 |
| direct_multistage_exp3 | fdffd | fdddd | no | 1 | 1 | 5.519 | 0.000 | 5.458 | 3.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | ffddf | fffff | no | 1 | 0 | 4.144 | 0.000 | 4.084 | 1.000 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | ffdff | fdddd | no | 1 | 1 | 23.516 | 17.000 | 6.451 | 4.500 | 0.000 | 0.000 | 0.000 |
| direct_multistage_exp3 | ffdff | fffff | no | 1 | 0 | 2.947 | 0.000 | 2.889 | 0.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | fffdf | fffff | no | 1 | 0 | 3.749 | 0.000 | 3.685 | 0.500 | 1.000 | 1.000 | 1.000 |
| direct_multistage_exp3 | ffffd | fdddd | no | 1 | 1 | 23.362 | 17.000 | 6.298 | 4.500 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | fdddd | fdddd | yes | 49 | 49 | 7.134 | 2.327 | 4.732 | 0.000 | 0.939 | 0.939 | 0.673 |
| epsilon_exp3 | ffddd | fdddd | no | 32 | 32 | 8.038 | 3.375 | 4.593 | 1.500 | 0.719 | 0.719 | 0.719 |
| epsilon_exp3 | dfddd | fdddd | no | 13 | 13 | 12.239 | 6.462 | 5.704 | 2.000 | 0.462 | 0.462 | 0.462 |
| epsilon_exp3 | fffdd | fffff | no | 9 | 0 | 4.452 | 0.000 | 4.385 | 1.000 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | ddddd | fffff | no | 8 | 0 | 7.521 | 1.250 | 6.196 | 2.500 | 0.875 | 0.875 | 1.000 |
| epsilon_exp3 | dfddd | fffff | no | 8 | 0 | 6.044 | 0.000 | 5.975 | 2.000 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | fdddd | fffff | no | 8 | 0 | 5.783 | 0.000 | 5.707 | 2.000 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | ffddf | fdddd | no | 8 | 8 | 7.103 | 1.750 | 5.291 | 3.000 | 0.875 | 0.875 | 0.875 |
| epsilon_exp3 | ddddd | fdddd | no | 7 | 7 | 7.670 | 1.714 | 5.879 | 0.500 | 1.000 | 1.000 | 0.714 |
| epsilon_exp3 | ffddd | fffff | no | 7 | 0 | 5.086 | 0.000 | 5.015 | 1.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | fdfdd | fdddd | no | 6 | 6 | 6.970 | 2.000 | 4.900 | 1.500 | 1.000 | 1.000 | 0.667 |
| epsilon_exp3 | dffdd | fdddd | no | 4 | 4 | 6.766 | 0.000 | 6.699 | 3.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | fddfd | fdddd | no | 4 | 4 | 14.578 | 9.250 | 5.256 | 1.500 | 0.250 | 0.250 | 0.250 |
| epsilon_exp3 | fffdd | fdddd | no | 4 | 4 | 8.237 | 3.500 | 4.670 | 3.000 | 0.750 | 0.750 | 0.750 |
| epsilon_exp3 | ffffd | fdddd | no | 3 | 3 | 21.348 | 15.000 | 6.288 | 4.500 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | dfddf | fdddd | no | 2 | 2 | 12.296 | 6.000 | 6.231 | 3.500 | 0.500 | 0.500 | 0.500 |
| epsilon_exp3 | dfdfd | fdddd | no | 2 | 2 | 11.537 | 6.000 | 5.469 | 3.500 | 0.500 | 0.500 | 0.500 |
| epsilon_exp3 | dffdd | fffff | no | 2 | 0 | 5.030 | 0.000 | 4.965 | 1.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | dffdf | fdddd | no | 2 | 2 | 22.910 | 14.000 | 8.843 | 5.000 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | dfffd | fdddd | no | 2 | 2 | 14.999 | 8.500 | 6.437 | 5.000 | 0.500 | 0.500 | 0.500 |
| epsilon_exp3 | fdddf | fdddd | no | 2 | 2 | 8.217 | 3.000 | 5.149 | 1.500 | 1.000 | 1.000 | 0.500 |
| epsilon_exp3 | fdfdd | fffff | no | 2 | 0 | 5.180 | 0.000 | 5.110 | 1.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | fdfdf | fdddd | no | 2 | 2 | 5.624 | 0.000 | 5.560 | 3.000 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | fdffd | fdddd | no | 2 | 2 | 22.181 | 15.500 | 6.614 | 3.000 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | fdffd | fffff | no | 2 | 0 | 4.207 | 0.000 | 4.143 | 1.000 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | ffdfd | fffff | no | 2 | 0 | 3.824 | 0.000 | 3.754 | 1.000 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | dfddf | fffff | no | 1 | 0 | 4.888 | 0.000 | 4.825 | 1.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | dfdff | fdddd | no | 1 | 1 | 21.333 | 14.000 | 7.270 | 5.000 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | dffff | fdddd | no | 1 | 1 | 20.887 | 14.000 | 6.832 | 6.500 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | fdddf | fffff | no | 1 | 0 | 5.124 | 0.000 | 5.065 | 1.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | fddff | fdddd | no | 1 | 1 | 19.701 | 14.000 | 5.635 | 3.000 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | fdfff | fdddd | no | 1 | 1 | 6.306 | 0.000 | 6.249 | 4.500 | 1.000 | 1.000 | 1.000 |
| epsilon_exp3 | ffdfd | fdddd | no | 1 | 1 | 19.242 | 14.000 | 5.173 | 3.000 | 0.000 | 0.000 | 0.000 |
| epsilon_exp3 | fffff | fdddd | no | 1 | 1 | 21.931 | 15.000 | 6.872 | 6.000 | 0.000 | 0.000 | 0.000 |
| risky_ps | fdddd | fdddd | yes | 42 | 42 | 6.802 | 2.000 | 4.726 | 0.000 | 1.000 | 1.000 | 0.667 |
| risky_ps | ddddd | fdddd | no | 26 | 26 | 6.244 | 0.923 | 5.246 | 0.500 | 1.000 | 1.000 | 0.846 |
| risky_ps | dfddd | fdddd | no | 21 | 21 | 10.655 | 5.143 | 5.442 | 2.000 | 0.571 | 0.571 | 0.571 |
| risky_ps | ffddd | fdddd | no | 19 | 19 | 9.227 | 4.421 | 4.737 | 1.500 | 0.632 | 0.632 | 0.632 |
| risky_ps | fdddd | fffff | no | 9 | 0 | 6.888 | 1.111 | 5.702 | 2.000 | 0.889 | 0.889 | 1.000 |
| risky_ps | fdfdd | fdddd | no | 9 | 9 | 7.158 | 2.000 | 5.086 | 1.500 | 1.000 | 1.000 | 0.667 |
| risky_ps | ffddd | fffff | no | 9 | 0 | 4.839 | 0.000 | 4.766 | 1.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | ddddd | fffff | no | 5 | 0 | 6.908 | 0.000 | 6.832 | 2.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | dffdd | fdddd | no | 5 | 5 | 10.214 | 4.800 | 5.349 | 3.500 | 0.600 | 0.600 | 0.600 |
| risky_ps | fffdd | fffff | no | 5 | 0 | 4.431 | 0.000 | 4.366 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | fffdd | fdddd | no | 4 | 4 | 13.507 | 7.000 | 6.439 | 3.000 | 0.500 | 0.500 | 0.500 |
| risky_ps | dfddd | fffff | no | 3 | 0 | 5.420 | 0.000 | 5.349 | 2.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | dfdfd | fdddd | no | 3 | 3 | 20.821 | 14.000 | 6.751 | 3.500 | 0.000 | 0.000 | 0.000 |
| risky_ps | fdddf | fdddd | no | 3 | 3 | 5.491 | 0.000 | 5.420 | 1.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | fddfd | fdddd | no | 3 | 3 | 14.101 | 8.667 | 5.362 | 1.500 | 0.333 | 0.333 | 0.333 |
| risky_ps | fdfdd | fffff | no | 3 | 0 | 5.244 | 0.000 | 5.170 | 1.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | fdffd | fdddd | no | 3 | 3 | 10.344 | 4.667 | 5.611 | 3.000 | 0.667 | 0.667 | 0.667 |
| risky_ps | dfddf | fffff | no | 2 | 0 | 4.776 | 0.000 | 4.713 | 1.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | dffdd | fffff | no | 2 | 0 | 4.920 | 0.000 | 4.852 | 1.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | dffff | fffff | no | 2 | 0 | 3.569 | 0.000 | 3.513 | 0.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | fdfdf | fdddd | no | 2 | 2 | 10.581 | 3.000 | 7.515 | 3.000 | 1.000 | 1.000 | 0.500 |
| risky_ps | ffffd | fdddd | no | 2 | 2 | 12.785 | 7.000 | 5.721 | 4.500 | 0.500 | 0.500 | 0.500 |
| risky_ps | dfdff | fffff | no | 1 | 0 | 3.959 | 0.000 | 3.896 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | dffdf | fdddd | no | 1 | 1 | 23.261 | 14.000 | 9.200 | 5.000 | 0.000 | 0.000 | 0.000 |
| risky_ps | dffdf | fffff | no | 1 | 0 | 4.495 | 0.000 | 4.432 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | dfffd | fdddd | no | 1 | 1 | 20.103 | 14.000 | 6.039 | 5.000 | 0.000 | 0.000 | 0.000 |
| risky_ps | dfffd | fffff | no | 1 | 0 | 4.247 | 0.000 | 4.187 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | fdddf | fffff | no | 1 | 0 | 5.425 | 0.000 | 5.347 | 1.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | fddff | fdddd | no | 1 | 1 | 19.922 | 14.000 | 5.851 | 3.000 | 0.000 | 0.000 | 0.000 |
| risky_ps | fdffd | fffff | no | 1 | 0 | 3.855 | 0.000 | 3.795 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | fdfff | fdddd | no | 1 | 1 | 20.278 | 14.000 | 6.213 | 4.500 | 0.000 | 0.000 | 0.000 |
| risky_ps | ffddf | fdddd | no | 1 | 1 | 20.049 | 14.000 | 5.982 | 3.000 | 0.000 | 0.000 | 0.000 |
| risky_ps | ffddf | fffff | no | 1 | 0 | 3.972 | 0.000 | 3.908 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | ffdfd | fffff | no | 1 | 0 | 3.891 | 0.000 | 3.830 | 1.000 | 1.000 | 1.000 | 1.000 |
| risky_ps | ffdff | fdddd | no | 1 | 1 | 20.864 | 14.000 | 6.802 | 4.500 | 0.000 | 0.000 | 0.000 |
| risky_ps | ffdff | fffff | no | 1 | 0 | 3.254 | 0.000 | 3.192 | 0.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | fffdf | fdddd | no | 1 | 1 | 20.606 | 14.000 | 6.545 | 4.500 | 0.000 | 0.000 | 0.000 |
| risky_ps | fffdf | fffff | no | 1 | 0 | 3.757 | 0.000 | 3.690 | 0.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | ffffd | fffff | no | 1 | 0 | 3.182 | 0.000 | 3.118 | 0.500 | 1.000 | 1.000 | 1.000 |
| risky_ps | fffff | fdddd | no | 1 | 1 | 20.357 | 14.000 | 6.304 | 6.000 | 0.000 | 0.000 | 0.000 |

## 高 Cost 样例

| seed | method | episode_index | dataset_index | actual_agent_pattern | required_task_pattern | matched | schedule_phase | oracle_action | final_action | raw_total_cost | raw_terminal_penalty | raw_reasoning_cost_component | raw_mode_mismatch_cost_component | exact_match | clear_success | aux_success | subset_mismatch |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | risky_ps | 28 | 13 | dfdfd | fdddd | no | target_post_switch | repair_subset | repair_subset | 24.458 | 17.000 | 7.389 | 3.500 | no | no | no | yes |
| 1 | direct_multistage_exp3 | 48 | 13 | fdddf | fdddd | no | target_post_switch | repair_subset | repair_subset | 24.176 | 15.000 | 9.114 | 1.500 | no | no | no | yes |
| 1 | epsilon_exp3 | 78 | 13 | dfffd | fdddd | no | target_post_switch | repair_subset | repair_subset | 24.093 | 17.000 | 7.030 | 5.000 | no | no | no | yes |
| 1 | direct_multistage_exp3 | 88 | 13 | ffdff | fdddd | no | target_post_switch | repair_subset | repair_subset | 23.516 | 17.000 | 6.451 | 4.500 | no | no | no | yes |
| 1 | direct_multistage_exp3 | 78 | 13 | ffffd | fdddd | no | target_post_switch | repair_subset | repair_subset | 23.362 | 17.000 | 6.298 | 4.500 | no | no | no | yes |
| 1 | risky_ps | 57 | 10 | dffdf | fdddd | no | target_post_switch | repair_subset | repair_subset | 23.261 | 14.000 | 9.200 | 5.000 | no | no | no | yes |
| 1 | epsilon_exp3 | 58 | 13 | fdffd | fdddd | no | target_post_switch | repair_subset | repair_subset | 23.242 | 17.000 | 6.166 | 3.000 | no | no | no | yes |
| 0 | epsilon_exp3 | 47 | 10 | dffdf | fdddd | no | target_post_switch | repair_subset | repair_subset | 23.114 | 14.000 | 9.047 | 5.000 | no | no | no | yes |
| 0 | epsilon_exp3 | 88 | 13 | ffffd | fdddd | no | target_post_switch | repair_subset | repair_subset | 22.869 | 17.000 | 5.807 | 4.500 | no | no | no | yes |
| 0 | epsilon_exp3 | 57 | 10 | dffdf | fdddd | no | target_post_switch | repair_subset | repair_subset | 22.706 | 14.000 | 8.639 | 5.000 | no | no | no | yes |
| 0 | direct_multistage_exp3 | 69 | 16 | dfffd | fdddd | no | target_post_switch | repair_all | repair_subset | 22.547 | 15.000 | 7.489 | 5.000 | no | no | no | yes |
| 1 | risky_ps | 58 | 13 | fffdd | fdddd | no | target_post_switch | repair_subset | repair_all | 22.519 | 14.000 | 8.446 | 3.000 | no | no | no | yes |

## 因果链观察

- direct 赢的关键不是更便宜的 LLM 调用，而是 terminal 更稳：aggregate reasoning cost direct 是 `5.248`，risky_ps 是 `5.168`；但 terminal penalty direct 是 `2.455`，risky_ps 是 `2.970`。
- `risky_ps` 在 post-switch 上 raw_total_cost `9.207`，direct 是 `8.677`；这说明当前 seed0/seed1 下 PS shared-update 仍没有把 target/deep 后段执行质量稳定到 direct 之上。
- smoke episode 记录没有持久化 Stage 4/5 原始 JSON plan，所以这里的“真实输出因果链”只能从 final_action、oracle_action、subset_mismatch、terminal penalty、executed/replay tools 和 path pattern 推断；更细的 selected/deferred blocker 级因果仍需要 fixed trace 产物。
- mismatch 方向：跨全部 method，exact matched rows 的 raw_total_cost 低于 mismatched rows；但 risky/direct/epsilon 的路径选择不是均匀实验设计，严格证明“同任务同算法不匹配更贵”仍应引用 fixed-path probe。

## 和上一版区别

- 这版在 C config 基础上加入 `prompt v1.1b`，并把 Stage 4/5 模型从默认 `gpt-4o-mini` 切到 `gpt-4.1-mini`；Stage 1/2/3 仍是 `gpt-4o-mini`。
- 没改 PS algorithm、baseline、dataset、terminal penalty 语义，也没引入 normalizer auto-correction；mode mismatch 仍是 report-only。
- 相比上一个 C smoke 观察，这一版的结论不是“PS 稳定第一”，而是 direct 在 seed0/seed1 聚合上更稳；这支持继续查 PS update/路径采样稳定性，而不是继续大规模 formal run。

## 产物

- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/method_summary_aggregate.csv`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/method_summary_by_seed.csv`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/matched_vs_mismatched.csv`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/agent_task_pattern_pairs.csv`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/dataset_method_breakdown.csv`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/high_cost_examples.csv`
- `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/analysis_summary.json`
- wandb: `https://wandb.ai/wangjinyu0711-microsoft/psagent-llm-smoke/runs/yj7y49kq`
- wandb upload status: `tmp/llm_v8_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d4_eta03_eps001_10x10_3methods_seed0_seed1_compare/wandb_upload_status.json`
