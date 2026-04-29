# Full 13-Method Smoke Fast/Deep Cost Success Analysis

Source: `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini`

## Field Availability

- This 13-method full smoke export does **not** include true `clean_success_no_fallback`, `bench_aux_eval` / `bench_success`, per-stage fallback flags, hard-transfer guard flags, or completion-pass flags.
- I reconstructed path fast/deep and route labels from `selected_path` using the same family tree seed and topology.
- `exact_match` is used as terminal/clean-success proxy. `policy_violation_count == 0` is a weak policy-clean proxy, not an auxiliary execution-success signal.

## Definitions

- `F/F`: fast agent on fast-required stage.
- `F/D`: fast agent on deep-required stage, potential under-thinking.
- `D/F`: deep agent on fast-required stage, potential over-thinking.
- `D/D`: deep agent on deep-required stage.
- `pair_class`: path majority versus task majority, e.g. `agent_fast__task_deep`.

## Method Means Ranked By Total

| method | n | terminal | reasoning | total | exact | policy_clean | stage_match | F/D per ep | D/F per ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `epsilon_exp3` | 100 | 7.090 | 5.154 | 12.316 | 0.300 | 0.460 | 0.680 | 0.620 | 0.980 |
| `direct_multistage_exp3_local` | 100 | 7.825 | 5.002 | 12.898 | 0.300 | 0.430 | 0.690 | 0.590 | 0.960 |
| `risky_ps_linear` | 100 | 7.915 | 5.022 | 13.007 | 0.220 | 0.380 | 0.594 | 0.930 | 1.100 |
| `direct_multistage_exp3` | 100 | 7.920 | 5.076 | 13.067 | 0.250 | 0.470 | 0.670 | 0.580 | 1.070 |
| `risky_ps_ix` | 100 | 8.045 | 5.072 | 13.187 | 0.220 | 0.460 | 0.646 | 0.690 | 1.080 |
| `risky_ps_old` | 100 | 8.475 | 5.108 | 13.654 | 0.230 | 0.440 | 0.654 | 0.660 | 1.070 |
| `risky_ps_safe_conditional_ix` | 100 | 8.545 | 5.232 | 13.847 | 0.230 | 0.420 | 0.586 | 0.920 | 1.150 |
| `naive_mixed` | 100 | 8.645 | 5.153 | 13.870 | 0.190 | 0.440 | 0.636 | 0.720 | 1.100 |
| `naive_mixed_avg` | 100 | 8.430 | 5.377 | 13.877 | 0.220 | 0.460 | 0.536 | 0.640 | 1.680 |
| `risky_ps` | 100 | 8.650 | 5.255 | 13.975 | 0.220 | 0.450 | 0.602 | 0.800 | 1.190 |
| `risky_ps_safe_conditional` | 100 | 8.875 | 5.117 | 14.061 | 0.180 | 0.430 | 0.576 | 1.170 | 0.950 |
| `risky_ps_direct_cost` | 100 | 8.920 | 5.182 | 14.172 | 0.190 | 0.440 | 0.612 | 0.770 | 1.170 |
| `random_path` | 100 | 9.115 | 5.156 | 14.341 | 0.200 | 0.400 | 0.656 | 0.700 | 1.020 |

## Overall Path/Task Majority Buckets

| pair_class | n | terminal | reasoning | total | exact | policy_clean | stage_match | F/D | D/F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `agent_fast__task_fast` | 46 | 1.576 | 4.691 | 6.332 | 0.413 | 0.804 | 0.639 | 0.000 | 1.804 |
| `agent_fast__task_deep` | 190 | 10.579 | 5.209 | 15.852 | 0.042 | 0.147 | 0.465 | 2.526 | 0.147 |
| `agent_deep__task_fast` | 279 | 1.328 | 5.066 | 6.466 | 0.398 | 0.896 | 0.248 | 0.000 | 3.760 |
| `agent_deep__task_deep` | 785 | 10.690 | 5.187 | 15.949 | 0.200 | 0.322 | 0.798 | 0.636 | 0.372 |

## Phase x Path/Task Majority

| phase | pair_class | n | terminal | reasoning | total | exact | policy_clean | stage_match |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| trap_pre_switch | `agent_fast__task_fast` | 46 | 1.576 | 4.691 | 6.332 | 0.413 | 0.804 | 0.639 |
| trap_pre_switch | `agent_deep__task_fast` | 279 | 1.328 | 5.066 | 6.466 | 0.398 | 0.896 | 0.248 |
| target_post_switch | `agent_fast__task_deep` | 190 | 10.579 | 5.209 | 15.852 | 0.042 | 0.147 | 0.465 |
| target_post_switch | `agent_deep__task_deep` | 785 | 10.690 | 5.187 | 15.949 | 0.200 | 0.322 | 0.798 |

## Method x Path/Task Majority

| method | pair_class | n | terminal | reasoning | total | exact | policy_clean | F/D | D/F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `epsilon_exp3` | `agent_fast__task_fast` | 5 | 1.700 | 3.939 | 5.705 | 0.200 | 0.800 | 0.000 | 1.800 |
| `epsilon_exp3` | `agent_fast__task_deep` | 9 | 11.333 | 5.376 | 16.774 | 0.000 | 0.333 | 2.333 | 0.000 |
| `epsilon_exp3` | `agent_deep__task_fast` | 20 | 1.175 | 5.403 | 6.648 | 0.450 | 0.900 | 0.000 | 3.700 |
| `epsilon_exp3` | `agent_deep__task_deep` | 66 | 8.712 | 5.141 | 13.926 | 0.303 | 0.318 | 0.621 | 0.227 |
| `direct_multistage_exp3_local` | `agent_fast__task_fast` | 6 | 0.500 | 4.365 | 4.931 | 0.667 | 1.000 | 0.000 | 1.500 |
| `direct_multistage_exp3_local` | `agent_fast__task_deep` | 9 | 6.500 | 5.112 | 11.676 | 0.111 | 0.222 | 2.556 | 0.222 |
| `direct_multistage_exp3_local` | `agent_deep__task_fast` | 19 | 1.737 | 4.776 | 6.584 | 0.316 | 0.842 | 0.000 | 3.526 |
| `direct_multistage_exp3_local` | `agent_deep__task_deep` | 66 | 10.424 | 5.110 | 15.607 | 0.288 | 0.288 | 0.545 | 0.273 |
| `risky_ps_linear` | `agent_fast__task_fast` | 3 | 1.833 | 4.871 | 6.768 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps_linear` | `agent_fast__task_deep` | 22 | 9.591 | 5.215 | 14.871 | 0.045 | 0.091 | 2.500 | 0.182 |
| `risky_ps_linear` | `agent_deep__task_fast` | 22 | 1.318 | 4.761 | 6.151 | 0.409 | 0.909 | 0.000 | 3.727 |
| `risky_ps_linear` | `agent_deep__task_deep` | 53 | 10.302 | 5.058 | 15.432 | 0.208 | 0.264 | 0.717 | 0.340 |
| `direct_multistage_exp3` | `agent_fast__task_fast` | 7 | 0.429 | 5.380 | 5.875 | 0.714 | 1.000 | 0.000 | 1.714 |
| `direct_multistage_exp3` | `agent_fast__task_deep` | 9 | 10.722 | 5.474 | 16.257 | 0.000 | 0.111 | 2.778 | 0.444 |
| `direct_multistage_exp3` | `agent_deep__task_fast` | 18 | 1.611 | 4.733 | 6.416 | 0.278 | 0.833 | 0.000 | 3.778 |
| `direct_multistage_exp3` | `agent_deep__task_deep` | 66 | 10.053 | 5.083 | 15.209 | 0.227 | 0.364 | 0.500 | 0.348 |
| `risky_ps_ix` | `agent_fast__task_fast` | 3 | 2.500 | 5.318 | 7.882 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps_ix` | `agent_fast__task_deep` | 10 | 8.900 | 5.013 | 13.977 | 0.100 | 0.200 | 2.500 | 0.000 |
| `risky_ps_ix` | `agent_deep__task_fast` | 22 | 1.295 | 4.855 | 6.223 | 0.409 | 0.909 | 0.000 | 3.727 |
| `risky_ps_ix` | `agent_deep__task_deep` | 65 | 10.454 | 5.143 | 15.668 | 0.169 | 0.338 | 0.677 | 0.308 |
| `risky_ps_old` | `agent_fast__task_fast` | 3 | 2.500 | 5.093 | 7.657 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps_old` | `agent_fast__task_deep` | 16 | 12.969 | 4.934 | 17.967 | 0.000 | 0.062 | 2.250 | 0.062 |
| `risky_ps_old` | `agent_deep__task_fast` | 22 | 1.205 | 5.041 | 6.318 | 0.409 | 0.909 | 0.000 | 3.727 |
| `risky_ps_old` | `agent_deep__task_deep` | 59 | 10.271 | 5.181 | 15.525 | 0.220 | 0.356 | 0.508 | 0.305 |
| `risky_ps_safe_conditional_ix` | `agent_fast__task_fast` | 3 | 2.333 | 5.281 | 7.678 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps_safe_conditional_ix` | `agent_fast__task_deep` | 19 | 11.132 | 5.266 | 16.461 | 0.000 | 0.053 | 2.684 | 0.316 |
| `risky_ps_safe_conditional_ix` | `agent_deep__task_fast` | 22 | 1.295 | 5.114 | 6.481 | 0.409 | 0.909 | 0.000 | 3.727 |
| `risky_ps_safe_conditional_ix` | `agent_deep__task_deep` | 56 | 10.848 | 5.264 | 16.184 | 0.232 | 0.339 | 0.732 | 0.375 |
| `naive_mixed` | `agent_fast__task_fast` | 1 | 1.500 | 4.349 | 5.914 | 0.000 | 1.000 | 0.000 | 2.000 |
| `naive_mixed` | `agent_fast__task_deep` | 20 | 9.275 | 4.968 | 14.308 | 0.000 | 0.150 | 2.500 | 0.000 |
| `naive_mixed` | `agent_deep__task_fast` | 24 | 1.271 | 5.226 | 6.569 | 0.417 | 0.875 | 0.000 | 3.833 |
| `naive_mixed` | `agent_deep__task_deep` | 55 | 11.764 | 5.203 | 17.041 | 0.164 | 0.345 | 0.400 | 0.291 |
| `naive_mixed_avg` | `agent_fast__task_deep` | 2 | 17.750 | 4.602 | 22.417 | 0.000 | 0.500 | 2.500 | 0.500 |
| `naive_mixed_avg` | `agent_deep__task_fast` | 25 | 1.440 | 5.281 | 6.793 | 0.400 | 0.880 | 0.000 | 4.120 |
| `naive_mixed_avg` | `agent_deep__task_deep` | 73 | 10.568 | 5.430 | 16.070 | 0.164 | 0.315 | 0.808 | 0.877 |
| `risky_ps` | `agent_fast__task_fast` | 3 | 1.833 | 4.839 | 6.736 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps` | `agent_fast__task_deep` | 11 | 11.364 | 5.298 | 16.724 | 0.091 | 0.182 | 2.727 | 0.273 |
| `risky_ps` | `agent_deep__task_fast` | 22 | 1.295 | 5.245 | 6.613 | 0.409 | 0.909 | 0.000 | 3.727 |
| `risky_ps` | `agent_deep__task_deep` | 64 | 11.031 | 5.271 | 16.373 | 0.172 | 0.328 | 0.781 | 0.438 |
| `risky_ps_safe_conditional` | `agent_fast__task_fast` | 3 | 2.500 | 5.291 | 7.855 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps_safe_conditional` | `agent_fast__task_deep` | 37 | 10.946 | 5.217 | 16.228 | 0.081 | 0.216 | 2.514 | 0.027 |
| `risky_ps_safe_conditional` | `agent_deep__task_fast` | 22 | 1.250 | 5.168 | 6.490 | 0.409 | 0.955 | 0.000 | 3.727 |
| `risky_ps_safe_conditional` | `agent_deep__task_deep` | 38 | 11.776 | 4.977 | 16.825 | 0.132 | 0.316 | 0.632 | 0.158 |
| `risky_ps_direct_cost` | `agent_fast__task_fast` | 3 | 1.833 | 5.046 | 6.943 | 0.333 | 0.667 | 0.000 | 2.000 |
| `risky_ps_direct_cost` | `agent_fast__task_deep` | 13 | 11.385 | 5.366 | 16.816 | 0.077 | 0.077 | 2.385 | 0.154 |
| `risky_ps_direct_cost` | `agent_deep__task_fast` | 22 | 1.114 | 4.881 | 6.067 | 0.409 | 0.909 | 0.000 | 3.727 |
| `risky_ps_direct_cost` | `agent_deep__task_deep` | 62 | 11.516 | 5.256 | 16.844 | 0.129 | 0.339 | 0.742 | 0.435 |
| `random_path` | `agent_fast__task_fast` | 6 | 1.750 | 3.444 | 5.257 | 0.333 | 0.833 | 0.000 | 1.500 |
| `random_path` | `agent_fast__task_deep` | 13 | 10.385 | 5.578 | 16.027 | 0.000 | 0.077 | 2.692 | 0.308 |
| `random_path` | `agent_deep__task_fast` | 19 | 1.342 | 5.291 | 6.704 | 0.421 | 0.895 | 0.000 | 3.737 |
| `random_path` | `agent_deep__task_deep` | 62 | 11.944 | 5.191 | 17.207 | 0.161 | 0.274 | 0.565 | 0.290 |

## Stage-Pair Observations Overall

Cost columns here are parent episode costs averaged over stage observations in that pair; `stage_tokens` is truly per-stage.

| stage_pair | stage_n | stage_tokens | episode_terminal | episode_reasoning | episode_total | exact | policy_clean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| F/F | 1148 | 3971.9 | 6.427 | 4.935 | 11.432 | 0.279 | 0.545 |
| F/D | 979 | 6115.6 | 11.038 | 5.200 | 16.304 | 0.072 | 0.194 |
| D/F | 1452 | 9652.5 | 3.608 | 5.194 | 8.873 | 0.341 | 0.751 |
| D/D | 2921 | 13555.2 | 10.545 | 5.188 | 15.805 | 0.202 | 0.320 |

## Reconstructed Archetype Overall

| archetype | n | terminal | reasoning | total | exact | F/D | D/F |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `neutral` | 587 | 7.871 | 5.081 | 13.021 | 0.215 | 0.973 | 1.220 |
| `target_safe_specialist_good` | 211 | 11.739 | 5.392 | 17.207 | 0.261 | 0.175 | 0.588 |
| `trap_like_bad` | 211 | 10.313 | 5.053 | 15.434 | 0.199 | 0.853 | 0.000 |
| `target_decoy_medium` | 96 | 10.448 | 5.262 | 15.783 | 0.125 | 1.646 | 0.271 |
| `trap_safe_overcautious` | 90 | 1.561 | 5.186 | 6.823 | 0.322 | 0.000 | 4.378 |
| `target_safe_majority_bad` | 55 | 7.018 | 5.037 | 12.127 | 0.000 | 0.600 | 0.618 |
| `trap_like_good` | 50 | 0.840 | 5.106 | 6.013 | 0.620 | 0.000 | 3.160 |

## Generated Files

- `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini/fastdeep_episode_detail_full13.csv`
- `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini/fastdeep_method_summary_full13.csv`
- `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini/fastdeep_pairclass_summary_full13.csv`
- `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini/fastdeep_phase_pairclass_summary_full13.csv`
- `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini/fastdeep_group_summary_full13.csv`
- `tmp/llm_repeated_smoke_profile_switch_lowtransfer_v4_10x10_d4_eta03_eps001_gpt4omini/fastdeep_stage_pair_summary_full13.csv`