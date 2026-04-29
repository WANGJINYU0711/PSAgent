# Fast/Deep Path vs Task Requirement Cost Analysis

Source: `tmp/llm_v4_profilefix_archetype_diagnostic_d4_eta03_eps001`

## Field availability

- True `clean_success_no_fallback`, `bench_aux_eval` / `bench_success`, and per-stage fallback fields are **not present** in this diagnostic export.
- I use `exact_match` as the terminal/clean-success proxy.
- I use `policy_violation_count == 0` as a weak auxiliary safety proxy, not an execution-success signal.

## Pairing Definitions

- `F/F`: fast agent on fast-required stage.
- `F/D`: fast agent on deep-required stage, potential under-thinking.
- `D/F`: deep agent on fast-required stage, potential over-thinking.
- `D/D`: deep agent on deep-required stage.
- `pair_class`: 5-stage path majority, e.g. `agent_deep__task_fast`.

## Method Means

| method | n | terminal | reasoning | total | exact_rate | stage_match_rate | F/D per ep | D/F per ep |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_multistage_exp3` | 20 | 12.075 | 5.226 | 17.371 | 0.150 | 0.580 | 0.950 | 1.150 |
| `epsilon_exp3` | 20 | 9.700 | 5.313 | 15.083 | 0.200 | 0.600 | 0.800 | 1.200 |
| `risky_ps_linear` | 20 | 8.675 | 5.067 | 13.813 | 0.300 | 0.660 | 0.650 | 1.050 |

## Method x Path/Task Majority

| method | pair_class | n | terminal | reasoning | total | exact_rate | stage_match | F/D | D/F |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_multistage_exp3` | `agent_fast__task_deep` | 5 | 17.700 | 5.422 | 23.190 | 0.000 | 0.560 | 2.200 | 0.000 |
| `direct_multistage_exp3` | `agent_deep__task_fast` | 5 | 1.400 | 4.597 | 6.067 | 0.400 | 0.280 | 0.000 | 3.600 |
| `direct_multistage_exp3` | `agent_deep__task_deep` | 10 | 14.600 | 5.443 | 20.113 | 0.100 | 0.740 | 0.800 | 0.500 |
| `epsilon_exp3` | `agent_fast__task_deep` | 4 | 8.000 | 5.778 | 13.845 | 0.000 | 0.550 | 2.250 | 0.000 |
| `epsilon_exp3` | `agent_deep__task_fast` | 5 | 1.300 | 4.825 | 6.194 | 0.600 | 0.280 | 0.000 | 3.600 |
| `epsilon_exp3` | `agent_deep__task_deep` | 11 | 14.136 | 5.365 | 19.573 | 0.091 | 0.764 | 0.636 | 0.545 |
| `risky_ps_linear` | `agent_fast__task_deep` | 4 | 11.000 | 5.353 | 16.418 | 0.000 | 0.500 | 2.250 | 0.250 |
| `risky_ps_linear` | `agent_deep__task_fast` | 5 | 1.100 | 4.516 | 5.685 | 0.600 | 0.360 | 0.000 | 3.200 |
| `risky_ps_linear` | `agent_deep__task_deep` | 11 | 11.273 | 5.213 | 16.560 | 0.273 | 0.855 | 0.364 | 0.364 |

## Stage-Pair Observations

Cost columns here are the parent episode cost averaged over stage observations in that pair; `stage_tokens` is genuinely per-stage.

| method | stage_pair | stage_n | stage_tokens | episode_terminal | episode_reasoning | episode_total | exact_rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_multistage_exp3` | F/F | 17 | 4180.2 | 10.382 | 5.069 | 15.521 | 0.176 |
| `direct_multistage_exp3` | F/D | 19 | 6560.3 | 17.842 | 5.445 | 23.355 | 0.000 |
| `direct_multistage_exp3` | D/F | 23 | 8606.5 | 4.043 | 4.794 | 8.909 | 0.348 |
| `direct_multistage_exp3` | D/D | 41 | 14427.5 | 14.610 | 5.431 | 20.111 | 0.098 |
| `epsilon_exp3` | F/F | 16 | 4164.0 | 6.219 | 5.018 | 11.304 | 0.375 |
| `epsilon_exp3` | F/D | 16 | 5733.8 | 10.875 | 5.614 | 16.557 | 0.000 |
| `epsilon_exp3` | D/F | 24 | 8948.6 | 5.021 | 5.103 | 10.194 | 0.417 |
| `epsilon_exp3` | D/D | 44 | 14864.4 | 13.091 | 5.425 | 18.587 | 0.091 |
| `risky_ps_linear` | F/F | 19 | 3932.8 | 5.789 | 4.818 | 10.678 | 0.474 |
| `risky_ps_linear` | F/D | 13 | 6233.2 | 11.192 | 5.269 | 16.527 | 0.000 |
| `risky_ps_linear` | D/F | 21 | 9179.9 | 4.071 | 4.767 | 8.909 | 0.429 |
| `risky_ps_linear` | D/D | 47 | 13496.1 | 11.202 | 5.245 | 16.521 | 0.255 |

## Episode Detail

| method | ep | phase | archetype | agent | task | stage pairs | majority | term | reason | total | exact | policy_clean | oracle | final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `direct_multistage_exp3` | 0 | trap/pre/switch | neutral | FDFDD | FFFFF | F/F,D/F,F/F,D/F,D/F | a=deep t=fast | 1.500 | 5.091 | 6.662 | N | Y | repair_all | repair_all |
| `direct_multistage_exp3` | 1 | trap/pre/switch | neutral | DFDDD | FFFFF | D/F,F/F,D/F,D/F,D/F | a=deep t=fast | 0.000 | 4.564 | 4.631 | Y | Y | repair_all | repair_all |
| `direct_multistage_exp3` | 2 | trap/pre/switch | neutral | FDDDD | FFFFF | F/F,D/F,D/F,D/F,D/F | a=deep t=fast | 0.000 | 4.658 | 4.729 | Y | Y | repair_all | repair_all |
| `direct_multistage_exp3` | 3 | trap/pre/switch | neutral | DFDDD | FFFFF | D/F,F/F,D/F,D/F,D/F | a=deep t=fast | 4.000 | 4.521 | 8.599 | N | N | repair_subset | repair_subset |
| `direct_multistage_exp3` | 4 | trap/pre/switch | neutral | DFDFD | FFFFF | D/F,F/F,D/F,F/F,D/F | a=deep t=fast | 1.500 | 4.149 | 5.715 | N | Y | repair_all | repair_all |
| `direct_multistage_exp3` | 5 | target/post/switch | neutral | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 13.000 | 5.005 | 18.079 | N | N | repair_subset | repair_subset |
| `direct_multistage_exp3` | 6 | target/post/switch | neutral | DFDDD | FDDDD | D/F,F/D,D/D,D/D,D/D | a=deep t=deep | 21.500 | 5.886 | 27.452 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 7 | target/post/switch | neutral | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 18.500 | 5.584 | 24.155 | N | Y | repair_all | transfer |
| `direct_multistage_exp3` | 8 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 2.000 | 5.014 | 7.090 | Y | N | repair_subset | repair_subset |
| `direct_multistage_exp3` | 9 | target/post/switch | neutral | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 21.500 | 5.545 | 27.112 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 10 | target/post/switch | trap_like_bad | FFDDD | FDDDD | F/F,F/D,D/D,D/D,D/D | a=deep t=deep | 16.500 | 5.076 | 21.638 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 11 | target/post/switch | neutral | FFDFF | FDDDD | F/F,F/D,D/D,F/D,F/D | a=fast t=deep | 17.000 | 4.165 | 21.230 | N | N | repair_all | repair_subset |
| `direct_multistage_exp3` | 12 | target/post/switch | trap_like_bad | FDDFF | FDDDD | F/F,D/D,D/D,F/D,F/D | a=fast t=deep | 18.500 | 6.812 | 25.372 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 13 | target/post/switch | neutral | DFDDD | FDDDD | D/F,F/D,D/D,D/D,D/D | a=deep t=deep | 21.500 | 4.590 | 26.159 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 14 | target/post/switch | neutral | DDDDD | FDDDD | D/F,D/D,D/D,D/D,D/D | a=deep t=deep | 1.000 | 4.794 | 5.869 | N | Y | transfer | transfer |
| `direct_multistage_exp3` | 15 | target/post/switch | neutral | DFFDD | FDDDD | D/F,F/D,F/D,D/D,D/D | a=deep t=deep | 13.000 | 6.383 | 19.457 | N | N | repair_subset | repair_subset |
| `direct_multistage_exp3` | 16 | target/post/switch | trap_like_bad | FDFDD | FDDDD | F/F,D/D,F/D,D/D,D/D | a=deep t=deep | 20.500 | 5.659 | 26.223 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 17 | target/post/switch | neutral | DDDDD | FDDDD | D/F,D/D,D/D,D/D,D/D | a=deep t=deep | 11.000 | 5.927 | 17.002 | N | Y | repair_all | repair_subset |
| `direct_multistage_exp3` | 18 | target/post/switch | neutral | FDDDF | FDDDD | F/F,D/D,D/D,D/D,F/D | a=deep t=deep | 18.500 | 5.012 | 23.584 | N | N | repair_subset | transfer |
| `direct_multistage_exp3` | 19 | target/post/switch | trap_like_bad | FFDDD | FDDDD | F/F,F/D,D/D,D/D,D/D | a=deep t=deep | 20.500 | 6.085 | 26.656 | N | N | repair_subset | transfer |
| `epsilon_exp3` | 0 | trap/pre/switch | trap_like_good | FFDDD | FFFFF | F/F,F/F,D/F,D/F,D/F | a=deep t=fast | 0.000 | 4.548 | 4.613 | Y | Y | repair_all | repair_all |
| `epsilon_exp3` | 1 | trap/pre/switch | neutral | DFDDF | FFFFF | D/F,F/F,D/F,D/F,F/F | a=deep t=fast | 0.000 | 4.233 | 4.296 | Y | Y | repair_all | repair_all |
| `epsilon_exp3` | 2 | trap/pre/switch | trap_like_good | FDDDF | FFFFF | F/F,D/F,D/F,D/F,F/F | a=deep t=fast | 0.000 | 4.399 | 4.458 | Y | Y | repair_all | repair_all |
| `epsilon_exp3` | 3 | trap/pre/switch | neutral | FDDDD | FFFFF | F/F,D/F,D/F,D/F,D/F | a=deep t=fast | 5.000 | 5.306 | 10.384 | N | Y | repair_subset | repair_all |
| `epsilon_exp3` | 4 | trap/pre/switch | neutral | DDDDD | FFFFF | D/F,D/F,D/F,D/F,D/F | a=deep t=fast | 1.500 | 5.641 | 7.217 | N | Y | repair_all | repair_all |
| `epsilon_exp3` | 5 | target/post/switch | neutral | DDDDD | FDDDD | D/F,D/D,D/D,D/D,D/D | a=deep t=deep | 2.000 | 5.714 | 7.790 | Y | N | repair_subset | repair_subset |
| `epsilon_exp3` | 6 | target/post/switch | neutral | DFDDD | FDDDD | D/F,F/D,D/D,D/D,D/D | a=deep t=deep | 21.500 | 5.832 | 27.403 | N | N | repair_subset | transfer |
| `epsilon_exp3` | 7 | target/post/switch | neutral | FFDDD | FDDDD | F/F,F/D,D/D,D/D,D/D | a=deep t=deep | 11.000 | 5.914 | 16.988 | N | Y | repair_all | repair_subset |
| `epsilon_exp3` | 8 | target/post/switch | neutral | FFDDD | FDDDD | F/F,F/D,D/D,D/D,D/D | a=deep t=deep | 3.000 | 4.448 | 7.518 | N | N | repair_subset | repair_subset |
| `epsilon_exp3` | 9 | target/post/switch | trap_like_bad | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 20.500 | 5.725 | 26.295 | N | N | repair_subset | transfer |
| `epsilon_exp3` | 10 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 17.000 | 4.188 | 21.261 | N | Y | repair_subset | repair_subset |
| `epsilon_exp3` | 11 | target/post/switch | neutral | DFFDD | FDDDD | D/F,F/D,F/D,D/D,D/D | a=deep t=deep | 18.500 | 5.405 | 23.973 | N | Y | repair_all | transfer |
| `epsilon_exp3` | 12 | target/post/switch | neutral | DFDDD | FDDDD | D/F,F/D,D/D,D/D,D/D | a=deep t=deep | 19.500 | 5.159 | 24.725 | N | N | repair_subset | transfer |
| `epsilon_exp3` | 13 | target/post/switch | neutral | FFFFD | FDDDD | F/F,F/D,F/D,F/D,D/D | a=fast t=deep | 5.000 | 5.595 | 10.659 | N | N | repair_subset | repair_subset |
| `epsilon_exp3` | 14 | target/post/switch | neutral | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 1.000 | 4.326 | 5.393 | N | Y | transfer | transfer |
| `epsilon_exp3` | 15 | target/post/switch | neutral | FDFFD | FDDDD | F/F,D/D,F/D,F/D,D/D | a=fast t=deep | 11.000 | 8.695 | 19.765 | N | Y | repair_subset | repair_subset |
| `epsilon_exp3` | 16 | target/post/switch | neutral | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 15.000 | 4.498 | 19.565 | N | N | repair_subset | repair_subset |
| `epsilon_exp3` | 17 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 11.000 | 5.231 | 16.307 | N | Y | repair_all | repair_subset |
| `epsilon_exp3` | 18 | target/post/switch | neutral | DDDDD | FDDDD | D/F,D/D,D/D,D/D,D/D | a=deep t=deep | 18.500 | 5.565 | 24.142 | N | N | repair_subset | transfer |
| `epsilon_exp3` | 19 | target/post/switch | neutral | DFDDD | FDDDD | D/F,F/D,D/D,D/D,D/D | a=deep t=deep | 13.000 | 5.835 | 18.904 | N | N | repair_subset | repair_subset |
| `risky_ps_linear` | 0 | trap/pre/switch | trap_like_good | FFDDD | FFFFF | F/F,F/F,D/F,D/F,D/F | a=deep t=fast | 0.000 | 4.564 | 4.631 | Y | Y | repair_all | repair_all |
| `risky_ps_linear` | 1 | trap/pre/switch | neutral | DFDDF | FFFFF | D/F,F/F,D/F,D/F,F/F | a=deep t=fast | 0.000 | 4.222 | 4.288 | Y | Y | repair_all | repair_all |
| `risky_ps_linear` | 2 | trap/pre/switch | neutral | FFDDD | FFFFF | F/F,F/F,D/F,D/F,D/F | a=deep t=fast | 0.000 | 3.954 | 4.029 | Y | Y | repair_all | repair_all |
| `risky_ps_linear` | 3 | trap/pre/switch | neutral | DFDDD | FFFFF | D/F,F/F,D/F,D/F,D/F | a=deep t=fast | 4.000 | 4.883 | 8.958 | N | N | repair_subset | repair_subset |
| `risky_ps_linear` | 4 | trap/pre/switch | neutral | DFFDD | FFFFF | D/F,F/F,F/F,D/F,D/F | a=deep t=fast | 1.500 | 4.956 | 6.520 | N | Y | repair_all | repair_all |
| `risky_ps_linear` | 5 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 2.000 | 5.011 | 7.087 | Y | N | repair_subset | repair_subset |
| `risky_ps_linear` | 6 | target/post/switch | neutral | DFDFF | FDDDD | D/F,F/D,D/D,F/D,F/D | a=fast t=deep | 5.000 | 5.994 | 11.057 | N | N | repair_subset | repair_subset |
| `risky_ps_linear` | 7 | target/post/switch | trap_like_bad | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 18.500 | 6.526 | 25.086 | N | Y | repair_all | transfer |
| `risky_ps_linear` | 8 | target/post/switch | trap_like_bad | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 16.500 | 6.162 | 22.734 | N | Y | repair_subset | transfer |
| `risky_ps_linear` | 9 | target/post/switch | neutral | DFFDD | FDDDD | D/F,F/D,F/D,D/D,D/D | a=deep t=deep | 15.000 | 5.048 | 20.121 | N | N | repair_subset | repair_subset |
| `risky_ps_linear` | 10 | target/post/switch | neutral | DFDDD | FDDDD | D/F,F/D,D/D,D/D,D/D | a=deep t=deep | 3.000 | 5.017 | 8.084 | N | N | repair_subset | repair_subset |
| `risky_ps_linear` | 11 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 0.000 | 4.898 | 4.972 | Y | Y | repair_all | repair_all |
| `risky_ps_linear` | 12 | target/post/switch | neutral | FFDDF | FDDDD | F/F,F/D,D/D,D/D,F/D | a=fast t=deep | 19.500 | 4.555 | 24.122 | N | N | repair_subset | transfer |
| `risky_ps_linear` | 13 | target/post/switch | neutral | DDDDD | FDDDD | D/F,D/D,D/D,D/D,D/D | a=deep t=deep | 20.500 | 5.511 | 26.088 | N | N | repair_subset | transfer |
| `risky_ps_linear` | 14 | target/post/switch | neutral | FFFDD | FDDDD | F/F,F/D,F/D,D/D,D/D | a=fast t=deep | 1.000 | 4.339 | 5.407 | N | Y | transfer | transfer |
| `risky_ps_linear` | 15 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 2.000 | 4.568 | 6.647 | Y | N | repair_subset | repair_subset |
| `risky_ps_linear` | 16 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 13.000 | 5.315 | 18.393 | N | N | repair_subset | repair_subset |
| `risky_ps_linear` | 17 | target/post/switch | neutral | FDDDD | FDDDD | F/F,D/D,D/D,D/D,D/D | a=deep t=deep | 11.000 | 5.334 | 16.413 | N | Y | repair_all | repair_subset |
| `risky_ps_linear` | 18 | target/post/switch | neutral | FFDDD | FDDDD | F/F,F/D,D/D,D/D,D/D | a=deep t=deep | 19.500 | 4.558 | 24.126 | N | N | repair_subset | transfer |
| `risky_ps_linear` | 19 | target/post/switch | neutral | DDDDD | FDDDD | D/F,D/D,D/D,D/D,D/D | a=deep t=deep | 21.500 | 5.920 | 27.495 | N | N | repair_subset | transfer |

## Generated Files

- `tmp/llm_v4_profilefix_archetype_diagnostic_d4_eta03_eps001/fastdeep_episode_detail.csv`: full 60-row episode detail.
- `tmp/llm_v4_profilefix_archetype_diagnostic_d4_eta03_eps001/fastdeep_group_summary.csv`: group summaries.
- `tmp/llm_v4_profilefix_archetype_diagnostic_d4_eta03_eps001/fastdeep_stage_pair_summary.csv`: stage-pair summaries.