# llm_v8_trapasymv3efficientanchor4of5_stage45_contract_promptv11b_stage45_gpt41mini_cconfig_d3_activegap_reason55_eps001_10x10_4methods_seed0 monitor report

- decision: `continue`
- reason: PS is not clearly behind at the episode threshold
- switch_episode: `33`

## Last-5 Pre-Switch Trap Probabilities

| method | n | root trap | stage4 trap | all-fast trap route | selected trap-like | selected all-fast |
|---|---:|---:|---:|---:|---:|---:|
| risky_ps | 5 | 0.599 | 0.986 | 0.591 | 1.000 | 1.000 |
| direct_multistage_exp3 | 5 | 0.531 | 0.551 | 0.292 | 0.400 | 0.400 |
| epsilon_exp3 | 5 | 0.437 | 0.913 | 0.399 | 0.400 | 0.400 |
| risky_ps_old | 5 | 0.588 | 0.991 | 0.583 | 1.000 | 1.000 |

## Post-Switch Means

| method | post n | total mean | raw mean | terminal mean |
|---|---:|---:|---:|---:|
| risky_ps | 43 | 0.3835 | 14.243 | 5.070 |
| direct_multistage_exp3 | 42 | 0.4269 | 15.854 | 5.667 |
| epsilon_exp3 | 43 | 0.4283 | 15.906 | 6.372 |
