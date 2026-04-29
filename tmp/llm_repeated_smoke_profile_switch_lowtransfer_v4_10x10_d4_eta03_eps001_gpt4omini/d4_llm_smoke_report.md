# LLM Smoke Report: d4_eta03_eps001_gpt4omini
Generated: 2026-04-28T02:46:27
## Setting
| field | value |
| --- | --- |
| setting name | `llm_lowtransfer_v4_d4_eta03_eps001_gpt4omini` |
| executor setting | `telecom_mms_agent_profile_only_clean_v4_hard_transfer_contract` |
| model | `gpt-4o-mini` |
| dataset | `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch_lowtransfer_smoke/tasks.json` |
| schedule buckets | `analysis/shared_basin_prefix_dedup_profile_switch_lowtransfer_smoke_schedule_buckets.json` |
| repeats / horizon | `10` / `100` |
| schedule mode | `trap_switch` |
| d / switch denominator | `4` |
| switch episode | `25` |
| eta | `0.3` |
| epsilon | `0.01` |
| methods | 13, method-level parallel |

## Main Findings
- Overall winner by `raw_total_cost_mean`: `epsilon_exp3` = 12.3156.
- Best PS overall: `risky_ps_linear` = 13.0068, rank 3.
- Post-switch winner by `post_raw_total_cost_mean`: `epsilon_exp3` = 14.2677.
- Best PS post-switch: `risky_ps_linear` = 15.2674, rank 3.
- Specialist winner by `specialist_raw_total_cost_mean`: `epsilon_exp3` = 15.0780.
- Best PS specialist: `risky_ps_linear` = 16.1289, rank 3.
- Transfer-control episodes: every method had `transfer_final_transfer_rate=1.0` on the transfer-oracle episodes in this run.

## Extended Metrics
| method | family | raw_total_cost_rank | raw_total_cost_mean | post_raw_total_cost_rank | post_raw_total_cost_mean | specialist_raw_total_cost_rank | specialist_raw_total_cost_mean | exact_match_mean | post_exact_match_mean | transfer_final_transfer_rate | mean_total_tokens |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epsilon_exp3 | baseline | 1 | 12.3156 | 1 | 14.2677 | 1 | 15.0780 | 0.3000 | 0.2667 | 1.0000 | 50013.4100 |
| direct_multistage_exp3_local | baseline | 2 | 12.8979 | 2 | 15.1349 | 2 | 16.0757 | 0.3000 | 0.2667 | 1.0000 | 48598.7900 |
| risky_ps_linear | PS | 3 | 13.0068 | 3 | 15.2674 | 3 | 16.1289 | 0.2200 | 0.1600 | 1.0000 | 47420.6200 |
| direct_multistage_exp3 | baseline | 4 | 13.0671 | 4 | 15.3348 | 4 | 16.2368 | 0.2500 | 0.2000 | 1.0000 | 49265.1400 |
| risky_ps_ix | PS | 5 | 13.1873 | 5 | 15.4425 | 5 | 16.3617 | 0.2200 | 0.1600 | 1.0000 | 48700.0900 |
| risky_ps_old | PS | 6 | 13.6542 | 6 | 16.0461 | 6 | 17.0059 | 0.2300 | 0.1733 | 1.0000 | 49379.3600 |
| risky_ps_safe_conditional_ix | PS | 7 | 13.8467 | 8 | 16.2538 | 7 | 17.1979 | 0.2300 | 0.1733 | 1.0000 | 49429.0900 |
| naive_mixed | baseline | 8 | 13.8697 | 9 | 16.3120 | 9 | 17.3222 | 0.1900 | 0.1200 | 1.0000 | 49723.2600 |
| naive_mixed_avg | baseline | 9 | 13.8775 | 7 | 16.2391 | 8 | 17.2369 | 0.2200 | 0.1600 | 1.0000 | 51434.7800 |
| risky_ps | PS | 10 | 13.9754 | 10 | 16.4247 | 10 | 17.4488 | 0.2200 | 0.1600 | 1.0000 | 50322.7100 |
| risky_ps_safe_conditional | PS | 11 | 14.0612 | 11 | 16.5303 | 11 | 17.5654 | 0.1800 | 0.1067 | 1.0000 | 47696.5300 |
| risky_ps_direct_cost | PS | 12 | 14.1724 | 12 | 16.8391 | 12 | 17.8718 | 0.1900 | 0.1200 | 1.0000 | 49624.7400 |
| random_path | baseline | 13 | 14.3410 | 13 | 17.0023 | 13 | 18.0452 | 0.2000 | 0.1333 | 1.0000 | 49963.0100 |

## Delta vs Previous v4 n7_eta02_eps001
| method | family | old_raw_rank | new_raw_rank | old_raw_total_cost_mean | new_raw_total_cost_mean | delta_raw_total_cost_mean | old_post_raw_total_cost_mean | new_post_raw_total_cost_mean | delta_post_raw_total_cost_mean | old_specialist_raw_total_cost_mean | new_specialist_raw_total_cost_mean | delta_specialist_raw_total_cost_mean | old_exact_match_mean | new_exact_match_mean | delta_exact_match_mean |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| epsilon_exp3 | baseline | 5 | 1 | 14.9734 | 12.3156 | -2.6578 | 16.3138 | 14.2677 | -2.0461 | 17.3458 | 15.0780 | -2.2678 | 0.1800 | 0.3000 | 0.1200 |
| direct_multistage_exp3_local | baseline | 7 | 2 | 15.3111 | 12.8979 | -2.4132 | 16.8184 | 15.1349 | -1.6835 | 17.9139 | 16.0757 | -1.8383 | 0.1300 | 0.3000 | 0.1700 |
| risky_ps_linear | PS | 13 | 3 | 15.8874 | 13.0068 | -2.8805 | 17.4458 | 15.2674 | -2.1784 | 18.6319 | 16.1289 | -2.5030 | 0.1400 | 0.2200 | 0.0800 |
| direct_multistage_exp3 | baseline | 1 | 4 | 14.1879 | 13.0671 | -1.1207 | 15.4850 | 15.3348 | -0.1502 | 16.3946 | 16.2368 | -0.1578 | 0.2500 | 0.2500 | 0.0000 |
| risky_ps_ix | PS | 3 | 5 | 14.8103 | 13.1873 | -1.6230 | 16.1713 | 15.4425 | -0.7288 | 17.1906 | 16.3617 | -0.8290 | 0.2400 | 0.2200 | -0.0200 |
| risky_ps_old | PS | 4 | 6 | 14.8871 | 13.6542 | -1.2330 | 16.2743 | 16.0461 | -0.2282 | 17.2584 | 17.0059 | -0.2525 | 0.2000 | 0.2300 | 0.0300 |
| risky_ps_safe_conditional_ix | PS | 8 | 7 | 15.3656 | 13.8467 | -1.5190 | 16.8247 | 16.2538 | -0.5709 | 17.8563 | 17.1979 | -0.6584 | 0.2300 | 0.2300 | 0.0000 |
| naive_mixed | baseline | 11 | 8 | 15.6387 | 13.8697 | -1.7691 | 17.0641 | 16.3120 | -0.7521 | 18.1473 | 17.3222 | -0.8251 | 0.1600 | 0.1900 | 0.0300 |
| naive_mixed_avg | baseline | 10 | 9 | 15.5769 | 13.8775 | -1.6994 | 16.9623 | 16.2391 | -0.7232 | 18.0837 | 17.2369 | -0.8468 | 0.2000 | 0.2200 | 0.0200 |
| risky_ps | PS | 12 | 10 | 15.8860 | 13.9754 | -1.9106 | 17.4297 | 16.4247 | -1.0049 | 18.6357 | 17.4488 | -1.1870 | 0.2000 | 0.2200 | 0.0200 |
| risky_ps_safe_conditional | PS | 2 | 11 | 14.6729 | 14.0612 | -0.6117 | 16.0697 | 16.5303 | 0.4606 | 17.0646 | 17.5654 | 0.5007 | 0.2000 | 0.1800 | -0.0200 |
| risky_ps_direct_cost | PS | 6 | 12 | 15.1034 | 14.1724 | -0.9311 | 16.5154 | 16.8391 | 0.3237 | 17.5259 | 17.8718 | 0.3459 | 0.1900 | 0.1900 | 0.0000 |
| random_path | baseline | 9 | 13 | 15.5394 | 14.3410 | -1.1984 | 17.0134 | 17.0023 | -0.0111 | 18.1310 | 18.0452 | -0.0858 | 0.2100 | 0.2000 | -0.0100 |

## Files
- `repeated_smoke_compare.csv/json/md`: runner overall table.
- `d4_extended_metrics.csv/md`: overall + post-switch + specialist + transfer/token metrics.
- `d4_vs_n7_delta.csv/md`: comparison against previous v4 n7 eta0.2 eps0.01 smoke.
