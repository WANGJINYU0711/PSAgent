# Profile-Switch Diagnostics Report

Generated: 2026-04-28T10:58:42

## Runs
- Layer 1: `tmp/llm_v4_profilefix_archetype_diagnostic_d4_eta03_eps001`
- Layer 2: `tmp/llm_v4_profilefix_flatcost_diagnostic_d4_eta03_eps001` with `PSAGENT_PROFILE_SWITCH_FLAT_PATH_COST=1`
- Layer 3 was not run because target-good archetype did not appear in either Layer 1 or Layer 2.

## Summary Table
| layer | method | raw_total_cost_mean | post_raw_total_cost_mean | pre_trap_stage1_frac | post_trap_stage1_frac | trap_like_good_n | post_trap_like_bad_n | post_target_safe_specialist_good_n | trap_like_bad_terminal_mean | path_cost_mean | final_transfer_nontransfer_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| layer1_profilefix | direct_multistage_exp3 | 17.3706 | 21.1384 | 0.0000 | 0.2667 | 0 | 4 | 0 | 19.0000 | 0.0696 | 9 |
| layer1_profilefix | epsilon_exp3 | 15.0827 | 18.0457 | 0.4000 | 0.0667 | 2 | 1 | 0 | 20.5000 | 0.0699 | 5 |
| layer1_profilefix | risky_ps_linear | 13.8129 | 16.5222 | 0.2000 | 0.1333 | 1 | 2 | 0 | 17.5000 | 0.0712 | 6 |
| layer2_flatcost | direct_multistage_exp3 | 13.6394 | 15.9566 | 0.0000 | 0.2667 | 0 | 4 | 0 | 19.0000 | 0.0693 | 4 |
| layer2_flatcost | epsilon_exp3 | 15.0615 | 18.0651 | 0.4000 | 0.1333 | 2 | 2 | 0 | 17.5000 | 0.0693 | 7 |
| layer2_flatcost | risky_ps_linear | 16.1619 | 19.5154 | 0.2000 | 0.1333 | 1 | 2 | 0 | 18.5000 | 0.0709 | 8 |

## Key Finding
- `trap_like_bad` appears post-switch and has high terminal penalty, so the trap-bad side is now active.
- `target_safe_specialist_good` appears 0 times in both layers, so the target-good side is still inactive. This is unreasonable for the intended PS-favorable setting.
- Flat path cost changes path-cost means slightly but cannot fix the missing target-good behavior.
