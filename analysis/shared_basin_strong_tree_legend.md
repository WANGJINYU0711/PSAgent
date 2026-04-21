# shared_basin_strong tree legend

Gold border marks full-share selection points (`FS`: the whole descendant child-subtree is `g=0`) and share leaves (`SL`: `g=0` leaf).

| Alias | Stage | g | FS subtree | Share leaf | Gold | Route lane | Role | Safe count | Mixed count | Next |
|---|---|---:|---|---|---|---|---|---:|---:|---|
| A1 | stage1 | 0 | no | no | no | public_stage1_intake | safe_core_user_grounding | 0 | 1 | B1,B2,B3 |
| A2 | stage1 | 0 | no | no | no | public_stage1_intake | safe_core_lookup_line | 0 | 1 | B1,B2,B3 |
| A3 | stage1 | 0 | no | no | no | public_stage1_intake | safe_core_context_verify | 0 | 1 | B1,B2,B3 |
| A4 | stage1 | 0 | no | no | no | mixed_stage1_intake | mixed_shared_edge_intake | 0 | 1 | B1,B2,B3,B4,B5 |
| A5 | stage1 | 1 | no | no | no | private_barrier_stage1 | private_barrier_intake_gate | 0 | 0 | B3,B4,B5 |
| B1 | stage2 | 0 | yes | no | yes | public_stage2_core | safe_core_account_core | 4 | 0 | C1,C2 |
| B2 | stage2 | 0 | no | no | no | public_stage2_core | safe_core_line_core | 0 | 4 | C2,C3,C4,C5 |
| B3 | stage2 | 0 | no | no | no | mixed_stage2_lane | mixed_shared_roaming_ready | 0 | 5 | C3,C4,C5 |
| B4 | stage2 | 0 | no | no | no | private_stage2_lane | private_edge_roaming_lane | 0 | 2 | C3,C4,C5 |
| B5 | stage2 | 1 | no | no | no | private_barrier_stage2 | private_barrier_roaming_gate | 0 | 2 | C3,C4,C5 |
| C1 | stage3 | 0 | yes | no | yes | public_stage3_core | safe_core_network_core | 4 | 0 | D1 |
| C2 | stage3 | 0 | yes | no | yes | public_stage3_verify | safe_core_network_verify | 8 | 0 | D1 |
| C3 | stage3 | 0 | yes | no | yes | public_stage3_edge | mixed_shared_edge_diagnosis | 13 | 0 | D2,D3 |
| C4 | stage3 | 0 | no | no | no | mixed_stage3_lane | private_edge_config_lane | 0 | 13 | D2,D3,D4,D5 |
| C5 | stage3 | 1 | no | no | no | private_barrier_stage3 | private_barrier_config_gate | 0 | 13 | D2,D3,D4,D5 |
| D1 | stage4 | 0 | yes | no | yes | public_stage4_core | safe_core_repair_core | 12 | 0 | E1,E2 |
| D2 | stage4 | 0 | yes | no | yes | public_stage4_verify | safe_core_repair_verify | 39 | 0 | E1,E2 |
| D3 | stage4 | 0 | yes | no | yes | mixed_stage4_lane | mixed_shared_repair_escalation | 39 | 0 | E1,E2,E3,E4 |
| D4 | stage4 | 0 | no | no | no | private_stage4_lane | private_edge_repair_lane | 0 | 26 | E1,E3,E4,E5 |
| D5 | stage4 | 1 | no | no | no | private_barrier_stage4 | private_barrier_edge_repair | 0 | 26 | E2,E4,E5 |
| E1 | stage5 | 0 | no | yes | yes | public_stage5_verify | safe_core_verify_core | 0 | 0 | - |
| E2 | stage5 | 0 | no | yes | yes | public_stage5_decision | safe_core_decision_core | 0 | 0 | - |
| E3 | stage5 | 0 | no | yes | yes | mixed_stage5_transfer | mixed_shared_transfer_ready | 0 | 0 | - |
| E4 | stage5 | 0 | no | yes | yes | private_stage5_edge | private_edge_resolution | 0 | 0 | - |
| E5 | stage5 | 1 | no | no | no | private_stage5_leaf | private_leaf_transfer_edge | 0 | 0 | - |
