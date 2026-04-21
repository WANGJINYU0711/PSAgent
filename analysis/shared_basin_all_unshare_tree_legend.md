# shared_basin_all_unshare tree legend

This is the `all_unshare` g-only variant of the current `shared_basin_strong` 4/5 tree. It reuses the same node roles, route labels, legal continuations, base costs, capability-profile semantics, and deliberation modes.

Intervention: All nodes are set to g=1; no full-share selection point exists.

Gold border marks full-share selection points (`FS`: the whole descendant child-subtree is `g=0`) and share leaves (`SL`: `g=0` leaf).

| Alias | Stage | g_variant | g_4of5 | FS subtree | Share leaf | Gold | Route lane | Role | Delib | Base cost | Next | 4/5 base agent id |
|---|---|---:|---:|---|---|---|---|---|---|---:|---|---|
| A1 | stage1 | 1 | 0 | no | no | no | public_stage1_intake | safe_core_user_grounding | deep | 0.156 | B1,B2,B3 | stage1_safe_core_user_grounding_g0_0 |
| A2 | stage1 | 1 | 0 | no | no | no | public_stage1_intake | safe_core_lookup_line | fast | 0.149 | B1,B2,B3 | stage1_safe_core_lookup_line_g0_1 |
| A3 | stage1 | 1 | 0 | no | no | no | public_stage1_intake | safe_core_context_verify | deep | 0.130 | B1,B2,B3 | stage1_safe_core_context_verify_g0_2 |
| A4 | stage1 | 1 | 0 | no | no | no | mixed_stage1_intake | mixed_shared_edge_intake | deep | 0.130 | B1,B2,B3,B4,B5 | stage1_mixed_shared_edge_intake_g0_3 |
| A5 | stage1 | 1 | 1 | no | no | no | private_barrier_stage1 | private_barrier_intake_gate | deep | 0.140 | B3,B4,B5 | stage1_private_barrier_intake_gate_g1_4 |
| B1 | stage2 | 1 | 0 | no | no | no | public_stage2_core | safe_core_account_core | deep | 0.138 | C1,C2 | stage2_safe_core_account_core_g0_0 |
| B2 | stage2 | 1 | 0 | no | no | no | public_stage2_core | safe_core_line_core | fast | 0.150 | C2,C3,C4,C5 | stage2_safe_core_line_core_g0_1 |
| B3 | stage2 | 1 | 0 | no | no | no | mixed_stage2_lane | mixed_shared_roaming_ready | deep | 0.154 | C3,C4,C5 | stage2_mixed_shared_roaming_ready_g0_2 |
| B4 | stage2 | 1 | 0 | no | no | no | private_stage2_lane | private_edge_roaming_lane | deep | 0.145 | C3,C4,C5 | stage2_private_edge_roaming_lane_g0_3 |
| B5 | stage2 | 1 | 1 | no | no | no | private_barrier_stage2 | private_barrier_roaming_gate | deep | 0.124 | C3,C4,C5 | stage2_private_barrier_roaming_gate_g1_4 |
| C1 | stage3 | 1 | 0 | no | no | no | public_stage3_core | safe_core_network_core | deep | 0.143 | D1 | stage3_safe_core_network_core_g0_0 |
| C2 | stage3 | 1 | 0 | no | no | no | public_stage3_verify | safe_core_network_verify | deep | 0.157 | D1 | stage3_safe_core_network_verify_g0_1 |
| C3 | stage3 | 1 | 0 | no | no | no | public_stage3_edge | mixed_shared_edge_diagnosis | deep | 0.124 | D2,D3 | stage3_mixed_shared_edge_diagnosis_g0_2 |
| C4 | stage3 | 1 | 0 | no | no | no | mixed_stage3_lane | private_edge_config_lane | deep | 0.147 | D2,D3,D4,D5 | stage3_private_edge_config_lane_g0_3 |
| C5 | stage3 | 1 | 1 | no | no | no | private_barrier_stage3 | private_barrier_config_gate | deep | 0.132 | D2,D3,D4,D5 | stage3_private_barrier_config_gate_g1_4 |
| D1 | stage4 | 1 | 0 | no | no | no | public_stage4_core | safe_core_repair_core | deep | 0.158 | E1,E2 | stage4_safe_core_repair_core_g0_0 |
| D2 | stage4 | 1 | 0 | no | no | no | public_stage4_verify | safe_core_repair_verify | deep | 0.120 | E1,E2 | stage4_safe_core_repair_verify_g0_1 |
| D3 | stage4 | 1 | 0 | no | no | no | mixed_stage4_lane | mixed_shared_repair_escalation | deep | 0.137 | E1,E2,E3,E4 | stage4_mixed_shared_repair_escalation_g0_2 |
| D4 | stage4 | 1 | 0 | no | no | no | private_stage4_lane | private_edge_repair_lane | deep | 0.140 | E1,E3,E4,E5 | stage4_private_edge_repair_lane_g0_3 |
| D5 | stage4 | 1 | 1 | no | no | no | private_barrier_stage4 | private_barrier_edge_repair | deep | 0.159 | E2,E4,E5 | stage4_private_barrier_edge_repair_g1_4 |
| E1 | stage5 | 1 | 0 | no | no | no | public_stage5_verify | safe_core_verify_core | deep | 0.146 | - | stage5_safe_core_verify_core_g0_0 |
| E2 | stage5 | 1 | 0 | no | no | no | public_stage5_decision | safe_core_decision_core | deep | 0.154 | - | stage5_safe_core_decision_core_g0_1 |
| E3 | stage5 | 1 | 0 | no | no | no | mixed_stage5_transfer | mixed_shared_transfer_ready | deep | 0.158 | - | stage5_mixed_shared_transfer_ready_g0_2 |
| E4 | stage5 | 1 | 0 | no | no | no | private_stage5_edge | private_edge_resolution | deep | 0.142 | - | stage5_private_edge_resolution_g0_3 |
| E5 | stage5 | 1 | 1 | no | no | no | private_stage5_leaf | private_leaf_transfer_edge | deep | 0.127 | - | stage5_private_leaf_transfer_edge_g1_4 |
