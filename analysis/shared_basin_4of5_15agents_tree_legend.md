# shared_basin 4/5 scaled tree legend (15 candidates per stage)

This is a profile-clone expansion of the current `shared_basin_strong` 4/5 tree. It keeps the original route labels, legal continuation patterns, base costs, capability-profile semantics, and deliberation modes. The only structural change is candidate count per stage.

Each of the five original profiles is cloned `3` times. Each stage therefore has `12` `g=0` share nodes and `3` `g=1` unshare nodes.

| Alias | Stage | g | Base alias | Clone | FS subtree | Share leaf | Gold | Route lane | Role | Delib | Base cost | Next count | Base agent id |
|---|---|---:|---|---:|---|---|---|---|---|---|---:|---:|---|
| A01 | stage1 | 0 | A1 | 1 | no | no | no | public_stage1_intake | safe_core_user_grounding | deep | 0.156 | 9 | stage1_safe_core_user_grounding_g0_0 |
| A02 | stage1 | 0 | A1 | 2 | no | no | no | public_stage1_intake | safe_core_user_grounding | deep | 0.156 | 9 | stage1_safe_core_user_grounding_g0_0 |
| A03 | stage1 | 0 | A1 | 3 | no | no | no | public_stage1_intake | safe_core_user_grounding | deep | 0.156 | 9 | stage1_safe_core_user_grounding_g0_0 |
| A04 | stage1 | 0 | A2 | 1 | no | no | no | public_stage1_intake | safe_core_lookup_line | fast | 0.149 | 9 | stage1_safe_core_lookup_line_g0_1 |
| A05 | stage1 | 0 | A2 | 2 | no | no | no | public_stage1_intake | safe_core_lookup_line | fast | 0.149 | 9 | stage1_safe_core_lookup_line_g0_1 |
| A06 | stage1 | 0 | A2 | 3 | no | no | no | public_stage1_intake | safe_core_lookup_line | fast | 0.149 | 9 | stage1_safe_core_lookup_line_g0_1 |
| A07 | stage1 | 0 | A3 | 1 | no | no | no | public_stage1_intake | safe_core_context_verify | deep | 0.130 | 9 | stage1_safe_core_context_verify_g0_2 |
| A08 | stage1 | 0 | A3 | 2 | no | no | no | public_stage1_intake | safe_core_context_verify | deep | 0.130 | 9 | stage1_safe_core_context_verify_g0_2 |
| A09 | stage1 | 0 | A3 | 3 | no | no | no | public_stage1_intake | safe_core_context_verify | deep | 0.130 | 9 | stage1_safe_core_context_verify_g0_2 |
| A10 | stage1 | 0 | A4 | 1 | no | no | no | mixed_stage1_intake | mixed_shared_edge_intake | deep | 0.130 | 15 | stage1_mixed_shared_edge_intake_g0_3 |
| A11 | stage1 | 0 | A4 | 2 | no | no | no | mixed_stage1_intake | mixed_shared_edge_intake | deep | 0.130 | 15 | stage1_mixed_shared_edge_intake_g0_3 |
| A12 | stage1 | 0 | A4 | 3 | no | no | no | mixed_stage1_intake | mixed_shared_edge_intake | deep | 0.130 | 15 | stage1_mixed_shared_edge_intake_g0_3 |
| A13 | stage1 | 1 | A5 | 1 | no | no | no | private_barrier_stage1 | private_barrier_intake_gate | deep | 0.140 | 9 | stage1_private_barrier_intake_gate_g1_4 |
| A14 | stage1 | 1 | A5 | 2 | no | no | no | private_barrier_stage1 | private_barrier_intake_gate | deep | 0.140 | 9 | stage1_private_barrier_intake_gate_g1_4 |
| A15 | stage1 | 1 | A5 | 3 | no | no | no | private_barrier_stage1 | private_barrier_intake_gate | deep | 0.140 | 9 | stage1_private_barrier_intake_gate_g1_4 |
| B01 | stage2 | 0 | B1 | 1 | yes | no | yes | public_stage2_core | safe_core_account_core | deep | 0.138 | 6 | stage2_safe_core_account_core_g0_0 |
| B02 | stage2 | 0 | B1 | 2 | yes | no | yes | public_stage2_core | safe_core_account_core | deep | 0.138 | 6 | stage2_safe_core_account_core_g0_0 |
| B03 | stage2 | 0 | B1 | 3 | yes | no | yes | public_stage2_core | safe_core_account_core | deep | 0.138 | 6 | stage2_safe_core_account_core_g0_0 |
| B04 | stage2 | 0 | B2 | 1 | no | no | no | public_stage2_core | safe_core_line_core | fast | 0.150 | 12 | stage2_safe_core_line_core_g0_1 |
| B05 | stage2 | 0 | B2 | 2 | no | no | no | public_stage2_core | safe_core_line_core | fast | 0.150 | 12 | stage2_safe_core_line_core_g0_1 |
| B06 | stage2 | 0 | B2 | 3 | no | no | no | public_stage2_core | safe_core_line_core | fast | 0.150 | 12 | stage2_safe_core_line_core_g0_1 |
| B07 | stage2 | 0 | B3 | 1 | no | no | no | mixed_stage2_lane | mixed_shared_roaming_ready | deep | 0.154 | 9 | stage2_mixed_shared_roaming_ready_g0_2 |
| B08 | stage2 | 0 | B3 | 2 | no | no | no | mixed_stage2_lane | mixed_shared_roaming_ready | deep | 0.154 | 9 | stage2_mixed_shared_roaming_ready_g0_2 |
| B09 | stage2 | 0 | B3 | 3 | no | no | no | mixed_stage2_lane | mixed_shared_roaming_ready | deep | 0.154 | 9 | stage2_mixed_shared_roaming_ready_g0_2 |
| B10 | stage2 | 0 | B4 | 1 | no | no | no | private_stage2_lane | private_edge_roaming_lane | deep | 0.145 | 9 | stage2_private_edge_roaming_lane_g0_3 |
| B11 | stage2 | 0 | B4 | 2 | no | no | no | private_stage2_lane | private_edge_roaming_lane | deep | 0.145 | 9 | stage2_private_edge_roaming_lane_g0_3 |
| B12 | stage2 | 0 | B4 | 3 | no | no | no | private_stage2_lane | private_edge_roaming_lane | deep | 0.145 | 9 | stage2_private_edge_roaming_lane_g0_3 |
| B13 | stage2 | 1 | B5 | 1 | no | no | no | private_barrier_stage2 | private_barrier_roaming_gate | deep | 0.124 | 9 | stage2_private_barrier_roaming_gate_g1_4 |
| B14 | stage2 | 1 | B5 | 2 | no | no | no | private_barrier_stage2 | private_barrier_roaming_gate | deep | 0.124 | 9 | stage2_private_barrier_roaming_gate_g1_4 |
| B15 | stage2 | 1 | B5 | 3 | no | no | no | private_barrier_stage2 | private_barrier_roaming_gate | deep | 0.124 | 9 | stage2_private_barrier_roaming_gate_g1_4 |
| C01 | stage3 | 0 | C1 | 1 | yes | no | yes | public_stage3_core | safe_core_network_core | deep | 0.143 | 3 | stage3_safe_core_network_core_g0_0 |
| C02 | stage3 | 0 | C1 | 2 | yes | no | yes | public_stage3_core | safe_core_network_core | deep | 0.143 | 3 | stage3_safe_core_network_core_g0_0 |
| C03 | stage3 | 0 | C1 | 3 | yes | no | yes | public_stage3_core | safe_core_network_core | deep | 0.143 | 3 | stage3_safe_core_network_core_g0_0 |
| C04 | stage3 | 0 | C2 | 1 | yes | no | yes | public_stage3_verify | safe_core_network_verify | deep | 0.157 | 3 | stage3_safe_core_network_verify_g0_1 |
| C05 | stage3 | 0 | C2 | 2 | yes | no | yes | public_stage3_verify | safe_core_network_verify | deep | 0.157 | 3 | stage3_safe_core_network_verify_g0_1 |
| C06 | stage3 | 0 | C2 | 3 | yes | no | yes | public_stage3_verify | safe_core_network_verify | deep | 0.157 | 3 | stage3_safe_core_network_verify_g0_1 |
| C07 | stage3 | 0 | C3 | 1 | yes | no | yes | public_stage3_edge | mixed_shared_edge_diagnosis | deep | 0.124 | 6 | stage3_mixed_shared_edge_diagnosis_g0_2 |
| C08 | stage3 | 0 | C3 | 2 | yes | no | yes | public_stage3_edge | mixed_shared_edge_diagnosis | deep | 0.124 | 6 | stage3_mixed_shared_edge_diagnosis_g0_2 |
| C09 | stage3 | 0 | C3 | 3 | yes | no | yes | public_stage3_edge | mixed_shared_edge_diagnosis | deep | 0.124 | 6 | stage3_mixed_shared_edge_diagnosis_g0_2 |
| C10 | stage3 | 0 | C4 | 1 | no | no | no | mixed_stage3_lane | private_edge_config_lane | deep | 0.147 | 12 | stage3_private_edge_config_lane_g0_3 |
| C11 | stage3 | 0 | C4 | 2 | no | no | no | mixed_stage3_lane | private_edge_config_lane | deep | 0.147 | 12 | stage3_private_edge_config_lane_g0_3 |
| C12 | stage3 | 0 | C4 | 3 | no | no | no | mixed_stage3_lane | private_edge_config_lane | deep | 0.147 | 12 | stage3_private_edge_config_lane_g0_3 |
| C13 | stage3 | 1 | C5 | 1 | no | no | no | private_barrier_stage3 | private_barrier_config_gate | deep | 0.132 | 12 | stage3_private_barrier_config_gate_g1_4 |
| C14 | stage3 | 1 | C5 | 2 | no | no | no | private_barrier_stage3 | private_barrier_config_gate | deep | 0.132 | 12 | stage3_private_barrier_config_gate_g1_4 |
| C15 | stage3 | 1 | C5 | 3 | no | no | no | private_barrier_stage3 | private_barrier_config_gate | deep | 0.132 | 12 | stage3_private_barrier_config_gate_g1_4 |
| D01 | stage4 | 0 | D1 | 1 | yes | no | yes | public_stage4_core | safe_core_repair_core | deep | 0.158 | 6 | stage4_safe_core_repair_core_g0_0 |
| D02 | stage4 | 0 | D1 | 2 | yes | no | yes | public_stage4_core | safe_core_repair_core | deep | 0.158 | 6 | stage4_safe_core_repair_core_g0_0 |
| D03 | stage4 | 0 | D1 | 3 | yes | no | yes | public_stage4_core | safe_core_repair_core | deep | 0.158 | 6 | stage4_safe_core_repair_core_g0_0 |
| D04 | stage4 | 0 | D2 | 1 | yes | no | yes | public_stage4_verify | safe_core_repair_verify | deep | 0.120 | 6 | stage4_safe_core_repair_verify_g0_1 |
| D05 | stage4 | 0 | D2 | 2 | yes | no | yes | public_stage4_verify | safe_core_repair_verify | deep | 0.120 | 6 | stage4_safe_core_repair_verify_g0_1 |
| D06 | stage4 | 0 | D2 | 3 | yes | no | yes | public_stage4_verify | safe_core_repair_verify | deep | 0.120 | 6 | stage4_safe_core_repair_verify_g0_1 |
| D07 | stage4 | 0 | D3 | 1 | yes | no | yes | mixed_stage4_lane | mixed_shared_repair_escalation | deep | 0.137 | 12 | stage4_mixed_shared_repair_escalation_g0_2 |
| D08 | stage4 | 0 | D3 | 2 | yes | no | yes | mixed_stage4_lane | mixed_shared_repair_escalation | deep | 0.137 | 12 | stage4_mixed_shared_repair_escalation_g0_2 |
| D09 | stage4 | 0 | D3 | 3 | yes | no | yes | mixed_stage4_lane | mixed_shared_repair_escalation | deep | 0.137 | 12 | stage4_mixed_shared_repair_escalation_g0_2 |
| D10 | stage4 | 0 | D4 | 1 | no | no | no | private_stage4_lane | private_edge_repair_lane | deep | 0.140 | 12 | stage4_private_edge_repair_lane_g0_3 |
| D11 | stage4 | 0 | D4 | 2 | no | no | no | private_stage4_lane | private_edge_repair_lane | deep | 0.140 | 12 | stage4_private_edge_repair_lane_g0_3 |
| D12 | stage4 | 0 | D4 | 3 | no | no | no | private_stage4_lane | private_edge_repair_lane | deep | 0.140 | 12 | stage4_private_edge_repair_lane_g0_3 |
| D13 | stage4 | 1 | D5 | 1 | no | no | no | private_barrier_stage4 | private_barrier_edge_repair | deep | 0.159 | 9 | stage4_private_barrier_edge_repair_g1_4 |
| D14 | stage4 | 1 | D5 | 2 | no | no | no | private_barrier_stage4 | private_barrier_edge_repair | deep | 0.159 | 9 | stage4_private_barrier_edge_repair_g1_4 |
| D15 | stage4 | 1 | D5 | 3 | no | no | no | private_barrier_stage4 | private_barrier_edge_repair | deep | 0.159 | 9 | stage4_private_barrier_edge_repair_g1_4 |
| E01 | stage5 | 0 | E1 | 1 | no | yes | yes | public_stage5_verify | safe_core_verify_core | deep | 0.146 | 0 | stage5_safe_core_verify_core_g0_0 |
| E02 | stage5 | 0 | E1 | 2 | no | yes | yes | public_stage5_verify | safe_core_verify_core | deep | 0.146 | 0 | stage5_safe_core_verify_core_g0_0 |
| E03 | stage5 | 0 | E1 | 3 | no | yes | yes | public_stage5_verify | safe_core_verify_core | deep | 0.146 | 0 | stage5_safe_core_verify_core_g0_0 |
| E04 | stage5 | 0 | E2 | 1 | no | yes | yes | public_stage5_decision | safe_core_decision_core | deep | 0.154 | 0 | stage5_safe_core_decision_core_g0_1 |
| E05 | stage5 | 0 | E2 | 2 | no | yes | yes | public_stage5_decision | safe_core_decision_core | deep | 0.154 | 0 | stage5_safe_core_decision_core_g0_1 |
| E06 | stage5 | 0 | E2 | 3 | no | yes | yes | public_stage5_decision | safe_core_decision_core | deep | 0.154 | 0 | stage5_safe_core_decision_core_g0_1 |
| E07 | stage5 | 0 | E3 | 1 | no | yes | yes | mixed_stage5_transfer | mixed_shared_transfer_ready | deep | 0.158 | 0 | stage5_mixed_shared_transfer_ready_g0_2 |
| E08 | stage5 | 0 | E3 | 2 | no | yes | yes | mixed_stage5_transfer | mixed_shared_transfer_ready | deep | 0.158 | 0 | stage5_mixed_shared_transfer_ready_g0_2 |
| E09 | stage5 | 0 | E3 | 3 | no | yes | yes | mixed_stage5_transfer | mixed_shared_transfer_ready | deep | 0.158 | 0 | stage5_mixed_shared_transfer_ready_g0_2 |
| E10 | stage5 | 0 | E4 | 1 | no | yes | yes | private_stage5_edge | private_edge_resolution | deep | 0.142 | 0 | stage5_private_edge_resolution_g0_3 |
| E11 | stage5 | 0 | E4 | 2 | no | yes | yes | private_stage5_edge | private_edge_resolution | deep | 0.142 | 0 | stage5_private_edge_resolution_g0_3 |
| E12 | stage5 | 0 | E4 | 3 | no | yes | yes | private_stage5_edge | private_edge_resolution | deep | 0.142 | 0 | stage5_private_edge_resolution_g0_3 |
| E13 | stage5 | 1 | E5 | 1 | no | no | no | private_stage5_leaf | private_leaf_transfer_edge | deep | 0.127 | 0 | stage5_private_leaf_transfer_edge_g1_4 |
| E14 | stage5 | 1 | E5 | 2 | no | no | no | private_stage5_leaf | private_leaf_transfer_edge | deep | 0.127 | 0 | stage5_private_leaf_transfer_edge_g1_4 |
| E15 | stage5 | 1 | E5 | 3 | no | no | no | private_stage5_leaf | private_leaf_transfer_edge | deep | 0.127 | 0 | stage5_private_leaf_transfer_edge_g1_4 |
