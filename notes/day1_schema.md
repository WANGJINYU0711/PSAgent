# Day 1 Schema Notes

Stage 1: user_grounding
Input:
- reason_for_call
- known_info
- task_instructions
Output:
- user_id
- target_reservation_query
- cancel_reason_label
- wants_refund
- pressure_flags

Stage 2: reservation_lookup
Input:
- stage1_output
- reservation index / snapshot
Output:
- reservation_id
- passenger_count
- cabin
- insurance
- created_at
- hours_since_booking
- any_flight_flown
- any_flight_cancelled_by_airline
- membership
- flight_statuses

Stage 3: eligibility_adjudication
Input:
- stage2_output
- codified_policy_rules
Output:
- eligible_refund
- eligible_cancel
- refusal_code
- rule_trace

Stage 4: execute_or_refuse
Input:
- stage3_output
Output:
- final_action in [cancel, refuse, transfer]
- executed_reservation_id
- user_message_template_id