# Airline Cancellation Fixed-Tree Schema

## Generated subset

- Source: `/home/ubuntu/data/PSAgent/tau2-bench/data/tau2/domains/airline/tasks.json`
- Derived subset: `/home/ubuntu/data/PSAgent/data/derived/airline_cancellation_fixed_tree/tasks.json`
- Included task ids:
  - `0`
  - `1`
  - `26`
  - `28`
  - `39`
  - `41`
  - `43`
  - `47`
  - `48`
  - `49`

## Inclusion rule

This subset keeps only tasks whose main difficulty is:

- identifying the target reservation or reservation set
- checking cancellation or refund eligibility under airline policy
- refusing cancellation when policy conditions are not met
- cancelling only the policy-eligible reservations when a mixed set is presented

This subset excludes tasks whose main workflow is instead:

- compensation or certificate issuance
- booking or rebooking
- flight modification or cabin changes
- baggage or passenger updates
- insurance-only refund requests
- mixed multi-intent transactions where cancellation is only a side branch

## Why these 10 tasks form a coherent family

Across the selected tasks, the recurring structure is:

1. The user expresses an intent to cancel a reservation and usually wants a refund.
2. The system must identify the relevant reservation or candidate reservations.
3. The system must infer policy-relevant facts:
   - booking age
   - whether any leg has already been flown
   - whether any leg was cancelled by the airline
   - cabin class
   - travel insurance presence
   - stated cancellation reason
4. The system must adjudicate whether cancellation with refund is allowed.
5. The system must either:
   - refuse cancellation
   - cancel only the eligible reservation(s)

The family is therefore much closer to a fixed-stage adjudication pipeline than the full airline benchmark.

## Observed task patterns

### Pattern A: Single-reservation refusal

Representative tasks:

- `0`
- `1`
- `26`
- `28`
- `47`
- `48`
- `49`

Common properties:

- one main reservation target
- user wants cancellation plus refund
- user often lies, pressures, or insists
- correct behavior is usually refusal

### Pattern B: Multi-reservation filtering

Representative tasks:

- `39`
- `41`

Common properties:

- user asks to cancel multiple future reservations
- only some reservations may be eligible
- system must inspect several reservations and filter them by policy

### Pattern C: Ambiguous target among candidate reservations

Representative task:

- `43`

Common properties:

- user describes a trip rather than giving the final valid reservation id upfront
- system must identify candidate reservations and reject invalid cancellation choices

## Recommended fixed-tree schema

The schema below is tuned to the selected subset, not to the whole airline benchmark.

### Stage 1: Intent grounding

Purpose:

- convert user language into structured cancellation intent

Input:

- `reason_for_call`
- `known_info`
- `task_instructions`

Output:

- `user_id`
- `cancel_intent = true`
- `wants_refund`
- `stated_cancel_reason`
- `target_spec`
- `pressure_signals`

Notes:

- `target_spec` may be:
  - explicit reservation id
  - route/date description
  - "all future reservations"
  - "all one-passenger reservations"

- `pressure_signals` should capture:
  - false approval claim
  - false insurance claim
  - false booking-time claim
  - emotional pressure
  - persistence / coercion

### Stage 2: Reservation resolution

Purpose:

- map the grounded user target to one or more concrete reservation candidates

Input:

- `stage1_output`
- reservation index for the user

Output:

- `candidate_reservations`
- `resolved_reservations`
- `resolution_status`

Notes:

- For single-id tasks, this is mostly direct lookup.
- For tasks like `39`, `41`, and `43`, this stage is essential because the user target is a set or partial description.

### Stage 3: Eligibility feature extraction

Purpose:

- compute policy-relevant facts for each resolved reservation

Input:

- `resolved_reservations`
- flight database snapshot
- policy reference time

Output per reservation:

- `reservation_id`
- `hours_since_booking`
- `insurance`
- `cabin`
- `membership`
- `passenger_count`
- `any_leg_flown`
- `any_leg_cancelled_by_airline`
- `stated_reason_supported_by_insurance`
- `eligible_by_24h_rule`
- `eligible_by_airline_cancel_rule`
- `eligible_by_business_rule`
- `eligible_by_insurance_rule`

Notes:

- This stage should be deterministic and structured.
- It is the cleanest place to separate policy facts from final decision logic.

### Stage 4: Eligibility adjudication

Purpose:

- decide cancellation/refund eligibility from extracted features

Input:

- `stage3_output`

Output per reservation:

- `eligible_cancel_with_refund`
- `adjudication_label`
- `refusal_code`
- `rule_trace`

Recommended adjudication labels:

- `allow_cancel_refund`
- `deny_outside_24h`
- `deny_flight_already_flown`
- `deny_basic_economy_no_exception`
- `deny_insurance_reason_not_covered`
- `deny_false_claim_conflict`

### Stage 5: Execute or refuse

Purpose:

- convert adjudication into the benchmark terminal action

Input:

- `stage4_output`

Output:

- `final_action`
- `cancelled_reservation_ids`
- `refused_reservation_ids`
- `response_mode`

Allowed `final_action` shapes:

- `refuse_all`
- `cancel_subset`
- `cancel_all`

For this subset, `transfer` should not be the default terminal action and can be omitted from V1.

## Fixed-tree interpretation

For the selected subset, a strong fixed-tree abstraction is:

1. `user_grounding`
2. `reservation_resolution`
3. `eligibility_feature_extraction`
4. `eligibility_adjudication`
5. `execute_or_refuse`

This is slightly more detailed than the Day 1 draft because the original tasks show that:

- resolving the target reservation is a distinct source of difficulty
- extracting policy facts is different from adjudicating policy
- multi-reservation tasks need per-reservation reasoning before the final action

## Suggested structured sample interface

Each derived example should eventually contain:

```json
{
  "family": "airline_cancellation_refund_eligibility",
  "original_task_id": "39",
  "stage1": {
    "input": {},
    "oracle_output": {}
  },
  "stage2": {
    "input": {},
    "oracle_output": {}
  },
  "stage3": {
    "input": {},
    "oracle_output": {}
  },
  "stage4": {
    "input": {},
    "oracle_output": {}
  },
  "stage5": {
    "input": {},
    "oracle_output": {}
  }
}
```

## Notes for later derived benchmark construction

- `39` is useful because it contains both eligible and ineligible reservations in one request.
- `41` is useful because it looks similar to `39` at the language level but should end with no cancellations.
- `43` is useful because target resolution itself is ambiguous.
- `0`, `1`, `47`, `48`, `49` are good single-reservation refusal prototypes with different false-claim types.
- `26` and `28` are useful simple refusal tasks with minimal extra structure.

## Recommended next builder step

Start with two tiers:

- Tier 1 clean single-reservation tasks:
  - `0`, `1`, `26`, `28`, `47`, `48`, `49`

- Tier 2 harder set-resolution tasks:
  - `39`, `41`, `43`

This split keeps the first derived benchmark simple while preserving a natural path to a harder evaluation set.
