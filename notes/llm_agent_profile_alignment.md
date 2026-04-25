# LLM Agent Profile Alignment Notes

## Core Goal

The primary objective is to make the LLM executor behave so that paths with better agent profiles for the current task naturally produce lower final cost.

In this context, "better agent profile" means alignment between:

- task stage capability requirements
- agent `attribute_skill`
- task stage deliberation requirements
- agent `deliberation_mode`

The desired signal is not only lower token/tool usage. The important outcome is lower terminal penalty / total cost for paths that are stronger according to the intended profile.

## Current LLM Diagnostic Setup

Recent diagnostics use:

- family: `shared_basin_strong_prefix_dedup_profile_switch`
- dataset: `data/derived/telecom_mms_fixed_tree_base_v2_100_capabilities_time_profile_switch/tasks.json`
- script: `scripts/run_llm_path_sweep_diagnostic.py`
- model: `gpt-4o-mini`
- sample: 4 tasks x 5 representative paths

This is not the old controlled-sim SVG/tree with `T/8` trap switch. It is the new profile-switch LLM family, with trap/target behavior coming from task profiles rather than a time switch.

## Version Results

### v1: `/tmp/llm_path_sweep_diagnostic`

- `tasks_with_action_diversity`: 1/4
- `mean_terminal_penalty_spread`: 2.875
- positive terminal spread: 3/4 tasks
- `offline_match_vs_terminal_corr_mean`: -0.385260

Interpretation:

- Resource usage was separated.
- Some behavior separation appeared.
- Offline path match did not predict terminal quality.
- Target/specialist tasks often still favored lower offline-match paths.

### v2: `/tmp/llm_path_sweep_diagnostic_v2`

- `tasks_with_action_diversity`: 0/4
- `mean_terminal_penalty_spread`: 0.75
- positive terminal spread: 2/4 tasks
- `offline_match_vs_terminal_corr_mean`: -0.134388

Interpretation:

- Stage3/4/5 hardening was too conservative at stage5.
- Many paths collapsed into `transfer`.
- Apparent correlation improvement was partly caused by loss of terminal differentiation.

### v3: `/tmp/llm_path_sweep_diagnostic_v3`

- `tasks_with_action_diversity`: 1/4
- `mean_terminal_penalty_spread`: 1.5
- positive terminal spread: 3/4 tasks
- `offline_match_vs_terminal_corr_mean`: -0.211925

Interpretation:

- v3 improved over v2 by reducing the default transfer bias.
- v3 is still worse than v1 on terminal spread and still does not meet the core goal.
- Target and specialist tasks still collapse to `transfer`.
- The remaining bottleneck is likely upstream stage4 repairability / `should_repair` planning for APN, roaming, SIM, and specialist blockers.

## Key Failure Modes

1. Stage5 transfer collapse

   Overly conservative terminal rules plus normalizer fallback can flatten path differences into `transfer`.

2. Stage4 under-repair for specialist blockers

   If stage4 marks APN/roaming/SIM-style blockers as deferred or transfer-required, stage5 has little room to express target-specialist strength.

3. Offline path match is not yet predictive

   The current static score measures intended capability alignment, but the LLM executor does not reliably convert that alignment into better terminal outputs.

4. Resource separation is not enough

   `max_rounds` and prompt contract changes clearly separate token/tool use, but final cost alignment requires profile-dependent decisions, not just different search budgets.

## Modification Principles

- Keep stage3 blocker-family elimination guardrails.
- Keep stage4 repair precondition guardrails, but make specialist/deep agents more capable of repairing when evidence is sufficient.
- Avoid making stage5 globally conservative.
- Deep agents should be strict about verification, then decisive after evidence is verified.
- Fast agents should narrow scope and be conservative when evidence is insufficient.
- Normalizer fallback should not default to `transfer` when stage4 says blockers are repairable.
- Do not change cost aggregation, evaluator, family, or dataset until executor behavior is better aligned.

## Next Likely Work

Focus on stage4 before stage5:

- inspect stage4 outputs in v3 for target/specialist tasks
- compare `repairability`, `per_blocker.should_repair`, and `transfer_reason` across representative paths
- adjust stage4 prompt so target/deep APN/roaming/repair specialists can mark specialist blockers as repairable when evidence supports it
- rerun the same 4 x 5 diagnostic and compare against v1/v2/v3
