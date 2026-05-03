# BarrierShare Controlled Sim Package

This folder is a standalone copy of the main non-LLM simulation code, the
tree JSONs, and the result sets discussed with the advisor:

1. The screenshot/table run:
   `outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_epsilon_sweep/`
2. The v10 nonzero-epsilon sweep:
   `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/`
3. The follow-up three-layer top eta/eps ablation summaries:
   `outputs/sim_three_layer_topetaeps_ablation_v1/`

Run commands below from this `sim/` directory.

## Contents

- `scripts/run_barriershare_controlled_sim.py`
  - Main controlled-simulation runner.
  - No LLM/API calls.
  - Key knobs: `--common-eta-override`, `--common-epsilon-override`,
    `--trap-switch-denominator`, `--tree-spec-cost-mode`.
- `tmp/run_sim_v10_nonzero_eps_d_eta_eps_sweep.py`
  - Aggregated d/eta/epsilon sweep runner for the v10 experiment.
- `tmp/run_sim_three_layer_topetaeps_ablation.py`
  - Follow-up top eta/eps ablation runner for v10/v11/v12 cost landscapes.
- `scripts/build_profile_switch_trap_asym_tree.py`
- `scripts/build_profile_switch_trap_asym_v2_neutral_tree.py`
- `scripts/build_profile_switch_trap_asym_v3_efficient_anchor_tree.py`
  - Builder scripts for the newer trap-asymmetric tree JSONs.
- `scripts/run_barriershare_controlled_bestof_sweep.py`
  - Best-of hyperparameter sweep helper.
- `baselines/`
  - PS and baseline policies.
- `envs/`
  - Minimal environment, tree, executor, and adapter code needed by the runner.
- `analysis/tree_specs/`
  - Tree specs used by the sim runs.
  - The original v10 sweep used
    `shared_basin_strong_4of5_prefix_dedup.json`.
  - The latest/current full-run tree setting is included as
    `shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json`.
  - Earlier trap-asym variants v1 and v2 are included for comparison.
- `analysis/*_validation.json`
  - Validation metadata used by the runner summaries. These are intentionally
    present both next to the tree specs and under `analysis/`, because the
    runner looks for validation files under `analysis/` when `--tree-spec` is
    under `analysis/tree_specs/`.
- `outputs/`
  - Compact result artifacts and summaries. The 4.5GB v10 per-combo raw `runs/`
    directory is intentionally not included; the sweep can regenerate it.

## Python

This package uses only the Python standard library for the controlled sim path.
Python 3.10+ is recommended.

Quick import/CLI check:

```bash
python scripts/run_barriershare_controlled_sim.py --help
python tmp/run_sim_v10_nonzero_eps_d_eta_eps_sweep.py --help
```

## Reproduce Screenshot Row

The screenshot row corresponds to:

- cost mode: `ps_favored_trap`
- tree: `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json`
- horizon: `1000`
- seeds: `0 1 2 3 4`
- eta: `0.01`
- epsilon: `0.02`
- best PS: `risky_ps`

Run:

```bash
python scripts/run_barriershare_controlled_sim.py \
  --output-dir outputs/reproduce_screenshot_eta001_eps002 \
  --tree-spec analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json \
  --tree-spec-role-mode spec_or_agent_id \
  --tree-spec-cost-mode ps_favored_trap \
  --horizon 1000 \
  --seeds 0 1 2 3 4 \
  --cost-noise 0.02 \
  --specialist-fraction 0.15 \
  --common-eta-override 0.01 \
  --common-epsilon-override 0.02 \
  --methods \
    risky_ps_old risky_ps risky_ps_ix \
    risky_ps_safe_conditional risky_ps_safe_conditional_ix risky_ps_direct_cost \
    epsilon_exp3 direct_multistage_exp3 naive_mixed random_path
```

Expected summary files:

- `outputs/reproduce_screenshot_eta001_eps002/controlled_sim_compare.csv`
- `outputs/reproduce_screenshot_eta001_eps002/ps_favored_trap_summary.md`

The original compact screenshot summary is already included at:

```text
outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_epsilon_sweep/eta_epsilon_sweep_summary.md
```

The already-run single config for `eta=0.01, eps=0.02` is included at:

```text
outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_0.01_eps_0.02/
```

## Reproduce v10 Nonzero-Epsilon Sweep

The original v10 sweep below uses the older prefix-dedup tree:

```text
analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json
```

Full sweep:

```bash
python tmp/run_sim_v10_nonzero_eps_d_eta_eps_sweep.py \
  --output-dir outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1_rerun \
  --tree-spec analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json \
  --d-values 5,6,7,8,10,12,16 \
  --eta-values 0.05,0.1,0.15,0.2,0.3,0.4,0.5 \
  --epsilon-values 0.005,0.01,0.02,0.05,0.1 \
  --horizon 1000 \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Fast local sweep around the strongest PS-win region:

```bash
python tmp/run_sim_v10_nonzero_eps_d_eta_eps_sweep.py \
  --output-dir outputs/v10_local_ps_region_rerun \
  --tree-spec analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json \
  --d-values 4,5,6 \
  --eta-values 0.3,0.4 \
  --epsilon-values 0.0025,0.005,0.0075,0.01 \
  --horizon 1000 \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

Original v10 compact outputs are included at:

```text
outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/
```

Important included files:

- `sweep_report.md`
- `combo_summaries.csv`
- `top_ps_winning_post_switch_configs.csv`
- `top_ps_winning_overall_configs.csv`
- `best_by_method_post_switch.csv`
- `best_by_method_overall.csv`

The original notes are included at:

```text
notes/sim_v10_nonzero_eps_d_eta_eps_sweep_full13_v1_report_2026-04-27.md
notes/sim_three_layer_topetaeps_ablation_v1_report_2026-04-27.md
```

## Latest Tree JSON

The newer trap-asymmetric profile-switch tree JSONs are included under:

```text
analysis/tree_specs/
```

Most important one for the latest/current full-run tree setting:

```text
analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json
```

Its validation metadata is included at both paths:

```text
analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5_validation.json
analysis/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5_validation.json
```

To run the controlled sim on this latest tree, pass it explicitly:

```bash
python scripts/run_barriershare_controlled_sim.py \
  --output-dir outputs/latest_tree_v3_eta03_eps001_d4_smoke \
  --tree-spec analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json \
  --tree-spec-role-mode spec_or_agent_id \
  --tree-spec-cost-mode ps_favored_trap_v10_avg_baited \
  --trap-switch-denominator 4 \
  --horizon 1000 \
  --seeds 0 1 2 3 4 \
  --cost-noise 0.02 \
  --specialist-fraction 0.15 \
  --common-eta-override 0.3 \
  --common-epsilon-override 0.01 \
  --methods \
    risky_ps_old risky_ps risky_ps_ix \
    risky_ps_safe_conditional risky_ps_safe_conditional_ix risky_ps_direct_cost \
    epsilon_exp3 direct_multistage_exp3 direct_multistage_exp3_local \
    naive_mixed_avg naive_mixed random_path
```

To regenerate the latest tree JSON itself:

```bash
python scripts/build_profile_switch_trap_asym_v3_efficient_anchor_tree.py
```

Earlier trap-asym tree JSONs are also included:

```text
analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v1.json
analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v2_neutral_4of5.json
```

## Reproduce Three-Layer Top Eta/Eps Ablation

The compact outputs are included at:

```text
outputs/sim_three_layer_topetaeps_ablation_v1/
```

Runner:

```bash
python tmp/run_sim_three_layer_topetaeps_ablation.py \
  --output-dir outputs/sim_three_layer_topetaeps_ablation_v1_rerun \
  --tree-spec analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json
```

To rerun the same ablation on the latest trap-asym v3 efficient-anchor tree,
change only `--tree-spec`:

```bash
python tmp/run_sim_three_layer_topetaeps_ablation.py \
  --output-dir outputs/sim_three_layer_topetaeps_ablation_v1_latest_tree_rerun \
  --tree-spec analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup_profile_switch_trap_asym_v3_efficient_anchor_4of5.json
```

## Known Good v10 Points

From the included v10 report:

- `d=6, eta=0.4, eps=0.005`
  - `switch_episode=floor(1000/6)=166`
  - overall PS-family winner: `risky_ps_old`
- `d=5, eta=0.4, eps=0.005`
  - `switch_episode=200`
  - overall PS-family winner: `risky_ps_old`
- `d=4, eta=0.3, eps=0.01`
  - found in the follow-up three-layer ablation
  - winner: `risky_ps_old`

## Notes

- `eps != 0` is controlled by `--common-epsilon-override`.
- In the main runner, `--common-epsilon-override` affects `epsilon_exp3` and
  PS-family methods. It does not affect `direct_multistage_exp3`, `naive_mixed`,
  or `random_path`.
- `d` in the notes is `--trap-switch-denominator`; switch happens at
  `floor(horizon / d)`.
- The packaged `scripts/run_barriershare_controlled_sim.py` includes a small
  compatibility fix for the legacy `ps_favored_trap` cost mode so the screenshot
  run can be regenerated from this folder directly.
