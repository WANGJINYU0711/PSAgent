# BarrierShare Controlled Sim Package

This folder is a standalone copy of the main non-LLM simulation code and the
two result sets discussed with the advisor:

1. The screenshot/table run:
   `outputs/barriershare_controlled_sim_prefix_dedup_ps_favored_trap_v9_eta_epsilon_sweep/`
2. The v10 nonzero-epsilon sweep:
   `outputs/sim_v10_fixed_nonzero_eps_d_eta_eps_sweep_full13_v1/`

Run commands below from this `sim/` directory.

## Contents

- `scripts/run_barriershare_controlled_sim.py`
  - Main controlled-simulation runner.
  - No LLM/API calls.
  - Key knobs: `--common-eta-override`, `--common-epsilon-override`,
    `--trap-switch-denominator`, `--tree-spec-cost-mode`.
- `tmp/run_sim_v10_nonzero_eps_d_eta_eps_sweep.py`
  - Aggregated d/eta/epsilon sweep runner for the v10 experiment.
- `scripts/run_barriershare_controlled_bestof_sweep.py`
  - Best-of hyperparameter sweep helper.
- `baselines/`
  - PS and baseline policies.
- `envs/`
  - Minimal environment, tree, executor, and adapter code needed by the runner.
- `analysis/tree_specs/shared_basin_strong_4of5_prefix_dedup.json`
  - Tree spec used by both included sim runs.
- `analysis/shared_basin_strong_4of5_prefix_dedup_validation.json`
  - Validation metadata used by the runner summaries.
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
