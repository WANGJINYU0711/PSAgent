# llm_v8 PS update stability smoke

Config: C terminalv4 + reasoning calibration v3 + report-only modecost, d=4, eta=0.3, epsilon=0.01, risky_ps only. Sim guard selected `loss_clip=100` and `prob_floor=0.002`; `prob_floor=0.005` was rejected as too large.

## Seed1 full 100 results

| rank | method | total | terminal | reasoning | modecost report | clear | aux | strict |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | `epsilon` | 9.612 | 5.030 | 4.512 | 1.490 | 72% | 75% | 64% |
| 2 | `direct` | 9.898 | 5.115 | 4.712 | 1.825 | 71% | 76% | 63% |
| 3 | `probfloor0002` | 10.293 | 5.730 | 4.492 | 1.625 | 69% | 74% | 66% |
| 4 | `eta_shared002` | 10.464 | 5.800 | 4.593 | 1.690 | 67% | 72% | 62% |
| 5 | `old_risky_ps` | 10.504 | 5.985 | 4.449 | 1.665 | 65% | 70% | 62% |

## Seed1 full post results

| rank | method | total | terminal | reasoning | modecost report | clear | aux | strict | transfers | rtp>=14 |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `epsilon` | 11.114 | 6.707 | 4.336 | 1.447 | 63% | 67% | 52% | 11 | 16 |
| 2 | `direct` | 11.153 | 6.580 | 4.503 | 1.860 | 63% | 68% | 52% | 7 | 18 |
| 3 | `probfloor0002` | 12.059 | 7.640 | 4.347 | 1.647 | 59% | 65% | 55% | 14 | 24 |
| 4 | `eta_shared002` | 12.301 | 7.733 | 4.497 | 1.733 | 56% | 63% | 49% | 11 | 27 |
| 5 | `old_risky_ps` | 12.436 | 7.980 | 4.387 | 1.700 | 53% | 60% | 49% | 11 | 23 |

## Seed0 early stop at t~77

Seed0 new variants were stopped after the t>=75 check because old `risky_ps` was stable first for the last 15 aligned episodes with margin >1.0 and the new variants had worse rolling totals. See `early_t75_check.csv`.


## Seed1 post majority-pair details for completed new variants

| method | pair | n | total | terminal | reasoning | clear | strict |
|---|---|---:|---:|---:|---:|---:|---:|
| `direct` | `mostly_deep_vs_mostly_deep_required` | 58 | 11.130 | 6.767 | 4.291 | 62% | 52% |
| `direct` | `mostly_fast_vs_mostly_deep_required` | 17 | 11.231 | 5.941 | 5.225 | 65% | 53% |
| `epsilon` | `mostly_deep_vs_mostly_deep_required` | 60 | 10.917 | 6.683 | 4.161 | 62% | 50% |
| `epsilon` | `mostly_fast_vs_mostly_deep_required` | 15 | 11.902 | 6.800 | 5.039 | 67% | 60% |
| `eta_shared002` | `mostly_deep_vs_mostly_deep_required` | 57 | 11.975 | 7.526 | 4.376 | 60% | 51% |
| `eta_shared002` | `mostly_fast_vs_mostly_deep_required` | 18 | 13.337 | 8.389 | 4.883 | 44% | 44% |
| `old_risky_ps` | `mostly_deep_vs_mostly_deep_required` | 61 | 11.663 | 7.361 | 4.231 | 57% | 52% |
| `old_risky_ps` | `mostly_fast_vs_mostly_deep_required` | 14 | 15.806 | 10.679 | 5.065 | 36% | 36% |
| `probfloor0002` | `mostly_deep_vs_mostly_deep_required` | 63 | 12.609 | 8.262 | 4.275 | 56% | 51% |
| `probfloor0002` | `mostly_fast_vs_mostly_deep_required` | 12 | 9.168 | 4.375 | 4.727 | 75% | 75% |

## Output files

- `summary.csv`: all/post summary for historical, full new, and early-stopped new runs.
- `post_pair.csv`: post majority-pair breakdown.
- `update_diag.csv`: path probability / estimated-loss diagnostics.
- `early_t75_check.csv`: aligned t>=75 early-stop decision table.
