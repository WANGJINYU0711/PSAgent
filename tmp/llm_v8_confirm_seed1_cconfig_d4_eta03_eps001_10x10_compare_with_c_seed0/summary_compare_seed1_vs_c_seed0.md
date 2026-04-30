# Confirmatory seed1 vs C seed0 comparison

## all

| rank | run | seed | method | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PSfamily_seed0 | 0 | `risky_ps` | 9.01 | 4.50 | 3.06 | 4.44 | 1.28 | 50% | 77% | 77% | 70% | 22% |
| 2 | C_seed0_3methods | 0 | `direct_multistage_exp3` | 9.53 | 4.91 | 3.04 | 4.55 | 1.34 | 52% | 73% | 78% | 64% | 18% |
| 3 | confirm_seed1 | 1 | `epsilon_exp3` | 9.61 | 5.03 | 3.15 | 4.51 | 1.49 | 52% | 72% | 75% | 64% | 31% |
| 4 | C_seed0_3methods | 0 | `risky_ps_linear` | 9.70 | 5.16 | 3.43 | 4.47 | 1.30 | 53% | 70% | 78% | 65% | 17% |
| 5 | confirm_seed1 | 1 | `direct_multistage_exp3` | 9.90 | 5.12 | 3.18 | 4.71 | 1.82 | 52% | 71% | 76% | 63% | 20% |
| 6 | C_seed0_3methods | 0 | `epsilon_exp3` | 10.50 | 5.61 | 3.38 | 4.82 | 1.81 | 53% | 67% | 73% | 62% | 15% |
| 7 | confirm_seed1 | 1 | `risky_ps` | 10.50 | 5.99 | 3.94 | 4.45 | 1.67 | 57% | 65% | 70% | 62% | 18% |

## post

| rank | run | seed | method | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | PSfamily_seed0 | 0 | `risky_ps` | 10.37 | 6.00 | 4.08 | 4.30 | 1.22 | 58% | 69% | 69% | 60% | 29% |
| 2 | C_seed0_3methods | 0 | `direct_multistage_exp3` | 10.80 | 6.41 | 4.02 | 4.31 | 1.30 | 59% | 65% | 71% | 53% | 24% |
| 3 | C_seed0_3methods | 0 | `risky_ps_linear` | 11.06 | 6.75 | 4.53 | 4.25 | 1.22 | 61% | 61% | 71% | 55% | 23% |
| 4 | confirm_seed1 | 1 | `epsilon_exp3` | 11.11 | 6.71 | 4.19 | 4.34 | 1.45 | 60% | 63% | 67% | 52% | 41% |
| 5 | confirm_seed1 | 1 | `direct_multistage_exp3` | 11.15 | 6.58 | 4.18 | 4.50 | 1.86 | 59% | 63% | 68% | 52% | 27% |
| 6 | C_seed0_3methods | 0 | `epsilon_exp3` | 11.84 | 7.47 | 4.51 | 4.29 | 1.83 | 63% | 56% | 64% | 49% | 20% |
| 7 | confirm_seed1 | 1 | `risky_ps` | 12.44 | 7.98 | 5.26 | 4.39 | 1.70 | 64% | 53% | 60% | 49% | 24% |
