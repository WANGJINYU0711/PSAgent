# C direct/epsilon vs PS-family C-config comparison

| run | split | method | total | terminal | legacy | reasoning | modecost report | terminal share | clear | aux | strict | exact mode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PS_family_C | all | risky_ps | 9.01 | 4.50 | 3.06 | 4.44 | 1.28 | 50% | 77% | 77% | 70% | 22% |
| C_baseline | all | direct_multistage_exp3 | 9.53 | 4.91 | 3.04 | 4.55 | 1.34 | 52% | 73% | 78% | 64% | 18% |
| PS_family_C | all | risky_ps_safe_conditional_ix | 9.55 | 5.08 | 3.35 | 4.39 | 1.48 | 54% | 71% | 75% | 67% | 15% |
| PS_family_C | all | risky_ps_safe_conditional | 9.58 | 5.02 | 3.40 | 4.49 | 1.46 | 53% | 73% | 73% | 67% | 18% |
| PS_family_C | all | risky_ps_ix | 10.13 | 5.41 | 3.90 | 4.65 | 1.50 | 54% | 72% | 78% | 68% | 17% |
| PS_family_C | all | risky_ps_direct_cost | 10.33 | 5.82 | 3.69 | 4.44 | 1.55 | 57% | 66% | 74% | 62% | 15% |
| PS_family_C | all | risky_ps_old | 10.41 | 5.89 | 4.09 | 4.45 | 1.47 | 57% | 66% | 76% | 63% | 17% |
| C_baseline | all | epsilon_exp3 | 10.50 | 5.61 | 3.38 | 4.82 | 1.81 | 54% | 67% | 73% | 62% | 15% |
| PS_family_C | all | risky_ps_linear | 11.19 | 6.64 | 4.47 | 4.47 | 1.44 | 60% | 62% | 75% | 59% | 21% |
| PS_family_C | post | risky_ps | 10.37 | 6.00 | 4.08 | 4.30 | 1.22 | 58% | 69% | 69% | 60% | 29% |
| C_baseline | post | direct_multistage_exp3 | 10.80 | 6.41 | 4.02 | 4.31 | 1.30 | 60% | 65% | 71% | 53% | 24% |
| PS_family_C | post | risky_ps_safe_conditional | 11.09 | 6.69 | 4.53 | 4.32 | 1.45 | 61% | 64% | 64% | 56% | 24% |
| PS_family_C | post | risky_ps_safe_conditional_ix | 11.15 | 6.78 | 4.47 | 4.30 | 1.48 | 61% | 61% | 67% | 56% | 20% |
| PS_family_C | post | risky_ps_ix | 11.72 | 7.21 | 5.19 | 4.43 | 1.45 | 62% | 63% | 71% | 57% | 23% |
| C_baseline | post | epsilon_exp3 | 11.84 | 7.47 | 4.51 | 4.29 | 1.83 | 64% | 56% | 64% | 49% | 20% |
| PS_family_C | post | risky_ps_direct_cost | 12.18 | 7.75 | 4.91 | 4.36 | 1.58 | 64% | 55% | 65% | 49% | 20% |
| PS_family_C | post | risky_ps_old | 12.21 | 7.85 | 5.45 | 4.28 | 1.46 | 65% | 55% | 68% | 51% | 23% |
| PS_family_C | post | risky_ps_linear | 13.34 | 8.86 | 5.96 | 4.41 | 1.42 | 67% | 49% | 67% | 45% | 28% |
