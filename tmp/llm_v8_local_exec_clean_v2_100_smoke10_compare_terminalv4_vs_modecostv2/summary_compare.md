# llm_v8 local_exec_clean_v2_100 smoke10 comparison

| run | split | method | total | terminal | legacy | reasoning | modecost | terminal share | clear | aux | strict | exact mode |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| terminalv4 | all | direct_multistage_exp3 | 12.08 | 7.14 | 5.18 | 4.86 | 0.00 | 59% | 63% | 74% | 59% | 24% |
| terminalv4 | all | epsilon_exp3 | 11.39 | 6.39 | 4.58 | 4.93 | 0.00 | 56% | 67% | 74% | 64% | 18% |
| terminalv4 | all | risky_ps_linear | 9.71 | 4.71 | 3.06 | 4.93 | 0.00 | 49% | 75% | 76% | 68% | 25% |
| terminalv4 | post | direct_multistage_exp3 | 14.47 | 9.52 | 6.91 | 4.87 | 0.00 | 66% | 51% | 65% | 45% | 32% |
| terminalv4 | post | epsilon_exp3 | 13.34 | 8.52 | 6.11 | 4.74 | 0.00 | 64% | 56% | 65% | 52% | 24% |
| terminalv4 | post | risky_ps_linear | 11.16 | 6.15 | 4.07 | 4.94 | 0.00 | 55% | 68% | 68% | 59% | 33% |
| terminalv4_modecostv2 | all | direct_multistage_exp3 | 12.10 | 5.81 | 3.87 | 6.22 | 1.38 | 48% | 70% | 72% | 61% | 23% |
| terminalv4_modecostv2 | all | epsilon_exp3 | 12.39 | 5.72 | 3.78 | 6.60 | 1.53 | 46% | 67% | 77% | 65% | 15% |
| terminalv4_modecostv2 | all | risky_ps_linear | 12.89 | 6.20 | 4.32 | 6.62 | 1.54 | 48% | 68% | 77% | 62% | 20% |
| terminalv4_modecostv2 | post | direct_multistage_exp3 | 13.96 | 7.75 | 5.15 | 6.14 | 1.31 | 56% | 60% | 63% | 48% | 31% |
| terminalv4_modecostv2 | post | epsilon_exp3 | 14.16 | 7.63 | 5.04 | 6.46 | 1.49 | 54% | 56% | 69% | 53% | 20% |
| terminalv4_modecostv2 | post | risky_ps_linear | 14.71 | 8.27 | 5.76 | 6.37 | 1.47 | 56% | 57% | 69% | 49% | 27% |
