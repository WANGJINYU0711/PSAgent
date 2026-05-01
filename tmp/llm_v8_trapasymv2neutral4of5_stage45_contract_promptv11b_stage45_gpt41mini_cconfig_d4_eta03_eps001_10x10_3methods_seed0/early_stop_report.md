# trap_asym_v2_neutral_4of5 seed0 early-stop report

## Decision
- early_stop: True
- min_episodes: 75
- raw_gap_threshold: 0.5
- ps_raw_mean: 9.091902
- best_baseline_method: epsilon_exp3
- best_baseline_raw_mean: 8.343095
- gap_vs_best_baseline: 0.748806

## First 75 episode fair comparison
| method | n | raw_total | terminal | exact | reasoning | path | mode_mismatch | shared | unshared |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| epsilon_exp3 | 75 | 8.304 | 3.000 | 0.813 | 5.238 | 0.066 | 1.993 | 0.880 | 0.120 |
| direct_multistage_exp3 | 75 | 8.122 | 2.827 | 0.840 | 5.230 | 0.065 | 1.920 | 0.907 | 0.093 |
| risky_ps | 75 | 9.092 | 3.720 | 0.733 | 5.307 | 0.065 | 2.013 | 0.893 | 0.107 |

## First 75 by phase
| method | pre n | pre raw | pre exact | post n | post raw | post terminal | post exact |
|---|---:|---:|---:|---:|---:|---:|---:|
| epsilon_exp3 | 25 | 4.727 | 1.000 | 50 | 10.092 | 4.500 | 0.720 |
| direct_multistage_exp3 | 25 | 4.476 | 1.000 | 50 | 9.945 | 4.240 | 0.760 |
| risky_ps | 25 | 4.769 | 1.000 | 50 | 11.253 | 5.580 | 0.600 |

## First 75 specialist/post target-heavy slice
| method | n | raw_total | terminal | exact | shared | unshared |
|---|---:|---:|---:|---:|---:|---:|
| epsilon_exp3 | 20 | 13.815 | 7.750 | 0.550 | 0.750 | 0.250 |
| direct_multistage_exp3 | 20 | 14.873 | 8.300 | 0.550 | 0.900 | 0.100 |
| risky_ps | 20 | 16.176 | 9.850 | 0.300 | 0.900 | 0.100 |

## Available episodes at stop
| method | completed | raw_total | terminal | exact | connection_errors_seen | return_code_1 |
|---|---:|---:|---:|---:|---:|---:|
| epsilon_exp3 | 77 | 8.343 | 3.000 | 0.818 | 9 | 1 |
| direct_multistage_exp3 | 80 | 8.577 | 3.163 | 0.812 | 3 | 1 |
| risky_ps | 75 | 9.092 | 3.720 | 0.733 | 12 | 1 |

## Notes
- The run used the v2 neutral 4/5 tree and otherwise matched the seed0 stage45 promptv11b gpt-4.1-mini C config.
- The run was stopped at the t=75 rule because risky_ps lagged the best baseline by more than 0.5 raw cost.
- Multiple OpenAI connection errors occurred; method-level checkpoints were resumed, and an outer bridge retry was added before the final risky_ps resume.
