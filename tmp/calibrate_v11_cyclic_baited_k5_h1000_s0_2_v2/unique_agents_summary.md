# Unique-agent 4/5-share controlled simulation

## Direct Answers
- risky_ps vs epsilon_exp3 gap is smaller: unique=-0.048333, old=0.000000.
- risky_ps vs direct_multistage_exp3 gap is smaller: unique=-0.037000, old=0.191784.
- Cross-prefix reuse removal does shrink the average risky_ps gap to epsilon/direct: unique=-0.042667, old=0.095892.
- naive_mixed rank changed from old=4 to unique=12; random_path changed from old=5 to unique=13.
- The unique tree has num_paths=440 versus old partial_4of5 num_paths=3125.
- This is a real caveat: fewer paths make exploration easier, so any comparison mixes structure cleanup with a smaller path space. If risky_ps still does not close the gap under fewer paths, that weakens the cross-prefix-reuse hypothesis.

## Unique-Agent Tree Results
| method | regret_per_t_mean | average_cost_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| naive_mixed_avg | 0.011260343782471012 | 0.02766666666666667 | 1.0 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.14692701044913767 | 0.16333333333333333 | 0.9813333333333333 | 440 | 1000 | 3 |
| risky_ps | 0.159260343782471 | 0.17566666666666667 | 0.987 | 440 | 1000 | 3 |
| risky_ps_ix | 0.16192701044913768 | 0.17833333333333334 | 0.9819999999999999 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.1769270104491377 | 0.19333333333333336 | 0.9886666666666667 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.1769270104491377 | 0.19333333333333336 | 0.9886666666666667 | 440 | 1000 | 3 |
| risky_ps_linear | 0.18959367711580435 | 0.206 | 0.9866666666666667 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.196260343782471 | 0.21266666666666667 | 0.9793333333333333 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.20759367711580434 | 0.224 | 0.975 | 440 | 1000 | 3 |
| risky_ps_old | 0.27959367711580435 | 0.296 | 0.9773333333333333 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.3389270104491377 | 0.3553333333333333 | 0.9780000000000001 | 440 | 1000 | 3 |
| naive_mixed | 0.661260343782471 | 0.6776666666666666 | 1.0 | 440 | 1000 | 3 |
| random_path | 0.7105936771158042 | 0.727 | 0.9266666666666667 | 440 | 1000 | 3 |

## Old Tree Reference: main_partial_4of5_L5_K5
| method | regret_per_t_mean | terminal_proxy_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| direct_multistage_exp3 | 0.0735857079360716 | 0.12053624211417821 | 0.98072 | 3125 | 5000 | 10 |
| risky_ps | 0.26536921693098714 | 0.31231975110909377 | 0.9079599999999999 | 3125 | 5000 | 10 |
| epsilon_exp3 | 0.26536921693098714 | 0.31231975110909377 | 0.9079599999999999 | 3125 | 5000 | 10 |
| naive_mixed | 0.34922459985788523 | 0.39617513403599186 | 0.02064 | 3125 | 5000 | 10 |
| random_path | 0.6295492983461468 | 0.6764998325242534 | 0.7983 | 3125 | 5000 | 10 |

## Delta vs Old Tree
| method | delta_regret_per_t | delta_terminal_proxy | delta_shared_path_fraction |
| --- | --- | --- | --- |
| risky_ps | -0.10610887314851614 | -0.1366530844424271 | 0.07904000000000011 |
| direct_multistage_exp3 | 0.12267463584639941 | 0.09213042455248846 | -0.0013866666666667582 |
| epsilon_exp3 | -0.0577755398151828 | -0.08831975110909376 | 0.0670400000000001 |
| naive_mixed | 0.3120357439245858 | 0.2814915326306748 | 0.97936 |
| random_path | 0.08104437876965742 | 0.050500167475746593 | 0.12836666666666674 |
