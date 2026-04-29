# Unique-agent 4/5-share controlled simulation

## Direct Answers
- risky_ps vs epsilon_exp3 gap is not smaller: unique=0.020667, old=0.000000.
- risky_ps vs direct_multistage_exp3 gap is smaller: unique=0.035667, old=0.191784.
- Cross-prefix reuse removal does shrink the average risky_ps gap to epsilon/direct: unique=0.028167, old=0.095892.
- naive_mixed rank changed from old=4 to unique=12; random_path changed from old=5 to unique=13.
- The unique tree has num_paths=440 versus old partial_4of5 num_paths=3125.
- This is a real caveat: fewer paths make exploration easier, so any comparison mixes structure cleanup with a smaller path space. If risky_ps still does not close the gap under fewer paths, that weakens the cross-prefix-reuse hypothesis.

## Unique-Agent Tree Results
| method | regret_per_t_mean | average_cost_mean | shared_path_fraction_mean | num_paths | horizon | seeds |
| --- | --- | --- | --- | --- | --- | --- |
| naive_mixed_avg | 0.009431367288761953 | 0.022000000000000002 | 1.0 | 440 | 1000 | 3 |
| direct_multistage_exp3 | 0.05343136728876196 | 0.066 | 0.9856666666666666 | 440 | 1000 | 3 |
| risky_ps_old | 0.05709803395542862 | 0.06966666666666667 | 0.9913333333333334 | 440 | 1000 | 3 |
| risky_ps_linear | 0.05743136728876195 | 0.07 | 0.9913333333333334 | 440 | 1000 | 3 |
| epsilon_exp3 | 0.06843136728876195 | 0.081 | 0.9883333333333333 | 440 | 1000 | 3 |
| risky_ps | 0.08909803395542863 | 0.10166666666666667 | 0.9893333333333333 | 440 | 1000 | 3 |
| risky_ps_ix | 0.0900980339554286 | 0.10266666666666667 | 0.989 | 440 | 1000 | 3 |
| direct_multistage_exp3_local | 0.09143136728876194 | 0.104 | 0.9836666666666667 | 440 | 1000 | 3 |
| risky_ps_safe_conditional | 0.10143136728876195 | 0.114 | 0.9923333333333333 | 440 | 1000 | 3 |
| risky_ps_safe_conditional_ix | 0.10143136728876195 | 0.114 | 0.9923333333333333 | 440 | 1000 | 3 |
| risky_ps_direct_cost | 0.323431367288762 | 0.336 | 0.9819999999999999 | 440 | 1000 | 3 |
| naive_mixed | 0.585431367288762 | 0.598 | 1.0 | 440 | 1000 | 3 |
| random_path | 0.6860980339554287 | 0.6986666666666667 | 0.9266666666666667 | 440 | 1000 | 3 |

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
| direct_multistage_exp3 | -0.020154340647309636 | -0.05453624211417821 | 0.004946666666666544 |
| epsilon_exp3 | -0.1969378496422252 | -0.23131975110909375 | 0.08037333333333341 |
| risky_ps | -0.1762711829755585 | -0.21065308444242709 | 0.08137333333333341 |
| naive_mixed | 0.23620676743087676 | 0.20182486596400812 | 0.97936 |
| random_path | 0.056548735609281864 | 0.022166834142413272 | 0.12836666666666674 |
