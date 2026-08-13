# Token-Fate Superiority Analysis

## Held-out conditional results
```text
             run                     slice          method    auroc  average_precision  delta_log_loss_vs_confidence  topk_realized_regret  topk_lift
math500_disjoint                       all confidence_only 0.892800           0.833527                      0.000000                  0.94   2.235172
math500_disjoint                       all       fate_lite 0.894487           0.832782                     -0.003371                  0.93   2.211394
math500_disjoint                       all      full_trace 0.894556           0.833025                     -0.003610                  0.92   2.187615
math500_disjoint      activation_0.00_0.25 confidence_only 0.879536           0.859435                      0.000000                  0.93   1.805221
math500_disjoint      activation_0.00_0.25       fate_lite 0.881062           0.858018                     -0.003386                  0.94   1.824632
math500_disjoint      activation_0.00_0.25      full_trace 0.881145           0.858558                     -0.003629                  0.95   1.844043
math500_disjoint       high_confidence_q75 confidence_only 0.814007           0.041537                      0.000000                  0.05   4.271987
math500_disjoint       high_confidence_q75       fate_lite 0.822493           0.060616                     -0.001551                  0.10   8.543974
math500_disjoint       high_confidence_q75      full_trace 0.826117           0.061769                     -0.001704                  0.10   8.543974
math500_disjoint early_high_confidence_q75 confidence_only 0.786597           0.046447                      0.000000                  0.06   4.022500
math500_disjoint early_high_confidence_q75       fate_lite 0.791812           0.067735                     -0.000524                  0.08   5.363333
math500_disjoint early_high_confidence_q75      full_trace 0.797017           0.069171                     -0.000838                  0.08   5.363333
```

## Example-bootstrap prediction deltas
```text
             run                     slice                      comparison            metric  eval_rows  bootstrap_examples  point_delta  bootstrap_ci_low  bootstrap_ci_high  bootstrap_samples bootstrap_unit
math500_disjoint       high_confidence_q75 fate_lite_minus_confidence_only             auroc      26230                 100     0.008486         -0.015288           0.033278               1000        example
math500_disjoint       high_confidence_q75 fate_lite_minus_confidence_only average_precision      26230                 100     0.019079         -0.003517           0.076943               1000        example
math500_disjoint early_high_confidence_q75 fate_lite_minus_confidence_only             auroc      16090                 100     0.005215         -0.025776           0.034602               1000        example
math500_disjoint early_high_confidence_q75 fate_lite_minus_confidence_only average_precision      16090                 100     0.021288         -0.006280           0.088400               1000        example
```

## High-confidence selector bootstrap
```text
                     run                          slice                     comparison            metric  eval_rows  bootstrap_examples  point_delta  bootstrap_ci_low  bootstrap_ci_high  bootstrap_samples bootstrap_unit
high_confidence_selector selector_high_confidence_early selector_minus_confidence_only             auroc      16090                 100     0.022798         -0.002943           0.040603               1000        example
high_confidence_selector selector_high_confidence_early selector_minus_confidence_only average_precision      16090                 100     0.031152         -0.004980           0.115872               1000        example
```

## Causal group comparisons
```text
              reference                                 comparison                           metric      comparison_design  reference_examples  comparison_examples  learned_rate  comparison_rate  delta_learned_minus_comparison  bootstrap_ci_low  bootstrap_ci_high  bootstrap_samples       bootstrap_unit
learned_high_confidence matched_same_trajectory_high_conf_low_risk         non_target_token_changed same_trajectory_paired                  43                   43      0.093023         0.046512                        0.046512               0.0           0.116279               1000 matched example pair
learned_high_confidence matched_same_trajectory_high_conf_low_risk    non_target_token_change_count same_trajectory_paired                  43                   43      1.116279         0.488372                        0.627907               0.0           1.534884               1000 matched example pair
learned_high_confidence matched_same_trajectory_high_conf_low_risk non_target_token_change_fraction same_trajectory_paired                  43                   43      0.017719         0.007752                        0.009967               0.0           0.025471               1000 matched example pair
learned_high_confidence matched_same_trajectory_high_conf_low_risk             suffix_token_changed same_trajectory_paired                  43                   43      0.069767         0.023256                        0.046512               0.0           0.116279               1000 matched example pair
learned_high_confidence matched_same_trajectory_high_conf_low_risk        suffix_token_change_count same_trajectory_paired                  43                   43      0.162791         0.023256                        0.139535               0.0           0.325581               1000 matched example pair
learned_high_confidence matched_same_trajectory_high_conf_low_risk              final_token_changed same_trajectory_paired                  43                   43      0.069767         0.023256                        0.046512               0.0           0.116279               1000 matched example pair
learned_high_confidence matched_same_trajectory_high_conf_low_risk           answer_correct_changed same_trajectory_paired                  43                   43      0.023256         0.000000                        0.023256               0.0           0.069767               1000 matched example pair
```
