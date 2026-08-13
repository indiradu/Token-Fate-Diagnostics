# Token-Fate Superiority Analysis

## Held-out conditional results
```text
          run                     slice          method    auroc  average_precision  delta_log_loss_vs_confidence  topk_realized_regret  topk_lift
heldout_gsm8k                       all confidence_only 0.882371           0.790350                      0.000000                  0.94   2.433394
heldout_gsm8k                       all       fate_lite 0.877236           0.777998                      0.009688                  0.86   2.226297
heldout_gsm8k                       all      full_trace 0.876245           0.776728                      0.012380                  0.87   2.252184
heldout_gsm8k      activation_0.00_0.25 confidence_only 0.857900           0.822196                      0.000000                  0.93   1.866362
heldout_gsm8k      activation_0.00_0.25       fate_lite 0.852983           0.812690                      0.009126                  0.90   1.806157
heldout_gsm8k      activation_0.00_0.25      full_trace 0.852351           0.811794                      0.011312                  0.89   1.786089
heldout_gsm8k       high_confidence_q75 confidence_only 0.887209           0.010139                      0.000000                  0.01   5.640601
heldout_gsm8k       high_confidence_q75       fate_lite 0.663295           0.003782                     -0.000600                  0.00   0.000000
heldout_gsm8k       high_confidence_q75      full_trace 0.639157           0.003617                     -0.000322                  0.00   0.000000
heldout_gsm8k early_high_confidence_q75 confidence_only 0.892914           0.013399                      0.000000                  0.00   0.000000
heldout_gsm8k early_high_confidence_q75       fate_lite 0.757774           0.010435                     -0.001021                  0.02   8.252117
heldout_gsm8k early_high_confidence_q75      full_trace 0.725604           0.010043                     -0.000822                  0.02   8.252117
```

## Example-bootstrap prediction deltas
```text
          run                     slice                      comparison            metric  eval_rows  bootstrap_examples  point_delta  bootstrap_ci_low  bootstrap_ci_high  bootstrap_samples bootstrap_unit
heldout_gsm8k       high_confidence_q75 fate_lite_minus_confidence_only             auroc     103223                 500    -0.223914         -0.317837          -0.120303               1000        example
heldout_gsm8k       high_confidence_q75 fate_lite_minus_confidence_only average_precision     103223                 500    -0.006358         -0.010745          -0.003331               1000        example
heldout_gsm8k early_high_confidence_q75 fate_lite_minus_confidence_only             auroc      56527                 500    -0.135140         -0.221870          -0.042274               1000        example
heldout_gsm8k early_high_confidence_q75 fate_lite_minus_confidence_only average_precision      56527                 500    -0.002964         -0.009674           0.007545               1000        example
```

## High-confidence selector bootstrap
```text
                     run                          slice                     comparison            metric  eval_rows  bootstrap_examples  point_delta  bootstrap_ci_low  bootstrap_ci_high  bootstrap_samples bootstrap_unit
high_confidence_selector selector_high_confidence_early selector_minus_confidence_only             auroc      79558                 500     0.003523         -0.010223           0.017947               1000        example
high_confidence_selector selector_high_confidence_early selector_minus_confidence_only average_precision      79558                 500    -0.000514         -0.008141           0.007850               1000        example
```
