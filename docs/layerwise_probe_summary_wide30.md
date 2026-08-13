# Layerwise Token-Fate Probe Summary

## Best Layer By Probe Family
| target             | family                      |   best_layer |   best_auc |   best_ap |   trace_auc |   delta_vs_trace |
|:-------------------|:----------------------------|-------------:|-----------:|----------:|------------:|-----------------:|
| all_trace_regret   | hidden                      |           24 |   0.778241 |  0.568194 |    0.906732 |     -0.128491    |
| all_trace_regret   | hidden_dynamics             |           28 |   0.685115 |  0.45121  |    0.906732 |     -0.221617    |
| all_trace_regret   | hidden_plus_logits          |           20 |   0.891908 |  0.7232   |    0.906732 |     -0.0148235   |
| all_trace_regret   | hidden_dynamics_plus_logits |           32 |   0.906826 |  0.76387  |    0.906732 |      9.36301e-05 |
| early_trace_regret | hidden                      |           32 |   0.734658 |  0.64741  |    0.881544 |     -0.146886    |
| early_trace_regret | hidden_dynamics             |           28 |   0.647883 |  0.509967 |    0.881544 |     -0.233661    |
| early_trace_regret | hidden_plus_logits          |           32 |   0.863683 |  0.769192 |    0.881544 |     -0.0178616   |
| early_trace_regret | hidden_dynamics_plus_logits |           16 |   0.885571 |  0.79885  |    0.881544 |      0.004027    |

## Layerwise Hidden Probes
| target             | family          |   layer |      auc |   average_precision |   eval_rows |   positive_rate |
|:-------------------|:----------------|--------:|---------:|--------------------:|------------:|----------------:|
| all_trace_regret   | hidden          |       4 | 0.686888 |            0.445411 |       13680 |        0.314254 |
| all_trace_regret   | hidden          |       8 | 0.71797  |            0.522986 |       13680 |        0.314254 |
| all_trace_regret   | hidden          |      12 | 0.733321 |            0.5469   |       13680 |        0.314254 |
| all_trace_regret   | hidden          |      16 | 0.703268 |            0.517026 |       13680 |        0.314254 |
| all_trace_regret   | hidden          |      20 | 0.738621 |            0.564857 |       13680 |        0.314254 |
| all_trace_regret   | hidden          |      24 | 0.778241 |            0.568194 |       13680 |        0.314254 |
| all_trace_regret   | hidden          |      28 | 0.770605 |            0.568356 |       13680 |        0.314254 |
| all_trace_regret   | hidden          |      32 | 0.66577  |            0.455632 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |       4 | 0.577218 |            0.335753 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |       8 | 0.653105 |            0.408591 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |      12 | 0.666059 |            0.462151 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |      16 | 0.666561 |            0.466712 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |      20 | 0.662123 |            0.461382 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |      24 | 0.631768 |            0.411228 |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |      28 | 0.685115 |            0.45121  |       13680 |        0.314254 |
| all_trace_regret   | hidden_dynamics |      32 | 0.575792 |            0.342322 |       13680 |        0.314254 |
| early_trace_regret | hidden          |       4 | 0.660506 |            0.535068 |        7680 |        0.423047 |
| early_trace_regret | hidden          |       8 | 0.67833  |            0.567354 |        7680 |        0.423047 |
| early_trace_regret | hidden          |      12 | 0.678648 |            0.580223 |        7680 |        0.423047 |
| early_trace_regret | hidden          |      16 | 0.662262 |            0.565653 |        7680 |        0.423047 |
| early_trace_regret | hidden          |      20 | 0.68553  |            0.583255 |        7680 |        0.423047 |
| early_trace_regret | hidden          |      24 | 0.720341 |            0.633525 |        7680 |        0.423047 |
| early_trace_regret | hidden          |      28 | 0.72261  |            0.622072 |        7680 |        0.423047 |
| early_trace_regret | hidden          |      32 | 0.734658 |            0.64741  |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |       4 | 0.589711 |            0.47562  |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |       8 | 0.614959 |            0.494238 |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |      12 | 0.632242 |            0.524833 |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |      16 | 0.62398  |            0.524629 |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |      20 | 0.63331  |            0.542827 |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |      24 | 0.581451 |            0.482519 |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |      28 | 0.647883 |            0.509967 |        7680 |        0.423047 |
| early_trace_regret | hidden_dynamics |      32 | 0.593278 |            0.471241 |        7680 |        0.423047 |