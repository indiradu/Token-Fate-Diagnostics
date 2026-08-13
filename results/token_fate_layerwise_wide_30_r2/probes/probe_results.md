# Token-Fate Probe Results

Best probe: `all_trace_regret/layer_32_hidden_dynamics_plus_logits` AUC `0.9068`

```text
                                                             probe  eval_rows  positive_rate      auc  average_precision  accuracy    brier
                                           all_trace_regret/logits      13680       0.314254 0.906732           0.763117  0.816813 0.126414
                                  all_trace_regret/layer_12_hidden      13680       0.314254 0.733321           0.546900  0.673392 0.206254
                         all_trace_regret/layer_12_hidden_dynamics      13680       0.314254 0.666059           0.462151  0.559211 0.245044
             all_trace_regret/layer_12_hidden_dynamics_plus_logits      13680       0.314254 0.905945           0.759670  0.815570 0.126615
                      all_trace_regret/layer_12_hidden_plus_logits      13680       0.314254 0.884310           0.716477  0.799635 0.138770
                                  all_trace_regret/layer_16_hidden      13680       0.314254 0.703268           0.517026  0.675512 0.209348
                         all_trace_regret/layer_16_hidden_dynamics      13680       0.314254 0.666561           0.466712  0.561842 0.244690
             all_trace_regret/layer_16_hidden_dynamics_plus_logits      13680       0.314254 0.904479           0.759212  0.813377 0.129050
                      all_trace_regret/layer_16_hidden_plus_logits      13680       0.314254 0.886047           0.718970  0.798099 0.138513
                                  all_trace_regret/layer_20_hidden      13680       0.314254 0.738621           0.564857  0.700292 0.198682
                         all_trace_regret/layer_20_hidden_dynamics      13680       0.314254 0.662123           0.461382  0.640570 0.219364
             all_trace_regret/layer_20_hidden_dynamics_plus_logits      13680       0.314254 0.904703           0.755009  0.816155 0.127622
                      all_trace_regret/layer_20_hidden_plus_logits      13680       0.314254 0.891908           0.723200  0.806433 0.131864
                                  all_trace_regret/layer_24_hidden      13680       0.314254 0.778241           0.568194  0.708114 0.189340
                         all_trace_regret/layer_24_hidden_dynamics      13680       0.314254 0.631768           0.411228  0.601170 0.230467
             all_trace_regret/layer_24_hidden_dynamics_plus_logits      13680       0.314254 0.904582           0.774081  0.810599 0.129201
                      all_trace_regret/layer_24_hidden_plus_logits      13680       0.314254 0.891637           0.725402  0.806360 0.139983
                                  all_trace_regret/layer_28_hidden      13680       0.314254 0.770605           0.568356  0.718713 0.191086
                         all_trace_regret/layer_28_hidden_dynamics      13680       0.314254 0.685115           0.451210  0.628363 0.227715
             all_trace_regret/layer_28_hidden_dynamics_plus_logits      13680       0.314254 0.906029           0.760400  0.814547 0.126848
                      all_trace_regret/layer_28_hidden_plus_logits      13680       0.314254 0.882941           0.703928  0.800292 0.139738
                                  all_trace_regret/layer_32_hidden      13680       0.314254 0.665770           0.455632  0.662427 0.226284
                         all_trace_regret/layer_32_hidden_dynamics      13680       0.314254 0.575792           0.342322  0.546564 0.245862
             all_trace_regret/layer_32_hidden_dynamics_plus_logits      13680       0.314254 0.906826           0.763870  0.817251 0.125800
                      all_trace_regret/layer_32_hidden_plus_logits      13680       0.314254 0.861458           0.652316  0.792982 0.145709
                                   all_trace_regret/layer_4_hidden      13680       0.314254 0.686888           0.445411  0.644444 0.231588
                          all_trace_regret/layer_4_hidden_dynamics      13680       0.314254 0.577218           0.335753  0.533699 0.233714
              all_trace_regret/layer_4_hidden_dynamics_plus_logits      13680       0.314254 0.903502           0.773534  0.808553 0.130108
                       all_trace_regret/layer_4_hidden_plus_logits      13680       0.314254 0.877173           0.703833  0.788012 0.143867
                                   all_trace_regret/layer_8_hidden      13680       0.314254 0.717970           0.522986  0.667617 0.212204
                          all_trace_regret/layer_8_hidden_dynamics      13680       0.314254 0.653105           0.408591  0.581871 0.225289
              all_trace_regret/layer_8_hidden_dynamics_plus_logits      13680       0.314254 0.902597           0.758923  0.810892 0.128983
                       all_trace_regret/layer_8_hidden_plus_logits      13680       0.314254 0.891594           0.741594  0.801023 0.136493
                                         early_trace_regret/logits       7680       0.423047 0.881544           0.795086  0.807422 0.135592
                                early_trace_regret/layer_12_hidden       7680       0.423047 0.678648           0.580223  0.631120 0.246543
                       early_trace_regret/layer_12_hidden_dynamics       7680       0.423047 0.632242           0.524833  0.569661 0.236258
           early_trace_regret/layer_12_hidden_dynamics_plus_logits       7680       0.423047 0.883812           0.794935  0.807161 0.134674
                    early_trace_regret/layer_12_hidden_plus_logits       7680       0.423047 0.854053           0.750772  0.785547 0.158505
                                early_trace_regret/layer_16_hidden       7680       0.423047 0.662262           0.565653  0.625000 0.248827
                       early_trace_regret/layer_16_hidden_dynamics       7680       0.423047 0.623980           0.524629  0.567839 0.242579
           early_trace_regret/layer_16_hidden_dynamics_plus_logits       7680       0.423047 0.885571           0.798850  0.804948 0.133761
                    early_trace_regret/layer_16_hidden_plus_logits       7680       0.423047 0.848789           0.747943  0.774089 0.162713
                                early_trace_regret/layer_20_hidden       7680       0.423047 0.685530           0.583255  0.634896 0.240443
                       early_trace_regret/layer_20_hidden_dynamics       7680       0.423047 0.633310           0.542827  0.594531 0.234680
           early_trace_regret/layer_20_hidden_dynamics_plus_logits       7680       0.423047 0.884612           0.796355  0.804688 0.134291
                    early_trace_regret/layer_20_hidden_plus_logits       7680       0.423047 0.861131           0.764272  0.788672 0.152194
                                early_trace_regret/layer_24_hidden       7680       0.423047 0.720341           0.633525  0.661589 0.229612
                       early_trace_regret/layer_24_hidden_dynamics       7680       0.423047 0.581451           0.482519  0.554818 0.247449
           early_trace_regret/layer_24_hidden_dynamics_plus_logits       7680       0.423047 0.882981           0.801835  0.800260 0.139206
                    early_trace_regret/layer_24_hidden_plus_logits       7680       0.423047 0.852366           0.763911  0.771094 0.161238
                                early_trace_regret/layer_28_hidden       7680       0.423047 0.722610           0.622072  0.664193 0.229910
                       early_trace_regret/layer_28_hidden_dynamics       7680       0.423047 0.647883           0.509967  0.612891 0.233542
           early_trace_regret/layer_28_hidden_dynamics_plus_logits       7680       0.423047 0.883404           0.803735  0.800391 0.139300
                    early_trace_regret/layer_28_hidden_plus_logits       7680       0.423047 0.857359           0.766379  0.781510 0.159183
                                early_trace_regret/layer_32_hidden       7680       0.423047 0.734658           0.647410  0.676302 0.228740
                       early_trace_regret/layer_32_hidden_dynamics       7680       0.423047 0.593278           0.471241  0.570312 0.238085
           early_trace_regret/layer_32_hidden_dynamics_plus_logits       7680       0.423047 0.883144           0.803302  0.799609 0.139447
                    early_trace_regret/layer_32_hidden_plus_logits       7680       0.423047 0.863683           0.769192  0.785026 0.152592
                                 early_trace_regret/layer_4_hidden       7680       0.423047 0.660506           0.535068  0.623828 0.264876
                        early_trace_regret/layer_4_hidden_dynamics       7680       0.423047 0.589711           0.475620  0.540365 0.239445
            early_trace_regret/layer_4_hidden_dynamics_plus_logits       7680       0.423047 0.884507           0.796782  0.804948 0.134689
                     early_trace_regret/layer_4_hidden_plus_logits       7680       0.423047 0.861231           0.768909  0.781380 0.157007
                                 early_trace_regret/layer_8_hidden       7680       0.423047 0.678330           0.567354  0.635677 0.252459
                        early_trace_regret/layer_8_hidden_dynamics       7680       0.423047 0.614959           0.494238  0.561198 0.238016
            early_trace_regret/layer_8_hidden_dynamics_plus_logits       7680       0.423047 0.882870           0.794770  0.807422 0.135257
                     early_trace_regret/layer_8_hidden_plus_logits       7680       0.423047 0.849994           0.759833  0.776042 0.167249
                              stable_high_conf_trace_regret/logits       2633       0.000000      NaN                NaN       NaN      NaN
                     stable_high_conf_trace_regret/layer_12_hidden       2633       0.000000      NaN                NaN       NaN      NaN
            stable_high_conf_trace_regret/layer_12_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
stable_high_conf_trace_regret/layer_12_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
         stable_high_conf_trace_regret/layer_12_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                     stable_high_conf_trace_regret/layer_16_hidden       2633       0.000000      NaN                NaN       NaN      NaN
            stable_high_conf_trace_regret/layer_16_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
stable_high_conf_trace_regret/layer_16_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
         stable_high_conf_trace_regret/layer_16_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                     stable_high_conf_trace_regret/layer_20_hidden       2633       0.000000      NaN                NaN       NaN      NaN
            stable_high_conf_trace_regret/layer_20_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
stable_high_conf_trace_regret/layer_20_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
         stable_high_conf_trace_regret/layer_20_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                     stable_high_conf_trace_regret/layer_24_hidden       2633       0.000000      NaN                NaN       NaN      NaN
            stable_high_conf_trace_regret/layer_24_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
stable_high_conf_trace_regret/layer_24_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
         stable_high_conf_trace_regret/layer_24_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                     stable_high_conf_trace_regret/layer_28_hidden       2633       0.000000      NaN                NaN       NaN      NaN
            stable_high_conf_trace_regret/layer_28_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
stable_high_conf_trace_regret/layer_28_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
         stable_high_conf_trace_regret/layer_28_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                     stable_high_conf_trace_regret/layer_32_hidden       2633       0.000000      NaN                NaN       NaN      NaN
            stable_high_conf_trace_regret/layer_32_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
stable_high_conf_trace_regret/layer_32_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
         stable_high_conf_trace_regret/layer_32_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                      stable_high_conf_trace_regret/layer_4_hidden       2633       0.000000      NaN                NaN       NaN      NaN
             stable_high_conf_trace_regret/layer_4_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
 stable_high_conf_trace_regret/layer_4_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
          stable_high_conf_trace_regret/layer_4_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
                      stable_high_conf_trace_regret/layer_8_hidden       2633       0.000000      NaN                NaN       NaN      NaN
             stable_high_conf_trace_regret/layer_8_hidden_dynamics       2633       0.000000      NaN                NaN       NaN      NaN
 stable_high_conf_trace_regret/layer_8_hidden_dynamics_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
          stable_high_conf_trace_regret/layer_8_hidden_plus_logits       2633       0.000000      NaN                NaN       NaN      NaN
```

## Fate Summary, Train
```text
          fate  count  fraction
 early_correct    705  0.400568
   oscillating    633  0.359659
late_corrected    369  0.209659
   early_wrong     49  0.027841
         other      3  0.001705
  stable_wrong      1  0.000568
```

## Fate Summary, Eval
```text
          fate  count  fraction
 early_correct    765  0.436644
   oscillating    617  0.352169
late_corrected    317  0.180936
   early_wrong     51  0.029110
         other      2  0.001142
```