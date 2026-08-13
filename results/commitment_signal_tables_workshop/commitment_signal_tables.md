# Commitment Signal Tables

## Cross-task summary
```text
            run                  signal  base_regret_rate    auroc  average_precision  topk_realized_regret  topk_lift
   illada_gsm8k learned_trace_predictor          0.463968 0.894731           0.850629                  0.93   2.004449
   illada_gsm8k              confidence          0.463968 0.893601           0.852095                  0.93   2.004449
llada_countdown learned_trace_predictor          0.275549 0.984906           0.948161                  0.99   3.592824
llada_countdown              confidence          0.275549 0.969857           0.895257                  0.94   3.411368
    llada_gsm8k learned_trace_predictor          0.397519 0.887263           0.802107                  0.86   2.163419
    llada_gsm8k              confidence          0.397519 0.886386           0.805422                  0.86   2.163419
  llada_math500 learned_trace_predictor          0.429347 0.901462           0.847280                  0.91   2.119500
  llada_math500              confidence          0.429347 0.899449           0.847010                  0.88   2.049626
```

## Signal leaderboard
```text
            run             target                  signal orientation    auroc  average_precision  topk_realized_regret  topk_lift
    llada_gsm8k   all_trace_regret learned_trace_predictor         raw 0.887263           0.802107                  0.86   2.163419
    llada_gsm8k   all_trace_regret              confidence    inverted 0.886386           0.805422                  0.86   2.163419
    llada_gsm8k   all_trace_regret                 entropy         raw 0.500000           0.397519                  0.25   0.628901
    llada_gsm8k   all_trace_regret                  margin    inverted 0.873104           0.755857                  0.76   1.911859
    llada_gsm8k   all_trace_regret                      kl         raw 0.578371           0.417132                  0.19   0.477965
    llada_gsm8k   all_trace_regret               top1_flip         raw 0.572614           0.451388                  0.53   1.333270
    llada_gsm8k   all_trace_regret               runlength    inverted 0.694169           0.559296                  0.24   0.603745
    llada_gsm8k   all_trace_regret right_boundary_nearness         raw 0.583069           0.458819                  0.48   1.207490
    llada_gsm8k early_trace_regret learned_trace_predictor         raw 0.873995           0.818386                  0.88   1.920392
    llada_gsm8k early_trace_regret              confidence    inverted 0.873177           0.820957                  0.87   1.898569
    llada_gsm8k early_trace_regret                 entropy         raw 0.500000           0.458240                  0.25   0.545566
    llada_gsm8k early_trace_regret                  margin    inverted 0.858411           0.776655                  0.72   1.571230
    llada_gsm8k early_trace_regret                      kl         raw 0.555908           0.464319                  0.24   0.523743
    llada_gsm8k early_trace_regret               top1_flip         raw 0.564589           0.503413                  0.66   1.440294
    llada_gsm8k early_trace_regret               runlength    inverted 0.679975           0.600547                  0.42   0.916551
    llada_gsm8k early_trace_regret right_boundary_nearness         raw 0.612147           0.541303                  0.48   1.047487
  llada_math500   all_trace_regret learned_trace_predictor         raw 0.901462           0.847280                  0.91   2.119500
  llada_math500   all_trace_regret              confidence    inverted 0.899449           0.847010                  0.88   2.049626
  llada_math500   all_trace_regret                 entropy         raw 0.500000           0.429347                  0.17   0.395951
  llada_math500   all_trace_regret                  margin    inverted 0.886826           0.799381                  0.80   1.863296
  llada_math500   all_trace_regret                      kl         raw 0.569679           0.440807                  0.15   0.349368
  llada_math500   all_trace_regret               top1_flip         raw 0.578740           0.487672                  0.61   1.420764
  llada_math500   all_trace_regret               runlength    inverted 0.695003           0.592964                  0.31   0.722027
  llada_math500   all_trace_regret right_boundary_nearness         raw 0.581784           0.496300                  0.53   1.234434
  llada_math500 early_trace_regret learned_trace_predictor         raw 0.891974           0.859405                  0.93   1.910091
  llada_math500 early_trace_regret              confidence    inverted 0.890151           0.859947                  0.91   1.869014
  llada_math500 early_trace_regret                 entropy         raw 0.500000           0.486888                  0.17   0.349156
  llada_math500 early_trace_regret                  margin    inverted 0.875808           0.817521                  0.72   1.478780
  llada_math500 early_trace_regret                      kl         raw 0.551947           0.486188                  0.19   0.390234
  llada_math500 early_trace_regret               top1_flip         raw 0.572213           0.537414                  0.65   1.335010
  llada_math500 early_trace_regret               runlength    inverted 0.687544           0.635937                  0.56   1.150162
  llada_math500 early_trace_regret right_boundary_nearness         raw 0.610747           0.577744                  0.61   1.252855
llada_countdown   all_trace_regret learned_trace_predictor         raw 0.984906           0.948161                  0.99   3.592824
llada_countdown   all_trace_regret              confidence    inverted 0.969857           0.895257                  0.94   3.411368
llada_countdown   all_trace_regret                 entropy         raw 0.500000           0.275549                  0.85   3.084748
llada_countdown   all_trace_regret                  margin    inverted 0.955440           0.820082                  0.67   2.431507
llada_countdown   all_trace_regret                      kl         raw 0.556815           0.290881                  0.51   1.850849
llada_countdown   all_trace_regret               top1_flip         raw 0.555664           0.326023                  0.54   1.959722
llada_countdown   all_trace_regret               runlength    inverted 0.836038           0.557033                  0.03   0.108873
llada_countdown   all_trace_regret right_boundary_nearness    inverted 0.833928           0.579271                  0.55   1.996013
llada_countdown early_trace_regret learned_trace_predictor         raw 0.986205           0.964446                  0.99   2.920750
llada_countdown early_trace_regret              confidence    inverted 0.972369           0.921557                  0.94   2.773237
llada_countdown early_trace_regret                 entropy         raw 0.500000           0.338954                  0.85   2.507714
llada_countdown early_trace_regret                  margin    inverted 0.954553           0.848305                  0.83   2.448709
llada_countdown early_trace_regret                      kl         raw 0.547040           0.348918                  0.51   1.504629
llada_countdown early_trace_regret               top1_flip         raw 0.558646           0.397021                  0.79   2.330699
llada_countdown early_trace_regret               runlength    inverted 0.866946           0.667247                  0.76   2.242192
llada_countdown early_trace_regret right_boundary_nearness    inverted 0.877528           0.732662                  0.70   2.065176
   illada_gsm8k   all_trace_regret learned_trace_predictor         raw 0.894731           0.850629                  0.93   2.004449
   illada_gsm8k   all_trace_regret              confidence    inverted 0.893601           0.852095                  0.93   2.004449
   illada_gsm8k   all_trace_regret                 entropy         raw 0.500000           0.463968                  0.00   0.000000
   illada_gsm8k   all_trace_regret                  margin    inverted 0.875910           0.799078                  0.67   1.444066
   illada_gsm8k   all_trace_regret                      kl         raw 0.549532           0.456571                  0.14   0.301745
   illada_gsm8k   all_trace_regret               top1_flip         raw 0.578395           0.520215                  0.68   1.465619
   illada_gsm8k   all_trace_regret               runlength    inverted 0.613736           0.564755                  0.38   0.819022
   illada_gsm8k   all_trace_regret right_boundary_nearness         raw 0.685539           0.606120                  0.65   1.400959
   illada_gsm8k early_trace_regret learned_trace_predictor         raw 0.879894           0.861121                  0.94   1.789042
   illada_gsm8k early_trace_regret              confidence    inverted 0.879381           0.862682                  0.92   1.750977
   illada_gsm8k early_trace_regret                 entropy         raw 0.500000           0.525421                  0.00   0.000000
   illada_gsm8k early_trace_regret                  margin    inverted 0.858307           0.813478                  0.82   1.560654
   illada_gsm8k early_trace_regret                      kl         raw 0.547770           0.516012                  0.22   0.418712
   illada_gsm8k early_trace_regret               top1_flip         raw 0.570008           0.572181                  0.67   1.275168
   illada_gsm8k early_trace_regret               runlength    inverted 0.584804           0.598905                  0.56   1.065812
   illada_gsm8k early_trace_regret right_boundary_nearness         raw 0.787360           0.770588                  0.85   1.617751
```

## Matched causal controls
```text
         candidate_group  candidates  forced_applied_rate  forced_token_survival_rate  final_token_change_rate  baseline_accuracy  forced_accuracy  answer_correct_change_rate
      matched_non_regret         100                  1.0                         1.0                     0.00               0.69             0.71                        0.06
matched_random_high_conf         100                  1.0                         1.0                     0.03               0.69             0.68                        0.01
      prospective_regret         100                  1.0                         1.0                     0.88               0.69             0.63                        0.16
```

## Oracle upper bound
```text
 examples  baseline_accuracy  oracle_accuracy  accuracy_delta  correctness_change_rate
      100               0.69             0.69             0.0                      0.0
```
