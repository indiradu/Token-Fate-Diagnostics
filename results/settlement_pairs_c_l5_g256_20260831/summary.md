# Settlement Fate Paired Bootstrap

```json
{
  "mode": "clamp",
  "runs": 1,
  "total_pairs": 80,
  "pairs_with_eos_eot": 0,
  "bootstrap_reps": 10000,
  "bootstrap_unit": "example_id clusters"
}
```

Bootstrap resamples example ids (cluster level), 95% percentile intervals.
Sign test is an exact binomial test on discordant pairs.

| slice     |   pairs |   examples | metric                          |   mean_delta |   ci_low |   ci_high |   sign_positive |   sign_negative |   sign_p_value |
|:----------|--------:|-----------:|:--------------------------------|-------------:|---------:|----------:|----------------:|----------------:|---------------:|
| c_l5_g256 |      80 |         40 | delta_non_target_changed        |       0.25   |   0.125  |    0.375  |              24 |               4 |    0.000179991 |
| c_l5_g256 |      80 |         40 | delta_non_target_change_count   |      28.175  |  14.5372 |   41.8378 |              33 |               7 |    4.2277e-05  |
| c_l5_g256 |      80 |         40 | delta_normalized_answer_changed |       0.1375 |   0.05   |    0.2375 |              12 |               1 |    0.00341797  |
| c_l5_g256 |      80 |         40 | delta_answer_changed            |       0.025  |   0      |    0.0625 |               2 |               0 |    0.5         |