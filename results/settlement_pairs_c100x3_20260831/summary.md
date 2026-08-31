# Settlement Fate Paired Bootstrap

```json
{
  "mode": "clamp",
  "runs": 1,
  "total_pairs": 300,
  "pairs_with_eos_eot": 0,
  "bootstrap_reps": 10000,
  "bootstrap_unit": "example_id clusters"
}
```

Bootstrap resamples example ids (cluster level), 95% percentile intervals.
Sign test is an exact binomial test on discordant pairs.

| slice       |   pairs |   examples | metric                          |   mean_delta |     ci_low |   ci_high |   sign_positive |   sign_negative |   sign_p_value |
|:------------|--------:|-----------:|:--------------------------------|-------------:|-----------:|----------:|----------------:|----------------:|---------------:|
| c100x3_eosx |     300 |        100 | delta_non_target_changed        |    0.0333333 | -0.02      | 0.0866667 |              36 |              26 |       0.252854 |
| c100x3_eosx |     300 |        100 | delta_non_target_change_count   |    1.13667   |  0.0365833 | 2.28667   |              44 |              31 |       0.165428 |
| c100x3_eosx |     300 |        100 | delta_normalized_answer_changed |    0.0133333 | -0.0266667 | 0.0533333 |              18 |              14 |       0.596615 |
| c100x3_eosx |     300 |        100 | delta_answer_changed            |    0.01      | -0.0133333 | 0.0333333 |               6 |               3 |       0.507812 |