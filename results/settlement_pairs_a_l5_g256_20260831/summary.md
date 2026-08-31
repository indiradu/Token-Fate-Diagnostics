# Settlement Fate Paired Bootstrap

```json
{
  "mode": "commit",
  "runs": 1,
  "total_pairs": 120,
  "pairs_with_eos_eot": 0,
  "bootstrap_reps": 10000,
  "bootstrap_unit": "example_id clusters"
}
```

Bootstrap resamples example ids (cluster level), 95% percentile intervals.
Sign test is an exact binomial test on discordant pairs.

| slice     |   pairs |   examples | metric                          |   mean_delta |   ci_low |   ci_high |   sign_positive |   sign_negative | sign_p_value   |
|:----------|--------:|-----------:|:--------------------------------|-------------:|---------:|----------:|----------------:|----------------:|:---------------|
| a_l5_g256 |     120 |         40 | delta_non_target_changed        |            0 |        0 |         0 |               0 |               0 |                |
| a_l5_g256 |     120 |         40 | delta_non_target_change_count   |            0 |        0 |         0 |               0 |               0 |                |
| a_l5_g256 |     120 |         40 | delta_normalized_answer_changed |            0 |        0 |         0 |               0 |               0 |                |
| a_l5_g256 |     120 |         40 | delta_answer_changed            |            0 |        0 |         0 |               0 |               0 |                |