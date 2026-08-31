# Settlement Fate Paired Bootstrap

```json
{
  "mode": "commit",
  "runs": 2,
  "total_pairs": 416,
  "pairs_with_eos_eot": 0,
  "bootstrap_reps": 10000,
  "bootstrap_unit": "example_id clusters"
}
```

Bootstrap resamples example ids (cluster level), 95% percentile intervals.
Sign test is an exact binomial test on discordant pairs.

| slice               |   pairs |   examples | metric                          |   mean_delta |     ci_low |   ci_high |   sign_positive |   sign_negative |   sign_p_value |
|:--------------------|--------:|-----------:|:--------------------------------|-------------:|-----------:|----------:|----------------:|----------------:|---------------:|
| a40x3               |     120 |         40 | delta_non_target_changed        |   0.025      | 0          | 0.0583333 |               3 |               0 |          0.25  |
| a40x3               |     120 |         40 | delta_non_target_change_count   |   0.1        | 0          | 0.225     |               3 |               0 |          0.25  |
| a40x3               |     120 |         40 | delta_normalized_answer_changed |   0.0166667  | 0          | 0.0416667 |               2 |               0 |          0.5   |
| a40x3               |     120 |         40 | delta_answer_changed            |   0          | 0          | 0         |               0 |               0 |        nan     |
| a100x3              |     296 |         99 | delta_non_target_changed        |   0          | 0          | 0         |               0 |               0 |        nan     |
| a100x3              |     296 |         99 | delta_non_target_change_count   |   0.027027   | 0          | 0.0816327 |               1 |               0 |          1     |
| a100x3              |     296 |         99 | delta_normalized_answer_changed |   0          | 0          | 0         |               0 |               0 |        nan     |
| a100x3              |     296 |         99 | delta_answer_changed            |   0          | 0          | 0         |               0 |               0 |        nan     |
| pooled              |     416 |        139 | delta_non_target_changed        |   0.00721154 | 0          | 0.0167866 |               3 |               0 |          0.25  |
| pooled              |     416 |        139 | delta_non_target_change_count   |   0.0480769  | 0.00719424 | 0.105516  |               4 |               0 |          0.125 |
| pooled              |     416 |        139 | delta_normalized_answer_changed |   0.00480769 | 0          | 0.0120192 |               2 |               0 |          0.5   |
| pooled              |     416 |        139 | delta_answer_changed            |   0          | 0          | 0         |               0 |               0 |        nan     |
| pooled (no EOS/EOT) |     416 |        139 | delta_non_target_changed        |   0.00721154 | 0          | 0.0167866 |               3 |               0 |          0.25  |
| pooled (no EOS/EOT) |     416 |        139 | delta_non_target_change_count   |   0.0480769  | 0.00719424 | 0.105516  |               4 |               0 |          0.125 |
| pooled (no EOS/EOT) |     416 |        139 | delta_normalized_answer_changed |   0.00480769 | 0          | 0.0120192 |               2 |               0 |          0.5   |
| pooled (no EOS/EOT) |     416 |        139 | delta_answer_changed            |   0          | 0          | 0         |               0 |               0 |        nan     |