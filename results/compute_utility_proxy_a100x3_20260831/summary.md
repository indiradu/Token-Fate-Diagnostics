# Compute-Utility Proxy (drift-gated freeze simulation)

```json
{
  "run_dir": "results/settlement_fate_a_premise_math500_100x3_20260831",
  "examples": 100,
  "tokens": 6200,
  "steps": 64,
  "gen_length": 64,
  "bootstrap_unit": "example_id clusters",
  "caveats": [
    "Phase-6 proxy: token-step row updates avoided, not wall-clock.",
    "Freeze-to-base policies only; freeze-to-current-state is unquantifiable from this data.",
    "Freeze-vs-remove output equivalence (A2 vs A3 arms) untested.",
    "Harm panel is exploratory: labels come from immediate-freeze interventions on a top-drift-biased sample.",
    "No drift trajectories exist for the gen-256 runs (--skip-drift-time-rows); this analysis covers gen=64 only."
  ]
}
```

## Policy Sweep
| policy                                        |    tau |   k |   frozen_token_fraction |   mean_settle_delay_steps | settle_delay_ci   |   row_update_fraction_avoided_gen_rows | gen_rows_ci      |   row_update_fraction_avoided_incl_prompt_freeze | incl_prompt_ci   |
|:----------------------------------------------|-------:|----:|------------------------:|--------------------------:|:------------------|---------------------------------------:|:-----------------|-------------------------------------------------:|:-----------------|
| freeze_at_commit (upper bound, no drift gate) | nan    | nan |               1         |                   0       |                   |                             0.209744   | [0.2003, 0.2193] |                                         0.783386 | [0.7735, 0.7931] |
| drift_gate tau=0.1 k=1                        |   0.1  |   1 |               0.259677  |                  24.8473  | [24.29, 25.38]    |                             0.0499217  | [0.0456, 0.0547] |                                         0.623565 | [0.6066, 0.6404] |
| drift_gate tau=0.1 k=2                        |   0.1  |   2 |               0.0974194 |                  29.8055  | [29.48, 30.10]    |                             0.0174888  | [0.0154, 0.0200] |                                         0.591132 | [0.5725, 0.6094] |
| drift_gate tau=0.1 k=4                        |   0.1  |   4 |               0.0280645 |                  31.6869  | [31.55, 31.82]    |                             0.00517578 | [0.0042, 0.0062] |                                         0.578819 | [0.5596, 0.5979] |
| drift_gate tau=0.15 k=1                       |   0.15 |   1 |               0.443871  |                  18.9415  | [18.21, 19.66]    |                             0.088223   | [0.0819, 0.0950] |                                         0.661866 | [0.6462, 0.6775] |
| drift_gate tau=0.15 k=2                       |   0.15 |   2 |               0.240323  |                  25.3605  | [24.76, 25.90]    |                             0.0465517  | [0.0421, 0.0516] |                                         0.620195 | [0.6031, 0.6371] |
| drift_gate tau=0.15 k=4                       |   0.15 |   4 |               0.0937097 |                  29.6579  | [29.24, 30.02]    |                             0.0185074  | [0.0158, 0.0218] |                                         0.59215  | [0.5739, 0.6106] |
| drift_gate tau=0.2 k=1                        |   0.2  |   1 |               0.600161  |                  13.7684  | [12.98, 14.56]    |                             0.121763   | [0.1138, 0.1297] |                                         0.695406 | [0.6811, 0.7097] |
| drift_gate tau=0.2 k=2                        |   0.2  |   2 |               0.40371   |                  19.9547  | [19.13, 20.73]    |                             0.0815934  | [0.0752, 0.0886] |                                         0.655236 | [0.6395, 0.6710] |
| drift_gate tau=0.2 k=4                        |   0.2  |   4 |               0.209194  |                  25.9106  | [25.32, 26.47]    |                             0.0428359  | [0.0384, 0.0477] |                                         0.616479 | [0.5992, 0.6338] |
| drift_gate tau=0.3 k=1                        |   0.3  |   1 |               0.793387  |                   7.52516 | [6.85, 8.24]      |                             0.161743   | [0.1528, 0.1708] |                                         0.735386 | [0.7233, 0.7474] |
| drift_gate tau=0.3 k=2                        |   0.3  |   2 |               0.66871   |                  11.1837  | [10.40, 12.00]    |                             0.138786   | [0.1301, 0.1479] |                                         0.712429 | [0.6992, 0.7253] |
| drift_gate tau=0.3 k=4                        |   0.3  |   4 |               0.506613  |                  15.9319  | [15.15, 16.68]    |                             0.107439   | [0.1004, 0.1150] |                                         0.681082 | [0.6663, 0.6958] |
| drift_gate tau=0.4 k=1                        |   0.4  |   1 |               0.885323  |                   4.4421  | [3.90, 5.04]      |                             0.181388   | [0.1725, 0.1908] |                                         0.755031 | [0.7439, 0.7662] |
| drift_gate tau=0.4 k=2                        |   0.4  |   2 |               0.797581  |                   6.89548 | [6.19, 7.67]      |                             0.166335   | [0.1570, 0.1759] |                                         0.739978 | [0.7282, 0.7517] |
| drift_gate tau=0.4 k=4                        |   0.4  |   4 |               0.688548  |                   9.81677 | [9.07, 10.59]     |                             0.14739    | [0.1384, 0.1564] |                                         0.721032 | [0.7082, 0.7336] |
| drift_gate tau=0.5 k=1                        |   0.5  |   1 |               0.931774  |                   2.91032 | [2.49, 3.45]      |                             0.1911     | [0.1820, 0.2001] |                                         0.764743 | [0.7538, 0.7753] |
| drift_gate tau=0.5 k=2                        |   0.5  |   2 |               0.870968  |                   4.32032 | [3.75, 4.98]      |                             0.182388   | [0.1730, 0.1919] |                                         0.756031 | [0.7450, 0.7669] |
| drift_gate tau=0.5 k=4                        |   0.5  |   4 |               0.787742  |                   6.23097 | [5.55, 6.99]      |                             0.170325   | [0.1609, 0.1797] |                                         0.743968 | [0.7321, 0.7554] |
| drift_gate tau=0.75 k=1                       |   0.75 |   1 |               0.974032  |                   1.59597 | [1.37, 1.93]      |                             0.199595   | [0.1904, 0.2089] |                                         0.773237 | [0.7631, 0.7834] |
| drift_gate tau=0.75 k=2                       |   0.75 |   2 |               0.943065  |                   1.90516 | [1.59, 2.31]      |                             0.197548   | [0.1882, 0.2069] |                                         0.771191 | [0.7608, 0.7815] |
| drift_gate tau=0.75 k=4                       |   0.75 |   4 |               0.887903  |                   2.65968 | [2.14, 3.29]      |                             0.192849   | [0.1834, 0.2025] |                                         0.766492 | [0.7562, 0.7771] |
| drift_gate tau=1.0 k=1                        |   1    |   1 |               0.994839  |                   1.11468 | [1.04, 1.22]      |                             0.202577   | [0.1934, 0.2119] |                                         0.77622  | [0.7660, 0.7863] |
| drift_gate tau=1.0 k=2                        |   1    |   2 |               0.974677  |                   1.19742 | [1.09, 1.34]      |                             0.202047   | [0.1929, 0.2114] |                                         0.77569  | [0.7655, 0.7858] |
| drift_gate tau=1.0 k=4                        |   1    |   4 |               0.937097  |                   1.41468 | [1.26, 1.61]      |                             0.200619   | [0.1915, 0.2099] |                                         0.774261 | [0.7640, 0.7842] |

## Semantic vs Commit vs Drift Settlement Timing
```json
{
  "tokens": 6400,
  "tokens_with_semantic_stability_before_commit": 6400,
  "tokens_drift_settled_tau0.3_k2": 4146,
  "mean_semantic_lead_steps": 9.93265625,
  "semantic_lead_ci": [
    9.5017578125,
    10.357234374999999
  ],
  "mean_drift_settle_lag_steps_after_commit": 2.047515677761698,
  "drift_lag_ci": [
    2.027953123676965,
    2.0700442033984774
  ],
  "note": "semantic stability arrives before commit; drift settlement arrives after \u2014 the gap is compute a drift-safe gate forfeits relative to a semantic-only gate."
}
```