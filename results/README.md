# Reported Results Manifest

This repository includes compact, inspectable aggregate artifacts behind the
reported tables and intervals. It intentionally excludes raw generation traces,
candidate rows, hidden-state arrays, locally fitted `.joblib` models, and
scheduler outputs; those files are unnecessary to inspect the reported metrics
and can contain complete model generations.

## Main Result Artifacts

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `commitment_signal_tables_workshop/` | cross-task signal tables, baseline comparison, oracle summary | Broad trace-regret prediction across tasks and signals. |
| `robustness_gsm8k_frozen_selector_analysis/` | example-level fixed-selector bootstrap | Main positive high-confidence GSM8K result. |
| `robustness_gsm8k_highconf_analysis_final/` | residual-selector bootstrap and confidence-conditioned tables | Shows the residual-only selector is statistically tied with confidence. |
| `robustness_llada_gsm8k_general_paired_500/` | aggregate intervention summary and pair bootstrap | Strict GSM8K intervention for the frozen general selector. |
| `robustness_llada_gsm8k_highconf_paired_500/` | aggregate intervention summary and pair bootstrap | Strict GSM8K intervention for the residual selector. |
| `robustness_llada_math500_highconf_paired/` | disjoint MATH500 aggregate intervention and bootstrap artifacts | Cross-task exploratory replication. |
| `logit_lens_suite_30_r3/` | layer and slice summaries plus attention-probe results | Mechanistic logit-lens and attention diagnostics. |
| `token_fate_layerwise_wide_30_r2/probes/` | aggregate layerwise probe metrics and fate summaries | Hidden-state probe comparison. |

## PhaseLock Premise Audit (2026-08-31 / 09-01)

Counterfactual audit of the PhaseLock gates. Every run's
`committed_token_drift.csv` carries the premise-A evidence (post-commit drift
under a fixed identity). `clamp` runs additionally freeze a committed token's
hidden row against a drift-matched same-trajectory control (premises B and C);
`commit` runs additionally force a stable token to lock early against a
same-position late-commit control (the auxiliary timing test). All runs exclude
EOS/EOT candidates and use disjoint example offsets.

**Label note (realigned 2026-09-01).** `A` is *semantic commitment does not
imply representation settlement* — its evidence is the post-commit drift in
`committed_token_drift.csv` (identity is fixed by construction because a
transferred token is never remasked), not an intervention. The `commit`-arm
runs below measure the **auxiliary** commitment-timing question — is committing
*earlier than the decoder would* harmful? — which is reported beside the ladder,
not as premise A. `B` is an existence claim about the set `C=1, R=0`; clamp
candidates are drift-stratified by construction, so no directory here contains
an unbiased population base rate. `U` is a separate systems proposition about
latency and is **not tested** by any of these runs.

| Directory | Contents | Role in the study |
| --- | --- | --- |
| `settlement_fate_a_premise_math500_40x3_20260831/` | commit-arm summary, run config | Auxiliary commitment-timing test, mixed MATH500, gen 64, 120 pairs. Also carries premise-A drift rows. |
| `settlement_fate_a_premise_math500_100x3_20260831/` | commit-arm summary, run config | Auxiliary timing replication, 296 pairs; paired deltas are zero. Premise-A drift: 92.0% of 6,400 rows drift > 0.25. |
| `settlement_fate_a_premise_math500l5_40x3_g256_20260831/` | commit-arm summary, run config | Auxiliary timing test, level-5 gen 256; null, but candidate confidence is higher by construction. Premise-A drift: 96.7% of 10,240 rows drift > 0.25. |
| `settlement_fate_c_premise_math500_100x3_20260831/` | clamp-arm summary, run config | Premises B and C, mixed MATH500, gen 64, 300 pairs. |
| `settlement_fate_c_premise_math500l5_40x2_g256_20260831/` | clamp-arm summary, run config | Premises B and C, level-5 gen 256, 80 pairs; the run where C resolves. |
| `settlement_pairs_a_pooled_20260831/` | example-clustered bootstrap and sign tests | Auxiliary earliness attribution: `+0.7` pp non-target change, 95% CI `[0.0, +1.7]`; raw early-commit harm is `0.93%` (5 of 536). Not premise A. |
| `settlement_pairs_c100x3_20260831/` | example-clustered bootstrap and sign tests | Premise C at gen 64: `+3.3` pp, 95% CI `[-2.0, +8.7]`, unresolved. |
| `settlement_pairs_c_l5_g256_20260831/` | example-clustered bootstrap and sign tests | Premise C at level-5 gen 256: `+25.0` pp, 95% CI `[+12.5, +37.5]`, sign test `p = 0.0002`. |
| `settlement_pairs_a_l5_g256_20260831/` | example-clustered bootstrap | Auxiliary timing test at level-5 gen 256; all deltas exactly zero. |
| `compute_utility_proxy_c100x3_20260831/` | drift-gate policy sweep, harm-by-drift bins | Algorithmic proxy for the U proposition (row updates avoided, **not latency**): freeze-at-commit ceiling `20.8%`; `tau=0.3` keeps `13.7%`. U itself is untested. |
| `compute_utility_proxy_a100x3_20260831/` | drift-gate policy sweep, settlement timing | Independent replication of the sweep plus semantic-vs-drift settlement times. |

Every strict-intervention directory includes its aggregate group summary and
paired bootstrap comparisons where applicable.

For definitions, split rules, and interpretation, read:

- `../docs/causal_robustness_protocol.md`
- `../docs/causal_robustness_results.md`
