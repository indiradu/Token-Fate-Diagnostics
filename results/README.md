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

Every strict-intervention directory includes its aggregate group summary and
paired bootstrap comparisons where applicable.

For definitions, split rules, and interpretation, read:

- `../docs/causal_robustness_protocol.md`
- `../docs/causal_robustness_results.md`
