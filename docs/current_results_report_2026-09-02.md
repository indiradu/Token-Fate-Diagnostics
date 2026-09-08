# Current Results Report - 2026-09-02

This report summarizes the current experimental state of the repository from
local artifacts. It intentionally includes positive, null, negative,
inconclusive, smoke, and availability results.

Primary index: `results/README.md`.

Verification run during inspection:

```bash
python3 -m pytest -q tests
# 26 passed in 1.78s

python3 -m compileall src scripts tests
# passed
```

## 1. Core Problem

The original repository problem was premature token commitment in diffusion
language models: an intermediate denoising token can look confident before its
final fate is known.

The current PhaseLock problem is sharper:

1. Token-identity stability is not the same as reference-state settlement.
2. Reference-freeze safety is not the same as commitment safety.
3. Row-compute removal is a systems utility question, not automatically implied
   by safe reference freezing.

The practical mistake being tested is: efficient diffusion-LM decoders often
treat "the token looks stable" as if it licenses several different actions at
once - fixing the token identity, freezing the reference state exposed to other
positions, and removing future row compute. The current experiments show these
must be separated.

## 2. Current Availability Problem

There is a real reproducibility/packaging issue right now.

`results/README.md` says compact aggregate artifacts are the checkable evidence,
while raw traces and generated text stay local because they may contain full
model outputs. That is sensible. But the newest Phase 1 and Phase 2 aggregate
directories are currently ignored by git and have zero tracked files.

These directories are checkable on this machine, but not available to someone
who clones the branch unless they are force-added:

- `results/lock_admission_bound_phase0_20260901/`
- `results/lock_predicate_audit_g64_20260901/`
- `results/lock_predicate_audit_l5g256_20260901/`
- `results/phase2_rulelock_g64/`
- `results/phase2_pairs_g64_20260902/`
- `results/phase2_rulelock_l5_g256/`
- `results/phase2_pairs_l5g256_20260902/`

Evidence:

```bash
git status --short --ignored results/phase2_pairs_g64_20260902
# !! results/phase2_pairs_g64_20260902/

git check-ignore -v results/phase2_pairs_g64_20260902/summary.md
# .gitignore:17:results/** results/phase2_pairs_g64_20260902/summary.md
```

So: the results are locally available and checkable, but the newest ones are not
yet packaged into git. This is the biggest operational problem with the current
state.

## 3. Experiment Family A - Token-Fate Prediction

Question: can token fate be predicted from decoding traces?

Source files:

- `results/commitment_signal_tables_workshop/cross_task_commitment_summary.csv`
- `results/commitment_signal_tables_workshop/commitment_signal_leaderboard.csv`

### Cross-task trace prediction

| Run | Learned trace AUROC | Confidence AUROC | Learned trace AP | Confidence AP | Interpretation |
| --- | ---: | ---: | ---: | ---: | --- |
| `illada_gsm8k` | 0.8947 | 0.8936 | 0.8506 | 0.8521 | Trace and confidence are essentially tied; AP is slightly worse for trace. |
| `llada_countdown` | 0.9849 | 0.9699 | 0.9482 | 0.8953 | Trace clearly beats confidence. |
| `llada_gsm8k` | 0.8873 | 0.8864 | 0.8021 | 0.8054 | AUROC barely improves; AP is slightly worse. |
| `llada_math500` | 0.9015 | 0.8994 | 0.8473 | 0.8470 | Trace barely improves over confidence. |

Analysis:

- Token fate is highly predictable.
- But the broad signal is mostly captured by confidence on GSM8K/MATH500/iLLaDA.
- The strong positive exception in this table is countdown.
- This means the paper should not claim novelty as "we can predict token fate."
  The stronger claim is where prediction remains useful after cheap stability
  signals are already high.

## 4. Experiment Family B - High-Confidence Prediction

Question: in the region where tokens look safe by confidence, do trace features
still add predictive information?

Source files:

- `results/robustness_gsm8k_frozen_selector_analysis/fixed_selector_bootstrap.csv`
- `results/robustness_gsm8k_highconf_analysis_final/high_confidence_selector_bootstrap.csv`
- `results/robustness_gsm8k_highconf_analysis_final/predictive_metric_bootstrap.csv`
- `results/robustness_llada_math500_highconf_paired/analysis/predictive_metric_bootstrap.csv`

### GSM8K: frozen general selector

The strongest predictive result:

| Slice | Metric | Delta vs confidence | 95% CI | Examples | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| fixed selector, high-confidence early | AUROC | +0.0174 | [+0.0066, +0.0272] | 500 | 79,558 |
| fixed selector, high-confidence early | AP | +0.0124 | [+0.0032, +0.0218] | 500 | 79,558 |

Analysis:

- This is the cleanest positive predictive result.
- It is statistically resolved.
- It supports the claim that trace information has residual value in the
  apparently safe high-confidence region.

### GSM8K: residual/high-confidence-only selector

| Slice | Metric | Delta vs confidence | 95% CI | Examples | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| selector high-confidence early | AUROC | +0.0035 | [-0.0102, +0.0179] | 500 | 79,558 |
| selector high-confidence early | AP | -0.0005 | [-0.0081, +0.0078] | 500 | 79,558 |

Analysis:

- This is a null result.
- The residual-only selector is statistically tied with confidence.
- This should remain visible because it prevents overclaiming that every learned
  trace model beats confidence.

### GSM8K: `fate_lite` high-confidence slices

| Slice | Metric | Delta vs confidence | 95% CI | Examples | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| high_confidence_q75 | AUROC | -0.2239 | [-0.3178, -0.1203] | 500 | 103,223 |
| high_confidence_q75 | AP | -0.0064 | [-0.0107, -0.0033] | 500 | 103,223 |
| early_high_confidence_q75 | AUROC | -0.1351 | [-0.2219, -0.0423] | 500 | 56,527 |
| early_high_confidence_q75 | AP | -0.0030 | [-0.0097, +0.0075] | 500 | 56,527 |

Analysis:

- This is not positive. The `fate_lite` setup is worse than confidence in
  AUROC on these high-confidence GSM8K slices.
- The useful predictive result is specifically the frozen general selector, not
  every lightweight trace variant.

### MATH500: high-confidence prediction

| Slice | Metric | Delta vs confidence | 95% CI | Examples | Rows |
| --- | --- | ---: | ---: | ---: | ---: |
| high_confidence_q75 | AUROC | +0.0085 | [-0.0153, +0.0333] | 100 | 26,230 |
| high_confidence_q75 | AP | +0.0191 | [-0.0035, +0.0769] | 100 | 26,230 |
| early_high_confidence_q75 | AUROC | +0.0052 | [-0.0258, +0.0346] | 100 | 16,090 |
| early_high_confidence_q75 | AP | +0.0213 | [-0.0063, +0.0884] | 100 | 16,090 |

Analysis:

- MATH500 is directionally positive but not statistically resolved.
- The example count is much smaller than GSM8K.
- This is exploratory replication, not a settled result.

## 5. Experiment Family C - Strict Counterfactual Commitment Interventions

Question: if the learned selector picks high-confidence risky tokens, does
forcing those token commitments cause more downstream output change than
confidence-matched controls?

Source files:

- `results/robustness_llada_gsm8k_general_paired_500/group_summary.csv`
- `results/robustness_llada_gsm8k_general_paired_500/analysis/causal_selector_comparisons_10k.csv`
- `results/robustness_llada_gsm8k_highconf_paired_500/group_summary.csv`
- `results/robustness_llada_gsm8k_highconf_paired_500/analysis/causal_selector_comparisons_10k.csv`
- `results/robustness_llada_math500_highconf_paired/analysis/causal_selector_comparisons.csv`

### GSM8K frozen general selector

| Endpoint | Learned rate | Matched control rate | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| non-target token changed | 6.91% | 5.99% | +0.92 pp | [-2.76, +4.61] |
| non-target token count | 0.770 | 0.825 | -0.055 | [-0.548, +0.406] |
| final token changed | 6.91% | 5.53% | +1.38 pp | [-2.30, +5.07] |
| answer correctness changed | 0.46% | 0.46% | 0.00 pp | [0.00, 0.00] |

Analysis:

- This is a strict causal null.
- The learned selector did not resolve larger downstream harm than the matched
  high-confidence low-risk control.
- This is central, not a footnote: prediction did not automatically translate
  into causal enrichment.

### GSM8K high-confidence residual selector

| Endpoint | Learned rate | Matched control rate | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| non-target token changed | 5.83% | 5.83% | 0.00 pp | [-3.14, +3.14] |
| non-target token count | 0.946 | 0.982 | -0.036 | [-0.700, +0.570] |
| final token changed | 4.48% | 5.38% | -0.90 pp | [-4.04, +2.24] |
| answer correctness changed | 0.45% | 0.45% | 0.00 pp | [0.00, 0.00] |

Analysis:

- Also null.
- This confirms the residual selector should not be described as causally
  superior to confidence on GSM8K.

### MATH500 strict intervention

| Endpoint | Learned rate | Matched control rate | Delta | 95% CI |
| --- | ---: | ---: | ---: | ---: |
| non-target token changed | 9.30% | 4.65% | +4.65 pp | [0.00, +11.63] |
| non-target token count | 1.116 | 0.488 | +0.628 | [0.00, +1.535] |
| final token changed | 6.98% | 2.33% | +4.65 pp | [0.00, +11.63] |
| answer correctness changed | 2.33% | 0.00% | +2.33 pp | [0.00, +6.98] |

Analysis:

- Directionally positive, but small and exploratory: 43 matched pairs.
- The interval touches zero for key endpoints.
- This should be reported as cross-task exploratory evidence, not as a resolved
  causal result.

## 6. Experiment Family D - Mechanistic / Representation Diagnostics

Question: do hidden states and logit-lens features contain information about
token fate?

Source files:

- `results/token_fate_layerwise_wide_30_r2/probes/probe_results.csv`
- `results/token_fate_layerwise_wide_30_r2/probes/fate_summary_eval.csv`
- `results/logit_lens_suite_30_r3/eval/layer_summary.csv`
- `results/logit_lens_suite_30_r3/eval/slice_summary.csv`
- `results/logit_lens_suite_30_r3/attention_probe/attention_probe_results.csv`

### Layerwise probes

| Target | Probe class | Best probe | AUROC | AP |
| --- | --- | --- | ---: | ---: |
| all trace regret | logits only | `all_trace_regret/logits` | 0.9067 | 0.7631 |
| all trace regret | hidden only | `layer_24_hidden` | 0.7782 | 0.5682 |
| all trace regret | hidden + logits | `layer_32_hidden_dynamics_plus_logits` | 0.9068 | 0.7639 |
| early trace regret | logits only | `early_trace_regret/logits` | 0.8815 | 0.7951 |
| early trace regret | hidden only | `layer_32_hidden` | 0.7347 | 0.6474 |
| early trace regret | hidden + logits | `layer_16_hidden_dynamics_plus_logits` | 0.8856 | 0.7989 |

Important null/negative detail:

- `stable_high_conf_trace_regret` is skipped because labels are single-class in
  this run. There is no valid AUROC/AP there.
- Hidden-only probes contain signal but do not match logits.
- Hidden+logit probes barely improve on logits.

### Fate categories in 30-example eval probe set

| Fate | Count | Fraction |
| --- | ---: | ---: |
| early_correct | 765 | 43.66% |
| oscillating | 617 | 35.22% |
| late_corrected | 317 | 18.09% |
| early_wrong | 51 | 2.91% |
| other | 2 | 0.11% |

### Logit lens

Layer-32 aggregate:

- `lens_top1_final_rate`: 83.21%
- `lens_top1_surface_rate`: 99.31%
- `final_top10_rate`: 97.80%

For trace-regret rows at layer 32:

| Slice | Rows | Lens top-1 final rate | Final top-10 rate | Mean final rank |
| --- | ---: | ---: | ---: | ---: |
| late, regret | 752 | 1.73% | 92.55% | 4.90 |
| early, regret | 1,541 | 1.10% | 84.10% | 7.17 |

Analysis:

- Wrong surface commitments often still keep the eventual final token nearby in
  the layer-32 distribution.
- This supports an interpretability story, not an online deployment result by
  itself.

### Attention probe

| Probe | AUROC | AP | Interpretation |
| --- | ---: | ---: | --- |
| trace features | 0.9049 | 0.6118 | Strong trace baseline. |
| attention only | 0.5430 | 0.1677 | Weak; nearly unhelpful alone. |
| trace + attention | 0.9057 | 0.5880 | AUROC barely moves, AP worsens. |
| trace + lens | 0.9994 | 0.9950 | Mechanistic/retrospective diagnostic, not a direct online policy. |

## 7. Experiment Family E - PhaseLock Premise A

Question: after the token identity is fixed, does the representation keep
moving?

Source files:

- `results/lock_admission_bound_phase0_20260901/summary.json`
- `results/settlement_fate_a_premise_math500_100x3_20260831/summary.json`
- `results/settlement_fate_a_premise_math500l5_40x3_g256_20260831/summary.json`
- `results/phase2_rulelock_g64/summary.json`
- `results/phase2_rulelock_l5_g256/summary.json`

### Predicate-free Phase 0 bound

Pooled over `a100x3` and `c100x3`, EOS/EOT excluded:

| Frozen horizon fraction | Tokens | Median oracle lower bound | 95% CI | Fraction > 0.25 |
| ---: | ---: | ---: | ---: | ---: |
| 1.0 | 11,309 | 0.438 | [0.432, 0.444] | 92.8% |
| 0.9 | 11,309 | 0.205 | [0.201, 0.210] | 36.0% |
| 0.75 | 11,309 | 0.120 | [0.116, 0.123] | 13.3% |
| 0.5 | 11,309 | 0.062 | [0.060, 0.065] | 4.4% |

Analysis:

- Lock-at-commit strongly supports A: identity-fixed tokens keep moving.
- But the free lower bound decays quickly as the lock is delayed.
- Therefore Phase 0 convicts naive immediate locks, but it cannot resolve
  delayed rules like SureLock. It never acquits.

### Premise A in long-horizon level-5 gen-256 regime

From the Phase 1/2 long-horizon artifacts:

- Premise A lock-at-commit: median lower bound 0.514, 96.8% above 0.25.
- Earliest logged lock after commit: median lower bound 0.354, 76.3% above 0.25.
- Phase 2 `phase2_rulelock_l5_g256`: median max drift 0.516 over 13,716 tokens
  with drift measurement.

Analysis:

- Representation movement is more sustained at gen 256 than at gen 64.
- This is the regime where compute-saving methods matter more, so the problem
  gets harder where it matters.

## 8. Experiment Family F - Auxiliary Early-Commit Timing

Question: if a token is already stable later, is forcing it earlier harmful?

Important: this is not premise A after the 2026-09-01 label realignment. It is
an auxiliary commitment-timing result.

Source files:

- `results/settlement_fate_a_premise_math500_40x3_20260831/summary.json`
- `results/settlement_fate_a_premise_math500_100x3_20260831/summary.json`
- `results/settlement_fate_a_premise_math500l5_40x3_g256_20260831/summary.json`
- `results/settlement_pairs_a_pooled_20260831/paired_bootstrap.csv`
- `results/settlement_pairs_a_l5_g256_20260831/paired_bootstrap.csv`

### Gen-64 pooled auxiliary timing

| Slice | Pairs | Examples | Endpoint | Delta | 95% CI | Sign test |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| pooled | 416 | 139 | non-target changed | +0.72 pp | [0.00, +1.68] | p = 0.25 |
| pooled | 416 | 139 | non-target count | +0.048 tokens | [+0.007, +0.106] | p = 0.125 |
| pooled | 416 | 139 | normalized answer changed | +0.48 pp | [0.00, +1.20] | p = 0.5 |
| pooled | 416 | 139 | correctness changed | 0.00 pp | [0.00, 0.00] | no discordant pairs |

### Level-5 gen-256 auxiliary timing

| Pairs | Examples | All deltas |
| ---: | ---: | --- |
| 120 | 40 | exactly zero for non-target change, change count, normalized-answer change, and correctness change |

Analysis:

- Early semantic commit timing is mostly null under the current candidate
  construction.
- This strengthens the pivot away from "early commit is the main harm" and
  toward "stale reference freeze is the main harm."

## 9. Experiment Family G - PhaseLock Premises B/C, Drift-Matched Freeze

Question: for semantically safe committed tokens, does freezing the stale
reference change downstream outputs, and does larger drift predict more harm?

Source files:

- `results/settlement_fate_c_premise_math500_100x3_20260831/summary.json`
- `results/settlement_pairs_c100x3_20260831/paired_bootstrap.csv`
- `results/settlement_fate_c_premise_math500l5_40x2_g256_20260831/summary.json`
- `results/settlement_pairs_c_l5_g256_20260831/paired_bootstrap.csv`
- Smoke/local: `results/settlement_fate_c_math500_10x3/summary.json`,
  `results/settlement_fate_c_nonmath_10x3/summary.json`,
  `results/settlement_fate_second_gsm8k_5/summary.json`,
  `results/settlement_fate_second_math500_5/summary.json`

### Gen-64 MATH500, 100 examples

Absolute group rates:

| Arm | Candidates | Mean drift | Non-target changed | Mean changed tokens | Normalized answer changed | Correctness changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high drift | 300 | 0.932 | 24.7% | 3.33 | 10.7% | 2.0% |
| matched low drift | 300 | 0.366 | 21.3% | 2.20 | 9.3% | 1.0% |

Paired bootstrap:

| Endpoint | Delta high-minus-low | 95% CI | Sign test |
| --- | ---: | ---: | ---: |
| non-target changed | +3.3 pp | [-2.0, +8.7] | p = 0.253 |
| non-target count | +1.14 tokens | [+0.04, +2.29] | p = 0.165 |
| normalized answer changed | +1.3 pp | [-2.7, +5.3] | p = 0.597 |
| correctness changed | +1.0 pp | [-1.3, +3.3] | p = 0.508 |

Analysis:

- B is supported in the weak sense that freezing committed tokens changes
  outputs in both arms.
- C is unresolved at gen 64: high drift is directionally worse, but the main
  binary endpoint crosses zero.

### Level-5 gen-256 MATH500, 40 examples

Absolute group rates:

| Arm | Candidates | Mean drift | Non-target changed | Mean changed tokens | Normalized answer changed | Correctness changed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| high drift | 80 | 4.960 | 53.8% | 48.79 | 18.8% | 2.5% |
| matched low drift | 80 | 0.417 | 28.8% | 20.61 | 5.0% | 0.0% |

Paired bootstrap:

| Endpoint | Delta high-minus-low | 95% CI | Sign test |
| --- | ---: | ---: | ---: |
| non-target changed | +25.0 pp | [+12.5, +37.5] | p = 0.00018 |
| non-target count | +28.18 tokens | [+14.54, +41.84] | p = 0.000042 |
| normalized answer changed | +13.75 pp | [+5.0, +23.75] | p = 0.0034 |
| correctness changed | +2.5 pp | [0.0, +6.25] | p = 0.5 |

Analysis:

- C resolves in the long-horizon level-5 regime for token/output-text endpoints.
- Correctness still does not resolve.
- The harm is real at output-token and normalized-answer-text levels, but not
  yet established as accuracy harm.

### Smoke/local checks

These are useful but not paper-grade on their own:

| Directory | Result |
| --- | --- |
| `settlement_fate_second_gsm8k_5` | 5 GSM8K examples; high and low drift clamp arms both caused 0 non-target changes. A supported, B/C not supported on this easy sample. |
| `settlement_fate_second_math500_5` | 5 MATH500 examples; high drift 50% non-target change vs low drift 40%; tiny and directional. |
| `settlement_fate_c_math500_10x3` | 10 MATH500 examples; high drift 23.3% vs low drift 13.3%; delta +10 pp, small. |
| `settlement_fate_c_nonmath_10x3` | 10 local non-math examples; high drift 40.0% vs low drift 10.0%; local/synthetic, not benchmark-grade. |

## 10. Experiment Family H - Prior-Method Lock Admission Audit

Question: do SureLock-/LESS-style admission rules avoid representational
movement if evaluated at their own delayed lock step?

Source files:

- `docs/lock_admission_audit.md`
- `results/lock_admission_bound_phase0_20260901/summary.json`
- `results/lock_predicate_audit_g64_20260901/predicate_sweep.csv`
- `results/lock_predicate_audit_g64_20260901/frozen_threshold_eval.csv`
- `results/lock_predicate_audit_l5g256_20260901/predicate_sweep.csv`
- `results/lock_predicate_audit_l5g256_20260901/frozen_threshold_eval.csv`

### Gen-64 Phase 1

Representative rows:

| Rule / point | Savings | Lock rate | Median lower | Fraction lower > 0.25 | Median upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| premise A lock-at-commit reference | not in sweep | not in sweep | 0.442 | 91.6% | not the same bound |
| earliest logged lock | 0.969 | 100.0% | 0.266 | 55.0% | 2.224 |
| SureLock-style KL, tau=0 | 0.093 | 22.6% | 0.037 | 4.5% | 0.552 |
| LESS-style JSD+persistence, tau=0 | 0.118 | 26.1% | 0.039 | 4.7% | 0.612 |
| confidence-only, similar low savings | 0.105 | 23.6% | 0.068 | 11.6% | 0.822 |
| SureLock-style KL, about 0.5 target | 0.507 | 71.0% | 0.094 | 13.9% | 1.266 |
| confidence-only, about 0.5 target | 0.527 | 70.4% | 0.162 | 28.5% | 1.563 |

Analysis:

- SureLock-/LESS-style rules do carry information beyond confidence.
- They reduce provable post-lock movement at matched savings.
- But no rule is acquitted because the upper bound remains large.
- A single-step delay after commit explains a lot of the apparent contradiction:
  premise A at commit is severe, but the earliest logged post-commit lock is
  already much less severe.

### Level-5 gen-256 Phase 1

Representative rows:

| Rule / point | Savings | Lock rate | Median lower | Fraction lower > 0.25 | Median upper |
| --- | ---: | ---: | ---: | ---: | ---: |
| premise A lock-at-commit | not in sweep | not in sweep | 0.514 | 96.8% | not the same bound |
| earliest logged lock | 0.992 | 100.0% | 0.354 | 76.3% | 5.974 |
| SureLock-style KL, tau=0 | 0.433 | 72.0% | 0.093 | 15.3% | 2.629 |
| LESS-style JSD+persistence, tau=0 | 0.466 | 69.8% | 0.101 | 15.6% | 3.101 |
| confidence-only, matched-ish | 0.405 | 67.8% | 0.138 | 28.2% | 3.219 |

Analysis:

- The qualitative pattern replicates: KL/JSD stability is better than
  confidence at matched savings.
- The strongest problem remains: even the most conservative `tau=0` KL gate
  admits many tokens, and 15.3% still have provable post-lock movement above
  0.25.
- The upper bound is vacuous at gen 256, so this audit convicts but cannot
  certify settlement.

## 11. Experiment Family I - Phase 2 Rule-Lock Interventions

Question: does stale reference movement actually change decoded output when the
same token is frozen at commit versus at a SureLock-style rule lock?

Design:

- Within-token pair.
- Target arm: same token frozen at commit step.
- Control arm: same token frozen at the rule's own lock step.
- Rule: SureLock-style KL gate at `tau=0`, meaning adjacent-step posterior is
  numerically identical.
- Candidates sampled per example with a fixed seed, not ranked by drift.

Source files:

- `results/phase2_rulelock_g64/summary.json`
- `results/phase2_rulelock_g64/freeze_group_summary.csv`
- `results/phase2_pairs_g64_20260902/paired_bootstrap.csv`
- `results/phase2_rulelock_l5_g256/summary.json`
- `results/phase2_rulelock_l5_g256/freeze_group_summary.csv`
- `results/phase2_pairs_l5g256_20260902/paired_bootstrap.csv`
- Tests: `tests/test_rule_lock_selection.py`

### Gen-64 Phase 2

Absolute harm:

| Arm | Candidates | Non-target changed | Mean changed tokens | Normalized answer changed | Correctness changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| freeze at commit | 269 | 20.8% | 2.69 | 6.7% | 1.5% |
| freeze at rule lock (`tau=0`) | 269 | 6.3% | 0.35 | 0.7% | 0.4% |

Paired deltas:

| Endpoint | Delta commit-minus-rule | 95% CI | Sign test |
| --- | ---: | ---: | ---: |
| non-target changed | +14.5 pp | [+10.0, +19.3] | p = 2.8e-9 |
| non-target count | +2.34 tokens | [+1.49, +3.26] | p = 1.3e-9 |
| normalized answer changed | +5.9 pp | [+3.0, +9.3] | p = 1.4e-4 |
| correctness changed | +1.1 pp | [0.0, +2.6] | p = 0.375 |

Analysis:

- The rule's delay buys real harm reduction.
- But rule-lock freezing is not harmless: 6.3% of interventions still change
  downstream tokens even at `tau=0`.
- Correctness remains unresolved.

### Level-5 gen-256 Phase 2

Absolute harm:

| Arm | Candidates | Non-target changed | Mean changed tokens | Normalized answer changed | Correctness changed |
| --- | ---: | ---: | ---: | ---: | ---: |
| freeze at commit | 108 | 37.0% | 33.96 | 20.4% | 1.9% |
| freeze at rule lock (`tau=0`) | 108 | 21.3% | 12.42 | 8.3% | 0.0% |

Paired deltas:

| Endpoint | Delta commit-minus-rule | 95% CI | Sign test |
| --- | ---: | ---: | ---: |
| non-target changed | +15.7 pp | [+8.3, +24.1] | p = 9.1e-4 |
| non-target count | +21.5 tokens | [+11.2, +33.0] | p = 9.0e-3 |
| normalized answer changed | +12.0 pp | [+4.6, +20.4] | p = 4.4e-3 |
| correctness changed | +1.9 pp | [0.0, +4.6] | p = 0.5 |

Analysis:

- The rule's protection does not scale cleanly.
- The benefit of waiting is about the same in binary rate terms as gen 64
  (+15.7 pp vs +14.5 pp), but residual harm at the rule lock grows sharply:
  6.3% to 21.3% non-target change, 0.7% to 8.3% normalized answer change, and
  0.35 to 12.42 changed tokens per intervention.
- This is the strongest current evidence that posterior stability is not enough
  for reference-freeze safety.
- Correctness is still underpowered: 0.0% in the rule-lock arm is not proof of
  safety.

## 12. Experiment Family J - Compute Utility Proxy

Question: how much row-update work would drift-gated freezing avoid?

Source files:

- `results/compute_utility_proxy_c100x3_20260831/summary.json`
- `results/compute_utility_proxy_c100x3_20260831/policy_sweep.csv`
- `results/compute_utility_proxy_c100x3_20260831/harm_by_drift_bin.csv`
- `results/compute_utility_proxy_a100x3_20260831/summary.json`
- `results/compute_utility_proxy_a100x3_20260831/policy_sweep.csv`

### Row-update proxy

| Run | Policy | Frozen token fraction | Mean settle delay | Row-update fraction avoided |
| --- | --- | ---: | ---: | ---: |
| c100x3 | freeze at commit ceiling | 100.0% | 0.0 | 20.75% |
| c100x3 | drift gate tau=0.3, k=2 | 66.9% | 11.24 | 13.67% |
| a100x3 | freeze at commit ceiling | 100.0% | 0.0 | 20.97% |
| a100x3 | drift gate tau=0.3, k=2 | 66.9% | 11.18 | 13.88% |

Harm by drift bin in c100x3 immediate-freeze sample:

| Max drift bin | Interventions | Non-target change rate | Normalized answer change rate |
| --- | ---: | ---: | ---: |
| (0.0, 0.3] | 47 | 4.3% | 2.1% |
| (0.3, 0.5] | 249 | 24.5% | 11.2% |
| (0.5, 0.75] | 18 | 33.3% | 11.1% |
| (0.75, 1.0] | 274 | 24.1% | 9.9% |
| (1.0, inf] | 12 | 25.0% | 16.7% |

Analysis:

- This is not a wall-clock result.
- It measures token-step row updates avoided, not latency or throughput.
- It also does not test freeze-vs-row-removal equivalence.
- It is useful as a U proxy, but U is not actually established.

## 13. Overall Current Analysis

What is supported:

- Token fate is predictable from traces.
- Confidence is a very strong baseline, and often nearly matches trace features.
- A frozen general trace selector has a resolved high-confidence GSM8K
  predictive advantage.
- Token-identity stability does not imply reference-state settlement.
- Reference freezing can change downstream output.
- Drift predicts freeze harm in the long-horizon level-5 gen-256 regime.
- SureLock-/LESS-style posterior-stability rules reduce harm/movement relative
  to naive early freezing and relative to confidence at comparable savings.
- Even the most conservative rule-lock point tested (`tau=0`) is not harmless,
  especially at gen 256.

What is null or unresolved:

- GSM8K strict causal enrichment beyond confidence is null.
- Residual high-confidence selector vs confidence is null on GSM8K.
- `fate_lite` is worse than confidence in GSM8K high-confidence AUROC.
- MATH500 high-confidence prediction is directionally positive but unresolved.
- Gen-64 drift-matched C is unresolved on the main binary endpoint.
- Accuracy/correctness effects are unresolved almost everywhere.
- U as actual latency/throughput improvement is untested.

What is the main scientific problem now:

The project has good evidence that posterior/token stability and
reference-freeze safety are different. The current blocker for a stronger
efficiency paper is not whether there is movement or token-level harm. The
blocker is proving an operational controller that gives a useful quality/compute
frontier under real cache/row-removal mechanics, not only a hidden-row clamp
proxy.

What is the main repository problem now:

The newest results are locally checkable but not git-packaged. If this branch is
shared, several manifest-referenced Phase 1/2 artifacts will be missing because
`results/**` is ignored unless files are force-added.

## 14. Specific Next Steps

1. Package availability first: force-add compact summaries/tables for every
   Phase 1/2 directory mentioned in `results/README.md`.
2. Keep raw traces, full generations, hidden arrays, and scheduler logs ignored.
3. Add a small check script or manifest validator that fails if
   `results/README.md` mentions a directory with no tracked compact artifacts.
4. Run Phase 2 over more than `tau=0` if the paper wants a threshold frontier.
5. Run the secondary admitted-versus-rejected matched arm if the paper wants a
   clean B/C test inside the rule-admitted population.
6. Do not claim wall-clock speedup until a real cached-K/V or row-removal
   implementation is measured.
7. Keep correctness claims conservative unless the experiment is powered for
   answer-correctness deltas.

## Appendix - Every Local Result Directory

`tracked files = 0` means the directory exists locally but would not be present
from git unless explicitly force-added.

| Directory | In manifest | Tracked files | Local summary | Key check files |
| --- | ---: | ---: | ---: | --- |
| `commitment_signal_tables_workshop` | yes | 5 | no |  |
| `compute_utility_proxy_a100x3_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `policy_sweep.csv` |
| `compute_utility_proxy_c100x3_20260831` | yes | 4 | yes | `summary.md`, `summary.json`, `policy_sweep.csv` |
| `counterfactual_commit_30_r2` | no | 2 | yes | `summary.md`, `summary.json` |
| `lock_admission_bound_phase0_20260901` | yes | 0 | yes | `summary.md`, `summary.json` |
| `lock_predicate_audit_g64_20260901` | yes | 0 | yes | `summary.md`, `summary.json` |
| `lock_predicate_audit_l5g256_20260901` | yes | 0 | yes | `summary.md`, `summary.json` |
| `logit_lens_suite_30_r3` | yes | 6 | no |  |
| `phase1_postcommit_g64_eval` | no | 0 | yes | `summary.md`, `summary.json` |
| `phase1_postcommit_g64_train` | no | 0 | yes | `summary.md`, `summary.json` |
| `phase1_postcommit_l5_g256` | no | 0 | yes | `summary.md`, `summary.json` |
| `phase1_smoke_tiny` | no | 0 | yes | `summary.md`, `summary.json` |
| `phase2_pairs_g64_20260902` | yes | 0 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `phase2_pairs_l5g256_20260902` | yes | 0 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `phase2_rulelock_g64` | yes | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `phase2_rulelock_l5_g256` | yes | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `phase2_smoke_tiny` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `robustness_gsm8k_frozen_selector_analysis` | yes | 1 | no |  |
| `robustness_gsm8k_highconf_analysis_final` | yes | 4 | no |  |
| `robustness_llada_gsm8k_general_paired_500` | yes | 4 | yes | `summary.md`, `summary.json`, `group_summary.csv` |
| `robustness_llada_gsm8k_highconf_paired_500` | yes | 5 | yes | `summary.md`, `summary.json`, `group_summary.csv` |
| `robustness_llada_math500_highconf_paired` | yes | 9 | no |  |
| `settlement_fate_a_premise_math500_100x3_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `commit_group_summary.csv` |
| `settlement_fate_a_premise_math500_40x3_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `commit_group_summary.csv` |
| `settlement_fate_a_premise_math500l5_40x3_g256_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `commit_group_summary.csv` |
| `settlement_fate_a_smoke_tiny` | no | 0 | yes | `summary.md`, `summary.json`, `commit_group_summary.csv` |
| `settlement_fate_audit_gsm8k_2` | no | 0 | no |  |
| `settlement_fate_audit_local_math_2` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_audit_smoke` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_b_premise_math500_10x2_20260831` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_b_premise_math500_2x1_20260831` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_b_premise_math500_3x2_20260831` | no | 0 | no |  |
| `settlement_fate_c_math500_10x3` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_c_nonmath_10x3` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_c_premise_math500_100x3_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_c_premise_math500_40x2_20260831` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_c_premise_math500l5_40x2_g256_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_clamp_smoke_tiny` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_second_gsm8k_20` | no | 0 | no |  |
| `settlement_fate_second_gsm8k_5` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_second_local_math_2` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_fate_second_math500_5` | no | 0 | yes | `summary.md`, `summary.json`, `freeze_group_summary.csv` |
| `settlement_pairs_a100x3_20260831` | no | 0 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `settlement_pairs_a_l5_g256_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `settlement_pairs_a_math500_20260831` | no | 0 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `settlement_pairs_a_pooled_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `settlement_pairs_c100x3_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `settlement_pairs_c_l5_g256_20260831` | yes | 3 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `settlement_pairs_math500_pooled_20260831` | no | 0 | yes | `summary.md`, `summary.json`, `paired_bootstrap.csv` |
| `token_fate_layerwise_wide_30_r2` | no | 8 | no |  |
