# High-Confidence Robustness Results

Date: August 12, 2026.

This note supersedes earlier high-confidence causal interpretations that used
outcomes including the forced token itself, loose matching, or overlapping
MATH500 train/evaluation offsets.

## Protocol

- Model: `GSAI-ML/LLaDA-8B-Instruct` with confidence decoding, 64 denoising
  steps, generation length 64, and block length 32.
- GSM8K prediction/intervention split: selector fitting traces from training
  examples 0-99; held-out test examples 100-599.
- MATH500 split: test examples 0-99 for selector fitting and 100-199 for
  held-out evaluation/intervention. These offsets are disjoint.
- High-confidence candidates are early positions with confidence above the
  training 75th-percentile threshold.
- Strict controls are label-free, lower-score positions in the same example,
  exact global denoising step, confidence caliper `0.05`, and relative output
  position caliper `8`.
- The primary causal endpoint is `non_target_token_changed`: whether forcing
  one target changes at least one *other* output token. All confidence
  intervals resample examples or matched pairs, never individual trace rows.

## Prediction Beyond Confidence

| Evaluation | Selector | Units | Delta AUROC vs confidence (95% CI) | Delta AP vs confidence (95% CI) |
| --- | --- | ---: | ---: | ---: |
| GSM8K | Frozen general trace selector | 500 examples | +0.0174 [+0.0066, +0.0272] | +0.0124 [+0.0032, +0.0218] |
| GSM8K | High-confidence residual selector | 500 examples | +0.0035 [-0.0102, +0.0179] | -0.0005 [-0.0081, +0.0078] |
| MATH500 | High-confidence residual selector | 100 examples | +0.0228 [-0.0029, +0.0406] | +0.0312 [-0.0050, +0.1159] |

The frozen general selector is the only high-confidence model whose predictive
advantage is clearly nonzero on held-out GSM8K. Its raw metrics are AUROC
`0.8695` versus `0.8521` for confidence and AP `0.0569` versus `0.0446`.
Training a selector only inside the rare high-confidence slice did not improve
over confidence reliably.

## Strict Downstream Intervention

| Evaluation | Selector | Matched pairs | Target rate | Control rate | Difference (95% CI) |
| --- | --- | ---: | ---: | ---: | ---: |
| GSM8K | Frozen general trace selector | 217 | 6.91% | 5.99% | +0.92 pp [-2.76, +4.61] |
| GSM8K | High-confidence residual selector | 223 | 5.83% | 5.83% | +0.00 pp [-3.14, +3.14] |
| MATH500 | High-confidence residual selector | 43 | 9.30% | 4.65% | +4.65 pp [0.00, +11.63] |

The GSM8K strict-control results do not establish causal enrichment beyond
confidence. MATH500 is directionally positive, but only 43 strict pairs were
available, and its interval touches zero. It is exploratory rather than a
cross-task confirmation.

## Interpretation

The defensible result is narrow:

> A frozen broad trace predictor can identify future regret among early,
> high-confidence tokens slightly better than confidence alone. The present
> evidence does not show that this predictive advantage selects commitments
> with larger downstream counterfactual effects after strict confidence- and
> trajectory-matched control.

This is still useful for the paper because it turns an apparent method claim
into a diagnostic finding with a precise limitation. It should not be written
as proof that token-fate prediction finds causally more consequential false
confidence.

## Artifacts

- `results/robustness_gsm8k_frozen_selector_analysis/fixed_selector_bootstrap.csv`
- `results/robustness_llada_gsm8k_highconf_paired_500/analysis/causal_selector_comparisons_10k.csv`
- `results/robustness_llada_gsm8k_general_paired_500/analysis/causal_selector_comparisons_10k.csv`
- `results/robustness_llada_math500_highconf_paired/analysis/`
- `docs/causal_robustness_protocol.md`
