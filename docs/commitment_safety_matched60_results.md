# Matched 60-example semantic-lock confirmation

## Setup

- Model: LLaDA-8B-Instruct
- Precision: BF16
- Generation: 128 tokens, 128 denoising updates, block length 32
- Serving batch: 4
- Evaluation: GSM8K 820–839, MATH500 450–469, HumanEval 120–139
- Timing: one warmup and four repeats for each method and batch
- Scope: Phase 1/A1 only. Token identities are committed early, while hidden/K/V
  representations continue to update densely. No Phase-2 freezing or Phase-3
  row removal is included in these wall-clock measurements.

## Frozen policies

`capped_confidence` preserves the ordinary confidence schedule. During only the
first two updates of each block, it may raise the transfer count to
`max(base_k, ceil(0.06 * active_tokens))`; only extra candidates with confidence
at least 0.95 are eligible.

`safety_full` preserves the ordinary schedule and adds at most two candidates
per sample and update when the calibrated direct commitment-safety probability
is at least 0.733200. Its features are confidence, margin, entropy, token age,
Trace probability, block age, maximum hidden drift across layers 8/16/24,
layer-24 attention influence, and drift availability.

## Pooled results

| Method | Accuracy | Exact dense | Answer changed | Regressions / improvements | Mean NFE | Actual runtime gain |
|---|---:|---:|---:|---:|---:|---:|
| Dense | 36.67% | 100.00% | 0.00% | 0 / 0 | 128.00 | 0.00% |
| Capped confidence | 36.67% | 98.33% | 1.67% | 0 / 0 | 126.07 | 1.36% [0.87%, 1.82%] |
| Direct safety full | 36.67% | 15.00% | 45.00% | 1 / 1 | 55.00 | 52.22% [49.73%, 54.38%] |

## Per-dataset results

| Dataset | Policy | Accuracy | Exact dense | Answer changed | Mean NFE | Runtime gain |
|---|---|---:|---:|---:|---:|---:|
| GSM8K | Capped confidence | 60% | 100% | 0% | 126.0 | 1.26% |
| GSM8K | Direct safety full | 60% | 10% | 20% | 51.6 | 54.62% |
| MATH500 | Capped confidence | 45% | 100% | 0% | 126.8 | 0.66% |
| MATH500 | Direct safety full | 45% | 20% | 30% | 61.2 | 46.32% |
| HumanEval | Capped confidence | 5% | 95% | 5% | 125.4 | 1.85% |
| HumanEval | Direct safety full | 5% | 15% | 85% | 52.2 | 54.65% |

The capped-confidence difference is `HumanEval/136`; both dense and capped
outputs are incorrect. Direct safety regresses `gsm8k-test-829` and improves
`gsm8k-test-839`. Both examples are outside the predictor's original GSM8K
820–827 end-to-end pilot subset.

## Decision

Capped confidence remains a useful safety control but is too conservative for
speed mode. Direct safety exposes a large compute opportunity and is much safer
than SureLock by paired regression count, but it does not pass the zero-
regression rule and changes nearly half of the answers. It therefore does not
replace the current `open_confidence` PhaseLock policy.

The likely failure is compositional: the classifier was trained on additive
single-token interventions, while deployment can make roughly 75 extra commits
per sample. The next direct-safety revision should train/calibrate on multi-token
commitment sets or enforce a cumulative per-sequence risk budget. Only a
zero-regression candidate should be replayed unchanged through fixed Phase 2
and Phase 3.
