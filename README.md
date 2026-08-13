# Token-Fate Diagnostics for Premature Commitment in Diffusion LMs

This repository contains the code and reported experimental artifacts for a
diagnostic study of premature token commitment in diffusion language models. We
use **token fate** as a behavioral object: whether a token proposed at an
intermediate denoising step agrees with the final decoded token.

The project uses `GSAI-ML/LLaDA-8B-Instruct` with confidence decoding on
GSM8K, MATH500, countdown, and an iLLaDA GSM8K robustness run.

## Main Finding

Token fate is strongly predictable from decoding traces, but broad trace regret
is often nearly captured by confidence. The clearest result is a held-out,
high-confidence GSM8K analysis: a frozen general trace predictor improves over
confidence by `+0.0174` AUROC (95% bootstrap CI `[+0.0066, +0.0272]`) and
`+0.0124` AP (95% CI `[+0.0032, +0.0218]`) across 500 held-out examples.

Strict prospective interventions use same-trajectory, same-step,
confidence-matched controls and a downstream-only endpoint that excludes the
forced token. These experiments do **not** establish that the trace score
selects commitments with larger downstream causal effects than confidence.
This prediction-versus-causation distinction is a central result, not a
negative result to hide.

## Repository Guide

- `src/regret_remasking/`: trace collection, token-fate labels, features, and
  predictor implementations.
- `scripts/`: training, evaluation, intervention, and bootstrap analysis entry
  points.
- `docs/`: experimental protocol and result summaries.
- `results/`: compact reported artifacts; see `results/README.md`.
- `tests/`: lightweight selector and feature tests.

## Key Results

| Analysis | Result | Interpretation |
| --- | --- | --- |
| Cross-task trace prediction | AUROC `0.887` GSM8K, `0.901` MATH500, `0.985` countdown, `0.895` iLLaDA GSM8K | Future token fate is predictable from traces. |
| High-confidence GSM8K | `+0.0174` AUROC vs confidence, 95% CI `[+0.0066, +0.0272]` | A frozen broad trace model retains a small predictive advantage when confidence considers tokens safe. |
| Strict GSM8K intervention | `+0.92` percentage points downstream change, 95% CI `[-2.76, +4.61]`, 217 pairs | No resolved causal enrichment beyond confidence. |
| Strict MATH500 intervention | `+4.65` percentage points, 95% CI `[0.00, +11.63]`, 43 pairs | Directionally positive but exploratory. |
| Hidden/logit analysis | Hidden-only AUROC `0.778`; final token in layer-32 top-10 for `84.1%` of early-regret cases | Hidden states contain partial signal; wrong commitments often retain the eventual token as a near alternative. |

Detailed evidence and all caveats are in
`docs/causal_robustness_results.md`.

## Reproduction

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

Run local checks that do not load LLaDA:

```bash
python3 -m compileall src scripts tests
python3 tests/test_features.py
python3 tests/test_counterfactual_selection.py
```

The exact experimental specifications and split rules are documented in
`docs/causal_robustness_protocol.md`.

## Scope

This is a diagnostic and interpretability project, not a state-of-the-art
decoding claim. The repository deliberately preserves the strict-control null
result alongside the positive prediction results.
