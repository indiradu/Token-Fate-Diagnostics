# PhaseLock screening log

Date: 2026-09-06

Model: `GSAI-ML/LLaDA-8B-Instruct`, `bf16`, seed `17`.

The nested protocol is:

- `A0`: ordinary dense decode;
- `P1`: semantic/token identity commitment with live representations;
- `P2`: replay the exact `P1` semantic plan and freeze references using the
  representation gate;
- `P3`: replay the exact `P2` reference plan and remove locked query/FFN rows.

Settings for the first screen: 64 denoising steps, 64 generated tokens,
32-token blocks, semantic-lock fraction 0.06, layer-24 cosine drift,
drift threshold 0.02, patience 2, minimum post-commit age 1. Policies were
`confidence`, `posterior`, `token_fate`, and `consensus`; the reference policy
was `drift`; the compute policy was `row_sparse`.

## Runs

| Dataset | Slurm job | Examples | Offset | Status |
|---|---:|---:|---:|---|
| GSM8K | 176218 | 20 | 100 | complete |
| MATH500 | 176239 | 20 | 100 | complete |
| HumanEval | 176270 | 10 | 0 | complete |

The failed GSM8K job 176217 is discarded: it stopped before producing valid
arm records because the initial feature map omitted three logged features. The
feature map was made parity-complete and job 176218 was rerun cleanly.

## GSM8K screen

The clean rerun used examples `gsm8k-test-100` through `gsm8k-test-119`.

- `A0` accuracy: 0.60, mean NFE 64.
- Semantic commitment reached mean NFE 48 and approximately 25% lower measured
  latency for every policy.
- Relative to A0, answer-change rates were 0.10 for confidence, posterior, and
  token-fate, and 0.15 for consensus. Reference-correct to candidate-incorrect
  harm was 0.05 for every policy.
- `P1→P2` was exact and answer-preserving for every policy; representation
  locking added about 15% latency on this implementation.
- `P2→P3` was exact and answer-preserving for every policy; row removal added
  about 6% latency rather than saving time.
- Consensus reached 0.65 accuracy on this screen; the other policies reached
  0.60. This is a small screening result, not a final ranking.

## MATH500 screen

The clean run used examples `math500-test-100` through `math500-test-119`.

- `A0` accuracy: 0.30, mean NFE 64.
- All semantic policies reached mean NFE 48 and approximately 25% lower
  measured latency.
- Relative to A0, answer-change rates were 0.30 for confidence, 0.20 for
  posterior and consensus, and 0.25 for token-fate. Reference-correct to
  candidate-incorrect harm was 0.20, 0.10, 0.10, and 0.15 respectively.
- `P1→P2` was exact and answer-preserving for every policy; representation
  locking added about 14.5% latency.
- `P2→P3` was exact and answer-preserving for every policy; row removal added
  about 4.7–5.0% latency.

## Current interpretation

The screen supports a real separation between semantic commitment and the
later reference/compute interventions: semantic commitment changes sequences
and sometimes answers, while the current conservative representation gate did
not add observable harm, and the current Python row-sparse backend did not
produce a wall-clock gain. It does **not** yet establish that representation
drift predicts answer harm: the first screen was recorded before the ledger
included the semantic-policy ID on every representation observation. The
logging fix is in the runner for subsequent runs, and the analyzer now writes
position-level drift/divergence summaries when that ID is present.

The negative P3 speed result is an implementation result. The backend computes
only active query/FFN rows but still assembles a full K/V tensor and calls the
dense attention primitive, so a fused sparse kernel or batch-aware packing is
still required before concluding that compute locking has no systems value.

HumanEval scoring is an isolated single-sample execution check (pass@1-style
screen), not the official pass@k estimator. The token-fate model in this first
HumanEval screen is transferred from GSM8K and is therefore exploratory.

## HumanEval screen

The run used `HumanEval/0` through `HumanEval/9`, with 128 generated tokens
and 64 denoising steps because code completions need more room than the math
screen.

- `A0` accuracy: 0.30, mean NFE 64.
- All semantic policies reached 0.20 accuracy, mean NFE 52, and approximately
  18.8% lower measured latency.
- Each semantic policy changed the answer status on 1/10 examples, and every
  observed change was a reference-correct to candidate-incorrect case.
- `P1→P2` was exact and answer-preserving for every policy; reference locking
  added about 8.6–9.1% latency.
- `P2→P3` was exact and answer-preserving for every policy; the row-sparse
  backend was approximately 1.9–2.2% slower.

Because this is only 10 tasks and single-sample execution, these HumanEval
numbers are a sanity screen, not a benchmark claim.

The next scheduled matrix is the representation-gate sweep: `strict`
(`τ=0.005`, patience 4), `default` (`0.02`, 2), `loose` (`0.05`, 1), and
`very_loose` (`0.10`, 1), all with minimum age 1. The compute follow-up pairs
the original row-sparse backend with `row_sparse_packed` under the identical
P2 reference plan.
