# Trace-assisted PhaseLock speed-mode results

## Result

The selected algorithm is `open_confidence`. It combines an aggressive
TraceLock-style semantic proposal path with an LM-confidence progress fallback,
then keeps representation lock and compute lock as independent downstream
decisions.

On the fixed 60-example GSM8K/MATH500/HumanEval comparison set, it preserved
the dense decoder's correctness status for every example and reduced estimated
online runtime by 19.24% [13.79%, 24.76%]. It is substantially faster than the
previous conservative PhaseLock, but it does not beat the TraceLock-LLaDA
transfer on either measured accuracy or speed in this slice.

## Algorithm

For every active block and denoising update:

1. Compute LLaDA token proposals, LM posterior confidence, and the trained
   TraceLock future-stability probabilities from layers 30/31/32.
2. Restrict the semantic decision to TraceLock's 32-token soft-crop window.
3. If any active token has future-stability probability greater than 0.95,
   commit every such token to its current argmax identity.
4. Otherwise, commit exactly one token: the active in-window token with maximum
   LM posterior confidence. The rejected alternatives used maximum Trace score
   or maximum Trace-probability-times-confidence for this fallback.
5. Continue updating committed-token representations. Independently freeze a
   reference only after the frozen multi-layer/influence gate passes: maximum
   drift across layers 8/16/24 <= 0.002, attention influence at layer 24 <= 0.5,
   patience 6, semantic age >= 4, and a 4% per-update lock cap.
6. Independently switch to fused row removal only when locked fraction >= 20%,
   at least eight steps remain, at least 1,500 row-forwards are projected to be
   saved, and the calibrated expected gain is at least 3%.

The key revision is step 4. The original maximum-Trace-score fallback caused
the same MATH500 regression under guarded, balanced, and open proposal caps.
The confidence fallback removed that regression and also shortened trajectories.

## Held-out comparison: 60 examples

| Method | Accuracy | Exact vs dense | Answer changed | Regressions | Improvements | Mean NFE | Runtime gain vs dense |
|---|---:|---:|---:|---:|---:|---:|---:|
| PhaseLock `open_confidence` | 36.67% | 86.67% | 8.33% | 0 | 0 | 100.73 | 19.24% [13.79, 24.76] |
| TraceLock-LLaDA transfer | 43.33% | 5.00% | 50.00% | 0 | 4 | 93.73 | 27.99% [23.92, 32.46] |
| SureLock, epsilon 5e-4 | 33.33% | 3.33% | 63.33% | 6 | 4 | 128.00 | 51.07% [47.56, 53.92] |
| SureLock, epsilon 5e-3 | 35.00% | 3.33% | 65.00% | 5 | 4 | 128.00 | 52.53% [48.80, 55.58] |

All methods use LLaDA-8B-Instruct, BF16, temperature zero, 128 generated tokens,
128 denoising steps, batch size four, the same 60 examples, one timing warmup,
and four timing repeats on the same GPU class. PhaseLock's value is an additive
estimate from exact fused replay plus measured controller overhead; SureLock
and TraceLock are online measurements.

PhaseLock-minus-TraceLock accuracy is -6.67 percentage points with the existing
paired 95% interval [-13.33, -1.67]. TraceLock therefore has the better measured
accuracy on this slice, not merely more output variation. Its runtime gain is
also 8.75 percentage points larger, although a direct cross-job latency
significance claim is not made.

Against SureLock, PhaseLock gives up roughly 32--33 percentage points of speed
but removes all 5--6 observed correctness regressions and retains far more of
the dense output. This is the clearest current safety/speed trade-off.

## Phase separation

| Transition | Held-out result | Interpretation |
|---|---:|---|
| A0 -> A1 | 52/60 exact; 0 regressions | Aggressive semantic lock changes some trajectories but preserves correctness in this run. |
| A1 -> A2 | 60/60 exact | The independent representation gate adds no observed output harm. |
| A2 -> A3 | 60/60 exact | Fused row removal is output-equivalent to reference freeze. |
| A3 vs A2 runtime | +1.36% [0.43, 2.44] | Compute lock adds a small but statistically positive pooled gain. |

Phase 3 switched on 6/15 batches. It switched on 3/5 GSM8K batches and 3/5
MATH500 batches, and abstained on 5/5 HumanEval batches. This is the intended
behavior of a distinct profitability decision: safe row removal is used only
where the latency model predicts enough work to amortize packing.

## Per-dataset selected-policy results

| Dataset | Accuracy A0/A3 | Exact A0 -> A1 | Answer changed | Mean NFE | Estimated online gain | Phase-3 gain |
|---|---:|---:|---:|---:|---:|---:|
| GSM8K | 60% / 60% | 80% | 10% | 98.8 | 21.39% [14.69, 26.91] | 2.10% [0.20, 4.63] |
| MATH500 | 45% / 45% | 90% | 5% | 114.8 | 8.48% [3.70, 12.63] | 1.80% [0.26, 3.60] |
| HumanEval | 5% / 5% | 90% | 10% | 88.6 | 27.41% [21.40, 35.63] | -0.01% [-0.02, 0.01] |

HumanEval Phase 3 did not switch, so its near-zero A3-versus-A2 timing
difference is measurement noise rather than row-removal performance.

## Decision

Keep `open_confidence` as the speed-mode semantic policy for the next integrated
implementation. Keep the current Phase-2 gate: it is empirically exact here and
is the safety condition that permits reference reuse. Keep the Phase-3 gate:
it contributes a small pooled gain and correctly abstains on unprofitable
HumanEval batches.

This is a robust candidate, not a final winning algorithm. Before a paper-level
speed claim, integrate the controllers and fused executor into one serving path,
repeat on a larger held-out sample, and investigate why the TraceLock transfer's
four beneficial answer changes produce higher measured accuracy.
