# Selective Fate-Gated Locking for Efficient Diffusion LM Decoding

## One-Sentence Thesis

Use token-fate prediction only where confidence is most likely to be
misleading: high-confidence tokens. Low-confidence positions already have a
cheap uncertainty signal and should usually remain active; high-confidence
positions should pass a second fate gate before they are semantically committed
or removed from future computation.

## Motivation

Masked diffusion language models repeatedly recompute every token position
across denoising steps, even after many positions appear stable. SureLock shows
that this waste can be reduced by permanently locking converged unmasked tokens,
caching their K/V states, and skipping later Q-projection and FFN computation.
It reports large algorithmic FLOP savings on LLaDA-8B while maintaining similar
quality.

Our current token-fate results suggest a different safety question. Broad
token-fate prediction is strong, but much of the all-token signal is already
captured by confidence. The interesting residual value appears in early
high-confidence tokens: on held-out GSM8K, a frozen general trace predictor
improves over confidence by `+0.0174` AUROC and `+0.0124` AP. This is exactly
the failure mode a locking method cares about: tokens that look safe according
to confidence but later change.

The research shift is therefore:

> Do not use token fate as a universal replacement for confidence. Use it as a
> selective second constraint for high-confidence lock candidates.

## Core Research Question

Can a selective token-fate gate improve the quality-efficiency frontier of
diffusion LM decoding by preventing false high-confidence locks while still
allowing aggressive commit locking and compute locking for genuinely settled
tokens?

## Positioning Against SureLock

SureLock is the closest prior work and should be treated as the main baseline,
not as something to rediscover.

SureLock:

- is training-free;
- identifies converged unmasked tokens using local posterior KL, optionally
  with a confidence gate;
- permanently locks selected positions;
- caches locked K/V states so active positions can still attend to them;
- skips future Q-projection and FFN rows for locked positions.

Our proposed contribution is different:

- use learned token-fate prediction to detect false confidence before locking;
- run the learned gate only on high-confidence candidates, limiting overhead;
- keep confidence as the cheap rule for low-confidence positions;
- separate semantic commit locking from systems-level compute locking;
- evaluate whether fate gating reduces premature-lock errors at the same FLOP
  or latency budget.

The key paper claim should not be "we lock tokens." It should be:

> Confidence and KL tell us which tokens look converged now; token fate asks
> whether a high-confidence token is likely to remain correct later.

## Proposed Algorithm

At denoising step `t`, for each token position `i`, compute the standard
posterior and cheap online trace features already available from decoding:
confidence, entropy, margin, KL/JSD to the previous step, top-1 flip history,
run length, block time fraction, local mask ratio, and context volatility.

Then route positions through a selective gate:

1. **Low-confidence route.**
   If confidence is below a threshold, do not run the fate model and do not
   lock. Confidence is already enough to mark the token as uncertain. The token
   remains active and eligible for later revision.

2. **High-confidence candidate route.**
   If confidence is high enough to be considered safe, run the token-fate
   predictor as a second constraint. Lock only if predicted future-regret risk
   is below a conservative threshold.

3. **Commit lock.**
   A commit-locked token has its current identity treated as final for decoding
   policy purposes. It is removed from future remasking or candidate revision,
   but its representation may still be refreshed for a small number of steps.

4. **Compute lock.**
   A compute-locked token additionally exits the expensive computation path.
   Its K/V states are cached; active positions continue attending to those
   cached states while Q-projection and FFN computation for the locked row are
   skipped.

5. **Optional audit or unlock budget.**
   Because permanent locking is risky, periodically audit a small sample of
   locked positions or unlock positions when nearby context volatility becomes
   unusually high. This should be an ablation, not part of the first claim.

In compact form:

```text
if confidence_i < tau_conf:
    keep_active(i)
elif fate_risk_i > tau_fate:
    keep_active(i)
else:
    commit_lock(i)
    if representation_stable_i and lock_age_i >= k:
        compute_lock(i)
```

## Commit Lock vs Compute Lock

The two locks should be evaluated separately.

**Commit lock** is a decoding decision. It says the token identity is stable
enough that the sampler should stop revising it. This can be tested before any
custom kernels by forcing the token to remain fixed in future denoising steps.

**Compute lock** is a systems decision. It says the model no longer needs to
recompute that position's query, attention output, and FFN row. This is where
actual latency and FLOP savings come from, but it is also where stale
representations can hurt neighboring tokens.

Separating the two gives a cleaner research path:

1. First prove that selective fate gating makes safer commit decisions than
   confidence/KL alone.
2. Then prove that compute locking preserves most of those decisions while
   delivering real inference savings.

## Hypotheses

1. **Selective fate gating beats all-token fate gating.**
   The learned fate model is most useful on high-confidence candidates, where
   confidence alone is under-informative. Running it everywhere adds overhead
   and spends model capacity on easy low-confidence cases.

2. **Selective fate gating reduces premature locks.**
   At a fixed lock rate or FLOP budget, `confidence + fate` should produce
   fewer final-token changes after lock than confidence-only or KL-only rules.

3. **Two-stage locking is safer than immediate compute locking.**
   A short commit-only delay before compute locking should catch some unstable
   high-confidence tokens while preserving most potential savings.

4. **The main gain is quality at matched compute, not just raw speed.**
   SureLock already establishes that locking can save compute. The new question
   is whether token-fate gating moves the quality-efficiency frontier.

## Experimental Plan

### Phase 1: Offline Trace Study

Use existing traces to estimate the oracle opportunity:

- irreversible settlement time per token;
- token-step compute spent after settlement;
- fraction of high-confidence tokens that later change;
- lock precision, premature-lock rate, and settlement delay for each gate.

Compare:

- confidence-only;
- KL/stability-only;
- SureLock-style KL plus confidence;
- token-fate model on all tokens;
- selective token-fate model only on high-confidence candidates;
- oracle settlement.

Primary offline metrics:

- premature-lock rate;
- lock precision;
- lock recall;
- mean steps saved after lock;
- recovered oracle savings;
- calibration of predicted future-regret risk.

### Phase 2: Commit-Lock Decoder Prototype

Modify the LLaDA decoding loop so selected tokens remain fixed after lock, but
continue normal model computation. This isolates semantic locking from kernel
optimization.

Evaluate on GSM8K, MATH500, countdown, and at least one open-ended generation
benchmark if feasible.

Primary metrics:

- exact-answer accuracy for math tasks;
- output divergence from unlocked baseline;
- token change rate after would-be lock;
- lock-induced answer change rate;
- number of denoising steps or token updates avoided by policy.

### Phase 3: Compute-Lock Proxy

Before writing custom kernels, simulate compute savings from the active-set
trajectory:

- active token count per step;
- estimated attention FLOPs `O(M_t N d)`;
- estimated FFN FLOPs `O(M_t d^2)`;
- cache memory overhead;
- expected speedup under ideal and realistic packing assumptions.

This determines whether the gate is worth systems work.

### Phase 4: Real Compute-Lock Implementation

Implement the systems path only after the commit-lock policy is validated.
Reuse the SureLock-style design where locked tokens keep cached K/V values and
active tokens attend to them.

Compare end-to-end:

- no locking;
- SureLock-style KL/confidence locking;
- selective fate-gated commit lock;
- selective fate-gated commit plus compute lock;
- optional audit/unlock variants.

Report:

- algorithmic FLOPs;
- wall-clock latency;
- throughput;
- quality metrics;
- premature-lock and lock-age diagnostics.

## Ablations

- Confidence threshold: fixed threshold vs percentile threshold.
- Fate threshold: conservative vs aggressive risk budgets.
- Gate placement: fate-before-KL, KL-before-fate, confidence-before-fate.
- Feature family: trace-only vs trace plus hidden/logit features.
- Lock delay: immediate compute lock vs wait `k` steps after commit lock.
- Scope: high-confidence only vs all-token fate prediction.
- Transfer: train on GSM8K, evaluate on MATH500/countdown/iLLaDA.
- Calibration: raw logistic score vs isotonic or temperature-calibrated score.

## Failure Modes and Risks

- **Current evidence is predictive, not causal.** The existing strict
  interventions do not prove that token-fate-selected tokens cause larger
  downstream changes. The new work must be evaluated as a locking safety and
  efficiency method, not as a causal claim.

- **Predictor overhead can erase savings.** This is why the fate model should
  run only on high-confidence candidates and should use features already
  computed during decoding whenever possible.

- **Permanent locks are brittle.** A rare late context shift can make a locked
  token stale. Commit-only delay, conservative thresholds, and audit/unlock
  ablations should be included.

- **All-token fate may look good but add little.** Broad AUROC can be high
  because low-confidence tokens are easy. The decisive test is marginal value
  over confidence in the lock-candidate slice.

- **Systems speedup may lag FLOP savings.** Active-set compaction, gather/scatter
  overhead, cache layout, and kernel launch overhead may dominate. The proxy
  phase should estimate this before implementation effort.

## Minimum Viable Paper

A credible first paper does not need a perfect custom kernel. It needs to show:

1. existing SureLock-style criteria can be fooled by false high-confidence
   tokens;
2. selective token-fate gating reduces premature locks in the high-confidence
   candidate slice;
3. commit-lock decoding preserves task quality better than confidence/KL
   locking at matched lock rate;
4. active-set simulations show enough remaining compute savings to justify
   compute-lock implementation.

If the real compute-lock implementation works, the stronger paper is:

> Selective fate-gated locking improves the quality-efficiency frontier of
> masked diffusion LM decoding by using future-stability prediction only where
> confidence-based convergence is unsafe.

## Current Best Framing

The idea should be presented as a selective safety layer for locking, not as a
general replacement for confidence:

> Low confidence says "do not lock yet." High confidence says "maybe lock."
> Token fate answers the missing question: "will this confident token still be
> the same at the end?"

## References to Check

- SureLock project page:
  <https://daioba.github.io/surelock/>
- ICLR 2026 poster page:
  <https://iclr.cc/virtual/2026/poster/10009632>
- OpenReview entry linked from the SureLock project page:
  <https://openreview.net/forum?id=PzhNnMepgl>
