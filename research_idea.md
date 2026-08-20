# Risk-Controlled Selective Locking for Diffusion Language Models

## One-Sentence Thesis

Irreversible token locking in diffusion language models should be treated as a
risk-controlled decision problem, not just a convergence-detection heuristic:
we separately estimate when a token is safe to commit, when its representation
is safe to freeze, and when removing its row from future computation preserves
final output quality.

## Core Claim

Existing efficient diffusion LM decoders mostly ask whether a token looks
settled now. Our proposal asks a harder question:

> What is the conditional risk of making this token irreversible?

The key distinction is that semantic convergence, representational convergence,
and computational dispensability are not the same object.

Throughout the project, use three hypothesis labels:

- **A:** Semantic commitment does not imply representation settlement.
- **B:** Semantic commitment does not imply compute safety.
- **C:** Representation drift predicts compute unsafety beyond matched controls.

These are labels, not new variables. Diagnostics such as non-target token
changes, answer changes, or output divergence are measurements used to test the
three hypotheses.

| Fate type | Question | Failure if wrong |
| --- | --- | --- |
| Semantic fate | Will the token identity change by the final sequence? | Premature token commitment |
| Representational fate | Will the token's hidden/K/V state keep changing materially? | Stale context for neighboring tokens |
| Computational fate | Can we skip future compute for this position without changing the final output or task result? | Accuracy or quality loss despite apparent stability |

The proposed paper should show when these fates agree, when they diverge, and
how a decoder can use that distinction to improve the quality-efficiency
frontier.

## Contribution Framing

### Contribution 1: Three notions of token settlement

We identify and operationalize three distinct notions of settlement in
diffusion LM decoding:

- **Semantic settlement:** the token identity has reached its final value.
- **Representational settlement:** the token's hidden state or K/V state has
  stopped changing in ways that matter to the rest of the sequence.
- **Computational settlement:** the token can be removed from future row-wise
  computation without changing the final output or task result.

This taxonomy separates a decoding question from a systems question. A token
can be semantically settled while still carrying useful evolving context for
other positions.

### Contribution 2: Observational stability is not compute safety

We show that observational token stability is systematically different from
interventional compute safety. The observational question is:

```text
P(x_i,t = x_i,T | H_t)
```

where `H_t` is the history available at denoising step `t`. This asks whether
the current token identity will match the final token.

The interventional compute-safety question is:

```text
P(Y_freeze(i,t) = Y_baseline | H_t)
```

This asks whether the final decoded sequence or task outcome remains unchanged
under an intervention that freezes position `i` at step `t`.

These probabilities need not agree. A token may already have its final identity
while its representation continues to affect neighboring tokens. Conversely, a
token may be observationally unstable in isolation but computationally
irrelevant to the task metric. This observational-versus-interventional gap is
the central scientific distinction of the project.

### Contribution 3: Irreversible decoding as constrained risk allocation

We formulate locking as a constrained allocation problem: maximize saved
compute while keeping semantic premature-lock risk and compute-freeze risk
below explicit budgets. This shifts the objective from "lock as many stable
tokens as possible" to "spend irreversible actions where their conditional
risk is acceptable."

### Contribution 4: Selective exploitation of the disagreement region

We exploit the region where cheap convergence signals and downstream risk
disagree. In practice, this is likely to include early high-confidence tokens:
they look safe enough to invite locking, but our current results show that
token-fate features still add predictive information over confidence there.
The method should improve the quality-compute Pareto frontier by applying
extra risk estimation only to these consequential lock candidates.

## Motivation

Masked diffusion language models repeatedly recompute every token position
across denoising steps. This wastes computation once many positions have become
stable. SureLock shows that positions can be permanently removed from future
compute by caching their K/V states and skipping later Q-projection and FFN
rows, yielding large algorithmic FLOP savings on LLaDA-8B.

However, locking is an irreversible action. A token that has high confidence or
low local posterior drift can still later change because surrounding context is
not settled. Our current token-fate results directly identify this danger:
trace-based fate prediction is broadly strong, but its most useful residual
value appears among early high-confidence tokens. On held-out GSM8K, a frozen
general trace predictor improves over confidence by `+0.0174` AUROC and
`+0.0124` AP in the high-confidence slice.

That result should not be framed as "we can predict token fate." Recent
learned commitment methods already press on that claim. Instead, it should be
framed as evidence that the value of extra prediction is concentrated in the
apparently safe lock-candidate region, where cheap convergence signals can be
misleading.

## Research Question

Can a diffusion LM decoder save compute under explicit lock-risk budgets by
using different evidence for three transitions?

```text
ACTIVE -> COMMITTED -> COMPUTE-FROZEN
```

The target is not maximum locking. The target is maximum saved compute subject
to bounded premature-lock and stale-representation risk.

## Decision-Theoretic Formulation

Let `pi` be a locking policy. A simple objective is:

```text
maximize_pi   E[compute_saved(pi)]
subject to    P(semantic_premature_lock | pi locks) <= eps_commit
              P(compute_freeze_harms_output | pi freezes) <= eps_compute
```

Equivalently, for a tunable tradeoff:

```text
maximize_pi E[compute_saved]
            - lambda_commit E[semantic_lock_regret]
            - lambda_compute E[compute_freeze_regret]
```

The learned token-fate model is therefore not the whole contribution. It is one
estimator of conditional lock risk, invoked only where cheap signals become
unreliable.

## Relation to Prior Work

### SureLock

SureLock is the main systems baseline. It is training-free, locks unmasked
positions when adjacent-step posterior KL indicates stability, optionally adds
a confidence gate, caches locked K/V, and skips Q-projection and FFN rows for
locked positions. It establishes that compute locking can reduce algorithmic
FLOPs substantially.

Our difference:

- SureLock asks whether the local posterior is stable enough to stop compute.
- We ask whether irreversible commit and compute-freeze decisions satisfy
  calibrated risk budgets.
- We can reuse SureLock's compute machinery, but replace or augment its lock
  admission rule.

### TraceLock and learned commitment policies

TraceLock-like methods are the closest conceptual threat. They learn from
completed diffusion traces whether an intermediate token agrees with the final
token, then use that prediction to decide commit-or-revise.

Our difference cannot merely be "we predict future token stability." The
stronger distinction is:

- learned semantic fate is only one risk estimator;
- we apply it selectively where cheap signals are ambiguous;
- we separate semantic commit risk from representation/cache risk;
- we evaluate the full compute-quality Pareto frontier, not just stability
  prediction.

### TACG, LESS, and trajectory-aware commitment

Trajectory-aware methods already address transient high-confidence spikes using
history, persistence, confidence, KL/JSD, or logit trajectory signals. LESS, for
example, combines confidence, top-1 persistence, and inter-step JSD as a
training-free commitment rule.

Our difference:

- those methods define stronger online stability criteria;
- we treat each criterion as a candidate risk estimator or baseline;
- the paper's main object is calibrated irreversible-decision risk under
  compute budgets.

### Polestar and representation-drift methods

Polestar is especially important because it connects token commitment and
cache/reuse through representation drift. This pressures any claim that
"semantic stability implies compute safety."

Our difference should be tested, not assumed:

- semantic fate: token identity stability;
- representational fate: hidden/K/V drift;
- computational fate: final output or task invariance under compute freezing.

If experiments show these are separable, that becomes the scientific core of
the paper.

## Proposed Method

At each denoising step `t`, for each position `i`, compute cheap online
features already available from the decoding loop:

- confidence and entropy;
- margin;
- local KL or JSD to the previous step;
- top-1 persistence and flip history;
- run length since last top-1 change;
- block time fraction;
- local mask ratio;
- context volatility around the position;
- optional lightweight hidden/K/V drift features.

Then apply a staged risk controller.

### Stage 1: Candidate Routing

Low-confidence positions are not lock candidates. They are obviously uncertain,
so running a learned fate model is usually wasted overhead.

High-confidence and apparently stable positions are candidates. These are the
positions where false confidence is dangerous and where a second risk estimate
has the highest value.

### Stage 2: Semantic Commit Gate

Estimate:

```text
r_commit(i,t) = P(x_i,t != x_i,T | trace history up to t)
```

Commit only if:

```text
r_commit(i,t) <= eps_commit
```

A committed token is no longer eligible for remasking or token replacement, but
its representation can still be recomputed for a short delay window. This
isolates semantic locking from systems-level compute freezing.

### Stage 3: Representation and Compute-Freeze Gate

Estimate or test whether the position is safe to remove from future row-wise
computation:

```text
r_repr(i,t) = P(representation remains materially useful if cached | history)
r_compute(i,t) = P(final output changes under compute freeze | history)
```

Compute-freeze only if:

```text
r_commit(i,t) <= eps_commit
r_compute(i,t) <= eps_compute
lock_age(i) >= k
```

The compute-frozen token keeps cached K/V so active positions can still attend
to it, but skips future Q-projection, attention output, and FFN row updates.

### Compact Policy Sketch

```text
if confidence_i < tau_conf:
    keep_active(i)
elif cheap_stability_i is clearly unsafe:
    keep_active(i)
else:
    r_commit = semantic_risk_model(trace_i_up_to_t)
    if r_commit > eps_commit:
        keep_active(i)
    else:
        commit_lock(i)

        if lock_age_i >= k:
            r_compute = compute_risk_model_or_drift_gate(trace_i_up_to_t)
            if r_compute <= eps_compute:
                compute_freeze(i)
```

## What Makes This Worth Studying

The main empirical hypothesis is not just that a learned classifier has better
AUROC. It is:

> The marginal value of expensive risk estimation is concentrated among
> apparently safe lock candidates, and semantic safety is not sufficient for
> compute safety.

The core empirical test is whether observational settlement predicts the
counterfactual effect of freezing. We should report both:

```text
P(x_i,t = x_i,T | H_t)
```

and:

```text
P(Y_freeze(i,t) = Y_baseline | H_t)
```

then measure where they disagree.

This produces three falsifiable hypotheses:

1. **A:** Some positions are semantically committed while their hidden or K/V
   representations still move materially.
2. **B:** Some semantically committed positions still affect neighboring tokens
   or the final task answer under a freeze intervention.
3. **C:** High post-commit representation drift identifies freeze-sensitive
   positions better than matched low-drift controls.

The method is worth building only if those hypotheses translate into useful
engineering behavior:

- selective risk estimation on lock candidates preserves most of the safety
  benefit at lower overhead than all-token prediction;
- risk-controlled admission beats raw convergence detection at matched lock
  rate or matched FLOP budget;
- the Pareto gain concentrates where cheap criteria say "lock" but the
  interventional risk is still nontrivial.

## Experimental Plan

### Phase 0: Baseline and Related-Work Reproduction

Implement or approximate the strongest relevant lock criteria:

- confidence-only;
- KL-only;
- confidence plus KL;
- persistence plus confidence;
- LESS-style confidence plus persistence plus JSD;
- TACG-style trajectory support, if feasible;
- TraceLock-style learned future-stability controller;
- SureLock criterion and SureLock compute-freeze machinery;
- representation-drift gate inspired by Polestar;
- oracle semantic settlement;
- oracle compute-freeze safety.

The goal is not to beat weak baselines. The paper only becomes credible if it
compares against the methods that reviewers will expect.

### Phase 1: Offline Fate Decomposition

Use completed traces to label three targets.

Semantic labels:

- Does the current top-1 token equal the final token?
- What is the earliest irreversible semantic settlement step?
- Does a high-confidence token later change?

Representational labels:

- How much do hidden states, K/V states, or logits drift after semantic
  settlement?
- Does representation drift remain high for semantically stable tokens?
- Which layers and positions show the largest semantic/representation gap?

Computational labels:

- If the token were compute-frozen at step `t`, would other tokens change?
- Does answer correctness or generation quality change?
- Is the effect local, suffix-wide, or global?
- How often does `x_i,t = x_i,T` hold while `Y_freeze(i,t) != Y_baseline`?
- How often does observational instability matter for the final task outcome?

This phase should produce the core scientific figure:

```text
semantic convergence != representational convergence != compute dispensability
```

### Phase 2: Lock-Risk Pareto Evaluation

For each candidate method, plot:

```text
x-axis: fraction of token-step FLOPs eliminated
y-axis: P(future token change | locked)
```

Also plot:

```text
x-axis: lock rate or estimated FLOPs saved
y-axis: premature-lock rate, lock precision, and settlement delay
```

The high-confidence disagreement subset should be explicit: token-steps where
confidence is high, but the current token still differs from the final token.
Evaluate how well each method finds or avoids these dangerous positions.

```text
semantic_miss_set = {
  (i,t): confidence_i,t > tau_conf and x_i,t != x_i,T
}
```

Also construct the compute-safety disagreement subset: token-steps where the
current token already matches the final token, but freezing that position
changes the baseline output. This subset is the direct evidence for B. If it is
large or structured, semantic settlement is insufficient for compute locking.

```text
compute_disagreement_set = {
  (i,t): x_i,t = x_i,T and Y_freeze(i,t) != Y_baseline
}
```

### Phase 3: Commit-Lock Decoder

Modify the decoding loop so committed tokens stay fixed but still receive full
model computation. This tests the semantic decision before adding systems
complexity.

Report:

- final-answer accuracy on GSM8K and MATH500;
- code or structured-task accuracy if available;
- output divergence from the unlocked baseline;
- lock-induced answer changes;
- premature-lock diagnostics by confidence slice;
- quality at matched lock rate.

### Phase 4: Compute-Freeze Proxy

Before writing custom kernels, simulate the compute path from active-set
trajectories.

Estimate:

- active rows per step;
- attention FLOPs `O(M_t N d)`;
- FFN FLOPs `O(M_t d^2)`;
- K/V cache memory;
- gather/scatter and packing overhead;
- expected speedup under realistic batch and sequence regimes.

This phase decides whether a real compute-lock implementation is justified.

### Phase 5: End-to-End Compute-Freeze Decoder

Implement the full compute-freeze path only after semantic and proxy results
are strong. Reuse the SureLock-style mechanism: locked tokens keep cached K/V,
while active tokens attend to all positions.

Primary end-to-end plots:

```text
x-axis: wall-clock speedup
y-axis: task quality
```

and:

```text
x-axis: accuracy drop budget
y-axis: FLOPs or latency saved
```

Report results on LLaDA and at least one additional diffusion LM family if
feasible. Dream or another open dLLM would make the generality claim much
stronger.

## Metrics

Safety:

- `P(future_change | locked)`;
- `P(answer_changes | locked)`;
- non-target token change rate after intervention;
- lock precision and recall;
- calibration error for `r_commit` and `r_compute`;
- risk-budget violation rate.

Efficiency:

- token-step updates avoided;
- algorithmic FLOPs;
- wall-clock latency;
- throughput;
- predictor overhead;
- memory overhead from caching.

Quality:

- exact-answer accuracy for math;
- pass rate for code;
- MT-Bench or instruction-following judge score;
- continuation Gen-PPL, with caution that short continuations may exaggerate
  surface differences;
- output divergence from the no-lock baseline.

## Ablations

- Selective fate model on high-confidence candidates vs all-token fate model.
- Confidence threshold: fixed, percentile, or calibrated risk threshold.
- Commit risk threshold `eps_commit`.
- Compute risk threshold `eps_compute`.
- Immediate compute freeze vs delayed freeze after `k` committed steps.
- Semantic-only gate vs semantic plus representation-drift gate.
- Trace-only features vs trace plus hidden/K/V drift features.
- Training-free stability criteria vs learned risk models.
- Calibration method: raw score, isotonic calibration, conformal threshold, or
  validation-set risk control.
- Transfer: train/calibrate on GSM8K, evaluate on MATH500, countdown, iLLaDA,
  and another dLLM.

## Minimum Viable Paper

A credible first version does not need a perfect custom sparse kernel. It needs
to establish the scientific and algorithmic claim.

Minimum evidence:

1. High-confidence false locks exist and are not fully captured by confidence,
   KL/JSD, or persistence.
2. Selective risk estimation reduces premature locks at matched lock rate or
   matched estimated compute.
3. Semantic convergence and representation convergence diverge in measurable
   cases.
4. A staged commit-then-compute-freeze policy dominates immediate compute
   freeze on a safety-efficiency Pareto curve.
5. Active-set simulations show enough compute savings to justify systems work.

Stronger systems paper:

1. Implement full compute freeze with cached K/V.
2. Match or improve SureLock quality at similar FLOPs.
3. Save more FLOPs or latency at the same quality drop.
4. Demonstrate gains on more than one dLLM family.

## Reviewer-Resistant Framing

Avoid this claim:

> We learn whether high-confidence tokens will stay fixed.

That is too close to learned future-stability commitment work.

Use this claim:

> Locking is a risk-sensitive irreversible decision. We separate semantic,
> representational, and computational fate, then allocate expensive risk
> estimation only to lock candidates where cheap convergence signals are
> unreliable.

The sharper title-level contribution is:

> Semantic stability is not compute stability.

If experiments support that sentence, the project becomes more than a decoder
heuristic.

## Risks

- **Related-work crowding.** TraceLock, trajectory-aware commitment methods,
  LESS-style mutual stability, SureLock, and Polestar all occupy nearby space.
  The paper needs a risk-control and fate-decomposition contribution, not just
  another commitment score.

- **Existing diagnostic AUROC is not enough.** The `+0.0174` AUROC and
  `+0.0124` AP high-confidence result motivates the idea but cannot be the
  headline. Reviewers will care about quality at fixed compute or compute at
  fixed quality.

- **Compute-risk labels are expensive.** Measuring whether compute-freezing one
  token changes the final sequence can require counterfactual decoding. Start
  with a small candidate set and proxy labels based on representation drift.

- **Predictor overhead can erase savings.** The learned model must be small and
  selectively invoked. Otherwise a training-free criterion may win in practice.

- **Wall-clock speedup may lag FLOPs.** Irregular active sets, cache layout, and
  gather/scatter overhead can reduce real speedups. FLOPs alone are not enough.

## Immediate Next Experiment

Build the offline lock-risk Pareto plot before implementing a new decoder.

For each method, evaluate:

```text
P(future_change | locked)
vs
fraction of token-step FLOPs eliminated
```

Methods:

- confidence;
- KL;
- confidence plus KL;
- persistence plus confidence;
- JSD/persistence-style stability;
- SureLock criterion;
- TraceLock-style learned semantic fate;
- selective semantic fate on high-confidence candidates;
- semantic plus representation-drift gate;
- oracle semantic settlement;
- oracle compute safety.

If selective risk control does not dominate on this plot, the idea should be
rethought before any systems implementation.

## References to Track

- SureLock: <https://daioba.github.io/surelock/>
- SureLock arXiv: <https://arxiv.org/abs/2602.06412>
- TraceLock / learned token-commitment policy: verify latest public version
  before citation.
- LESS: <https://www.alphaxiv.org/overview/2606.16908>
- TACG: <https://www.emergentmind.com/papers/2607.03236>
- Polestar: <https://papers.cool/arxiv/2607.14107>
