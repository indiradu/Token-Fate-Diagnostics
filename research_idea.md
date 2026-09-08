# PhaseLock: Separating Commitment Risk, Reference Risk, and Compute Utility in Diffusion Language Models

## One-Sentence Thesis

Irreversible token locking in diffusion language models should be treated as a
phased decision problem, not just a convergence-detection heuristic: we
separately estimate when a token identity is safe to commit, when the reference
state exposed to other tokens is safe to freeze, and when removing its row from
future computation is actually profitable on the target runtime.

## Core Claim

Existing efficient diffusion LM decoders mostly ask whether a token looks
settled now. Our proposal asks a harder question:

> What is the conditional risk of making this token irreversible?

The key distinction is that observational token/prediction stability,
interventional commitment safety, interventional reference-freeze safety, and
systems profitability are not the same object. Exact compute removal usually
requires a fixed reference state; once that reference is exact and all state
visible to other positions is frozen, the remaining question is mostly whether
row removal is worthwhile on the hardware, not whether the unobservable row
computation carries additional model-quality information.

Throughout the project, use three scientific hypothesis labels and keep the
systems proposition separate:

- **A:** Token-identity stability does not imply reference-state settlement.
- **B:** Commitment safety does not imply reference-freeze safety.
- **C:** Reference-state drift predicts the causal harm of reference freezing
  beyond matched controls.
- **U:** Reference-freeze safety does not imply compute profitability.

Above the ladder sits one further level we name but do not test:

- **Level 0 (meaning vs. tokens):** the same meaning can be carried by
  different tokens, so a token-identity change is only an *upper bound* on a
  meaning change. It matters for how outcomes are read — a changed token is not
  automatically a changed answer, which is why every result reports
  normalized-answer change alongside token change — but it is not a gate:
  nothing about meaning-level equivalence changes which rows can be removed
  from compute, and the efficient-decoding literature does not gate on it
  either. Treat it as a measurement caveat, not a hypothesis.

Terminology guardrail: do not call same-token identity "semantic convergence"
unless the metric really measures meaning. In this proposal, "semantic lock" is
shorthand for token-identity commitment. A token can keep the same surface form
while its meaning contribution changes with surrounding context, and two
different surface forms can express the same answer.

A separate, literature-adjacent question is whether committing a token
*earlier than the decoder would* is harmful. That is commitment-timing risk;
learned-commitment work (TraceLock and similar) already presses on it, and it
does not change what can be removed from compute. We measure it as an
auxiliary result, not as a premise.

`A` through `C` are model-behavior hypotheses. `U` is a systems utility
proposition whose truth depends on active-set size, sequence length, batch
size, cache layout, kernels, and hardware. Diagnostics such as non-target token
changes, answer changes, output divergence, latency, or throughput are
measurements used to test these claims.

| Stage | Decision | Gate type | Question | Failure if wrong |
| --- | --- | --- | --- | --- |
| Commitment | Token-identity lock | Causal quality risk | Can we irreversibly fix this token identity now? | The intervention changes sequence or task outcome. |
| Reference freeze | Reference lock | Causal representation risk | Can other tokens safely see a cached hidden/K/V reference? | Stale context changes downstream tokens or task outcome. |
| Compute removal | Row removal | Systems utility | Is skipping future row-wise compute faster after overheads? | Sparse execution adds overhead or fails to save latency. |

The proposed paper should show when token stability and reference-state
settlement diverge, when reference freeze is unsafe despite commitment safety,
and when safe reference freezes translate into real compute savings.

## Estimands and Examples

The cleanest version of PhaseLock uses different estimands for different
decisions.

First define token/prediction stability on the baseline trajectory:

```text
T_i,t = 1[argmax p_i,s = argmax p_i,T for every s >= t]
```

`T_i,t = 1` means the model's preferred token at position `i` no longer changes
after step `t`. This is a token-identity claim, not a meaning claim.

Define reference-state settlement at a tolerance relevant to the actual cached
state:

```text
R_i,t(delta) = 1[max_{s >= t, layer} D_KV(i, layer, t, s) <= delta]
```

where `D_KV` should measure the K/V or hidden reference exposed to active
tokens, not arbitrary private computation. A is the claim that
`P(R_i,t = 0 | T_i,t = 1) > 0`, but the meaningful result is not "some float
changed." The movement must be above numerical/background variation and large
enough to plausibly perturb active-token logits, attention outputs, or decoded
text.

Example for A: position `i` has top-1 token `"7"` from step 20 through the final
decode, but the masked tokens around it are still being filled in. As that
context changes, the token's K/V vectors continue moving by a large fraction of
their norm. The token is stable; the reference that other positions attend to
has not settled. A non-example is a row whose hidden state changes only at
floating-point noise scale and never changes any downstream logit.

For intervention safety, separate the outputs:

```text
Y0 = baseline output
YC = output when token identity is committed, but representation is recomputed
YF = output when token identity is committed and its reference state is frozen
YR = output when token identity is committed, reference is frozen, and row compute is removed
```

B has two useful readings:

```text
B1: P(YF != YC | YC = Y0) > 0
B2: P(YF != YC | observed_past_drift <= delta) > 0
```

B1 is the main causal premise: committing the token can be harmless while
freezing its stale reference changes later tokens. B2 is the online-control
premise: low drift over the last few observed steps does not prove future
reference settlement. If representation convergence meant oracle knowledge
that all future K/V states were identical, freezing would be safe in a
conventional transformer. The failure mode is weaker and more realistic:
past-local stability is not a certificate about the future.

Example for B1: forcing a stable `"7"` to remain `"7"` leaves the final answer
unchanged, so `YC = Y0`. But freezing the K/V row from step 20 makes a later
position attend to an outdated reference and changes `"21"` to `"24"`, so
`YF != YC`.

Example for B2: a SureLock-style KL gate sees two consecutive identical
posteriors and admits the token. One step later, newly unmasked context changes
the token's K/V state and the cached reference becomes stale. The online signal
was calm; the future reference was not settled.

C should be stated as a prediction-of-harm claim, not as "drift influences
answer quality":

```text
C: D_pre-freeze predicts 1[YF != YC] after controlling for cheap stability signals
```

Raw drift is probably insufficient. Harm depends on whether active tokens are
sensitive to the stale reference. A better first-order proxy is:

```text
reference risk ~= K/V drift * attention exposure
```

Large drift on a token that no active position attends to may be harmless.
Moderate drift on a token that many active positions attend to can matter.
More formally, the ideal object is closer to
`||J_active<-KV * Delta_KV||`: the active tokens' sensitivity to the stale K/V
perturbation, not the perturbation norm alone.

## Contribution Framing

### Contribution 1: Two risk gates plus one utility gate

We identify and operationalize three distinct decisions in diffusion LM
decoding:

- **Semantic lock:** the token identity is no longer eligible for remasking or
  replacement.
- **Reference freeze:** the token's hidden state or K/V state is cached as the
  reference exposed to other positions.
- **Row removal:** the token row is removed from future Q-projection,
  attention-output, and FFN updates while active positions still attend to its
  cached reference.

The first two are causal quality-risk gates. The third is a systems utility
gate. This separation matters because exact reference freeze should make the
locked row's later private computation unobservable to active tokens; removing
that computation is then mainly a question of latency, memory, packing, and
kernel overheads.

### Contribution 2: Token stability is not reference-state settlement

We show that fixing a token's identity is systematically different from that
token's reference state becoming settled. The identity question is trivially
closed once a token is committed: in confidence decoding a transferred token is
never remasked, so

```text
x_i,t = x_i,T   for all t after commit
```

holds by construction. The reference-state question is open:

```text
D_i,t = ||h_i,t - h_i,commit|| / ||h_i,commit||
```

which asks how far the committed row keeps moving after its identity stops
moving. The meaningful claim is not that `D` is merely nonzero, but that it can
remain large enough to matter to the cached hidden/K/V state that active tokens
see.

The interventional commitment-safety question is:

```text
C_i,t = 1[Y_commit(i,t) = Y_baseline]
```

This asks whether the final decoded sequence or task outcome remains unchanged
under an intervention that commits position `i` at step `t`.

These quantities need not agree. A token may already match its final identity
while forcing it early still changes the trajectory, or it may differ from its
eventual final token while the task answer remains unchanged. This
observational-versus-interventional gap is the central scientific distinction
from future-stability commitment policies.

### Contribution 3: Irreversible decoding as constrained risk allocation

We formulate locking as a constrained allocation problem with two causal risk
budgets and one utility gate: maximize saved compute while keeping commitment
risk and stale-reference risk below explicit budgets, then remove row-wise
compute only where expected runtime gain is positive. This shifts the objective
from "lock as many stable tokens as possible" to "spend irreversible actions
where their intervention risk is acceptable and their systems payoff is real."

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

Can a diffusion LM decoder save compute by controlling two intervention risks
and applying compute removal only when the runtime utility is positive?

```text
ACTIVE -> SEMANTIC-LOCKED -> REFERENCE-FROZEN -> COMPUTE-REMOVED
```

The target is not maximum locking. The target is maximum saved compute subject
to bounded commitment risk, bounded stale-reference risk, and positive compute
utility.

## Decision-Theoretic Formulation

Let `pi` be a locking policy. A simple objective is:

```text
maximize_pi   E[compute_saved(pi)]
subject to    R_commit(pi) <= eps_commit
              R_ref(pi) <= eps_ref
              E[g_compute(pi) | runtime state] >= g_min
```

Equivalently, for a tunable tradeoff:

```text
maximize_pi E[compute_saved]
            - lambda_commit E[commitment_regret]
            - lambda_ref E[stale_reference_regret]
            + lambda_gain E[g_compute]
```

The learned token-fate model is therefore not the whole contribution. It is one
possible proxy for commitment risk, invoked only where cheap signals become
unreliable; the final controlled quantities are intervention risks and realized
runtime gain, not just future-token stability.

## Relation to Prior Work

### SureLock

SureLock is the main systems baseline. It is training-free, locks unmasked
positions when adjacent-step posterior KL indicates stability, optionally adds
a confidence gate, caches locked K/V, and skips Q-projection and FFN rows for
locked positions. It establishes that row-wise compute removal can reduce
algorithmic FLOPs substantially.

Our difference:

- SureLock asks whether the local posterior is stable enough to stop compute.
- We ask whether token-identity lock and reference-freeze decisions satisfy
  calibrated intervention-risk budgets, then whether row removal has positive
  runtime utility.
- We can reuse SureLock's compute machinery, but replace or augment its lock
  admission rule with a phase-specific controller.

### TraceLock and learned commitment policies

TraceLock-like methods are the closest conceptual threat. They learn from
completed diffusion traces whether an intermediate token agrees with the final
token, then use that prediction to decide commit-or-revise.

Our difference cannot merely be "we predict future token stability." The
stronger distinction is:

- learned token fate is only one risk estimator;
- we apply it selectively where cheap signals are ambiguous;
- we separate token-identity lock risk from reference/cache risk and compute-removal
  utility;
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

### Ada-DLM, Prophet, and semantic convergence

Ada-DLM pressures any claim that PhaseLock is the first to notice scalar
criteria can miss semantic convergence. It uses confidence-trajectory features
to identify semantically stabilized tokens and includes system-level
optimization. Prophet pressures exact-sequence objectives from a different
direction: task answers may converge before full sequence refinement finishes.

Our difference:

- existing semantic-convergence methods improve when to stop or commit;
- PhaseLock treats same-token stability as only one observable and tests whether
  it predicts interventional commitment and reference-freeze safety;
- for task settings such as math and code, report both sequence-level outcomes
  and task-level outcomes.

### Polestar and representation-drift methods

Polestar is especially important because it connects token commitment and
cache/reuse through representation drift. This pressures any claim that token
stability implies freeze safety.

Our difference should be tested, not assumed:

- token fate: token identity stability;
- reference fate: hidden/K/V drift and cached-reference safety;
- computational fate: runtime utility under row-wise compute removal.

If experiments show these are separable, that becomes the scientific core of
the paper.

### Windowed caching and token-role systems

Window-Diffusion and related cache/pruning systems already give tokens
different computational roles such as active, buffer, cached, or far-field
positions. PhaseLock should not claim novelty from having multiple token states
alone. The contribution is that transitions are treated as intervention-risk
or utility decisions: token-identity lock controls commitment risk, reference freeze
controls causal representation risk, and row removal controls runtime utility.

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

Then apply a PhaseLock controller.

### Stage 1: Candidate Routing

Low-confidence positions are not lock candidates. They are obviously uncertain,
so running a learned fate model is usually wasted overhead.

High-confidence and apparently stable positions are candidates. These are the
positions where false confidence is dangerous and where a second risk estimate
has the highest value.

### Stage 2: Semantic Lock / Commitment-Risk Gate

Estimate:

```text
r_commit(i,t) = P(Y_commit(i,t) != Y_baseline | history up to t)
```

Semantic-lock only if:

```text
r_commit(i,t) <= eps_commit
```

A semantically locked token is no longer eligible for remasking or token
replacement, but its representation can still be recomputed. This isolates
token-identity intervention safety from reference-state safety and systems-level
compute removal.

### Stage 3: Reference-Freeze Risk Gate

Estimate or test whether the position is safe to expose through a cached hidden
row or cached K/V reference:

```text
r_ref(i,t) = P(Y_reference_freeze(i,t) != Y_commit(i,t) | history, commit-safe)
```

Reference-freeze only if:

```text
r_commit(i,t) <= eps_commit
r_ref(i,t) <= eps_ref
semantic_lock_age(i) >= k_ref
```

This step freezes the state other positions see. It does not yet have to claim
full systems speedup: the implementation may still recompute the row for
diagnostics, delayed validation, fallback, or partial-update experiments.

### Stage 4: Compute-Utility Gate

Estimate whether removing the reference-frozen row from future row-wise
computation is profitable after systems overhead:

```text
g_compute(i,t) = latency_reference_frozen - latency_row_removed
```

Compute-remove only if:

```text
r_ref(i,t) <= eps_ref
E[g_compute(i,t) | active_rows, seq_len, batch, hardware, cache_layout] >= g_min
```

The compute-removed token keeps cached K/V so active positions can still attend
to it, but skips future Q-projection, attention output, and FFN row updates. In
an exact implementation this should be output-equivalent to reference freeze;
the gate exists because sparse execution can fail to save latency after
packing, gather/scatter, cache, and kernel overheads.

### Compact Policy Sketch

```text
if confidence_i < tau_conf:
    keep_active(i)
elif cheap_stability_i is clearly unsafe:
    keep_active(i)
else:
    r_commit = commitment_risk_model(trace_i_up_to_t)
    if r_commit > eps_commit:
        keep_active(i)
    else:
        semantic_lock(i)

        if semantic_lock_age_i >= k_ref:
            r_ref = reference_risk_model_or_drift_gate(trace_i_up_to_t)
            if r_ref <= eps_ref:
                reference_freeze(i)

        if reference_frozen(i):
            g_compute = compute_utility_model(runtime_state)
            if g_compute >= g_min:
                remove_row_compute(i)
```

## What Makes This Worth Studying

The main empirical hypothesis is not just that a learned classifier has better
AUROC. It is:

> The marginal value of expensive risk estimation is concentrated among
> apparently safe lock candidates, and token stability is not freeze safety.

The core empirical test is whether observational settlement predicts
interventional safety. We should report:

```text
T_i,t = 1[token identity stays fixed on the baseline trajectory]
D_i,t = hidden/KV drift from the candidate freeze reference
C_i,t = 1[YC = Y0]      (commitment safety)
R_i,t = 1[YF = YC]      (reference-freeze safety)
```

Then measure the two disagreement sets:

```text
T_i,t = 1, D_i,t meaningfully above tolerance
C_i,t = 1, R_i,t = 0
```

The first set shows tokens whose identity is stable while their reference state
is not settled. The second set shows commit-safe but reference-freeze-harmful
tokens.

This produces three scientific hypotheses and one systems proposition:

1. **A:** Token-identity stability does not imply reference-state settlement: a
   token's identity can be permanently fixed while `D_i,t` stays meaningfully
   large. In a decoder that never remasks a transferred token, the antecedent
   holds by construction, so `A` is an observation rather than an intervention.
2. **B:** Commitment safety does not imply reference-freeze safety: `C=1` can
   occur while `R=0`.
3. **C:** High pre-freeze reference-state drift predicts reference-freeze harm
   beyond matched low-drift controls, especially when the token has high
   attention exposure from active positions.
4. **U:** Reference-freeze safety does not imply compute profitability: `R=1`
   does not guarantee `g_compute > 0`.

The method is worth building only if those hypotheses translate into useful
engineering behavior:

- selective risk estimation on lock candidates preserves most of the safety
  benefit at lower overhead than all-token prediction;
- risk-controlled phase admission beats raw convergence detection at matched
  semantic-lock rate, reference-freeze rate, or FLOP budget;
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
- SureLock criterion and SureLock cached-reference plus row-removal machinery;
- representation-drift gate inspired by Polestar;
- oracle token settlement;
- oracle commitment safety;
- oracle reference-freeze safety;
- oracle compute utility.

The goal is not to beat weak baselines. The paper only becomes credible if it
compares against the methods that reviewers will expect.

### Phase 1: Offline Fate Decomposition

Use completed traces and counterfactual replays to label observational fate,
commitment safety, reference-freeze safety, and compute utility.

Observational labels:

- Does the current top-1 token equal the final token?
- What is the earliest irreversible token-settlement step?
- Does a high-confidence token later change?

Commitment-intervention labels:

- If the token identity is committed at step `t` but full computation continues,
  does the final sequence change?
- Does the normalized answer or task correctness change?
- How often does a fixed identity coexist with an unsettled reference state?

Reference-freeze labels:

- How much do hidden states, K/V states, or logits drift after token
  settlement?
- If the token's hidden row or K/V reference is frozen after commitment, do
  other tokens change?
- Does answer correctness or generation quality change?
- Does reference-state drift remain high for commit-safe tokens?
- Which layers and positions show the largest commitment/reference gap?

Compute-utility labels:

- Given a reference-frozen row, does removing row-wise compute produce identical
  outputs up to numerical precision?
- How many active rows remain per step?
- What are the FLOP, cache-memory, packing, gather/scatter, and wall-clock gains
  under the target batch and sequence regimes?

This phase should produce the core scientific figure:

```text
token identity settled != reference state settled != reference-freeze safe
reference-freeze safety != compute profitability
```

### Phase 2: Four-Arm Counterfactual Audit

For high-confidence candidate token-steps, replay from the same state under
four interventions:

```text
A0 = baseline
A1 = token identity committed, representation still updated
A2 = token identity committed + hidden/K/V reference frozen
A3 = token identity committed + hidden/K/V reference frozen + row compute removed
```

Use `A0` versus `A1` to measure commitment risk, `A1` versus `A2` to measure
incremental reference-freeze risk, and `A2` versus `A3` to verify whether row
removal is output-equivalent once references are exact. If `A2` and `A3`
match up to numerical precision, do not define a separate compute quality-risk
estimator; keep only the compute-utility gate.

Report exact sequence change, non-target token change count, normalized answer
change, correctness change, next-step logit divergence, and terminal logit
divergence.

### Phase 3: Selective Risk-Control Evaluation

For each candidate method, plot:

```text
coverage / semantic-lock rate vs empirical commitment-intervention risk
coverage / reference-freeze rate vs empirical reference-freeze risk
```

Also plot:

```text
x-axis: lock rate or estimated FLOPs saved
y-axis: empirical risk-budget violation rate
```

The high-confidence disagreement subset should be explicit: token-steps where
confidence is high and cheap criteria invite locking, but the intervention
fails.

```text
semantic_miss_set = {
  (i,t): confidence_i,t > tau_conf and x_i,t != x_i,T
}
```

Also construct the two intervention disagreement subsets:

```text
commitment_false_safety_set = {
  (i,t): x_i,t = x_i,T and Y_commit(i,t) != Y_baseline
}

reference_false_safety_set = {
  (i,t): Y_commit(i,t) = Y_baseline
         and Y_reference_freeze(i,t) != Y_commit(i,t)
}
```

### Phase 4: Semantic-Lock Decoder

Modify the decoding loop so semantically locked tokens stay fixed but still
receive full model computation. This tests the semantic decision before adding
reference-cache or compute-lock systems complexity.

Report:

- final-answer accuracy on GSM8K and MATH500;
- code or structured-task accuracy if available;
- output divergence from the unlocked baseline;
- lock-induced answer changes;
- premature-lock diagnostics by confidence slice;
- quality at matched lock rate.

### Phase 5: Reference-Freeze Proxy

Before writing custom kernels, intervene on cached hidden rows or K/V states to
test whether semantically locked tokens can safely become fixed references.
This phase should estimate stale-reference risk without claiming wall-clock
speedup.

Report:

- non-target token changes under cached-reference interventions;
- normalized-answer changes under cached-reference interventions;
- drift quantiles before reference lock;
- matched high-drift versus low-drift control effects.

### Phase 6: Compute-Utility Proxy

After reference-freeze safety is plausible, simulate the compute path from
active-set trajectories.

Estimate:

- active rows per step;
- attention FLOPs `O(M_t N d)`;
- FFN FLOPs `O(M_t d^2)`;
- K/V cache memory;
- gather/scatter and packing overhead;
- expected speedup under realistic batch and sequence regimes.

This phase decides whether a real row-removal implementation is justified.

### Phase 7: End-to-End Row-Removal Decoder

Implement the full row-removal path only after semantic-lock and
reference-freeze proxy results are strong. Reuse the SureLock-style mechanism:
compute-removed tokens keep cached K/V, while active tokens attend to all
positions.

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

- `P(Y_commit != Y_baseline | semantic-locked)`;
- `P(Y_reference_freeze != Y_commit | reference-frozen)`;
- non-target token change rate after intervention;
- normalized-answer change rate and task-correctness change rate;
- sequence-level versus task-level intervention safety;
- calibration error for `r_commit` and `r_ref`;
- risk-budget violation rate.

Efficiency:

- token-step updates avoided;
- algorithmic FLOPs;
- wall-clock latency;
- throughput;
- compute utility `g_compute`;
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
- Commitment risk threshold `eps_commit`.
- Reference risk threshold `eps_ref`.
- Compute utility threshold `g_min`.
- Immediate reference freeze vs delayed reference freeze after `k_ref` semantic-lock
  steps.
- Immediate row removal vs delayed row removal after reference freeze.
- Semantic-only gate vs semantic plus reference-drift gate vs full PhaseLock
  with compute-utility gating.
- Reference freeze without row removal vs reference freeze plus row removal.
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

1. Token-stable positions whose reference states keep drifting beyond a useful
   tolerance exist at a nontrivial rate, establishing that token stability does
   not deliver a stable cached reference for free.
2. Commit-safe tokens that are reference-freeze-harmful exist at a nontrivial
   rate.
3. Reference-state drift, preferably exposure-weighted, predicts
   reference-freeze harm beyond matched controls.
4. Selective risk control reduces commitment and reference-freeze failures at
   matched lock/freeze coverage.
5. `A2` reference-freeze and `A3` row-removal outputs match up to numerical
   precision under an exact implementation, validating compute utility as a
   systems gate.
6. Active-set simulations show enough compute utility to justify systems work.

Stronger systems paper:

1. Implement full row removal with cached K/V.
2. Match or improve SureLock quality at similar FLOPs.
3. Save more FLOPs or latency at the same quality drop.
4. Demonstrate gains on more than one dLLM family.

## Reviewer-Resistant Framing

Avoid this claim:

> We learn whether high-confidence tokens will stay fixed.

That is too close to learned future-stability commitment work.

Use this claim:

> Locking is a risk-sensitive irreversible decision. We separate semantic,
> reference-freeze, and compute-utility decisions, then allocate expensive risk
> estimation only to the intervention phases where cheap convergence signals are
> unreliable.

The sharper title-level contribution is:

> Token stability is not freeze safety.

The secondary systems sentence is:

> Freeze safety is not compute profitability.

If experiments support those sentences, the project becomes more than a decoder
heuristic or another commitment score.

## Risks

- **Related-work crowding.** TraceLock, trajectory-aware commitment methods,
  LESS-style mutual stability, Ada-DLM, Prophet, Window-Diffusion, SureLock,
  and Polestar all occupy nearby space. The paper needs an intervention-risk
  and utility-gating contribution, not just another commitment score or token
  state taxonomy.

- **Existing diagnostic AUROC is not enough.** The `+0.0174` AUROC and
  `+0.0124` AP high-confidence result motivates the idea but cannot be the
  headline. Reviewers will care about quality at fixed compute or compute at
  fixed quality.

- **Commitment- and reference-risk labels are expensive.** Measuring whether a
  semantic lock or cached reference changes the final sequence can require
  counterfactual decoding. Start with a small candidate set and proxy labels
  based on representation drift.

- **Predictor overhead can erase savings.** The learned model must be small and
  selectively invoked. Otherwise a training-free criterion may win in practice.

- **Wall-clock speedup may lag FLOPs.** Irregular active sets, cache layout, and
  gather/scatter overhead can reduce real speedups. FLOPs alone are not enough.

## Immediate Next Experiment

Run the four-arm counterfactual audit before implementing a new decoder.

For each high-confidence candidate token-step, compare:

```text
A0 = baseline
A1 = token-identity lock, full representation updates
A2 = token-identity lock + reference freeze
A3 = token-identity lock + reference freeze + row removal
```

This directly estimates:

```text
A0 vs A1: commitment risk
A1 vs A2: reference-freeze risk
A2 vs A3: output equivalence of row removal after exact reference freeze
```

Then fit or compare risk policies:

- confidence;
- KL;
- confidence plus KL;
- persistence plus confidence;
- JSD/persistence-style stability;
- SureLock criterion;
- TraceLock-style learned token fate;
- selective commitment-risk model on high-confidence candidates;
- token-identity lock plus reference-drift reference-freeze gate;
- full PhaseLock: commitment-risk gate plus reference-risk gate plus
  compute-utility gate;
- oracle token settlement;
- oracle commitment safety;
- oracle reference-freeze safety.

If selective risk control does not dominate on this plot, the idea should be
rethought before any systems implementation.

## References to Track

- SureLock: <https://daioba.github.io/surelock/>
- SureLock arXiv: <https://arxiv.org/abs/2602.06412>
- TraceLock / learned token-commitment policy:
  <https://arxiv.org/abs/2605.24697>
- Ada-DLM / semantic-aware adaptive denoising:
  <https://aclanthology.org/2026.acl-long.819/>
- Prophet / early answer convergence:
  <https://proceedings.iclr.cc/paper_files/paper/2026/hash/daadbff4d4ea884ca3d9d389a1dfc61c-Abstract-Conference.html>
- Window-Diffusion:
  <https://arxiv.org/abs/2601.20332>
- LESS: <https://arxiv.org/abs/2606.16908>
- TACG: <https://arxiv.org/abs/2607.03236>
- Polestar: <https://arxiv.org/abs/2607.14107>
- Deferred Commitment Decoding: <https://arxiv.org/abs/2601.02076>
- CoCommit: <https://arxiv.org/abs/2607.04469>
- Fast-dLLM: <https://arxiv.org/abs/2505.22618>
- dKV-Cache: <https://arxiv.org/abs/2505.15781>
- dLLM-Cache: <https://arxiv.org/abs/2506.06295>
- d2Cache: <https://arxiv.org/abs/2509.23094>
