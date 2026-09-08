# PhaseLock Clear Idea Report - 2026-09-05

This report incorporates the critique in `discuss.md` into the active
PhaseLock framing. The main change is conceptual precision: PhaseLock is not
about generic "convergence." It is about which irreversible action is safe to
take, using only the information available online.

## Short Version

A diffusion LM can make a token look settled before the computation attached to
that token is safe to freeze or remove.

PhaseLock separates three decisions:

1. **Token-identity lock:** keep this token fixed from now on.
2. **Reference freeze:** cache the token's hidden/K/V state as the reference
   that other positions attend to.
3. **Row removal:** skip future row-wise computation for that token because the
   cached reference is enough and the runtime actually gets faster.

The project should not claim merely that hidden states move after token
stability. That is too easy. The real claim is that token-stable positions can
have reference-state movement large enough to matter for downstream computation,
and that cheap online stability signals do not fully certify freeze safety.

## The Four Core Claims

### A. Token Stability Is Not Reference-State Settlement

Plain statement:

> The model can keep predicting the same token while the reference state for
> that token is still changing.

More precise statement:

```text
P(reference_not_settled | token_identity_stable) > 0
```

Here "token identity stable" means the top token at a position stays the same
for the rest of the baseline trajectory. It does not mean the same thing as
"semantic convergence" in the meaning-level sense.

Example:

- At step 20, position 50 predicts token `"7"`.
- From step 20 to the end, position 50 always remains `"7"`.
- But masked context around it continues to fill in.
- The K/V vector for position 50 keeps changing as the context evolves.
- Other tokens that attend to position 50 would see a different reference if we
  froze it at step 20.

This supports A only if the movement is meaningful: above numerical noise,
above normal background variation, and relevant to the hidden/K/V reference
that active tokens actually use.

Non-example:

- The token remains `"7"`.
- The hidden row changes at floating-point noise scale.
- No downstream logits, tokens, or answer text change.

That is not a useful PhaseLock result.

### B1. Commitment Safety Is Not Reference-Freeze Safety

Plain statement:

> Fixing the token identity can be harmless while freezing its stale reference
> changes the output.

Use three outputs:

```text
Y0 = baseline output
YC = output after committing the token identity, while still recomputing its representation
YF = output after committing the token identity and freezing its reference
```

The clean B1 failure is:

```text
YC = Y0, but YF != YC
```

Example:

- Baseline answer: `"21"`.
- If we force one already-stable token to stay fixed, the answer remains
  `"21"`. This means the token-identity intervention was safe.
- If we also freeze that token's K/V reference from an earlier step, a later
  token changes and the answer becomes `"24"`.
- The problem was not committing the token. The problem was serving active
  positions a stale reference.

This is the most important PhaseLock premise because it isolates semantic/token
commitment from cache/reference safety.

### B2. Low Observed Drift Is Not A Future-Settlement Certificate

Plain statement:

> A token can look locally stable over the last few steps and still have future
> reference-state movement.

This is an online-control claim. The word "observed" matters.

If an oracle told us that all future K/V states would be identical to the
current K/V state, reference freezing should be safe in a conventional
transformer. But online rules do not have that oracle. They only see past and
current states.

Example:

- A SureLock-style KL rule sees adjacent-step posterior KL equal to zero.
- A LESS-style rule sees persistent top-1 plus low JSD.
- The token is admitted for locking.
- Later, newly unmasked context changes the token's K/V state.
- The frozen reference becomes stale even though the recent online signal was
  calm.

B2 says that past-local calm is not the same as future reference settlement.

### C. Drift Should Predict Causal Freeze Harm

Plain statement:

> Reference-state drift should help predict when freezing the reference will
> actually change the output.

The safer hypothesis is not "drift influences answer quality." That language
overclaims the mechanism. The estimand is:

```text
D_pre_freeze predicts 1[YF != YC]
```

after controlling for cheap stability signals such as confidence, entropy,
posterior KL/JSD, step, remaining horizon, position, token type, local mask
ratio, context volatility, and attention received from active tokens.

Raw drift is not enough. Harm depends on whether active tokens are sensitive to
the stale reference.

Useful proxy:

```text
reference risk ~= K/V drift * attention exposure
```

Examples:

- High drift, low exposure: a token's K/V changes a lot, but no active token
  attends to it. Freezing may be harmless.
- Moderate drift, high exposure: many active tokens attend to the stale token.
  Even a smaller perturbation may change logits or decoded text.

The ideal object is closer to:

```text
||J_active<-KV * Delta_KV||
```

That is, how much the stale K/V perturbation changes the active tokens'
computation, not just how large the perturbation is in isolation.

### U. Freeze Safety Is Not Compute Profitability

Plain statement:

> Even if freezing the reference is safe, removing row compute may not actually
> make the system faster.

Example:

- A policy freezes many references without changing outputs.
- But active rows become irregular.
- Packing, gather/scatter, cache layout, and kernel overhead dominate.
- Wall-clock latency does not improve.

That means U is a systems proposition, not a model-behavior proposition. It
requires runtime measurement, not only a quality proxy.

## What The Paper Should Not Claim

Avoid:

> We found that token-stable hidden states still move.

That is weak because small harmless movement is expected.

Avoid:

> We prove semantic convergence is unsafe.

Same-token identity is not meaning-level semantic convergence.

Avoid:

> Drift causes answer degradation.

The current safer claim is that drift predicts the causal harm of freezing,
especially when weighted by attention exposure. Causation belongs to the
intervention comparison, not the raw drift statistic.

## The Clean Paper Claim

The clean claim is:

> Token stability is not freeze safety, and freeze safety is not compute
> profitability.

Expanded:

> Efficient diffusion-LM decoders should not collapse token commitment,
> reference caching, and row removal into one convergence test. These are
> separate decisions with separate risks: token-identity risk, stale-reference
> risk, and runtime-utility risk.

## Minimal Clear Example

Imagine one position in a math solution.

Baseline:

```text
... 7 + 14 = 21
```

At step 20, token `"7"` appears stable.

Case 1: token lock only.

```text
Commit "7"; keep recomputing its hidden/KV state.
Final output still says: 7 + 14 = 21
```

Commitment was safe.

Case 2: token lock plus early reference freeze.

```text
Commit "7"; freeze its step-20 K/V as the reference.
Later active tokens attend to that stale reference.
Final output changes: 7 + 17 = 24
```

Reference freeze was unsafe, even though token commitment was safe.

Case 3: delayed reference freeze.

```text
Wait until posterior KL is very low.
Freeze the later K/V reference.
Output usually changes less often, but still sometimes changes.
```

The delay helps, but it is not a proof of safety.

Case 4: row removal.

```text
If the reference is frozen safely, skip future row-wise compute for "7".
Measure whether this actually speeds up the model on the target hardware.
```

This is the utility question. It is not answered by token stability or by
freeze safety alone.

## Practical Consequence

The PhaseLock controller should have separate gates:

```text
active token
  -> token-identity lock if commitment risk is low
  -> reference freeze if stale-reference risk is low
  -> row removal if expected runtime gain is positive
```

The strongest next research direction is not another broad token-fate AUROC.
It is a risk frontier: at matched lock/freeze coverage or matched compute
budget, does a PhaseLock-style controller reduce `YF != YC` failures compared
with confidence, KL/JSD, persistence, and SureLock-/LESS-style rules?
