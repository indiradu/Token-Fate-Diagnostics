# Direct commitment-safety PhaseLock protocol

## Objective

This experiment replaces Phase 1's observational-stability proxy with a model
trained on the intervention it is meant to authorize:

\[
\hat p_{t,i}=P(C_{t,i}=1\mid
c,m,H,a,\tau,b,d,I).
\]

The features are LM confidence \(c\), top-two margin \(m\), entropy \(H\),
token-identity age \(a\), TraceLock stability probability \(\tau\), within-block
age \(b\), maximum hidden cosine drift over layers 8/16/24 \(d\), and sampled
layer-24 attention influence \(I\).

The strict primary label is sequence safety:

\[
C_{t,i}=1 \iff y^{\mathrm{commit}(t,i)}=y^{\mathrm{dense}}.
\]

Answer preservation, target-token preservation, and no correctness regression
are logged as secondary outcomes. They are not substituted for the primary
label during this pilot.

## Intervention

Candidates are unresolved rows that the ordinary confidence transfer schedule
would not select at update \(t\). The candidate's current top token is added to
the ordinary transfer set. No ordinary transfer is displaced. The committed
token identity is then held fixed while every hidden representation continues
to update densely. This is an A1 semantic intervention; it neither freezes
hidden/K/V references nor removes rows.

Candidates are sampled without future labels. For each example, six rows are
spread across the range of a composite observational-safety proxy and, when
possible, across distinct output positions. This deliberately includes both
stable-looking and unstable-looking states while avoiding selection on the
counterfactual outcome.

Each intervention is replayed at batch size one, matching the candidate's
batch-one baseline trajectory. This avoids treating BF16 ranking changes caused
only by a different GEMM batch shape as effects of the commitment intervention.
If exact replay makes a registered target part of the ordinary transfer set,
or its registered top token no longer matches, the candidate has no valid
distinct treatment. It is excluded before labeling and recorded in
`replay_exclusions.csv`; it is never silently assigned a safe outcome.

## Leakage controls

- All candidates from an example remain in one partition.
- Development and held-out example ranges are disjoint.
- The fit/calibration split is grouped by `(dataset, example_id)`.
- The deployment threshold is chosen only on calibration examples.
- Held-out intervention labels are used once for evaluation, never to tune the
  threshold.
- The end-to-end policy is evaluated on the same held-out prompts after the
  causal-prediction report is fixed.

## Models and gate

Two calibrated logistic models are tested:

- `cheap`: confidence, margin, entropy, token age, TraceLock probability, and
  block age;
- `full`: all cheap features plus multi-layer drift, attention influence, and
  a previous-hidden-state availability indicator.

The simple model is intentional for the first causal test: it is calibrated,
interpretable, cheap to execute as one fused linear score, and difficult to
overfit silently on a small intervention set.

For each model, the selected threshold maximizes calibration coverage subject
to zero observed unsafe commitments and at least eight selected calibration
rows. If no threshold satisfies that rule, the learned gate disables itself;
the ordinary schedule still guarantees decoding progress. The held-out Wilson
upper bound is reported so zero observed failures is not misrepresented as
proof of zero risk.

Confidence and TraceLock receive thresholds selected with the identical causal
label, calibration split, minimum count, and zero-failure rule.

## End-to-end policy

At each update, Phase 1 first applies the ordinary transfer schedule. It then
adds at most two unresolved candidates whose predicted commitment safety
exceeds the fixed threshold. There is no learned fallback. This cap limits
interactions between single-token interventions during the first pilot.

The benchmark compares dense, confidence, TraceLock, cheap direct safety, and
full direct safety. It records exact sequence agreement, answer changes,
correctness, NFE, semantic plans, controller time, and wall-clock time.

The saved semantic plans are suitable for a later fixed replay through
PhaseLock Phase 2 and Phase 3. Using drift and influence as Phase-1 features
does not perform representation locking: Phase 2 still makes an independent
decision about freezing a committed row.

## Pilot slices

| Dataset | Development | Held out |
|---|---:|---:|
| GSM8K | 700–707 | 820–827 |
| MATH500 | 400–407 | 450–457 |
| HumanEval | 100–107 | 120–127 |

All runs use LLaDA-8B-Instruct, BF16, 128 generated tokens, 128 denoising
updates, and block length 32. This 24-development/24-held-out pilot is for
algorithm selection. A larger disjoint confirmation is required before a
paper-level safety or speed claim.

## Decision rule

The revision advances only if a direct-safety model improves held-out causal
ranking or safe coverage over both confidence and TraceLock, does not introduce
a correctness regression in end-to-end decoding, and reduces NFE. A model that
predicts well but loses wall-clock time remains a diagnostic Phase-1 policy
until its controller is made profitable. The selected Phase-1 semantic plans
must then be replayed unchanged through the frozen Phase-2 and Phase-3 audit.
