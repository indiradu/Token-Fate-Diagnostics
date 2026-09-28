# Trace-assisted PhaseLock speed protocol

## Question

Can PhaseLock obtain most of TraceLock's semantic speed while preserving its
separate representation-safety and compute-profitability decisions?

The experiment does not merge the three locks. Phase 1 proposes token-identity
commits, Phase 2 independently decides when committed references may be frozen,
and Phase 3 independently decides whether row-sparse execution is predicted to
be profitable.

## Fixed setup

- Model: LLaDA-8B-Instruct, BF16, one RTX 5000 Ada GPU.
- Datasets: GSM8K, MATH500, and HumanEval test splits.
- Schedule: 128 generation tokens, 128 denoising steps, block length 32.
- Batch size: 4.
- Seed: 251.
- Trace controller: the already-trained 8,000-example LLaDA transfer controller,
  threshold 0.95, layers 30/31/32, window width 32.
- Phase 2: frozen `mli_ultra` gate using representation layers 8/16/24,
  influence layer 24, drift threshold 0.002, influence threshold 0.5,
  patience 6, minimum semantic age 4, maximum lock fraction 0.04, and at most
  eight sampled queries.
- Phase 3: the existing exact fused executor and the pre-existing RTX 5000 Ada
  latency table. The gate requires locked fraction >= 0.20, at least eight
  remaining steps, at least 1,500 projected saved row-forwards, and predicted
  gain >= 0.03.

All policies use the same model, prompts, examples, schedule, trained controller,
projection, hardware, executor, and timing method as the prior TraceLock and
SureLock comparison. No controller is retrained on evaluation examples.

## Phase-1 speed policies

Each policy first restricts candidates to TraceLock's 32-token soft-crop window
and requires future-stability probability greater than 0.95. It then applies a
language-model confidence veto and a per-update cap over currently masked
positions in that window. If no position passes the threshold, the highest
future-stability position in the same window is used as the Trace-style progress
fallback.

| Policy | LM confidence | Maximum committed fraction per update |
|---|---:|---:|
| guarded | 0.90 | 0.25 |
| balanced | 0.80 | 0.50 |
| open | none | 1.00 |

The policies are fixed before calibration results are inspected.

## Data separation

Calibration uses 24 examples that do not overlap the prior baseline evaluation:

- GSM8K: indices 800--807.
- MATH500: indices 430--437.
- HumanEval: indices 100--107.

Final evaluation uses exactly the 60 examples used by the published-baseline
comparison:

- GSM8K: indices 820--839.
- MATH500: indices 450--469.
- HumanEval: indices 120--139.

## Selection rule

A policy is eligible only if, over all 24 calibration examples:

1. A0 -> A1 has zero correctness regressions;
2. A1 -> A2 has zero correctness regressions and exact decoded-output rate 1;
3. A2 -> A3 has zero correctness regressions and exact decoded-output rate 1.

Among eligible policies, select the policy with the largest pooled estimated
online gain relative to its paired dense A0 runtime. Ties within one percentage
point choose the more conservative policy in the order guarded, balanced, open.
If no policy is eligible, report the failure and select none; do not relax the
quality rule after seeing the data.

Only the selected policy is run on the final 60-example evaluation set.

## Reported measurements

- Accuracy and normalized-answer transitions for A0, A1, A2, and A3.
- Exact decoded-output agreement for every adjacent arm.
- Mean model forward evaluations (NFE).
- Semantic-controller and Phase-2 controller overhead.
- Fused-executor A3-versus-A0 and A3-versus-A2 paired runtime gains.
- Estimated online gain including measured Phase-1 and incremental Phase-2
  controller overhead.
- Reference-freeze counts, removed row-forwards, and sparse-switch rate.

The current planner and fused executor are replayed separately so the inclusive
number is an estimate, not yet a single-kernel serving measurement. Quality and
output-equivalence claims are exact replays; runtime claims must retain this
qualification until the controllers and fused executor are integrated.

## Invalidated pilot

The first smoke and partial calibration jobs (`196370`, `196380`, `196381`, and
the cancelled `196382`--`196388` sweep) are excluded. Their controller attention
was soft-cropped, but their candidate mask was not, so positions outside the
published TraceLock decision window could be selected. The corrected sweep is
identified by the `cropfix` output prefix and starts at job `196438`.

## Calibration outcome and fallback revision

The corrected guarded, balanced, and open policies all failed the original
selection rule on the same example, `math500-test-434`. Each policy had one
A0 -> A1 correctness regression and three improvements over the 24 calibration
examples. All A1 -> A2 and A2 -> A3 outputs were exact. No original policy was
selected and no held-out evaluation was launched.

The shared failure isolates the Trace-style progress fallback: when no token
passes the 0.95 future-stability threshold, it selects the maximum Trace score
without considering language-model confidence. Two preregistered revisions keep
the fast open proposal path and alter only this fallback:

| Policy | Threshold-passing proposals | No-threshold fallback |
|---|---|---|
| open_confidence | all tokens with Trace probability > 0.95 | maximum LM confidence |
| open_joint | all tokens with Trace probability > 0.95 | maximum Trace probability x LM confidence |

Both are screened first on the same eight MATH500 calibration examples. A policy
with any correctness regression is dropped. Every surviving policy is then run
on the eight GSM8K and eight HumanEval calibration examples. The original
zero-regression and exact downstream-equivalence rule remains unchanged; among
eligible revisions, estimated online gain selects the final policy, with
`open_confidence` preferred when gains are within one percentage point. Only
that selected revision may be evaluated on the held-out 60 examples.

## Final outcome

`open_joint` failed the MATH500 screen with one regression and was dropped.
`open_confidence` had zero regressions on all 24 calibration examples and was
selected. On calibration it preserved aggregate accuracy, made A1 -> A2 and
A2 -> A3 exactly output-equivalent, reduced mean NFE from 128 to 92.17, and had
an estimated online gain of 25.38%.

The selected policy then ran once on the fixed 60-example held-out set with four
timing repeats. It preserved every example's correctness status, retained exact
A0 sequences on 52/60 examples, and changed normalized answers on 5/60. Both
downstream transitions were exact on 60/60. Mean NFE was 100.73. The estimated
online gain was 19.24% [13.79%, 24.76%], and the independently gated Phase-3
increment was 1.36% [0.43%, 2.44%]. The compute gate switched on 6/15 batches
and abstained on every HumanEval batch.

Runtime remains an additive online estimate: exact fused A3 replay plus measured
Phase-1 controller and incremental Phase-2 controller overhead. It is not yet a
single integrated serving-kernel measurement.
