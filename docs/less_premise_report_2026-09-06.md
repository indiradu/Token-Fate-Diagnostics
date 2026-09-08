# LESS Premise Report - 2026-09-06

## Bottom Line

LESS's token-admission rule is useful for deciding when to commit a token
identity. It is not, by itself, evidence that the token's hidden representation
has stopped changing. Reusing that admission to freeze one hidden row changed
downstream output in 40/108 Level-5 MATH interventions and 23/140 HumanEval
interventions. This does not show that normal LESS decoding is unsafe: published
LESS continues recomputing hidden representations.

## Question

Does a token accepted by LESS's joint semantic-stability rule also have a
settled internal representation that can safely be frozen?

This is deliberately stronger than the claim made by LESS. Published LESS
changes when masked positions are committed and reduces the number of reverse
steps; it does not freeze hidden or K/V rows. The intervention below tests
whether a LESS admission could be reused as a reference-freeze decision.

## Reproduction Target

The baseline follows the [LESS paper](https://arxiv.org/abs/2606.16908) and the
[released LLaDA implementation](https://github.com/amr-mohamedd/LESS-Is-More/blob/4bfb6723ce703a52e0233e2229b8b374a2d0ccf0/llada_sampling/llada_less.py)
at upstream commit `4bfb6723ce703a52e0233e2229b8b374a2d0ccf0`:

- confidence threshold `c = 0.75`;
- top-8 coarsened inter-step JSD threshold `d = 0.04`;
- two stored prior top-1 winners, so acceptance first becomes possible on the
  third matching observation;
- histories reset at each active block;
- all jointly accepted positions commit together;
- when none are accepted, the released LLaDA path immediately falls back to
  the scheduled number of highest-confidence positions;
- greedy decoding, generation length 256, 256 maximum steps, block length 32.

The paper pseudocode differs from the released LLaDA code: it describes a
two-step skip budget and a persistence definition that can be read as requiring
only one prior winner when `P=2`. Those paper-spec semantics were not mixed
into the primary result and remain a sensitivity test.

## Intervention

For each example:

1. Run LESS normally. This is `YC`: committed token identities are fixed, but
   hidden rows continue to be recomputed.
2. Select two semantic-safe tokens that were admitted by the full LESS rule,
   excluding fallback commitments and EOS/EOT.
3. Replay the same LESS sampler.
4. Starting at the selected token's first post-commit forward pass, clamp its
   hidden row at layers 16, 24, and 32 to the baseline row from that pass.
   This is `YF`.
5. Compare `YF` with the LESS baseline `YC`.

Before a row is accepted for causal analysis, replay must exactly reproduce the
full baseline prefix. At every step before the freeze, the complete input-token
vector, raw top-1 vector, and LESS transfer mask must match. At the freeze step,
the input-token vector must match; top-1 and transfer decisions may then differ
because the intervention is active. The analyzer rejects the run if any prefix
is invalid.

All 248 interventions passed this check. Every selected token was committed
identically in replay, and no selected target token changed. Reported token
changes are therefore downstream of the frozen row.

### What the metrics mean

- **Downstream output changed:** at least one generated token other than the
  deliberately frozen target differs between `YC` and `YF`.
- **Answer/program changed:** the extracted MATH answers are not symbolically
  equivalent, or the normalized HumanEval programs differ. This does not imply
  that correctness changed.
- **Original accuracy:** the normal LESS output `YC` is checked against gold.
- **Modified accuracy:** the frozen-row output `YF` is checked against gold.
- **Correct -> wrong / wrong -> correct:** the two directions are counted
  separately, so wrong-to-wrong answer changes are not mislabeled as accuracy
  changes.

## Results

Intervals use 10,000 example-clustered bootstrap resamples. Level-5 is the full
held-out 54-example partition (`offset=80`) of the 134 Level-5 records; the first
80 records were reserved by the existing experiment split. HumanEval is the
tested 70-example slice of 164 records. Each example contributes two sampled
eligible admissions.

| Metric | Level-5 MATH, gen256 | HumanEval, gen256 |
| --- | ---: | ---: |
| Examples | 54 | 70 |
| Interventions | 108 | 140 |
| Mean LESS baseline NFEs | 126.3 / 256 | 49.7 / 256 |
| LESS-accepted share of committed tokens | 66.6% | 88.5% |
| Downstream output changed | 37.0% [27.8, 47.2] | 16.4% [9.3, 24.3] |
| Mean downstream tokens changed | 25.91 [16.88, 35.69] | 7.82 [3.54, 13.01] |
| Answer/program changed | 20.4% [13.0, 28.7] | 16.4% [9.3, 24.3] |
| Original accuracy | 7.4% [1.9, 14.8] | 58.6% [47.1, 70.0] |
| Modified accuracy | 7.4% [1.9, 14.8] | 57.1% [45.7, 68.6] |
| Accuracy delta | 0.0 pp | -1.4 pp [-3.6, 0.0] |
| Correct -> wrong | 0 / 108 | 2 / 140 |
| Wrong -> correct | 0 / 108 | 0 / 140 |
| Scoring infrastructure failures | 0 | 0 |

### Concrete examples

- **Answer changed, correctness did not:** on `math500-l5-327`, the normal LESS
  answer was `3125`; freezing one admitted row changed it to `5`. The gold answer
  is `15`, so both outputs are wrong. This is an answer change but not a
  correctness change.
- **Reasoning changed, answer did not:** on `math500-l5-425`, one intervention
  changed 175 downstream tokens while both decodes still extracted `9r^2`.
  This is an output change but not an answer change.
- **Correct program became wrong:** on `HumanEval/5`, the normal program appends
  the delimiter and then the next number. The frozen-row program reverses those
  two operations, so it fails the tests.
- **A second correctness failure:** on `HumanEval/25`, freezing an admitted row
  replaces the factorization loop with a truncated loop that appends the
  remaining composite value. The normal program passes; the modified one fails.

These examples explain how 23 HumanEval programs can change while only two
correctness labels change, and how 22 Level-5 answers can change with no
correctness transition.

### Representation movement after LESS commitment

For each token, drift is the relative L2 distance from its first post-commit row.
The table reports the median, across tokens, of the mean of the per-layer maximum
drifts at layers 16, 24, and 32. EOS/EOT rows are excluded.

| Dataset | LESS-accepted body tokens | Median max post-commit drift | Fraction above 0.25 |
| --- | ---: | ---: | ---: |
| Level-5 MATH | 8,878 | 0.469 | 91.9% |
| HumanEval | 10,035 | 0.448 | 94.9% |

LESS acceptance selects somewhat lower-drift tokens than fallback commitment,
but it does not select representation-settled tokens. Fallback-token medians
were 0.517 on Level-5 MATH and 0.521 on HumanEval.

## Interpretation

The supported claim is:

> LESS's semantic commitment rule is not a reference-settlement certificate.

The result is clear at the output level. Freezing one accepted row changes
downstream output in 40/108 Level-5 interventions and 23/140 HumanEval
interventions. This replicates the PhaseLock separation across symbolic math
and executable code.

The accuracy claim is weaker. Level-5 had four correct baseline examples and no
intervention changed a correctness label. HumanEval had two
passing-to-failing transitions and a -1.4 percentage-point aggregate change,
but the interval includes zero. This is not resolved evidence of systematic
accuracy harm.

The LESS sampler itself is not contradicted. Its normal baseline used only
49.3% of the maximum forward evaluations on Level-5 MATH and 19.4% on
HumanEval. Baseline accuracy was unchanged from the prior fixed-schedule
Level-5 run and higher on this HumanEval sample. The failure appears only when
the semantic admission is given the additional job of authorizing reference
freezing.

## Relation To SureLock

| Dataset | Strategy reused for freeze | Mean freeze step | Downstream change | Answer/program change |
| --- | --- | ---: | ---: | ---: |
| Level-5 MATH | LESS admission | 66.8 | 37.0% | 20.4% |
| Level-5 MATH | SureLock-style lock | 169.5 | 21.3% | 8.3% |
| HumanEval | LESS admission | 15.7 | 16.4% | 16.4% |
| HumanEval | SureLock-style lock | 114.8 | 10.0% | 10.0% |

This table must not be read as "LESS is worse than SureLock." LESS acts much
earlier, uses a different baseline sampler, and is designed for semantic
commitment rather than row freezing. A fair strategy ranking requires matched
compute savings or matched remaining horizon and a common-token analysis.

In 56.5% of Level-5 interventions and 30.7% of HumanEval interventions, row
freezing changed the number of subsequent LESS forward evaluations. The
reported output difference is therefore the total effect of stale-reference
reuse inside an adaptive sampler, including its downstream effect on later
LESS admission decisions.

## Decision

Keep the result. It supports PhaseLock's separation of semantic commitment from
reference freezing and does so with a faithful released-code LESS rule rather
than the earlier post-hoc JSD proxy.

The next discriminating experiment is not another natural-threshold comparison.
It is a matched-savings comparison among LESS, SureLock, confidence-only, and a
representation-drift gate, with both policy-level and common-token analyses.

## Limitations

- The reference-freeze arm is hypothetical reuse, not published LESS behavior.
- The effect rates are conditional on eligible LESS admissions with measurable
  post-commit future state, with at most two randomly sampled per example.
- Level-5 covers the held-out 54-example partition, not all 134 Level-5 records;
  HumanEval covers 70 of 164 tasks.
- Block length 32 matches this repository's SureLock experiments; the paper's
  headline LLaDA evaluations use block length 64.
- The released-code and paper-pseudocode variants differ and only the released
  variant was run.
- The clamp covers hidden rows at layers 16, 24, and 32, not actual K/V cache
  tensors or row removal.
- One deterministic decode and one candidate-sampling seed were used.
- Accuracy effects remain underpowered.

## Artifacts

- `results/less_premise_l5g256_54x2_20260906/`
- `results/less_premise_reanalysis_l5g256_54x2_20260906/`
- `results/less_premise_humaneval_g256_70x2_20260906/`
- `results/less_premise_reanalysis_humaneval_g256_70x2_20260906/`
- `results/less_vs_surelock_premise_suite_20260906/`
