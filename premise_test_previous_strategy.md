# Premise Tests of Previous Strategies

This document explains, in simple terms, what we tested with two previous
strategies:

1. a **SureLock-style stability rule**;
2. the released **LESS** decoding rule.

The central question was the same in both cases:

> If a strategy says a token is stable, is the token's internal representation
> also stable enough to stop recomputing it?

Our answer is:

> **No. Token stability is not a reliable certificate of representation
> stability.**

Freezing an apparently stable token's hidden representation often changed
later output. However, the experiments do **not** yet show a reliable decrease
in final task accuracy.

## The Basic Idea

A token has two different kinds of state:

- **Token identity:** the visible token, such as `5`, `therefore`, or `return`.
- **Internal representation:** the hidden vectors used by other tokens during
  later transformer computations.

The visible token can remain fixed while its hidden representation continues
changing as the surrounding masked text develops.

We compared two executions:

- **Original (`YC`):** keep the chosen token identity fixed, but continue
  recomputing its hidden representation normally.
- **Modified (`YF`):** keep the same token identity and freeze selected hidden
  rows, so later steps repeatedly see the old representation.

The selected target token did not change. Therefore, when later tokens changed,
the cause was the stale hidden representation rather than a different target
token.

## How To Read The Measurements

These measurements answer different questions:

- **Output changed:** at least one downstream token changed. The final answer
  may still be the same.
- **Answer/program changed:** the extracted mathematical answer or generated
  program changed. Both versions may still be wrong, or both may still pass.
- **Original accuracy:** how often the normal recomputing output matched the
  gold answer or passed the tests.
- **Modified accuracy:** how often the frozen-representation output matched the
  gold answer or passed the tests.
- **Correct -> wrong:** the original output was correct but freezing made it
  incorrect.
- **Wrong -> correct:** the original output was incorrect but freezing made it
  correct.

For example, changing an answer from one wrong value to another wrong value is
an **answer change**, but it is not a **correctness change**.

# Report 1: SureLock-Style Rule

## What SureLock-Style Stability Means Here

The rule waits until an already committed token has the same posterior across
two adjacent steps. We used the strictest threshold: adjacent-step posterior KL
had to be numerically zero.

This is a strong posterior-stability signal. The experiment asks whether that
signal also means the hidden reference row is safe to freeze.

## Main SureLock Results

| Dataset | Interventions | Output changed | Answer/program changed | Original accuracy | Modified accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Level-5 MATH, gen256 | 108 | 23 (21.3%) | 9 (8.3%) | 7.4% | 7.4% |
| HumanEval, gen256 | 140 | 14 (10.0%) | 14 (10.0%) | 54.3% | 55.0% |

On Level-5 MATH, freezing a row selected by the SureLock-style rule changed
downstream output in more than one intervention in five. Nine interventions
changed the mathematical answer. Nevertheless, no correctness label changed:
the original and modified accuracies were both 7.4%.

On HumanEval, 14 programs changed. Thirteen retained their previous correctness
status. One changed from failing to passing, and none changed from passing to
failing. This is evidence that stale references affect generated code, but it
is not evidence that freezing systematically lowers code accuracy.

## SureLock Example

An earlier evaluator incorrectly displayed one original answer as `36`. The
fixed evaluator correctly reads the change as:

- original answer: `193/36`;
- modified answer: `5`;
- gold answer: `13/6`.

The answer changed, but both values were wrong. Therefore correctness did not
change.

## Horizon Result

SureLock-style freezing became more disruptive at generation length 256 than
at generation length 64.

For Level-5 MATH:

- gen64 output-change rate: 7.4%;
- gen256 output-change rate: 21.3%.

The paired horizon increase was 13.9 percentage points, with a 95% bootstrap
interval of 2.8 to 25.0 points. Longer remaining trajectories give a stale row
more opportunities to influence later tokens.

## SureLock Conclusion

The SureLock-style posterior gate is informative: waiting until the posterior
stabilizes reduces harm compared with freezing immediately. But even numerical
posterior identity does not guarantee that hidden representations have settled.

Supported claim:

> **Posterior stability is not sufficient for reference-freeze safety.**

Not supported by these experiments:

> **SureLock-style freezing systematically reduces final accuracy.**

# Report 2: LESS

## What LESS Stability Means Here

The LESS baseline follows the released LLaDA implementation. A position is
accepted when all three conditions hold:

1. confidence is at least `0.75`;
2. the top prediction matches the two stored previous predictions;
3. top-8 distribution drift, measured by coarsened JSD, is at most `0.04`.

If no position passes, LESS falls back to the scheduled highest-confidence
positions. We excluded these fallback commitments from the intervention. Every
tested token was genuinely admitted by the full LESS rule.

Published LESS commits token identities and continues recomputing hidden
representations. It does **not** freeze hidden or K/V rows. Our experiment asks
a stronger hypothetical question: can a LESS admission also authorize hidden-
row freezing?

## Main LESS Results

| Dataset | Interventions | Output changed | Answer/program changed | Original accuracy | Modified accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Level-5 MATH, gen256 | 108 | 40 (37.0%) | 22 (20.4%) | 7.4% | 7.4% |
| HumanEval, gen256 | 140 | 23 (16.4%) | 23 (16.4%) | 58.6% | 57.1% |

All 248 LESS interventions reproduced the complete pre-freeze baseline history:
input tokens, raw top predictions, and LESS transfer decisions. None changed the
selected target token. This makes the downstream comparison especially clean.

On Level-5 MATH, 40 interventions changed downstream output and 22 changed the
mathematical answer. No intervention changed correctness. Original and modified
accuracy were both 7.4%.

On HumanEval, 23 interventions changed the generated program. Two changed a
passing program into a failing program. There were no failing-to-passing
changes. Accuracy moved from 58.6% to 57.1%, but its bootstrap interval included
zero, so this is not decisive evidence of systematic accuracy loss.

## LESS Examples

### Answer changed but remained wrong

For `math500-l5-327`:

- original LESS answer: `3125`;
- frozen-row answer: `5`;
- gold answer: `15`.

This counts as an answer change. It does not count as a correctness change
because both answers are wrong.

### Many tokens changed but the answer stayed the same

For `math500-l5-425`, one intervention changed 175 downstream tokens, but both
executions still produced the extracted answer `9r^2`.

This counts as an output change. It does not count as an answer change.

### A passing program became incorrect

For `HumanEval/5`, the normal program appended the delimiter and then the next
number. Freezing one admitted row reversed those two operations. The modified
program failed the tests.

For `HumanEval/25`, freezing one row truncated the factorization loop and caused
the program to append a remaining composite value. The normal program passed;
the modified program failed.

## Representations Continued Moving After LESS Admission

We measured relative L2 movement from the first post-commit hidden row at
layers 16, 24, and 32. EOS/EOT rows were excluded.

| Dataset | LESS-admitted tokens | Median maximum drift | Fraction above 0.25 |
| --- | ---: | ---: | ---: |
| Level-5 MATH | 8,878 | 0.469 | 91.9% |
| HumanEval | 10,035 | 0.448 | 94.9% |

Therefore, a token passing LESS's confidence, persistence, and JSD checks
usually did **not** have a settled hidden representation under this drift
definition.

## LESS Conclusion

Supported claim:

> **LESS token admission is not a reference-settlement certificate.**

Not supported by these experiments:

> **Published LESS is unsafe.**

Published LESS does not perform our hidden-row freeze. The result only rejects
using LESS admission as permission for that additional operation.

# What The Two Reports Mean Together

Both strategies identify useful forms of token or posterior stability, but
neither signal is sufficient for deciding that an old hidden representation can
be reused indefinitely.

| Strategy signal | What it establishes | What it does not establish |
| --- | --- | --- |
| SureLock-style adjacent posterior stability | The output distribution stopped changing across the checked steps | The hidden reference will remain safe for all future steps |
| LESS confidence + persistence + JSD | A token is a strong candidate for identity commitment | Its hidden/KV representation has finished adapting to context |

The strongest combined conclusion is:

> **Token commitment and representation freezing are different decisions and
> require different safety tests.**

The experiments provide clear evidence of output sensitivity and answer/program
changes. They do not yet provide clear evidence of systematic accuracy harm.

## Do Not Rank LESS And SureLock From These Percentages

LESS has higher raw change rates than the SureLock-style rule in these two
regimes, but this is not a fair strategy ranking.

- LESS freezes much earlier on average.
- The methods use different baseline samplers.
- The interventions do not provide equal compute savings.
- The selected token sets differ.

A fair comparison must match the amount of row computation avoided and should
also compare both rules on common eligible tokens.

## Recommended Next Test

The next strategy should be **Polestar-Cache**, because it directly uses
representation or attention drift to decide when stale cache entries need to be
refreshed. This is a stronger test than another token-stability rule.

Compare Polestar-Cache, LESS-derived freezing, SureLock-style freezing, and a
confidence-only rule at matched realized row-update savings. If Polestar reduces
harm at the same savings, representation-aware refreshing is the right
direction. If it does not, the next gate should combine drift with attention
exposure or directly predict causal freeze risk.

### Initial Polestar Result

We implemented the smallest defensible Polestar-Cache-inspired proxy before a
full run. It measured per-head attention KL, selected the strongest online
refresh event in each of four Level-5/gen256 examples, and compared keeping the
old block-entry row with refreshing to the current row.

All eight replays were valid and all target tokens stayed unchanged. Neither
the stale nor refreshed arm changed any downstream token (`0/4` versus `0/4`),
so the paired refresh benefit was zero. This validates the instrumentation but
does not justify scaling this proxy.

It is not a full Polestar result. Full Polestar uses centroid clustering,
repeated sparse per-layer KV refreshes, suffix updates, quantization, and a
custom optimized cache runtime that are absent here. See
`docs/polestar_cache_proxy_smoke_report_2026-09-07.md`.

## Official Elastic-Cache Result

We next tested the official Elastic-Cache implementation against matched full
recomputation at generation length 256.

On 54 Level-5 MATH examples, every token sequence changed and 43 mathematical
answers changed. Both arms still solved 5/54 problems, but two correct answers
became wrong and two wrong answers became correct.

On 70 HumanEval tasks, 46 token sequences and 45 programs changed. Baseline
accuracy was 41/70 and Elastic accuracy was 40/70, with seven correct-to-wrong
and six wrong-to-correct transitions.

Elastic ran 1.31x faster on Level-5 and 1.49x faster on HumanEval in this setup.
Its layer-recompute counter was much smaller than the measured runtime change,
showing why algorithmic compute reduction and real speed must be evaluated
separately.

The result supports output and per-example sensitivity, but not directional
aggregate accuracy harm. Full details are in
`docs/elastic_cache_official_report_2026-09-07.md`.

## Important Limitations

- The interventions clamp hidden rows at layers 16, 24, and 32. They are not a
  complete production K/V-cache implementation.
- Each run used one deterministic decode and one candidate-sampling seed.
- Level-5 uses the held-out 54-example partition of 134 records.
- HumanEval uses 70 of 164 tasks.
- Accuracy changes are rare, so the accuracy conclusions remain underpowered.
- The SureLock test is SureLock-style rather than a complete reproduction of
  the method's systems implementation.
- The LESS test uses block length 32; some headline LESS evaluations use block
  length 64.

## Detailed Reports And Artifacts

- SureLock technical report:
  `docs/surelock_premise_expanded_report_2026-09-06.md`
- LESS technical report: `docs/less_premise_report_2026-09-06.md`
- SureLock aggregate results: `results/surelock_premise_suite_20260906/`
- LESS/SureLock comparison:
  `results/less_vs_surelock_premise_suite_20260906/`
