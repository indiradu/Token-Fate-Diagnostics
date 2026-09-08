# SureLock Premise Expanded Report - 2026-09-06

## Executive Result

The expanded experiments support one narrow claim and reject a stronger one:

- **Supported:** a SureLock-style adjacent-posterior stability gate does not
  certify that freezing the admitted token's reference representation will
  preserve the decoded output.
- **Not supported:** freezing at this gate measurably reduces task accuracy.
  Output and extracted-answer changes occur, but correctness transitions are
  rare, balanced or favorable in these samples, and every accuracy-delta
  interval includes zero.

The strongest result remains Level-5 MATH at gen256: freezing a SureLock-
admitted row changes downstream tokens in 23/108 interventions (21.3%, 95% CI
[13.0%, 30.6%]) and changes the mathematical answer in 9/108 (8.3%, [3.7%,
13.9%]), while original and modified accuracy are both 7.4%.

## Experimental Object

For each sampled, semantic-safe token admitted by the SureLock-style
adjacent-step KL gate at `tau=0`:

- `YC`: normal LLaDA replay. Token identity is fixed after transfer, while its
  hidden row is recomputed as the surrounding sequence develops.
- `YF`: the same replay, but the token's hidden row at layers 16, 24, and 32 is
  clamped to the row cached at the SureLock-style lock step.

The target token remained unchanged and replay-valid in every reported
intervention. The causal endpoint is therefore downstream divergence caused by
freezing the reference row, not by changing the selected token.

This is a hidden-row freeze proxy. It is not yet an implementation of actual
cached K/V tensors or row removal.

## Evaluation Repair

The previous evaluator reduced a MATH answer to its final number. It could, for
example, display a boxed `193/36` answer as `36`. The replacement evaluator:

- preserves nested boxed expressions;
- checks scalar expressions through a whitelisted AST converted directly to
  SymPy objects, without `eval` or SymPy string evaluation;
- handles fractions, radicals, variables, infinities, intervals, unions,
  tuples, sets, matrices, `plus/minus`, textual labels, and base notation;
- parses all 500/500 MATH-500 gold answers;
- reports literal extracted-answer change separately from mathematical-value
  change.

The earlier `36 -> 5` example is corrected to `193/36 -> 5`; the gold answer is
`13/6`, so it remains a wrong-to-wrong change.

HumanEval programs run under Bubblewrap with a read-only filesystem, private
temporary directory, network namespace, CPU/memory/file-size limits, and a
wall timeout. All 164 canonical HumanEval solutions pass this scorer. Sandbox
setup failures are reported separately and excluded from accuracy denominators;
candidate-program timeouts remain ordinary task failures.

## Main Suite

Intervals are 10,000-replicate percentile intervals that resample example-id
clusters. Original and modified accuracy use the same intervention-weighted
denominator.

| Regime | Ex. | Int. | Downstream change | Answer/program change | Original acc. | Modified acc. | Acc. delta | Correct -> wrong | Wrong -> correct |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Levels 1-4, gen64 | 54 | 107 | 5.6% [0.9, 12.0] | 2.8% [0.0, 7.5] | 46.7% [33.6, 59.8] | 46.7% [33.6, 59.8] | 0.0 pp | 0.0% | 0.0% |
| Levels 1-4, gen256 | 54 | 108 | 14.8% [7.4, 23.1] | 1.9% [0.0, 4.6] | 46.3% [33.3, 59.3] | 46.3% [33.3, 59.3] | 0.0 pp [-2.8, 2.8] | 0.9% | 0.9% |
| Level 5, gen64 | 54 | 108 | 7.4% [1.9, 14.8] | 1.9% [0.0, 4.6] | 9.3% [1.9, 16.7] | 10.2% [2.8, 18.5] | +0.9 pp [0.0, 2.8] | 0.0% | 0.9% |
| Level 5, gen256 | 54 | 108 | 21.3% [13.0, 30.6] | 8.3% [3.7, 13.9] | 7.4% [1.9, 14.8] | 7.4% [1.9, 14.8] | 0.0 pp | 0.0% | 0.0% |
| HumanEval, gen256 | 70 | 140 | 10.0% [4.3, 17.1] | 10.0% [4.3, 17.1] | 54.3% [42.9, 65.7] | 55.0% [43.6, 66.4] | +0.7 pp [0.0, 2.1] | 0.0% | 0.7% |
| AIME 2024, gen256 | 10 | 20 | 10.0% [0.0, 25.0] | 5.0% [0.0, 15.0] | 0.0% | 0.0% | 0.0 pp | 0.0% | 0.0% |

`Answer/program change` means mathematical-value change for MATH/AIME and
normalized source-program change for HumanEval.

## Difficulty x Horizon

The 2x2 uses fixed examples within each difficulty group. Level 5 contains 54
problems from the existing Level-5 file. The comparison group is a fixed,
stratified sample of 54 Level-1--4 problems. Contrasts pair examples across
gen64 and gen256 within each group; token positions need not match across
generation regimes.

| Endpoint | Levels 1-4: gen256 - gen64 | Level 5: gen256 - gen64 | Interaction |
| --- | ---: | ---: | ---: |
| Downstream-change rate | +9.3 pp [-0.9, 19.4] | +13.9 pp [2.8, 25.0] | +4.6 pp [-10.2, 19.4] |
| Mean changed-token count | +6.16 [2.47, 10.68] | +11.88 [5.51, 19.08] | +5.72 [-2.05, 13.93] |
| Answer-value change | -0.9 pp [-6.5, 3.7] | +6.5 pp [0.9, 13.0] | +7.4 pp [0.0, 15.7] |

The defensible interpretation is:

1. Longer generation clearly increases the *magnitude* of downstream
   disruption in both difficulty groups.
2. Level-5 answer changes rise at gen256, while Levels 1-4 do not in this
   sample.
3. The binary downstream-change interaction is unresolved, and the answer-
   change interaction touches zero. We therefore cannot yet claim that problem
   difficulty itself amplifies the horizon effect.

The large baseline-accuracy gap confirms that the difficulty split is real:
Levels 1-4 score about 46%, while Level 5 scores about 7-9% under these prompts.

## Cross-Domain Result

The 70-example HumanEval run establishes that the phenomenon is not math-only.
Fourteen of 140 frozen-reference interventions changed the generated program,
with a mean 2.27 downstream tokens changed per intervention [0.77, 4.14].

Thirteen changed programs retained their prior correctness status. One changed
from failing to passing the official tests; none changed from passing to
failing. This is evidence for reference sensitivity, not evidence that the
intervention systematically harms code correctness.

## Harder-Math Probe

The first ten AIME 2024 problems produced two downstream-divergent interventions
and one answer change among 20 interventions. Original and modified accuracy
were both zero. The probe therefore says that LLaDA can generate trajectories
on AIME where reference freezing changes output, but it cannot estimate an
accuracy effect. Scaling AIME is not currently a good use of compute without a
stronger math backbone or prompting protocol.

## Conclusions

1. SureLock-style posterior stability is informative but not sufficient for
   reference-freeze safety. Residual downstream changes replicate across MATH
   and executable code.
2. Generation horizon is an important stress variable. It raises changed-token
   magnitude in both easy/moderate and Level-5 MATH.
3. Difficulty may matter specifically for answer-level divergence at gen256,
   but the interaction evidence is not decisive.
4. No experiment here establishes negative accuracy impact. Answer changes and
   correctness changes are different estimands; most changed answers are
   wrong-to-wrong, and the few correctness transitions are balanced or
   favorable.
5. The next controller should predict or bound `YF != YC`, not merely posterior
   stability. A subsequent systems experiment must still test real K/V caching
   and row removal before making speed-quality claims.

## Artifact Map

- Full suite: `results/surelock_premise_suite_20260906/`
- 2x2 contrasts: `results/surelock_premise_2x2_analysis_20260906/`
- Level-5/gen256 regrade:
  `results/surelock_premise_reanalysis_l5g256_20260906/`
- Level-5/gen64 regrade:
  `results/surelock_premise_reanalysis_2x2_l5_g64_20260906/`
- Levels-1--4/gen256 regrade:
  `results/surelock_premise_reanalysis_2x2_lt5_g256_20260906/`
- Levels-1--4/gen64 regrade:
  `results/surelock_premise_reanalysis_2x2_lt5_g64_20260906/`
- HumanEval regrade:
  `results/surelock_premise_reanalysis_humaneval_g256_70x2_20260906/`
- AIME regrade:
  `results/surelock_premise_reanalysis_aime2024_g256_10x2_20260906/`

## Remaining Risks

- The intervention clamps hidden rows only at layers 16, 24, and 32. It is not
  identical to a production cache implementation.
- Each cell uses one deterministic decode and one candidate-sampling seed.
- Correctness transitions remain rare, so accuracy-harm claims are underpowered.
- HumanEval covers the first 70 of 164 tasks; AIME covers only 10 of 30.
- The custom MATH grader parses every gold answer in MATH-500, but a standard
  external symbolic grader would still be valuable as an independent check.
