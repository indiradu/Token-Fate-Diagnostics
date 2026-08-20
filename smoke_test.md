# Settlement Fate Smoke Test

Date: 2026-08-20

This note records the first smoke tests for the core settlement-fate idea:
semantic commitment, representation settlement, and compute safety are not the
same property.

## Three Hypotheses

Use only three labels for the hypotheses:

- **A:** Semantic commitment does not imply representation settlement.
- **B:** Semantic commitment does not imply compute safety.
- **C:** Representation drift predicts compute unsafety beyond matched controls.

Here, "non-target change" is just an audit outcome: after clamping one committed
position, at least one other final output token changes. It is not a separate
hypothesis.

### A: Semantic commitment does not imply representation settlement

Once the normal confidence decoder commits a token, the token identity is fixed
by the decoding process. The question is whether that token's hidden
representation keeps changing afterward.

Smoke status: supported.

Evidence:

- `results/settlement_fate_second_gsm8k_5`
  - 5 real GSM8K examples.
  - 320 committed tokens.
  - 310 post-commit drift measurements.
  - Median max hidden drift: `0.4701`.
  - P90 max hidden drift: `0.9360`.
- `results/settlement_fate_second_math500_5`
  - 5 real MATH500 examples.
  - 320 committed tokens.
  - 310 post-commit drift measurements.
  - Median max hidden drift: `0.4608`.
  - P90 max hidden drift: `0.7917`.

Interpretation: even after token identity is committed, the row representation
continues to move substantially. This supports the distinction between
semantic settlement and representation settlement.

### B: Semantic commitment does not imply compute safety

The stronger question is whether freezing a committed token's representation
can change other output tokens. We tested this with a stale-row hidden-state
clamp: after commitment, replay the decode while clamping that token's hidden
row to its cached post-commit representation.

Smoke status: supported as a proxy phenomenon.

Evidence:

- `results/settlement_fate_second_gsm8k_5`
  - 10 clamp interventions.
  - No non-target output changes.
- `results/settlement_fate_second_math500_5`
  - 20 clamp interventions.
  - High-drift group: 10 candidates, non-target change rate `0.5`.
  - Matched low-drift group: 10 candidates, non-target change rate `0.4`.
  - Some interventions changed extracted answer content:
    - `105 -> 255`
    - `11 -> 10`
  - One target-unique case: `math500-test-101:190:4`, where the high-drift
    clamp changed 12 non-target positions and the matched low-drift control
    changed none.

Interpretation: harder MATH500 examples show that already committed tokens can
still be compute-sensitive under stale-row clamping. This supports the
semantic-settled versus compute-safe distinction.

### C: Representation drift predicts compute unsafety beyond controls

The method-level hypothesis is stronger: high post-commit representation drift
should identify committed tokens that are more unsafe to freeze than matched
low-drift tokens.

Smoke status: smoke-supported directionally, but not robustly established.

Evidence:

- Initial MATH500 smoke run: high-drift clamps changed non-target tokens more often than
  matched low-drift clamps:
  - high drift: `5/10`
  - matched low drift: `4/10`
  - pairwise high-minus-low change rate: `+0.1`
  - pairwise high-minus-low changed-token count: `+1.2`
- Follow-up MATH500 run, `results/settlement_fate_c_math500_10x3`:
  - 10 examples, 30 matched high/low pairs, 60 interventions.
  - High drift: non-target change rate `0.2333`, mean changed-token count
    `3.8333`, normalized-answer change rate `0.1333`.
  - Matched low drift: non-target change rate `0.1333`, mean changed-token
    count `1.0`, normalized-answer change rate `0.0333`.
  - Pairwise high-minus-low non-target change rate: `+0.1`.
  - Pairwise high-minus-low changed-token count: `+2.8333`.
  - Mean target-unique changed-token count: `2.8667`; mean shared changed-token
    count: `0.9667`.
  - High-only non-target-change pairs: `4`; low-only pairs: `1`.
  - High-only normalized-answer-change pairs: `3`; low-only pairs: `0`.
- Follow-up non-math smoke run,
  `results/settlement_fate_c_nonmath_10x3`:
  - 10 local non-math reasoning/classification examples, 30 matched high/low
    pairs, 60 interventions.
  - High drift: non-target change rate `0.4`, mean changed-token count
    `5.7333`, normalized-answer change rate `0.2667`.
  - Matched low drift: non-target change rate `0.1`, mean changed-token count
    `1.8333`, normalized-answer change rate `0.0333`.
  - Pairwise high-minus-low non-target change rate: `+0.3`.
  - Pairwise high-minus-low changed-token count: `+3.9`.
  - Mean target-unique changed-token count: `3.9667`; mean shared changed-token
    count: `1.7667`.
  - High-only non-target-change pairs: `10`; low-only pairs: `1`.
  - High-only normalized-answer-change pairs: `7`; low-only pairs: `0`.

Interpretation: the follow-up tests make C more plausible. High-drift committed
tokens produced more target-unique output changes than matched low-drift
controls on both MATH500 and a small local non-math slice. This is still a smoke
result: sample sizes are small, the non-math slice is synthetic/local, and the
intervention is still a stale-row hidden-state clamp rather than the final
SureLock-style cached-K/V mechanism.

## Concrete Examples

### A example: committed but still moving

In `results/settlement_fate_second_math500_5`, the normal decoder committed 320
tokens and produced 310 post-commit drift measurements. The median max hidden
drift after commitment was `0.4608`, with P90 `0.7917`. This is the cleanest
smoke evidence for A: the token identity is already committed, but the hidden
row is still changing substantially.

One concrete row is `math500-test-100:312:35`: the high-drift committed token
had max drift `1.0186` after commitment. That supports A, but by itself does
not prove B or C.

### B example: committed token still affects later output

In `math500-test-101:190:4`, clamping the high-drift committed position changed
12 non-target output positions. The matched low-drift control changed none.

Baseline ending:

```text
... the sum of the reciprocals of the products of each root and its conjugate
is simply the number of pairs of roots, which is 5. Thus, the value is \boxed{5}.
```

High-drift clamp ending:

```text
... the sum of the reciprocals of the products of each root and its conjugate
is equal to 1, and there are 5 such pairs. Thus, the value is \boxed{5}.
```

The final boxed answer stayed `5`, but the reasoning text changed. This is good
evidence for B as a compute-sensitivity proxy: the committed token was not inert
under the stale-row clamp.

### B but not C example: shared answer change

In `math500-test-104`, both a high-drift clamp and its matched low-drift control
changed the final answer from `11` to `10`.

Baseline:

```text
... this sum is $6 + 5 = \boxed{11}.$
```

Clamp output:

```text
... this sum is $5 + 5 = \boxed{10}.$
```

This supports B because freezing a committed token can alter the task answer.
It does not support C, because the matched low-drift control also changed the
answer. This example is generic clamp sensitivity, not drift-specific evidence.

### C example: target-unique answer change

In `results/settlement_fate_c_math500_10x3`, pair
`math500-test-107:129:32` is a clean C example. The high-drift target had max
drift `1.1199`; its matched low-drift control had max drift `0.3332`.

Baseline:

```text
... All of these are odd numbers greater than zero, so the intersection has
\boxed{9} elements.
```

High-drift clamp:

```text
... Of these, the first five are odd. Therefore, the intersection has
\boxed{5} elements.
```

Matched low-drift clamp:

```text
... All of these are odd numbers greater than zero, so the intersection has
\boxed{9} elements.
```

The high-drift clamp changed 12 non-target positions and changed the normalized
answer `9 -> 5`; the matched low-drift control changed neither.

### C non-math example

In `results/settlement_fate_c_nonmath_10x3`, pair
`nonmath-smoke-007:75:9` is a non-math smoke example. The task asks whether a
ticket marked `void` may enter. The high-drift target had max drift `6.6972`;
its matched low-drift control had max drift `0.3999`.

Baseline:

```text
The ticket marked 'void' may not enter.

\boxed{\text{May not enter}}
```

High-drift clamp:

```text
The ticket marked 'void' may not enter.

\boxed{\text{No entry}}
```

Matched low-drift clamp:

```text
The ticket marked 'void' may not enter.

\boxed{\text{May not enter}}
```

The high-drift clamp changed 5 non-target positions and changed the normalized
answer text; the matched low-drift control changed neither. This is a small
synthetic example, not a benchmark result, but it shows that the phenomenon is
not limited to MATH500-style math prompts.

### Negative example: easier GSM8K run

In `results/settlement_fate_second_gsm8k_5`, committed tokens still showed
substantial hidden drift, but none of the 10 clamp interventions changed
non-target outputs. That supports A without supporting B or C on this easier
sample.

## Current Bottom Line

We have smoke support for all three hypotheses:

1. **A:** Committed token identity can be fixed while the representation keeps
   moving.
2. **B:** Freezing a committed token's representation can change downstream
   outputs.
3. **C:** Representation drift reliably predicts which committed tokens are
   unsafe to freeze beyond matched controls.

C is the weakest of the three: the follow-up runs are directional and concrete,
but they are still too small to treat as robust proof.

## Caveats

- The intervention is a stale-row hidden-state clamp, not a SureLock-style
  cached-K/V implementation.
- Samples are still small: the strongest C follow-up uses 10 MATH500 examples
  and 10 local non-math smoke examples.
- The non-math slice is synthetic/local and uses the existing generic boxed
  answer prompt path; it is useful as a smoke check, not as a benchmark.
- Current matched controls are reasonable but imperfect; some pairs still have
  nontrivial position/time differences.
- Correctness-change is too coarse because wrong-to-different-wrong answer
  changes matter. Future runs should use normalized-answer change directly.

## Next Robustness Test

The next run should stress-test hypothesis C:

- Use more MATH500 examples and at least one real non-math benchmark.
- Increase the number and quality of matched controls.
- Report target-unique changed positions, shared changed positions, and
  normalized-answer changes.
- Compare multiple drift quantiles within the same example, not only top drift
  versus one matched low-drift control.
- Add bootstrap confidence intervals or a paired sign test before claiming C as
  established.
