# Published-baseline comparison results

Status: SureLock matched-policy runs are complete; PhaseLock controller-cost
and TraceLock-LLaDA transfer runs are in progress. This document is updated
only from completed GPU artifacts. Dense decoding is retained as an audit
anchor; the requested headline baselines are SureLock and TraceLock.

## Reproducibility status

- **SureLock:** paper-spec Algorithm-1 reimplementation because the project
  page does not release code. The matched run uses the shared fused row
  executor, so it tests the SureLock locking policy under the same optimized
  executor as PhaseLock rather than reproducing the paper's PyTorch runtime.
- **TraceLock:** the official repository at
  `bc4134c6c6716e9d92719fc582b6770616030840` is not executable because
  `tracelock.models` is absent. It also releases only the Dream projection,
  not the LLaDA projection/checkpoint. The matched baseline is therefore
  labeled **TraceLock-LLaDA transfer adaptation**.

## Matched setup

Shared setup: LLaDA-8B-Instruct, bf16, temperature 0, generation length 128,
block/window 32, batch size 4, one warmup, four rotating-order repetitions,
and paired batch bootstrap timing.

The SureLock policy uses the paper's raw-posterior KL gate, 20th-percentile
confidence cutoff, and both reported epsilon values (`5e-4`, `5e-3`). Once a
token locks, the shared exact fused-row executor freezes its reference and
removes its row. PhaseLock uses the previously frozen `safe_c095_s2` semantic
plan, `mli_ultra` representation plan, and workload-aware compute gate. No
threshold was fitted on these 60 evaluation examples.

## Three-dataset aggregate (60 examples)

This is the completed first-pass table. The SureLock columns are retained.
The PhaseLock column is now classified as a **single-example-plan transfer
diagnostic**, not the final PhaseLock comparator: a subsequent audit found
20/20 GSM8K plan signatures changed when the frozen controller was evaluated
at batch size four. A corrected exact-batch A1 -> A2 -> A3 run is queued and
will replace that column in the headline table.

| Method | Accuracy | Exact vs dense | Answer changed | Dense correctness regressions | Mean NFE | Runtime gain vs dense |
|---|---:|---:|---:|---:|---:|---:|
| Dense audit anchor | 36.67% | 100% | 0% | 0% | 128.00 | -- |
| SureLock, eps 5e-4 | 33.33% | 3.33% | 63.33% | 10.00% (6/60) | 128.00 | +51.07% [47.53, 53.94] |
| SureLock, eps 5e-3 | 35.00% | 3.33% | 65.00% | 8.33% (5/60) | 128.00 | +52.53% [48.93, 55.63] |
| PhaseLock, frozen-plan replay | 38.33% | 61.67% | 20.00% | 0% (0/60) | 126.07 | +1.63% [1.05, 2.10] |

PhaseLock-minus-SureLock accuracy is `+5.0 pp [-5.0, +15.0]` at `5e-4`
and `+3.33 pp [-6.67, +13.33]` at `5e-3`. Both confidence intervals cross
zero, so this pilot does **not** prove an accuracy advantage. It does establish
a strong behavioral difference on these slices: PhaseLock has zero observed
correctness regressions and 20% answer change, whereas SureLock has 5--6
regressions and 63--65% answer change. Conversely, SureLock is much more
aggressive and its executor is about 51--53% faster than dense, while the
conservative PhaseLock replay saves only 1.63%.

The raw PhaseLock timing excludes online Phase-1/Phase-2 plan construction;
SureLock's timing includes its online KL controller. Therefore the percentages
above are useful executor-policy measurements, but are not yet a fair
end-to-end speed ranking between the two methods. Dedicated online controller
measurements are running.

## GSM8K 820--839

| Method | Accuracy | Exact vs dense | Answer changed | Dense correctness regressions | Mean NFE | Runtime gain vs dense |
|---|---:|---:|---:|---:|---:|---:|
| Dense audit anchor | 60% | 100% | 0% | 0% | 128 | -- |
| SureLock, eps 5e-4 | 55% | 0% | 35% | 10% (2/20) | 128 | +42.22% [39.17, 44.84] |
| SureLock, eps 5e-3 | 55% | 0% | 40% | 10% (2/20) | 128 | +43.90% [40.68, 46.83] |
| PhaseLock, frozen-plan replay | 65% | 60% | 15% | 0% (0/20) | 126 | +1.89% [0.25, 3.25] |

Direct PhaseLock-minus-SureLock accuracy is +10 percentage points for both
thresholds, but the paired bootstrap interval is `[-10, +30]` points. The
slice is too small to claim accuracy superiority. The directional result is
that SureLock is much more aggressive and faster, while PhaseLock is more
conservative and produced fewer answer changes and no correctness regression
on this slice.

The direct raw latency ratio is not a final end-to-end PhaseLock claim:
SureLock pays its online KL controller cost, while the PhaseLock row executor
replays saved Phase-1/Phase-2 plans. Dedicated controller-overhead runs are
queued and will be incorporated before the final speed table.

## MATH500 450--469

| Method | Accuracy | Exact vs dense | Answer changed | Dense correctness regressions | Mean NFE | Runtime gain vs dense |
|---|---:|---:|---:|---:|---:|---:|
| Dense audit anchor | 45% | 100% | 0% | 0% | 128 | -- |
| SureLock, eps 5e-4 | 40% | 5% | 60% | 15% (3/20) | 128 | +46.42% [41.56, 49.66] |
| SureLock, eps 5e-3 | 45% | 5% | 60% | 10% (2/20) | 128 | +48.40% [43.67, 51.81] |
| PhaseLock, frozen-plan replay | 45% | 60% | 10% | 0% (0/20) | 127 | +1.00% [0.56, 1.63] |

## HumanEval 120--139

| Method | Pass rate | Exact vs dense | Answer changed | Dense correctness regressions | Mean NFE | Runtime gain vs dense |
|---|---:|---:|---:|---:|---:|---:|
| Dense audit anchor | 5% | 100% | 0% | 0% | 128.0 | -- |
| SureLock, eps 5e-4 | 5% | 5% | 95% | 5% (1/20) | 128.0 | +58.73% [53.11, 62.11] |
| SureLock, eps 5e-3 | 5% | 5% | 95% | 5% (1/20) | 128.0 | +59.76% [53.47, 63.63] |
| PhaseLock, frozen-plan replay | 5% | 65% | 35% | 0% (0/20) | 125.2 | +1.88% [1.03, 2.32] |

HumanEval has only one dense-correct example in this slice, so equal aggregate
pass rate is not evidence of output equivalence: each SureLock setting loses
one dense-correct solution and gains a different solution while changing 19
of 20 answers. PhaseLock changes 7 of 20 answers but preserves the one
dense-correct solution.

## PhaseLock batch-scope audit and corrected smoke

Re-running the frozen Phase-1/Phase-2 controllers at serving batch size four
produced plan-signature mismatches for 20/20 GSM8K and 20/20 MATH500 examples
relative to the old single-example plans. The old PhaseLock column above is
therefore retained only as a transfer diagnostic.

The corrected four-example GSM8K smoke generated plans on the exact batch and
forced the saved A1 token identities during A2/A3 replay:

| Transition | Exact output | Answer/correctness changes | Timing |
|---|---:|---:|---:|
| A0 -> A1 semantic commitment | 4/4 | 0 | -- |
| A1 -> A2 representation freeze | 4/4 | 0 | A2 is 0.32% slower than A0 |
| A2 -> A3 row removal | 4/4 | 0 | A3 is 2.09% faster than A2 |
| A0 -> A3 total executor | 4/4 | 0 | A3 is 1.78% faster than A0 |

The separately measured Phase-2 controller overhead is 1.29% on GSM8K, 1.39%
on MATH500, and 1.25% on HumanEval. The dataset-stratified pooled overhead is
1.30% [1.15%, 1.51%]. The planner leaves A1 tokens unchanged, but all 60 old
single-example plan signatures differ under batch-four serving. Subtracting
1.29% from the 1.78% GSM8K smoke executor gain gives only a rough +0.49%
margin; it is not an exact end-to-end measurement and a single batch has no
useful timing confidence interval. Full corrected runs are queued.

The corrected TraceLock batch-four plumbing smoke also completed, but its
32-trajectory controller committed exactly one token per sample-step and used
all 128 steps. Its roughly 2% timing difference is treated as noise/a smoke
check, not a TraceLock result. The scientific comparison uses the queued
8,000-trace controller.

## Pending before the final baseline table

- corrected batch-matched PhaseLock A1 -> A2 -> A3 results on all three
  datasets;
- online PhaseLock controller overhead on all three datasets;
- the 8,000-trace TraceLock-LLaDA transfer training run, using a preregistered
  equal mix of GSM8K, Natural Questions, Alpaca, and KodCode;
- matched TraceLock comparisons on all three datasets;
- a direct PhaseLock-versus-TraceLock pooled comparison.

GPU chain at the time of this update: the first matched TraceLock/PhaseLock
smoke `193173` failed before Python because the submission exported
`PHASELOCK_PLAN_DIR` while the launcher requires `PLAN_DIR`; it contributes no
result. Corrected replacement `193609` passed: PhaseLock A1/A3 both matched
dense on 4/4 examples, while the small TraceLock smoke controller preserved
normalized answers but still used all 128 steps. Corrected PhaseLock full jobs
`193185`--`193187` are now running/queued, followed by TraceLock trace
generation/training/evaluation `193064`--`193084`.

## Trace-generation recovery

The final KodCode trace job `193067` reached 1,037 complete examples and was
cancelled at its four-hour limit. The first three 2,000-example source shards
completed normally. KodCode also rejects many prompts above the preregistered
512-token input cap, so the 1,037th complete example occurred at shuffled raw
position 1,831; resuming from raw offset 1,037 would have duplicated and
omitted examples.

The generator now has a fail-safe resume mode that inventories existing step
files, accepts a sample only when all 38 sampled steps are present, regenerates
an interrupted partial sample, skips complete sample IDs, and stops at an
explicit total-completed target. The real interrupted directory contained
exactly 1,037 complete KodCode examples and zero partial examples. Recovery is
split into jobs `194922` (target 1,500) and `194923` (target 2,000), avoiding a
second four-hour timeout. Existing training job `193081` now depends on
`194923`; evaluations `193082`--`193084` retain their original dependencies.
No model, trace, controller, dataset, seed, projection, threshold, or evaluation
setting changed.

The original controller job `193081` then failed before its first optimizer
step because its comma-separated `EXPECTED_SOURCE_COUNTS` value was parsed by
Slurm as multiple export fields, leaving only `gsm8k=2000`. The observed pool
was already correct. An independent audit verified four manifest files, 8,000
unique examples, exactly 2,000 examples per source, and 304,000 step files
(271,776 train and 32,224 validation). Replacement training job `195659`
inherits the complete expected-count string through the submitted environment;
evaluation job `193082` was reconnected to it. The replacement passed the same
audit in-job and began optimization normally.
