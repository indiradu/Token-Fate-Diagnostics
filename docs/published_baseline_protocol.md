# Published baseline protocol for PhaseLock

## Comparison question

The headline comparison is PhaseLock versus SureLock and TraceLock. A normal
dense decode is retained only as a shared internal anchor for measuring output
changes, correctness changes, NFE, and runtime gains. It is not presented as
the main competing method.

## Reproduction status

### SureLock

SureLock's project page currently says that code is coming soon. We therefore
use a paper-spec reimplementation of Algorithm 1, not an official-code claim.
The implementation uses:

- the raw posterior before temperature scaling;
- local `KL(p_t || p_{t-1})` with first-step abstention;
- the top 20 percent confidence gate implemented as the 20th percentile of
  uncertainty among active, unmasked candidates after ordinary mask update;
- the paper's `5e-4` and `5e-3` KL operating points;
- immediate all-layer block-input and K/V freezing after a lock;
- exact fused active-row execution on every subsequent step, without a
  PhaseLock profitability or representation-safety gate.

This preserves the baseline's defining coupling: SureLock uses one posterior
criterion to trigger representation caching and row removal directly.

The matched benchmark uses PhaseLock's `128/32` generation/block schedule so
the methods see the same backbone, prompts, number of generated positions, and
ordinary token-transfer schedule. This is a transfer evaluation of the
published SureLock controller. It is not a reproduction of SureLock's main
paper table, whose default is a fully parallel block (`block length =
generation length`) on WikiText-103 and MT-Bench. A separate source-setting
check is required before claiming the paper's reported numbers were
reproduced.

### TraceLock

The official public repository was cloned at commit
`bc4134c6c6716e9d92719fc582b6770616030840`. It cannot execute as published:

- the release explicitly provides only the Dream path, not the LLaDA path used
  for the matched PhaseLock comparison;
- its training and generation files import `tracelock.models.tracelock`, but
  the entire `tracelock/models` directory is absent from the commit, all
  branches, tags, and repository history;
- only the Dream activation projection is released; the paper's LLaDA
  projection and trained LLaDA TraceLock controller are not released.

The attempted import fails with
`ModuleNotFoundError: No module named 'tracelock.models'`. This is recorded as
an upstream reproducibility failure, not repaired code presented as official.

The release README describes a newer Dream-only reproduction recipe (7,000
traces, five encoder layers, 36,000 updates, a 0.99 deployment threshold, and
a four-token threshold-accept cap). Those settings do not match the paper's
LLaDA method (`8,000` traces, three encoder layers, about `10,000` updates,
and threshold `0.95`). The matched LLaDA benchmark follows the paper, not the
Dream-only release recipe; the two specifications are never mixed.

For a matched LLaDA transfer comparison, we implement the paper-specified
controller:

- last three hidden snapshots and two adjacent late-layer differences;
- a 3-layer Transformer encoder, 8 heads, width 384, FFN width 768, dropout
  0.1, and positional encoding;
- a learned threshold token and token-minus-threshold decision margin;
- weighted BCE with mistake weight 4 and threshold regularization;
- random visible-generation-prefix crop with ratio sampled uniformly from
  `[0.25, 1]`;
- local soft-crop deployment and probability threshold 0.95, with a top-score
  fallback when no active token passes.

The 8,000-trace training pool is preregistered as 2,000 prompts from each task
source named by the paper: GSM8K, Natural Questions Slim Short Answer,
Alpaca-Cleaned, and KodCode-HumanEval-like. Evaluation indices are disjoint
from the GSM8K training split. The equal four-way mix is explicit because the
paper gives the total trace count and source families but not their mixture
weights.

Because the LLaDA projection checkpoint is unavailable, the transfer uses a
seeded, frozen Gaussian projection shared across the three late hidden layers.
Every result must be labeled **TraceLock-LLaDA transfer adaptation**, not an
official TraceLock reproduction.

The paper states that malformed, very short, repetitive open-ended traces and
invalid code traces are filtered, but it does not publish the exact filters or
thresholds and the executable filtering implementation is unavailable because
of the missing source module. The transfer run therefore records this as a
second deviation; it must not be described as an exact checkpoint
reproduction.

## Matched evaluation

All methods use `GSAI-ML/LLaDA-8B-Instruct`, bfloat16, temperature zero, one
RTX 5000 Ada GPU, generation length 128, maximum 128 denoising steps, and
window/block length 32. The evaluation slices are the untouched PhaseLock
confirmation sets:

- GSM8K test indices 820--839;
- MATH500 test indices 450--469;
- HumanEval test indices 120--139.

`128/32` is a published LLaDA TraceLock evaluation regime. GSM8K and
HumanEval are in-domain task families for TraceLock; MATH500 is an explicit
out-of-domain transfer check. All three are transfer evaluations for
SureLock relative to its reported WikiText-103/MT-Bench main experiments.

The default serving comparison uses batch size 4. Each batch receives one
warmup and four timed repetitions with rotating method order and CUDA
synchronization. Task scoring is outside the timed interval. Runtime gains use
paired batch medians and a 10,000-sample batch bootstrap confidence interval.

For each method, report:

- task accuracy/pass@1;
- exact-token and task-answer change rates against the shared dense anchor;
- correctness regressions and improvements against the anchor;
- mean executed denoising steps/NFE;
- controller overhead as a fraction of wall time where applicable;
- physically removed row-forwards and wall-clock gain where applicable.

The primary table must include direct PhaseLock-minus-SureLock and
PhaseLock-minus-TraceLock quality deltas. Runtime is split into two auditable
views: (1) executor-only replay, where every method replays a saved lock plan,
and (2) online end-to-end time, where every method pays its controller cost.
An online SureLock number must not be compared as a final headline against a
PhaseLock fixed-plan replay number. A method is not called faster unless the
lower endpoint of its 95 percent paired bootstrap interval is above zero.

The currently scheduled online-controller runs are diagnostic for output,
task quality, NFE, and controller cost. PhaseLock's plan-replay latency is not
a final end-to-end number. A separate online PhaseLock controller run measures
its Phase-2 planning overhead; the final speed table reports the replay and
controller costs in separate columns. No asymmetric raw ratio is promoted as
the main comparison, and any conservative combined estimate is labeled as an
estimate rather than an exact end-to-end measurement.

## Frozen-policy PhaseLock comparator

The comparator is not retuned on baseline outputs:

- Phase 1: frozen `safe_c095_s2` semantic controller;
- Phase 2: frozen `mli_ultra` multi-layer plus influence-aware representation
  controller;
- Phase 3: latency-calibrated exact fused global/bucketed execution with
  abstention (`min locked fraction=0.20`, `min remaining steps=8`, projected
  saved rows `1500`, predicted gain floor `0.03`).

The original confirmation plans were generated one example at a time. An
audit found that all 20 GSM8K plan signatures change when the same frozen
controller is run at the serving batch size of four, so transferring those
single-example plans into a batch-four executor is excluded from the final
headline comparison.

For the corrected comparison, the frozen controllers first observe A1 on the
exact evaluation batch. The resulting semantic positions **and token
identities**, and the Phase-2 reference decisions, are then replayed through
A2 and A3 on that same batch. We report A0-to-A1, A1-to-A2, and A2-to-A3
changes separately. This changes only plan scope, not any learned parameter,
threshold, cap, patience, layer, influence gate, or compute gate. No SureLock
or TraceLock output is used to alter PhaseLock.
