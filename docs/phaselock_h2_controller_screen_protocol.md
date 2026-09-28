# PhaseLock H2 controller and representation screen

Status: implementation/development screen, not a confirmed new method.

## Objective and measurements

Improve task accuracy at H2-like speed, and ultimately match or improve BOTH
SureLock and TraceLock accuracy with lower online latency. Dense agreement is
diagnostic only. No criterion demands zero individual regressions if improvements
compensate in task accuracy. Execution-only A2/A3 equivalence remains a separate
correctness invariant, not the quality objective.

Use LLaDA-8B-Instruct BF16, temperature 0, 128 generated positions, maximum 128
updates, width 32, batches of four. Same examples, tokenizer and prompts, scoring,
GPU and caller-level synchronized timing for every arm. Include prompt preparation,
live decisions, row packing, cleanup and output decoding, but exclude model loading
and scoring. No cached-plan replay or estimated controller-overhead subtraction.
Record generation-loop time separately. Rotate method order by batch and repeat;
use one warmup and three measurements. Repeats are not additional accuracy samples.
This first screen is a latency screen, not a FLOPs measurement or kernel profiler.

SureLock is the existing paper-spec transfer implementation (KL<=0.005, top-20%
candidate confidence quantile, ordinary token schedule and immediate row freezing).
TraceLock is the existing trained LLaDA adaptation, not an official checkpoint.
Its sliding width-32 crop differs from the fixed width-32 blocks used by the other
methods. Preserve this defining behavior and disclose it; matching the width does
not imply identical commitment schedules. Source, controller and projection hashes,
prompt hashes, package versions, GPU and output tokens are saved with every run.

## Controlled candidate pool

P1 H2: compute full-posterior log probabilities and KL once per update. Among masked
tokens in the current block, select uncertainty below the candidate-relative 20th
percentile AND KL<=0.005. If none survive, commit the most confident allowed token.
P2 H2: apply the same Sure rule to active unmasked positions AFTER commitment,
including prompt/padding and newly committed positions. Freeze all-layer input/output
and K/V snapshots. P3 immediately removes frozen query/FFN rows using the existing
exact bucketed runtime (dense attention over all K/V references).

Predeclared variants are in `configs/phaselock_h2_controller_screen_v1.json`:

- Original H2 control (unchanged decisions).
- Live P2: no reference freezing or row removal; isolate P1 quality and NFE.
- Generated-only P2: remove prompt eligibility; isolate a confound in old H2/H3.
- Delay 1 or 2: preserve prompt eligibility; freeze generated references only after
  that many subsequent forwards have seen their committed identities.
- Multi-layer drift: preserve prompt eligibility; require max cosine drift over
  layers 8/16/24 <=0.01, one passing update, and two post-commit observations.
  Only still-live rows qualify. Frozen zero-drift snapshots are NOT safety evidence.
- Refresh 16: delay-1 rule plus release all references every 16 actual evaluations;
  semantic identities stay committed. Clear posterior/drift history, perform a live
  forward, then reacquire references. All rebuilding and dense refresh costs count.
- P1 confidence floor 0.90, and independently a cap of ceil(10% of remaining allowed
  tokens) per update. These guard optional commits; the one-token fallback remains.
- One predeclared crossing: confidence-0.90 P1 + delay-1 P2.

Prompt reuse can still cause errors: retaining it is a controlled hypothesis, not a
safety guarantee. A representation veto cannot undo an already incorrect semantic
commitment. Refresh repairs references only, never token identities. Further pool
combinations must be chosen on development/calibration results, not on confirmation.

## Data, scoring, calibration and claim limits

Smoke: GSM8K 800:803, one measurement. Development: GSM8K 800:807, MATH500 430:437,
HumanEval 100:107. These were already used in earlier development; they are explicitly
NOT untouched test data. Screening runs do not establish generalization.

The legacy MATH500 scorer extracts numbers, which is not a validated symbolic
equivalence metric. Save full outputs and label its numbers screening proxies.
Before selection for final accuracy claims, validate/re-score MATH500 with a suitable
symbolic grader. HumanEval 128-token results may be truncation-limited; follow with
matched 256/256 settings before concluding quality superiority.

The re-scoring pass is implemented separately using pinned Math-Verify 0.9.0,
latex2sympy2_extended 1.11.0 and antlr4 4.13.2, in an isolated grader environment.
It parses gold LaTeX and predictions (boxed answers have priority), checks symbolic
equivalence, and records parsed expressions and failed prediction extractions.
Unparseable gold stops the pass for inspection. Original numeric scores and raw
records are preserved. Six hand-constructed positive/negative tests are required;
this validates basic integration, not every mathematical answer type. See
https://github.com/huggingface/Math-Verify for the grader's documented API.

Report per-dataset correctness and paired changes, total median batch latency ratio
versus both baselines, NFE, broad H2 controller/setup time, peak memory, reference
and prompt-lock counts and repeat determinism. Bootstrap examples for exploratory
quality differences and batches for latency; do not count repeats as independent
examples. Tiny pilot intervals are descriptive, not noninferiority certification.

Primary final target is no accuracy loss vs either baseline (0 percentage-point
margin), with a positive speed gain vs BOTH; do not quietly widen the tolerance to
select a winner. Development Pareto candidates may advance without satisfying this
claim. A separate calibration split, sample-size plan, policy freeze and audited
unseen confirmation manifest are required before an outperformance claim. Earlier
820:839/450:469/120:139 slices have been repeatedly inspected. Even later reserved
slices may have been used by other runs and need a provenance audit.

For learned controllers, split by example before calibration; train against answer
correctness loss and whole proposed-group interventions, not only sequence equality
or independent token labels. Calibration quality (Brier/reliability/risk–coverage)
and the adaptive feedback of deployment require separate evaluation. Do not call a
confidence floor a calibrated probability of commitment safety.

## Cost accounting and next stages

H2 heuristics have no learned-controller training, but development/calibration costs
are real. TraceLock has trace-generation + controller-training costs. Direct safety
adds counterfactual-labeling and fitting, and any shared Trace feature training.
Unknown historical costs remain null, never zero. Report inference-only results and
amortized cost separately: training/label/calibration GPU-seconds + N * measured
serving seconds at the stated batch/workload, allocating shared costs explicitly.

Next: inspect the live screen, validate scoring, establish independent calibration
data, investigate controller errors/overhead, cross the best semantic and reference
policies, and run longer-output confirmation against both baselines. No automatic
confirmation or claim of robustness is authorized by a favorable tiny screen alone.
Keep existing experiment artifacts. This completed development revision is part
of the major-experiment handoff; unfinished follow-up campaigns remain excluded.
