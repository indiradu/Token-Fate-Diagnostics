# Project Description

Token-Fate Diagnostics is a Python research repository for studying premature
token commitment in diffusion language models. It treats token fate as the
question of whether an intermediate denoising token agrees with the final
decoded token, then evaluates trace features, hidden-state probes, and strict
counterfactual interventions on LLaDA-style confidence decoding.

The public repo contains source code under `src/regret_remasking/`, experiment
and analysis entrypoints under `scripts/`, lightweight tests under `tests/`,
protocol/result notes under `docs/`, and compact aggregate reported artifacts
under `results/`. Raw traces, hidden-state arrays, local model files, and
scheduler outputs are intentionally excluded.

The current published interpretation is that token fate is strongly predictable
from traces, but confidence explains much of the broad trace-regret signal. The
clearest positive held-out result is a frozen general trace predictor on
high-confidence GSM8K tokens, while strict confidence- and trajectory-matched
interventions do not establish a resolved causal enrichment beyond confidence.

The [PhaseLock locking strategy document](locking_strateggies.md) explains the
three staged policies, exact decision rules, replay semantics, and implementation
limitations, with source references pinned to the merged `main` revision.

Working state inspected on 2026-09-15: the checkout is
`pivot_efficient_compute` at `4d16746`. Recent work completed the local
Fast-dLLM v1 component evaluation on 54 Level-5 MATH examples and 70 HumanEval
tasks, then documented PhaseLock's locking policies. The
[Fast-dLLM report](docs/fast_dllm_official_report_2026-09-08.md) and its runner,
analyzer, and tests remain untracked; result artifacts are local and ignored.
The results index does not yet include this evaluation.

The staged PhaseLock controller described by the strategy document exists on
local `main` at `c43be08`, not in this checkout. Its documented limitations
remain: the prediction run-length counter is not carried forward, replay forces
saved token identities, and compute removal has no online profitability gate.
Final-output agreement under that replay cannot establish reference-freeze
safety. Calibrated causal-risk gates and their quality/runtime benefit remain
research goals.

The September 15 inspection found no newer substantive experiment artifacts.
Fresh checks passed all 10 Fast-dLLM helper tests and parsed all 50 Python
source, script, and test files. Both saved Fast-dLLM datasets contain the
expected 620 complete outputs; all eight pairwise output-change and
correctness-transition summaries per dataset agree with their saved rows.
These checks did not rerun model generation or task scoring.

The results index also retains a superseded Polestar smoke-only status: a
54-example, 108-intervention proxy run is recorded in
`experiments_pass_strategies.md` and local artifacts. Its paired refresh benefit
remains inconclusive. That inspection recommended a reference-freeze test that
lets downstream token predictions respond freely. The earlier baseline queue
lists Dynamic-dLLM next, then TraceLock.

Planning snapshot from earlier on 2026-09-16: the requested workload was a
[draft plan for two locking experiments](.omx/plans/two-locking-experiments-2026-09-16.md):
SureLock-style KL-gated representation freezing and TraceLock-style learned
token-identity commitment. The draft evaluates the policies separately on the
same LLaDA backbone, with free continuation, paired quality/runtime comparisons,
and example-disjoint training and calibration. Its proposed first token-lock
policy uses the existing compact trace features; it is explicitly an adaptation,
not a reproduction of the published hidden-state controller. At that planning
checkpoint, no new model run or implementation had started.

Execution inspection on 2026-09-16 at 20:06 Asia/Dubai: the later
[execution report](.omx/reports/task-2026-09-16-175242.md) records that work
started in `/tmp/token-fate-two-locking-20260916`, on
`exp/two-locking-policy-20260916` based on `4d16746`. That worktree is absent
on the currently inspected host, `ws-l2-008`; Git marks its registration as
prunable. The branch still points to its base with no committed implementation
changes, and `results/locking_policy_eval_20260916/` is absent. The report's
eight passing core tests and GPU smoke retry are historical status, not evidence
of completed policy evaluation. No relevant host-local process was found;
the scheduler query timed out, so remote execution status remains unknown.

At that inspection, the workload was the started but locally unverifiable
two-policy quality/runtime study, superseding the older baseline queue.
Locate the earlier session's working files and GPU logs before restarting it;
preserve the worktree registration while assessing recovery. Fresh inspection
passed 10 Fast-dLLM helper tests, parsed all 50 Python files, and independently
checked 620 saved outputs plus eight pairwise output/correctness comparisons
per dataset. Model generation and task scoring were not rerun. Details are in
the [inspection report](.omx/reports/task-2026-09-16-200639.md).

Planning request on 2026-09-16: draft the two experiments before further
implementation. The [revised experiment plan](.omx/plans/two-locking-experiments-v2-2026-09-16.md)
specifies a shared LLaDA baseline, KL-gated dense/sparse reference-freeze arms,
and compact trace-supervised fixed-budget/adaptive token-commitment arms with
a confidence-only comparator. It defines example-disjoint development splits,
free continuation, frozen operating points, numerical backend checks, and
paired quality/runtime evaluation. The earlier plan and execution report are
preserved as history. This revision is a design artifact; no new implementation,
model run, or recovery operation was performed while drafting it.

Implementation update on 2026-09-16: the user requested implementation, now
present in the persistent `.omx/worktrees/two-locking-v2/` checkout on branch
`exp/two-locking-v2`. It includes both policy families, frozen manifests,
trace collection and calibrated training, development-only operating-point
selection, paired analysis, and backend validation. The
[execution guide](.omx/worktrees/two-locking-v2/docs/two_locking_experiments.md)
contains runnable commands. The original dirty source files remain unchanged.

Fresh verification passed 132 CPU tests, scoped lint/type checks, and source
syntax checks. The real-model short pipeline produced 20 complete outputs
after collecting two fit/two calibration traces and training both predictors.
A full-length engineering check produced another 12 complete outputs at
gen256/block32, with zero masks or token-identity violations in both runs.
These small checks are not final quality/runtime results.

Large-lock bfloat16 sparse execution has a measured limitation: the strict
matched-state test with 128 frozen rows fails dense/sparse equivalence, including
five fresh-reference argmax changes. The validator explicitly flags this for
review; a broad numerical envelope no longer counts as equivalence. The sparse
arm must remain experimental until this is resolved or separately accounted for.
Full development collection, threshold selection, and the 54/70 study remain
unrun. Detailed implementation and verification evidence is in the
[implementation report](.omx/reports/task-2026-09-16-204805.md).

Local verification that does not load LLaDA:

```bash
python3 -m compileall src scripts tests
python3 tests/test_features.py
python3 tests/test_counterfactual_selection.py
```

Full-test execution update: the user requested all existing evaluation samples.
Slurm arrays `196642` and `196675` are running the frozen 54 MATH + 70 HumanEval test set with
all six current policies and three repeats (2,232 expected measured outputs).
The current saved prototype checkpoints and thresholds are held fixed; there
is no retraining or tuning on test labels. Source is frozen in the campaign at
`results/locking_full_test_20260916_231245/`. The existing sparse-equivalence
limitation remains attached to its results. Status and completion evidence are
recorded in `.omx/reports/task-2026-09-16-231245.md`.

Automatic strict merge and paired analysis are scheduled as job `196679`, after
both arrays finish. The scheduler currently permits one evaluation task at a
time; all samples remain queued. Progress is in each shard’s `records.jsonl`,
and `finalization.json` will record the eventual aggregate success or failure.

Original-plan execution restarted on2026-09-17 after the user requested the
proper400-fit/100-calibration/100-selection study. Campaign:
`results/locking_original_plan_20260917_204619/`. The first20 full-length training
trajectories are validated; array198482 is collecting the rest. Evaluation is
gated on complete training, calibration, development selection, and backend
acceptance. The earlier prototype campaign's final merge failed on two missing
shards and is preserved separately. The authoritative current report is
`.omx/reports/task-2026-09-17-204619.md`.

### Original-plan execution, 2026-09-17

The active full study is `results/locking_original_plan_20260917_204619`: 400
GSM8K fit, 100 calibration, 100 selection, then the frozen 54 MATH / 70 HumanEval
panel with three timing repeats. Collection is active; final results are pending.
The sparse BF16 backend remains numerically approximate despite passing all
192 real-layer float32 semantic checks. Its current implementation skips
projection/FFN rows while preserving full attention geometry. Keep that status
visible when interpreting speed/quality. The authoritative workflow is
`orchestration/study.py` plus `orchestration/driver.py`; the alternative
subagent-submitted downstream jobs were canceled before execution. See
`.omx/reports/task-2026-09-17-204619.md` and campaign `validation/numerical_status.md`.

### Original-plan status, 2026-09-18

Training/calibration/selection and 16-case smoke passed. 14 MATH evaluation
cases (all 12 policies, 3 repeats) finished; 110 cases remain. A GPU contention
guard stopped the initial evaluation. The busy host is now excluded and
recovery array 199263 is running on another host, with callback 199264.
Interim results: results/locking_original_plan_20260917_204619/reports/partial_math_14_20260918_112503/partial_results.md. These are partial,
not full-panel results. Session report: .omx/reports/task-2026-09-18-113348.md.

### Paper-mechanism redesign,2026-09-20

The previous original-plan campaign completed all124cases. A user-authorized
redesign now lives in .omx/worktrees/paper-locking-v3, with campaign
results/locking_paper_redesign_20260920_161812. It implements real active
attention/head removal for representation caching and a contextual hidden-state
Transformer/dynamic-threshold/moving-window token controller. Fresh mixed-domain
hidden traces are required; old scalar traces are not a replacement. The plan is
.omx/plans/paper-locking-redesign-2026-09-20.md and active report is
.omx/reports/task-2026-09-20-paper-locking-redesign.md. Full rerun is being prepared.


Execution update on 2026-09-20: the two paper-mechanism experiments have been redesigned and launched in `results/locking_paper_redesign_20260920_161812`, using `.omx/worktrees/paper-locking-v3`. E1 now removes frozen query/FFN/head computation and retains frozen per-layer K/V; E2 uses a contextual hidden-state controller with 832 projected features. These are LLaDA mechanism ports. The exact 54 MATH + 70 HumanEval evaluation panel is retained with three timing repeats and explicit speedup percentages. Source tests: 199 passed; orchestration tests: 25 passed; real GPU numerical, hidden-trace, and controller integration preflights passed. Full typing/style checks are not clean and are recorded in the report.

E1 smoke array 202023/callback 202024 and E2 collection array 202035/callback 202036 are submitted. The first full E2 shard accepted 40/40 traces; a helper uses allocation 201790 alongside the queued jobs. Full training targets 8,000 accepted fit and 400 validation examples, then 10,000 controller updates. Scientific outcomes are pending. See [task-2026-09-20-182157.md](.omx/reports/task-2026-09-20-182157.md) and each campaign's `e1_completion.json` / `e2_completion.json` for subsequent completion evidence.


2026-09-21 result check: E1 completed all124cases,7policies,3repeats. The vocabulary-KL gate at epsilon0.005 (then misidentified as representation KL) was7.92% faster overall with40/124correct versus43/124baseline; strict epsilon0 had no meaningful speed gain. See [the complete accuracy/time table](results/locking_paper_redesign_20260920_161812/e1_analysis/results_table.md). E2 has3,710accepted traces across94completed shards and no trained-model evaluation yet. Its batch memory request was corrected from96 to64GiB and the busyGPUhost excluded; a separate code-scoring error remains under diagnosis. Current report: [.omx/reports/task-2026-09-21-144935.md](.omx/reports/task-2026-09-21-144935.md).


2026-09-21 correction and execution: the completed earlier E1 used vocabulary-probability KL to gate representation freezing. It is retained as a baseline and does not answer the user's intended representation-KL question. The user selected final-layer hidden representations for the corrected online test. New campaign `results/representation_kl_last_hidden_20260921_203301` uses KL between softmax-normalized final pre-vocabulary hidden vectors at consecutive decoding steps, temperature1. The seven policies include representation-KL sparse thresholds0/0.0005/0.005, dense/no-lock controls, B0 and a fresh token-KL control. All54 MATH +70 HumanEval cases and3timing repeats are retained. Source223tests and orchestration9tests passed; realGPU capture/work/signal/diagnostic checks passed. Smoke array203974/callback203975 now run before automatic full evaluation. Offline current-to-final analysis is deferred until after this online experiment. No corrected representation-KL speed/quality result has been claimed yet.


### Corrected representation-KL result, 2026-09-22

The corrected online campaign completed all 124 samples × 7 policies × 3 timing repeats. KL is measured over softmax-normalized final pre-vocabulary hidden features at consecutive decoding steps (temperature 1). Representation epsilon 0.0005 achieves 9.61% measured speedup with 43/124 correct, the same aggregate count as B0, but fixes six cases and introduces six errors. Epsilon 0.005 achieves 10.10% speedup with 41/124 correct; the token-KL control achieves 7.74% with 40/124. All methods use 256 decode calls; epsilon 0.0005 reduces reported query/projection/FFN/head token rows by 27.10%. These are exploratory results on this panel, not a guarantee of answer preservation or all-layer convergence.

See [the online report](results/representation_kl_last_hidden_20260921_203301/analysis/online_representation_kl_results.md) and [dataset-level results](results/representation_kl_last_hidden_20260921_203301/practical_full_results_table.md). All 2,604 records passed integrity checks; an independent architect approved the online experiment. The temporarily deferred E2 collection queue has resumed. The user-requested offline follow-up passed its two-case GPU replay check and is running on all 124 cases from a separate `offline/frozen_v1` snapshot, using unfrozen trajectories and an extra completed-context final forward; the online artifacts remain fixed.


### Completed online and offline representation-KL study, 2026-09-22

The [combined results](results/representation_kl_last_hidden_20260921_203301/representation_kl_results.md) now include the full 124-case offline audit. Every no-lock replay matched the accepted tokens/calls/raw-KL diagnostics; all grids and provenance passed independent verification. At epsilon0.0005, 13,541/24,578 hypothetical first admissions (55.09%) still exceed the same final-state KL cutoff; median raw-vector relative-L2 distance to final is55.54%. Thus the online9.61% speedup at43/124correct is a useful approximate tradeoff on this panel, not proof of final-state or all-layer convergence. Source/thresholds and online data remained fixed. The separately running E2/TraceLock collection has resumed after all temporary holds were released.


### FLOP accounting for both locking methods, 2026-09-22

The [FLOP report](docs/locking_flops_results_2026-09-22.md) covers the full 124-case panel for both methods. SureLock-style algorithmic GEMMs are counted separately from vocabulary-head and learned-controller GEMMs. Representation KL 0.0005 reduces extended GEMMs 27.09%, alongside its previously measured 9.61% speedup and 43/124 correct. A separate frozen TraceLock 0.95 FLOP-only replay reduces extended GEMMs 68.12% including projector/controller cost, with 79.60 mean forwards versus 256. This pass does not establish E2 correctness or wall-clock speedup; the original E2 quality/runtime pipeline remains separate. All 22 focused tests, scoped typing/lint,14 E1 operator/trajectory checks,2 E2 operator/trajectory checks and124 E2 profile integrity checks passed. Original experiment sources and records remain unchanged.


### SureLock and semantic token-KL comparison, 2026-09-22

A new user-authorized paired campaign is running in `results/surelock_token_vs_tracelock_20260922_2145`. It compares a shared moving-window confidence baseline, SureLock-style vocabulary-KL state caching (epsilon 0.005, confidence percentile 20), a new vocabulary-KL token-commitment arm (epsilon 0.005, no state caching), and frozen TraceLock at 0.95. All 124 samples and three timing repeats are included (1,488 expected records). Source tests (230), orchestration tests (8), scoped typing/lint and 16 full-length development outputs passed before source/config freezing. The runner automatically validates and reports correctness, measured speedup and FLOPs after completion. The original E2 pipeline continues separately. See [task-2026-09-22-220750.md](.omx/reports/task-2026-09-22-220750.md) for exact definitions, provenance and the running process.


### Status check, 2026-09-28

Both campaigns left running on 2026-09-22 have finished; no jobs are queued
besides the interactive allocation. This check read saved artifacts only and
did not rerun anything.

- **SureLock-vs-TraceLock comparison** (`results/surelock_token_vs_tracelock_20260922_2145/results.md`,
  1,488/1,488 records, moving-window-32 B0 at 45/124): SureLock-cache-0.005
  scores 43/124 at +5.96% speed and 27.46% fewer FLOPs. KL-token-commit-0.005
  drops to 26/124 (+451%). T-adaptive-0.95 (TraceLock) scores 43/124 at +208.68%
  speed and 68.12% fewer FLOPs.
- **E2 TraceLock evaluation** (`results/locking_paper_redesign_20260920_161812/e2_analysis/summary.md`,
  complete 2026-09-22T22:58Z): `C-adaptive-0.956667` is a training-free
  confidence threshold (commit every eligible token with confidence at or above
  the threshold, else the most confident one). Its threshold was set on the
  disjoint 200-example selection split to match TraceLock's mean forward count
  (73.9 vs 74.5). It matches B0 exactly on both datasets (45/124, sequence change
  8.6% HumanEval / 5.6% MATH). The learned T-adaptive-0.95 scores 43/124 with
  74–100% sequence change. Speedups: C-adaptive +536.9% HumanEval / +90.4% MATH;
  T-adaptive +469.9% / +92.5%. T-adaptive's −2 correct is inside its intervals
  (accuracy delta HumanEval [−0.157, +0.057], MATH [−0.056, +0.093]). So there is
  no evidence that the learned controller beats a confidence threshold at the
  same forward count on this panel. This tested a TraceLock port, not PhaseLock's
  own three-gate controller. It weakens Contribution 4 in `research_idea.md` (that
  learned fate prediction adds value over confidence among lock candidates), and
  with it any plan to build PhaseLock's commitment gate as a learned model on
  similar features.
- **Two baselines:** the representation-KL/E1 campaigns use a block-32 B0
  (43/124); E2 and the comparison campaign use a moving-window-32 B0 (45/124).
- **Stale docs:** `docs/locking_experiment_results_2026-09-22.md` still reports
  E2 as evaluating. `results/README.md` has no rows for Fast-dLLM, E1/E2,
  representation KL, FLOPs, or the comparison campaign.


### Aligned standard-LLaDA campaign, 2026-09-29

The user asked to align the experiment setting with the collaborator's work, then rerun their strategies and baseline under it.
The chosen setting is:
- the standard LLaDA low-confidence sampler: block 32, one token per step, 256 tokens and 256 steps;
- batch 1, BF16, seed 23;
- the fixed 124-question panel of 54 MATH Level-5 and 70 HumanEval;
- three timing repeats.

The protocol for both collaborators is `docs/aligned_protocol_2026-09-29.md`. The campaign is `results/aligned_standard_g256_20260929/`, run as Slurm job 211194 (after two setup attempts, 211189 and 211191). It runs six policies:
- B0;
- RepKL-freeze-0.0005;
- SureLock-gen-0.005;
- SureLock-prompt-0.005 (prompt rows eligible);
- TraceLock-0.95;
- Confidence-0.9 (new `C-block` arm: Fast-dLLM-style threshold inside blocks).

The source is pass 7's frozen source plus the C-block arm. Before launch, 233 of 234 suite tests passed; the one failure is a pre-existing path-dependent test. The smoke stage checks exact output parity with the earlier identical B0, RepKL and TraceLock runs before the full run starts. No result is claimed yet; see the campaign's `state.json`, `results.md` and `completion.json`.

**Result (completed 2026-09-29 23:56 local, job 211194):**
- All 2,232 records passed validation.
- Exact output parity was 124/124 for B0 and RepKL (vs pass 4) and for TraceLock (vs pass 7).
- B0 scored 43/124.

| Policy | Correct /124 | Speedup vs B0 [95% CI] | Mean calls | FLOP cut | Outputs changed |
|---|---:|---:|---:|---:|---:|
| RepKL-freeze-0.0005 | 43 | +10.5% [+9.3, +11.8] | 256 | 27.1% | 99/124 |
| SureLock-gen-0.005 | 44 | +7.7% [+6.5, +9.0] | 256 | 27.5% | 97/124 |
| SureLock-prompt-0.005 | 45 | +28.0% [+24.4, +31.9] | 256 | 62.6% | 106/124 |
| TraceLock-0.95 | 43 | +208.1% [+177.2, +248.1] | 79.6 | 68.1% | 109/124 |
| Confidence-0.9 | 44 | +271.8% [+232.3, +324.2] | 67.9 | 72.9% | 26/124 |

Every accuracy interval includes zero. Confidence-0.9 broke 0 answers and fixed 1, and changed only 26 outputs. Prompt freezing adds about 20 points of speed over generated-only SureLock.

See `results/aligned_standard_g256_20260929/results.md`, `analysis/paired/summary.md` and `analysis/parity.json`.


### Collaborator methods under the aligned protocol, started 2026-09-30

The campaign is `results/aligned_collab_g256_20260930/`. Her code is frozen at commit 24cd51b. It is scored with our prompts and our scorer, and every job reruns our B0 on the same GPU.

Her checkpoints are private, so her learned models are being retrained here with her own recipe:
- TraceLock-LLaDA traces: 4 × 2,000, 128/128/32, seed 42. Her projection sha reproduces exactly.
- Controller training.
- Direct-safety data and predictor.

Declared adaptations:
- a transformers-5 loader shim;
- controller max length 1024, because two panel prompts exceed 512 tokens;
- open_confidence Phase 3 abstains, because her branch has no latency calibrator.

Jobs:
- Non-learned methods (dense, SureLock 5e-4/5e-3, capped confidence, H2 guard90_delay1): job 212013, complete.
- Retraining is complete: traces (8,000 examples), controller (212006), and safety predictor (212020; threshold 0.808).
- Learned methods: job 212840, complete (`analysis_learned/`). Job 212021 had stopped at a reporting KeyError in
  `gate.py` after a successful smoke run.
- Combined table of all 13 methods vs B0: `results/aligned_collab_g256_20260930/combined_results.md`.

Learned results (same-job B0 15.930 s/example, 124/124 parity):

| Method | Correct /124 | Δ correct (pp) [95% CI] | Speedup [95% CI] | Mean calls |
|---|---:|---:|---:|---:|
| collab-TraceLock-LLaDA | 41 | −1.61 [−8.87, +5.65] | +176.42% [+144.17, +219.40] | 89.6 |
| collab-open-confidence | 42 | −0.81 [−5.65, +4.03] | +150.67% [+124.19, +185.10] | 88.2 |
| collab-safety-full | 45 | +1.61 [−2.42, +6.45] | +100.57% [+94.21, +107.53] | 109.1 |

The open-confidence speedup uses her additive time estimate. Its online planner gave identical outputs and
measured +167.4%.

Non-learned results (2026-10-01; `results/aligned_collab_g256_20260930/analysis_phaseA/results.md`). Same-job B0
reproduces the aligned campaign's B0 on 124/124, and her dense matches our B0 on 124/124 (but is 3.7% slower).

| Method | Correct /124 | Δ correct vs B0 (pp) [95% CI] | Speedup vs same-job B0 [95% CI] | Mean calls | Broken / fixed |
|---|---:|---:|---:|---:|---:|
| B0 (15.535 s/example) | 43 | — | — | 256 | — |
| collab-dense | 43 | 0 | −3.67% [−3.81, −3.54] | 256 | 0 / 0 |
| collab-SureLock-5e-4 | 34 | −7.26 [−14.52, −0.81] | +19.02% [+16.39, +21.82] | 256 | 14 / 5 |
| collab-SureLock-5e-3 | 34 | −7.26 [−14.52, +0.00] | +19.25% [+16.57, +22.11] | 256 | 15 / 6 |
| collab-capped-confidence | 43 | 0 | −0.74% [−1.04, −0.43] | 244.9 | 0 / 0 |
| collab-H2-guard90-delay1 | 47 | +3.23 [−4.03, +10.48] | +107.73% [+95.76, +121.36] | 143.3 | 8 / 12 |

Her SureLock is the only method in either campaign whose accuracy interval excludes zero (5e-4). The changed
outputs are coherent but take different reasoning paths, so this is not a decoding artifact.
