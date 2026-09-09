# PhaseLock semantic-lock development log

## 2026-09-09: isolated selector screen

### Question

Which online signal is safest for adding irreversible token-identity commitments
to the ordinary LLaDA denoising schedule?

### Intervention

- A0: ordinary blockwise LLaDA decoding; each step transfers the scheduled
  highest-confidence tokens.
- A1(selector): preserves those same scheduled confidence transfers and lets
  exactly one selector choose only the additional accelerated transfers.
- Representation updates remain live for every committed token.
- No hidden/K/V freeze and no row-compute removal are enabled.

This decomposition is important: a selector must not replace the normal
scheduled transfers, otherwise A0 versus A1 confounds an extra early-lock
intervention with a different base decoder.

### Standalone selectors

1. `confidence`: largest current top-1 probability.
2. `margin`: largest top-1 minus top-2 probability.
3. `entropy`: smallest current posterior entropy.
4. `posterior_kl`: smallest adjacent-step KL divergence. The first step has no
   posterior history, so its extra transfer uses confidence and is explicitly
   logged as a fallback.
5. `persistence_only`: longest consecutive run of an unchanged top-1 proposal;
   confidence is used only as a `1e-6` tie-breaker.
6. `token_fate_only`: smallest predicted probability that the current proposal
   differs from its eventual token under an ordinary completed trace.

The existing fused policies (`posterior`, `persistence`, `token_fate`, and
`consensus`) are deliberately excluded from this screen.

`trace BO` is not reported as a separate result: neither the TraceLock paper
nor its public repository defines a method by that name. TraceLock uses a
learned hidden-state trajectory controller whose released implementation is
model-specific, so relabeling the repository's token-fate probe as official
TraceLock would be misleading. Here `token_fate_only` is the available learned
trajectory baseline; a faithful LLaDA TraceLock port remains separate work.

### Matched settings

- Model: `GSAI-ML/LLaDA-8B-Instruct`
- Steps / generation length / block length: `64 / 64 / 32`
- Added semantic-lock floor: `6%` of currently available positions
- Seed: `17`
- Smoke sample: two examples per dataset
- Dataset slices: GSM8K test offset 100, MATH500 test offset 300, HumanEval
  offset 0. MATH500 offset 300 is disjoint from the fate probe's examples
  0--199; unlike GSM8K, MATH-500 exposes only a test split.
- HumanEval token-fate scorer: GSM8K-trained cross-task probe, therefore
  exploratory rather than a task-matched learned baseline

### Metrics

- A0-to-A1 exact sequence match and token agreement
- A0-correct/A1-wrong answer harm and A0-wrong/A1-correct answer gain
- accuracy delta
- NFE, masked-token-forward, and wall-clock reductions
- retrospective accelerated-lock precision: fraction of selector-controlled
  commitments whose token equals A0's eventual token at the same position
- high-confidence accelerated-lock error rate

Retrospective lock precision diagnoses semantic fate but is not causal proof of
commitment safety. A0 versus A1 answer harm is the intervention test.

### Implementation changes

- Added pure entropy, posterior-KL, persistence, and token-fate priorities.
- Split transfers into `scheduled` and `accelerated` ledger events.
- Added a first-step posterior-history fallback rather than treating zero KL as
  stability evidence.
- Compute the learned fate score only for fate-based policies, so other methods
  do not pay its controller overhead.
- Added a semantic-specific analyzer and Slurm runner.
- Fixed the trajectory-state update so the current top-1 run length is carried
  into the next denoising step. A regression test covers this transition.
- Local verification: 24 tests passed.

### Integrity log

- Initial jobs `182122`--`182124` failed before decoding because stale Slurm
  GPU accounting placed them on a device with only 57 MiB free. No results from
  those jobs are used.
- Jobs were pinned to `gpu-01`, where Slurm's free device and actual GPU memory
  agreed.
- The first MATH500 smoke at offset 100 overlaps the task-matched fate probe's
  training examples. It is retained only as a functionality check and is not
  evidence.
- The first full GSM8K/HumanEval outputs and partial MATH500 run revealed that
  the newly computed top-1 run length was not carried to the next denoising
  step. This made `persistence_only` a confidence tie-break and gave the fate
  model a constant run-length feature. Those screens are invalid for model
  selection.
- Corrected full-screen jobs: HumanEval `182156`, GSM8K `182157`, and disjoint
  MATH500 `182158`.

Status: corrected standalone screens complete; only the corrected run set below
is used for model selection.

### Corrected standalone screen result

Valid runs used 20 GSM8K examples at test offset 100, 20 MATH500 examples at
offset 300, and 10 HumanEval tasks at offset 0. Every method made 16 optional
locks per example and reduced NFE from 64 to 48.

| selector | pooled lock precision | exact output | answer harm | runtime gain |
|---|---:|---:|---:|---:|
| confidence | 0.569 | 0.260 | 0.080 | 0.247 |
| entropy | 0.515 | 0.200 | 0.160 | 0.247 |
| margin | 0.515 | 0.240 | 0.040 | 0.246 |
| persistence only | 0.509 | 0.160 | 0.080 | 0.246 |
| posterior KL only | 0.388 | 0.000 | 0.100 | 0.246 |
| learned token fate only | 0.535 | 0.240 | 0.060 | 0.219 |

No standalone selector is robust at this budget. Margin has the lowest observed
answer harm, confidence has the highest retrospective lock precision, and the
learned fate selector does not dominate either while paying about 2.8
percentage points of runtime gain in controller overhead. Persistence reverses
across domains: it is strong on HumanEval but weak on GSM8K and MATH500.

### PhaseLock semantic candidate v1

The error rate rises after repeated optional commitments within a block. The
next candidate therefore separates admission from ranking and may abstain:

- `capped_confidence`: allow an optional lock only in the first two denoising
  steps of a block and only at confidence at least 0.95; rank by confidence.
- `capped_persistence`: use the identical admission gate and optional-lock cap,
  but rank admitted candidates by consecutive top-1 persistence.

This is evaluated on untouched offsets (GSM8K 200, MATH500 320, HumanEval 10).
The fixed development settings are not retuned after inspecting holdout output.
Holdout jobs: MATH500 `182185`, GSM8K `182186`, HumanEval `182187`.

### Frozen holdout result

Across 20 GSM8K, 20 MATH500, and 10 HumanEval examples:

| selector | optional locks | lock precision | exact output | answer harm | NFE gain | runtime gain |
|---|---:|---:|---:|---:|---:|---:|
| capped confidence | 75 | 1.000 | 1.000 | 0.000 | 0.0234 | 0.0235 |
| capped persistence | 77 | 1.000 | 1.000 | 0.000 | 0.0241 | 0.0240 |

For capped confidence, the 95% Wilson interval is `[0.951, 1.000]` for
retrospective lock precision and `[0.000, 0.071]` for answer harm. Capped
persistence produces only two more optional locks and about 0.05 percentage
points more runtime gain, so the holdout does not establish a meaningful
ranking advantage over capped confidence. The main improvement comes from the
admission threshold, abstention, and per-block intervention cap.

## 2026-09-10: frozen confirmation

The gate was kept unchanged for a larger untouched confirmation: GSM8K offset
220 with 50 examples (`182196`), MATH500 offset 340 with 50 examples
(`182197`), and HumanEval offset 20 with 20 examples (`182198`). No threshold,
cap, selector, seed, or decoding setting was changed after the first holdout.

### Confirmation-only result

Across the 120 new confirmation examples:

| selector | optional locks | lock precision (95% Wilson) | exact output | answer harm (95% Wilson) | NFE gain | runtime gain |
|---|---:|---:|---:|---:|---:|---:|
| capped confidence | 194 | 1.000 `[0.981, 1.000]` | 1.000 | 0.000 `[0.000, 0.031]` | 0.0253 | 0.0245 |
| capped persistence | 195 | 1.000 `[0.981, 1.000]` | 1.000 | 0.000 `[0.000, 0.031]` | 0.0254 | 0.0247 |

### All untouched evaluations

Pooling the 50-example first holdout and 120-example frozen confirmation gives
170 untouched examples:

| selector | optional locks | lock precision (95% Wilson) | exact output | answer harm (95% Wilson) | NFE gain | runtime gain |
|---|---:|---:|---:|---:|---:|---:|
| capped confidence | 269 | 1.000 `[0.986, 1.000]` | 1.000 | 0.000 `[0.000, 0.022]` | 0.0247 | 0.0242 |
| capped persistence | 272 | 1.000 `[0.986, 1.000]` | 1.000 | 0.000 `[0.000, 0.022]` | 0.0250 | 0.0245 |

There are no observed semantic-lock failures in these untouched slices, but
zero observed failures is not proof of universal safety. The intervals bound
what this sample supports: for capped confidence, the lower 95% bound on
retrospective lock precision is 0.986, and the upper 95% bound on per-example
answer harm is 0.022.

The persistence ranker still does not establish a useful advantage: it makes
only three additional commitments and improves runtime by about 0.027
percentage points over capped confidence across all untouched examples. The
defensible candidate is therefore the simpler capped-confidence admission
rule. Its measured quality safety comes with a deliberately modest efficiency
gain: about 2.47% NFE and 2.42% wall-clock reduction in this setup.

These experiments validate only PhaseLock's semantic stage (A0 versus A1).
Representations remain live and row compute remains dense, so they do not test
or establish representation-lock or compute-lock safety.

### Reproduction

The standalone screen can be submitted one dataset at a time with:

```bash
sbatch --export=ALL,OUTPUT_DIR=results/semantic_screen,DATASET=gsm8k,LIMIT=20,OFFSET=100,SEMANTIC_POLICIES=confidence\,margin\,entropy\,posterior_kl\,persistence_only\,token_fate_only cluster/run_semantic_lock.slurm
```

The frozen candidate uses the same runner with the admission gate enabled:

```bash
sbatch --export=ALL,OUTPUT_DIR=results/semantic_capped,DATASET=gsm8k,LIMIT=50,OFFSET=220,SEMANTIC_POLICIES=capped_confidence\,capped_persistence,SEMANTIC_MIN_CONFIDENCE=0.95,SEMANTIC_OPTIONAL_STEPS_PER_BLOCK=2 cluster/run_semantic_lock.slurm
```

Set `REGRET_MODEL_PATH` for `token_fate_only`. Leave it unset for the cheap
selectors and capped candidate so they do not load or execute the learned
scorer.
