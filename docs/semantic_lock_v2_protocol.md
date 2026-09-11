# PhaseLock semantic-lock v2 development protocol

## Frozen reference

The validated semantic-lock v1 implementation is preserved at Git tag
`semantic-lock-v1-baseline` (commit `8876fdf`). Its policy is:

- keep ordinary LLaDA scheduled confidence transfers unchanged;
- allow optional transfers only in the first two steps of each block;
- require top-1 confidence at least `0.95`;
- use the `0.06` total-transfer target to bound optional promotion;
- rank eligible optional transfers by confidence and abstain when none qualify;
- keep representations live and row compute dense.

The v1 baseline is rerun in every v2 development job as
`baseline_c095_s2`; historical results are not mixed with newly timed arms.
The one-arm manifest `configs/semantic_lock_v1_baseline.json` is the canonical
way to regenerate only this A1 policy and its exact commitment plans.

## Scope

This experiment changes only the semantic commitment controller. Every A1 arm
uses dense computation, keeps hidden/K/V representations live, and writes its
exact semantic commitment plan to `phase_plans.jsonl` for later A2 replay.
Normal scheduled transfers are identical across A0 and all A1 arms. The tested
controller affects only additional early commitments.

Representation cosine-drift threshold, post-commit semantic age, and
transformer layer are intentionally excluded. They are A1-to-A2 variables;
using them in the main semantic gate would blur the semantic/representation
separation that PhaseLock is intended to test.

For semantic locking, the corresponding online quantities are:

- adjacent-step posterior KL as semantic-distribution drift;
- consecutive unchanged top-1 proposals as pre-commit patience;
- confidence, top-2 margin, entropy, and persistence as ranking signals.

## Development arms

The manifest `configs/semantic_lock_v2_development.json` defines 29 arms:

1. frozen v1 baseline: confidence `>=0.95`, first two block steps;
2. loose control: confidence `>=0.90`, first four block steps;
3. four matched-admission alternatives to the baseline ranker: margin,
   negative entropy, persistence, and posterior KL;
4. margin admission at `0.05`, `0.10`, and `0.20` under the loose window;
5. a `5 x 4` trajectory grid under the loose window:
   - maximum adjacent-step KL: `0.002, 0.005, 0.01, 0.02, 0.05`;
   - minimum unchanged-proposal run length: `1, 2, 3, 4`;
   - posterior history is required, so a missing previous posterior cannot be
     interpreted as zero drift.

All arms use the same `0.06` transfer target and seed `17`.

## Development and confirmation split

The grid is selected only on the already-used development slices:

- GSM8K test offset 100, 20 examples;
- MATH500 offset 300, 20 examples;
- HumanEval offset 0, 10 tasks.

After one candidate is selected, its configuration is frozen and evaluated on
new untouched slices. No threshold is changed after examining confirmation.

## Predeclared selection rule

Selection is safety-first and lexicographic:

1. reject any arm with nonzero A0-correct/A1-wrong harm on any dataset;
2. among the remainder, prefer exact A0 sequence agreement on every
   development example;
3. require retrospective optional-lock precision no worse than v1;
4. among safety-equivalent arms, maximize NFE reduction and optional locks per
   example;
5. use measured runtime only as a secondary tie-breaker because all arms run
   sequentially in one process; remeasure shortlisted arms in a matched timing
   run before making a wall-clock claim;
6. prefer the cheaper/simpler controller when efficiency differences are
   within noise.

Aggregate accuracy alone is not a safety criterion because answer harms can be
cancelled by answer gains.

## Integrity log

- Job `185075` failed during model placement before decoding because Slurm
  assigned a physically occupied GPU on `gpu-01`. No output is used.
- Job `185081` was cancelled during initialization after verifying that all
  physical devices on Slurm-idle `gpu-04` were occupied by stale/untracked
  processes. No output is used.
- Job `185090` is the completed one-example, 29-arm smoke test on Slurm GPU
  index 2 of `gpu-02`, whose physical memory availability was verified before
  launch. It produced all 29 A1 outputs, all 29 exact semantic plans, and all
  29 semantic-analysis rows.
- Completed development jobs are GSM8K `185100` and MATH500 `185101`.
- Dependent HumanEval job `185102` was cancelled during initialization before
  decoding because Slurm reassigned it to an index with insufficient physical
  memory. No output is used. After rechecking per-index memory, HumanEval was
  resubmitted as `185284` to a Slurm index with 32 GB free.

## Development round 1 result

Jobs `185100`, `185101`, and `185284` completed all 50 development examples
and all 29 arms. The frozen baseline reproduced its safety result: 77/77
optional locks matched A0, every output was exactly equal to A0, and mean NFE
reduction was 2.41%.

| policy family | locks/example | lock precision | exact A0 | answer harm | NFE gain |
|---|---:|---:|---:|---:|---:|
| frozen `c0.95 / steps<2` baseline | 1.54 | 1.000 | 1.000 | 0.000 | 0.0241 |
| loose `c0.90 / steps<4` control | 3.64 | 0.989 | 0.980 | 0.000 | 0.0569 |
| matched strict rankers | 1.54--1.58 | 1.000 | 1.000 | 0.000 | 0.0241--0.0247 |
| `c0.90 / steps<4` KL-patience grid | 1.84--2.88 | 0.989--0.993 | 0.980 | 0.000 | 0.0288--0.0450 |

All looser trajectory gates changed the sequence for the same MATH500 example,
so the entire family is rejected by the predeclared exact-A0 criterion. The
automatic rule provisionally ranks `rank_persistence_c095_s2` first among safe
arms, but its improvement over the baseline is only two optional locks over 50
examples and 0.0625 percentage points of NFE. That is too small to establish a
meaningful algorithmic improvement or justify confirmation by itself.

### Shared failure diagnosis

The divergent case is `math500-test-316`. The strictest tested trajectory gate
commits the wrong token at relative position 62 on global step 32, the first
step of block 2, despite apparently strong signals:

- confidence: `0.9108`;
- top-2 margin: `0.8261`;
- adjacent-step posterior KL: `1.61e-5`;
- global unchanged-proposal run length: `33`.

The error changes 21 later sequence positions, although both final normalized
answers happen to remain correct. It demonstrates that a proposal can look
stable while its block is inactive and become unsafe when that block first
enters the active denoising context. Correctness-only evaluation would miss
this trajectory change.

## Development round 2: active-block evidence

Round 2 distinguishes global history from evidence accumulated while the
token's block is active:

- active-block proposal run length resets at every block boundary;
- `semantic_min_block_age` prevents optional commitment until the block has
  received the requested number of active denoising updates;
- adjacent-step KL gates that use active age at least one compare two forwards
  from the same active block.

The 25-arm manifest `configs/semantic_lock_v2b_active_block.json` tests active
ages `1`, `2`, and `4`, plus the same five KL thresholds and four block-local
patience values. Job `185334` tested all arms on the diagnosed failure case
before any full rerun.

On `math500-test-316`, the loose age-zero control made three optional locks,
including the known wrong lock at relative position 62; optional-lock precision
was 1/3 and the final sequence differed from A0. The active-age controls behaved
as follows:

| target policy | optional locks | lock precision | exact A0 | NFE gain |
|---|---:|---:|---:|---:|
| loose `c0.90 / age0 / steps<4` | 3 | 0.333 | 0 | 0.0469 |
| warm `c0.90 / age1 / steps<4` | 2 | 1.000 | 1 | 0.0312 |
| warm `c0.90 / age2 / steps<4` | 2 | 1.000 | 1 | 0.0312 |
| warm `c0.90 / age4 / steps<8` | 1 | 1.000 | 1 | 0.0156 |

The exact plans show why: age zero locks position 62 at global step 32
(`block=1`, `step_in_block=0`), while age one forbids that decision and commits
only later positions. The strictest KL gates safely abstain on this example;
the `0.05` KL gates make one correct optional commitment. This targeted result
validates the repair mechanism but is not used as evidence of generalization.

The full round-2 development jobs use the same slices and manifest as the first
round. GSM8K job `185372` completed 20 examples, 25 A1 arms, and exactly 500
saved plans. MATH500 job `185538` completed 20 examples and 500 plans; HumanEval
job `185680` completed 10 tasks and 250 plans.

### Frozen round-2 decision

The cross-dataset ranker was run once after all 50 examples completed. The
frozen baseline and selected candidate are:

| policy | optional locks | locks/example | precision (Wilson 95% lower) | exact A0 | harm | NFE gain |
|---|---:|---:|---:|---:|---:|---:|
| baseline `c0.95 / age0 / steps<2` | 77 | 1.54 | 1.000 (0.952) | 1.000 | 0.000 | 0.0241 |
| selected `c0.90 / age4 / steps<8` | 176 | 3.52 | 1.000 (0.979) | 1.000 | 0.000 | 0.0550 |
| rejected loose `c0.90 / age0 / steps<4` | 182 | 3.64 | 0.989 (0.961) | 0.980 | 0.000 | 0.0569 |

The selected policy is intentionally simple: rank by confidence, require
confidence at least `0.90`, make no optional commitment during the first four
active updates of a block, and allow optional commitments on block steps 4--7.
It uses no semantic KL threshold because no KL-gated arm matched its safe
coverage on this development screen. The wider step window and the active age
must be treated as one selected policy; this sweep does not estimate their
independent causal effects.

The exact candidate is frozen in `configs/semantic_lock_v2_confirmation.json`.
Untouched confirmation uses GSM8K offset 270 (50 examples), MATH500 offset 410
(50 examples), and HumanEval offset 40 (20 tasks). These thresholds will not be
changed after confirmation results are observed.

## Untouched confirmation result

Jobs `185710` (GSM8K), `185727` (MATH500), and `185734` (HumanEval) completed
all 120 held-out examples. Integrity checks found 120 A0 records, two A1 files
per dataset, and exactly 240 saved plans. The development candidate did **not**
confirm:

| policy | optional locks | locks/example | precision (Wilson 95% lower) | exact A0 | harm | NFE gain | runtime gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| frozen v1 baseline | 177 | 1.475 | 1.000 (0.979) | 1.000 | 0.000 | 0.0230 | 0.0229 |
| delayed development candidate | 455 | 3.792 | 0.989 (0.975) | 0.975 | 0.000 | 0.0592 | 0.0585 |

The candidate changed two of 50 GSM8K sequences and one of 50 MATH500
sequences; HumanEval remained exact on 20/20. There were five wrong optional
locks in total. All three changed sequences preserved their normalized answer
and benchmark correctness, so an answer-only evaluation would incorrectly
declare the policy safe.

The failed locks do not support a single posterior-drift cutoff. Four GSM8K
errors had adjacent-step KL from `0.032` to `0.161`, but the MATH500 error had
KL `2.6e-5`, confidence `0.951`, block-local run length `5`, and low context
volatility. Thus waiting for four active updates removes the original stale
inactive-block failure but does not make commitment universally safe.

The delayed candidate is rejected under the predeclared exact-A0 and lock-
precision criteria. `semantic_lock_v1_baseline.json` remains the accepted
semantic policy. The failed confirmation is retained as experiment provenance;
it must not be presented as the PhaseLock semantic algorithm. Any future v3
controller must be developed as a new round and evaluated on a new holdout,
rather than retuning this confirmation result.
