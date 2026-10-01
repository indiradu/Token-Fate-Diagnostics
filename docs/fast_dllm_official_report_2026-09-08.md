# Fast-dLLM v1 Component Evaluation - 2026-09-08

## Bottom Line

Fast-dLLM combines two different acceleration ideas:

1. commit several high-confidence tokens in one model call;
2. reuse cached K/V representations instead of recomputing them.

Our matched component test shows that these two operations have very different
effects. Confidence-based parallel commitment changed some outputs, but it was
much less disruptive than caching. Adding a prefix cache changed 53/54 Level-5
MATH outputs and 45/70 HumanEval outputs, even when the token-commitment rule
was otherwise held fixed.

Aggregate accuracy did not decrease reliably. Cache-induced correctness changes
went in both directions. The supported conclusion is therefore:

> Committed token identity is not a certificate that its cached representation
> is functionally equivalent to continued recomputation.

This uses the official Fast-dLLM v1 LLaDA generators inside this repository's
local matched harness. It is not a reproduction of the paper's benchmark
accuracy.

## Provenance And Protocol

- Official repository: [NVlabs/Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
- Paper: [Fast-dLLM: Training-free Acceleration of Diffusion LLM by Enabling KV Cache and Parallel Decoding](https://arxiv.org/abs/2505.22618)
- Pinned upstream commit: `a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b`
- Upstream implementation: `v1/llada`
- Model: `GSAI-ML/LLaDA-8B-Instruct`, bfloat16
- Generation length: 256
- Block length: 32
- Parallel-commitment confidence threshold: 0.9
- Temperature: 0
- Hardware: one NVIDIA RTX 5000 Ada per run
- Tasks: 54 held-out Level-5 MATH examples and 70 HumanEval tasks

The model was instantiated on the meta device and existing checkpoint shards
were dispatched directly to GPU. This avoids host-memory exhaustion and does
not change the upstream model or generation algorithms.

The repository's prompts and symbolic/code scorers were used so these results
are comparable with the preceding PhaseLock strategy tests. The paper's
few-shot `lm-eval` protocol was not used.

## Five Arms

| Arm | Commitment schedule | Representation reuse |
| --- | --- | --- |
| `fixed_full` | Fixed quota, 256 total iterations | None; full recomputation |
| `prefix_fixed` | Same fixed quota | Official prefix K/V cache |
| `parallel_full` | Confidence threshold 0.9 | None; full recomputation |
| `prefix_parallel` | Confidence threshold 0.9 | Official prefix K/V cache |
| `dual_parallel` | Confidence threshold 0.9 | Official prefix and masked-suffix cache |

These direct comparisons answer different questions:

| Comparison | What changes |
| --- | --- |
| `fixed_full -> parallel_full` | Commitment rule only |
| `fixed_full -> prefix_fixed` | Prefix cache only |
| `prefix_fixed -> prefix_parallel` | Commitment rule under the same prefix cache |
| `parallel_full -> prefix_parallel` | Prefix cache under the same confidence rule |
| `prefix_parallel -> dual_parallel` | Extra dual-cache reuse |

The thresholded full and prefix generators receive nominal `steps=8`, as in
the upstream launchers, but their official code loops until each block is fully
decoded. They are not eight-forward-pass methods. Their realized NFE is
reported below.

The upstream GSM8K launcher gives dual cache a 256-step budget, while its
HumanEval launcher gives it 8. The dual generator uses a bounded loop, so the
short setting can stop before every mask is removed. We used 256 for dual cache
on both datasets, allowing up to 32 iterations per block. All outputs finished
with zero masks.

## Quality Results

### Level-5 MATH

| Comparison | Output changed | Answer changed | Accuracy | Correct -> wrong | Wrong -> correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Commitment only | 9/54 (16.7%) | 4/54 (7.4%) | 5/54 -> 5/54 | 0 | 0 |
| Prefix cache only | 53/54 (98.1%) | 37/54 (68.5%) | 5/54 -> 6/54 | 3 | 4 |
| Commitment within prefix cache | 3/54 (5.6%) | 2/54 (3.7%) | 6/54 -> 6/54 | 0 | 0 |
| Add prefix cache to parallel | 53/54 (98.1%) | 38/54 (70.4%) | 5/54 -> 6/54 | 3 | 4 |
| Prefix+parallel -> dual+parallel | 52/54 (96.3%) | 37/54 (68.5%) | 6/54 -> 4/54 | 3 | 1 |

Key 95% example-bootstrap intervals:

- Commitment-only output change: 7.4% to 27.8%.
- Commitment-only answer change: 1.9% to 14.8%.
- Prefix-cache-only output change: 94.4% to 100%.
- Prefix-cache-only answer change: 55.6% to 79.6%.
- Prefix-cache-only accuracy delta: -7.4 to +11.1 percentage points.
- Dual versus prefix-parallel accuracy delta: -11.1 to +3.7 points.

### HumanEval

For code, answer/program change means the extracted executable program changed.

| Comparison | Output changed | Program changed | Accuracy | Correct -> wrong | Wrong -> correct |
| --- | ---: | ---: | ---: | ---: | ---: |
| Commitment only | 17/70 (24.3%) | 17/70 (24.3%) | 38/70 -> 39/70 | 0 | 1 |
| Prefix cache only | 45/70 (64.3%) | 44/70 (62.9%) | 38/70 -> 42/70 | 3 | 7 |
| Commitment within prefix cache | 9/70 (12.9%) | 9/70 (12.9%) | 42/70 -> 41/70 | 1 | 0 |
| Add prefix cache to parallel | 45/70 (64.3%) | 44/70 (62.9%) | 39/70 -> 41/70 | 4 | 6 |
| Prefix+parallel -> dual+parallel | 39/70 (55.7%) | 38/70 (54.3%) | 41/70 -> 39/70 | 6 | 4 |

Key 95% example-bootstrap intervals:

- Commitment-only program change: 14.3% to 34.3%.
- Prefix-cache-only program change: 51.4% to 74.3%.
- Prefix-cache-only accuracy delta: -2.9 to +14.3 percentage points.
- Prefix cache added to parallel accuracy delta: -5.7 to +11.4 points.
- Dual versus prefix-parallel accuracy delta: -11.4 to +5.7 points.

All paired correctness sign tests were non-significant. These runs show
per-example instability, not a directional accuracy loss.

## Accuracy By Arm

| Arm | Level-5 MATH | HumanEval |
| --- | ---: | ---: |
| `fixed_full` | 5/54 (9.3%) | 38/70 (54.3%) |
| `prefix_fixed` | 6/54 (11.1%) | 42/70 (60.0%) |
| `parallel_full` | 5/54 (9.3%) | 39/70 (55.7%) |
| `prefix_parallel` | 6/54 (11.1%) | 41/70 (58.6%) |
| `dual_parallel` | 4/54 (7.4%) | 39/70 (55.7%) |

Equal or higher aggregate accuracy does not imply output preservation. For
example, prefix caching raised HumanEval accuracy from 38 to 42 while changing
10 individual correctness labels: 3 passing programs failed and 7 failing
programs passed.

## Concrete Examples

### Confidence-only commitment changed an answer but not correctness

For `math500-l5-22`, fixed full recomputation produced answer `2`, while
confidence-parallel full recomputation produced `1`. The gold answer was `5`,
so the answer changed but both outputs remained wrong.

### Prefix caching changed a correct answer to a wrong answer

For `math500-l5-24`, full recomputation produced the correct answer `10`.
Prefix caching with the same fixed commitment schedule produced `3.5`.

The reverse also occurred. For `math500-l5-100`, full recomputation produced
`505`, while prefix caching produced the correct `145 degrees`.

### Confidence-only commitment fixed one program

For `HumanEval/16`, fixed full recomputation counted `set(string)`, which is
case-sensitive and failed. Confidence-parallel full recomputation used
`set(string.lower())` and passed. This was the only correctness change in the
commitment-only HumanEval comparison.

### Prefix caching changed a passing program into a failing one

For `HumanEval/43`, full recomputation correctly checked whether `-num` had
already been seen. Prefix caching with the same fixed schedule instead checked
whether `num` had been seen, changing a passing implementation into a failing
one.

### Dual cache introduced a specific code error

For `HumanEval/11`, prefix-cache-plus-parallel compared characters directly and
preserved leading zeros in binary XOR. Dual-cache-plus-parallel converted the
strings to integers and back, dropping leading zeros and failing the tests.

These examples explain why output change, answer/program change, and correctness
change must be reported separately.

## Efficiency Results

`Relative input positions` is the number of `input_ids` positions passed to
model forwards divided by the fixed-full count. It is a transparent query-work
proxy, not FLOPs: cached key length, attention cost, memory traffic, softmax,
and cache bookkeeping are not included.

| Arm | Level-5 mean NFE | Level-5 relative input positions | Level-5 observed speedup | HumanEval mean NFE | HumanEval relative input positions | HumanEval observed speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `fixed_full` | 256.0 | 100.0% | 1.00x | 256.0 | 100.0% | 1.00x |
| `prefix_fixed` | 256.0 | 37.7% | 1.52x | 256.0 | 37.7% | 1.54x |
| `parallel_full` | 113.3 | 45.0% | 2.23x | 32.9 | 13.3% | 7.51x |
| `prefix_parallel` | 113.7 | 19.2% | 3.31x | 35.0 | 7.1% | 10.08x |
| `dual_parallel` | 125.2 | 6.8% | 2.90x | 37.1 | 4.0% | 9.27x |

Dual cache passed substantially fewer input positions than prefix cache plus
parallel commitment, but it was slower on both datasets:

- Level-5: `0.878x` relative speed, meaning about 12.2% slower.
- HumanEval: `0.920x` relative speed, meaning about 8.0% slower.

This supports PhaseLock's compute-utility distinction: a more aggressive work
proxy does not guarantee a faster implementation.

The bootstrap timing intervals quantify variation across the sampled tasks.
They are not repeat-run or hardware-performance uncertainty intervals because
each task was timed once.

## What This Means For The Premises

### Supported: commitment and reference reuse are different decisions

The cleanest comparison is prefix cache only. Token commitment follows the same
fixed schedule in both arms. The previous blocks contain already committed
token identities, but reusing their cached K/V representations changed nearly
every Level-5 trajectory and about two-thirds of HumanEval programs.

This is system-level evidence that stable token identity does not imply a
functionally settled representation.

### Supported: the cache effect dominates here

On Level-5, commitment-only changed 9 outputs, while adding prefix caching under
the same confidence rule changed 53. Under an existing prefix cache, switching
from fixed to confidence commitment changed only 3 outputs.

HumanEval shows the same ordering, though less extremely: 17 commitment-only
changes versus 45 changes from adding prefix caching to the parallel decoder.

### Not tested directly

Fast-dLLM does not measure hidden-state drift magnitude in this experiment.
Therefore, these results do not establish:

- that high measured representation drift predicts more harm than low drift;
- that representation convergence is sufficient or insufficient for safe
  commitment;
- which cached token, layer, or step caused a changed answer.

Those questions still require event-level drift measurements and controlled
interventions.

## Novelty Implication

Fast-dLLM already publishes baseline, cache-only, parallel-only, and combined
launchers. PhaseLock should not claim that separating caching from parallel
commitment at the system-ablation level is new. Fast-dLLM also directly crowds
any broad novelty claim around KV caching or confidence-based parallel decoding.

The experiment helps a narrower and stronger claim:

> Existing acceleration methods expose aggregate quality/speed tradeoffs, while
> PhaseLock treats token commitment, reference freezing, and compute removal as
> separate per-event safety decisions and evaluates their causal consequences.

The paired results make that distinction concrete, but PhaseLock must still
show that its own gates reduce harmful changes at matched realized speed or
saved work.

## Limitations

- Only the generation/model code is official Fast-dLLM. Prompts, task slices,
  scoring, timing, and pairwise analysis are local.
- The Level-5 and HumanEval runs used different GPUs of the same model.
- Each task used one deterministic decode and one timing observation.
- Exact output comparison covers the full 256-token response, including text
  after a semantically complete answer; answer/program comparison is therefore
  the more task-relevant endpoint.
- HumanEval covers 70 of 164 tasks.
- The dual-cache step budget follows the upstream GSM8K launcher rather than
  the inconsistent shorter HumanEval launcher.
- Input-token-position counts are not FLOPs or total cache cost.
- Accuracy-transition counts are small, and all directional intervals include
  zero.

## Validity And Artifacts

Strict artifact validation confirmed:

- 54 unique Level-5 examples and 70 unique HumanEval tasks;
- five arms per task, for 620 complete outputs;
- 256 response tokens per output;
- zero unresolved masks;
- no missing required values;
- NFE exactly equal to observed model-forward count in every arm.

Artifacts:

- Level-5 run: `results/fast_dllm_official_l5g256_54_20260908/`
- Level-5 reanalysis: `results/fast_dllm_official_reanalysis_l5g256_54_20260908/`
- HumanEval run: `results/fast_dllm_official_humaneval_g256_70_20260908/`
- HumanEval reanalysis: `results/fast_dllm_official_reanalysis_humaneval_g256_70_20260908/`
- Gen64 smoke: `results/fast_dllm_official_smoke_l5g64_1_20260908/`
- Gen256 smoke: `results/fast_dllm_official_smoke_l5g256_1_20260908/`
