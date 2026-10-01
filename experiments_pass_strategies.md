2. Experiments

SureLock / LESS / Polestar / TraceLock / Ada-DLM

Premise Test

Premise A → Semantic/token convergence
does not imply
representation convergence.

Premise B → Representation convergence
does not imply
safe to commit/lock.

Investigate:

Compute might depend on …
Final answer [“proved/prove”] could change.
Premise C → Representation drift influences answer changes and quality.
High drift → higher/more change
Low drift → fewer changes

## Completed: Polestar-Cache Proxy, Level-5 Gen256

Date: 2026-09-07

### Question

When a Polestar-style attention-drift signal identifies a cache refresh event,
does refreshing the hidden reference reduce the harm caused by stale reuse?

### What We Compared

- `YC`: normal full-recompute baseline.
- `Y_stale`: keep using the old row cached at block entry.
- `Y_refresh`: at the same step and for the same token, update to the current
  baseline row and then hold that row.

The test used the strongest per-head attention-KL event in each of the 54
held-out Level-5 MATH examples. Generation length was 256 and block length was
32. Rows were clamped at layers 16, 24, and 32.

### Validity

- 54 examples and 54 matched stale/refreshed pairs.
- 108 intervention replays.
- Full pre-intervention replay valid: 108/108.
- Selected target token unchanged: 108/108.
- 78,624 selected refresh events were logged and retained; top-KL selection was
  not truncated by the event reservoir.
- Runtime: 3,629.5 seconds, or 60.5 minutes on one GPU.

### Results

| Metric | Stale row | Refreshed row |
| --- | ---: | ---: |
| Downstream output changed | 11/54 (20.4%) | 14/54 (25.9%) |
| Mathematical answer changed | 4/54 (7.4%) | 1/54 (1.9%) |
| Accuracy | 5/54 (9.3%) | 4/54 (7.4%) |
| Correct -> wrong | 0/54 | 0/54 |
| Wrong -> correct | 1/54 | 0/54 |

Paired comparison:

- Output changes: 3 stale-only versus 6 refresh-only; exact sign-test
  `p = 0.508`.
- Answer changes: 3 stale-only versus 0 refresh-only; exact sign-test
  `p = 0.25`.
- Mean downstream token-count difference, stale minus refresh: `-3.15` tokens.

### Interpretation

The result is mixed:

- Refreshing reduced answer changes from 4 to 1.
- Refreshing did not reduce token-level output changes; it increased them from
  11 to 14.
- Accuracy did not improve after refreshing.
- Neither paired difference is statistically decisive.

Therefore this experiment does not establish that the proxy refresh helps or
hurts. It also does not support a simple claim that high attention KL identifies
rows whose refresh will preserve the exact output.

### Scope Guardrail

This is a token-level, per-head attention-KL, single-refresh hidden-row proxy.
It is not full Polestar-Cache. It omits centroid proxy attention, quantization,
repeated sparse per-layer KV updates, suffix update packets, alternate full
cache refreshes, and the optimized Triton runtime. It cannot support Polestar
TPF, TPS, speedup, or full-quality claims.

Artifacts:

- `results/polestar_cache_proxy_topkl_l5g256_54x1_20260907/`
- Proxy design and initial smoke report:
  `docs/polestar_cache_proxy_smoke_report_2026-09-07.md`

## Completed: Official Elastic-Cache, Gen256

Elastic-Cache was tested using the official LLaDA generator and model code at
upstream commit `1960d8fc6231205a1ae4ebba3898d475e339f7e1`. The comparison arm
uses full recomputation with the same model, confidence threshold, window,
prompt, and EOS logic.

| Metric | Level-5 MATH (54) | HumanEval (70) |
| --- | ---: | ---: |
| Output changed | 100.0% | 65.7% |
| Answer/program changed | 79.6% | 64.3% |
| Baseline accuracy | 9.3% | 58.6% |
| Elastic accuracy | 9.3% | 57.1% |
| Correct -> wrong | 2 | 7 |
| Wrong -> correct | 2 | 6 |
| Relative layer-recompute count | 10.8% | 22.8% |
| Measured speedup | 1.31x | 1.49x |

Interpretation: Elastic-Cache changes exact trajectories frequently while
roughly preserving aggregate accuracy. Individual correctness still changes in
both directions. The large gap between its layer-recompute counter and measured
speedup supports treating compute utility as separate from reference safety.

The layer-recompute ratio is not FLOPs. The timing baseline uses the upstream
attention-monitoring model rather than vanilla optimized LLaDA. See
`docs/elastic_cache_official_report_2026-09-07.md` for confidence intervals and
limitations.

## Next Strategies To Test

Do not test every nearby paper indiscriminately. The next method should add a
new control or mechanism rather than repeat the token-stability question already
covered by SureLock and LESS.

## Completed: Fast-dLLM v1 Component Test, Gen256

Date: 2026-09-08

We tested the official Fast-dLLM v1 LLaDA generators at upstream commit
`a9b81e4caa240c8cad4f7dc1889ff4852a0fca5b` inside our local matched harness.
Five arms separated fixed versus confidence commitment and full recomputation
versus prefix or dual caching.

| Direct comparison | Level-5 output / answer changes | HumanEval output / program changes |
| --- | ---: | ---: |
| Commitment only | 9/54 / 4/54 | 17/70 / 17/70 |
| Prefix cache only | 53/54 / 37/54 | 45/70 / 44/70 |
| Commitment within prefix cache | 3/54 / 2/54 | 9/70 / 9/70 |
| Add prefix cache to parallel | 53/54 / 38/54 | 45/70 / 44/70 |
| Prefix+parallel -> dual+parallel | 52/54 / 37/54 | 39/70 / 38/70 |

Accuracy by arm:

| Arm | Level-5 | HumanEval |
| --- | ---: | ---: |
| Fixed full recomputation | 5/54 | 38/70 |
| Prefix cache, fixed commitment | 6/54 | 42/70 |
| Parallel commitment, full recomputation | 5/54 | 39/70 |
| Prefix cache + parallel commitment | 6/54 | 41/70 |
| Dual cache + parallel commitment | 4/54 | 39/70 |

The cache comparisons changed correctness in both directions, but no accuracy
delta was statistically resolved. The important result is the component
separation: confidence commitment alone was much less disruptive than K/V
reuse, especially on Level-5. Committed token identity was not a certificate
that its cached representation was equivalent to continued recomputation.

Observed speedups versus fixed full recomputation were 2.23x/7.51x for
confidence commitment only, 3.31x/10.08x for prefix cache plus commitment, and
2.90x/9.27x for dual cache plus commitment on Level-5/HumanEval. Dual cache used
fewer input query positions but ran 12.2% and 8.0% slower than prefix cache plus
parallel commitment, reinforcing the distinction between a work proxy and
realized runtime.

This is not a reproduction of the paper's benchmark accuracy; only the
upstream model/generators are official. Full report:
`docs/fast_dllm_official_report_2026-09-08.md`.

### Deferred: d2Cache

**Why deferred:** d2Cache uses a two-stage adaptive token-selection policy for
KV updates, but it does not directly study token-identity commitment. It remains
relevant to PhaseLock's reference-freeze stage, but Elastic-Cache already gives
us one official adaptive KV-cache baseline. The current priority is the
commitment/reference boundary rather than another cache selector.

Use the same quality and runtime endpoints and compare at matched realized
speedup or matched cache work if this method is revisited.

Official code: `https://github.com/Kamichanw/d2Cache`.

### Priority 1: Dynamic-dLLM

**Why next:** Dynamic-dLLM combines Dynamic Cache Updating with Adaptive
Parallel Decoding. It is a strong end-to-end baseline, but it changes cache and
token commitment together, so it is harder to identify which component caused
an output or accuracy change. Test it after Fast-dLLM establishes the simpler
cache-plus-confidence control, and use component ablations where available.

Official code: `https://github.com/TianyiWu233/DYNAMIC-DLLM`.

### Priority 2: TraceLock - Separate Commitment Test

TraceLock learns token-commitment timing from completed trajectories. It should
be tested for commitment risk (`Y_commit` versus baseline), not treated as a KV
cache or reference-freeze strategy. It remains useful for defending the claim
that PhaseLock is more than another future-token predictor.

Official code: `https://github.com/BobSun98/TraceLock`.

#### Where TraceLock Acts

The first three strategies above are cache-oriented, with one qualification:

- **Fast-dLLM v1:** KV caching plus parallel token commitment.
- **d2Cache:** adaptive token-level KV caching.
- **Dynamic-dLLM:** dynamic caching plus adaptive parallel commitment.

TraceLock acts at an earlier decision boundary. It decides which still-masked
token proposals become permanently committed to the visible sequence. It does
not freeze hidden/KV references or remove committed rows from per-step model
computation.

PhaseLock separates three decisions:

1. token-identity commitment;
2. hidden/KV reference freezing;
3. row-compute removal when the runtime gain is worthwhile.

TraceLock overlaps only decision 1. It is therefore a direct baseline for the
PhaseLock commitment-risk estimator, but not for the reference-risk or
compute-utility gates.

#### Label Difference

TraceLock trains on a future-stability label:

```text
S(i,t) = 1[current proposed token equals the final baseline-trace token]
```

PhaseLock's stronger commitment-safety label is interventional:

```text
C(i,t) = 1[forcing commitment at this step preserves the final output]
```

`S=1` does not logically imply `C=1`. A token may already equal its final
baseline value while forcing it early changes other commitment decisions, the
reasoning trajectory, or the task answer. This observational-versus-causal gap
is the main scientific comparison with TraceLock.

#### Released Implementation Constraints

The official release at commit
`bc4134c6c6716e9d92719fc582b6770616030840` currently provides:

- a Dream-v0-Instruct-7B pipeline, not a LLaDA pipeline;
- code to generate traces, train TraceLock, and evaluate it;
- a pretrained Dream activation-projection autoencoder;
- no pretrained TraceLock commitment-policy checkpoint.

The authors report that the 7,000-trace reproduction uses approximately 245 GB
for traces and recommends at least 350 GB free space. Their reproduced run used
8 NVIDIA A40 GPUs. A small official smoke uses 8 traces, 20 training steps, and
8 math plus 8 code examples on one GPU, but it validates infrastructure only.

#### Correct Test Plan

Phase 1 - faithful Dream reproduction:

1. Run the official 8-trace/20-step smoke on one GPU.
2. Compare native entropy, Fast-dLM threshold 0.9, and TraceLock.
3. Report average steps, output and answer/program changes, original and
   TraceLock accuracy, and both correctness-transition directions.
4. Do not compare the Dream numbers directly with the LLaDA cache tables.

Phase 2 - causal commitment audit:

1. Record the token-step events admitted by TraceLock.
2. For sampled admitted events, force only that commitment while continuing
   full representation recomputation.
3. Compare the intervention output with the same baseline trajectory.
4. Measure whether future-token agreement `S` actually predicts commitment
   safety `C`.

Any hidden/KV freeze applied at a TraceLock commitment must be labeled a
hypothetical reuse experiment, not published TraceLock behavior.

#### Decision

TraceLock is worth testing because it pressures PhaseLock's first stage and its
learned-token-fate novelty. It should remain separate from the cache benchmark.
Before paying for the full 7,000-trace reproduction, request the trained policy
checkpoint from the authors or run only the official 8-sample infrastructure
smoke. Porting the released controller to LLaDA would require new LLaDA traces,
projection features, and retraining, so it would no longer be a direct official
reproduction.

### Lower Priority Or Deferred

- **Ada-DLM and TACG:** primarily add trajectory-aware commitment signals. LESS
  already showed that a stronger commitment signal is not a reference-freeze
  certificate, so these are partly redundant for the current premise.
- **Polestar full implementation:** revisit if official code becomes available.
  The current result is only a single-refresh hidden-row proxy.
- **More natural-threshold proxy tests:** defer. The next useful comparisons
  should use actual cache implementations and matched quality/speed points.

### Recommended Order

1. Dynamic-dLLM, with cache-only and commitment-only ablations if exposed.
2. TraceLock as a separate commitment-only study.
3. d2Cache only if another cache-selector baseline is still needed.

Across the completed and remaining methods, this set covers the useful decision
boundaries:

- Fast-dLLM: simple KV cache plus confidence-based parallel commitment.
- Dynamic-dLLM: adaptive cache budget plus adaptive parallel commitment.
- TraceLock: learned token commitment without KV/reference freezing.

Stop after these if they establish the commitment/reference separation clearly.
Additional cache methods are useful only if they materially change the
quality-speed frontier.
