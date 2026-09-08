# Polestar-Cache Proxy Smoke Report - 2026-09-07

## Bottom Line

The Polestar attention-drift signal is technically measurable in this LLaDA
environment, and the paired stale-versus-refreshed intervention is valid.
However, the four-example top-KL smoke produced no output change in either arm,
so it gave no evidence that one selected refresh improved the result. We should
not spend the compute for a full 54-example run of this proxy.

This is **not a test of full Polestar-Cache**. The authors have not linked a
code release, and the paper's full method depends on a custom dual-cache path,
spherical hidden-state clustering, centroid proxy attention, quantization,
sparse layer-wise update packets, suffix updates, and Triton optimizations.

## What We Tested

The test keeps the parts of the
[Polestar paper](https://arxiv.org/abs/2607.14107) that can be reproduced
without inventing the missing cache runtime:

- LLaDA-8B-Instruct;
- generation length 256 and block length 32;
- a two-block local prefix window;
- a refresh trigger after more than three decoded tokens;
- attention-distribution KL measured independently per head and then averaged;
- refresh selection from the highest-drift half of eligible prefix tokens.

The paper selects the top four of eight hidden-state clusters. This proxy is
more fine-grained: it uses exact full-recompute attention for individual tokens
and selects the top half directly. It only studies committed generated prefix
tokens; it does not model prompt or suffix cache entries.

At each online-selected refresh event, we construct a matched pair:

- `YC`: the ordinary full-recompute confidence decode;
- `Y_stale`: starting at the refresh step, keep serving the row cached at block
  entry;
- `Y_refresh`: at the same step for the same token, update to the current
  baseline row and then hold that row.

Rows are clamped at layers 16, 24, and 32. This isolates the value of one
refresh under the existing hidden-row proxy. It does not simulate Polestar's
complete sequence of sparse KV refreshes.

## Validation

The four-example Level-5/gen256 run logged:

- 51,584 finite local-prefix attention-KL rows;
- 5,824 online-selected refresh events;
- all 5,824 events retained, so top-KL selection was not reservoir-truncated;
- one strongest event per example;
- four matched pairs and eight intervention replays;
- exact pre-intervention replay for 8/8 arms;
- unchanged target tokens for 8/8 arms.

The selected attention-KL values were 1.659, 1.693, 2.619, and 3.441. The
stale-to-refresh row movement was nonzero in every pair, with mean relative L2
movement ranging from 0.121 to 0.725. The null result therefore was not caused
by selecting identical rows or degenerate KL values.

## Results

| Metric | Stale block-entry row | Refreshed current row |
| --- | ---: | ---: |
| Interventions | 4 | 4 |
| Downstream output changed | 0/4 | 0/4 |
| Answer changed | 0/4 | 0/4 |
| Original baseline accuracy | 1/4 | 1/4 |
| Modified accuracy | 1/4 | 1/4 |
| Correct -> wrong | 0/4 | 0/4 |
| Wrong -> correct | 0/4 | 0/4 |

The paired stale-minus-refresh differences were all zero. In this smoke, the
old rows were already harmless, so refreshing them could not demonstrate a
benefit.

## Interpretation

What the smoke establishes:

1. Per-head attention KL can be captured without changing the baseline output.
2. The published decoded-token refresh schedule produces non-degenerate online
   events in the Level-5/gen256 regime.
3. Stale and refreshed references can be compared at one exact causal boundary
   with full-prefix replay validation.
4. The four strongest events in these four examples do not support a positive
   single-refresh effect.

What the smoke does not establish:

1. It does not show that Polestar-Cache succeeds or fails.
2. It does not test centroid proxy attention or top-cluster assignment.
3. It does not perform repeated sparse per-layer KV refreshes.
4. It does not measure cache-row savings, TPF, TPS, or wall-clock speedup.
5. With only four pairs, it cannot estimate a population effect or meaningful
   confidence interval.

The null can mean that these particular cached rows were not causally
influential even though their own attention distributions moved. It can also
mean that a single hidden-row refresh is a poor approximation to Polestar's
layer-wise, repeated KV updates. The experiment cannot distinguish those
explanations.

## Decision

**Keep the instrumentation; stop this proxy before the full sweep.**

The predeclared scale criterion was a positive stale-minus-refresh downstream
effect. The strongest-event smoke produced exactly zero, so a 54-example run of
the same proxy is not justified.

A proper Polestar-Cache evaluation now requires either:

1. obtaining the authors' implementation; or
2. building and validating an actual dual-cache, repeated sparse-refresh path
   before measuring quality and runtime.

Until then, results from this proxy must not be described as Polestar accuracy
or efficiency results.

## Artifacts

- Primary smoke:
  `results/polestar_cache_proxy_topkl_smoke_l5g256_4x1_20260907/`
- Initial random-event infrastructure smoke:
  `results/polestar_cache_proxy_smoke_l5g256_2x1_20260907/`
- Attention-capture feasibility probe:
  `results/polestar_attention_capture_smoke_g64_1_20260907/`

## Remaining Risks

- The paper does not specify enough implementation detail for exact runtime
  parity, and no official code repository was found.
- The proxy uses token-level attention rather than centroid proxy attention.
- It freezes hidden rows at three layers rather than updating actual K/V cache
  entries at every layer.
- The stress sample contains only four Level-5 examples.
- Candidate selection deliberately uses the highest-KL event per example, so it
  is a stress test rather than an unbiased effect-rate estimate.
