# Official Elastic-Cache Evaluation - 2026-09-07

## Bottom Line

Elastic-Cache produced substantially different outputs from matched full
recomputation while preserving nearly the same aggregate accuracy. It also
reduced the upstream layer-recompute counter substantially and achieved a real,
but much smaller, wall-clock speedup on this hardware.

This is stronger evidence than the earlier Polestar proxy because the cached
arm uses the authors' actual Elastic-Cache LLaDA generator and model code.

## Implementation Provenance

- Official repository: [VILA-Lab/Elastic-Cache](https://github.com/VILA-Lab/Elastic-Cache)
- Paper: [Attention Is All You Need for KV Cache in Diffusion LLMs](https://arxiv.org/abs/2510.14973)
- Pinned upstream commit: `1960d8fc6231205a1ae4ebba3898d475e339f7e1`
- Model: `GSAI-ML/LLaDA-8B-Instruct`

The upstream model initially exceeded the interactive host-memory limit during
ordinary loading. We instantiated it on the meta device and dispatched the
existing checkpoint shards directly to GPU. This changes loading only, not the
model or Elastic-Cache generation algorithm.

## Comparison

Both arms use the upstream Elastic-Cache model implementation and the same:

- prompt and tokenizer;
- confidence threshold `0.9`;
- sliding window length `16`;
- generation length `256`;
- cache trigger threshold `gamma=0.9`;
- one tracked most-attended token;
- EOS behavior.

The two arms are:

- **Full recomputation:** the complete sequence passes through every layer at
  every denoising iteration.
- **Elastic-Cache:** the official generator reuses cached states and adaptively
  chooses where deeper-layer recomputation restarts.

This compares the end-to-end cache policy, not a single-token intervention.

## Quality Results

Intervals are 10,000-replicate example bootstrap intervals.

| Metric | Level-5 MATH, 54 examples | HumanEval, 70 tasks |
| --- | ---: | ---: |
| Generated tokens changed | 54/54, 100% [100, 100] | 46/70, 65.7% [54.3, 75.7] |
| Answer/program changed | 43/54, 79.6% [68.5, 88.9] | 45/70, 64.3% [52.9, 75.7] |
| Baseline accuracy | 5/54, 9.3% [1.9, 16.7] | 41/70, 58.6% [47.1, 70.0] |
| Elastic accuracy | 5/54, 9.3% [1.9, 16.7] | 40/70, 57.1% [45.7, 68.6] |
| Accuracy delta | 0.0 pp [-7.4, 7.4] | -1.4 pp [-11.4, 8.6] |
| Any correctness change | 4/54, 7.4% [1.9, 14.8] | 13/70, 18.6% [10.0, 28.6] |
| Correct -> wrong | 2/54 | 7/70 |
| Wrong -> correct | 2/54 | 6/70 |

The paired correctness sign test is `p=1.0` on both datasets. The samples do not
show a directional accuracy loss, but they do show substantial per-example
instability hidden by the aggregate accuracy.

### Concrete Level-5 transitions

Correct to wrong:

- `math500-l5-444`: `1/2` changed to `pi/4`; gold is `1/2`.
- `math500-l5-473`: `7` changed to `5`; gold is `7`.

Wrong to correct:

- `math500-l5-327`: `0` changed to `15`; gold is `15`.
- `math500-l5-400`: `14/2` changed to `14/3`; gold is `14/3`.

These examples show why equal aggregate accuracy does not mean output-equivalent
or per-example-safe caching.

## Efficiency Results

| Metric | Level-5 MATH | HumanEval |
| --- | ---: | ---: |
| Mean baseline NFE | 111.8 [105.3, 118.8] | 31.9 [27.5, 36.6] |
| Mean Elastic NFE | 125.1 [117.1, 133.4] | 32.2 [28.0, 36.5] |
| Elastic - baseline NFE | +13.2 [7.1, 19.6] | +0.3 [-2.1, 2.5] |
| Relative layer-recompute count | 10.8% [10.0, 11.6] | 22.8% [20.6, 25.2] |
| Measured wall-clock speedup | 1.31x [1.24, 1.40] | 1.49x [1.37, 1.62] |

`Relative layer-recompute count` weights the upstream per-example
`num_computed / total_computed` counter by Elastic NFE and compares it with the
full-recompute layer count. It is not a FLOP measurement: it does not account
for sequence length, attention shape, memory traffic, or cache bookkeeping.

The gap between roughly 77-89% fewer layer recomputations and only 1.31-1.49x
measured speedup supports the project's utility distinction: algorithmic work
reduction does not translate proportionally into runtime improvement.

## Interpretation

### What is supported

1. Actual adaptive KV caching can substantially change exact generated output.
2. Aggregate accuracy can remain nearly constant while individual correctness
   changes in both directions.
3. Elastic-Cache produced a real speedup on this RTX 5000 Ada setup.
4. NFE alone is misleading: Level-5 Elastic used more denoising iterations but
   still ran faster by recomputing far fewer layers.
5. Compute-proxy savings and measured speedup are different quantities.

### What is not supported

1. The results do not show systematic accuracy harm; both accuracy-delta
   intervals include zero.
2. They do not reproduce the paper's reported throughput numbers or benchmark
   protocol.
3. They do not establish that every Elastic cache decision is unsafe.
4. They do not isolate which particular cached token or layer caused each
   output change.

## Novelty Implication

Elastic-Cache further crowds any broad claim that attention-aware or
layer-aware KV refresh is new. PhaseLock should not be presented as the first
method to distinguish stable and stale cache behavior.

The remaining distinction is causal and decision-specific: PhaseLock separates
token commitment, reference freezing, and compute utility, then evaluates exact
output, answer/program, correctness, and runtime consequences separately. The
Elastic result helps that framing because aggregate accuracy hides 4 Level-5
and 13 HumanEval correctness transitions, while the compute counter greatly
overstates realized wall-clock gain.

## Limitations

- The cached arm is official Elastic-Cache, but the matched full-recompute arm
  is our adapter around the same upstream model implementation.
- The official model implementation computes attention-monitoring information;
  this is not a vanilla optimized LLaDA latency baseline.
- Runs use this repository's prompts, zero-shot task slices, and symbolic/code
  scorers rather than the paper's complete lm-eval few-shot protocol.
- HumanEval covers 70 of 164 tasks; Level-5 covers the held-out 54-example
  partition.
- Timing uses one RTX 5000 Ada GPU and one execution order, with full
  recomputation followed by Elastic for each example.
- One deterministic decode was used; no seed or hardware replication was run.
- The layer-recompute counter is not FLOPs.

## Artifacts

- Level-5 run: `results/elastic_cache_official_l5g256_54_20260907/`
- Level-5 bootstrap: `results/elastic_cache_official_reanalysis_l5g256_54_20260907/`
- HumanEval run: `results/elastic_cache_official_humaneval_g256_70_20260907/`
- HumanEval bootstrap: `results/elastic_cache_official_reanalysis_humaneval_g256_70_20260907/`
- Determinism smoke: `results/elastic_cache_official_smoke_l5g64_1_repeat_20260907/`
