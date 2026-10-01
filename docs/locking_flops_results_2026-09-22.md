# FLOP counts for both locking experiments

Completed on 22 September 2026: the full 124-example representation-locking analysis and a separate full 124-example TraceLock FLOP-only pass. Both use the identical 54 MATH Level-5 and 70 HumanEval prompts, LLaDA-8B-Instruct, BF16, seed 23 and 256 output positions.

**Representation KL 0.0005 saves 27.09% of extended matrix-operation FLOPs and previously measured 9.61% speedup. TraceLock 0.95 saves 68.12% of extended matrix-operation FLOPs, including the learned controller; this profiling pass does not establish its answer accuracy or wall-clock speedup.**

## What FLOPs mean here

[SureLock section 2.4](https://arxiv.org/html/2602.06412v1#S2.SS4) counts algorithmic matrix-multiplication operations. We use two FLOPs per multiply-add. Our paper-compatible backbone count includes attention projections, attention score/value products and feed-forward matrices. We separately report an extended count adding the vocabulary head and, for TraceLock, its feature projector and controller.

Both counts exclude elementwise operations (including KL, softmax, normalization, activations and RoPE), cache/memory traffic, and kernel padding. They are not hardware instruction counters. TFLOPs below means trillions of operations per complete generation, not operations per second. Training and model loading are excluded.

## Experiment 1: representation locking

Observe the same token’s final-layer representation at consecutive decoding steps. Softmax its 4,096 features at temperature 1, then gate all-layer representation caching using KL(current || previous). Sparse execution actually skips frozen query, projection, feed-forward and output-head rows. Dense freezing retains the full matrix workload.

All 124 cases, seven policies and three original timing repeats are included. FLOPs use the recorded actual work counters and each prompt’s full sequence length; accuracy uses repeat 1 and timing uses each example’s median of three repeats. The repeated work counters agree.

| Policy | Correct / 124 | Backbone TFLOPs/example | Backbone reduction | Extended TFLOPs/example | Extended reduction | Measured speedup |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 43 | 1,464.924 | 0.00% | 1,571.957 | 0.00% | +0.00% |
| Representation monitoring; no freezing | 43 | 1,464.924 | 0.00% | 1,571.957 | 0.00% | -5.37% |
| Representation KL 0.005; dense freezing | 40 | 1,464.924 | 0.00% | 1,571.957 | 0.00% | -7.31% |
| Representation KL 0; sparse freezing | 44 | 1,185.119 | 19.10% | 1,271.701 | 19.10% | +4.13% |
| Representation KL 0.0005; sparse freezing | 43 | 1,068.145 | 27.09% | 1,146.174 | 27.09% | +9.61% |
| Representation KL 0.005; sparse freezing | 41 | 1,055.263 | 27.96% | 1,132.349 | 27.97% | +10.10% |
| Token-probability KL 0.005; sparse representation freezing | 40 | 1,054.513 | 28.02% | 1,131.545 | 28.02% | +7.74% |

Speedup = 100 × (baseline time / method time − 1). Negative values mean slower. FLOP reduction = 100 × (1 − method FLOPs / baseline FLOPs), using summed counts across paired examples, not the average of individual percentage reductions.

At representation cutoff 0.0005, the correct count stays 43/124, but six baseline errors are fixed and six new errors appear. This is a measured tradeoff on our panel, not answer preservation. The token-probability KL control still freezes representations; it is distinct from TraceLock token commitment.

All E1 methods make 256 model calls per example. Monitoring and dense freezing save no backbone GEMMs but still add runtime overhead, explaining why FLOP reductions and wall-clock speedups differ.

## Experiment 2: TraceLock token commitment

The frozen trained controller uses cutoff 0.95 and a 32-token moving window. Its features come from the last three hidden states and their differences. It commits token identities while keeping backbone representations live. The controller sees the prompt through the moving window’s end, and its Transformer has one additional threshold token. Both details are included in the count.

We replayed all 124 prompts and recorded every controller-call prefix length. The baseline count is the explicitly analytical cost of 256 full-sequence forward calls on each identical prompt, with no learned controller. Fixed-quota confidence decoding has exactly that matrix workload regardless of which token is committed first. The original E2 answer-quality/runtime evaluation remains separate.

| Dataset | Samples | Mean model calls | Backbone reduction | Extended TFLOPs/example | Extended reduction | Projector + controller TFLOPs/example |
|---|---:|---:|---:|---:|---:|---:|
| humaneval | 70 | 42.61 | 82.87% | 269.830 | 82.84% | 0.532 |
| math500 | 54 | 127.54 | 49.12% | 800.953 | 49.03% | 1.573 |
| overall | 124 | 79.60 | 68.18% | 501.125 | 68.12% | 0.985 |

Overall: extended matrix work drops from 1,571.957 to 501.125 TFLOPs per generation. The 0.985 TFLOPs average projector/controller cost is already included. Mean model calls drop from 256 to 79.60. No correctness or speedup value is inferred from this FLOP-only pass.

## Verification

- All original 2,604 E1 records retain their saved checksum. The full 124-case counts were reconstructed, and 14 full-length GPU profiles (two cases × seven policies) matched both the analytical FLOP formulas and the original outputs/work exactly.
- All 124 E2 profiles completed with no remaining masks or token-identity violations. Every controller call has a recorded prefix length; zero representation rows were frozen.
- One math and one coding E2 profile were independently checked against uninstrumented decoding. Their outputs/work and observed matrix counts matched exactly, including fused Transformer and attention operations. Other E2 rows use those validated formulas with their own observed dimensions/counts.
- All 22 focused tests, scoped typing on three scripts and fatal lint passed. Independent architect review approved the formulas and final artifacts. The completed progress sidecars have been reconciled.
- Executed E2 profiling sources are preserved in `results/locking_flops_20260922/source_v1/`. A later maintained-script change only fixes completion-status reporting. No frozen experiment source, checkpoint, original timings or original scoring data was changed.

## Files and reproducibility

Added [accounting](../scripts/locking_flop_accounting.py), [profiling](../scripts/profile_locking_flops.py), and [profile summarization](../scripts/summarize_locking_flop_profiles.py), with three corresponding test files. Counting is isolated from the experiment implementation; no dependencies were added. The reporting simplification is to separate paper-compatible backbone work, added model/controller work, and measured runtime.

- [E1 aggregate counts](../results/representation_kl_last_hidden_20260921_203301/flop_accounting/aggregate_flops.csv)
- [E1 per-example counts](../results/representation_kl_last_hidden_20260921_203301/flop_accounting/percase_flops.csv)
- [E2 aggregate counts](../results/locking_flops_20260922/e2_analysis/aggregate_flops.csv)
- [E2 per-example counts](../results/locking_flops_20260922/e2_analysis/percase_flops.csv)
- [E2 source/checkpoint provenance](../results/locking_flops_20260922/e2_full_profile_v1/provenance.json)
- [Final verification](../results/locking_flops_20260922/final_verification.json)

The results remain specific to this panel and these frozen operating points. Lower algorithmic FLOPs do not prove stable representations, preserved answer quality, or an equal percentage speed gain.
