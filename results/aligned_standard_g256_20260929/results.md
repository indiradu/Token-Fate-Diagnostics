# Aligned standard-LLaDA comparison

Setting: LLaDA-8B-Instruct BF16, standard low-confidence sampler (block 32, one token per step), 256 output tokens / 256 steps, batch 1, seed 23, 54 MATH Level-5 + 70 HumanEval, one warm-up and three timing repeats with rotating policy order. Correctness uses repeat 1; speed uses paired sums of per-example median latency. Intervals are 10,000-sample bootstraps over examples.

| Policy | MATH /54 | Code /70 | Total /124 | Δ correct vs B0 (pp) [95% CI] | s / example | Speedup vs B0 [95% CI] | Mean calls | Extended TFLOPs | FLOP cut | Outputs changed | Broken / fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 5 | 38 | 43 | +0.00 [+0.00, +0.00] | 16.028 | +0.00% [+0.00, +0.00] | 256.0 | 1,572.0 | 0.00% | 0/124 | 0 / 0 |
| RepKL-freeze-0.0005 | 4 | 39 | 43 | +0.00 [-5.65, +5.65] | 14.503 | +10.52% [+9.26, +11.75] | 256.0 | 1,146.2 | 27.09% | 99/124 | 6 / 6 |
| SureLock-gen-0.005 | 6 | 38 | 44 | +0.81 [-4.84, +6.45] | 14.877 | +7.74% [+6.48, +8.96] | 256.0 | 1,140.4 | 27.45% | 97/124 | 6 / 7 |
| SureLock-prompt-0.005 | 5 | 40 | 45 | +1.61 [-4.84, +8.06] | 12.520 | +28.02% [+24.35, +31.87] | 256.0 | 587.4 | 62.63% | 106/124 | 8 / 10 |
| TraceLock-0.95 | 6 | 37 | 43 | +0.00 [-7.26, +7.26] | 5.202 | +208.13% [+177.20, +248.14] | 79.6 | 501.1 | 68.12% | 109/124 | 10 / 10 |
| Confidence-0.9 | 5 | 39 | 44 | +0.81 [+0.00, +2.42] | 4.312 | +271.75% [+232.31, +324.21] | 67.9 | 426.4 | 72.88% | 26/124 | 0 / 1 |

Policies: B0 is the standard LLaDA sampler. RepKL-freeze-0.0005 freezes a generated token's all-layer state and skips its rows once final-layer representation KL ≤ 0.0005. SureLock-gen/-prompt use vocabulary KL ≤ 0.005 plus the top-20% confidence gate; -gen freezes generated tokens only, -prompt also freezes prompt rows. Both freeze only after the token has been seen unmasked, and never EOS/EOT. TraceLock-0.95 is the trained controller with its own 32-token sliding window. Confidence-0.9 commits every block position with confidence ≥ 0.9 (else the single most confident), Fast-dLLM-style, inside the standard blocks.

Parity with earlier identical configurations is in analysis/parity.json; per-dataset bootstrap tables are in analysis/paired/summary.md; FLOPs follow the SureLock §2.4 convention (2 FLOPs/MAC, GEMMs only) plus the vocabulary head and controller.
