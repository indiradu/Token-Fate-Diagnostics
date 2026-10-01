# Collaborator methods under the aligned protocol

Same job and GPU as the B0 anchor. Batch 1, 256/256/32, our 124 prompts and our scorer, 1 warm-up + 3 timed repeats (rotating order). Her decoders from commit 24cd51b; timing is a wall timer around each call (includes tokenization, like ours). Intervals: 10,000-sample bootstrap over questions.

| Method | MATH /54 | Code /70 | Total /124 | Δ correct vs B0 (pp) [95% CI] | s / example | Speedup vs B0 [95% CI] | Mean calls | Outputs changed vs B0 | Broken / fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 5 | 38 | 43 | +0.00 [+0.00, +0.00] | 15.535 | +0.00% [+0.00, +0.00] | 256.0 | 0/124 | 0 / 0 |
| collab-dense | 5 | 38 | 43 | +0.00 [+0.00, +0.00] | 16.128 | -3.67% [-3.81, -3.54] | 256.0 | 0/124 | 0 / 0 |
| collab-SureLock-5e-4 | 3 | 31 | 34 | -7.26 [-14.52, -0.81] | 13.053 | +19.02% [+16.39, +21.82] | 256.0 | 111/124 | 14 / 5 |
| collab-SureLock-5e-3 | 4 | 30 | 34 | -7.26 [-14.52, +0.00] | 13.028 | +19.25% [+16.57, +22.11] | 256.0 | 113/124 | 15 / 6 |
| collab-capped-confidence | 5 | 38 | 43 | +0.00 [+0.00, +0.00] | 15.650 | -0.74% [-1.04, -0.43] | 244.9 | 1/124 | 0 / 0 |
| collab-H2-guard90-delay1 | 7 | 40 | 47 | +3.23 [-4.03, +10.48] | 7.479 | +107.73% [+95.76, +121.36] | 143.3 | 106/124 | 8 / 12 |

Parity: same-job B0 vs the aligned campaign's B0: 124/124; her dense vs our B0: 124/124 (details in parity.json).
