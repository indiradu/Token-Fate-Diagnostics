# Collaborator methods under the aligned protocol

Same job and GPU as the B0 anchor. Batch 1, 256/256/32, our 124 prompts and our scorer, 1 warm-up + 3 timed repeats (rotating order). Her decoders from commit 24cd51b; timing is a wall timer around each call (includes tokenization, like ours). Intervals: 10,000-sample bootstrap over questions.

| Method | MATH /54 | Code /70 | Total /124 | Δ correct vs B0 (pp) [95% CI] | s / example | Speedup vs B0 [95% CI] | Mean calls | Outputs changed vs B0 | Broken / fixed |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| B0 | 5 | 38 | 43 | +0.00 [+0.00, +0.00] | 15.930 | +0.00% [+0.00, +0.00] | 256.0 | 0/124 | 0 / 0 |
| collab-TraceLock-LLaDA | 4 | 37 | 41 | -1.61 [-8.87, +5.65] | 5.763 | +176.42% [+144.17, +219.40] | 89.6 | 109/124 | 12 / 10 |
| collab-open-confidence | 4 | 38 | 42 | -0.81 [-5.65, +4.03] | 6.355 | +150.67% [+124.19, +185.10] | 88.2 | 52/124 | 5 / 4 |
| collab-safety-full | 7 | 38 | 45 | +1.61 [-2.42, +6.45] | 7.942 | +100.57% [+94.21, +107.53] | 109.1 | 50/124 | 3 / 5 |

Parity: same-job B0 vs the aligned campaign's B0: 124/124; her dense vs our B0: 0/0 (details in parity.json).
