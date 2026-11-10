# Depth-refresh cells, stage D_math54half: HumanEval 0-163, 512 tokens, word lock C-0.9, S32 cache

n = 54 questions common to all arms. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000 resamples, seed 23). Cost model: call = 30.8 ms + 0.0705 ms x (mean rows over layers), fitted on the layer benchmark (dense + query-only cells).

| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | paired vs S32:main [CI] | speed vs D:C-0.9 [CI] | speed vs S32:main | calls | dense | partial | mean rows (layer-avg) | model time / measured |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 18/54 | +0 [+0, +0] (0/0) | +1 [-4, +6] | 1.00x [1.00, 1.00] | 0.32x | 208.5 | 208.5 | 0.0 | 1132 | - |
| S32:main | 17/54 | -1 [-6, +4] (4/3) | - | 3.16x [2.91, 3.45] | 1.00x | 227.4 | 20.6 | 0.0 | 121 | 0.87 |
| K4 | 16/54 | -2 [-7, +3] (5/3) | -1 [-6, +4] | 2.00x [1.89, 2.12] | 0.63x | 223.8 | 73.0 | 0.0 | 382 | 0.80 |
| W20-dense | 16/54 | -2 [-8, +4] (5/3) | -1 [-5, +3] | 2.66x [2.46, 2.89] | 0.84x | 216.8 | 40.4 | 0.0 | 230 | 0.83 |
| P4-l11-blank | 18/54 | +0 [-6, +5] (4/4) | +1 [-4, +6] | 2.82x [2.61, 3.06] | 0.89x | 216.2 | 20.9 | 41.9 | 219 | 0.86 |
| W20-blank11 | 20/54 | +2 [-4, +8] (4/6) | +3 [-1, +8] | 2.82x [2.60, 3.05] | 0.89x | 238.9 | 22.0 | 19.7 | 159 | 0.86 |

Parity: S32:main tokens equal the stage-1 S32:C-0.9 tokens on 54/54 questions. Speed intervals under about 5% are not resolved (one timed repeat).
