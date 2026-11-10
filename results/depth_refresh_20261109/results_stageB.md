# Depth-refresh cells, stage B: HumanEval 0-163, 512 tokens, word lock C-0.9, S32 cache

n = 164 questions common to all arms. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000 resamples, seed 23). Cost model: call = 30.8 ms + 0.0705 ms x (mean rows over layers), fitted on the layer benchmark (dense + query-only cells).

| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | paired vs S32:main [CI] | speed vs D:C-0.9 [CI] | speed vs S32:main | calls | dense | partial | mean rows (layer-avg) | model time / measured |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 68/164 | +0 [+0, +0] (0/0) | +9 [-1, +19] | 1.00x [1.00, 1.00] | 0.45x | 53.8 | 53.8 | 0.0 | 690 | - |
| S32:main | 59/164 | -9 [-19, +1] (17/8) | - | 2.24x [2.11, 2.37] | 1.00x | 53.2 | 7.4 | 0.0 | 132 | 0.92 |
| P2-l11-blank | 68/164 | +0 [-9, +9] (10/10) | +9 [+1, +17] | 2.05x [1.94, 2.16] | 0.92x | 51.5 | 7.8 | 20.2 | 225 | 0.95 |
| P4-l11-blank | 69/164 | +1 [-8, +10] (10/11) | +10 [+3, +17] | 2.19x [2.05, 2.34] | 0.98x | 51.1 | 7.7 | 8.6 | 175 | 0.93 |
| K2 | 67/164 | -1 [-8, +6] (6/5) | +8 [-1, +17] | 1.58x [1.49, 1.66] | 0.71x | 47.3 | 27.7 | 0.0 | 425 | 0.91 |
| K4 | 68/164 | +0 [-9, +9] (10/10) | +9 [+2, +17] | 1.97x [1.86, 2.08] | 0.88x | 46.3 | 17.1 | 0.0 | 280 | 0.91 |

Parity: S32:main tokens equal the stage-1 S32:C-0.9 tokens on 164/164 questions. Speed intervals under about 5% are not resolved (one timed repeat).
