# Depth-refresh cells, stage C_slurm

n = 164 questions common to all arms. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000 resamples, seed 23). Cost model: call = 30.8 ms + 0.0705 ms x (mean rows over layers), fitted on the layer benchmark (dense + query-only cells).

| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | paired vs S32:main [CI] | speed vs D:C-0.9 [CI] | speed vs S32:main | calls | dense | partial | mean rows (layer-avg) | model time / measured |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 68/164 | +0 [+0, +0] (0/0) | - | 1.00x [1.00, 1.00] | 0.43x | 53.8 | 53.8 | 0.0 | 690 | - |
| S32:main | 59/164 | -9 [-19, +1] (17/8) | - | 2.35x [2.21, 2.49] | 1.00x | 53.2 | 7.4 | 0.0 | 132 | 0.91 |
| W10-dense | 69/164 | +1 [-7, +9] (8/9) | +10 [+1, +20] | 1.98x [1.87, 2.11] | 0.84x | 47.8 | 17.1 | 0.0 | 289 | 0.88 |
| W20-dense | 69/164 | +1 [-8, +10] (10/11) | +10 [+2, +18] | 2.10x [1.99, 2.22] | 0.89x | 50.9 | 13.1 | 0.0 | 222 | 0.89 |
| W10-blank11 | 65/164 | -3 [-12, +6] (12/9) | +6 [-1, +13] | 2.18x [2.03, 2.35] | 0.93x | 53.4 | 7.7 | 10.3 | 186 | 0.91 |
| W20-blank11 | 64/164 | -4 [-13, +5] (13/9) | +5 [-2, +13] | 2.25x [2.10, 2.42] | 0.96x | 53.2 | 7.6 | 5.6 | 164 | 0.91 |
| K2 | 67/164 | -1 [-8, +6] (6/5) | +8 [-1, +17] | 1.62x [1.53, 1.70] | 0.69x | 47.3 | 27.7 | 0.0 | 425 | 0.87 |
| K4 | 68/164 | +0 [-9, +9] (10/10) | +9 [+2, +17] | 2.04x [1.93, 2.15] | 0.87x | 46.3 | 17.1 | 0.0 | 280 | 0.89 |

Parity with the stage-1 S32 tokens: S32:main 164/164. Speed intervals under about 5% are not resolved (one timed repeat).
