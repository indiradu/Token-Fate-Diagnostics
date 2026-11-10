# Depth-refresh cells, stage A: HumanEval 0-163, 512 tokens, word lock C-0.9, S32 cache

n = 164 questions common to all arms. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000 resamples, seed 23). Cost model: call = 30.8 ms + 0.0705 ms x (mean rows over layers), fitted on the layer benchmark (dense + query-only cells).

| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | paired vs S32:main [CI] | speed vs D:C-0.9 [CI] | speed vs S32:main | calls | dense | partial | mean rows (layer-avg) | model time / measured |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 68/164 | +0 [+0, +0] (0/0) | +9 [-1, +19] | 1.00x [1.00, 1.00] | 0.45x | 53.8 | 53.8 | 0.0 | 690 | - |
| S32:main | 59/164 | -9 [-19, +1] (17/8) | - | 2.24x [2.11, 2.38] | 1.00x | 53.2 | 7.4 | 0.0 | 132 | 0.92 |
| P1-l0-context | 64/164 | -4 [-12, +4] (11/7) | +5 [-4, +14] | 1.50x [1.40, 1.63] | 0.67x | 50.3 | 7.7 | 42.6 | 403 | 0.94 |
| P1-l11-context | 59/164 | -9 [-18, -1] (14/5) | +0 [-9, +9] | 1.78x [1.68, 1.90] | 0.80x | 48.9 | 7.6 | 41.3 | 313 | 0.94 |
| P1-l11-all | 71/164 | +3 [-5, +11] (6/9) | +12 [+2, +22] | 1.49x [1.40, 1.59] | 0.66x | 44.9 | 7.9 | 37.1 | 508 | 0.90 |
| P1-l24-all | 67/164 | -1 [-10, +8] (11/10) | +8 [+0, +16] | 1.83x [1.73, 1.95] | 0.82x | 50.6 | 7.6 | 43.0 | 278 | 0.92 |
| P1-l11-blank | 72/164 | +4 [-5, +13] (9/13) | +13 [+4, +22] | 1.85x [1.75, 1.97] | 0.83x | 50.2 | 7.7 | 42.4 | 327 | 0.96 |

Parity: S32:main tokens equal the stage-1 S32:C-0.9 tokens on 164/164 questions. Speed intervals under about 5% are not resolved (one timed repeat).
