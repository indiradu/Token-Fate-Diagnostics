# Depth-refresh cells, stage D_math108

n = 108 questions common to all arms. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000 resamples, seed 23). Cost model: call = 30.8 ms + 0.0705 ms x (mean rows over layers), fitted on the layer benchmark (dense + query-only cells).

| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | paired vs S32:main [CI] | speed vs D:C-0.9 [CI] | speed vs S32:main | calls | dense | partial | mean rows (layer-avg) | model time / measured |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 31/108 | +0 [+0, +0] (0/0) | - | 1.00x [1.00, 1.00] | 0.47x | 162.7 | 162.7 | 0.0 | 628 | - |
| S32:main | 31/108 | +0 [-7, +7] (6/6) | - | 2.11x [2.00, 2.22] | 1.00x | 179.4 | 14.1 | 0.0 | 68 | 0.93 |
| K4 | 33/108 | +2 [-4, +8] (4/6) | +2 [-3, +8] | 1.70x [1.61, 1.80] | 0.81x | 169.9 | 53.1 | 0.0 | 211 | 0.91 |
| W20-dense | 35/108 | +4 [-4, +12] (6/10) | +4 [-3, +11] | 1.97x [1.87, 2.07] | 0.93x | 174.2 | 27.2 | 0.0 | 117 | 0.92 |
| P4-l11-blank | 33/108 | +2 [-6, +10] (7/9) | +2 [-4, +8] | 2.07x [1.97, 2.17] | 0.98x | 171.7 | 14.2 | 34.3 | 109 | 0.94 |
| W20-blank11 | 27/108 | -4 [-11, +3] (8/4) | -4 [-9, +0] | 2.13x [2.03, 2.24] | 1.01x | 173.3 | 14.2 | 13.1 | 85 | 0.93 |

Parity with the stage-1 S32 tokens: S32:main 108/108. Speed intervals under about 5% are not resolved (one timed repeat).
