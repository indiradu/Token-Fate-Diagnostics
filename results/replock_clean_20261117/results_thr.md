# Threshold sweep: the cache factor as the word lock gets stronger (LLaDA-8B)
m = calls per 32-block of the dense decode; tau_d = dense ms per call; tau_c = 33.6 ms (cached call, from the cost benchmark).
Formula: S_cache(m) = m*tau_d / (tau_d + (m-1)*tau_c) (one dense call per block, the rest cached; no refresh cost, no call inflation). I = S_cache(threshold) / S_cache(B0 point): the interaction of the multiplication rule (1 = the cache factor does not depend on the word lock).

## he164
| threshold | prompts | m (calls/block) | S32 calls/block | word-lock factor (dense over B0) | S32 cache factor | R0 cache factor | formula S_cache(m) | I (R0) | R0 total over B0 | correct dense / S32 / R0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 60 | 1.7 | 1.8 | 6.9x | 1.33x | 1.28x | 1.34x | 0.66 | 8.7x | 13 / 12 / 13 |
| 0.6 | 60 | 2.0 | 2.0 | 5.5x | 1.45x | 1.41x | 1.44x | 0.73 | 7.9x | 14 / 20 / 17 |
| 0.7 | 60 | 2.1 | 2.3 | 5.0x | 1.43x | 1.39x | 1.47x | 0.72 | 7.6x | 29 / 28 / 30 |
| 0.8 | 60 | 2.3 | 2.5 | 4.8x | 1.50x | 1.47x | 1.54x | 0.76 | 7.2x | 37 / 29 / 34 |
| 0.9 | 60 | 2.8 | 2.9 | 4.2x | 1.65x | 1.56x | 1.66x | 0.81 | 6.3x | 38 / 32 / 34 |
| 0.95 | 60 | 3.3 | 3.4 | 3.6x | 1.71x | 1.60x | 1.75x | 0.83 | 5.6x | 38 / 34 / 35 |
| 0.99 | 60 | 4.4 | 4.7 | 2.7x | 1.84x | 1.77x | 1.92x | 0.92 | 4.6x | 38 / 33 / 35 |
| 1.0 | 20 | 12.1 | 12.5 | 1.0x | 2.19x | 1.93x | 2.28x | 1.00 | 1.9x | 13 / 10 / 12 |

## gsm_held
| threshold | prompts | m (calls/block) | S32 calls/block | word-lock factor (dense over B0) | S32 cache factor | R0 cache factor | formula S_cache(m) | I (R0) | R0 total over B0 | correct dense / S32 / R0 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.5 | 60 | 4.6 | 4.9 | 7.5x | 1.34x | 1.29x | 1.43x | 0.88 | 9.5x | 26 / 21 / 18 |
| 0.6 | 60 | 5.2 | 5.8 | 6.1x | 1.32x | 1.30x | 1.46x | 0.88 | 7.9x | 32 / 27 / 25 |
| 0.7 | 60 | 6.3 | 6.9 | 5.1x | 1.36x | 1.34x | 1.49x | 0.91 | 6.5x | 27 / 32 / 23 |
| 0.8 | 60 | 7.9 | 8.8 | 4.0x | 1.38x | 1.42x | 1.52x | 0.96 | 5.4x | 28 / 28 / 30 |
| 0.9 | 60 | 10.3 | 11.4 | 3.1x | 1.40x | 1.41x | 1.54x | 0.96 | 4.2x | 31 / 28 / 31 |
| 0.95 | 60 | 12.2 | 13.4 | 2.6x | 1.42x | 1.39x | 1.55x | 0.95 | 3.5x | 31 / 26 / 32 |
| 0.99 | 60 | 16.1 | 17.3 | 2.0x | 1.45x | 1.41x | 1.57x | 0.96 | 2.8x | 32 / 24 / 38 |
| 1.0 | 20 | 31.3 | 30.6 | 1.0x | 1.61x | 1.47x | 1.59x | 1.00 | 1.5x | 11 / 14 / 13 |
