# R0 ablation (LLaDA-8B, Conf-0.9)
Speed = dense time / arm time over the prompts; paired change in correct answers vs dense and vs R0 (95% bootstrap over prompts).

## he164 (n=100)
| arm | schedule | speed vs dense | calls | correct | Δ vs dense | Δ vs R0 |
|---|---|---|---|---|---|---|
| DENSE |  | 1.00x | 48.7 | 55 |  |  |
| S32 |  | 1.66x | 54.5 | 45 | -10 [-18, -2] | -3 [-9, +3] (incl. 0) |
| R0 | layer 10, blank, k=2 | 1.58x | 51.9 | 48 | -7 [-14, +0] (incl. 0) |  |
| R0_A25 | layer 8, blank, k=2 | 1.60x | 50.2 | 52 | -3 [-10, +4] (incl. 0) | +4 [-1, +9] (incl. 0) |
| R0_A50 | layer 16, blank, k=2 | 1.58x | 53.5 | 51 | -4 [-12, +4] (incl. 0) | +3 [-4, +10] (incl. 0) |
| R0_B05 | layer 10, blank, k=2 | 1.58x | 51.9 | 48 | -7 [-14, +0] (incl. 0) | +0 [+0, +0] (incl. 0) |
| R0_B20 | layer 10, blank, k=2 | 1.58x | 51.9 | 48 | -7 [-14, +0] (incl. 0) | +0 [+0, +0] (incl. 0) |
| FIX_K2 | layer 10, blank, k=2 | 1.58x | 51.9 | 48 | -7 [-14, +0] (incl. 0) | +0 [+0, +0] (incl. 0) |
| FIX_K4 | layer 10, blank, k=4 | 1.63x | 52.9 | 53 | -2 [-8, +4] (incl. 0) | +5 [+0, +10] (incl. 0) |
| FIX_K8 | layer 10, blank, k=8 | 1.65x | 54.0 | 52 | -3 [-10, +3] (incl. 0) | +4 [-2, +10] (incl. 0) |

## kod_held (n=100)
| arm | schedule | speed vs dense | calls | correct | Δ vs dense | Δ vs R0 |
|---|---|---|---|---|---|---|
| DENSE |  | 1.00x | 72.8 | 40 |  |  |
| S32 |  | 2.11x | 81.0 | 35 | -5 [-13, +3] (incl. 0) | -4 [-11, +2] (incl. 0) |
| R0 | layer 10, blank, k=3 | 2.02x | 78.4 | 39 | -1 [-9, +7] (incl. 0) |  |
| R0_A25 | layer 8, blank, k=3 | 2.04x | 77.0 | 40 | +0 [-8, +8] (incl. 0) | +1 [-3, +6] (incl. 0) |
| R0_A50 | layer 16, blank, k=2 | 2.02x | 77.6 | 38 | -2 [-9, +5] (incl. 0) | -1 [-6, +5] (incl. 0) |
| R0_B05 | layer 10, blank, k=3 | 2.02x | 78.4 | 39 | -1 [-9, +6] (incl. 0) | +0 [+0, +0] (incl. 0) |
| R0_B20 | layer 10, blank, k=2 | 1.98x | 77.1 | 38 | -2 [-10, +5] (incl. 0) | -1 [-8, +5] (incl. 0) |
| FIX_K2 | layer 10, blank, k=2 | 1.98x | 77.1 | 38 | -2 [-10, +6] (incl. 0) | -1 [-8, +5] (incl. 0) |
| FIX_K4 | layer 10, blank, k=4 | 2.07x | 78.1 | 37 | -3 [-11, +5] (incl. 0) | -2 [-7, +3] (incl. 0) |
| FIX_K8 | layer 10, blank, k=8 | 2.13x | 78.0 | 39 | -1 [-9, +7] (incl. 0) | +0 [-5, +5] (incl. 0) |

## gsm_held (n=100)
| arm | schedule | speed vs dense | calls | correct | Δ vs dense | Δ vs R0 |
|---|---|---|---|---|---|---|
| DENSE |  | 1.00x | 81.0 | 52 |  |  |
| S32 |  | 1.39x | 90.5 | 46 | -6 [-18, +5] (incl. 0) | -5 [-18, +9] (incl. 0) |
| R0 | layer 10, all, k=4 | 1.38x | 84.8 | 51 | -1 [-13, +11] (incl. 0) |  |
| R0_A25 | layer 8, all, k=4 | 1.35x | 85.8 | 47 | -5 [-17, +7] (incl. 0) | -4 [-13, +5] (incl. 0) |
| R0_A50 | layer 16, all, k=3 | 1.39x | 84.4 | 48 | -4 [-15, +7] (incl. 0) | -3 [-15, +9] (incl. 0) |
| R0_B05 | layer 10, all, k=6 | 1.38x | 87.2 | 48 | -4 [-17, +9] (incl. 0) | -3 [-14, +8] (incl. 0) |
| R0_B20 | layer 10, all, k=2 | 1.30x | 83.0 | 51 | -1 [-13, +11] (incl. 0) | +0 [-11, +11] (incl. 0) |
| FIX_K2 | layer 10, blank, k=2 | 1.39x | 85.1 | 45 | -7 [-18, +5] (incl. 0) | -6 [-17, +5] (incl. 0) |
| FIX_K4 | layer 10, blank, k=4 | 1.41x | 87.0 | 45 | -7 [-18, +5] (incl. 0) | -6 [-18, +6] (incl. 0) |
| FIX_K8 | layer 10, blank, k=8 | 1.39x | 89.5 | 47 | -5 [-16, +7] (incl. 0) | -4 [-16, +8] (incl. 0) |
