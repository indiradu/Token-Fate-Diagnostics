# Noise-matched control of the stale K/V on LLaDA-MoE (HumanEval; premise P3)
Arms: P-stale (= plain S32), P-fresh (oracle), P-rand1a/b (fresh + noise of the size of the staleness).

n = 164 prompts
| arm | correct | Δ vs P-fresh | Δ vs P-stale | identical to dense decode | identical to plain S32 |
|---|---|---|---|---|---|
| P-fresh | 100 |  | +8 [-2, +18] (incl. 0) | 108/164 | 37/164 |
| P-stale | 92 | -8 [-18, +2] (incl. 0) |  | 36/164 | 164/164 |
| P-rand1a | 11 | -89 [-101, -76] | -81 [-94, -68] | 6/164 | 6/164 |
| P-rand1b | 15 | -85 [-98, -71] | -77 [-90, -64] | 8/164 | 8/164 |

Pre-registered contrast (as on LLaDA-8B): mean of the two noise seeds minus P-stale: -79 [-90, -66]   (LLaDA-8B: +11.0 [+2, +20])