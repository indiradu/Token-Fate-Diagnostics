# Extra-prompt parity of R0 against dense Conf-0.9 (LLaDA-8B)
Paired change in correct answers (R0 minus dense); percentage points of prompts; 95% bootstrap over prompts.

| set | prompts | dense correct | R0 correct | Δ answers | Δ points | speed vs dense |
|---|---|---|---|---|---|---|
| kod_ext (new) | 650 | 213 | 206 | -7 [-26, +11] (incl. 0) | -1.08 | 2.01x |
| math_ext (new) | 200 | 78 | 72 | -6 [-16, +4] (incl. 0) | -3.00 | 1.96x |
| gsm_ext (new) | 100 | 62 | 54 | -8 [-18, +2] (incl. 0) | -8.00 | 1.37x |
| he164 (Study 5) | 164 | 68 | 64 | -4 [-13, +5] (incl. 0) | -2.44 | 1.58x |
| kod_held (Study 5) | 150 | 59 | 60 | +1 [-8, +10] (incl. 0) | +0.67 | 1.95x |
| gsm_held (Study 5) | 100 | 52 | 51 | -1 [-13, +11] (incl. 0) | -1.00 | 1.36x |
| math108 (Study 5) | 108 | 31 | 33 | +2 [-5, +9] (incl. 0) | +1.85 | 1.98x |

Pooled, new prompts only: -2.21 pts [-4.74, +0.32] over 950 prompts (interval includes 0)

Pooled, new + Study 5 (LLaDA-8B): -1.56 pts [-3.60, +0.54] over 1472 prompts (interval includes 0)