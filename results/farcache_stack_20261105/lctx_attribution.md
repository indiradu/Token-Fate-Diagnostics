# Is the stacked loss the learned word lock (Lctx-0.8) or the cache? (post hoc, 2026-11-07)

CPU only, from `records/stage1.jsonl` + `records/stage2.jsonl` (no new decodes). Each row is a paired difference in correct answers over the same questions, 95% interval from a paired bootstrap over questions (5,000 to 10,000 resamples). Not pre-registered.
Lctx effect = (Lctx-0.8 cell) minus (C-0.8 cell) with the same cache setting, so the threshold is the same and only the rule differs.

| set | Lctx effect, dense | Lctx effect, S32 cache | interaction (S32 minus dense) | cache effect with C-0.8 (S32 minus D) | Lctx effect, S16 cache |
|---|---|---|---|---|---|
| MATH 108, 512 | -3 [-11, +5] | -4 [-11, +2] | -1 [-9, +8] | -1 [-8, +6] | -9 [-17, -2] |
| MATH 54, 1024 | -1 [-6, +4] | -1 [-7, +5] | 0 [-9, +9] | 0 [-5, +6] | -6 [-11, -1] |
| HumanEval 164, 512 | -3 [-11, +5] | -6 [-15, +3] | -3 [-14, +8] | -6 [-16, +4] | -24 [-36, -13] |
| GSM8K 200, 256 | -5 [-21, +11] | +9 [-7, +25] | +14 [-10, +38] | -5 [-23, +13] | -4 [-21, +13] |

Reading: Lctx alone costs 1 to 5 answers against C-0.8 on every set (never resolved). Stacking it on the S32 cache is additive (interaction intervals all include 0). The S16 + Lctx cell is the only resolved Lctx-specific loss (MATH 108, MATH 54, HumanEval). On HumanEval the cache costs 6 to 9 answers under every word lock including C-0.9 (-9 [-19, +1]).
Pooled over the three distinct question sets (n = 472; GSM8K is budget-limited, so pooling is only indicative): Lctx vs C-0.8 dense -11 [-30, +8]; in S32 -1 [-20, +19]; in S16 -37 [-59, -15].
