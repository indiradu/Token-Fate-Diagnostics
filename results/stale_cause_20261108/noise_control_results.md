# Analysis 1: noise-matched control for the stale K/V of the S32 representation cache

HumanEval 0-163 (n = 164), 512 tokens, word lock C-0.9, LLaDA-8B-Instruct bf16, block 32. Reference: dense C-0.9 (68 correct, from `farcache_stack_20261105/records/stage1.jsonl`; deterministic decode).
The cached rows are the prompt, written-answer and blank rows outside the query set of a call. Arms differ only in where the attention of a non-dense call reads their K/V (cache, dense pass, dense pass + random perturbation of norm = scale x ||stale - fresh|| per row and layer). Bootstrap over questions, 10,000 resamples. No speed is measured.

## 0. Parity

| check | tasks equal |
|---|---|
| P-stale == stage-1 S32:C-0.9 (tokens) | 164/164 |
| P-fresh == stage-1 dense C-0.9 (all 512 tokens) | 133/164 |
| P-fresh == stage-1 dense C-0.9 (tokens up to the first EOS/EOT) | 133/164 |
| P-fresh and dense agree on pass/fail | 160/164 |

P-fresh tasks whose tokens differ from the dense decode: HumanEval/10, HumanEval/102, HumanEval/110, HumanEval/111, HumanEval/118, HumanEval/12, HumanEval/123, HumanEval/124, HumanEval/127, HumanEval/133, HumanEval/136, HumanEval/140, HumanEval/141, HumanEval/153, HumanEval/156, HumanEval/160, HumanEval/161, HumanEval/21, HumanEval/25, HumanEval/32, HumanEval/37, HumanEval/39, HumanEval/4, HumanEval/64, HumanEval/68, HumanEval/75, HumanEval/81, HumanEval/85, HumanEval/88, HumanEval/94, HumanEval/99.

## 1. Answers

Net = gained - lost against the reference in the column header (lost = reference passes, arm fails); 95% interval by bootstrap over questions. `text differs` = the generated text differs from the reference text.

| arm | correct | net vs dense [CI] (lost/gained) | net vs P-fresh [CI] | net vs P-stale [CI] | text differs from dense | text differs from P-stale | median first differing token vs dense |
|---|---|---|---|---|---|---|---|
| S32 (stage 1) | 59 | -9 [-19, +1] (17/8) | -9 [-18, +0] | +0 [+0, +0] | 0.84 | - | - |
| P-fresh | 68 | +0 [-4, +4] (2/2) | +0 [+0, +0] | +9 [+0, +18] | 0.19 | 0.84 | 512 (identical to dense: 133) |
| P-stale | 59 | -9 [-19, +1] (17/8) | -9 [-18, +0] | +0 [+0, +0] | 0.84 | 0.00 | 117 (identical to dense: 26) |
| P-rand0.5 | 66 | -2 [-7, +3] (4/2) | -2 [-8, +3] | +7 [-2, +16] | 0.48 | 0.84 | 512 (identical to dense: 86) |
| P-rand1a | 72 | +4 [-6, +14] (10/14) | +4 [-5, +13] | +13 [+3, +23] | 0.73 | 0.83 | 158 (identical to dense: 44) |
| P-rand1b | 68 | +0 [-8, +8] (9/9) | +0 [-9, +9] | +9 [-2, +20] | 0.75 | 0.85 | 151 (identical to dense: 41) |
| P-rand2 | 0 | -68 [-81, -56] (68/0) | -68 [-81, -56] | -59 [-71, -47] | 1.00 | 1.00 | 4 (identical to dense: 0) |
| B-stale | 66 | -2 [-9, +5] (8/6) | -2 [-10, +6] | +7 [-2, +16] | 0.68 | 0.77 | 161 (identical to dense: 52) |
| B-rand1 | 71 | +3 [-4, +11] (6/9) | +3 [-4, +11] | +12 [+1, +23] | 0.63 | 0.84 | 203 (identical to dense: 60) |

## 2. Pre-registered contrasts

- N1 (mean of two seeds) - dense: +2.0 answers [-5.5, +9.5]
- N1 (mean of two seeds) - P-fresh: +2.0 answers [-5.5, +9.5]
- N1 (mean of two seeds) - P-stale: +11.0 answers [+2.0, +20.0]
- seed a - seed b (P-rand1): +4.0 answers [-6.0, +14.0]
- B-rand1 - B-stale: +5.0 answers [-5.0, +15.0]

## 3. Which tasks are lost

Dense solves 68 tasks. `overlap` = tasks lost by both the arm and P-stale; `expected` = lost(arm) x lost(P-stale) / 68 (independent losses).

| arm | lost | overlap with P-stale lost | expected by chance |
|---|---|---|---|
| P-fresh | 2 | 2 | 0.5 |
| P-stale | 17 | 17 | 4.2 |
| P-rand0.5 | 4 | 3 | 1.0 |
| P-rand1a | 10 | 7 | 2.5 |
| P-rand1b | 9 | 4 | 2.2 |
| P-rand2 | 68 | 17 | 17.0 |
| B-stale | 8 | 5 | 2.0 |
| B-rand1 | 6 | 4 | 1.5 |

## 4. How far is the stale cache from a dense pass (measured in the P-stale / P-fresh arms)

Mean over rows, calls and questions of ||K_cache - K_dense|| / ||K_dense|| (per row, all kv heads and dims flattened) at layers 1-31 (layer 0 depends on the token only), and the same for V. Rows are the cached rows of non-dense calls.

| class | rel K (layers 1-31) | rel V (layers 1-31) | rel K at layers 8 / 16 / 24 / 31 | rows per question (layer 1) |
|---|---|---|---|---|
| prompt | 0.095 | 0.142 | 0.025 / 0.096 / 0.144 / 0.206 | 9585 |
| prior_written | 0.128 | 0.193 | 0.043 / 0.144 / 0.188 / 0.241 | 6927 |
| inblock_old | 0.285 | 0.475 | 0.173 / 0.383 / 0.339 / 0.377 | 644 |
| next_blank | 0.365 | 0.647 | 0.126 / 0.528 / 0.532 / 0.385 | 1442 |
| far_blank | 0.228 | 0.405 | 0.075 / 0.313 / 0.396 / 0.202 | 13593 |

- Realized noise norm / target (stale-fresh) norm in P-rand0.5: mean 0.5012, min 0.4700, max 0.5176 (perturbed classes only).
- Realized noise norm / target (stale-fresh) norm in P-rand1a: mean 1.0012, min 0.9920, max 1.0339 (perturbed classes only).
- Realized noise norm / target (stale-fresh) norm in P-rand2: mean 2.0010, min 2.0000, max 2.0429 (perturbed classes only).
- Realized noise norm / target (stale-fresh) norm in B-rand1: mean 1.0004, min 1.0000, max 1.0108 (perturbed classes only).

