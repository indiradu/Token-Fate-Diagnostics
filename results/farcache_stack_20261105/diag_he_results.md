# Added after seeing interim HumanEval: why does the cache lose answers on code?

(Not part of PREREGISTERED.md. WORD LOCK = C-0.9 everywhere here; the cells change the cache, the EOS truncation or the refresh frequency one at a time. HumanEval 0-163, gen 512, dense reference D:C-0.9 in the same process.)

## 1. Anatomy of the lost answers in the stage-1 HumanEval cells

Of 68 tasks that dense C-0.9 solves, the number lost by each cell; and how many cells lose each task.

Tasks lost by at least one cache cell: 43; by all six: 2; by at least four: 11 (HumanEval/16, HumanEval/43, HumanEval/25, HumanEval/9, HumanEval/105, HumanEval/158, HumanEval/47, HumanEval/12, HumanEval/128, HumanEval/49, HumanEval/82).


### Stage-1 cells, all HumanEval tasks (n = 164; reference D:C-0.9: 68 correct)

| cell | speed vs ref [CI] | calls | dense calls | rows/call | correct | net vs ref [CI] (lost/gained) | syntactically valid | max token run >= 8 | first EOS earlier than ref by >= 20 (of lost) | text differs from ref |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 1.000 [1.000, 1.000] | 53.8 | 53.8 | 690 | 68/164 | +0 [+0, +0] (0/0) | 162/164 | 1 | 0/0 | 0.00 |
| D:C-0.8 | 1.208 [1.169, 1.246] | 44.6 | 44.6 | 690 | 69/164 | +1 [-5, +7] (4/5) | 160/164 | 1 | 0/4 | 0.39 |
| D:Lctx-0.8 | 1.297 [1.241, 1.355] | 41.5 | 41.5 | 690 | 66/164 | -2 [-9, +5] (7/5) | 156/164 | 1 | 1/7 | 0.63 |
| S16:C-0.9 | 2.142 [2.007, 2.283] | 55.7 | 6.3 | 114 | 62/164 | -6 [-16, +3] (15/9) | 162/164 | 1 | 2/15 | 0.87 |
| S16:C-0.8 | 2.640 [2.486, 2.793] | 42.9 | 6.6 | 146 | 64/164 | -4 [-14, +6] (15/11) | 160/164 | 1 | 1/15 | 0.86 |
| S16:Lctx-0.8 | 3.047 [2.845, 3.267] | 34.3 | 7.3 | 182 | 40/164 | -28 [-39, -17] (32/4) | 97/164 | 4 | 3/32 | 0.90 |
| S32:C-0.9 | 2.242 [2.110, 2.379] | 53.2 | 7.4 | 132 | 59/164 | -9 [-19, +1] (17/8) | 159/164 | 1 | 7/17 | 0.84 |
| S32:C-0.8 | 2.615 [2.457, 2.785] | 43.5 | 7.6 | 167 | 63/164 | -5 [-15, +5] (16/11) | 159/164 | 1 | 5/16 | 0.84 |
| S32:Lctx-0.8 | 2.858 [2.689, 3.038] | 37.0 | 7.7 | 185 | 57/164 | -11 [-21, -2] (18/7) | 155/164 | 1 | 4/18 | 0.85 |

Scorer status of the lost answers (all cache cells): {'failed_tests': 113}.


## 2. One-ingredient-at-a-time cells (own process, same 164 tasks)


### Diagnostic process (n = 164; reference D:C-0.9: 68 correct)

| cell | speed vs ref [CI] | calls | dense calls | rows/call | correct | net vs ref [CI] (lost/gained) | syntactically valid | max token run >= 8 | first EOS earlier than ref by >= 20 (of lost) | text differs from ref |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 1.000 [1.000, 1.000] | 53.8 | 53.8 | 690 | 68/164 | +0 [+0, +0] (0/0) | 162/164 | 1 | 0/0 | 0.00 |
| S32:main | 2.243 [2.112, 2.380] | 53.2 | 7.4 | 132 | 59/164 | -9 [-19, +1] (17/8) | 159/164 | 1 | 7/17 | 0.84 |
| S32:every+eos | 1.192 [1.162, 1.228] | 45.5 | 45.5 | 690 | 68/164 | +0 [+0, +0] (0/0) | 162/164 | 1 | 0/0 | 0.00 |
| S32:block-noeos | 1.685 [1.581, 1.792] | 61.7 | 16.0 | 236 | 58/164 | -10 [-19, -1] (17/7) | 160/164 | 1 | 7/17 | 0.84 |
| S32:nofar | 1.677 [1.567, 1.791] | 49.6 | 7.5 | 425 | 69/164 | +1 [-8, +10] (10/11) | 161/164 | 1 | 4/10 | 0.78 |
| S32:block+k2 | 1.582 [1.499, 1.669] | 47.3 | 27.7 | 425 | 67/164 | -1 [-8, +6] (6/5) | 163/164 | 1 | 1/6 | 0.58 |
| S32:block+k4 | 1.973 [1.867, 2.086] | 46.3 | 17.1 | 280 | 68/164 | +0 [-9, +9] (10/10) | 164/164 | 1 | 3/10 | 0.72 |
| S16:every+eos | 1.124 [1.050, 1.200] | 47.9 | 47.9 | 690 | 69/164 | +1 [-3, +5] (2/3) | 162/164 | 1 | 0/2 | 0.42 |
| S16:block-noeos | 1.517 [1.417, 1.624] | 75.3 | 11.5 | 145 | 62/164 | -6 [-16, +3] (15/9) | 162/164 | 1 | 2/15 | 0.88 |

Parity: S32:main tokens equal the stage-1 S32:C-0.9 tokens on 164/164 tasks (same question, same code path).

- S32:main: lost 17; overlap with the tasks lost by S32:main 17
- S32:every+eos: lost 0; overlap with the tasks lost by S32:main 0
- S32:block-noeos: lost 17; overlap with the tasks lost by S32:main 17
- S32:nofar: lost 10; overlap with the tasks lost by S32:main 5
- S32:block+k2: lost 6; overlap with the tasks lost by S32:main 4
- S32:block+k4: lost 10; overlap with the tasks lost by S32:main 8
- S16:every+eos: lost 2; overlap with the tasks lost by S32:main 1
- S16:block-noeos: lost 15; overlap with the tasks lost by S32:main 8
