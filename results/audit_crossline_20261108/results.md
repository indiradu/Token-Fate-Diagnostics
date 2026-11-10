# Results: cross-line consistency audit (HumanEval 164, gen 512)

Pre-registration: `PREREGISTERED_AUDIT.md` (+ Addendum 1 and 2, both written before the labels were unblinded). Word lock = C-0.9 / C-0.8 / Lctx-0.8; representation cache = S32 / S16 (dense refresh at each 32-block start), D = dense. Rows concerned: answer rows of HumanEval programs. Order of operations: pool built (class counts printed only), two blind labelling passes saved, then this script (automatic statistics and unblinding).

## 1. Sets per cell vs the dense C-0.9 reference (68 of 164 correct)

| cell | LOST | GAINED | CHANGED-SAME-PASS | CHANGED-SAME-FAIL | IDENTICAL | net | lost rate among 68 dense-pass tasks [exact 95%] |
|---|---|---|---|---|---|---|---|
| S32:C-0.9 | 17 | 8 | 32 | 81 | 26 | -9 | 17/68 = 0.25 [0.15, 0.37] |
| S32:Lctx-0.8 | 18 | 7 | 33 | 82 | 24 | -11 | 18/68 = 0.26 [0.17, 0.39] |
| S32:C-0.8 | 16 | 11 | 34 | 77 | 26 | -5 | 16/68 = 0.24 [0.14, 0.35] |
| S16:C-0.9 | 15 | 9 | 37 | 82 | 21 | -6 | 15/68 = 0.22 [0.13, 0.34] |
| S16:C-0.8 | 15 | 11 | 36 | 79 | 23 | -4 | 15/68 = 0.22 [0.13, 0.34] |
| D:C-0.8 | 4 | 5 | 18 | 37 | 100 | +1 | 4/68 = 0.06 [0.02, 0.14] |
| D:Lctx-0.8 | 7 | 5 | 26 | 65 | 61 | -2 | 7/68 = 0.10 [0.04, 0.20] |
| S16:Lctx-0.8 | 32 | 4 | 22 | 89 | 17 | -28 | 32/68 = 0.47 [0.35, 0.60] |

Dense word-lock controls (D:C-0.8, D:Lctx-0.8) change the word lock only. Their LOST counts (4 and 7 of 68) are the churn that is NOT due to the cache; every main cache cell loses 15-18 of 68. S16:Lctx-0.8 (32 lost) is the broken-code transfer failure and is excluded from the main contrasts (section 8).

Union of tasks lost: 34 tasks by at least one of the 5 main cache cells, 10 tasks by at least one dense control (of 68).
Excess (cache lost minus dense-control lost, counts): S32:C-0.9: 17 - {4, 7}, S32:Lctx-0.8: 18 - {4, 7}, S32:C-0.8: 16 - {4, 7}, S16:C-0.9: 15 - {4, 7}, S16:C-0.8: 15 - {4, 7}

## 2. Per-task multiplicity (5 main cache cells)

| tasks lost by >= k cells | observed | null mean (cells lose uniformly random dense-pass tasks) | null 2.5-97.5% | P(null >= observed) |
|---|---|---|---|---|
| 1 | 34 | 50.6 | [46, 55] | 1.0000 |
| 2 | 26 | 23.3 | [19, 27] | 0.1426 |
| 3 | 13 | 6.2 | [3, 10] | 0.0007 |
| 4 | 5 | 0.9 | [0, 3] | 0.0005 |
| 5 | 3 | 0.1 | [0, 1] | 0.0000 |

The observed concentration is far above the independent-cells null, as expected (tasks differ in fragility, and S32/S16 cells share a mechanism); this says lost tasks are task-specific, not that a cause is known.
Dense controls: tasks lost by both controls: 1; overlap with cache-lost tasks: 9 of 10.

## 3. First divergence from the dense C-0.9 text (token level, 512 gen positions, 32-blocks)

Pairs are (cell, task) with a changed text; a task lost by several cells appears once per cell. d = first differing gen position (up to the shorter first-EOS + 1); relative position = d / dense first-EOS; offset = d mod 32. The call that wrote each token is not stored, so offset in block is only a proxy for time since the last dense refresh (S cells) and has no cache meaning in D cells.

| group | n pairs | median d [IQR] | median relative position | offset 0-7 | 8-15 | 16-23 | 24-31 | d < 32 (first block) | same-length prefix (d None) |
|---|---|---|---|---|---|---|---|---|---|
| cache LOST | 81 | 107 [36, 142] | 0.61 | 7 (9%) | 23 (28%) | 27 (33%) | 24 (30%) | 20 (25%) | 0 |
| cache GAINED | 46 | 109 [24, 177] | 0.54 | 7 (15%) | 6 (13%) | 13 (28%) | 20 (43%) | 13 (28%) | 0 |
| cache CHANGED-SAME-PASS | 172 | 82 [15, 150] | 0.58 | 26 (15%) | 62 (36%) | 39 (23%) | 45 (26%) | 64 (37%) | 0 |
| cache CHANGED-SAME-FAIL | 401 | 78 [13, 154] | 0.36 | 55 (14%) | 146 (36%) | 107 (27%) | 93 (23%) | 166 (41%) | 0 |
| cache all changed | 700 | 94 [16, 151] | 0.50 | 95 (14%) | 237 (34%) | 186 (27%) | 182 (26%) | 263 (38%) | 0 |
| S32 cells all changed | 416 | 94 [15, 150] | 0.50 | 53 (13%) | 141 (34%) | 110 (26%) | 112 (27%) | 157 (38%) | 0 |
| S16 cells (C) all changed | 284 | 94 [16, 153] | 0.50 | 42 (15%) | 96 (34%) | 76 (27%) | 70 (25%) | 106 (37%) | 0 |
| dense-control LOST | 11 | 119 [92, 205] | 0.61 | 2 (18%) | 6 (55%) | 2 (18%) | 1 (9%) | 2 (18%) | 0 |
| dense-control GAINED | 10 | 110 [6, 212] | 0.67 | 4 (40%) | 3 (30%) | 0 (0%) | 3 (30%) | 3 (30%) | 0 |
| dense-control all changed | 167 | 135 [38, 205] | 0.65 | 47 (28%) | 57 (34%) | 39 (23%) | 24 (14%) | 39 (23%) | 0 |
| S16:Lctx-0.8 (transfer) LOST | 32 | 128 [34, 158] | 0.63 | 9 (28%) | 12 (38%) | 6 (19%) | 5 (16%) | 6 (19%) | 0 |
| S16:Lctx-0.8 (transfer) all changed | 147 | 76 [13, 147] | 0.37 | 36 (24%) | 50 (34%) | 32 (22%) | 29 (20%) | 57 (39%) | 0 |

Tests (pair level, not independent: a task can occur in several cells; descriptive):

| comparison (share with property) | group A | group B | one-sided Fisher p (A > B) | two-sided p |
|---|---|---|---|---|
| offset 0-7 (just after a block start): cache LOST vs cache other changed | 7/81 | 88/619 | 0.946 | 0.226 |
| offset 24-31 (late in block): cache LOST vs cache other changed | 24/81 | 158/619 | 0.253 | 0.422 |
| d < 32 (first block): cache LOST vs cache other changed | 20/81 | 243/619 | 0.997 | 0.010 |
| d < 32 (first block): cache LOST vs dense-control all changed | 20/81 | 39/167 | 0.467 | 0.874 |
| offset 0-7: cache all changed vs dense-control all changed | 95/700 | 47/167 | 1.000 | 0.000 |

Mean relative first-divergence position: cache LOST 0.521 (n=81) vs cache other changed 0.427 (n=619); permutation one-sided P(LOST earlier or equal) = 0.993.
Expected under no timing effect: each offset bin 25% (the dense refresh occurs at the start of each 32-block, so bin 0 holds the freshest rows).

## 4. Edit class of the changed programs (normalized lines vs dense C-0.9)

TEXT-ONLY = identical after dropping comments/docstrings/blank lines; LOCAL = one hunk of at most 2 lines per side; BLOCK = one hunk of 3+ lines; STRUCTURAL = 2+ separate hunks; UNPARSABLE = cell code does not parse.

| group | n | TEXT-ONLY | LOCAL | BLOCK | STRUCTURAL | UNPARSABLE |
|---|---|---|---|---|---|---|
| cache LOST | 81 | 0 (0%) | 16 (20%) | 17 (21%) | 44 (54%) | 4 (5%) |
| cache GAINED | 46 | 0 (0%) | 9 (20%) | 2 (4%) | 33 (72%) | 2 (4%) |
| cache CHANGED-SAME-PASS | 172 | 23 (13%) | 36 (21%) | 58 (34%) | 55 (32%) | 0 (0%) |
| cache CHANGED-SAME-FAIL | 401 | 10 (2%) | 59 (15%) | 115 (29%) | 195 (49%) | 22 (5%) |
| cache all changed | 700 | 33 (5%) | 120 (17%) | 192 (27%) | 327 (47%) | 28 (4%) |
| S32 cells all changed | 416 | 15 (4%) | 70 (17%) | 113 (27%) | 196 (47%) | 22 (5%) |
| S16 cells (C) all changed | 284 | 18 (6%) | 50 (18%) | 79 (28%) | 131 (46%) | 6 (2%) |
| dense-control LOST | 11 | 0 (0%) | 2 (18%) | 0 (0%) | 5 (45%) | 4 (36%) |
| dense-control GAINED | 10 | 0 (0%) | 3 (30%) | 2 (20%) | 5 (50%) | 0 (0%) |
| dense-control all changed | 167 | 16 (10%) | 34 (20%) | 38 (23%) | 69 (41%) | 10 (6%) |
| S16:Lctx-0.8 (transfer) LOST | 32 | 0 (0%) | 0 (0%) | 3 (9%) | 7 (22%) | 22 (69%) |
| S16:Lctx-0.8 (transfer) all changed | 147 | 9 (6%) | 16 (11%) | 26 (18%) | 24 (16%) | 72 (49%) |

LOCAL-or-TEXT-ONLY share and tests:

| comparison | group A | group B | one-sided Fisher p (A > B) | two-sided p |
|---|---|---|---|---|
| STRUCTURAL: cache LOST vs dense-control LOST | 44/81 | 5/11 | 0.407 | 0.750 |
| STRUCTURAL: cache LOST vs cache other changed | 44/81 | 283/619 | 0.090 | 0.156 |
| LOCAL (strict one hunk <= 2 lines): cache LOST vs dense-control LOST | 16/81 | 2/11 | 0.633 | 1.000 |
| LOCAL: cache LOST vs cache other changed | 16/81 | 104/619 | 0.300 | 0.531 |
| UNPARSABLE: cache LOST vs dense-control LOST | 4/81 | 4/11 | 1.000 | 0.006 |

## 5. Static signatures (all 6 reported for every set; S0 = does not parse)

Share of programs having the signature. Programs are the cell's own program (the pair table is by cell and task, so a task lost by 3 cells counts 3 times). 'new vs ref' counts signatures present in the cell program and absent in the dense C-0.9 program of the same task.

### 5a. By set (pooled over the 5 main cache cells; dense controls pooled over the 2 controls)

| set | n | S0_unparsable | S1_undefined_name | S2_unused_local | S3_recompute_after_assign | S4_unguarded_index_possibly_empty | S5_unreachable_code | S6_implicit_none_path | stray fence |
|---|---|---|---|---|---|---|---|---|---|
| cache LOST | 81 | 4 (5%) | 8 (10%) | 3 (4%) | 1 (1%) | 8 (10%) | 1 (1%) | 2 (2%) | 2 (2%) |
| cache GAINED | 46 | 2 (4%) | 0 (0%) | 1 (2%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 2 (4%) |
| cache CHANGED-SAME-PASS | 172 | 0 (0%) | 0 (0%) | 0 (0%) | 1 (1%) | 9 (5%) | 0 (0%) | 0 (0%) | 0 (0%) |
| cache CHANGED-SAME-FAIL | 401 | 22 (5%) | 25 (6%) | 12 (3%) | 3 (1%) | 4 (1%) | 1 (0%) | 7 (2%) | 7 (2%) |
| cache IDENTICAL | 120 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| dense-control LOST | 11 | 4 (36%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 4 (36%) |
| dense-control GAINED | 10 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| dense-control CHANGED-SAME | 146 | 6 (4%) | 5 (3%) | 3 (2%) | 3 (2%) | 6 (4%) | 0 (0%) | 2 (1%) | 3 (2%) |
| dense-control IDENTICAL | 161 | 2 (1%) | 3 (2%) | 2 (1%) | 2 (1%) | 3 (2%) | 0 (0%) | 1 (1%) | 0 (0%) |
| S16:Lctx-0.8 LOST | 32 | 22 (69%) | 5 (16%) | 1 (3%) | 0 (0%) | 1 (3%) | 1 (3%) | 0 (0%) | 2 (6%) |

### 5b. New signature vs the dense program of the same task (present in the cell program, absent in the dense C-0.9 program)

| set | n | S0_unparsable | S1_undefined_name | S2_unused_local | S3_recompute_after_assign | S4_unguarded_index_possibly_empty | S5_unreachable_code | S6_implicit_none_path |
|---|---|---|---|---|---|---|---|---|
| cache LOST | 81 | 4 (5%) | 8 (10%) | 3 (4%) | 1 (1%) | 1 (1%) | 1 (1%) | 2 (2%) |
| cache GAINED | 46 | 2 (4%) | 0 (0%) | 1 (2%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| cache CHANGED-SAME-PASS | 172 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 1 (1%) | 0 (0%) | 0 (0%) |
| cache CHANGED-SAME-FAIL | 401 | 21 (5%) | 25 (6%) | 12 (3%) | 2 (0%) | 0 (0%) | 1 (0%) | 1 (0%) |
| dense-control LOST | 11 | 4 (36%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| dense-control GAINED | 10 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) |
| dense-control CHANGED-SAME | 146 | 6 (4%) | 5 (3%) | 3 (2%) | 2 (1%) | 0 (0%) | 0 (0%) | 0 (0%) |
| S16:Lctx-0.8 LOST | 32 | 22 (69%) | 5 (16%) | 1 (3%) | 0 (0%) | 0 (0%) | 1 (3%) | 0 (0%) |

### 5c. By cell, all 164 tasks, split by the cell's own pass/fail

| cell | outcome | n | S0_unparsable | S1_undefined_name | S2_unused_local | S3_recompute_after_assign | S4_unguarded_index_possibly_empty | S5_unreachable_code | S6_implicit_none_path | stray fence |
|---|---|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | pass | 68 | 0 (0%) | 0 (0%) | 0 (0%) | 1 (1%) | 3 (4%) | 0 (0%) | 0 (0%) | 0 |
| D:C-0.9 | fail | 96 | 2 (2%) | 3 (3%) | 2 (2%) | 2 (2%) | 3 (3%) | 0 (0%) | 2 (2%) | 0 |
| S32:C-0.9 | pass | 59 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 1 (2%) | 0 (0%) | 0 (0%) | 0 |
| S32:C-0.9 | fail | 105 | 5 (5%) | 6 (6%) | 3 (3%) | 0 (0%) | 4 (4%) | 0 (0%) | 2 (2%) | 0 |
| S32:Lctx-0.8 | pass | 57 | 1 (2%) | 0 (0%) | 0 (0%) | 0 (0%) | 3 (5%) | 0 (0%) | 0 (0%) | 1 |
| S32:Lctx-0.8 | fail | 107 | 10 (9%) | 8 (7%) | 0 (0%) | 0 (0%) | 2 (2%) | 1 (1%) | 3 (3%) | 5 |
| S32:C-0.8 | pass | 63 | 1 (2%) | 0 (0%) | 0 (0%) | 1 (2%) | 1 (2%) | 0 (0%) | 0 (0%) | 1 |
| S32:C-0.8 | fail | 101 | 5 (5%) | 6 (6%) | 4 (4%) | 1 (1%) | 3 (3%) | 0 (0%) | 1 (1%) | 1 |
| S16:C-0.9 | pass | 62 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 2 (3%) | 0 (0%) | 0 (0%) | 0 |
| S16:C-0.9 | fail | 102 | 2 (2%) | 4 (4%) | 3 (3%) | 2 (2%) | 2 (2%) | 0 (0%) | 1 (1%) | 1 |
| S16:C-0.8 | pass | 64 | 0 (0%) | 0 (0%) | 1 (2%) | 0 (0%) | 2 (3%) | 0 (0%) | 0 (0%) | 0 |
| S16:C-0.8 | fail | 100 | 4 (4%) | 9 (9%) | 5 (5%) | 1 (1%) | 1 (1%) | 1 (1%) | 2 (2%) | 2 |
| D:C-0.8 | pass | 69 | 0 (0%) | 0 (0%) | 0 (0%) | 1 (1%) | 3 (4%) | 0 (0%) | 0 (0%) | 0 |
| D:C-0.8 | fail | 95 | 4 (4%) | 3 (3%) | 3 (3%) | 2 (2%) | 2 (2%) | 0 (0%) | 2 (2%) | 2 |
| D:Lctx-0.8 | pass | 66 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 3 (5%) | 0 (0%) | 0 (0%) | 0 |
| D:Lctx-0.8 | fail | 98 | 8 (8%) | 5 (5%) | 2 (2%) | 2 (2%) | 1 (1%) | 0 (0%) | 1 (1%) | 5 |
| S16:Lctx-0.8 | pass | 40 | 0 (0%) | 0 (0%) | 0 (0%) | 0 (0%) | 1 (2%) | 1 (2%) | 1 (2%) | 0 |
| S16:Lctx-0.8 | fail | 124 | 72 (58%) | 8 (6%) | 1 (1%) | 0 (0%) | 3 (2%) | 1 (1%) | 1 (1%) | 9 |

Signature detectors were unit-tested on synthetic snippets (`scripts/test_signatures.py`). They are crude (S3 and S4 approximate dataflow by line order; an early `return` guard for S4 is not modelled), so counts are indicative and absence is not evidence of absence.

## 6. Blind labelling: pool and reliability

Pool: 77 items; class counts (an item may carry several tags): CL 38, DL 10, CGd 10, DGd 8, CSP 4, CSF 6, DSP 3, DSF 3.

Self-agreement between pass 1 and pass 2 (same labeller, memory of pass 1 not erased; an upper bound on reliability): raw 76/77 = 0.99; Cohen kappa 5 categories 0.98; kappa (b vs not-b) 0.96; raw agreement on (b vs not-b) 76/77.

Confusion (rows pass 1, columns pass 2):

| p1 \ p2 | a | b | c | d | e |
|---|---|---|---|---|---|
| a | 26 | 0 | 0 | 0 | 0 |
| b | 0 | 15 | 0 | 0 | 0 |
| c | 0 | 1 | 19 | 0 | 0 |
| d | 0 | 0 | 0 | 9 | 0 |
| e | 0 | 0 | 0 | 0 | 7 |

Items where (b)-vs-not-(b) flipped between passes: 1 (these are the b-index / b-dead / b-type borderline items; the primary analysis uses pass 1).

### 6a. Validity check: pass-1 label vs the actual pass/fail of the labelled program

| actual outcome of the labelled program | n | (a) | (b) | (c) | (d) | (e) |
|---|---|---|---|---|---|---|
| fails tests | 70 | 26 | 15 | 20 | 2 | 7 |
| passes tests | 7 | 0 | 0 | 0 | 7 | 0 |

Blind reading flagged a defect (a/b/c/e) in 68/70 programs that really fail tests (0.97) and in 0/7 programs that pass (a/b/c; (d) was assigned to 7/7). Failing programs labelled (d) (no defect found): b51ced (HumanEval/51, ['DL']), d14b17 (HumanEval/116, ['CGd', 'DGd']).

## 7. Category distribution by class (pass 1)

Classes: CL = cache LOST (labelled program = the cache cell's); DL = dense word-lock-control LOST (labelled program = D:C-0.8 / D:Lctx-0.8's); CGd = the dense C-0.9 (wrong) program on a task a cache cell GAINED; DGd = same for a dense control; CSF/CSP = cache changed-same-fail/-pass; DSF/DSP = dense control changed-same-fail/-pass. All classes except CSP, DSP are failing programs.

| class | n | (a) local slip | (b) cross-line | (c) algorithm / spec | (d) correct | (e) other | (b) share [exact 95%] | strict (b) share |
|---|---|---|---|---|---|---|---|---|
| CL | 38 | 16 | 9 | 10 | 0 | 3 | 9/38 = 0.24 [0.11, 0.40] | 6/38 |
| DL | 10 | 4 | 1 | 1 | 1 | 3 | 1/10 = 0.10 [0.00, 0.45] | 0/10 |
| CGd | 10 | 4 | 2 | 3 | 1 | 0 | 2/10 = 0.20 [0.03, 0.56] | 1/10 |
| DGd | 8 | 3 | 3 | 1 | 1 | 0 | 3/8 = 0.38 [0.09, 0.76] | 0/8 |
| CSF | 6 | 2 | 0 | 3 | 0 | 1 | 0/6 = 0.00 [0.00, 0.46] | 0/6 |
| DSF | 3 | 0 | 1 | 2 | 0 | 0 | 1/3 = 0.33 [0.01, 0.91] | 0/3 |
| CSP | 4 | 0 | 0 | 0 | 4 | 0 | 0/4 = 0.00 [0.00, 0.60] | 0/4 |
| DSP | 3 | 0 | 0 | 0 | 3 | 0 | 0/3 = 0.00 [0.00, 0.71] | 0/3 |
| dense-version-wrong pooled (DL + CGd + DGd, unique items) | 24 | 9 | 5 | 5 | 2 | 3 | 5/24 = 0.21 [0.07, 0.42] | 1/24 |

(b) subtypes among (b)-labelled items (broad: type, undef, index; strict: dup, dead, logic, init, empty):

| class | b-undef | b-type | b-index | b-empty | b-dup | b-dead | b-logic | b-init |
|---|---|---|---|---|---|---|---|---|
| CL | 0 | 1 | 1 | 1 | 0 | 1 | 3 | 1 |
| DL | 0 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| CGd | 1 | 0 | 0 | 0 | 1 | 0 | 0 | 0 |
| DGd | 1 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| CSF | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| DSF | 0 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |

CL items labelled (e): 3; of which stray fence / syntax: 3.

### 7a. Primary contrast: share of (b) cross-line inconsistency among failing programs

One-sided Fisher exact (A > B); permutation test (10,000, seed 20261108) of the difference in shares as a cross-check; n are item counts (unique task+text), and several CL items can come from the same task.

**primary (pass 1, all labelled items)**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 9/38 = 0.24 | 1/10 = 0.10 | +14 pp | 0.322 | 0.664 | 0.321 |
| CL vs CGd (dense counterpart of cache-GAINED) | 9/38 = 0.24 | 2/10 = 0.20 | +4 pp | 0.587 | 1.000 | 0.588 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 9/38 = 0.24 | 5/24 = 0.21 | +3 pp | 0.525 | 1.000 | 0.524 |

**excluding the 12 hypothesis-generating tasks (4, 9, 16 + 9 named in diag_he_results.md)**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 4/18 = 0.22 | 0/6 = 0.00 | +22 pp | 0.288 | 0.539 | 0.287 |
| CL vs CGd (dense counterpart of cache-GAINED) | 4/18 = 0.22 | 2/10 = 0.20 | +2 pp | 0.642 | 1.000 | 0.643 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 4/18 = 0.22 | 4/20 = 0.20 | +2 pp | 0.589 | 1.000 | 0.590 |

**(e) removed from denominators (logic errors only)**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 9/35 = 0.26 | 1/7 = 0.14 | +11 pp | 0.461 | 1.000 | 0.455 |
| CL vs CGd (dense counterpart of cache-GAINED) | 9/35 = 0.26 | 2/10 = 0.20 | +6 pp | 0.535 | 1.000 | 0.528 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 9/35 = 0.26 | 5/21 = 0.24 | +2 pp | 0.568 | 1.000 | 0.573 |

**strict (b) only (dup, dead, logic, init, empty)**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 6/38 = 0.16 | 0/10 = 0.00 | +16 pp | 0.225 | 0.320 | 0.227 |
| CL vs CGd (dense counterpart of cache-GAINED) | 6/38 = 0.16 | 1/10 = 0.10 | +6 pp | 0.546 | 1.000 | 0.554 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 6/38 = 0.16 | 1/24 = 0.04 | +12 pp | 0.160 | 0.232 | 0.165 |

**one item per task per class**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 5/27 = 0.19 | 1/10 = 0.10 | +9 pp | 0.475 | 1.000 | 0.474 |
| CL vs CGd (dense counterpart of cache-GAINED) | 5/27 = 0.19 | 2/10 = 0.20 | -1 pp | 0.727 | 1.000 | 0.725 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 5/27 = 0.19 | 5/24 = 0.21 | -2 pp | 0.713 | 1.000 | 0.710 |

**C-0.9 cache cells only (S32:C-0.9, S16:C-0.9)**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 7/22 = 0.32 | 1/10 = 0.10 | +22 pp | 0.193 | 0.380 | 0.196 |
| CL vs CGd (dense counterpart of cache-GAINED) | 7/22 = 0.32 | 2/10 = 0.20 | +12 pp | 0.405 | 0.681 | 0.406 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 7/22 = 0.32 | 5/24 = 0.21 | +11 pp | 0.305 | 0.508 | 0.306 |

**high-confidence labels only**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 5/31 = 0.16 | 0/9 = 0.00 | +16 pp | 0.258 | 0.570 | 0.261 |
| CL vs CGd (dense counterpart of cache-GAINED) | 5/31 = 0.16 | 2/8 = 0.25 | -9 pp | 0.863 | 0.617 | 0.862 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 5/31 = 0.16 | 3/20 = 0.15 | +1 pp | 0.619 | 1.000 | 0.622 |

**pass 2 labels**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 10/38 = 0.26 | 1/10 = 0.10 | +16 pp | 0.262 | 0.416 | 0.256 |
| CL vs CGd (dense counterpart of cache-GAINED) | 10/38 = 0.26 | 2/10 = 0.20 | +6 pp | 0.517 | 1.000 | 0.519 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 10/38 = 0.26 | 5/24 = 0.21 | +5 pp | 0.431 | 0.764 | 0.433 |

**labels agreed in both passes only**

| contrast (A vs B) | (b) share A | (b) share B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs DL (dense word-lock control lost) | 9/37 = 0.24 | 1/10 = 0.10 | +14 pp | 0.308 | 0.664 | 0.309 |
| CL vs CGd (dense counterpart of cache-GAINED) | 9/37 = 0.24 | 2/10 = 0.20 | +4 pp | 0.570 | 1.000 | 0.571 |
| CL vs pooled dense-wrong (DL + CGd + DGd) | 9/37 = 0.24 | 5/24 = 0.21 | +3 pp | 0.503 | 1.000 | 0.491 |

Additional references (primary labels):

| contrast | (b) A | (b) B | diff | Fisher p (A>B) | two-sided p | perm p |
|---|---|---|---|---|---|---|
| CL vs CSF (cache fail->fail, other class of cache-changed failing programs) | 9/38 = 0.24 | 0/6 = 0.00 | +24 pp | 0.230 | 0.319 | 0.227 |
| CGd vs DL (dense wrong where cache fixed vs dense control lost) | 2/10 = 0.20 | 1/10 = 0.10 | +10 pp | 0.500 | 1.000 | 0.500 |
| CL vs DSF (dense-control fail->fail) | 9/38 = 0.24 | 1/3 = 0.33 | -10 pp | 0.858 | 1.000 | 0.856 |

### 7b. Pre-registered decision rule applied to the primary analysis

(b) share in CL = 0.24; in DL = 0.10; in CGd = 0.20; one-sided Fisher p CL>DL = 0.322, CL>CGd = 0.587.  **Verdict by the pre-registered rule: NOT SUPPORTED.**

The same rule on each sensitivity variant (for transparency; the pre-registered verdict is the primary one):

| variant | CL (b) | DL (b) | CGd (b) | p CL>DL | p CL>CGd | rule outcome |
|---|---|---|---|---|---|---|
| primary (pass 1, all labelled items) | 9/38 | 1/10 | 2/10 | 0.322 | 0.587 | NOT SUPPORTED |
| excluding the 12 hypothesis-generating tasks (4, 9, 16 + 9 named in diag_he_results.md) | 4/18 | 0/6 | 2/10 | 0.288 | 0.642 | NOT SUPPORTED |
| (e) removed from denominators (logic errors only) | 9/35 | 1/7 | 2/10 | 0.461 | 0.535 | UNRESOLVED |
| strict (b) only (dup, dead, logic, init, empty) | 6/38 | 0/10 | 1/10 | 0.225 | 0.546 | NOT SUPPORTED |
| one item per task per class | 5/27 | 1/10 | 2/10 | 0.475 | 0.727 | NOT SUPPORTED |
| C-0.9 cache cells only (S32:C-0.9, S16:C-0.9) | 7/22 | 1/10 | 2/10 | 0.193 | 0.405 | WEAKLY SUPPORTED |
| high-confidence labels only | 5/31 | 0/9 | 2/8 | 0.258 | 0.863 | NOT SUPPORTED |
| pass 2 labels | 10/38 | 1/10 | 2/10 | 0.262 | 0.517 | UNRESOLVED |
| labels agreed in both passes only | 9/37 | 1/10 | 2/10 | 0.308 | 0.570 | NOT SUPPORTED |

### 7c. CL items labelled (b) in pass 1 (note = my one-line justification; the programs themselves stay local)

| id | task | cell(s) | conf | note |
|---|---|---|---|---|
| c3253b | HumanEval/4 | S32:C-0.8, S32:C-0.9 | H | absolute_deviations already a scalar sum, return sums it again |
| 937e16 | HumanEval/9 | S32:C-0.8, S32:C-0.9 | H | b-empty rolling_max starts [] then rolling_max[-1] read in first iteration |
| 0c568f | HumanEval/69 | S32:C-0.9 | H | b-logic assign greatest_num=num before the guard if num>greatest_num which is then dead |
| 7dad86 | HumanEval/72 | S32:C-0.9 | L | b-logic undefined is_palindromic and early return makes the weight check dead code |
| af020f | HumanEval/105 | S32:C-0.9 | H | b-type dict keys are str but looked up with int digits |
| bbe9f9 | HumanEval/69 | S32:Lctx-0.8 | H | b-logic assigns greatest_integer=num before the guard num>greatest_integer (dead guard), same pattern as earlier item |
| a53df0 | HumanEval/128 | S32:Lctx-0.8 | L | b-dead result initialised to 0 and returned times sign but never accumulated; zero sign also *1 (second slip) |
| 61ae7e | HumanEval/105 | S16:C-0.9 | L | b-index 0-based list indexed with 1-based digits (offset slip; relation between list and index) |
| 3c17ee | HumanEval/114 | S16:C-0.9 | L | b-init current_sum starts at -inf and loop skips nums[0]; init conflicts with the min recurrence |

### 7d. CL items labelled (a), (c), (d), (e) in pass 1

| id | task | cell(s) | label | conf | note |
|---|---|---|---|---|---|
| a861a0 | HumanEval/12 | S32:C-0.9, S32:Lctx-0.8 | c | L | compares == to max_length which is never updated; needs operator and update changes, lines not in conflict |
| d9c9bd | HumanEval/16 | S16:C-0.8, S16:C-0.9, S32:C-0.8, S32:C-0.9 | a | H | drops .lower() |
| db2fc8 | HumanEval/25 | S32:C-0.9 | c | H | only handles factor 3 then appends rest; wrong plan |
| 5e5069 | HumanEval/43 | S32:C-0.8, S32:C-0.9, S32:Lctx-0.8 | a | H | num in seen should be -num |
| cbfb7b | HumanEval/47 | S16:C-0.8, S16:C-0.9, S32:C-0.8, S32:C-0.9 | a | H | median without sorting |
| 2ff192 | HumanEval/49 | S32:C-0.9 | c | H | iterates digits of n; wrong algorithm |
| a8e3be | HumanEval/56 | S32:C-0.9 | a | H | elif char == ')' instead of '>' |
| 7b344a | HumanEval/79 | S32:C-0.9 | a | H | final return lacks "db" prefix; wrong in isolation (zero branch right) |
| c249c7 | HumanEval/82 | S32:C-0.8, S32:C-0.9 | a | H | 3 is treated as non prime (length % 3 == 0 before a ==3 case) |
| 47e98d | HumanEval/85 | S32:C-0.8, S32:C-0.9 | a | H | range(lst) instead of range(len(lst)) |
| f92494 | HumanEval/114 | S32:C-0.9 | c | H | greedy reset-on-negative update rule is the wrong algorithm |
| 58c447 | HumanEval/128 | S32:C-0.9 | a | H | zero treated as negative; one missing branch |
| 0b731c | HumanEval/5 | S32:Lctx-0.8 | a | H | number and delimiter appends swapped; one adjacent hunk |
| 833a2b | HumanEval/11 | S32:Lctx-0.8 | c | L | nested loops over a and b plus shadowed char==char; whole plan wrong not a relation |
| 303e18 | HumanEval/16 | S32:Lctx-0.8 | a | H | drops .lower() |
| fb4fa8 | HumanEval/21 | S32:Lctx-0.8 | a | H | typo dctest in the trailing __main__ block (NameError when run as script) |
| 6aa5a4 | HumanEval/25 | S32:Lctx-0.8 | c | H | hard-coded prime ladder plus append n-1; wrong plan |
| 4fa300 | HumanEval/48 | S32:Lctx-0.8 | e | H | `import doctestmod()` syntax error in trailing main block |
| 840e1a | HumanEval/49 | S32:Lctx-0.8 | e | H | stray closing fence line in extracted code |
| 06bf7b | HumanEval/60 | S32:Lctx-0.8 | a | H | missing //2 |
| 67099a | HumanEval/72 | S32:Lctx-0.8 | a | H | no else branch so non-palindrome returns None instead of False |
| 880b25 | HumanEval/73 | S32:Lctx-0.8 | e | H | stray closing fence line in extracted code |
| a082cf | HumanEval/74 | S32:Lctx-0.8 | a | H | elif total2 < total2 typo |
| 1096df | HumanEval/82 | S32:Lctx-0.8 | a | H | length 2 treated as non prime; one missing case |
| f29a8b | HumanEval/89 | S32:Lctx-0.8 | a | H | chr((ascii+shift)%26) lacks ord('a') offset; one expression |
| e566fe | HumanEval/158 | S32:Lctx-0.8 | c | L | garbled: compares len(unique_chars) with itself, max_word unbound; lines individually meaningless |
| ce2cb3 | HumanEval/25 | S16:C-0.9 | c | H | hardcoded prime ladder, general factorisation plan missing; consistent but wrong plan |
| 91303d | HumanEval/9 | S16:C-0.9 | c | H | returns only the final maximum, no rolling list; wrong plan |
| 5b4467 | HumanEval/92 | S16:C-0.9 | c | H | compares each number to x+y only; coherent wrong plan |

### 7e. The collaborator's three named tasks as labelled (blind)

- HumanEval/4 (CL; c3253b): pass 1 = b, pass 2 = b; absolute_deviations already a scalar sum, return sums it again
- HumanEval/9 (CL; 937e16): pass 1 = b, pass 2 = b; b-empty rolling_max starts [] then rolling_max[-1] read in first iteration
- HumanEval/9 (CL; 91303d): pass 1 = c, pass 2 = c; returns only the final maximum, no rolling list; wrong plan
- HumanEval/16 (CL,DL; d9c9bd): pass 1 = a, pass 2 = a; drops .lower()
- HumanEval/16 (CL; 303e18): pass 1 = a, pass 2 = a; drops .lower()

## 8. S16:Lctx-0.8 (broken-code transfer failure), reported separately

LOST 32, GAINED 4. Programs that do not parse: 72/164 (stray fence line among them: 9); among its LOST programs 22/32 do not parse. Edit classes and signatures are in the group tables above ("S16:Lctx-0.8 (transfer)").

## 9. Parse / stray-fence share per cell (all 164)

| cell | does not parse | stray fence line in extracted code |
|---|---|---|
| D:C-0.9 | 2 | 0 |
| S32:C-0.9 | 5 | 0 |
| S32:Lctx-0.8 | 11 | 6 |
| S32:C-0.8 | 6 | 2 |
| S16:C-0.9 | 2 | 1 |
| S16:C-0.8 | 4 | 2 |
| D:C-0.8 | 4 | 2 |
| D:Lctx-0.8 | 8 | 5 |
| S16:Lctx-0.8 | 72 | 9 |

## 10. Supplementary analyses (NOT pre-registered; added after seeing the labels, exploratory)

### 10a. Whole category distribution, CL vs the pooled dense-version-wrong programs

| category | CL | pooled dense-wrong | two-sided Fisher p |
|---|---|---|---|
| (a) | 16/38 = 0.42 | 9/24 = 0.38 | 0.794 |
| (b) | 9/38 = 0.24 | 5/24 = 0.21 | 1.000 |
| (c) | 10/38 = 0.26 | 5/24 = 0.21 | 0.764 |
| (e) | 3/38 = 0.08 | 3/24 = 0.12 | 0.669 |

Overall chi-square statistic 3.80; permutation p (10,000, seed 20261108) = 0.466. The distribution of failure types in cache-LOST programs is not distinguishable from that of the dense decoder's own wrong programs at this n.

### 10b. Tail-block artefacts (stray fence, syntax error or typo in a trailing `__main__` block, doctest-hostile docstrings)

The scorer runs the whole extracted code as a script, so a trailing block can fail a program whose function is right. Automatic: share of programs that carry any top-level statement beyond defs/imports (a `__main__` block or top-level print). Manual (from labels): CL / DL items whose note names a fence, syntax error, `__main__` typo or doctest.

| set | n | has tail block | stray fence | does not parse |
|---|---|---|---|---|
| cache LOST | 81 | 22 (27%) | 2 (2%) | 4 (5%) |
| cache GAINED | 46 | 16 (35%) | 2 (4%) | 2 (4%) |
| cache CHANGED-SAME-PASS | 172 | 53 (31%) | 0 (0%) | 0 (0%) |
| cache CHANGED-SAME-FAIL | 401 | 135 (34%) | 7 (2%) | 22 (5%) |
| cache IDENTICAL | 120 | 2 (2%) | 0 (0%) | 0 (0%) |
| dense-control LOST | 11 | 3 (27%) | 4 (36%) | 4 (36%) |
| dense-control GAINED | 10 | 3 (30%) | 0 (0%) | 0 (0%) |

CL: items whose labelled defect is in the tail/formatting (fence, syntax error, main-block typo; labels a/e): 4/38: fb4fa8 (HumanEval/21), 4fa300 (HumanEval/48), 840e1a (HumanEval/49), 880b25 (HumanEval/73)

DL: items whose labelled defect is in the tail/formatting (fence, syntax error, main-block typo; labels a/e): 3/10: cb1ccb (HumanEval/73), 25f3c8 (HumanEval/49), 3fcdef (HumanEval/114)

A failure at HumanEval/51 (a dense-control program with a correct body but a `doctest.testmod()` main block) was labelled (d) blind and is one of the two failing programs the reading missed. Checked after unblinding by running only the prompt docstring with that main block: `doctest.testmod()` raises `ValueError: line 6 of the docstring ... has inconsistent leading whitespace` (the docstring contains a literal `\n` inside an example), exit code 1, so the program is scored as failed although its function is right. The other missed failure (HumanEval/116, negative numbers) is a spec/test mismatch.

### 10c. Offset-in-block of the first divergence, word-lock-matched view (C-0.9 cache cells vs the dense controls)

| group | n | offset 0-7 | 8-15 | 16-23 | 24-31 | d < 32 |
|---|---|---|---|---|---|---|
| S32:C-0.9 + S16:C-0.9, all changed | 281 | 30 (11%) | 100 (36%) | 77 (27%) | 74 (26%) | 102 (36%) |
| S32:C-0.9 + S16:C-0.9, LOST | 32 | 4 (12%) | 12 (38%) | 9 (28%) | 7 (22%) | 8 (25%) |
| dense controls, all changed | 167 | 47 (28%) | 57 (34%) | 39 (23%) | 24 (14%) | 39 (23%) |
| offset 0-7: C-0.9 cache cells all changed vs dense controls (one-sided p tests A > B) | 30/281 | 47/167 | 1.000 | 0.000 |

The first-quarter share is lower for cache-changed programs than for dense-control-changed programs (cache-caused divergences are less likely right after a dense refresh), but cache LOST programs do not differ from other cache-changed programs. Pair-level, descriptive.
