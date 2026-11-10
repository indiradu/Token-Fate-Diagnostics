# Analysis 2: oracle repair at the first divergence

HumanEval 0-163, 512 tokens. S32 representation cache (dense call at every block start) vs the dense decode of the same word lock (S32 engine, dense call every call, first-EOS truncation = the dense decoder's tokens up to the first EOS).
A task's FIRST DIVERGENCE is the first call at which the cached decode's write event (which blank positions were given which token) differs from the dense decode's. The cached decode is resumed from the last block-start dense call before it with one row class repaired.
`oracle`: the class's cached K/V are replaced by those of a dense pass over the current tokens. `compute`: the class's rows are computed in the call instead of read from the cache. `seg`: repaired only in the calls of the block of the divergence; `end`: from there to the end of the decode.
RESTORED = the repaired decode's write events equal the dense decode's through the first-divergence call (its own first divergence from dense is later than the original, or none). PASSES = the final program passes the tests.

## Word lock C-0.9

Census tasks: 164; lost tasks (dense passes, S32 fails): 17. Census parity with stage 1: dense decode equal to the stage-1 dense tokens up to the first EOS on 164/164, cached decode equal to the stage-1 S32 tokens on 164/164.

### First divergence of the cached decode from the dense decode

Position = answer position (token index in the generated text) of the lowest position whose write differs; fraction = position / dense answer length (first EOS); calls after block start = divergence call - dense call that opened the block. kind: token = the two decodes wrote different tokens at the same position; a_only / b_only = the cached / the dense decode wrote a position the other did not (word-lock set difference).

| tasks | n | diverged | kinds | answer position | fraction of dense answer | position in block | block index | calls after block start |
|---|---|---|---|---|---|---|---|---|
| lost | 17 | 17 | {'b_only': 8, 'a_only': 9} | 14 [10, 17] | 0.06 [0.05, 0.09] | 14 [10, 16] | 0 [0, 0] | 1 [1, 3] |
| retained (all other census tasks) | 147 | 147 | {'a_only': 51, 'b_only': 92, 'token': 4} | 12 [8, 14] | 0.05 [0.03, 0.09] | 12 [8, 14] | 0 [0, 0] | 1 [1, 2] |

Null control (resume from the dense call, nothing repaired, must equal the cached decode): 17/17.

### Repair of one row class, 17 lost tasks

RESTORED: write events equal the dense decode's through the original divergence call. POS TOKEN: the token at the first divergent position equals the dense token (defined only when the divergence is a position that the cached decode wrote: kind token or a_only). IDENTICAL: final tokens equal dense to the first EOS. PASSES: HumanEval tests pass. `class non-empty`: tasks where the repaired class had rows at some repaired call (oracle) or added rows to the query set (compute). Wilson 95% intervals.

| class | mode | window | tasks | class non-empty | RESTORED | POS TOKEN | IDENTICAL | PASSES |
|---|---|---|---|---|---|---|---|---|
| (a) blank rows of the next block | oracle | seg | 17 | 17 | 6/17 (0.17-0.59) | 8/9 (0.56-0.98) | 0/17 (0.00-0.18) | 4/17 (0.10-0.47) |
| (a) blank rows of the next block | oracle | end | 17 | 17 | 6/17 (0.17-0.59) | 8/9 (0.56-0.98) | 1/17 (0.01-0.27) | 12/17 (0.47-0.87) |
| (a) blank rows of the next block | compute | seg | 17 | 17 | 6/17 (0.17-0.59) | 8/9 (0.56-0.98) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (a) blank rows of the next block | compute | end | 17 | 17 | 6/17 (0.17-0.59) | 8/9 (0.56-0.98) | 1/17 (0.01-0.27) | 10/17 (0.36-0.78) |
| (b) blank rows beyond the next block (and past-EOS blanks) | oracle | seg | 17 | 17 | 2/17 (0.03-0.34) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 4/17 (0.10-0.47) |
| (b) blank rows beyond the next block (and past-EOS blanks) | oracle | end | 17 | 17 | 2/17 (0.03-0.34) | 7/9 (0.45-0.94) | 2/17 (0.03-0.34) | 8/17 (0.26-0.69) |
| (b) blank rows beyond the next block (and past-EOS blanks) | compute | seg | 17 | 17 | 1/17 (0.01-0.27) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (b) blank rows beyond the next block (and past-EOS blanks) | compute | end | 17 | 17 | 1/17 (0.01-0.27) | 7/9 (0.45-0.94) | 1/17 (0.01-0.27) | 7/17 (0.22-0.64) |
| (c) written rows of the current block older than one step | oracle | seg | 17 | 17 | 0/17 (0.00-0.18) | 6/9 (0.35-0.88) | 1/17 (0.01-0.27) | 4/17 (0.10-0.47) |
| (c) written rows of the current block older than one step | oracle | end | 17 | 17 | 0/17 (0.00-0.18) | 6/9 (0.35-0.88) | 2/17 (0.03-0.34) | 11/17 (0.41-0.83) |
| (c) written rows of the current block older than one step | compute | seg | 17 | 16 | 0/17 (0.00-0.18) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (c) written rows of the current block older than one step | compute | end | 17 | 16 | 0/17 (0.00-0.18) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 6/17 (0.17-0.59) |
| (d) prompt rows | oracle | seg | 17 | 17 | 2/17 (0.03-0.34) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (d) prompt rows | oracle | end | 17 | 17 | 2/17 (0.03-0.34) | 7/9 (0.45-0.94) | 1/17 (0.01-0.27) | 11/17 (0.41-0.83) |
| (d) prompt rows | compute | seg | 17 | 17 | 4/17 (0.10-0.47) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 2/17 (0.03-0.34) |
| (d) prompt rows | compute | end | 17 | 17 | 4/17 (0.10-0.47) | 7/9 (0.45-0.94) | 0/17 (0.00-0.18) | 8/17 (0.26-0.69) |
| (f) written rows of earlier blocks | oracle | seg | 17 | 1 | 0/17 (0.00-0.18) | 6/9 (0.35-0.88) | 0/17 (0.00-0.18) | 0/17 (0.00-0.18) |
| (f) written rows of earlier blocks | oracle | end | 17 | 17 | 0/17 (0.00-0.18) | 6/9 (0.35-0.88) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (f) written rows of earlier blocks | compute | seg | 17 | 1 | 0/17 (0.00-0.18) | 6/9 (0.35-0.88) | 0/17 (0.00-0.18) | 0/17 (0.00-0.18) |
| (f) written rows of earlier blocks | compute | end | 17 | 17 | 0/17 (0.00-0.18) | 6/9 (0.35-0.88) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (e) all cached blank rows (positive control) | oracle | seg | 17 | 17 | 7/17 (0.22-0.64) | 8/9 (0.56-0.98) | 0/17 (0.00-0.18) | 4/17 (0.10-0.47) |
| (e) all cached blank rows (positive control) | oracle | end | 17 | 17 | 7/17 (0.22-0.64) | 8/9 (0.56-0.98) | 3/17 (0.06-0.41) | 11/17 (0.41-0.83) |
| (e) all cached blank rows (positive control) | compute | seg | 17 | 17 | 5/17 (0.13-0.53) | 8/9 (0.56-0.98) | 0/17 (0.00-0.18) | 4/17 (0.10-0.47) |
| (e) all cached blank rows (positive control) | compute | end | 17 | 17 | 5/17 (0.13-0.53) | 8/9 (0.56-0.98) | 3/17 (0.06-0.41) | 12/17 (0.47-0.87) |
| (g) all cached rows (dense-equivalent control) | oracle | seg | 17 | 17 | 15/17 (0.66-0.97) | 9/9 (0.70-1.00) | 1/17 (0.01-0.27) | 5/17 (0.13-0.53) |
| (g) all cached rows (dense-equivalent control) | oracle | end | 17 | 17 | 15/17 (0.66-0.97) | 9/9 (0.70-1.00) | 13/17 (0.53-0.90) | 15/17 (0.66-0.97) |
| (g) all cached rows (dense-equivalent control) | compute | seg | 17 | 17 | 17/17 (0.82-1.00) | 9/9 (0.70-1.00) | 1/17 (0.01-0.27) | 5/17 (0.13-0.53) |
| (g) all cached rows (dense-equivalent control) | compute | end | 17 | 17 | 17/17 (0.82-1.00) | 9/9 (0.70-1.00) | 17/17 (0.82-1.00) | 17/17 (0.82-1.00) |

### Per task (oracle, window = seg): which classes restore the divergence

| task | first-divergence call | kind | answer position | next_blank | far2_blank | inblock_old | prompt | prior_written | all_blank | all_cached |
|---|---|---|---|---|---|---|---|---|---|---|
| HumanEval/105 | 2 | a_only | 10 | .f | .f | .P | .f | .f | .f | Rf |
| HumanEval/114 | 1 | b_only | 20 | Rf | .f | .f | .f | .f | Rf | Rf |
| HumanEval/12 | 1 | a_only | 12 | Rf | .f | .f | RP | .f | Rf | RP |
| HumanEval/128 | 2 | a_only | 13 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/16 | 9 | b_only | 80 | .P | .P | .P | .f | .f | .P | .P |
| HumanEval/25 | 1 | a_only | 19 | .f | .f | .f | .f | .f | Rf | Rf |
| HumanEval/4 | 1 | b_only | 14 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/43 | 1 | b_only | 5 | .f | .f | .f | .f | .f | Rf | Rf |
| HumanEval/47 | 1 | a_only | 10 | Rf | Rf | .f | .f | .f | Rf | Rf |
| HumanEval/49 | 1 | b_only | 28 | Rf | .f | .f | .f | .f | Rf | Rf |
| HumanEval/56 | 1 | b_only | 1 | .P | .P | .f | .P | .f | .P | RP |
| HumanEval/69 | 1 | a_only | 5 | .f | Rf | .f | Rf | .f | .f | Rf |
| HumanEval/72 | 3 | a_only | 14 | RP | .P | .P | .P | .f | .P | RP |
| HumanEval/79 | 3 | b_only | 17 | .f | .f | .P | .f | .f | .f | Rf |
| HumanEval/82 | 3 | a_only | 15 | .f | .f | .f | .f | .f | .f | .f |
| HumanEval/85 | 1 | a_only | 10 | RP | .P | .f | .f | .f | RP | RP |
| HumanEval/9 | 4 | b_only | 16 | .f | .f | .f | .f | .f | .f | Rf |

R = restored through the divergence call, . = not; P = final program passes, f = fails.

## Word lock Lctx-0.8

Census tasks: 17; lost tasks (dense passes, S32 fails): 17. Census parity with stage 1: dense decode equal to the stage-1 dense tokens up to the first EOS on 17/17, cached decode equal to the stage-1 S32 tokens on 17/17.

### First divergence of the cached decode from the dense decode

Position = answer position (token index in the generated text) of the lowest position whose write differs; fraction = position / dense answer length (first EOS); calls after block start = divergence call - dense call that opened the block. kind: token = the two decodes wrote different tokens at the same position; a_only / b_only = the cached / the dense decode wrote a position the other did not (word-lock set difference).

| tasks | n | diverged | kinds | answer position | fraction of dense answer | position in block | block index | calls after block start |
|---|---|---|---|---|---|---|---|---|
| lost | 17 | 17 | {'b_only': 6, 'a_only': 11} | 13 [10, 16] | 0.06 [0.05, 0.09] | 12 [10, 16] | 0 [0, 0] | 1 [1, 1] |

Null control (resume from the dense call, nothing repaired, must equal the cached decode): 17/17.

### Repair of one row class, 17 lost tasks

RESTORED: write events equal the dense decode's through the original divergence call. POS TOKEN: the token at the first divergent position equals the dense token (defined only when the divergence is a position that the cached decode wrote: kind token or a_only). IDENTICAL: final tokens equal dense to the first EOS. PASSES: HumanEval tests pass. `class non-empty`: tasks where the repaired class had rows at some repaired call (oracle) or added rows to the query set (compute). Wilson 95% intervals.

| class | mode | window | tasks | class non-empty | RESTORED | POS TOKEN | IDENTICAL | PASSES |
|---|---|---|---|---|---|---|---|---|
| (a) blank rows of the next block | oracle | seg | 17 | 17 | 1/17 (0.01-0.27) | 9/11 (0.52-0.95) | 0/17 (0.00-0.18) | 1/17 (0.01-0.27) |
| (a) blank rows of the next block | oracle | end | 17 | 17 | 1/17 (0.01-0.27) | 9/11 (0.52-0.95) | 2/17 (0.03-0.34) | 11/17 (0.41-0.83) |
| (a) blank rows of the next block | compute | seg | 17 | 17 | 2/17 (0.03-0.34) | 10/11 (0.62-0.98) | 0/17 (0.00-0.18) | 1/17 (0.01-0.27) |
| (a) blank rows of the next block | compute | end | 17 | 17 | 2/17 (0.03-0.34) | 10/11 (0.62-0.98) | 1/17 (0.01-0.27) | 9/17 (0.31-0.74) |
| (b) blank rows beyond the next block (and past-EOS blanks) | oracle | seg | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (b) blank rows beyond the next block (and past-EOS blanks) | oracle | end | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 7/17 (0.22-0.64) |
| (b) blank rows beyond the next block (and past-EOS blanks) | compute | seg | 17 | 17 | 1/17 (0.01-0.27) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 0/17 (0.00-0.18) |
| (b) blank rows beyond the next block (and past-EOS blanks) | compute | end | 17 | 17 | 1/17 (0.01-0.27) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 6/17 (0.17-0.59) |
| (c) written rows of the current block older than one step | oracle | seg | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 2/17 (0.03-0.34) |
| (c) written rows of the current block older than one step | oracle | end | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 1/17 (0.01-0.27) | 5/17 (0.13-0.53) |
| (c) written rows of the current block older than one step | compute | seg | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 1/17 (0.01-0.27) |
| (c) written rows of the current block older than one step | compute | end | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 1/17 (0.01-0.27) | 3/17 (0.06-0.41) |
| (d) prompt rows | oracle | seg | 17 | 17 | 2/17 (0.03-0.34) | 9/11 (0.52-0.95) | 0/17 (0.00-0.18) | 2/17 (0.03-0.34) |
| (d) prompt rows | oracle | end | 17 | 17 | 2/17 (0.03-0.34) | 9/11 (0.52-0.95) | 1/17 (0.01-0.27) | 6/17 (0.17-0.59) |
| (d) prompt rows | compute | seg | 17 | 17 | 1/17 (0.01-0.27) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 2/17 (0.03-0.34) |
| (d) prompt rows | compute | end | 17 | 17 | 1/17 (0.01-0.27) | 8/11 (0.43-0.90) | 1/17 (0.01-0.27) | 5/17 (0.13-0.53) |
| (f) written rows of earlier blocks | oracle | seg | 17 | 1 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 0/17 (0.00-0.18) | 0/17 (0.00-0.18) |
| (f) written rows of earlier blocks | oracle | end | 17 | 17 | 0/17 (0.00-0.18) | 8/11 (0.43-0.90) | 2/17 (0.03-0.34) | 7/17 (0.22-0.64) |
| (f) written rows of earlier blocks | compute | seg | 17 | 1 | 0/17 (0.00-0.18) | 9/11 (0.52-0.95) | 0/17 (0.00-0.18) | 1/17 (0.01-0.27) |
| (f) written rows of earlier blocks | compute | end | 17 | 17 | 0/17 (0.00-0.18) | 9/11 (0.52-0.95) | 2/17 (0.03-0.34) | 10/17 (0.36-0.78) |
| (e) all cached blank rows (positive control) | oracle | seg | 17 | 17 | 1/17 (0.01-0.27) | 9/11 (0.52-0.95) | 0/17 (0.00-0.18) | 4/17 (0.10-0.47) |
| (e) all cached blank rows (positive control) | oracle | end | 17 | 17 | 1/17 (0.01-0.27) | 9/11 (0.52-0.95) | 3/17 (0.06-0.41) | 11/17 (0.41-0.83) |
| (e) all cached blank rows (positive control) | compute | seg | 17 | 17 | 2/17 (0.03-0.34) | 9/11 (0.52-0.95) | 0/17 (0.00-0.18) | 3/17 (0.06-0.41) |
| (e) all cached blank rows (positive control) | compute | end | 17 | 17 | 2/17 (0.03-0.34) | 9/11 (0.52-0.95) | 1/17 (0.01-0.27) | 8/17 (0.26-0.69) |
| (g) all cached rows (dense-equivalent control) | oracle | seg | 17 | 17 | 15/17 (0.66-0.97) | 11/11 (0.74-1.00) | 1/17 (0.01-0.27) | 3/17 (0.06-0.41) |
| (g) all cached rows (dense-equivalent control) | oracle | end | 17 | 17 | 15/17 (0.66-0.97) | 11/11 (0.74-1.00) | 15/17 (0.66-0.97) | 16/17 (0.73-0.99) |
| (g) all cached rows (dense-equivalent control) | compute | seg | 17 | 17 | 17/17 (0.82-1.00) | 11/11 (0.74-1.00) | 1/17 (0.01-0.27) | 3/17 (0.06-0.41) |
| (g) all cached rows (dense-equivalent control) | compute | end | 17 | 17 | 17/17 (0.82-1.00) | 11/11 (0.74-1.00) | 17/17 (0.82-1.00) | 17/17 (0.82-1.00) |

### Per task (oracle, window = seg): which classes restore the divergence

| task | first-divergence call | kind | answer position | next_blank | far2_blank | inblock_old | prompt | prior_written | all_blank | all_cached |
|---|---|---|---|---|---|---|---|---|---|---|
| HumanEval/11 | 1 | a_only | 10 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/12 | 1 | a_only | 12 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/14 | 1 | a_only | 6 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/158 | 1 | a_only | 11 | .f | .f | .f | .f | .f | .P | Rf |
| HumanEval/16 | 4 | a_only | 73 | .f | .P | .f | Rf | .f | Rf | RP |
| HumanEval/21 | 1 | a_only | 6 | .f | .P | .P | .P | .f | .P | RP |
| HumanEval/25 | 1 | a_only | 16 | .f | .f | .f | .f | .f | .f | .f |
| HumanEval/43 | 1 | b_only | 17 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/48 | 1 | b_only | 10 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/5 | 1 | b_only | 12 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/58 | 1 | a_only | 14 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/60 | 1 | b_only | 30 | Rf | .f | .f | .f | .f | .f | Rf |
| HumanEval/69 | 1 | a_only | 3 | .f | .f | .f | .f | .f | .P | Rf |
| HumanEval/72 | 2 | a_only | 15 | .P | .P | .P | .P | .f | .P | .P |
| HumanEval/74 | 2 | b_only | 16 | .f | .f | .f | .f | .f | .f | Rf |
| HumanEval/82 | 2 | b_only | 26 | .f | .f | .f | Rf | .f | .f | Rf |
| HumanEval/87 | 1 | a_only | 13 | .f | .f | .f | .f | .f | .f | Rf |

R = restored through the divergence call, . = not; P = final program passes, f = fails.

