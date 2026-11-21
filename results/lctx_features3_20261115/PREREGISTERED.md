# Does information from OUTSIDE the position's output distribution (token provenance, math structure, the neighbour window, call-relative ranks) improve the learned word lock, and how much could any per-position feature gain at most (one-call lookahead reference)? Offline, CPU only, on the existing G1 traces (written 2026-11-15, before any code of this directory and before any feature below has been computed)

**Words.** WORD LOCK = which blanks are written per call. **Lctx** = the HistGradientBoosting word lock (200 rounds, lr 0.1, no early stopping, seed 23; `features2_lib.default_template()`); **six** = its six inputs (confidence, entropy, margin, dist_written, local_mask_ratio, pos_in_block). **G1** = the one-word-per-call traces of `lctx_features2_20261111` (80 MATH level-5 questions, 60 fit / 20 threshold by `manifests/split.json`, 512 tokens, 32-blocks, P4; `records/traces_G1/*.npz` on the HPC). **G2** = the same 80 questions decoded by C-0.8 multi-write (`records/traces_G2/`). Label y = the proposal at that call equals the final token. **Share** = the share of rows admitted at the pooled precision of C-0.8 on the threshold rows (the lambda rule of the earlier ablations). STATE at call t = the response tokens written before call t (a position written at call s < t holds the token of its `written_now` row at call s); every other response position is a blank.

## Why (evidence in hand)
- `lctx_features2_20261111` and its addenda: 30 decode-time inputs, two learners, five precision targets: a learned per-position lock admits at most 2-3 points more than C-0.8 at the same precision, and the only resolved gain comes from the neighbours' confidence (group b, +0.92 points [+0.18, +1.41] under the MLP). Groups a, c, d, e, f, g are all readings of the output distribution at the position or of the forward pass that forms it; a tree is invariant to monotone transforms of a single input, so such features add nothing unless they carry new information.
- `lctx_setlock_20261112` (set-level label by answer harm): failed; the placebo harms as much as c0.5. The label route is closed at this setting.
- `lctx_setlock_20261112/validation/vote_words_branches.md` (exploratory): extra words that were wrong were numbers more often than words (error rate 15.9% vs 12.4%).
The one unused source of information is WHERE the proposed token comes from and WHAT surrounds it in the text: whether it is copied from the question or computed, whether it sits right after `=`, whether its neighbours form a confident run. None of the 30 inputs reads the prompt or the written text. And none of the experiments so far says how much ANY per-position feature could gain at all; the lookahead reference below bounds that from the side of information the decoder would have one call later.

## Questions
- **Q1 (gate).** Does any of four new groups, computed from the trace tokens, the prompt and the tokenizer only (no forward pass), raise the share at C-0.8's precision over six by >= 5 points, beyond capacity and shuffle controls?
- **Q2 (reference, descriptive).** How much do features the decoder only has ONE CALL LATER (and the neighbours' fates) raise the share over six? This is a reference for the size of the information that per-position features miss, not a candidate rule.
- **Q3 (shift diagnostic).** Are the call-relative features (group j) distributed the same in G1 and G2 (the history features of `lctx_features2` moved up to 16x)?

## Rows (fixed here; differs from the earlier ablations, reason given)
Primary row filter: every G1 row whose proposal is not EOS/EOT **AND that is not written on its own call** (`written_now = False`). In one-word-per-call traces the written row has y = 1 by construction and is the call's most confident blank, so any feature relative to the call (group j: rank 1 within the call) would identify it and gain from the decode's own choice, not from information. The earlier ablations kept those rows (512 of 8,448 per question); for comparability the six control is ALSO fitted on the old filter and must reproduce `lctx_features2_20261111/validation/ablation_G1.json`'s six numbers (V2). Every set of this document is fitted and thresholded on the new filter, with the C-0.8 target re-computed on it.

## Features (all per row, all from (call, position, proposal token, confidence of the eligible blanks at the call), the reconstructed STATE, the question's prompt token ids (`common.prompt_ids`, the collection's chat template) and a tokenizer table built once (`assets/token_info.npz`, sha256 recorded); NaN where undefined; native NaN in the learner)
Tokenizer table, per vocabulary id, from the decoded string (special tokens and partial UTF-8 tokens get 0 everywhere): `digit` (contains a digit), `eq` (contains `=`), `op` (contains any of `+ - * / ^ < >` or the substrings `\times`, `\cdot`, `\frac`), `dollar` (count of `$`), `lbrace` / `rbrace` (counts of `{` / `}`).
- **(h) PROVENANCE** (where the proposal could come from)
  - `h_in_prompt` (0/1): the proposal id occurs among the prompt ids.
  - `h_in_written` (0/1): the proposal id occurs among the written response tokens of the STATE.
  - `h_bigram_prompt` (0/1/NaN): the pair (token at position-1, proposal) occurs as consecutive ids in the prompt; NaN if position-1 is a blank (position 0 uses the last prompt token).
  - `h_bigram_written` (0/1/NaN): the same pair occurs in the written response tokens (consecutive written positions only).
  - `h_num_novel` (0/1): the proposal is a `digit` token AND does not occur among the prompt ids (a computed value, not a copied one).
- **(i) STRUCTURE** (what surrounds the position in the written text)
  - `i_digit` (0/1): the proposal is a `digit` token.
  - `i_after_eq` (0/1/NaN): the nearest written token to the left, skipping tokens that decode to whitespace only, at most 3 positions back, is an `eq` token; 0 if it is written and not `eq`; NaN if no written token within 3 positions.
  - `i_after_op` (0/1/NaN): the same for `op`.
  - `i_in_math` (0/1): the number of `$` in the written tokens left of the position is odd.
  - `i_brace_depth` (int): sum of `lbrace` minus `rbrace` over the written tokens left of the position, clipped at 0.
- **(k) NEIGHBOUR WINDOW** (extends group b, the only group with a resolved gain; from the confidences of the eligible blanks at the same call)
  - `k_min_w2`: minimum confidence over the other eligible blanks within +-2 positions (NaN if none).
  - `k_mean_w4`: mean confidence over the other eligible blanks within +-4 positions (NaN if none).
  - `k_run`: length of the maximal run of consecutive positions, containing this one, in which every position is either written or an eligible blank with confidence >= 0.5 (0 if this blank's own confidence < 0.5).
  - `k_written_w4`: number of written positions within +-4.
- **(j) CALL-RELATIVE** (invariant by construction to how many words a call writes)
  - `j_rank_frac`: rank of the confidence among the call's eligible blanks, divided by their number (1 = most confident).
  - `j_conf_minus_med`: confidence minus the median confidence of the call's eligible blanks.
  - `j_block_written`: share of the current block's positions written in the STATE.
- **(O) LOOKAHEAD REFERENCE** (NOT decode-time; never a candidate)
  - `o_conf_next`: confidence of the same blank at call t+1 (rows not written at t are eligible at t+1 in this geometry; NaN otherwise).
  - `o_same_next` (0/1/NaN): the proposal at t+1 equals the proposal at t.
  - `o_nbr_y_l`, `o_nbr_y_r`: the label y of the nearest eligible blank on each side at call t (the neighbours' fates; NaN when none).

## Sets, learner, bootstrap (`orchestration/fit_ablation3.py`, to be written; learner, lambda rule and 2,000-resample cluster bootstrap over the 20 threshold questions (seed 23, target and lambda re-chosen per resample) as in `lctx_features2_20261111/orchestration/fit_ablation.py`)
- Controls and references: `conf`, `six`, `six+b` (the existing neighbour group, from the traces).
- **Candidates:** `six+h`, `six+i`, `six+k`, `six+j`, `six+hikj`. Secondary: `six+b+k` minus `six+b` (does the window add beyond the nearest neighbours), `six+b+hikj`.
- **Capacity / shuffle controls** (width of `six+hikj` = 17 columns): `six+noise17` (17 iid uniform columns, seed 23) and `six+hikj_shuf` (the 17 columns permuted within question, fit and threshold rows alike).
- **Reference:** `six+O`, `six+b+hikj+O`.
Readouts per set: AUROC; share at C-0.8's pooled precision and the achieved precision; paired differences to `six` and to `conf`; permutation importance (AUROC drop, 3 repeats) for `six+hikj` and `six+b+hikj`. Cost: wall time of computing groups h, i, k, j per call from the STATE (vectorised numpy, CPU), reported in ms.

## Validity (`validation/validity3.json`; any failure stops before a number is read)
- **V1** STATE reconstruction: in every G1 question each response position is written exactly once, and the written token equals that row's `final_token` (one word per call: a written token is final).
- **V2** the six control on the OLD filter reproduces `ablation_G1.json` (AUROC 0.935392, share 0.395136) to 1e-6.
- **V3** neighbour consistency: where the nearest left/right eligible blank lies within +-2, `k_min_w2` <= that blank's `nbr_conf_l` / `nbr_conf_r`; `k_run` >= 1 exactly when confidence >= 0.5.
- **V4** ranges: `j_rank_frac` in (0, 1], `j_block_written` in [0, 1], `i_brace_depth` >= 0, every 0/1 feature in {0, 1, NaN}; the prompt ids are those of `common.prompt_ids` (sha256 per question stored).

## Gate (`orchestration/check_gate3.py`; fails closed; the same conditions as `lctx_features2_20261111` so the verdicts are comparable)
Best set = the candidate with the largest share (point estimate). The gate passes only if all of:
- **G0** V1-V4 hold.
- **G1** best set's share minus `six`'s >= 0.05.
- **G2** its AUROC minus `six`'s >= +0.01 with the 2.5th percentile of the paired bootstrap > 0.
- **G3** `six+noise17` and `six+hikj_shuf` each gain < 0.01 share over `six`.
- **G4** the per-call cost of the best set's new features <= 4 ms.

## Predictions (point [80% range]; mine, before any feature is computed)
- Share over `six` on the new filter, points: `six+h` +1.0 [0, +3]; `six+i` +0.5 [-0.3, +2]; `six+k` +1.5 [+0.5, +3]; `six+j` +0.5 [-0.5, +2]; `six+hikj` +2.5 [+1, +5]; `six+b+k` minus `six+b` +0.3 [-0.3, +1].
- Reference `six+O`: +12 [+5, +25]; P(reference >= +5 points) = 75%.
- Importance: `h_num_novel` and `k_min_w2` the largest of the new features.
- **P(gate passes) = 15%.** The likely failure is G1 again: provenance is new information, but at precision 0.996 the admitted rows are nearly all correct already, and the remaining errors depend on words not yet written.

## Decision rules
- **Gate passes:** the rule is written (`rules/Z/`, sha256) and a decode stage is pre-registered separately before any decode (deployment geometry, MATH `math_l5_eval54` and HumanEval 164, the frontier rule of the earlier stages). Group j's G2 shift is read before that document is written.
- **Gate fails, reference `six+O` >= +5 points:** the information exists one call later but cheap decode-time features do not reach it. The per-position feature line is closed; the remaining route is DEFERRAL (decide on a blank one call later), which is a different question and needs its own document.
- **Gate fails, reference `six+O` < +5 points:** even one-call lookahead and the neighbours' fates cannot move the share by the gate's size; the per-position word lock with this label is closed for good, and no further feature work is done on it.

## Not claimed / known limits
(1) One-word-per-call geometry; the deployment decoder writes several words per call; only group j is built to be invariant, and only the G2 diagnostic checks it. (2) The 20 threshold questions are the same as in the three earlier ablations; the new features were designed without looking at their rows, but the question set itself is not fresh. (3) MATH level 5 only: groups h and i are math-flavoured by design, and their transfer to code is not tested here (HumanEval is the decode evaluation set). (4) The lookahead reference is not an upper bound on every feature: it bounds what one call of waiting and the neighbours' fates would add, which is the information the earlier closing statements said was missing. (5) The token table's `op` list is fixed here and not tuned. (6) LLaDA-8B-Instruct only.

## Outcome (appended 2026-11-15 after the run on the login node, 283 s; `validation/ablation3_G1.md`, `validation/gate3.json`)
**Gate: FAILS** on G1 and G2 (G0, G3, G4 pass). Best candidate `six+hikj`: share +1.23 points [+0.67, +1.97] over `six`, AUROC +0.0017 [+0.0011, +0.0023]; over `six+b` +0.22 [-0.37, +1.16].
- **Validity.** V1: all 80 questions reconstruct (512 positions written once, written token = final). V2: the six control on the old filter reproduces `ablation_G1.json` exactly (AUROC 0.935392, share 0.395136, difference 0.0). V3: 0 violations. V4: no range violation; prompt length = first response position in all 80.
- **Rows.** New filter: fit 458,265, threshold 148,386; C-0.8 pooled precision 0.99514, share 34.6%; raw-confidence AUROC 0.9268; `six` share 36.1%.
- **Per group (share over `six`, points).** h (provenance) -0.04 [-0.33, +0.45]; i (structure) +0.18 [-0.33, +0.44]; k (window) +0.92 [-0.08, +1.99]; j (call-relative) +0.69 [-0.15, +1.45]; `six+b` +1.01 [+0.28, +1.68]; `six+b+k` minus `six+b` +0.15 [-0.32, +0.78]. Controls: noise17 +0.03, hikj_shuf +0.05 (the design resolves differences of this size).
- **Importance.** In `six+hikj` the new information is group k (AUROC drop 0.030, mostly `k_mean_w4`); groups h (0.0005), i (0.0006) and j (0.002) are near zero. With group b present, k's drop falls to 0.008: the window repeats what the nearest neighbours already say.
- **Lookahead reference.** `six+O` +10.39 points (AUROC +0.023 [+0.019, +0.027]); its share interval is unstable ([-34.9, +12.6], P(>0) = 0.94: in about 6% of resamples the lambda rule admits almost nothing), so the point is read with that caveat; `six+b+hikj+O` +14.03 [+10.76, +16.25] is stable. Information about the blank's fate that the decoder has one call later is worth about ten points of share; `o_conf_next` carries most of it (importance 0.21, larger than confidence).
- **G2 shift (Q3).** Group j keeps its distribution under multi-write (`j_rank_frac` median 0.50 -> 0.46, `j_conf_minus_med` -0.0006 -> -0.007), as built; group k does not (`k_min_w2` median 0.35 -> 0.15, `k_run` p90 407 -> 54), because the blanks left after a multi-write call have less confident neighbours. G1 models on G2 not-written rows: six 0.841, six+hikj 0.844, six+b+hikj 0.848.
- **Predictions vs outcome.** h +1.0 [0, +3]: missed (-0.04); i +0.5 (+0.18, inside); k +1.5 (+0.92, inside); j +0.5 (+0.69, inside); hikj +2.5 [+1, +5] (+1.23, inside); b+k minus b +0.3 (+0.15, inside); reference +12 [+5, +25] (+10.4, inside); importance led by `h_num_novel` and `k_min_w2`: missed (k_mean_w4 leads, h near zero); P(gate) 15%: did not pass.
- **Decision (rule "gate fails, reference `six+O` >= +5 points").** Information about a blank's fate exists one call later (about +10 points of share) but cheap decode-time features do not reach it: provenance and math structure add nothing, the window adds only what group b already gives. The per-position feature line is closed. The remaining route is DEFERRAL (decide on a blank one call later, using its next-call confidence), which needs its own document; nothing is run under this one.

### Correction to the decision (appended 2026-11-15, EXPLORATORY check `orchestration/explore_deferral.py`, `validation/explore_deferral.md`, before any deferral document was written)
The reading "deferral is the remaining route" does not hold. Deferring means deciding on the blank at call t+1, where the decoder knows the BACKWARD history of that row (its previous confidence `p_conf_prev`, whether its proposal changed `p_same_prev`); the reference instead gave the call-t row FORWARD information. Measured on the same rows, split, learner and lambda rule (500 resamples): `six+P` -0.10 points [-0.45, +0.23] over `six`; `six+a+P` +0.25 [-0.05, +0.59]; `six+b+P` -0.37 [-0.60, +0.13] against `six+b`. The backward history adds nothing, while forward `six+P+O` gives +12.27 [+10.31, +14.12]. So the +10 points of the reference measure only that a later call is better informed, and the confidence rule at that later call already uses that information. No deferral experiment is pre-registered: its premise fails offline. Corrected decision: the per-position word lock with this label is closed, including deferral.
