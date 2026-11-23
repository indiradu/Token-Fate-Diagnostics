# Lctx-F3 word lock: hand-over (2026-11-22)

This is the new word lock for the sweep: **Lctx-F3**. It is a frozen `ContextRule` in the same format as `Lctx_c0.8.pkl`, with the engine change it needs, already applied to your frozen engine `source_m1`.

## Corrections (2026-11-23), after your check

**1. "MATH-108" is two different sets, and F3 is not clean on yours.** You are right. Our MATH-108 is our `math_l5_eval54.jsonl` (sha256 `2d83e351…`, the `eval54_questions` of `provenance/split.json`) followed by `data/math500_levels1to4_54_seed23.jsonl` (sha256 `c07bbddb…`), in that order. Its level 1–4 half is the same 54 questions as your `math_l14.jsonl`. Its level-5 half is not your `math_l5.jsonl`: the two share 21 questions. Of the other 33 in yours, 24 are in our fit split and 9 in our threshold split. The old line "none is in MATH-108" was checked against our MATH-108, not yours. So:
- Every MATH number in this note and in our sweep is on **our** MATH-108, where F3 has seen none of the 108.
- On **your** MATH-108, F3 has seen 33 of the 54 level-5 questions. Use the 75 outside our split (21 level-5 + 54 level 1–4), as you planned.
- `manifests/math108_ours.json` lists our 108 IDs in order, with both source files and their sha256. `math500-l5-241` and `math500-l5-394` (the GPU check below) are in our set, not yours.

Our sweep already covers those 75 questions (dense, no cache, no early stop; paired bootstrap over the 75, 2000 resamples, seed 23). Neither F3 nor your Lctx-0.8 saw any of them:

| Word lock (D, sweep) | Correct /75 | of which level 5 /21 | level 1–4 /54 | Calls | Calls per block | Net vs B0 [95% CI] | Net vs Lctx-0.8 [95% CI] |
|---|---|---|---|---|---|---|---|
| B0 | 27 | 2 | 25 | 512 | 32 | - | +2 [−5, +9] |
| D:C-0.9 | 26 | 2 | 24 | 157.4 | 9.8 | −1 [−6, +3] | +1 [−7, +8] |
| D:Lctx-0.8 (yours) | 25 | 1 | 24 | 110.3 | 6.9 | −2 [−9, +5] | - |
| D:Lctx-six | 26 | 2 | 24 | 107.6 | 6.7 | −1 [−6, +4] | +1 [−5, +7] |
| **D:Lctx-F3** | **30** | 2 | 28 | 109.7 | 6.9 | +3 [−4, +9] | +5 [−1, +12] |
| D:Learn2PD | 25 | 1 | 24 | 128.9 | 8.1 | −2 [−8, +4] | +0 [−7, +7] |

**2. token_info counts: one table, two ways of counting.** The npz you have (sha256 `d772a075…`) is the table the rule was fitted with: `fit_ablation3.py` checks this hash before it fits (lines 47–50). `dollar`, `lbrace` and `rbrace` hold a count per token (up to 4: some tokens contain `$$` or `}}`), while `digit, eq, op, ws` are 0/1. `assets.json` reported the number of tokens with a count above 0 (230, 343, 253). Your figures are the summed counts (255, 374, 307), which our own `load_lctx_f3.py` also printed. The engine uses the counts: running `$` parity and brace depth. `assets.json` now records both, and the loader prints both.

**3. Token check against our sweep.** `validation/sweep_f3_token_sha256.json` has, for the four check questions, the sha256 of our D:Lctx-F3 sweep answer as `np.asarray(tokens, dtype=np.int64).tobytes()` (all 512 positions), plus first EOS and calls. Compute the same over `res.tokens` from your D run. If one differs, tell us and we will send that answer's tokens.

**4. Held-out sets.** KodCode-held + GSM8K-held results are in "Held-out results" below. There Lctx-F3 vs your Lctx-0.8 is +2 [−11, +15]: the +7 on HumanEval + our MATH-108 does not carry over.

## Contents

| Path | What it is | sha256 |
|---|---|---|
| `rules/Lctx-F3.pkl` | `ContextRule("gbm")` whose payload is a sklearn `HistGradientBoostingClassifier`. It has 23 inputs. λ = 0.9729116277668666 | `acd2e4e973c48b50fb5dc86ec6acf01379eed8aed790b41e9d9a7647cd8db4fb` |
| `rules/rules.json` | λ, target precision, the feature list in order, fit and threshold row counts, AUROC and share on the threshold rows | |
| `assets/token_info.npz` | Token table for the 17 new inputs: int8 `[vocab]` arrays `digit, eq, op, ws, dollar, lbrace, rbrace`, built from the LLaDA-8B-Instruct tokenizer | `d772a075f528f8df10e2fc5b1ecc457a6cc3dd65b1bde44205f98275500a686e` |
| `source_m1_f3/` | Your `results/model_transfer_20261113/source_m1` with `engine_f3.patch` applied. Only `context_rules.py` and `selective_cache.py` change, and `lock_features3.py` is new; every other file matches your SHA256 file (kept as `SHA256_source_m1.json`) | aggregate `adeb87c06aefc9fe4164cd6dbeeb85ed55e1c1f6a58020e4fe5d825d5a1b80a7` |
| `engine_f3.patch` | The same change as a diff, if you would rather apply it to a later engine. About 25 lines change in the two existing files | |
| `load_lctx_f3.py` | Loads the rule and token table, checking hash, λ and feature list. `python load_lctx_f3.py` runs a quick CPU check | |
| `check_f3_engine.py`, `validation/check_f3_engine.json` | Our GPU check of `source_m1_f3` (below) | |
| `provenance/fit_rules.py`, `provenance/split.json` | The script that fitted the rule, and the question split. They are here for reading: running them needs our G1 traces, which stay on our side | |

The engine change is off by default. It adds `SelectiveConfig.token_info` and `SelectiveResult.f3_state`. The 17 inputs are computed only when the rule's feature list names one of them, so every other arm runs exactly as before.

## How to run it in your harness

```python
# FAMILY_SRC=<repo>/results/share_lctx_f3_20261122/source_m1_f3
sys.path.insert(0, "<repo>/results/share_lctx_f3_20261122")
from load_lctx_f3 import load_lctx_f3
rule, sha, token_info = load_lctx_f3()
ctx["rule"] = rule
cfg = C.cached_cfg("Lctx-F3", ctx, token_info=token_info, **FIXED["P4"])     # or dense=True for D, {} for S32
```

`cached_cfg` already maps any lock starting with `Lctx` to `lock="rule"` and `threshold=0.9`. The rule's own λ is the cut. The 0.9 is used only to pick the single best blank when nothing passes λ.

## Your conditions

- **Settings fixed on tune questions only.** The rule was fitted on 60 MATH-500 level-5 questions, and λ was set on 20 others. λ is the lowest score whose pooled precision on those threshold rows matches C-0.8's (0.99514). Learner: 200 rounds, learning rate 0.1, seed 23. The traces come from LLaDA-8B-Instruct with C-0.8 decoding (our G1 traces). None of the 80 is in our MATH-108 or in HumanEval-164, but 33 are in your MATH-108's level-5 half (see Corrections, 1). KodCode and GSM8K are different datasets. The input set (six plus groups h, i, k, j) was chosen by a pre-registered rule: the largest admitted share on the threshold rows (`results/lctx_features3_20261115/PREREGISTERED.md`).
- **At least one blank per call.** If no blank passes λ, the engine writes the most confident one (`fallback_calls` counts these).
- **Blocks of 32.** Yes: `shape="block"` with `block_length=32`, as in `cached_cfg`.

## Caveats

1. **Why F3 rather than F2 or the six-input refit.** We picked F3 to send after seeing the comparison grid and the word-lock sweep, which use HumanEval-164 and our MATH-108. The rule's settings never saw those sets, but our choice among our three rules did. KodCode-held and GSM8K-held are untouched by any of our choices, so they are the clean test (results below).
2. **The offline gain is small.** On the threshold rows, F3 admits +1.23 points [+0.67, +1.97] more blanks than the six-input refit at the same precision. That is below the +5 points our pre-registered gate asked for, so the gate failed. The decode gains below are larger than this offline number suggests, and their intervals include zero.
3. **Model.** The rule and token table are fitted on LLaDA-8B-Instruct. Our LLaDA-1.5 sweep reuses them unchanged (transfer, not a refit); that table follows when the run finishes.
4. **sklearn.** The pickle was written with scikit-learn 1.8.0 (numpy 2.4.6, torch 2.5.1). sklearn 1.6.1 loads it with a version warning.

## Our results so far (LLaDA-8B-Instruct, HumanEval-164 + our MATH-108 = 272 questions)

MATH-108 here is ours (eval54 + level 1–4, see Corrections, 1), not yours.

These use paired bootstrap over questions, 2000 resamples, seed 23. The word-lock sweep is dense with no early stop: all 512 positions are decoded, so every answer has 16 blocks.

| Word lock (D, sweep) | Correct /272 | Speed vs B0 | Calls | Calls per block (calls ÷ 16) |
|---|---|---|---|---|
| B0 | 98 | 1.00x | 512 | 32 |
| D:C-0.9 | 95 | 5.35x | 98.3 | 6.1 |
| D:TraceLock (T-block) | 87 | 4.39x | 116.5 | 7.3 |
| D:Lctx-0.8 (yours) | 97 | 7.21x | 72.2 | 4.5 |
| D:Lctx-six (our refit) | 94 | 7.35x | 70.8 | 4.4 |
| D:Lctx-F2 | 90 | 6.75x | 74.5 | 4.7 |
| **D:Lctx-F3** | **104** | 7.20x | 71.6 | 4.5 |

Paired against your Lctx-0.8, from the comparison grid (same answers as the sweep; only the cost accounting differs):

| | Net vs Lctx-0.8 [95% CI] |
|---|---|
| D:Lctx-F3 | +7 [−3, +17] |
| P4:Lctx-F3 | +3 [−8, +14] |

**Calls per block.** In the sweep, mean calls per block is calls ÷ 16. The records hold only total calls, not a per-block breakdown. `check_f3_engine.py` logs the per-block calls (`call_blocks`, `log_detail=True`) for D, S32 and P4; the numbers it returned are in `validation/check_f3_engine.json`. To get these for every lock in your grid, set `log_detail=True` in `cached_cfg` and count `res.call_blocks`.

## GPU check of this package

`check_f3_engine.py` loads `source_m1_f3` alone (nothing from our `src/`) and decodes two questions of our MATH-108 (both outside yours) and two HumanEval-164 questions with Lctx-F3 under D, S32 and P4. The D arm uses our sweep's settings, and its tokens are compared with our sweep records for those questions.

**Result (job 313387, RTX 5000 Ada, 2026-11-22): all four D decodes match our sweep records token for token**, so the patched `source_m1` and our engine give the same answers for this lock. Calls per block (16 blocks of 32), in `validation/check_f3_engine.json`:

| Question | D (no early stop) | S32 | P4 |
|---|---|---|---|
| math500-l5-241 | 128 calls: 4 19 7 3 6 5 7 3 9 10 3 8 12 15 14 3 | 133: 4 14 12 2 8 7 8 4 15 24 8 7 3 3 5 9 | 136: 4 16 10 3 9 8 6 3 14 22 15 9 5 3 7 2 |
| math500-l5-394 | 119: 12 7 9 5 8 9 8 7 2 4 7 3 8 9 14 7 | 150: 12 6 11 11 3 4 5 19 18 11 8 6 7 8 15 6 | 144: 12 6 10 11 3 4 5 20 18 9 17 4 7 7 7 4 |
| HumanEval/0 | 23: 3 1 1 1 5 2 1 1 1 1 1 1 1 1 1 1 | 14: 3 1 1 1 6 2, then stops at EOS | 14: same as S32 |
| HumanEval/1 | 52: 4 2 1 4 7 13 10 3 1 1 1 1 1 1 1 1 | 54: 5 2 1 6 12 8 14 6, then EOS | 48: 5 2 1 6 10 8 11 5, then EOS |

With EOS truncation (S32, P4), `words_per_call` = 512 ÷ calls, so it counts the positions filled after the first EOS. Count words per call up to the first EOS if you need it comparable to D.

## Held-out results (added 2026-11-23)

Your KodCode-held (first 150 rows of `kod_held400`, 512 tokens, scored in the bwrap sandbox) and GSM8K-held (rows 100–199 of `gsm8k_tune200`, 256 tokens), LLaDA-8B-Instruct, same sweep setting (dense, no cache, no early stop, block 32). These records log `calls_per_block` per answer. Paired bootstrap over questions within each set, 2000 resamples, seed 23; * = 95% interval excludes 0.

| Word lock (D, sweep) | KodCode /150 | Net vs B0 | GSM8K /100 | Net vs B0 | Pooled /250 | Net vs B0 | Net vs Lctx-0.8 | Speed vs B0 | Calls per block (KodCode, GSM8K) |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 60 | - | 52 | - | 112 | - | +11 [−3, +25] | 1.00x | 32, 32 |
| D:C-0.9 | 61 | +1 [+0, +3] | 49 | −3 [−9, +3] | 110 | −2 [−8, +4] | +9 [−5, +24] | 5.96x | 4.4, 10.2 |
| D:TraceLock (T-block) | 54 | −6 [−14, +2] | 52 | +0 [−13, +13] | 106 | −6 [−21, +9] | +5 [−10, +21] | 4.62x | 5.7, 11.7 |
| D:Lctx-0.8 (yours) | 50 | −10 [−18, −2] * | 51 | −1 [−12, +11] | 101 | −11 [−25, +3] | - | 8.21x | 3.2, 7.1 |
| D:Lctx-six | 38 | −22 [−32, −12] * | 49 | −3 [−12, +7] | 87 | −25 [−39, −10] * | −14 [−28, +0] | 8.24x | 3.2, 6.9 |
| D:Lctx-F2 | 51 | −9 [−16, −2] * | 45 | −7 [−16, +3] | 96 | −16 [−28, −4] * | −5 [−18, +9] | 7.66x | 3.3, 7.2 |
| **D:Lctx-F3** | 53 | −7 [−14, +0] | 50 | −2 [−11, +7] | 103 | −9 [−20, +3] | +2 [−11, +15] | 8.20x | 3.2, 7.1 |
| D:Learn2PD | 58 | −2 [−7, +2] | 54 | +2 [−6, +10] | 112 | +0 [−9, +9] | +11 [−2, +24] | 7.16x | 3.7, 8.3 |

Every Lctx rule loses code answers against B0 on KodCode; F3 loses the fewest of them, and its interval just reaches zero. Learn2PD (official filter, cut 0.96) keeps B0's accuracy at 7.2x.

We also ran KLASS and EB-Sampler from d2Cache ourselves, since you had not run them; they are not your runs. With d2Cache's default (no confidence threshold), its KLASS writes one word per call, which is B0, so we gave it the 0.9 confidence threshold; we also cast its bf16 probabilities to float64, without which it crashes on torch 2.5. Neither changes accuracy against B0 or d2c:Conf-0.9 beyond chance on any of the four sets. The full tables are in our report.
