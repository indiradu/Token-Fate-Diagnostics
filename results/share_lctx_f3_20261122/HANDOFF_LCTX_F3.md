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

## LLaDA-1.5 on the held-out sets (added 2026-11-24)

The same seven arms, question lists and settings as the LLaDA-8B held-out table above, on LLaDA-1.5 (`GSAI-ML/LLaDA-1.5` @ 84346fd). Every lock was fitted on LLaDA-8B and is used unchanged (transfer, not refit). Learn2PD is not run on LLaDA-1.5: its weights are LLaDA-8B only, and we skipped it there as in our LLaDA-1.5 HumanEval + MATH run (your open point 2).

| Word lock (D, sweep) | KodCode /150 | Net vs B0 | GSM8K /100 | Net vs B0 | Pooled /250 | Net vs B0 | Net vs Lctx-0.8 | Speed vs B0 | Calls per block (KodCode, GSM8K) |
|---|---|---|---|---|---|---|---|---|---|
| B0 | 62 | - | 50 | - | 112 | - | +7 [−8, +21] | 1.00x | 32, 32 |
| D:C-0.9 | 62 | +0 [−3, +3] | 47 | −3 [−10, +3] | 109 | −3 [−10, +4] | +4 [−10, +18] | 5.77x | 4.6, 9.9 |
| D:TraceLock (T-block) | 56 | −6 [−14, +2] | 57 | +7 [−5, +19] | 113 | +1 [−13, +15] | +8 [−6, +21] | 4.55x | 5.8, 11.6 |
| D:Lctx-0.8 (yours) | 49 | −13 [−21, −5] * | 56 | +6 [−6, +18] | 105 | −7 [−21, +8] | - | 8.00x | 3.3, 7.0 |
| D:Lctx-six | 44 | −18 [−28, −8] * | 53 | +3 [−9, +15] | 97 | −15 [−31, +1] | −8 [−21, +5] | 7.84x | 3.4, 6.8 |
| D:Lctx-F2 | 45 | −17 [−26, −9] * | 41 | −9 [−20, +2] | 86 | −26 [−40, −11] * | −19 [−32, −6] * | 7.56x | 3.4, 7.2 |
| **D:Lctx-F3** | 53 | −9 [−17, −2] * | 54 | +4 [−9, +16] | 107 | −5 [−20, +9] | +2 [−12, +16] | 7.98x | 3.3, 6.9 |

The LLaDA-8B pattern repeats: on KodCode every Lctx rule loses answers against B0, while C-0.9 holds; on GSM8K no arm differs from B0. Lctx-F3 vs your Lctx-0.8 is +2 on both models.

## FLOP pass (added 2026-11-24, your plan section 2)

One counter, `torch.utils.flop_counter.FlopCounterMode` (GEMMs and attention, 2 FLOPs per multiply-add), on LLaDA-8B-Instruct. LLaDA-1.5 has an identical `config.json`, so its per-call FLOPs are the same.

- **Part A.** One dense forward of prompt + answer blanks for the first 20 prompts of each set (80 prompts). The counted backbone (LM head excluded) equals SureLock's formula `L*T*(4*D*T + 4D^2 + 4D*Dkv + 6D*Dff)` exactly on all 80, and every engine record's row counts confirm that each call is dense, so backbone per question = calls × per-call backbone.
- **Part B.** HumanEval/0 and HumanEval/1 decoded with every arm under the counter, to separate the LM head and each lock's own GEMMs.

Backbone FLOP cut vs B0, all questions of the four sets:

| Word lock | LLaDA-8B (522 q) | LLaDA-1.5 (522 q) | Lock's own FLOPs, % of backbone |
|---|---|---|---|
| D:Lctx-0.8 | 86.9% | 86.8% | 0 (tree model on the CPU) |
| D:Lctx-six | 87.1% | 86.7% | 0 |
| D:Lctx-F2 | 86.4% | 86.6% | 0.25% (its probes run the LM head on an intermediate layer) |
| **D:Lctx-F3** | 86.9% | 86.8% | 0 |
| D:Learn2PD | 85.0% | not run | ~0 (32×32 filter, about 4 kFLOPs per call) |
| D:C-0.9 | 81.9% | 81.7% | 0 |
| D:TraceLock | 78.0% | 78.1% | 0.14% (projector + controller) |
| d2c:Conf-0.9 | 82.1% | not run | 0 |
| d2c:KLASS | 69.9% | not run | elementwise only (not counted) |
| d2c:EB-Sampler | 48.8% | not run | elementwise only (not counted) |

- In this dense sweep the FLOP cut is the call cut, because every call costs the same within a question.
- **LM head, reported separately.** B0 computes it on every position, about 310 TFLOPs per question on LLaDA-8B. Our engine computes it on about one block's rows, about 2 TFLOPs.
- **TraceLock.** The counter gives 0.78x the `locking_flop_accounting` formula per call, so earlier tables that used the formula slightly overstate TraceLock's FLOPs.
- **d2Cache** was not run under the counter. Its backbone uses the same per-call count; its head is assumed to cover every position.

## Remaining sweep-plan items (added 2026-11-25)

Everything in your word-lock sweep plan is now run. Same setting throughout: dense, no cache, no early stop, block 32, temperature 0, the same four question sets (MATH-108 = ours, see Corrections). Paired bootstrap over questions within each set, 2000 resamples, seed 23; * = 95% interval excludes 0.

### Dream-v0-Instruct-7B (d2Cache code)

`Dream-org/Dream-v0-Instruct-7B` @ 05334cb in d2Cache 79fb5f6 (its `dream-inst` loader, which shifts Dream's outputs by one position), Dream's own chat template, same prompts and scorer. As your plan says, only the model-agnostic word locks run here. B0 = d2Cache vanilla with no threshold (one word per call, block 32), and every arm runs in the same code, so these speeds compare directly.

| Arm | Correct /522 | Net vs B0 | Speed vs B0 | Words per call | HumanEval /164 | MATH /108 | KodCode /150 | GSM8K /100 |
|---|---|---|---|---|---|---|---|---|
| d2c:B0 | 294 | - | 1.00x | 1.00 | 92 | 40 | 87 | 75 |
| d2c:Conf-0.9 | 292 | −2 [−9, +5] | 5.93x | 5.61 | 91 | 41 | 83 | 77 |
| d2c:KLASS (confidence 0.9) | 294 | +0 [−3, +3] | 3.07x | 3.61 | 91 | 40 | 87 | 76 |
| d2c:EB-Sampler | 297 | +3 [+0, +7] | 2.43x | 2.55 | 92 | 41 | 87 | 77 |

### LLaDA-1.5: Learn2PD and the d2Cache decoders

D:Learn2PD uses the official LLaDA-8B filter unchanged, which settles your open point 2 as "transferred, not refit". Its net vs B0 and speed are measured against the same B0 records as the earlier LLaDA-1.5 tables.

| Arm | HE + MATH /272 | Net vs B0 | Speed vs B0 | KodCode + GSM8K /250 | Net vs B0 | Speed vs B0 |
|---|---|---|---|---|---|---|
| B0 | 107 | - | 1.00x | 112 | - | 1.00x |
| D:Learn2PD | 99 | −8 [−17, +0] | 6.53x | 108 | −4 [−15, +7] | 6.80x |
| d2c:Conf-0.9 | 110 | +3 [−6, +12] | 3.74x | 115 | +3 [−8, +14] | 3.92x |
| d2c:KLASS | 108 | +1 [−8, +9] | 1.95x | 114 | +2 [−8, +12] | 1.99x |
| d2c:EB-Sampler | 108 | +1 [−8, +10] | 1.31x | 114 | +2 [−8, +12] | 1.28x |

- **d2Cache speeds are not directly comparable with ours.** d2Cache's code costs about 45% more per call than our engine, so compare d2c rows with each other for speed and use words per call across codes.
- **A numerics effect on LLaDA-1.5 MATH.** d2c:Conf-0.9 gets 36 and our D:C-0.9 gets 29 with the same calls (160 vs 163). Both codes run the same rule; bf16 rounding differs between them, which on LLaDA-1.5 moves 7 MATH answers. That makes −7 [−13, −1] for ours against theirs. On HumanEval, KodCode and GSM8K the two agree within chance.

### Checks (plan section 3)

- **d2Cache vs our engine, Conf-0.9, first 20 HumanEval prompts.** LLaDA-8B: 12 of 20 answers identical up to the end of text, calls 960 vs 945. LLaDA-1.5: 13 of 20 identical, calls 948 vs 963. This matches your Study 5 result (12 of 20, calls within 7%).
- **Learn2PD vs the official `generate_learn2parallel`.** Setup: `ims-kdks/Learning-to-Parallel-Decoding` @ 06bfd7f, its `layer_2_flan.pth` (sha256 0d375569…), LLaDA-8B, first 5 HumanEval prompts, gen 512, block 32, τ 0.96 on the sigmoid, same prompt tokens. All 5 answers are identical in all 512 tokens. Calls are 196 vs 193: two questions take one or two extra calls, from the confidence numerics (official bf16 softmax of the raw argmax vs our float32 softmax with the mask token suppressed).

### Fastest arm without detectable answer loss, per cell (plan section 2)

"No detectable loss" means the 95% interval of net vs B0 is not wholly below 0. Speed is wall-clock vs that model's B0 on the same questions.

| Model | Set | Fastest arm without loss | Net vs B0 | Speed vs B0 | Answer tokens/s | Faster arm that loses |
|---|---|---|---|---|---|---|
| LLaDA-8B | HumanEval | D:Lctx-0.8 | −1 [−9, +6] | 12.52x | 61 | |
| LLaDA-8B | MATH-108 (ours) | D:Lctx-six | +0 [−8, +8] | 4.32x | 46 | |
| LLaDA-8B | KodCode-held | D:Lctx-F3 | −7 [−14, +0] (on the boundary) | 9.67x | 48 | |
| LLaDA-8B | GSM8K-held | D:Lctx-six | −3 [−12, +7] | 4.44x | 77 | |
| LLaDA-1.5 | HumanEval | D:Lctx-F3 | −8 [−18, +1] | 12.09x | 62 | D:Lctx-0.8, −11 [−20, −2] * |
| LLaDA-1.5 | MATH-108 (ours) | D:Lctx-six | −1 [−9, +7] | 4.42x | 45 | |
| LLaDA-1.5 | KodCode-held | D:Learn2PD | −2 [−8, +4] | 7.83x | 42 | D:Lctx-0.8, −13 [−21, −5] * |
| LLaDA-1.5 | GSM8K-held | D:Lctx-six | +3 [−9, +15] | 4.57x | 80 | |
| Dream-7B | HumanEval | d2c:Conf-0.9 | −1 [−4, +2] | 11.93x | 26 | |
| Dream-7B | MATH-108 (ours) | d2c:Conf-0.9 | +1 [−2, +4] | 2.75x | 27 | |
| Dream-7B | KodCode-held | d2c:Conf-0.9 | −4 [−9, +0] | 10.12x | 29 | |
| Dream-7B | GSM8K-held | d2c:Conf-0.9 | +2 [+0, +5] | 2.86x | 35 | |

### Your open points (plan section 6)

1. **Baselines:** re-measured inside this sweep (B0 and Conf-0.9 in every model and code), not reused from Study 5.
2. **Learn2PD on LLaDA-1.5:** run with the transferred LLaDA-8B weights (table above).
3. **Dream:** in scope, with the model-agnostic arms only (table above).
4. **Our lock:** delivered as a frozen `ContextRule` pickle (Lctx-F3, this note).
5. **The grid (Study 6):** these word locks are the word-lock axis; we have not run the cache × word-lock grid for F3 beyond our own compare_grid.

## Dream-v0-Instruct-7B: the LLaDA-fitted word locks transferred unchanged (added 2026-11-26)

Beyond your plan, which limited Dream to the model-agnostic word locks. D:C-0.9, D:Lctx-0.8, D:Lctx-six, D:Lctx-F3 and D:Learn2PD run on Dream with the fits from LLaDA-8B, unchanged:
- The Lctx rules get a Dream token-class table, built by the same classifier as `build_assets.py` (vocab 152064, sha256 `f47fcf6fb62dc624…`).
- Learn2PD uses the official `layer_2_flan.pth` (tau 0.96 on the sigmoid).

D:Lctx-F2 and D:TraceLock read LLaDA's hidden states (width 4096; Dream's is 3584), so they cannot run on Dream.

These arms run in our own Dream loop, not in d2Cache:
- Each call is one full forward.
- The loop applies d2Cache's logit shift (`prepare_logits_for_generation`).
- The word lock is the repo engine's `_lock_write` (or `_lock_write_l2pd`).

Loop check against d2Cache on 20 HumanEval prompts:
- Our B0 vs d2c:B0: 19/20 identical texts, 84.2 vs 83.1 ms per call, 13 vs 13 correct.
- Our C-0.9 vs d2c:Conf-0.9: 19/20 identical, 843 vs 827 calls, 12 vs 12 correct.

Speeds are therefore measured against the same d2c:B0 records as the table above.

Records and analysis are in `results/wordlock_sweep_20261125_dream_locks/` on the cluster (`orchestration/run_dream_locks.py`, `analyze_dream_all.py`, `results.md`, `validation/loop_vs_d2c.json`). All 5 arms × 522 questions were scored, with no duplicates.

| Arm | Correct /522 | Net vs B0 | Speed vs B0 | Words per call | HumanEval net | MATH net | KodCode net | GSM8K net |
|---|---|---|---|---|---|---|---|---|
| D:C-0.9 | 292 | −2 [−9, +5] | 5.96x | 5.57 | −1 | +1 | −4 | +2 |
| D:Learn2PD | 285 | −9 [−20, +2] | 7.02x | 6.59 | −8 [−14, −3] * | +5 [+0, +11] | −7 [−14, +0] | +1 |
| D:Lctx-F3 | 275 | −19 [−38, −2] * | 8.12x | 7.80 | −12 [−22, −2] * | +3 | −9 [−19, +0] | −1 |
| D:Lctx-0.8 | 265 | −29 [−47, −11] * | 8.28x | 7.88 | −13 [−23, −3] * | −1 | −16 [−25, −7] * | +1 |
| D:Lctx-six | 260 | −34 [−51, −17] * | 8.20x | 7.82 | −14 [−23, −5] * | −1 | −16 [−25, −7] * | −3 |

What this shows:
- D:C-0.9 gives the same 292 answers as d2c:Conf-0.9.
- The transferred locks write more words per call on Dream than on LLaDA: Lctx is 13–16x on code and 3.9–4.2x on maths.
- They lose answers on code only. Every Lctx rule loses on HumanEval; Lctx-0.8 and six also lose on KodCode.
- None loses on MATH or GSM8K.
- Lctx-F3 vs Lctx-0.8 on Dream: +10 [−8, +28], not resolved.
- Learn2PD vs Lctx-0.8: +20 [+2, +38].

Across all three models (12 cells per lock), the cells without a detectable loss are:
- C-0.9: 12/12
- Learn2PD: 10/12
- Lctx-F3: 10/12
- Lctx-0.8: 7/12
- Lctx-six: 7/12
- Lctx-F2: 5/8
- TraceLock: 7/8
