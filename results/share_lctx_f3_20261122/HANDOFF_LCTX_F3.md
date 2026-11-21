# Lctx-F3 word lock: hand-over (2026-11-22)

This is the new word lock for the sweep: **Lctx-F3**. It is a frozen `ContextRule` in the same format as `Lctx_c0.8.pkl`, with the engine change it needs, already applied to your frozen engine `source_m1`.

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

- **Settings fixed on tune questions only.** The rule was fitted on 60 MATH-500 level-5 questions, and λ was set on 20 others. λ is the lowest score whose pooled precision on those threshold rows matches C-0.8's (0.99514). Learner: 200 rounds, learning rate 0.1, seed 23. The traces come from LLaDA-8B-Instruct with C-0.8 decoding (our G1 traces). We checked the 80 question IDs against your manifests: none is in MATH-108 or HumanEval-164. KodCode and GSM8K are different datasets. The input set (six plus groups h, i, k, j) was chosen by a pre-registered rule: the largest admitted share on the threshold rows (`results/lctx_features3_20261115/PREREGISTERED.md`).
- **At least one blank per call.** If no blank passes λ, the engine writes the most confident one (`fallback_calls` counts these).
- **Blocks of 32.** Yes: `shape="block"` with `block_length=32`, as in `cached_cfg`.

## Caveats

1. **Why F3 rather than F2 or the six-input refit.** We picked F3 to send after seeing the comparison grid and the word-lock sweep, which use HumanEval-164 and MATH-108. The rule's settings never saw those sets, but our choice among our three rules did. KodCode-held and GSM8K-held are untouched by any of our choices, so they are the clean test.
2. **The offline gain is small.** On the threshold rows, F3 admits +1.23 points [+0.67, +1.97] more blanks than the six-input refit at the same precision. That is below the +5 points our pre-registered gate asked for, so the gate failed. The decode gains below are larger than this offline number suggests, and their intervals include zero.
3. **Model.** The rule and token table are fitted on LLaDA-8B-Instruct. Our LLaDA-1.5 sweep reuses them unchanged (transfer, not a refit); that table follows when the run finishes.
4. **sklearn.** The pickle was written with scikit-learn 1.8.0 (numpy 2.4.6, torch 2.5.1). sklearn 1.6.1 loads it with a version warning.

## Our results so far (LLaDA-8B-Instruct, HumanEval-164 + MATH-108 = 272 questions)

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

`check_f3_engine.py` loads `source_m1_f3` alone (nothing from our `src/`) and decodes two MATH-108 and two HumanEval-164 questions with Lctx-F3 under D, S32 and P4. The D arm uses our sweep's settings, and its tokens are compared with our sweep records for those questions.

**Result (job 313387, RTX 5000 Ada, 2026-11-22): all four D decodes match our sweep records token for token**, so the patched `source_m1` and our engine give the same answers for this lock. Calls per block (16 blocks of 32), in `validation/check_f3_engine.json`:

| Question | D (no early stop) | S32 | P4 |
|---|---|---|---|
| math500-l5-241 | 128 calls: 4 19 7 3 6 5 7 3 9 10 3 8 12 15 14 3 | 133: 4 14 12 2 8 7 8 4 15 24 8 7 3 3 5 9 | 136: 4 16 10 3 9 8 6 3 14 22 15 9 5 3 7 2 |
| math500-l5-394 | 119: 12 7 9 5 8 9 8 7 2 4 7 3 8 9 14 7 | 150: 12 6 11 11 3 4 5 19 18 11 8 6 7 8 15 6 | 144: 12 6 10 11 3 4 5 20 18 9 17 4 7 7 7 4 |
| HumanEval/0 | 23: 3 1 1 1 5 2 1 1 1 1 1 1 1 1 1 1 | 14: 3 1 1 1 6 2, then stops at EOS | 14: same as S32 |
| HumanEval/1 | 52: 4 2 1 4 7 13 10 3 1 1 1 1 1 1 1 1 | 54: 5 2 1 6 12 8 14 6, then EOS | 48: 5 2 1 6 10 8 11 5, then EOS |

With EOS truncation (S32, P4), `words_per_call` = 512 ÷ calls, so it counts the positions filled after the first EOS. Count words per call up to the first EOS if you need it comparable to D.
