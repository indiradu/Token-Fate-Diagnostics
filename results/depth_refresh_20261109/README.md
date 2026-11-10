# Refreshing the representation cache: the K4 and P4 cells, with the learned context word lock (2026-11-09 to 2026-11-11)

**Status: partly complete.** Stages A to D and the HumanEval half of stage E are decoded and scored (summaries below). Stage E's MATH half, stage F (Elastic-Cache's protocol), and the follow-ups in `../lctx_refit_20261110`, `../lctx_features_20261110` and `../tracelock_stack_20261111` were queued or running when this was written; their results are NOT in this commit. Nothing here is a final claim: every number is one benchmark's worth of evidence at a resolution of about +-6 answers. Pre-registrations with predictions, misses and decision rules: `PREREGISTERED.md` (Outcome sections appended after each stage).

## Words
WORD LOCK = which blank positions get a word per call (saves calls). REPRESENTATION CACHE = which rows' K/V are read from a store instead of recomputed (saves compute per call). **S32** = the cache computes only the current 32-block's blanks plus the rows written on the previous call; prompt rows, older written rows and later-block blank rows are read from the store; a dense call (every row, all 32 layers) opens each block; first-EOS truncation. Dense C-0.9 = the reference decoder with the 0.9 confidence word lock. LLaDA-8B-Instruct bf16, temperature 0, block 32, RTX 5000 Ada, batch 1.

## The highlighted cells (all stacked on S32)
- **K4** = S32 + a **dense** call every 4th call (every row, all layers).
- **P4** = S32 + a **blank-row** partial refresh every 4th call: the later-block blank rows rejoin the computation at layer 11 from their stored layer inputs and are recomputed up to layer 31; prompt and written rows stay cached. A partial refresh from layer 11 costs 67.5 ms against 80.7 ms for a dense call, but only the blank rows are recomputed in the variant used here (56.0 ms; see `records/bench_layer_cost.json`, local).
- **Lctx-0.8** = the learned context word lock (HistGradientBoosting on confidence, entropy, margin, distance to written text, local mask ratio and position in block; fitted once on MATH level-5 one-word-per-call traces; cutoff chosen so its pooled precision matches the confidence threshold C-0.8; never refit; `../longlen_wordlock_20261103`).

## Headline numbers
HumanEval 164 tasks, 512 tokens (dense C-0.9 solves 68); correct answers, net against dense in brackets, speed against dense:
| word lock | dense | S32 | K4 | P4 |
|---|---|---|---|---|
| C-0.9 | 68, 1.00x | 59 (-9), 2.30x | 68 (+0), 2.00x | 69 (+1), 2.25x |
| C-0.8 | 69 (+1), 1.21x | 63 (-5), 2.68x | 67 (-1), 2.36x | 67 (-1), 2.74x |
| **Lctx-0.8** | 66 (-2), 1.29x | 57 (-11), 2.93x | 64 (-4), 2.57x | 62 (-6), 2.92x |
- **P4 against K4**, same word lock, same process: 12% (C-0.9), 16% (C-0.8) and 13% (Lctx-0.8) faster at +1, +0 and -2 answers (paired intervals include 0). On MATH (C-0.9, S32 underneath; plain S32 loses nothing there): 108 questions at 512 tokens, 33 correct against dense 31 at 2.07x, 1.22x [1.16, 1.27] faster than K4; 54 questions at 1024 tokens, 18 correct against dense 18 at 2.82x, 1.41x [1.29, 1.54] faster than K4 (intervals do not overlap: the advantage grows with length, a dense call costing 152 ms at 1024 tokens).
- **Lctx-0.8 against C-0.8** in the same cache: 6 to 9% faster, 3 to 6 fewer correct answers on HumanEval, no paired interval excludes 0. It holds the pre-registered rule (more than 5% faster at no more than 3 answers lost) in 2 of 4 caches on HumanEval; MATH half of stage E pending. P4 recovers part of Lctx's S32 loss (62 against 57 correct, +5 [-2, +13]).
- **Elastic-Cache's recipe in our engine** (16-blank window, its word lock, our blank-row refresh every 4th call vs its adaptive trigger): 72 against 64 correct at 2.11x against 2.07x (+8 [-1, +17] answers, speed ratio 1.02 [0.95, 1.09]); not a speed win by the pre-registered rule.
- Backbone FLOPs (SureLock convention) fall 79 to 84% for the cached cells while wall-clock speed is 2.0 to 2.9x: FLOP cuts and latency cuts differ (`metrics_stageE_he164.md`).

## Limits (read before quoting)
- The P4 setting (blank rows, layer 11, every 4th call) was chosen after stage A on the same 164 HumanEval tasks; MATH is untuned for it but cannot show recovery. The repo has no held-out code set (no MBPP data or scorer).
- 164 tasks resolve about +-6 answers (+-4 at 54 questions); speeds are one timed repeat, differences under about 5% are not resolved.
- Several pre-registered predictions missed (listed in each Outcome section), among them first-divergence timing and the blank-row oracle; the stage-A result (blank-row refresh restores answers) conflicts with the oracle repair in `../stale_cause_20261108` and is not reconciled.
- HumanEval from stage B on was decoded with scoring deferred (no code sandbox on the decoding hosts) and scored afterwards with `bubblewrap` installed into a private conda environment; stages A and B reproduced their decode-time scores exactly (2,132 records, 0 differ).
- Elastic-Cache here is its recipe re-implemented in our engine with the fused attention kernel, not the released code.

## Files
`PREREGISTERED.md`; `results_stage*.md` (tables with bootstrap intervals), `metrics_stageE_he164.md` (calls, ms per call, tokens per second, backbone FLOPs, K/V memory, output change); `orchestration/` (stage runner `run_depth_cells.py`, analysis, rescoring, Slurm scripts, layer-cost benchmark). Records (`records/*.jsonl`), the frozen source copy, logs and manifests stay local.
