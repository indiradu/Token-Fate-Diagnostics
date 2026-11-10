# Learn2PD as a word lock, against confidence thresholds (C-c) and Lctx; does a learned rule beat a lowered threshold? (2026-11-08)

**STATUS: stage 1 complete** (HumanEval 164 and MATH 108, both at 512 tokens). Stage 2 (GSM8K 200, MATH 54 at 1024) NOT run (no budget). Nothing committed or pushed; GPU released; 7.62 GPU-hours (cap about 8). Pre-registration with Outcome: `PREREGISTERED.md`. Tables with intervals: `results.md` (all pooled and per-stratum tables, pairwise comparisons, equal-speed table, truncation split, load sensitivity); numbers `summary.json`; per-cell append-only records `records/`; official-code reading `OFFICIAL_CODE_NOTES.md`; frozen decoders `source/` (sha256 `source/SHA256.json`; `validation/source_diff_vs_farcache.json` lists the only three files that differ from the far-cache campaign's source: `l2pd_rule.py` new, `paper_decode.py` and `selective_cache.py` additive and default-off); frozen filters and taus `filters/`; held-out/fit logs `validation/fit_report.json`; training-split lists and overlap check `manifests/train_*.jsonl`, `validation/train_disjoint.json`.

## Words
WORD LOCK = which blank positions get a word written in a call (saves calls). REPRESENTATION CACHE = which rows have their K/V cached instead of recomputed (saves compute per call). Row types: prompt rows, written answer rows, blank rows. **Lctx-0.8** is the learned context word lock whose 0.8 is a PRECISION TARGET (cutoff lambda = 0.9605 on a gradient-boosted score, MATH-fitted, never refit), not a confidence cutoff. **C-c** = write every blank of the block with confidence >= c, else the single most confident (Fast-dLLM's threshold rule). **Learn2PD (L2PD)** = a 2,112-parameter MLP over the 32 block confidences; a blank is written iff sigmoid(logit) > tau, and every call also writes the most confident blank. Cells: **D** = dense word lock (no truncation); **T** = dense + first-EOS truncation twin (a dense call on every call); **S32** = selective cache (query rows = the current 32-block's blank rows, all 32 block rows for L2PD, + rows written on the previous call; a dense call at every block start; first-EOS truncation). Datasets: HumanEval 164 (code, gen 512) and MATH 108 (54 Level 5 + 54 Level 1-4, gen 512). LLaDA-8B-Instruct bf16, temperature 0, block 32, fused SDPA, one timed repeat, one process per set, threads 1.

## Headline (speed = ratio of summed decode seconds; net = correct - dense C-0.9's correct, paired bootstrap over questions; resolution about +-6 answers)
HumanEval 164 (dense C-0.9: 68 correct, 53.8 calls):

| cell | speed vs D:C-0.9 | speed vs T:C-0.9 | net [95%] | lost/gained |
|---|---|---|---|---|
| D:C-0.8 | 1.208 | 1.014 | +1 [-5, +7] | 4/5 |
| D:Lctx-0.8 | 1.297 | 1.089 | -2 [-9, +5] | 7/5 |
| D:L2PD-off (official weights, tau 0.96) | 1.176 | 0.987 | -3 [-8, +2] | 5/2 |
| D:L2PD-pub (retrained, tau 0.96) | 1.088 | 0.913 | **-12 [-20, -5]** | 14/2 |
| D:L2PD-match (retrained, tau 0.946) | 1.197 | 1.005 | **-17 [-26, -9]** | 19/2 |
| T:C-0.9 (truncation only) | 1.191 | - | 0 | 0/0 |
| S32:C-0.9 | 2.253 | 1.891 | -9 [-19, +1] | 17/8 |
| S32:C-0.8 | 2.628 | 2.206 | -5 [-15, +5] | 16/11 |
| S32:C-0.7 | 3.016 | 2.531 | **-19 [-30, -8]** | 26/7 |
| S32:C-0.6 | 3.392 | 2.847 | **-34 [-45, -23]** | 37/3 |
| S32:Lctx-0.8 | 2.869 | 2.408 | **-11 [-21, -2]** | 18/7 |
| S32:L2PD-pub | 2.569 | 2.156 | **-20 [-31, -9]** | 26/6 |
| S32:L2PD-match | 2.821 | 2.368 | **-23 [-34, -12]** | 29/6 |

MATH 108 (dense C-0.9: 31 correct, 162.7 calls; Level 5 only 6/54, so answer differences are hard to resolve):

| cell | speed vs D:C-0.9 | speed vs T:C-0.9 | net [95%] | lost/gained |
|---|---|---|---|---|
| D:C-0.8 | 1.253 | 1.230 | 0 [-5, +5] | 3/3 |
| D:Lctx-0.8 | 1.426 | 1.400 | -3 [-11, +5] | 10/7 |
| D:L2PD-off | 1.222 | 1.199 | 0 [-4, +4] | 2/2 |
| D:L2PD-pub | 1.191 | 1.169 | -1 [-6, +4] | 4/3 |
| D:L2PD-match | 1.350 | 1.326 | -1 [-9, +6] | 8/7 |
| T:C-0.9 (truncation only) | 1.019 | - | 0 | 0/0 |
| S32:C-0.9 | 2.070 | 2.033 | 0 [-7, +7] | 6/6 |
| S32:C-0.8 | 2.459 | 2.414 | -1 [-9, +6] | 8/7 |
| S32:C-0.7 | 2.948 | 2.894 | +2 [-5, +9] | 6/8 |
| S32:C-0.6 | 3.389 | 3.328 | -1 [-9, +6] | 8/7 |
| S32:Lctx-0.8 | 2.697 | 2.647 | -5 [-13, +2] | 10/5 |
| S32:L2PD-pub | 2.425 | 2.381 | -5 [-13, +3] | 11/6 |
| S32:L2PD-match | 2.677 | 2.628 | -2 [-9, +5] | 7/5 |

## The answer in plain words
1. **Truncation is a real part of the cache speed-up on code and nothing on MATH.** Dense + first-EOS truncation alone (T:C-0.9, token-identical to dense up to the first EOS, parity-checked) is 1.191x on HumanEval (short answers padded with EOS) and 1.019x on MATH 108 (most MATH answers fill the 512 budget). So against the dense-with-truncation reference the S32 cache is 1.89x [1.76, 2.03] (C-0.9) on HumanEval instead of 2.25x, and 2.03x [1.93, 2.14] on MATH instead of 2.07x; truncation explains 22% of the HumanEval log-speed-up and 3% of MATH's. The earlier campaign's cache numbers on code overstated the cache's own contribution by about a fifth.
2. **(a) Learn2PD vs C-c vs Lctx.** With the official released weights at the published tau 0.96 (dense), Learn2PD is *the same word lock as C-0.8*: 1.222x vs C-0.8's 1.253x on MATH (0.975x of C-0.8 [0.951, 0.998]) and 1.176x vs 1.208x on HumanEval, with the same answers on MATH (0 [-4, +4]) and -3 [-8, +2] on HumanEval (-4 [-9, 0] against C-0.8's own count). It is slower than Lctx-0.8 (0.857x [0.823, 0.889] on MATH). The filter's own cost is 0.3 ms per call (Lctx 1.6 ms). The retrained filter with tau 0.96 is more conservative (1.19x on MATH; 137 calls vs C-0.8's 130); call-matched to C-0.8 (tau 0.946, calibrated on 14 held-out training questions) it is 1.35x on MATH with the same answers as dense C-0.9 (-1 [-9, +6]) -- a little faster than C-0.8 (1.078x [1.043, 1.114]), 5% slower than Lctx (0.947x [0.913, 0.982]) and statistically the same on answers (+2 [-5, +9] over Lctx) -- **but on HumanEval it breaks: -12 and -17 answers** (lost 14/2 and 19/2): a filter trained on MATH and GSM8K traces does not transfer to code, while the official weights (FLAN-trained) do. Nowhere does Learn2PD beat both C-0.8 and Lctx-0.8 on speed and answers together. In the S32 cache the retrained call-matched filter reaches 2.677x on MATH, level with S32:Lctx-0.8 (2.697x; 0.993x [0.952, 1.035]) at net -2 vs -5, and 2.821x on HumanEval at -23 vs Lctx's -11.
3. **(b) Does the learned rule beat simply lowering the threshold at equal speed?** **No evidence that it does; on MATH the lowered threshold is at least as good, on code the comparison is a draw.** MATH, S32: S32:C-0.7 is faster than S32:Lctx-0.8 (2.948x vs 2.697x, pairwise 1.093x) with +2 [-5, +9] vs -5 [-13, +2] answers (Lctx - C-0.7: -7 [-14, 0], lost 11 / gained 4); S32:C-0.6 reaches 3.389x (26% above Lctx) at -1 [-9, +6]. S32:L2PD-match (2.677x, -2) sits between C-0.8 (2.459x, -1) and C-0.7 (2.948x, +2): on the C-c line, not above it. All of these differences are inside the +-6 resolution. HumanEval, S32: lowering the threshold is very costly (C-0.8 -5, C-0.7 -19, C-0.6 -34) and Lctx (2.869x, -11) lies between C-0.8 and C-0.7 in speed and above the line joining them by about 3 answers (Lctx vs C-0.7: +8 [-1, +17] at 0.951x speed): a point-value advantage for Lctx on code that does not clear the resolution. Summed over the two sets Lctx S32 loses 16 answers and C-0.7 loses 17, with C-0.7 the faster of the two on both sets (3.02x / 2.95x vs 2.87x / 2.70x). The retrained Learn2PD cells are below the line on code. This repeats the earlier word-lock finding that at the 0.9-type target a learned rule is not worth much against a lower confidence threshold; the lower-threshold C-0.7 / C-0.6 controls were what was missing.
4. **What lowering the threshold costs.** On MATH nothing is visible down to C-0.6 (nets -1 to +2, all within +-6) with calls falling to 101.6 (from 162.7). On HumanEval the S32 cache cell loses 9 answers already at C-0.9 and the loss grows with a lower threshold (-5, -19, -34). The C-0.9 dense reference is the right baseline for MATH quality but its HumanEval correct count (68/164) is itself low.

## How this Learn2PD differs from the official one (read the code, not only the paper)
See `OFFICIAL_CODE_NOTES.md`. The official repository (github.com/ims-kdks/Learning-to-Parallel-Decoding, commit 06bfd7f) DOES ship weights for LLaDA-8B-Instruct, block 32 (`llada/layer_2_flan.pth`), and arm `L2PD-off` uses them unchanged. Where the code contradicts the task description or the paper: (i) tau is applied to the sigmoid, strictly (`sigmoid(logit) > 0.96`); (ii) every step also writes the single most confident blank (the vanilla top-k stays on), so Learn2PD is not "filter else best blank" and its fallback share is not comparable to C-c's -- we report the share of calls on which the filter accepted nothing (MATH 33-41%) and on which the forced word was outside its set (37-46%); (iii) the filter reads the confidence of all 32 block slots, written ones included, so our cache cell computes all 32 block rows each call (MATH rows per call 96 and 103 for S32:L2PD-pub and -match vs 82 for S32:C-0.8, which makes a similar number of calls); (iv) the official EoT cuts at the most confident *masked* position predicted EOS even if nothing was written, not "once an EOS is unmasked" -- ours (all T and S32 cells) cuts after the first written EOS and is therefore less aggressive; (v) the official notebook trains with lr 0.003 for 40,000 epochs, the paper says 1e-3 and 5,000; (vi) the released weights were trained on raw (no chat template) FLAN prompts at gen 256, mostly padding blocks. Our retrained filter: same architecture and loss, paper numbers (AdamW 1e-3, 5,000 epochs chosen over {500, 1000, 2000, 5000} on held-out questions), but 72 training questions (42 MATH-train at 512, 30 GSM8K-train at 256; 2,216 EGP rows; the paper used 2,640 FLAN questions), chat-template prompts, our confidence convention (float32, mask-suppressed argmax; no mask proposal was ever rejected in a dense decode), and labels from this repo's one-word-per-call decode. On 662 held-out EGP rows the retrained filter's BCE is 0.256 vs the official weights' 0.246 (AUROC 0.907 vs 0.912), so it is about as good as the official one *on math-like traces*. tau: published 0.96 on the sigmoid. The precision match to C-0.8 (the Lctx rule) gave 0.963, i.e. the published tau is already precision-matched; the second variant is therefore the call-count match (0.946). Code is a transfer for the retrained filter (no code training split in the repo) and it failed there.

## What could not be done / not run
- Dense C-0.7 and C-0.6, the twins of Lctx-0.8 and L2PD-pub, and an S32 cell with the official weights were cut for the budget (pre-registered cut list). Lowered thresholds are compared in the S32 cells; the dense lowered-threshold comparison uses C-0.8 only. The twin-based truncation share is therefore measured for C-0.9 and L2PD-match only.
- Stage 2 (GSM8K 200, MATH 54 at 1024) not run.
- The official EoT argmax cut, and the official dual-cache decoder, were not run (our own S32 engine with block rows is the cache cell); weights from the official repo were not trained by us.
- Held-out calibration questions for tau were Level 1-4 MATH + GSM8K only (a role-assignment slip found after collection), so the call match did not reproduce exactly on the test sets (L2PD-match 7% fewer calls than C-0.8 on MATH, equal on HumanEval).
- One timed repeat; node load rose (5-25) for about 25 HumanEval questions (another session's pytest run), see the load-sensitivity table; ratios move by 1-2%.
- MATH dense C-0.9 solves only 31/108 (6/54 on Level 5): MATH quality differences between cells are rarely distinguishable.

## Process notes
Order: official code read -> unit tests (`tests/test_l2pd_rule.py`, 19; suite 413 passed) -> EGP traces (training split only) -> filter fit -> calibration decodes on held-out training questions -> tau frozen in `PREREGISTERED.md` -> crash test on training questions -> parity (28 checks, all matched) -> HumanEval -> MATH. Both test sets ran in single processes; no process was killed or restarted. The official repo was cloned into its own directory (`/home/rui.gao/Desktop/l2pd_untrusted`, outside the project), read, never executed; the weight file was loaded with `torch.load(weights_only=True)`.
