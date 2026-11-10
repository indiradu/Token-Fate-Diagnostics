# Which evaluation quantities do the cache and word-lock papers report besides wall-clock speed? (survey, 2026-11-10)

Purpose: decide which metrics our comparison must report so that it can be read next to the literature. Setting of ours: RTX 5000 Ada, batch 1, LLaDA-8B-Instruct bf16, HumanEval 164 at 512 tokens, MATH 108 at 512, MATH 54 at 1024, GSM8K 200 at 256. Machine-readable version: `survey.json` (216 records, one per paper and field).

## 0. How to read this, and what was not verified

- **Method.** Each paper's arXiv HTML page (or the one named in the notes) was fetched through a summarizing fetch tool with a fixed extraction prompt. Every "QUOTED" string below is a short quote **as extracted by that tool; none was checked against the PDF.** Tables and figures that are images were not read, so any number that exists only in a figure is marked as such.
- **Status labels used everywhere:** QUOTED (short verbatim, as extracted), PARAPHRASED, NOT REPORTED (the paper text visible to me does not contain it), NOT ACCESSIBLE (page truncated or unread), TOOL OBSERVATION (a consistency check done by the fetch tool, for example "Table 2 disagrees with the abstract"; I did not verify these and do not rely on them).
- **Papers fetched and read (18):** SureLock, Elastic-Cache, Fast-dLLM v1, Fast-dLLM v2, dLLM-Cache, dKV-Cache, Learn2PD, Window-Diffusion, FlashDLM (FreeCache), d2Cache, dInfer, DyLLM (plus the official README), TraceLock, Polestar, LESS, Sangam, MaskKV ("Mask Tokens as Prophet"), Dynamic-dLLM.
- **Partly unread:** DyLLM (App. E.2 and F), LESS (App. G onward). Elastic-Cache App. A proof tail only skimmed. SureLock's HTML needed two fetches; the second covered Appendices J-N.
- **Not fetched** (listed in `related_papers.md`, no efficiency numbers extracted): Ada-DLM, TACG, Deferred Commitment Decoding, CoCommit, Adaptive Parallel Decoding, Plan for Speed, DAEDAL, Self Speculative Decoding, Prophet (early commit, Li et al.), CDLM, DiffuSpec, ReFusion, FLARE, Learning Unmasking Policies.
- **Dynamic-dLLM vs DyLLM.** These are two different papers. DyLLM is arXiv 2603.08026 (ICML 2026, github scale-snu/DyLLM). Dynamic-dLLM is arXiv 2606.26120 (ICLR 2026, github TianyiWu233/DYNAMIC-DLLM); its id came from a web search and the page was fetched directly. I did not open the OpenReview page for DyLLM.
- **TraceLock** is arXiv 2605.24697 ("The Path Matters: Learning a Token-Commitment Policy for Diffusion Language Models") as listed in `related_papers.md`.

## 1. Setup comparison (what each speed number was measured on)

| Paper (arXiv) | GPU | Batch | Precision | Lengths / block | Baseline the speedup is against | Timing repeats |
|---|---|---|---|---|---|---|
| SureLock 2602.06412 | NOT REPORTED | 4 (Exp 1,3); 1,2,4,8 (Exp 4) | NOT REPORTED | 64-1024; one block = N_gen; steps = N_gen | same sampler without locking | NOT REPORTED |
| Elastic-Cache 2510.14973 | A100 80GB | 4 / 8 only in Table 14; else NOT REPORTED | NOT REPORTED | 256, 512, 1024; cache block 32 | vanilla LLaDA, "we re-run"; Table 8 uses Fast-dLLM = 1.0x | NOT REPORTED |
| Fast-dLLM v1 2505.22618 | A100 80GB | "single batch" (Fig 1); sweep in App C.5 | NOT REPORTED | 256, 512, 1024; block 32 | vanilla, no cache, one token per step | NOT REPORTED |
| Fast-dLLM v2 2509.26328 | A100, H100 | 1,4,32,64 (figures) | NOT REPORTED | NOT REPORTED | AR Qwen2.5-7B-Instruct | NOT REPORTED |
| dLLM-Cache 2506.06295 | RTX 4090 | 8 (Tables 1-2), 1 (Table 3) | NOT REPORTED for inference | per task (Table 12) | uncached same model | NOT REPORTED |
| dKV-Cache 2505.15781 | A6000 (LLaDA), H20 (Dream) | NOT REPORTED | NOT REPORTED | per Table 5 | few-step baselines chosen to be speed-matched | NOT REPORTED |
| Learn2PD 2509.25188 | 4x A6000 | NOT REPORTED | NOT REPORTED | 256, 1024; block 32 | vanilla LLaDA TPS at same length | NOT REPORTED |
| Window-Diffusion 2601.20332 | A6000 | NOT REPORTED | **FP32** | 256-1024 | unmodified Dream/LLaDA, parallel decoding disabled | NOT REPORTED |
| FlashDLM 2505.21467 | RTX 6000 Ada 48GB | NOT REPORTED | NOT REPORTED | max_new_tokens 1024 | vanilla Dream-7B-Instruct / LLaDA | NOT REPORTED (torch.cuda.Event) |
| d2Cache 2509.23094 | RTX 3090 24GB | NOT REPORTED | NOT REPORTED | 256 / 512 / 1024; block 32 | Vanilla one token per step | NOT REPORTED, single run |
| dInfer 2510.08666 | 8x H800 | **1** | NOT REPORTED | 1024, block 64 | Fast-dLLM (dual cache) and vLLM Qwen2.5-3B | NOT REPORTED |
| DyLLM 2603.08026 | H100 PCIe 80GB | 16 (breakdown); Table 2 NOT REPORTED | NOT REPORTED | 256 steps 256 | original implementation | NOT REPORTED |
| TraceLock 2605.24697 | A40 | NOT REPORTED | NOT REPORTED | 128, 256; window 32-128 | confidence, Fast-dLLM, Learn2PD | profiling 100 samples; repeats NOT REPORTED |
| Polestar 2607.14107 | A100 80GB, GH200 | **1** (App D.11; Sec 5.1 per one fetch) | NOT REPORTED | 256, 512; block 32 | Fast-dLLM, Elastic-Cache, d2Cache etc. | NOT REPORTED |
| LESS 2606.16908 | MI210 (wall-clock single GPU) | NOT REPORTED | NOT REPORTED | 256, block 64 | fixed 256-step sampler, Prophet, KLASS | 2 warmups; repeats NOT REPORTED |
| Sangam 2607.04206 | 8x H100 | queueing | bf16 | 1024/512 appended masks, block 32 | in-system Fast-dLLM, batch 1 | NOT REPORTED |
| MaskKV 2510.09309 | A100 (8x); RTX 4090 for long context | NOT REPORTED | bf16 in memory formula | LongBench, 32K prompt | dLLM-Cache full cache | NOT REPORTED |
| Dynamic-dLLM 2606.26120 | Pro6000; RTX 4090 in App E.3 | NOT REPORTED | NOT REPORTED | 256 / 512; block 32 | vanilla, dKV-Cache, dLLM-Cache, Fast-dLLM | NOT REPORTED |

Observations from the table (PARAPHRASED from the rows above):
- **No paper reports more than a passing statement on timing repeats.** LESS Table 9 shows +/- values and two warmup runs; TraceLock profiles 100 samples for a correction factor. Nobody reports a confidence interval on speed.
- **Only dInfer and Polestar state batch 1 for their speed or memory numbers** (my two Polestar fetches disagree on whether Sec 5.1 or only the App D.11 memory trace says so). SureLock, Elastic-Cache, dLLM-Cache and DyLLM use batch 4, 8, 8 and 16 in at least some of their speed tables, so their speed gains sit in a more compute-bound regime than ours.
- **No other paper runs on our GPU class at batch 1 with bf16 in the text I could read.** The closest hardware are FlashDLM (RTX 6000 Ada, 48GB) and d2Cache (RTX 3090, 24GB).

## 2. Per-paper extraction (efficiency, quality, FLOPs-versus-latency, unusual)

Every line is PARAPHRASED unless marked. Full status per field is in `survey.json`.

### 2.1 SureLock (2602.06412, ICLR 2026) - the closest compute-lock baseline
- **FLOPs (PARAPHRASED, footnote 3):** "algorithmic FLOPs" count GEMMs only: Q/K/V/output projections, QK^T, AV, FFN. Element-wise ops, LayerNorm, softmax, RoPE and cache I/O are excluded. The vocabulary (LM-head) matmul is not in the formula and not discussed. Per-step ratio = active positions M_t / total positions T_b. Table 1 ratios 0.475-0.55, Table 2 0.50-0.70, challenging HumanEval set 0.459-0.470 (Table 6, App J).
- **Active-position ratio r-bar** (micro-averaged, Table 1, Fig 6-7): the only lock-fraction measure; no lock counts.
- **E2E-TPS (QUOTED, App K):** "The numerator is the total number of unmasked tokens produced by the iterative sampling process." Timed from before the first decoding call to after the last, after device sync, so it includes inter-batch gaps and launch overhead. Step-TPS = newly unmasked tokens per step / step wall time. CUDA events with synchronization. Whether the numerator excludes EOS/padding is not stated.
- **Not reported:** per-sample latency, NFE as a column (steps S are tabulated), memory, cache refresh counts, GPU model, precision, timing repeats.
- **Quality beyond accuracy:** Gen-PPL of WikiText-103 continuations under LLaMA-3-8B (App I.1); MT-Bench with gpt-4o judge, one decimal, no variance (App I.2); HumanEval Pass@1 on the **first 70 problems** only at N_gen = S = 256: baseline 18.6, SureLock 20.3 (eps 5e-4) and 18.6 (eps 5e-3) (App J, Table 6). Output-change rate, seeds, CIs, tests: NOT REPORTED.
- **FLOPs versus latency (QUOTED):** Exp-4: "no runtime gain was observed under relatively light computational load settings" (batch 1, 64 tokens); Limitations: FLOPs are "a kernel- and hardware-agnostic proxy for theoretical savings"; gains from vanilla PyTorch called "conservative" (App L). This is the strongest primary-source statement of the caveat (see Section 3).
- **Fetch-tool observations (unverified):** text-versus-table mismatches (Gen-PPL ">=1.21x" vs 1.37x in Table 2; "about 0.5x" vs 0.459-0.470).

### 2.2 Elastic-Cache (2510.14973)
- **Throughput (QUOTED, Sec 4.1 / App B):** tokens per second "averaged until emitting" EOS, following Fast-dLLM. Padding handling: NOT REPORTED.
- **Latency:** whole-benchmark wall time only (App C.3, Table 14: GSM8K 512 tokens LLaDA-1.5, batch 4, 1 GPU: Fast-dLLM 1.86 h vs 0.83 h; batch 8, 2 GPU: 1.00 h vs 0.56 h).
- **Peak GPU memory (Table 12, LLaDA-1.5 GSM8K at 256/512/1024):** baseline 19.04/19.62/20.79 GB, Fast-dLLM 20.49/21.42/23.26, Elastic-Cache 18.11/18.13/19.37. Measurement method NOT REPORTED.
- **Cache update frequency** (Fig 3b-c; text 20% at gamma 0.95; 5.6%-17.2% in a schedule study) and a MAC count for the update trigger (Table 13: 6.7e7 vs 1.3e10 MACs, "<0.5% overhead").
- **Quality beyond accuracy, NFE, FLOPs, variance, output agreement:** NOT REPORTED.
- **Speedup reference changes between tables** (vanilla in Table 1, Fast-dLLM = 1.0x in Table 8). Many speedup and accuracy inconsistencies are flagged by the fetch tool (unverified).
- Our project has already reproduced the official code on RTX 5000 Ada (`docs/elastic_cache_official_report_2026-09-07.md`).

### 2.3 Fast-dLLM v1 (2505.22618) and v2 (2509.26328)
- **v1 throughput (QUOTED, Sec 4.1):** "average number of output tokens generated per second", counted until `<eos>`. **NFE** is named and used for the threshold study (Table 8). Latency only as a single LLaDA-V example (63.0 s to 6.8 s, App B.4). FLOPs, memory: NOT REPORTED. Accuracy only; no variance.
- **v1 caveat:** the cache is "an approximate decoding process" (Sec 3.2); at larger batch sizes LLaDA is compute-bound (App C.5), where the authors study caching alone because parallel decoding is "affected by dummy input tokens".
- **v2:** block-diffusion fine-tune of Qwen2.5. Throughput on GSM8K "from 39.1 to 101.7 tokens/s" at threshold 0.9; "up to 1.5x higher throughput at batch size 64" on A100, up to 1.8x on H100; caching gains "negligible" at small batch. Throughput timing scope, latency, FLOPs, memory, generation length: NOT REPORTED. It is a different model, so its numbers cannot be compared with LLaDA runs.

### 2.4 dLLM-Cache (2506.06295)
- **TPS and FLOPs per token** are both reported (Tables 1-2). FLOPs (QUOTED): "average Floating Point Operations (FLOPs) per token", with a tera-FLOPs column label (fetch-tool observation: unit conflict); measured versus analytic is NOT REPORTED, analytic formulas in App A.7.
- **Memory:** Table 3 "Memory (GB)" 16.94 to 17.93 GB (+5.8% by our arithmetic from the tool's numbers; text says about 5%); Table 7 "Memory"; Sec 5 calls it peak GPU usage.
- **Refresh controls:** prompt interval K_p (about 33-69 on average by model), response interval K_r (about 4-6), update ratio rho = 0.25 (Sec 3.2, 4.1; per-task Tables 13-16).
- **FLOPs versus latency (QUOTED, Sec 5):** "practical speedup from adaptive partial updates is constrained by fixed operational overheads". Fetch-tool observation: FLOPs speedups exceed TPS speedups in Tables 1-2 (for example LLaDA Base GSM8K 5.02x FLOPs vs 3.17x TPS) and the paper does not discuss the gap.
- LongBench (Table 6) is the only long-context quality panel; cosine similarity of K/V/AttnOut/FFNOut is a diagnostic only. App A.10 (QUOTED): "it is not fundamentally lossless".

### 2.5 dKV-Cache (2505.15781)
- tokens/s with undefined "(x)" speedups; "cache ratio" = average over steps of cached tokens divided by processed tokens (QUOTED definition, Sec 4.1; values only in plots); refresh intervals, not counts (Decode 8, Greedy 2).
- **Baselines are speed-matched few-step samplers** ("comparable to or slower than ours", Sec 4.1), so its accuracy comparisons depend on that matching choice.
- **Batch 1 (App A, D):** memory-bound gather/scatter means caching "can actually underperform" the baseline; speedup is batch-dependent (PARAPHRASED).
- LLaDA timed on A6000 and Dream on H20, so cross-model speed is not comparable.

### 2.6 Learn2PD (2509.25188) and the notes' word-lock context
- TPS (QUOTED): "the average number of tokens generated per second"; speedup = TPS ratio to vanilla LLaDA (3.41 TPS at 256, 0.54 at 1024). The numerator is not defined; this matters for End-of-Text-Prediction (EoTP) runs.
- EoTP: about 89.59% of baseline computation is decoding padding after [EoT] (Sec 3.3, App B); EoTP alone gives 5.98x at 1024 (Table 5). Dual cache reaches 57.51x with accuracy 79.83 to 74.00 (Table 2).
- The oracle "15-20x" claim is step-based, not wall-clock. Filter precision/recall/oracle agreement: NOT REPORTED (loss curve only). Latency, FLOPs, memory, NFE: NOT REPORTED.
- Our own reading of the official code (`results/learn2pd_control_20261108/OFFICIAL_CODE_NOTES.md`) differs from the paper text on the tau comparison and the EoT cut; that note is the authority for the code.

### 2.7 Window-Diffusion (2601.20332)
- FP32 on A6000; parallel decoding disabled for fairness (Sec 5.2). Throughput tokens/s (Table 2) **and** latency in seconds (Table 3). NFE, FLOPs, memory, token-role counts: NOT REPORTED.
- Fetch-tool observation: the headline 99x (Table 3, MBPP, WD-Adaptive) compares early termination with fixed-length baselines; fixed-length speedups are 2.3-6.6x (Table 2).

### 2.8 FlashDLM (2505.21467)
- **Latency (QUOTED):** "end-to-end problem solving time per problem" (Table 3 caption), measured "via torch.cuda.Event". **Peak GPU memory (QUOTED):** "recorded directly via torch.cuda.max_memory_allocated()", with separate DLM, guider and total columns. Throughput, NFE, FLOPs: NOT REPORTED. Quality: accuracy only.
- Fetch-tool observation: baseline and guided runs used different max output tokens (256 vs 1024).

### 2.9 d2Cache (2509.23094)
- Throughput (QUOTED) "average number of tokens generated per second", latency (QUOTED) "average inference time per sample", and peak memory only at one setting (Table 7, Dream-Instruct, 1024 tokens) plus a theoretical estimate (App D.1). NFE, FLOPs, refresh counts: NOT REPORTED. Single run per configuration.
- Fetch-tool observation: "(x)" speedups follow throughput ratios, latency ratios differ (Dream GSM8K 4.7x vs 4.0x); main-table d2Cache uses NAR decoding where baselines use semi-AR.

### 2.10 dInfer (2510.08666)
- **TPF** = (1/N) sum_i T_i / F_i and **TPS** = (1/N) sum_i T_i / t_i, with T_i the tokens generated **before the first EOS** and F_i the diffusion iterations (Sec 4.1, PARAPHRASED). These are means of per-sequence ratios.
- Batch size 1, generation length 1024, block 64 (QUOTED, Sec 4.1). Early termination ("halts the diffusion loop and fills all remaining blocks with EOS") is credited with 15-40% efficiency (Sec 3).
- FLOPs, memory, latency: NOT REPORTED. **The project notes say dInfer states that FLOPs savings do not imply latency savings. A full read plus a targeted search of the paper text found no such statement; the paper never discusses FLOPs.** See Section 3.

### 2.11 DyLLM (2603.08026) and Dynamic-dLLM (2606.26120)
- **DyLLM:** throughput (QUOTED) "the average number of tokens generated per second", H100, batch 16 for the breakdown, Table 2 batch NOT REPORTED. Refresh: full-sequence pass every T_full = 4 steps. No latency, NFE, FLOPs, memory. README names throughput only (about 9.6x). Fetch-tool observation: for Dream at nu = 1, Fast-dLLM DualCache is faster than DyLLM on all four benchmarks.
- **Dynamic-dLLM:** TPS ratio (Tables 1-3), seconds per inference with controller overhead listed separately (Table 8: at 256 tokens baseline 29.16 s, DCU 9.41 s of which 1.94 s is extra time, APD 6.87 s with 0.11 s extra), and number of inference steps NIS (Table 7). FLOPs, memory, update ratio: NOT REPORTED.

### 2.12 TraceLock (2605.24697)
- The efficiency axis is **average executed decoding steps**, normalized by the maximum scheduled steps (Sec 4.1, Table 1). Wall-clock appears only in App D, Table 5: seconds per sample, a measured prototype and an estimated corrected "TraceLock*" using a profiled factor (100 samples, CUDA events). Throughput, FLOPs: NOT REPORTED. No K/V caching.
- **Step reductions do not translate into latency (PARAPHRASED, App D, Table 5):** the prototype is slower than Fast-dLLM in several LLaDA rows (Math 128/32: 6.39 s vs 4.14 s as extracted).
- Quality: LLM-judge QA ranking with Llama-3.1-8B-Instruct (auxiliary), and AUROC / AP / Acc@0.5 for the commit predictor (App E, Table 6). Single seed 42; no CIs.

### 2.13 Polestar (2607.14107)
- TPS = end-to-end wall-clock decoding time over the full fixed-length trajectory (App D.1); TPF = average tokens decoded per step (footnote 1). **Peak memory (App D.11):** traced from just before decoding, includes weights, KV buffers and method buffers; LLaDA-8B-Instruct, GSM8K, 512 tokens, block 32, **batch 1**, A100: Elastic-Cache 19.5 GB, d2Cache 18.5 GB, Polestar 18.6 GB. Per-layer latency breakdown (App D.10): 1.63 ms saved by system optimizations, 2.71 ms critical-path overhead before overlap, <=0.10 ms after.
- Quality beyond accuracy: 86% agreement of the drift-based refresh policy with the top-25% most-stale positions (Sec 5.3).

### 2.14 LESS (2606.16908)
- Steps: "the average number of reverse denoising steps executed per sample". FLOPs ratio FLOPs_Base / FLOPs_LESS for Dream-7B GSM8K only (Table 10; analytic Eq 23 and measured forwards). Wall-clock for one model and one benchmark (Table 9: Base 19.45, Prophet 7.20, KLASS 7.62, LESS 5.16 s per sample, 3.77x).
- **FLOPs versus wall-clock:** the FLOPs speedup (5.01x at T = 256) and the wall-clock speedup (3.77x) differ and the text does not fully reconcile them (as extracted).
- Diagnostics: per-step top-1 flip rate, fallback invocation rate (2.1-8.9%). The only paper here with a reported timing protocol detail (two warmup runs, synchronized timing, +/- values).

### 2.15 Sangam (2607.04206) and MaskKV (2510.09309)
- **Sangam** is a serving paper: E2E latency mean and P99 under Poisson QPS, sustained QPS; TTFT and TBT "deliberately" not used because re-prefills make them ill-defined. No quality evaluation, no FLOPs, no KV memory in GB. Its Fast-dLLM baseline is in-system at batch 1, called "a floor rather than a headline result".
- **MaskKV** is a long-context memory paper: peak GPU memory and KV size (Eq 17, bf16), Table 5 ablation 68.13 to 23.42 GB at 32K, "up to 31x" TPS speedup at 32K, LongBench relative score retention (94.33% LLaDA at 256 pairs). Whether "256 pairs" is per head or total is ambiguous in the extracted text.

## 3. FLOPs savings versus latency savings: what the papers actually say

The notes (`related_papers.md`, item 28) say dInfer "important because FLOPs savings do not automatically imply latency savings". **That attribution is not supported by dInfer's text** (full read, then a targeted search for FLOPs, latency, compute-bound, overhead wording; none found). The papers that do carry the point, with locators:

| Paper | Statement | Status |
|---|---|---|
| SureLock | Exp-4: "no runtime gain was observed under relatively light computational load settings"; Limitations: FLOPs "a kernel- and hardware-agnostic proxy for theoretical savings"; irregular token-sparse compute and cache access overheads can offset FLOP savings | QUOTED + PARAPHRASED |
| dLLM-Cache Sec 5 | "practical speedup from adaptive partial updates is constrained by fixed operational overheads" | QUOTED |
| dLLM-Cache Tables 1-2 | FLOPs speedups above TPS speedups (e.g. 5.02x vs 3.17x) | TOOL OBSERVATION |
| dKV-Cache App A, D | at batch 1, memory-bound gather/scatter, caching "can actually underperform" | PARAPHRASED |
| Fast-dLLM v2 Sec 4.3 | caching gains "negligible" at small batch, "substantial" when compute-bound | QUOTED + PARAPHRASED |
| TraceLock App D, Table 5 | fewer steps do not mean lower latency; controller overhead | PARAPHRASED |
| LESS Table 9 vs Table 10 | 5.01x FLOPs vs 3.77x wall clock | PARAPHRASED |
| Polestar App D.10, Dynamic-dLLM Table 8 | per-component overhead in ms or seconds | PARAPHRASED |

This agrees with our own finding (`surelock analysis.md` Section 2): a rough fit of about 41 ms fixed plus 22 ms scaling with active rows per call, so FLOP savings alone cap the speedup.

## 4. Conflicting definitions between papers

| Quantity | Conventions found |
|---|---|
| Throughput numerator | tokens until first EOS (Fast-dLLM, Elastic-Cache, dInfer [per-sequence mean of T_i/t_i]); "total number of unmasked tokens produced" (SureLock E2E-TPS, includes padding unless excluded; not stated); "average number of tokens generated per second" with numerator not defined (Learn2PD, d2Cache, DyLLM, dLLM-Cache, Window-Diffusion, Dynamic-dLLM, Polestar over the fixed-length trajectory) |
| Averaging | mean of per-sequence ratios (dInfer) vs aggregate tokens over wall time (SureLock E2E-TPS) vs unspecified; fetch-tool observation: d2Cache throughput and latency ratios disagree |
| Speedup reference | vanilla one token per step (dLLM-Cache, d2Cache, Learn2PD, Fast-dLLM v1, Dynamic-dLLM); same sampler without locking (SureLock); Fast-dLLM = 1.0x (Elastic-Cache Table 8, dInfer); few-step samplers matched in speed (dKV-Cache); AR model (Fast-dLLM v2, dInfer vs vLLM) |
| Latency | per sample (d2Cache, LESS, TraceLock, Dynamic-dLLM); per problem via cuda events (FlashDLM); whole-benchmark hours (Elastic-Cache Table 14); E2E mean and P99 under load (Sangam) |
| FLOPs | GEMM-only algorithmic ratio, LM head not in formula (SureLock); "FLOPs per token" with tera-FLOP column (dLLM-Cache); analytic per-forward formula (LESS) |
| NFE or steps | NFE (Fast-dLLM v1); executed steps (LESS, TraceLock); TPF (dInfer, Polestar); NIS (Dynamic-dLLM) |
| Cache refresh | interval K_p / K_r and ratio rho (dLLM-Cache); interval (dKV-Cache, DyLLM T_full); update frequency plot (Elastic-Cache); none (SureLock, Fast-dLLM v1 other than per block) |
| Memory | peak allocated including weights (FlashDLM, Polestar D.11); cache storage formula (dLLM-Cache, d2Cache, MaskKV); different models and lengths in every paper |
| Early stop | counted inside speedup without separation (Learn2PD EoTP, dInfer, Window-Diffusion WD-Adaptive) |

## 5. Proposed comprehensive comparison

### 5.1 What we already save, checked against the record files

I grepped the record directories (not the code) for each field. Records are `results/<run>/records/*.jsonl`.

| Field | Meaning | Present in |
|---|---|---|
| `tokens`, `text`, `correct`, `example_id`, `arm`, `gen`, `prompt_len` | output and correctness; paired join on `example_id` and `arm` | all runs checked |
| `latency_s`, `nfe`, `first_eos`, `dense_calls` | wall time per decode, calls, index of first EOS in the answer, dense calls | all runs checked |
| `partial_calls`, `rows_per_call_mean` | partial (cached) calls, mean computed rows per call (mean over layers) | depth_refresh, lctx_refit (partial); `rows_per_call_mean` also in learn2pd_control, farcache_stack, staleness |
| `layer_rows`, `refresh_kind` | per call and per layer computed-row counts; kind of each call | depth_refresh stages A-E and lctx_refit stageG, not older runs |
| `refresh_calls` | refresh call count | farcache_stack |
| `rule_time_s` | time spent in the gate/controller | learn2pd_control, farcache_stack, depth_refresh stage E, lctx_refit stageG |
| **`peak_mem_gib`** | `torch.cuda.max_memory_allocated()/2**30` per decode, includes about 15 GiB of weights | **learn2pd_control (all four files) and farcache_stack stage1/stage2. NOT in staleness, in depth_refresh records (0 of the stage files carry it), or in lctx_refit. `run_depth_cells.py` in the working copy now sets it, so new depth runs will.** |
| `fallback_calls`, `eos_position`, `loadavg1`, `proc` | misc. diagnostics | various |

`results/depth_refresh_20261109/orchestration/metrics_extra.py` (already in the working copy) computes calls, ms per call, answer tokens/s, algorithmic TFLOPs (SureLock convention, head excluded), K/V cache memory, and "text differs from dense C-0.9". I did not run it.

Timing code, read: in `src/regret_remasking/paper_decode.py` (lines 304-305 and 798-799) and in `src/regret_remasking/selective_cache.py` (lines 587-588) `latency_s` is `time.perf_counter()` taken after a CUDA synchronize at both the start and the end of the decode, so it is device-synchronized like SureLock's, FlashDLM's and LESS's timing. In `paper_decode.py` the start precedes prompt tokenization, so `latency_s` includes tokenization, and the 'rule time' is separate. I did not check any other decoder (for example the Elastic-Cache or Fast-dLLM official-code runs). Still to verify: which arm is the "one word per call" reference for each length, and whether it exists at 512 and 1024.

### 5.2 Ranked metric table

"Have" = computable from saved records now; "Need" = new instrumentation or runs.

| Rank | Metric | Reported by | Definition to adopt | Have / need | Pitfalls when set beside published numbers |
|---|---|---|---|---|---|
| 1 | Task accuracy (pass@1 or exact match), with count and paired bootstrap over questions | all | correct answers / questions per set; greedy decoding; give the delta in answers versus the dense reference, with a bootstrap over `example_id` | Have (`correct`) | Published sets differ (SureLock HumanEval = first 70 problems; shots and lm-eval harness in Fast-dLLM, dLLM-Cache, d2Cache); our MATH 108 is a subset; do not compare absolute accuracy |
| 2 | Output throughput, tokens/s | Fast-dLLM, Elastic-Cache, dInfer, Learn2PD, dLLM-Cache, d2Cache, DyLLM, Dynamic-dLLM, Polestar, Window-Diffusion, SureLock (E2E-TPS) | primary: answer tokens (index of first EOS, from `first_eos`) / `latency_s`, reported both as ratio of means and mean of per-sample ratios (dInfer uses the latter); secondary: `gen`/`latency_s` (all positions, closest to SureLock's "total unmasked tokens") | Have | The numerator is the main incompatibility between papers (Section 4). State both. Never compare tokens/s across GPUs |
| 3 | Speedup ratio | all efficiency papers | latency of dense reference / latency of arm, paired per question; three references reported separately: dense C-0.9 (our reference), dense C-0.9 with end-of-text truncation (matched, our `T:` arm), and one-word-per-call vanilla if available | Have for C-0.9 and its truncation twin; vanilla only if the run exists at that length | Published speedups are almost all against one-token-per-step vanilla (Fast-dLLM, dLLM-Cache, d2Cache, Learn2PD, Dynamic-dLLM), SureLock against the same sampler without locking. Our dense reference already writes many tokens per call, so our numbers are smaller and not comparable. Early termination is inside several published speedups (Learn2PD, dInfer, Window-Diffusion) |
| 4 | Latency per sample (s), mean and median | d2Cache, FlashDLM, Window-Diffusion, LESS, TraceLock, Dynamic-dLLM, Elastic-Cache (whole-benchmark) | seconds per question from `latency_s`, with a paired bootstrap interval; one warmup decode excluded | Have (`latency_s` is device-synchronized, Section 5.1) | Per-problem versus whole-benchmark definitions differ; single-run timing noise is not reported by anyone. Ours: add a repeated-timing subset to bound noise (Need) |
| 5 | Calls (NFE) and tokens per forward (TPF) | Fast-dLLM v1 (NFE), LESS and TraceLock (steps), dInfer and Polestar (TPF), Dynamic-dLLM (NIS) | NFE = `nfe`; TPF = answer tokens / `nfe` per sample, as dInfer | Have | Fewer steps is not fewer FLOPs once partial calls are used; TraceLock and LESS both show steps do not map to latency |
| 6 | Algorithmic FLOP reduction | SureLock, dLLM-Cache, LESS | SureLock convention: per layer and per computed row 4dT + 4d^2 + 4d*d_kv + 6d*d_ff (with the actual per-layer row counts), ratio to dense; state that the LM head is excluded, and also give a variant with the head | Have for runs with `layer_rows`; for older runs use total computed row-layers = `rows_per_call_mean` x 32 layers x `nfe` (`rows_per_call_mean` is the layer average, `sum(layer_rows)/32/nfe` in `run_depth_cells.py`; the ratio to dense survives without the factor 32, absolute TFLOPs do not) | SureLock's ratio omits the head; by our arithmetic (d = 4096, vocabulary about 126k) head is about 7% of a dense row's cost. dLLM-Cache counts per token with another unit. FLOP cuts do not predict latency (Section 3) |
| 7 | Peak GPU memory (GiB) and analytic K/V cache size | dLLM-Cache, Elastic-Cache, FlashDLM, d2Cache, Polestar, MaskKV | `torch.cuda.max_memory_allocated()` reset before each decode; report absolute and the increase over the dense arm; analytic K/V bytes from rows (as `metrics_extra.py`) | Partial: present in learn2pd_control and farcache_stack; missing in depth_refresh and staleness records; new depth runs will have it | Peak includes about 15 GiB of weights, so relative differences look tiny (Elastic-Cache: 18.1 vs 19.6 GB); different models, lengths and batch sizes across papers |
| 8 | Cache refresh frequency and recompute ratio | Elastic-Cache (update frequency), dLLM-Cache (K_p, K_r, rho), dKV-Cache (interval, cache ratio), DyLLM (T_full) | dense calls / calls, partial calls / calls, refresh calls / calls, and recomputed rows / total rows (SureLock's active-position ratio r-bar) | Have (`dense_calls`, `partial_calls`, `refresh_calls`, `layer_rows`) | Papers give intervals or a plot, rarely a count; definitions are not common |
| 9 | Controller/gate overhead | TraceLock (App D factor), Dynamic-dLLM (Table 8), Polestar (D.10 ms), Learn2PD ("negligible", unmeasured) | `rule_time_s` / `latency_s` and the absolute seconds | Have where `rule_time_s` exists (see 5.1) | Gate time must be GPU-synchronized to mean anything (our code has a `sync_gate_timer` flag) |
| 10 | Per-call time (ms/call), dense versus partial | Polestar (per-layer ms), dLLM-Cache and dKV-Cache (arguments only) | mean latency / calls overall; per-call timing split by dense and partial calls | Overall: have (`latency_s`/`nfe`). Split by call kind: Need (no per-call timing saved) | Our fixed-cost estimate (about 41 ms per call) comes from a fit, not a direct measurement |
| 11 | Output-change rate versus the dense baseline (token-identical up to first EOS; share of differing positions) | **no paper** (SureLock, Elastic-Cache, dLLM-Cache all NOT REPORTED) | exact equality of `tokens[:first_eos]` against the dense arm on the same question, plus fraction of differing positions over the shorter answer | Have (`tokens`, `first_eos`; `metrics_extra.py`) | Our addition. Not comparable to anything published; do not present as a literature convention |
| 12 | Correct-to-wrong and wrong-to-correct counts (lost/gained) | no paper | paired counts versus the dense arm per question set | Have (`correct`) | Our addition; needs the bootstrap interval alongside |
| 13 | Answer-value change (final extracted answer differs from the dense answer) | no paper | compare scored final answers | Have if the scored files carry the extracted answer; otherwise Need | Our addition |
| 14 | Quality beyond accuracy: Gen-PPL, LLM-judge, LongBench | SureLock (Gen-PPL, MT-Bench), TraceLock (judge), dLLM-Cache and MaskKV (LongBench) | none proposed: not applicable to code and math at the lengths here | Need (a scoring model) | Cannot be compared with our correctness panel. This is the gap SureLock's "comparable quality" claim falls into (see `surelock analysis.md` Section 6) |
| 15 | Statistical reporting: bootstrap CIs, timing repeats, seeds | LESS (+/- on latency, warmup), SureLock none, d2Cache none | paired bootstrap over questions for accuracy deltas and latency ratios; repeat a subset to bound timing noise; decoding is greedy so no seed variance | Accuracy CI: have. Timing repeats: Need | Almost nobody reports intervals, so our CIs are a strength, not a convention |
| 16 | Early-termination effect | Learn2PD (EoTP), dInfer, Window-Diffusion (WD-Adaptive) | speedup with and without end-of-text truncation, reported separately | Have (the truncation twin arm; `first_eos`) | We measured truncation as 22% of the HumanEval log-speedup (`findings_2026-11-09.md`); published headline speedups often include it |

### 5.3 Metrics we cannot support yet (larger batch, longer context, other regimes)

- **Batch-size scaling** and compute-bound speedups: SureLock Exp-4 (B = 1, 2, 4, 8), Fast-dLLM v1 App C.5 and Fast-dLLM v2 Fig 5/6b, dKV-Cache App D, DyLLM (batch 16), Elastic-Cache Table 14 (batch 4, 8). Our batch 1 is the regime where SureLock and dKV-Cache say gains vanish.
- **Serving metrics:** QPS, P99 latency, queueing (Sangam); tensor/expert parallel, CUDA Graph and torch.compile effects (dInfer); multi-GPU latency (Elastic-Cache Table 14).
- **Long-context memory and speed:** 32K prompts and KV budgets (MaskKV), LongBench (dLLM-Cache, MaskKV, Elastic-Cache), our longest generation is 1024.
- **Throughput on different hardware:** absolute tokens/s is not transferable between RTX 5000 Ada and A100, H100, H800, A6000, 4090, 3090, MI210 or H20.
- **Models other than LLaDA-8B-Instruct:** Dream, LLaDA-1.5, LLaDA-MoE, LLaDA-V (Fast-dLLM v2 uses a different model altogether).

### 5.4 Which planned claims would be incomparable with the published numbers

1. **Any absolute speedup versus our dense C-0.9 reference** set next to published "Nx" figures (Fast-dLLM 27.6x, Elastic-Cache 25.2x on GSM8K 512 (Table 1, as extracted), Learn2PD 57.51x, dInfer about 10x, Window-Diffusion 99x). Those are against one-token-per-step vanilla, often with early termination counted, often at larger batch.
2. **FLOP-cut percentages** (SureLock 45-53% and our 27.5% and 62.6% versus its 0.47-0.55x ratios): SureLock's setting is one fully parallel block, steps = length, batch 4, and quality judged by Gen-PPL and MT-Bench; ours uses blocks of 32 and exact correctness. Also the head is excluded in its formula.
3. **Accuracy-preservation claims** ("no loss"): published accuracy panels are few-shot lm-eval or MT-Bench; SureLock's HumanEval is the first 70 problems; ours is 164 problems at 512 tokens, MATH subsets, GSM8K 200. Compare paired deltas inside our own protocol only.
4. **Memory comparisons:** peak memory from our 15 GiB-weights process versus papers' different models, batch sizes and lengths (Polestar D.11 is the only batch-1 memory trace at 512 tokens: 18.5-19.5 GB on an A100).
5. **Tokens/s** in absolute terms, because of the numerator problem (Section 4) and the hardware.
6. **"Fewer steps" as a result** (TraceLock, LESS, Dynamic-dLLM NIS) versus our call counts: step counts there are measured against a fixed full budget, not against a confidence-0.9 reference.
7. **Claims of "same output as dense"** have no published counterpart in the cache and word-lock papers read here (only Self Speculative Decoding, not fetched, is listed in the notes as an exact-output method).

## 6. What I could not access or verify

- No PDFs were opened; everything came through the summarizing fetch tool on arXiv HTML pages. Tables lost column headers in some papers (dKV-Cache Tables 1, 2, 5; FlashDLM Table 3), and figures are images, so numbers that live only in figures (SureLock Fig 3, Elastic-Cache Fig 3, Fast-dLLM v2 Fig 1/5/6, dKV-Cache Fig 3) are not in this survey.
- DyLLM App. E.2 and F, LESS App. G onward, and Elastic-Cache App. A tail were not read. No paywalled source was needed.
- The venue labels in `related_papers.md` (ICLR/ICML 2026) were not re-checked.
- The "fetch-tool observation" items are labelled and not relied on; any that matter for a paper (for example speedup arithmetic) should be re-read in the PDF before citing.
- The existence of a one-word-per-call arm at 512 and 1024 tokens was not verified; `latency_s` device synchronization was verified only for `paper_decode.py` and `selective_cache.py`.
