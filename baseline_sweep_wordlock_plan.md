# Word-lock sweep: word locks under ONE shared representation setting (plan written 2026-11-19, before any run of this sweep)

Status: PLAN ONLY. Same format as the representation-cache sweep (Study 5, `results/replock_clean_20261117/PREREGISTERED.md`), with the roles swapped: there the word lock was fixed and the representation caches varied; here the representation setting is fixed and the word locks vary. Open points are marked OPEN.

Terms. **Word lock** = which blanks get a word on a call (saves calls). **Representation cache** = which rows' K/V are reused (saves compute per call). Row types: prompt, written, blank. In this sweep there is NO representation cache: every call is a dense forward, so the only thing that changes between arms is the word lock, and speed differences come from the number of calls.

Standing rules. Speed comes first; answer parity is acceptable (a gate, not the target). Quality reference = the model's ORIGINAL decoding (one token per step, block 32, no acceleration). Tables sorted by speed. Answer change = paired delta with CI, and whether the CI includes 0; no tolerance imposed (the project owner decides tolerance and the primary metric).

## 1. Design
- **Representation setting fixed: dense, no cache.** A dense call every call, no early stop (all positions decoded, as the released code does), block 32, temperature 0, bf16, batch 1, one GPU.
- **Baseline per model = the original decoder** (one token per step, block 32, all gen-length calls). One baseline per model; nothing is compared across models.
- **Reference row: Conf-0.9** (write every blank of the block with confidence >= 0.9, else the single best blank; Fast-dLLM's threshold rule), which is the word lock of the representation-cache sweep.
- **Arms (word locks).** Established ones, released code first, our implementation only where no code exists:

| # | Word lock | Source | How it is run | Settings (frozen, from the source) |
|---|---|---|---|---|
| 1 | Conf-0.9 (Fast-dLLM threshold) | released code | d2Cache repo `generation=vanilla generation.threshold=0.9` and our engine `C-block`; both reported | threshold 0.9, at least one word per call |
| 2 | KLASS | released code (`github.com/Kamichanw/d2Cache`, `generation=klass`) | d2Cache repo, unpatched | `kl_threshold` 0.01, `kl_history_length` 2 |
| 3 | EB-Sampler | released code (same repo, `generation=eb_sampler`) | d2Cache repo, unpatched | `gamma` 0.001 |
| 4 | Learn2PD | official code (`github.com/ims-kdks/Learning-to-Parallel-Decoding`), released LLaDA-8B-Instruct block-32 filter weights | our engine (`X-block`, `block_rows`), validated against the official generate on a few prompts (notes: `results/learn2pd_control_20261108/OFFICIAL_CODE_NOTES.md`) | filter accepts blank if sigmoid > 0.96; every call also writes the single most confident blank |
| 5 | TraceLock | the collaborator's code (frozen at 24cd51b) with the shared controller and projector checkpoints (branch `share/collab-checkpoints`) | our engine (`T-block` / `lock="trace"`) | cutoff 0.95 |
| 6 | Lctx-0.8, the original selective lock | the collaborator's rule (`Lctx_c0.8.pkl`, same branch) | our engine (`lock="rule"` / `X-block`) | lam 0.8, six inputs, frozen |
| 7 | **NEW: the collaborator's word lock** (modified Lctx / new feature sets) | supplied by the collaborator | see section 5 | frozen before the sweep (hyperparameters never tuned on the sweep sets) |

  Learn2PD weights exist for LLaDA-8B only; on LLaDA-1.5 it is run with the 8B weights unchanged and labelled "transferred, not refit" (OPEN: or skip it there). TraceLock and Lctx were fitted on LLaDA-8B traces; same label on LLaDA-1.5.
- **Models:** LLaDA-8B-Instruct and LLaDA-1.5 first. Dream-7B only after its smoke run, and only with the arms that are model-agnostic (Conf-0.9, KLASS, EB-Sampler, and the collaborator's if it is); Learn2PD, TraceLock and Lctx have no Dream weights or fits.
- **Sets:** HumanEval (164, 512 tokens), KodCode held (150, 512), GSM8K held (100, 256), MATH-108 (512). Full sets, same prompts, tokenizer and scorer as Study 5 (`common.py`), not lm-eval's.

## 2. Metrics (per model, set, arm)
| Metric | Definition |
|---|---|
| Speed | wall-clock speedup over the original decoder, paired over prompts with a bootstrap CI; tokens/s |
| Call count | forward calls (all calls are dense here, so the call ratio is also the FLOP ratio of the backbone) |
| Words per call | mean words written per call (how parallel the word lock is) |
| Call time | mean ms per call, and for learned locks the extra time per call the lock itself costs (filter, controller, projector, rule scoring) |
| FLOP cut | backbone FLOPs vs the original from one counter on 20 prompts per cell, in a separate pass; vocabulary head and any controller/projector/filter cost reported separately (SureLock's convention) |
| Accuracy | paired delta vs the original with its CI, and whether the CI includes 0 |
| Peak memory | existing per-record field |

The fastest arm with no detectable answer loss is named per cell. Tables sorted by speed.

## 3. Control
- **Per-stack dense anchor.** The d2Cache-repo arms (1-3) and the engine arms (1, 4-7) come from different stacks, so each job also runs Conf-0.9 on the same prompts in the same process; speed is reported against the global baseline and against that stack's own Conf-0.9 run.
- **Equivalence check before any full run:** Conf-0.9 in the d2Cache repo vs our engine on 20 HumanEval prompts (Study 5 result: identical output on 12 of 20, calls within 7%; numerics differ, so identical tokens are not expected).
- **Learn2PD validation:** our engine vs the official `generate_learn2parallel` on 5 prompts, calls and outputs compared before use.
- Arms interleaved per prompt inside one process; speed anchor on the first 20 prompts in every job.
- The original decoder and Conf-0.9 baselines of Study 5 (same engine, same prompts, same settings) are reused as the shared baseline rows (OPEN: or re-measure inside this sweep).

## 4. Cost estimate and order of work
Rough, from Study 5 timings (dense call ~92-94 ms; original 512 calls; Conf-0.9 about 54 calls per prompt on HumanEval): an (arm, cell) takes about 0.25-1 GPU-h depending on the arm's calls per prompt; about 6 new arms x 8 cells, about 20-30 GPU-h with baselines reused, plus about 4 GPU-h for the FLOP pass. One GPU card, 2 running jobs at a time.
1. Build the runner for engine arms (Conf-0.9 anchor, Learn2PD, TraceLock, Lctx-0.8, and a slot for the collaborator's lock) and the strategy switch for the d2Cache-repo runner (klass, eb_sampler); smoke each (2 prompts) and check the FLOP fields before any mass submission.
2. Equivalence and Learn2PD validation (section 3).
3. LLaDA-8B on the four sets, then LLaDA-1.5; scoring of the code sets on a CPU compute node; analysis tables per model.
4. Dream if its smoke run works.

## 5. For the collaborator: what to deliver and how it plugs in
- **Preferred: a frozen `ContextRule` pickle** (the same format as `Lctx_c0.8.pkl`: rule kind, feature list, threshold `lam`, optional fitted estimator) with the training recipe and data identifiers. It then runs through `lock="rule"` with no engine change if its features are in `context_rules.FEATURES`; new features need a code change to `lock_features.py` (send the feature definitions).
- **Otherwise: a decoding arm** (a function that, given the confidences and context of the blanks of the current block, returns which blanks get a word) plus any checkpoints and their hashes.
- Must always write at least one blank per call, work inside blocks of 32, and need no look-ahead beyond the current call.
- **Freeze before the sweep:** hyperparameters, thresholds and checkpoints are fixed on tune sets (not HumanEval-164, KodCode-held, GSM8K-held or MATH-108) and recorded with hashes.
- **What the sweep reports for it:** speed, calls, words per call, FLOP cut and accuracy against the original decoder, next to the six established locks above.
- Their parity work on Lctx (the earlier version lost 3-6 HumanEval answers vs Conf-0.8) is judged the same way as every arm: paired delta with CI, no tolerance imposed here.

## 6. Open points
1. Reuse the Study 5 baselines or re-measure inside this sweep.
2. Learn2PD on LLaDA-1.5: transferred weights or skip.
3. Whether Dream is in scope for this sweep.
4. The collaborator's delivery date and format (pickle or arm).
5. The grid of representation cache x word lock (Study 6) uses these same word locks; this sweep supplies its word-lock axis.
