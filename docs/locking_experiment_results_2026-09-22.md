# Locking experiments: settings and results

Verified from saved artifacts on 22 September 2026. The corrected representation-KL experiment and its offline follow-up are complete. The separate TraceLock-style token-locking experiment has completed training and is still evaluating settings on development examples.

## Shared evaluation settings

| Setting | Value |
|---|---|
| Model | LLaDA-8B-Instruct, checkpoint `08b83a6feb34df1a6011b80c3c00c7563e963b07` |
| Evaluation samples | All 124 samples in our fixed panel: 54 MATH Level-5 + 70 HumanEval; not the entire public benchmarks |
| Generation | 256 output positions; block length 32 for representation-locking runs |
| Precision and batch | BF16; one example at a time |
| Seed and proposals | Seed 23; greedy token proposals |
| Hardware | NVIDIA RTX 5000 Ada Generation |
| Timing | Three repeats; median per example, then average across examples |
| Included in timing | Inference, stability checks and caching; excludes loading, scoring and file writes |
| Speedup | `100 × (baseline time / method time − 1)`; negative means slower |

These are the completed representation experiment's settings. TraceLock's final evaluation is planned on the same sample panel, generation budget, precision, seed and timing repeats, but uses a moving token window.

## Experiment 1: representation KL decides when to freeze representations

**Question:** If a token's hidden representation changes very little from one decoding step to the next, can we reuse its stored internal states and save computation?

**What we compare:** the same token's **last-layer representation now versus its last-layer representation at the previous decoding step**. We do not compare different layers, and the online rule does not see the final decoding step.

The representation is the 4,096-number, final-normalized vector immediately before vocabulary prediction. We turn it into a probability-like vector with softmax at temperature 1, then calculate KL(current || previous) in FP32. The three fixed cutoffs are 0, 0.0005 and 0.005. Smaller cutoffs demand a smaller change. There is no additional vocabulary-confidence filter. Only generated positions already unmasked in the current forward input can freeze; prompt and end-marker positions are excluded.

**What freezes:** stored representations and attention keys/values across all layers for the selected token, together with its output logits. Only the last-layer signal is checked; each layer does not get its own stability check. The sparse version skips the frozen token's query/projection/feed-forward/output-head computation on subsequent steps. The dense control keeps doing the full computation while imposing the stored states.

| Method | KL cutoff | Correct / 124 | Accuracy | Seconds/sample | Speedup |
|---|---:|---:|---:|---:|---:|
| Baseline: no representation freezing | — | 43 | 34.68% | 15.776 | 0.00% |
| Measure representation KL, never freeze | — | 43 | 34.68% | 16.671 | −5.37% |
| Representation freeze, keep full computation | 0.005 | 40 | 32.26% | 17.020 | −7.31% |
| Representation freeze, skip frozen computation | 0 | 44 | 35.48% | 15.151 | +4.13% |
| Representation freeze, skip frozen computation | **0.0005** | **43** | **34.68%** | **14.392** | **+9.61%** |
| Representation freeze, skip frozen computation | 0.005 | 41 | 33.06% | 14.328 | +10.10% |
| Token-probability KL → representation freeze, skip computation | 0.005 | 40 | 32.26% | 14.642 | +7.74% |

The last row changes the measured signal to vocabulary probabilities. It still freezes representations; it is **not** the TraceLock token-locking experiment. Equal numerical cutoffs in the two different spaces do not imply equally strict gates.

At representation cutoff 0.0005:

| Dataset | Baseline correct | Freezing correct | Speedup | Paired 95% speedup interval |
|---|---:|---:|---:|---:|
| MATH Level-5 | 5/54 | 4/54 | +9.59% | +7.97% to +11.29% |
| HumanEval | 38/70 | 39/70 | +9.63% | +7.86% to +11.31% |
| Combined panel | 43/124 | 43/124 | +9.61% | — |

**Meaning:** cutoff 0.0005 gives a promising measured speed/quality tradeoff on this panel. The total correct count matches baseline, but six previously wrong answers become correct and six previously correct answers become wrong. It does not preserve every answer or establish equal quality on new data. Cutoff 0.005 is slightly faster and loses two correct answers overall.

All methods still make 256 model calls per sample. The gain comes from doing less work inside each call: cutoff 0.0005 removes 27.10% of the reported query/projection/feed-forward/head token rows. That is not a FLOP percentage. Checking stability alone costs time; retaining full computation while caching also costs time.

## Offline follow-up: does looking stable mean being close to the final state?

**Question:** At the first moment the online rule would allow freezing, is the representation already close to where it will end up?

**Settings:** replay all 124 examples without freezing. After each complete generation, perform one extra forward pass on its completed token sequence to obtain the final representation. Look back at the first point where current-versus-previous KL passes each cutoff, and measure current-versus-final KL there. Use the same final layer, normalization, temperature and eligibility rules as online.

This is an after-the-run diagnostic. Future information never controls the online experiment. The final reference is the model's own generated sequence, not a known correct answer. These are hypothetical freeze decisions on unfrozen trajectories, not measurements of what would happen after intervening.

| KL cutoff | Token positions that would trigger freezing | Still above the same cutoff from the final representation |
|---|---:|---:|
| 0 | 18,095 | 13,905 / 18,095 = 76.84% |
| 0.0005 | 24,578 | 13,541 / 24,578 = 55.09% |
| 0.005 | 24,931 | 12,977 / 24,931 = 52.05% |

Each position is counted separately within each example. There are 25,048 eligible example-position pairs overall. The zero-cutoff row is especially sensitive to tiny numerical differences and does not establish exact equality.

At cutoff 0.0005, the median current-to-final KL at the hypothetical freeze point is 0.004785, above the 0.0005 cutoff. Of 11,037 positions initially close to their final state under this test, 3,115 (28.22%) later move outside the cutoff again. The softmax proxy is very concentrated, so a small proxy KL can coexist with a substantial change in the original vector.

**Meaning:** small change since the previous step does not guarantee that a representation has finished changing. The online speed gain is real in this test, but the offline result does not support a claim of final-state or all-layer convergence. Correctness and speedup are not endpoints of this offline diagnostic.

## Experiment 2: TraceLock-style controller decides which token identities to commit

**Question:** Can a learned controller recognize stable token predictions early and commit several at once, reducing decoding steps?

**Settings:** the controller reads the last three layer representations and their two layer-to-layer differences, compressed to 832 features. A three-layer Transformer predicts which token identities to commit. It uses a 32-token moving window and acceptance cutoff 0.95; when no token passes, it commits one fallback token. Token identities become fixed, while hidden representations and attention keys/values continue updating.

Training uses 8,000 examples (4,000 math, 2,000 code, 2,000 QA), 400 validation examples and 10,000 training updates. A separate 200-example development set is used to select comparison settings. Planned final testing uses the same 54 MATH + 70 HumanEval panel, 256-token budget and three timing repeats.

| Stage / result | Verified status |
|---|---|
| Training | Complete: 8,000 fit, 400 validation, 10,000 updates |
| Development setting selection | In progress; snapshot at 16:34 Dubai on 22 September: 2,250/3,400 records, 132/200 complete cases |
| Final 124-case evaluation | Not yet available |
| Final answer correctness | Pending |
| Final speedup | Pending |

Training completed at 12:37:33 Dubai on 22 September. Development records are not final test results. No final TraceLock speed/quality claim is supported yet.

**Update (2026-10-01):** E2 has since completed on the 124-case panel. Final results are in `results/locking_paper_redesign_20260920_161812/e2_analysis/summary.md`.
- TraceLock (T-adaptive 0.95): 43 of 124 correct against 45 for the moving-window baseline, with speedups of +92.45% on MATH and +469.9% on HumanEval.
- Confidence threshold (C-adaptive 0.956667), matched to TraceLock's call count: kept all 45.
- The aligned rerun at the standard block-32 setting is in `docs/aligned_protocol_2026-09-29.md`.

## Verification and evidence

Fresh reporting checks matched the online raw-record SHA-256 and all 2,604 records to the saved verification; recomputed all seven aggregate accuracy/speedup rows; and checked all nine offline dataset/threshold count ratios. Existing full-run verification confirms all 124 offline replays match the accepted no-lock outputs and calls. This reporting pass did not rerun the model or modify experiment code.

The thresholds and fixed sample panel make these exploratory results. BF16 sparse computation is approximate, including at a zero KL cutoff. Last-layer feature KL is a chosen proxy, not a proof that every cached layer is stable.

- [Combined representation report](../results/representation_kl_last_hidden_20260921_203301/representation_kl_results.md)
- [Online aggregate CSV](../results/representation_kl_last_hidden_20260921_203301/analysis/online_representation_kl_aggregate.csv)
- [Offline threshold CSV](../results/representation_kl_last_hidden_20260921_203301/offline/analysis/offline_audit_thresholds.csv)
- [Frozen online settings](../results/representation_kl_last_hidden_20260921_203301/orchestration/e1_policy_config.json)
- [Combined verification](../results/representation_kl_last_hidden_20260921_203301/validation/final_combined_verification.json)
- [TraceLock training completion](../results/locking_paper_redesign_20260920_161812/e2_train/full/success.json)
- [TraceLock development settings](../results/locking_paper_redesign_20260920_161812/orchestration/e2_selection_policy_config.json)
- [TraceLock development run configuration](../results/locking_paper_redesign_20260920_161812/e2_selection/selection/attempt_01/resolved_config.json)
