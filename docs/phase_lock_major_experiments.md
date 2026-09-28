# PhaseLock major experiment handoff

This handoff contains the three requested PhaseLock policies, exact benchmark
code, compact/full result records, baseline implementations, and the most useful
negative controls. It does **not** include the unfinished accuracy-speed v2 run.

## Results at a glance

| Policy | Evaluation | Accuracy | Speed result | Main conclusion |
|---|---:|---:|---:|---|
| `open_confidence` | 60 examples | 36.67% | 19.24% estimated online gain vs dense, 95% CI [13.79%, 24.76%] | Preserved dense correctness status on all 60, but was slower and less accurate than the TraceLock-LLaDA adaptation. |
| `capped_confidence` | 60 examples | 36.67% | 1.36% measured online gain vs dense, 95% CI [0.87%, 1.82%] | Strong conservative semantic-lock control: 0 regressions, 0 improvements, 59/60 exact; too conservative for speed mode. |
| H2 `guard90_delay1` | 24 development examples | 41.67% | 1.192x vs SureLock and 1.597x vs TraceLock | Best H2 development candidate: matched TraceLock aggregate accuracy while running faster than both baselines. Not a held-out or final robustness result. |
| Direct commitment-safety predictor | 60 examples | 36.67% | 52.22% measured online gain vs dense, 95% CI [49.73%, 54.38%] | Important negative/upper-bound result: 1 regression and 1 improvement with 45% answer changes; single-token intervention training did not make grouped deployment safe. |

All rows use LLaDA-8B-Instruct, BF16, temperature zero, batch size four,
128 generated tokens and at most 128 updates. The 60-example runs use 20 each
from GSM8K, MATH500 and HumanEval. The H2 screen uses eight per dataset and is
explicitly development-only. MATH500 in the H2 result was re-scored with pinned
Math-Verify 0.9.0; earlier 60-example tables retain their original repository
scorer and should not be silently mixed with the symbolic H2 metric.

## Exact algorithms

### Open confidence

For the current 32-token soft-crop window, commit every active token whose
TraceLock future-stability probability exceeds 0.95. If none pass, commit the
single token with maximum LLaDA posterior confidence. Committed identities keep
updating their representations. The downstream representation gate requires
maximum cosine drift across layers 8/16/24 <= 0.002, layer-24 influence <= 0.5,
patience six, semantic age >= four and a 4% per-update cap. Packed execution is
enabled only when the independent profitability gate predicts at least 3% gain.

The reported 19.24% is an additive estimate: exact fused replay plus measured
controller overhead. SureLock and TraceLock comparison times were measured
online. Read `phaselock_trace_assisted_results.md` before making speed claims.

### Capped confidence

Retain the ordinary confidence-based transfer schedule. During only the first
two updates of each block, permit the transfer count to rise to
`max(base_k, ceil(0.06 * remaining_active))`; only extra candidates with LLaDA
confidence >= 0.95 are eligible. This is Phase 1/A1 only: representations remain
dense and live. Its 1.36% gain is measured end-to-end online.

### H2 confidence-0.90 plus one-update delay

Phase 1 first applies the full-Sure rule to masked positions in the current
block: candidate-relative top-20% confidence and posterior KL <= 0.005. Optional
semantic commitments must additionally have confidence >= 0.90; if none pass,
commit the maximum-confidence allowed token to guarantee progress.

Phase 2 retains Sure-style prompt-reference reuse but prevents a newly committed
generated token from freezing the stale representation computed while it was
still masked. It becomes eligible only after one subsequent forward pass has
processed its committed identity. Phase 3 immediately uses the exact bucketed
row-removal executor for frozen query/FFN rows while retaining dense attention
over all K/V references.

This result is from only 24 reused development examples. Aggregate accuracy
matched TraceLock, but per-dataset tradeoffs remained: H2 scored 5/8 GSM8K,
4/8 symbolic MATH500 and 1/8 HumanEval, versus TraceLock's 6/8, 2/8 and 2/8.
It is a candidate to advance, not evidence of general superiority.

## Where to look

- `scripts/benchmark_phaselock_trace_semantic.py` and
  `scripts/analyze_phaselock_trace_semantic.py`: open-confidence controller and
  analysis.
- `scripts/benchmark_commitment_safety_policy.py` and
  `scripts/analyze_matched_semantic_confirmation.py`: capped confidence and
  direct-safety comparison.
- `scripts/benchmark_phaselock_online_screen.py` and
  `scripts/benchmark_phaselock_sure_hybrid.py`: live H2 controller/executor.
- `scripts/benchmark_surelock_vs_phaselock.py` and
  `scripts/benchmark_tracelock_vs_phaselock.py`: matched baseline implementations.
- `src/regret_remasking/phase_lock_sparse.py`: bucketed reference freezing and
  packed row-removal runtime.
- `configs/phaselock_h2_controller_screen_v1.json`: exact H2 screen policy pool.
- `results/phase_lock_major_experiments/`: full result records, timing rows,
  run configurations, frozen plans where available, and aggregate summaries.

The earlier representation-lock and compute-lock studies are also retained on
this branch under `results/representation_lock_v2/` and
`results/phase_lock_compute_v1/`. They establish the complementary insight that
safe reference freezing can exist without profitable row removal, so the three
PhaseLock decisions should remain separate.

## Reproduction cautions

- SureLock is a paper-spec transfer implementation because official code was not
  available at the time of the experiment.
- TraceLock is a trained LLaDA transfer adaptation because the released project
  lacked the LLaDA implementation/checkpoint. Do not label it an official
  reproduction.
- TraceLock training incurred at least 9.17 recorded GPU-hours; the true cost is
  higher because one resumed trace-generation segment lacks its original cost.
- Existing model/controller checkpoints are not committed. Their paths and hashes
  are recorded in run configurations; regenerate or provide them separately.
- Raw model generations are included for auditability. No hidden-state dumps,
  model weights, credentials or scheduler logs are included.
