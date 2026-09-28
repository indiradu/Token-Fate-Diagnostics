# Controller cost ledger (2026-09-27 audit)

Recorded successful-loop durations, not complete Slurm billed GPU-hours:

| TraceLock component | Seconds | Evidence under `results/` |
|---|---:|---|
| Alpaca 2,000 traces | 6,817.2005 | `tracelock_llada_8000_paper_mix_traces_20260914/summary_alpaca_000000.json` |
| GSM8K 2,000 traces | 11,000.1389 | same directory, `summary_gsm8k_000000.json` |
| Natural Questions 2,000 traces | 5,950.4747 | same directory, `summary_natural_questions_000000.json` |
| KodCode resume segment (500 new; 1,500 reused) | 7,260.8875 | same directory, `summary_kodcode_000000.json` |
| Controller fit, 10,000 updates | 1,973.9992 | `tracelock_llada_8000_paper_mix_controller_20260914/summary.json` |

Sum of these recorded segments: 33,002.7008 seconds = 9.1674 hours.
This is a LOWER BOUND on recorded sequential work, not the total TraceLock training
cost. It omits the original 1,500 KodCode traces, failed/restarted work, model-load
and allocation overhead. The resumed KodCode summary must not be counted as all
2,000 traces' cost. The training manifest confirms 8,000 unique prompts (2,000 per
source), 271,776 train step samples and 32,224 validation step samples.

Full cost remains unknown until missing logs/accounting are recovered. Never
replace unknown historical costs with zero. A PhaseLock policy using Trace features
inherits the appropriate trace/controller cost even if inference uses cached assets.

The current H2 heuristic pool does not train a learned controller. Its GPU screening
and calibration budget must still be reported as development cost, separately from
serving latency. Direct-safety interventions, classifier fitting and calibration
need their own ledger; no new direct-safety model has been trained in this revision.

For a fixed serving workload, report inference-only latency plus an amortized view:
`T_total(N) = T_offline + N * T_serving_per_example`, with batch amortization explicit.
Break-even against another trained method depends on the DIFFERENCE in offline costs
and serving costs; it is not just TraceLock's training time divided by PhaseLock's
latency. Comparable task accuracy is a prerequisite for efficiency claims.
