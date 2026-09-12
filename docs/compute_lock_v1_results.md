# Compute lock v1: fixed-plan packed-row profitability

## Outcome

The packed A3 backend is output-safe for the tested
`robust_delayed_one_update` representation plans, but its wall-clock value is
regime-dependent. It is **not profitable at 64 denoising steps / 64 generated
tokens** and is only modestly profitable at **128 steps / 128 generated
tokens**.

This is direct evidence for the PhaseLock distinction between reference safety
and compute value: every A3 replay was reference-safe and output-equivalent,
while the runtime gain changed sign with the workload.

## Matched experiment

For every example and semantic mode, the benchmark:

1. generated the exact semantic commitment plan in dense A1;
2. replayed that plan in a second dense run to generate a fixed
   `robust_delayed_one_update` reference plan;
3. asserted that the dense planner output exactly matched A1;
4. replayed the same semantic and reference plans in dense-reference A2 and
   packed-row A3;
5. warmed both backends once, then measured five A2/A3 pairs while alternating
   execution order;
6. synchronized CUDA and timed model decode through the final token copy while
   excluding model load, plan generation, tokenizer decoding, and task scoring.

The runtime metric is

`g_compute = 1 - latency_A3 / latency_A2`,

so positive values mean packed A3 is faster. Confidence intervals resample
examples, not repeated timings. A profitability pass requires exact A3=A2
outputs and a 95% bootstrap lower bound above zero.

The confirmation slices were the same slices used by representation lock v2:

| Dataset | Slice | Schedule | Examples | Timed pairs | Job |
|---|---|---:|---:|---:|---:|
| GSM8K | indices 300--309 | 64 steps / 64 tokens | 10 | 100 | `189345` |
| MATH500 | indices 400--409 | 64 steps / 64 tokens | 10 | 100 | `189360` |
| HumanEval | tasks 20--24 | 128 steps / 128 tokens | 5 | 50 | `189370` |

All jobs ran in bf16 on one clean NVIDIA RTX 5000 Ada Generation GPU with
PyTorch 2.5.1+cu121. The Slurm runner fails before model load if the allocated
GPU already has more than 512 MiB in use.

## Results

All 250 timed comparisons satisfied A2=A1 and A3=A2 exactly. There were zero
generation, answer, or correctness changes.

| Dataset | Semantic mode | Total rows removed | Median A2 (s) | Median A3 (s) | Mean gain | Example-bootstrap 95% CI | Profitable? |
|---|---|---:|---:|---:|---:|---:|---:|
| GSM8K | safe `c095/s2` | 19.99% | 3.410 | 3.527 | -3.13% | [-3.45%, -2.70%] | no |
| GSM8K | aggressive `c090/age4/s8` | 20.05% | 3.343 | 3.452 | -3.20% | [-3.59%, -2.69%] | no |
| MATH500 | safe `c095/s2` | 21.77% | 3.356 | 3.471 | -3.15% | [-3.44%, -2.72%] | no |
| MATH500 | aggressive `c090/age4/s8` | 22.02% | 3.114 | 3.224 | -3.20% | [-3.49%, -2.76%] | no |
| HumanEval | safe `c095/s2` | 25.04% | 7.711 | 7.654 | +0.96% | [+0.33%, +1.98%] | yes |
| HumanEval | aggressive `c090/age4/s8` | 25.35% | 7.150 | 7.096 | +1.01% | [+0.36%, +2.04%] | yes |

Execution-order means agree closely: the largest difference between
`A2_then_A3` and `A3_then_A2` was 0.30 percentage points. The sign of every
dataset/mode result was unchanged by order.

Pooling unlike schedules is useful only as an overall workload-mix check. It
is negative: -2.32% for the safe mode and -2.36% for the aggressive mode. This
pooled number must not hide the positive 128-step regime.

## Algorithm decision

`robust_delayed_one_update` remains the representation-lock default. Packed
row removal should **not** be unconditional:

- keep dense compute for the tested 64/64 regime;
- treat packed A3 at 128/128 with roughly 25% projected total-row removal as a
  promising but still provisional eligible regime;
- make the compute phase use a workload gate based on active rows, total
  sequence length, and remaining denoising steps, not the dataset name;
- switch from dense to packed only when predicted remaining savings exceed
  packing, gather/scatter, and dispatch overhead by a safety margin.

The current backend packs active query/projection/FFN rows and keeps a
persistent full-length K/V workspace, but still invokes dense attention. These
results characterize this implementation; they do not rule out larger gains
from a fused sparse kernel or batch-aware packing.

The next controlled experiment should cross the schedule within datasets
(64/64 and 128/128 on the same task slices) and sweep the online switch point.
That is required before promoting the observed 128-step boundary into a fixed
PhaseLock compute policy.

## Artifacts

- `scripts/benchmark_phase_lock_compute.py`: paired benchmark runner.
- `scripts/analyze_phase_lock_compute.py`: integrity checks and aggregation.
- `cluster/run_phase_lock_compute_benchmark.slurm`: fail-closed GPU runner.
- `results/phase_lock_compute_v1/`: run configs, raw timing pairs, summaries,
  and pooled validation.

