# Representation lock v2: fixed-plan causal development

## Outcome

The recommended PhaseLock representation policy is now
`robust_delayed_one_update`: after a token is semantically committed, keep its
representation live for one complete model update, then freeze it. The
immediate same-step variant is retained only as an aggressive frontier mode.

One-step cosine drift is **not** part of the recommended v2 gate. In these
experiments it reduced reference opportunity but did not identify a safer
intervention region than the one-update delay.

This is a robust development result, not a proof of universal safety. It must
be revalidated for a different model, decoding schedule, freeze backend, or
substantially longer generation regime.

## Why v1 needed correction

Representation v1 selected later locks online inside A2. Earlier A2 freezes
could therefore change hidden states used to choose later locks. That made the
policy plan endogenous to the intervention.

V2 removes that leakage:

1. Generate and save the exact semantic commitment plan in dense A1.
2. Replay that semantic plan in another dense run and generate the complete
   reference plan without freezing any model state.
3. Assert that the dense planner's token sequence is exactly equal to A1.
4. Replay the fixed reference plan in A2. A2 is not allowed to adapt its plan.
5. Reject a replay that freezes an uncommitted or already-frozen position.

The runner defaults to `--reference-plan-source dense_a1`. The former online
A2 mode remains available only for backward-compatible diagnostics.

## Matched low-drift versus high-drift intervention

For each low-drift policy, v2 constructs a high-drift stress control from the
same dense A1 trajectory. The control receives the low-drift plan's exact lock
count at every global denoising step, but ranks the currently available,
age-matched references by largest rather than smallest drift.

The tested low-drift gates were:

- layer 8, threshold 0.05, patience 1, minimum age 1;
- layer 8, threshold 0.02, patience 2, minimum age 1;
- layer 8, threshold 0.005, patience 4, minimum age 1;
- layer 24, threshold 0.05, patience 1, minimum age 1.

Each policy used a 0.06 per-step cap. An uncapped one-update `always` arm was
included as a stress control. The development examples were selected before
the new A2 outcomes: five GSM8K and five MATH500 examples with large future
representation paths in the existing diagnostics, plus five HumanEval tasks.

Across 15 development examples, two semantic modes, and nine representation
arms there were 270 A1-to-A2 comparisons. All 270 were exact sequence matches,
with zero answer changes and zero correctness changes. The 120 matched
low/high pairs had exact per-step lock-count agreement. Their selected
`(step, position)` event overlap was 0.000--0.0035, so the controls selected
genuinely different interventions. Mean high-minus-low source drift ranged
from 0.056 to 0.068 at layer 8 and from 0.210 to 0.367 at layer 24. Despite
that separation, neither side caused observed output harm.

Valid development jobs:

| Dataset | Examples | Job | Result directory |
|---|---:|---:|---|
| GSM8K hard set | 5 | `188518` | `results/representation_lock_v2_targeted_gsm8k_20260912` |
| MATH500 hard set | 5 | `188562` | `results/representation_lock_v2_targeted_math500_20260912` |
| HumanEval development | 5 | `188619` | `results/representation_lock_v2_humaneval_20260912` |

Job `188504` was a successful one-example implementation smoke and is not
counted separately because its example also appears in the GSM8K hard set.

## Untouched confirmation

The policy decision was then tested on examples not used for gate selection:

- GSM8K indices 300--309;
- MATH500 indices 400--409;
- HumanEval tasks 20--24.

The confirmation compared capped drift, one-update delay, and immediate
same-step freeze under both saved semantic modes. All 150 A1-to-A2 comparisons
were exact sequence matches. Answer-change, correctness-change, and
A1-correct-to-A2-incorrect rates were all zero.

Pooled across the 25 confirmation examples:

| Semantic mode | Policy | Exact A1=A2 | Reference opportunity |
|---|---|---:|---:|
| safe `c095/s2` | capped drift | 1.000 | 0.452 |
| safe `c095/s2` | one-update delay | 1.000 | 0.480 |
| safe `c095/s2` | immediate stress | 1.000 | 0.494 |
| aggressive `c090/age4/s8` | capped drift | 1.000 | 0.431 |
| aggressive `c090/age4/s8` | one-update delay | 1.000 | 0.484 |
| aggressive `c090/age4/s8` | immediate stress | 1.000 | 0.498 |

Valid confirmation jobs:

| Dataset | Examples | Job | Result directory |
|---|---:|---:|---|
| GSM8K | 10 | `188695` | `results/representation_lock_v2_confirm_gsm8k_20260912` |
| MATH500 | 10 | `188720` | `results/representation_lock_v2_confirm_math500_20260912` |
| HumanEval | 5 | `188744` | `results/representation_lock_v2_confirm_humaneval_20260912` |

HumanEval was scored as an isolated single-sample development run, not as an
official pass@k evaluation.

Dense A2 was about 14% slower than A1 in this instrumented implementation.
That is expected because A2 installs full-layer freeze hooks while still
computing every row. Reference opportunity is potential A3 work removal, not a
wall-clock speed claim. Compute profitability must be measured separately with
the packed row-sparse backend.

## Algorithm decision

`robust_delayed_one_update` is selected because it:

- passed every fixed-plan development and untouched confirmation comparison;
- removes threshold and layer calibration from the online decision;
- preserves substantially more useful lock lifetime than capped drift;
- gives up only about 1.4--1.5 opportunity points versus immediate freezing;
- guarantees one observed post-commit update before making the reference
  irreversible.

`aggressive_immediate_same_step` is saved for the speed-quality frontier but
is not the default. Its observed safety is encouraging, yet same-step locking
collapses semantic and representation decisions and would weaken the paper's
staged safety story.

## Scientific interpretation

These results do not show that representation and semantic convergence are
the same signal. The earlier observational diagnostics still show substantial
post-semantic representation movement. They show something narrower and
important: for the tested LLaDA-8B A2 backend, that movement did not translate
into output sensitivity after semantic commitment.

Therefore the current evidence does **not** support claiming that one-step
representation drift is required for safe reference freezing, nor that higher
drift causes more A2 answer harm in this regime. Premise B/C remains a research
hypothesis to test under broader or stronger interventions, not a premise that
the current algorithm should hard-code.

The next PhaseLock stage should use `robust_delayed_one_update` to test A2
versus A3 output equivalence and real wall-clock value of packed row removal.
That compute experiment must replay the exact same semantic and reference
plans and report packing/dispatch overhead separately from model compute.

## Artifacts

- `configs/representation_lock_modes.json`: saved default and aggressive modes.
- `configs/representation_lock_v2_matched.json`: causal low/high development grid.
- `configs/representation_lock_v2_confirmation.json`: untouched confirmation grid.
- `scripts/analyze_matched_representation_lock.py`: count/timing and drift-separation audit.
- `results/representation_lock_v2/`: compact development and confirmation summaries.
