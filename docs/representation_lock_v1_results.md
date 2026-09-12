# Representation lock v1 experiment log

## Algorithm under test

The experiment fixes the two semantic modes in
`configs/semantic_lock_modes.json` and replays each mode's exact A1 commitment
plan in every A2 arm. The proposed representation policy is `capped_drift`:
low cosine drift plus patience and post-commit age determine eligibility, then
the lowest-drift eligible references are frozen under a 0.06 per-step cap.

The online observation layer controls the gate only. Once selected, a position's
input/output hidden rows are frozen at every transformer block, making its
derived attention reference fixed while A2 still computes dense active rows.

## Cluster provenance

All valid jobs used LLaDA-8B-Instruct, bf16, temperature 0, seed 17, and one
RTX 5000 Ada GPU allocated by Slurm on `gpu-01` physical index 4.

The first submission, job `188149`, is invalid and excluded. It stopped before
the first A2 arm because the cluster had an older `run_four_arm_audit.py` whose
`ReferenceRuntime` signature did not match the PhaseLock runner. The validated
backend was synced and all reruns used fresh output directories.

Valid coarse smoke jobs:

| Dataset | Job | Slice | Generation length | A1->A2 pairs |
|---|---:|---:|---:|---:|
| GSM8K | `188165` | offset 100, 1 example | 64 | 54 |
| MATH500 | `188197` | offset 300, 1 example | 64 | 54 |
| HumanEval | `188227` | offset 0, 1 task | 128 | 54 |

Valid focused-development jobs:

| Dataset | Job | Slice | Generation length | A1->A2 pairs |
|---|---:|---:|---:|---:|
| GSM8K | `188267` | offset 100, 5 examples | 64 | 140 |
| MATH500 | `188276` | offset 300, 5 examples | 64 | 140 |
| HumanEval | `188277` | offset 0, 5 tasks | 128 | 140 |

HumanEval uses isolated single-sample execution scoring. It is a development
screen, not an official pass@k result.

## Coarse smoke result

All 162 matched comparisons were exact A1->A2 sequence matches. There were no
answer changes, correctness changes, or A1-correct to A2-incorrect cases.

The policy axes changed useful lock lifetime in the expected direction. Pooled
across the two semantic modes and three examples, mean reference-opportunity
fractions were:

- layer 8: 0.395; layer 16: 0.354; layer 24: 0.319;
- threshold 0.002: 0.309; 0.01: 0.360; 0.05: 0.399;
- patience 1: 0.382; patience 2: 0.361; patience 4: 0.325.

The best coarse arm was layer 8, threshold 0.05, patience 1, minimum age 1.
Its mean reference opportunity was 0.428 under the safe semantic mode and
0.419 under the aggressive semantic mode in the larger development screen.

## Focused development result

The shortlist contains 12 capped-drift variants plus uncapped low-drift and
always-freeze-after-one-update stress controls. Across 15 examples and both
semantic modes:

- 420/420 A1->A2 comparisons were exact sequence matches;
- normalized-answer/code-generation change rate was 0/420;
- correctness change rate was 0/420;
- A1-correct to A2-incorrect harm was 0/420.

Pooled time-weighted reference opportunity:

| Semantic mode | Best capped drift | Uncapped drift | Always freeze |
|---|---:|---:|---:|
| safe `c095/s2` | 0.428 | 0.467 | 0.478 |
| aggressive `c090/age4/s8` | 0.419 | 0.470 | 0.481 |

The best capped arm was `cdrift_l08_t050_p1_a1_f006` for both semantic modes.
Its dense A2 implementation was about 13.8% slower than A1 because the audit
adds full-layer freeze hooks without removing row compute. That is not an A3
compute result and is not treated as a speed claim.

## Interpretation

This establishes that capped drift is operational, produces substantial
time-weighted reference coverage, and preserved A1 exactly on the development
screen. It does **not** establish that the drift gate or cap is necessary:
uncapped and always-freeze controls also preserved A1 exactly and exposed more
reference opportunity on these examples.

Therefore representation v1 is a candidate, not a frozen final policy. The
next discriminating experiment should freeze high-drift matched references or
use examples where the teammate's causal audit observed representation-freeze
harm. If high-drift interventions fail while low-drift matched controls remain
safe, retain capped drift. If always-freeze remains equally safe on a larger
untouched set, simplify the representation phase to a one-update delay and
treat drift primarily as a diagnostic rather than claiming it is required by
the algorithm.
