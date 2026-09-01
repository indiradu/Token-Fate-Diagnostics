# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A research repository studying **premature token commitment in diffusion language
models** (`GSAI-ML/LLaDA-8B-Instruct`, confidence decoding). The unit of analysis
is *token fate*: whether the token proposed at an intermediate denoising step
equals the final decoded token. Read `README.md` for the published findings,
`docs/causal_robustness_protocol.md` for split rules and control definitions, and
`research_idea.md` + `smoke_test.md` for the current direction.

The branch `pivot_efficient_compute` extends this to **PhaseLock**: three
separate decisions — semantic lock (token identity fixed), reference freeze
(hidden/K-V state cached as what other positions see), and row removal (row
dropped from future compute). The first two are causal risk gates; the third is
a systems *utility* gate. Four labels are used consistently across
`research_idea.md` and `smoke_test.md` — do not invent new ones:

- **A:** semantic commitment does not imply representation settlement
- **B:** commitment safety does not imply reference-freeze safety
- **C:** representation drift predicts reference-freeze harm beyond matched controls
- **U:** reference-freeze safety does not imply compute profitability

There is also a **Level 0** (the same meaning can be carried by different
tokens) that is named in `research_idea.md` but deliberately not tested, and an
**auxiliary** commitment-timing question (is committing *earlier than the
decoder would* harmful?) that is measured but is not a premise. Do not promote
either into the A/B/C/U set.

`research_idea.md` is reframed often and `smoke_test.md` may lag behind it. Read
both before restating any claim, and treat `research_idea.md` as current. Note
that `A` was realigned on 2026-09-01 back to its original `smoke_test.md`
meaning (representation settlement); results written before that date may use
`A` for the commitment-timing question instead — see
`results/README.md` and the artifact for the corrected mapping.

## Commands

The package is **not installed**; only `run_settlement_fate_audit.py` and the
tests insert `src/` into `sys.path` themselves. Every other script needs
`PYTHONPATH=src`.

```bash
# Local checks, no GPU / no LLaDA download
python3 -m compileall src scripts tests
python3 -m pytest -q tests                     # pyproject sets pythonpath=["src"]
python3 tests/test_features.py                 # tests also run as plain scripts
python3 tests/test_counterfactual_selection.py

# GPU: settlement-fate audit (current pivot entry point)
python3 scripts/run_settlement_fate_audit.py --dataset math500 --split test \
  --limit 10 --offset 105 --output-dir results/<run> \
  --max-interventions-per-example 3 --empty-cache
# Fast shape check on the same script (tiny decode):
#   --dataset countdown --limit 1 --steps 4 --gen-length 8 --block-length 4 --layers 32

# GPU: trace + hidden-state dataset, run once per split
PYTHONPATH=src python3 scripts/run_token_fate_probe_dataset.py --dataset gsm8k \
  --split train --limit 100 --output-dir results/<run>/train
PYTHONPATH=src python3 scripts/run_token_fate_probe_dataset.py --dataset gsm8k \
  --split test --offset 100 --limit 500 --output-dir results/<run>/eval

# CPU: train / analyze over the CSVs produced above
PYTHONPATH=src python3 scripts/train_token_fate_probes.py \
  --train-dir results/<run>/train --eval-dir results/<run>/eval --output-dir results/<run>/probes
PYTHONPATH=src python3 scripts/analyze_token_fate_superiority.py \
  --run gsm8k=results/<run> --output-dir results/<analysis>
```

Decode-shape constraints are runtime errors, not warnings: `gen_length` must be
divisible by `block_length`, and `steps` divisible by `gen_length / block_length`.

Model loading honors `LLADA_DEVICE_MAP` (`1` = pin to `--device`, `auto` = accelerate
offload), plus `LLADA_GPU_MAX_MEMORY`, `LLADA_CPU_MAX_MEMORY`, `LLADA_OFFLOAD_FOLDER`.
Use `--jsonl-path` (fields `question`/`problem`/`prompt` and `answer`/`solution`/
`gold_answer`) for local data, e.g. `data/nonmath_reasoning_smoke.jsonl`.

## Architecture

Two script families, connected only by CSV artifacts on disk:

1. **GPU runners** — `run_settlement_fate_audit.py`, `run_token_fate_probe_dataset.py`,
   `run_logit_lens_token_fate.py`, `run_counterfactual_commit.py`, `run_regret_pilot.py`.
   They load LLaDA, replay a decode, and emit `metadata.csv` / `*.csv` / `summary.json`.
2. **CPU analyzers** — `train_*.py`, `analyze_*.py`, `summarize_*.py`. They only read
   those artifacts; they never touch the model. All model fitting is sklearn.

### The trace row is the data contract

`FEATURE_NAMES` in `src/regret_remasking/__init__.py` is the canonical feature list
(`confidence, entropy, margin, kl, top1_flip, runlength, t_frac, local_mask_ratio,
context_volatility`). Every trace row also carries `example_id`, `global_step`,
`block`, `step_in_block`, `block_t_frac`, `position`, `relative_position`,
`top_token`, `score`, `selected`, and — filled in **after** the decode finishes —
`final_token` and `trace_regret = int(top_token != final_token)`. Adding a feature
means updating `FEATURE_NAMES`, the `feature_map` in each decode loop, and any
saved `.joblib` bundle's `features` list (`RegretScorer` reads that list, not the
global one).

Note `t_frac` is global-step fraction while `block_t_frac` is within-block; the
"early" slice used throughout the protocol is `block_t_frac <= 0.5`, and it is
emitted by the probe/audit scripts, *not* by `llada_trace.decode_batch`.

### Duplicated decode loops — keep them in sync

`llada_trace.decode_batch` is the reference implementation, but
`run_token_fate_probe_dataset.py`, `run_logit_lens_token_fate.py`,
`run_settlement_fate_audit.py`, and `run_counterfactual_commit.py` each reimplement
the denoising loop inline (they need hidden states, hooks, or forced commits mid-loop)
while importing the primitives — `add_gumbel_noise`, `get_num_transfer_tokens`,
`prepare_prompts`, `infer_mask_token_id`. A change to scoring, transfer-count, or
block iteration must be mirrored across all of them or runs stop being comparable.

### Run-directory convention

Analysis scripts take `--run LABEL=RUN_DIR` and expect `RUN_DIR/train/metadata.csv`
and `RUN_DIR/eval/metadata.csv`. That layout comes from invoking
`run_token_fate_probe_dataset.py` twice with `--output-dir RUN_DIR/train` and
`--output-dir RUN_DIR/eval`; hidden states land beside each as
`hidden_layer_<n>.npy` (random-projected to `--projection-dim`, so they are not
raw activations). Train and eval must use disjoint examples — the original MATH500
run was discarded for using offset 0 on both.

## Experimental discipline

These are substantive constraints, not style preferences:

- **Selectors are fit on training slices only**, and thresholds (e.g. the
  confidence quantile) are computed from training rows, then frozen. Future
  `trace_regret` labels must never be used to pick intervention targets or controls.
- **Strict controls** in `run_counterfactual_commit.py` are same-trajectory,
  same-global-step, confidence-matched (|Δconf| ≤ 0.05, |Δposition| ≤ 8), and the
  primary endpoint **excludes the forced position**. `tests/test_counterfactual_selection.py`
  pins exactly these invariants — if a change makes it fail, the change is weakening
  the causal design.
- **Bootstrap over examples/pairs, never over dense token rows.**
- The strict-intervention **null result is deliberately preserved** (see README and
  `docs/causal_robustness_results.md`). Do not restate prediction results as causal
  ones or soften the reported intervals.

## Artifacts and git

`.gitignore` excludes `results/**` except `results/README.md`, plus `paper/`,
`cluster/`, and superseded `docs/` notes. Compact reported artifacts are tracked by
**explicit force-add**:

```bash
git add -f results/<dir>/summary.md results/<dir>/summary.json   # etc.
```

Only commit small aggregate tables/summaries. Raw traces, `metadata.csv`,
`hidden_layer_*.npy`, `.joblib` models, and full generations stay local — they can
contain complete model outputs. When adding a reported artifact, also add its row to
`results/README.md`. `.gitattributes` disables whitespace checks on generated CSVs
because model text can end in spaces.
