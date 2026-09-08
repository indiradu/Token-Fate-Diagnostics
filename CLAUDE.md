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
separate decisions — token-identity lock (also called semantic lock in older
notes), reference freeze (hidden/K-V state cached as what other positions see),
and row removal (row dropped from future compute). The first two are causal risk
gates; the third is a systems *utility* gate. `research_idea.md` is the current
label authority:

- **A:** token-identity stability does not imply reference-state settlement
- **B:** commitment safety does not imply reference-freeze safety
- **C:** reference-state drift predicts reference-freeze harm beyond matched controls
- **U:** reference-freeze safety does not imply compute profitability

There is also a **Level 0** (the same meaning can be carried by different
tokens) that is named in `research_idea.md` but deliberately not tested, and an
**auxiliary** commitment-timing question (is committing *earlier than the
decoder would* harmful?) that is measured but is not a premise. Do not promote
either into the A/B/C/U set.

Terminology guardrails from `discuss.md` are now incorporated into
`research_idea.md`: same-token identity is token/prediction stability, not
semantic convergence in the meaning-level sense. Premise A is not the trivial
claim that hidden floats change; the drift has to be above numerical/background
variation and relevant to the cached reference state active tokens see. Premise
B separates `YC` (token committed while representation is recomputed) from `YF`
(token committed and reference frozen). Premise C is a prediction-of-causal-harm
claim; raw drift alone may be weak, and `K/V drift * attention exposure` is the
preferred first-order risk proxy.

`research_idea.md` is reframed often and `smoke_test.md` may lag behind it. Read
both before restating any claim, and treat `research_idea.md` as current. Note
that `A` was realigned on 2026-09-01 back to its original `smoke_test.md`
meaning (reference-state settlement); results written before that date may use
`A` for the commitment-timing question instead — see
`results/README.md` and the artifact for the corrected mapping.

## Commands

The package is **not installed**. Scripts that import `regret_remasking` need
`PYTHONPATH=src`, except `run_settlement_fate_audit.py` and all four tests,
which insert `src/` into `sys.path` themselves. `analyze_settlement_pairs.py`
and `simulate_compute_utility.py` are standalone (numpy/pandas only) and need
no path setup.

```bash
# Local checks, no GPU / no LLaDA download
python3 -m compileall src scripts tests
python3 -m pytest -q tests                     # pyproject sets pythonpath=["src"]
python3 tests/test_features.py                 # tests also run as plain scripts
python3 tests/test_counterfactual_selection.py # also test_drift_update.py,
                                               # test_settlement_commit_selection.py

# GPU: settlement-fate audit (current pivot entry point). --intervention-mode
# picks which gate the run measures: clamp = reference freeze (B/C),
# commit = early forced lock (auxiliary timing). Always --exclude-eos-eot.
python3 scripts/run_settlement_fate_audit.py --dataset math500 --split test \
  --limit 10 --offset 105 --output-dir results/<run> \
  --intervention-mode clamp --exclude-eos-eot \
  --max-interventions-per-example 3 --empty-cache
# Long generations need --skip-drift-time-rows for throughput, which forfeits
# the U proxy (see Experimental discipline):
#   --gen-length 256 --block-length 32 --steps 256 --skip-drift-time-rows
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

# CPU: paired deltas for the audit runs above (--run is repeatable; pooling
# happens across runs of the SAME mode only)
python3 scripts/analyze_settlement_pairs.py \
  --run g64=results/<clamp_run_a> --run l5g256=results/<clamp_run_b> \
  --output-dir results/<pairs_analysis>

# CPU: premise-U compute proxy, needs drift_time_rows.csv from a clamp run;
# --timing-run adds the semantic-vs-drift settlement panel from a commit run
python3 scripts/simulate_compute_utility.py --run-dir results/<clamp_run> \
  --timing-run results/<commit_run> --output-dir results/<utility_proxy>

# CPU: lock-admission audit of prior methods (docs/lock_admission_audit.md).
# Phase 0 needs nothing new; Phase 1 needs post_commit_posterior.csv.
python3 scripts/analyze_lock_admission_bound.py \
  --run a100x3=results/<run_a> --run c100x3=results/<run_c> \
  --output-dir results/<phase0>
python3 scripts/analyze_lock_predicate_audit.py \
  --train-run train=results/<g64_train> --run eval=results/<g64_eval> \
  --output-dir results/<phase1>
```

**GPU jobs need `#SBATCH --propagate=NONE` (found 2026-09-01).** This
workspace's interactive session carries `RLIMIT_AS` (`ulimit -v`) of **16 GiB**,
soft *and* hard, and the cluster's `PropagateResourceLimitsExcept` covers only
`MEMLOCK,STACK,NOFILE` — so every `sbatch` inherits that 16 GiB address-space
cap and LLaDA dies during model construction no matter what `--mem` says. The
error is misleading: `DefaultCPUAllocator: can't allocate memory ... 100663296
bytes` at the same `nn.Linear` every time, identical at `--mem=32G`, `48G`, and
`80G`. `--mem=48G` is the right request; the cap is the bug. The same cap
applies to local analysis in this session, so a mysterious ENOMEM in a big
pandas job has the same cause.

Two related environment facts. The installed `transformers` is **5.12.1**, past
this repo's `<5` pin, and the base env has **no `accelerate`** — so
`LLADA_DEVICE_MAP=1` raises a hard `ValueError`. The repo `.venv` is
`--system-site-packages` plus accelerate; GPU jobs use `.venv/bin/python`.
LLaDA's remote code also hardcodes `model_config.init_device = "cpu"` in
`LLaDAModelLM.__init__`, so construction is always on CPU (in bf16, ~15G) and
neither `low_cpu_mem_usage`, `LLADA_DEVICE_MAP=1`, nor passing
`init_device="meta"` changes that.

Decode-shape constraints are runtime errors, not warnings: `gen_length` must be
divisible by `block_length`, and `steps` divisible by `gen_length / block_length`.

Model loading honors `LLADA_DEVICE_MAP` (`1` = pin to `--device`, `auto` = accelerate
offload), plus `LLADA_GPU_MAX_MEMORY`, `LLADA_CPU_MAX_MEMORY`, `LLADA_OFFLOAD_FOLDER`.
Use `--jsonl-path` (fields `question`/`problem`/`prompt` and `answer`/`solution`/
`gold_answer`) for local data, e.g. `data/nonmath_reasoning_smoke.jsonl`.

## Architecture

Two script families, joined by CSV artifacts on disk:

1. **GPU runners** — `run_settlement_fate_audit.py`, `run_token_fate_probe_dataset.py`,
   `run_logit_lens_token_fate.py`, `run_counterfactual_commit.py`, `run_regret_pilot.py`.
   They load LLaDA, replay a decode, and emit `metadata.csv` / `*.csv` / `summary.json`.
2. **CPU analyzers** — `train_*.py`, `analyze_*.py`, `simulate_*.py`, `summarize_*.py`.
   They only read those artifacts; they never touch the model. All model fitting is
   sklearn.

The one exception to "joined only by CSVs": `run_settlement_fate_audit.py` imports
`counterfactual_token_effects` from `run_counterfactual_commit.py` and
`parse_layers` / `register_layer_hooks` / `resolve_hook_layers` from
`run_logit_lens_token_fate.py` as sibling **modules**. That resolves only because
invoking it as a file path puts `scripts/` at `sys.path[0]` — so run it as
`python3 scripts/run_settlement_fate_audit.py`, never `python3 -m`, and renaming
those functions in either sibling breaks the audit. `tests/test_drift_update.py` and
`tests/test_settlement_commit_selection.py` import from the audit script the same
way, inserting `scripts/` themselves.

### The audit artifact contract

Which premise a settlement-fate run can speak to is set by `--intervention-mode`,
and each downstream analyzer reads one specific file:

| Artifact in `RUN_DIR` | Written when | Evidence for | Consumed by |
| --- | --- | --- | --- |
| `committed_token_drift.csv` | always | **A** (post-commit drift, no intervention needed) | read directly |
| `freeze_pair_summary.csv` | `--intervention-mode clamp` | **B**, **C** | `analyze_settlement_pairs.py` |
| `commit_pair_summary.csv` | `--intervention-mode commit` | auxiliary timing (*not* A) | `analyze_settlement_pairs.py` |
| `drift_time_rows.csv` | unless `--skip-drift-time-rows` | **U** proxy | `simulate_compute_utility.py` |
| `proposal_ledger.csv` | always | settlement-timing panel | `simulate_compute_utility.py --timing-run` |

Both analyzers emit `paired_bootstrap.csv` / `policy_sweep.csv` plus
`summary.json` and `summary.md`; the `summary.md` files are the force-added
reported artifacts.

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

`EOS_ID` / `EOT_ID` (126081 / 126348) are imported from `llada_trace` by the
runners but **hardcoded** in `analyze_settlement_pairs.py`; keep them in step.

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
- **Bootstrap over example IDs, never over pairs or dense token rows** — several
  pairs come from one trajectory. `analyze_settlement_pairs.py` already clusters
  this way; keep it that way.
- `tests/test_settlement_commit_selection.py` pins `choose_commit_candidates` and
  `tests/test_drift_update.py` pins `update_committed_drift` / `relative_l2`. Same
  rule: a change that breaks them is weakening the design, not fixing a test.
- The strict-intervention **null result is deliberately preserved** (see README and
  `docs/causal_robustness_results.md`). Do not restate prediction results as causal
  ones or soften the reported intervals.

### Settlement-fate run flags the code does not enforce

- **Always pass `--exclude-eos-eot`.** EOS/EOT padding drifts roughly twice as far
  as body tokens but is inert under clamping, so without the flag it floods the
  high-drift arm with null interventions and washes out the contrast. This alone
  once looked like a failed replication.
- **Premise C is invisible at gen 64** — the drift distribution is compressed and
  harm saturates above drift ~0.3, so high-vs-low contrasts measure nothing. Never
  benchmark a drift-gated policy at short generations only.
- **`--skip-drift-time-rows` makes a run unusable for `simulate_compute_utility.py`.**
  It exists for gen-256 throughput; the real fix is append-mode row writing rather
  than the current quadratic rewrite.
- **Commit-arm candidates are not comparable across generation lengths.** The
  chooser takes top-N by confidence per example, so a 256-token example yields a
  much safer candidate set than a 64-token one — any cross-regime comparison there
  is comparing token populations, not difficulty or length.
- **Never pool clamp and commit runs.** `analyze_settlement_pairs.py` raises rather
  than mixing modes; don't work around it.

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
