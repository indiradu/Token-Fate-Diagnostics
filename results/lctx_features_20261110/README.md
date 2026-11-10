# Learned context word lock Lctx: do extra decode-time features help? (2026-11-10)

**STATUS: built and queued (jobs 290320 featC, 290321 featH); nothing has been decoded and no fit has been made on real labels yet.** Pre-registration (gate, predictions, decision rules): `PREREGISTERED.md`, written before any job of this directory. Nothing is committed to git. Words: WORD LOCK = which blanks are written in a call; REPRESENTATION CACHE = which rows' K/V are reused. Datasets: collection and fit on MATH level-5 (the original rule's 60 fit + 20 threshold questions, one word per call, 512 tokens, 32-blocks, representation cache = blank-row refresh every 4th call from layer 11 = the refit's R-S32c geometry); decodes (only if the gate passes) on HumanEval 164 and MATH 108.

## Question
Does adding stability (a), neighbour (b), proposal (c) or frontier (d) features, all available at decode time without extra forward passes, to the six inputs of Lctx-0.8 raise AUROC and the share of blanks admitted at C-0.8's precision enough (gate: +5 points of share AND +0.01 AUROC with the lower bootstrap bound above 0) to justify decoding?

## Chain (Slurm, ws-ia; both jobs pend behind the user's running jobs)
| job | name | dependency | what | expected |
|---|---|---|---|---|
| 290320 | featC | afterany:290058:290265 | collection smoke + 80 questions (R-S32c with extras), CPU ablation on the node (`fit_ablation.py`), `check_gate.py` (informational), DRY decode smoke (2 MATH questions x 4 arms, ungated six+all rule) | about 1.0 h [0.8, 1.6], limit 3:30 |
| 290321 | featH | afterany:290320 | `check_gate.py` first: exit 0 without loading the model if the gate failed; otherwise stage H (4 arms: D:C-0.9, P4:C-0.8, P4:Lctx-0.8, P4:Lctx-X) on MATH 108 then HumanEval 164 (smoke each), MATH analysis | gate fails: seconds; passes: about 1.5 h [1.1, 2.2], limit 4:00 |

**Automatic:** everything above, including the rule file `rules/X/` (written only on a pass), `validation/ablation_S32c.{json,md}`, `validation/gate.json`, `validation/validity.json`, MATH 108 scoring and `results_stageH_math108.md/json`.
**Manual afterwards (not automatic):**
1. Read `validation/ablation_S32c.md` and `validation/gate.json` when JOB1 ends (label-level result and gate).
2. Only if stage H ran: HumanEval scoring (no bwrap on this host): `export PATH=/tmp/claude-3430/-home-rui-gao-Desktop-Token-Fate-Diagnostics/a6c9e73d-5398-4784-8f1d-2dd0b921122b/scratchpad/bwrap_env/bin:$PATH; cd results/lctx_features_20261110/orchestration; PY=/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python; $PY rescore_records.py H_he164; $PY analyze_stage_h.py he164; $PY analyze_stage_h.py both`.
3. Append the outcome to `PREREGISTERED.md` (do not edit the predictions) and add a row to `results/README.md` if reported artifacts are force-added.

## Layout
- `orchestration/`: `features_lib.py` (CPU helpers copied from the refit's `refit_lib.py`, plus `derive_features/derive_check`, the post-hoc re-derivation behind check V2), `common.py` (copy of the refit's, one geometry, extras, token-class table), `build_token_class.py`, `collect_traces.py` (GPU, resumable, V1/V2 inside), `fit_ablation.py` (CPU), `check_gate.py`, `run_feature_cells.py` (stage H), `analyze_stage_h.py`, `rescore_records.py`, `freeze_source.py`, `slurm_collect.sbatch`, `slurm_eval.sbatch`.
- `source/` frozen copy of `src/regret_remasking` (36 files; differs from `lctx_refit_20261110/source` only in `context_rules.py`, `selective_cache.py`, new `lock_features.py`). `assets/token_class.{npy,json}`; `manifests/` (copies of the refit's, sha256 equal).
- `records/` (gitignored with the rest of `results/**`): `traces_S32c/<id>.npz`, `collect_S32c.jsonl`, `dry_stageH_math108.jsonl`, `stageH_<set>.jsonl`. `rules/X` (gated), `rules/dry` (ungated, smoke only), `validation/`, `logs/`.

## Engine changes (default off; src/regret_remasking)
`lock_features.py` (new): the 15 features and the one-call state. `selective_cache.py`: config fields `collect_extra`, `token_class`; result field `extra_state`; `_rule_features` takes `res, tokens` and adds the extras when `lock_features.wants_extra(cfg)` (collect_extra or a rule that names an extra feature). `context_rules.py`: the feature-name whitelist is extended. The original six-input rule and decode paths are unchanged (tests pin token equality).

## What is and is not verified
- Unit tests and a CPU pipeline run on synthetic labels only (see Amendment 1). **No GPU contact yet**: the first is the smoke inside JOB1. No fit on real labels, so no result.
- A GBM fitted on one-word-per-call rows is deployed on multi-write rows; the word-count normalisation of the history features makes them comparable in units, not in distribution; `kl_prev` is the most shift-prone. No offline number tests this shift; only stage H does.
- The gate selects the best of five sets on the same 20 threshold questions that fix lambda (not corrected). Resolution of the decode comparisons is about +-6 answers.
