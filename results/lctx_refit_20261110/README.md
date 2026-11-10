# Lctx refit in the geometry of use (16-window and 32-block + blank-row refresh), 2026-11-10

**STATUS: built and queued; nothing has been decoded or fitted yet** (no result exists in this directory). Pre-registration: `PREREGISTERED.md` (written to disk before the jobs were submitted; nothing is committed to git). Words: WORD LOCK = which blanks are written per call; REPRESENTATION CACHE = which rows' K/V are reused. Datasets: collection on MATH level-5 (the original rule's 60 fit + 20 threshold questions), evaluation on HumanEval 164 and MATH 108, both at 512 tokens.

## Question
Does a Lctx-0.8 rule refitted on one-word-per-call traces collected in the geometry of use fix the 16-window (S16) transfer failure of the original rule (HumanEval 164 net -28 [-39, -17] against dense C-0.9, 63% parse; MATH 108 -11 [-19, -3])? And does fitting under the S32 cache with our blank-row refresh (R-S32c) change the rule under that cache (P4)?

## Chain (Slurm, ws-ia, one GPU job at a time, each pends behind the earlier ones)
| job | name | dependency | what | expected |
|---|---|---|---|---|
| 290057 | refitC | afterany:290035 (stage F) | smoke + collection R-S16 (80 q), smoke + collection R-S32c (80 q), then the CPU fit of both geometries on the node, then `check_rules.py` | about 55 min [40, 90], limit 2:30 |
| 290058 | refitG | afterany:290057 | `check_rules.py` first (stops if rules are missing), then stage G on MATH 108 and HumanEval 164 (smoke each), MATH analysis | about 2.4 h [1.8, 3.3], limit 4:30 |

(290035 = stage F and 289937 = stage E are not touched. The user's 2-running-job cap makes both of mine wait behind them.)

**Automatic:** collection, the fit (rule pickles, `rules.json` with sha256, `validation/refit_*.md`), the rule check, stage G decodes, MATH 108 scoring (in-process) and `results_stageG_math108.md/json`. If job 1 fails, job 2 stops at its rule check before loading the model (afterany fires it regardless).
**Manual steps afterwards (not automatic):**
1. Read `validation/refit_S16.md`, `validation/refit_S32c.md` (label-level results R0 and R4 of the pre-registration) as soon as job 1 ends.
2. HumanEval scoring (this host has no bwrap): `export PATH=/tmp/claude-3430/-home-rui-gao-Desktop-Token-Fate-Diagnostics/a6c9e73d-5398-4784-8f1d-2dd0b921122b/scratchpad/bwrap_env/bin:$PATH`, then
   `cd results/lctx_refit_20261110/orchestration && /home/rui.gao/Desktop/Token-Fate-Diagnostics/results/repfreeze_collect_20261006/venv_cu126/bin/python rescore_records.py G_he164` (writes `records/stageG_he164.scored.jsonl`), then `... python analyze_stage_g.py he164` (writes `results_stageG_he164.md/json`, with the parse share).
3. Fill in the outcome sections (append to `PREREGISTERED.md`, do not edit the predictions) and add a row to `results/README.md` if reported artifacts are force-added.

## Layout
- `PREREGISTERED.md`; `README.md`.
- `orchestration/`: `build_manifests.py` (fit/threshold manifest + disjointness audit), `common.py` (copied from depth_refresh, adapted; collection and evaluation configs), `collect_traces.py` (GPU, resumable), `refit_lib.py` + `fit_refit.py` (CPU fit; **a reconstruction**, see below), `check_rules.py`, `run_refit_cells.py` (stage G), `rescore_records.py`, `analyze_stage_g.py`, `freeze_source.py`, `slurm_collect.sbatch`, `slurm_eval.sbatch`.
- `source/` frozen copy of `src/regret_remasking` (35 files, `SHA256.json`; aggregate 514c02f3a9ed812c...; `selective_cache.py` 8d3c2374...). Every GPU script imports from it. It differs from `depth_refresh_20261109/source` only by the default-off `collect_rows` option and the factoring of `_rule_features`.
- `manifests/` `fit_thr.jsonl` (80), `he164.jsonl`, `math_l5.jsonl`, `math_l14.jsonl` (the last three are copies, sha256 equal to the farcache_stack manifests); `validation/disjoint.json` (audit: disjoint by id, normalised text, 200-character substring from MATH 108 and HumanEval 164).
- `records/` (gitignored with the rest of `results/**`): `traces_<GEOM>/<id>.npz`, `collect_<GEOM>.jsonl`, `stageG_<set>.jsonl`. `rules/R-S16`, `rules/R-S32c`: the refit rules (after job 1). `logs/`.

## Notes on what is and is not verified
- The logging path is unit-tested on the toy model (default-off invariance in both shapes, logged features bit-equal to what a recording rule is given, labels from final tokens, the pooled-lambda rule, rule round trip); full suite 451 passed, 2 skipped. The collection loop and the fit were run end to end on CPU with the toy model and fake ids (outputs deleted). **No GPU smoke has been run yet**; the first GPU contact is the smoke inside job 1 and inside job 2.
- **The fit is a reconstruction.** The original fitter and its training rows are not in the repository (the worktree that held them is gone). Learner and hyperparameters are cloned from the original pickle; the lambda rule and the row filter are pinned from `rules.json`, `fit_feature_ranges.json` and the earlier session's description. It cannot be shown to reproduce the original pickle; no control "dense B0 collected and fitted with this pipeline" (R-D32, about 1 GPU-hour) is run.
- Collection has no first-EOS truncation (the original fit was dense B0 without it); evaluation arms have (as in every earlier stage). Primary fit keeps first-call-of-block rows, which the original dropped (the secondary `orig_filter` fit is label-level only).
- Resolution of the answer comparisons is about +-6 answers at these sizes; speed is one timed repeat.
