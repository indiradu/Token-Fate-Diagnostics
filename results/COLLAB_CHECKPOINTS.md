# Checkpoints shared for the collaborator comparison (2026-11-18)

Requested: the original selective word lock (Lctx-0.8) and the TraceLock controller and projector, to compare against the new feature sets. Files are byte-identical copies of the originals in the working checkout (SHA-256 below); nothing was retrained or modified.

| What | Path | SHA-256 |
|---|---|---|
| Lctx-0.8 context-rule word lock (pickled `ContextRule`, fitted on MATH level-5 one-word-per-call traces, frozen) | `results/longlen_wordlock_20261103/rules/math512/Lctx_c0.8.pkl` | `6fe725ea2e893d4602e0e1e4c9ca74b9e1d48104015c641e5d339a6aac692f70` |
| Rule metadata | `results/longlen_wordlock_20261103/rules/math512/rules.json` | `0623c22fc123bf019de96a1e04e1256ae5a953ae5ae7a0c297f143c420bc1269` |
| Manifests used by that word-lock campaign (MATH level 5, levels 1-4, audit) | `results/longlen_wordlock_20261103/manifests/` | `math_l5.jsonl` `4fbeff1a...`, `math_l14.jsonl` `90393752...`, `audit.json` `00f75edc...` |
| TraceLock controller (E2 full, attempt 01) | `results/locking_paper_redesign_20260920_161812/e2_train/full/attempt_01/best_controller.pt` | `a36a47e8ac8f88874c024150ec008b3a26a0b0de27a17bd8a789ac4e602f93c8` |
| Controller config, metrics, provenance | same directory: `config.json`, `metrics.json`, `provenance.json` | provenance `7c4b9df9...` |
| TraceLock projector | `results/locking_paper_redesign_20260920_161812/projector/projector.pt` (+ `projector.pt.json`) | `6cba72b5a2a2c01a2982308136280a3c68855d85f5218b96e7c1d4522431155d` |

Notes
- The projector is NOT under `e2_train/.../attempt_01/projector/` (that directory does not exist); the controller's `provenance.json` points to `results/locking_paper_redesign_20260920_161812/projector/projector.pt`, and its `projection_sha256` equals the projector hash above.
- `Lctx_c0.9.pkl` (the 0.9-threshold variant in the same folder) is not included; ask if it is needed.
- Lctx-0.8 is used as `lock="rule"` with a `ContextRule` in the selective-cache engine (`src/regret_remasking/context_rules.py`, `selective_cache.py`); TraceLock as `lock="trace"` / arm `T-block` (`trace_lock.py`). The pickle needs this repo's `regret_remasking` package on the path to load (and trusts the pickle: load only from this repo).
- These are binary model files force-added past the `results/**` ignore on purpose; `.gitignore` is unchanged.
