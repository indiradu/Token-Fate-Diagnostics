"""CPU. Fit the refit Lctx-0.8 rule on one collection geometry and freeze it (PREREGISTERED.md, part 2).
Usage: <python with sklearn+pandas> fit_refit.py GEOM      GEOM in {S16, S32c}
Reads records/traces_<GEOM>/<id>.npz of the 60 fit + 20 threshold questions (all 80 must exist), writes
  rules/R-<GEOM>/Lctx_refit_<GEOM>_c0.8.pkl  (a ContextRule that loads with ContextRule.load) and rules/R-<GEOM>/rules.json (same fields as the original rules.json, sha256 included)
  validation/fit_feature_ranges_<GEOM>.json  (same format as the original validation/fit_feature_ranges.json)
  validation/refit_<GEOM>.json / .md          (admitted shares, AUROC, original rule on the new rows, feature-range shift)
The ORIGINAL rule (results/longlen_wordlock_20261103/rules/math512) is only read. Primary variant 'all' (every non-EOS/EOT proposal row); secondary 'orig_filter' (also drops the first call of each
block, the original fit's filter) is fitted and reported but not saved. See refit_lib.py for what is reconstructed."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
SRC = Path(__file__).resolve().parents[1] / "source/src"
sys.path.insert(0, str(SRC if SRC.exists() else Path(__file__).resolve().parents[3] / "src"))
import refit_lib as RL  # noqa: E402
from regret_remasking.context_rules import ContextRule  # noqa: E402

OUT = Path(__file__).resolve().parents[1]
WL = OUT.parents[1] / "results/longlen_wordlock_20261103"
C_TARGET = 0.8
SEED = 23


def g(x, f):
    return "n/a" if x is None else format(x, f)


def boot_auroc_diff(thr: pd.DataFrame, s_new: np.ndarray, s_old: np.ndarray, n: int = 300) -> dict:
    """Cluster bootstrap over questions of AUROC(new) - AUROC(old) on the threshold rows."""
    rng = np.random.default_rng(SEED)
    ids = thr["example_id"].to_numpy()
    uniq = np.unique(ids)
    groups = {u: np.nonzero(ids == u)[0] for u in uniq}
    y = thr["y"].to_numpy()
    out = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([groups[u] for u in pick])
        if y[idx].min() == y[idx].max():
            continue
        out.append(RL.auroc(y[idx], s_new[idx]) - RL.auroc(y[idx], s_old[idx]))
    return {"mean": float(np.mean(out)), "ci95": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))], "n_boot": len(out)}


def main(geom: str) -> None:
    meta0 = json.loads((WL / "rules/math512/rules.json").read_text())
    fit_ids, thr_ids = list(meta0["fit_questions"]), list(meta0["thr_questions"])
    tdir = OUT / f"records/traces_{geom}"
    missing = [q for q in fit_ids + thr_ids if not (tdir / f"{q}.npz").exists()]
    if missing:
        raise SystemExit(f"{geom}: {len(missing)} of 80 trace files missing (e.g. {missing[:3]}); nothing fitted")
    fit_all, thr_all = RL.load_traces(tdir, fit_ids), RL.load_traces(tdir, thr_ids)
    rule0 = ContextRule.load(WL / "rules/math512/Lctx_c0.8.pkl")
    template = rule0.payload
    ranges0 = json.loads((WL / "validation/fit_feature_ranges.json").read_text())["ranges"]
    res: dict = {"geometry": geom, "fit_questions": len(fit_ids), "thr_questions": len(thr_ids), "rows_logged": {"fit": int(len(fit_all)), "thr": int(len(thr_all))}, "variants": {}}
    out_dir = OUT / f"rules/R-{geom}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for variant in ("all", "orig_filter"):
        fit, thr = RL.select_rows(fit_all, variant), RL.select_rows(thr_all, variant)
        model = RL.fit_gbm(fit, template, SEED)
        s_thr = model.predict_proba(thr[RL.FEATURES].to_numpy("float32"))[:, 1]
        s_old = rule0.scores({f: thr[f].to_numpy() for f in RL.FEATURES})
        y = thr["y"].to_numpy()
        conf = thr["confidence"].to_numpy()
        target = float(y[conf >= C_TARGET].mean())
        lam = RL.pooled_lambda(s_thr, y, target)
        if lam is None:
            raise SystemExit(f"{geom}/{variant}: no admissible threshold")
        lam_old_recal = RL.pooled_lambda(s_old, y, target)
        v = {"rows": {"fit": int(len(fit)), "thr": int(len(thr))}, "y_mean": {"fit": float(fit["y"].mean()), "thr": float(y.mean())},
             "C-0.8": {"target_precision": target, "admitted_share": float((conf >= C_TARGET).mean())},
             "refit": {"lambda": lam, "admitted_share": float((s_thr >= lam).mean()), "precision": float(y[s_thr >= lam].mean()), "auroc": RL.auroc(y, s_thr)},
             "original_rule_as_is": {"lambda": rule0.lam, "admitted_share": float((s_old >= rule0.lam).mean()), "precision": float(y[s_old >= rule0.lam].mean()) if (s_old >= rule0.lam).any() else None,
                                      "auroc": RL.auroc(y, s_old)},
             "original_rule_recalibrated_here": {"lambda": lam_old_recal, "admitted_share": float((s_old >= lam_old_recal).mean()) if lam_old_recal is not None else None},
             "confidence_only_auroc": RL.auroc(y, conf), "auroc_refit_minus_original_boot": boot_auroc_diff(thr, s_thr, s_old),
             "first_call_rows_share_thr": float(thr_all["first_call_in_block"].mean())}
        res["variants"][variant] = v
        if variant == "all":
            name = f"Lctx_refit_{geom}_c0.8"
            rule = ContextRule("gbm", lam, features=list(RL.FEATURES), payload=model, name=name)
            path = out_dir / f"{name}.pkl"
            rule.save(path)
            back = ContextRule.load(path)
            assert np.array_equal(back.scores({f: thr[f].to_numpy() for f in RL.FEATURES}), s_thr)
            RL.dump(out_dir / "rules.json", {"dataset": f"math512-collected-{geom}", "gen_cap": 512, "geometry": geom, "fit_questions": sorted(fit_ids), "thr_questions": sorted(thr_ids),
                                             "rules": {name: {"lambda": lam, "target_precision": target, "features": list(RL.FEATURES), "sha256": RL.sha256(path),
                                                              "threshold_set_admitted_share": v["refit"]["admitted_share"], "precision_on_threshold_set": v["refit"]["precision"]}}})
            rng = RL.feature_ranges(fit)
            RL.dump(OUT / f"validation/fit_feature_ranges_{geom}.json", {"rows": f"{geom} one-word-per-call fit rows (60 questions), non-EOS/EOT proposals, all calls incl. the first call of a block", "gen_cap": 512, "ranges": rng})
            res["feature_range_shift_vs_original"] = {f: {"original": {k: ranges0[f][k] for k in ("min", "p0.1", "p50", "p99.9", "max")}, "new": {k: rng[f][k] for k in ("min", "p0.1", "p50", "p99.9", "max")}} for f in RL.FEATURES}
            res["share_of_new_fit_rows_outside_original_range"] = RL.share_outside(fit, ranges0)
            res["dist_written_above_original_max_31_share"] = float((fit["dist_written"] > 31).mean())
            res["rule"] = {"file": str(path.relative_to(OUT)), "sha256": RL.sha256(path), "lambda": lam}
    RL.dump(OUT / f"validation/refit_{geom}.json", res)
    a, b = res["variants"]["all"], res["variants"]["orig_filter"]
    lines = [f"# Refit Lctx-0.8 on {geom} one-word-per-call traces", "",
             f"Rows (non-EOS/EOT proposals): fit {a['rows']['fit']}, threshold {a['rows']['thr']}; label mean fit {a['y_mean']['fit']:.3f}, threshold {a['y_mean']['thr']:.3f}.", "",
             "| quantity (threshold-question rows) | refit (primary) | refit, original row filter | original rule as is | original rule, lambda recalibrated here | C-0.8 |", "|---|---|---|---|---|---|",
             f"| lambda | {a['refit']['lambda']:.4f} | {b['refit']['lambda']:.4f} | {a['original_rule_as_is']['lambda']:.4f} | {g(a['original_rule_recalibrated_here']['lambda'], '.4f')} | 0.8 (confidence) |",
             f"| admitted share | {a['refit']['admitted_share']:.3f} | {b['refit']['admitted_share']:.3f} | {a['original_rule_as_is']['admitted_share']:.3f} | {g(a['original_rule_recalibrated_here']['admitted_share'], '.3f')} | {a['C-0.8']['admitted_share']:.3f} |",
             f"| pooled precision of the admitted rows | {a['refit']['precision']:.4f} | {b['refit']['precision']:.4f} | {g(a['original_rule_as_is']['precision'], '.4f')} | (target) | {a['C-0.8']['target_precision']:.4f} |",
             f"| AUROC (label = proposal equals final token) | {a['refit']['auroc']:.4f} | {b['refit']['auroc']:.4f} | {a['original_rule_as_is']['auroc']:.4f} | same | confidence alone {a['confidence_only_auroc']:.4f} |", "",
             f"AUROC refit minus original, cluster bootstrap over the {res['thr_questions']} threshold questions: {a['auroc_refit_minus_original_boot']['mean']:+.4f} {a['auroc_refit_minus_original_boot']['ci95']}.", "",
             "Feature ranges, new fit rows vs the original fit rows (min / 0.1% / median / 99.9% / max) and the share of new rows outside the original [min, max]:", "",
             "| feature | original | new | share outside original range |", "|---|---|---|---|"]
    for f in RL.FEATURES:
        o, n = res["feature_range_shift_vs_original"][f]["original"], res["feature_range_shift_vs_original"][f]["new"]
        fmt = lambda d: " / ".join(f"{d[k]:.3g}" for k in ("min", "p0.1", "p50", "p99.9", "max"))
        lines.append(f"| {f} | {fmt(o)} | {fmt(n)} | {res['share_of_new_fit_rows_outside_original_range'][f]:.4f} |")
    (OUT / f"validation/refit_{geom}.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
