"""CPU. Offline feature ablation for the learned context word lock (PREREGISTERED.md, section Offline ablation) on the R-S32c one-word-per-call traces of MATH level-5 (60 fit, 20 threshold questions).
Usage: <python with sklearn+pandas+numpy> fit_ablation.py [--boot N]      (needs all 80 trace files records/traces_S32c/<id>.npz and records/collect_S32c.jsonl)
Writes validation/ablation_S32c.json and .md, validation/validity.json; then evaluates the gate (check_gate.evaluate_gate) and, ONLY if it passes, writes rules/X/Lctx_X_c0.8.pkl + rules/X/rules.json.

Same learner as the refit and the original rule (sklearn.base.clone of the original pickle's HistGradientBoosting, 200 rounds, lr 0.1, no early stopping, seed 23), the refit's primary row filter (every logged row whose
proposal is not EOS/EOT), the refit's lambda rule (pooled precision of C-0.8 on the threshold rows; lowest score threshold whose pooled admitted precision is still >= the target; ties never split).
Cluster bootstrap over the 20 threshold questions (default 2,000 resamples, seed 23): in EVERY resample the C-0.8 target and the lambda of every set are re-chosen, so the intervals cover the whole procedure;
the paired differences to the six-input control (and to confidence alone) use the same resamples. The bootstrap is exact (weights on the globally sorted rows), validated against the direct computation."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "4"))     # HistGradientBoosting threads (common.py is not imported here: it pins 1 thread)
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(OUT / "source/src" if (OUT / "source/src").exists() else ROOT / "src"))
import features_lib as FL  # noqa: E402
from check_gate import evaluate_gate  # noqa: E402
from regret_remasking.context_rules import ContextRule  # noqa: E402

WL = ROOT / "results/longlen_wordlock_20261103"
REFIT = ROOT / "results/lctx_refit_20261110"
SEED, C_TARGET = 23, 0.8
SIX = FL.FEATURES
SETS: dict[str, list[str]] = {"six": SIX, **{f"six+{g}": SIX + FL.GROUPS[g] for g in "abcd"}, "six+all": SIX + FL.EXTRA,
                              "conf": ["confidence"], **{f"conf+{g}": ["confidence"] + FL.GROUPS[g] for g in "abcd"}}


class Boot:
    """Exact weighted statistics of one score vector on the threshold rows: a bootstrap resample is a vector of question multiplicities m; every row gets weight m[question]."""

    def __init__(self, y: np.ndarray, q: np.ndarray, score: np.ndarray):
        order = np.argsort(-score, kind="stable")
        s = score[order]
        self.y, self.q = y[order].astype(np.float64), q[order]
        self.start = np.concatenate([[0], np.nonzero(s[1:] != s[:-1])[0] + 1])      # tie groups (never split)
        self.score_sorted = s

    def stats(self, m: np.ndarray, target: float) -> tuple[float, float, float]:
        """(AUROC, admitted share, admitted precision) at the lowest threshold whose pooled admitted precision is still >= target."""
        w = m[self.q].astype(np.float64)
        pos = np.add.reduceat(w * self.y, self.start)
        neg = np.add.reduceat(w * (1.0 - self.y), self.start)
        P, N = pos.sum(), neg.sum()
        auc = float((pos * ((N - np.cumsum(neg)) + 0.5 * neg)).sum() / (P * N)) if P > 0 and N > 0 else float("nan")
        tot = np.cumsum(pos + neg)
        prec = np.cumsum(pos) / np.maximum(tot, 1e-300)
        ok = np.nonzero((tot > 0) & (prec >= target - 1e-15))[0]
        if len(ok) == 0:
            return auc, 0.0, float("nan")
        g = ok[-1]
        return auc, float(tot[g] / tot[-1]), float(prec[g])


def c08(conf: np.ndarray, y: np.ndarray, q: np.ndarray, m: np.ndarray) -> tuple[float, float]:
    """(target precision, share) of the confidence >= 0.8 rule under multiplicities m."""
    w = m[q].astype(np.float64)
    sel = conf >= C_TARGET
    den = (w * sel).sum()
    return (float((w * sel * y).sum() / den) if den > 0 else float("nan")), float(den / w.sum())


def ci(v: np.ndarray) -> list[float]:
    v = v[~np.isnan(v)]
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def diff_block(point: float, boot: np.ndarray) -> dict:
    b = boot[~np.isnan(boot)]
    return {"point": point, "mean": float(b.mean()), "ci95": ci(b), "p_gt0": float((b > 0).mean())}


def perm_importance(model, X: np.ndarray, y: np.ndarray, feats: list[str], n_rep: int = 3) -> dict:
    """Drop in AUROC on the threshold rows when a feature (or all columns of a group jointly) is shuffled; mean of n_rep shuffles, seed 23."""
    rng = np.random.default_rng(SEED)
    base = FL.auroc(y, model.predict_proba(X)[:, 1])
    units = {f: [i] for i, f in enumerate(feats)}
    groups = {"six(original inputs)": [i for i, f in enumerate(feats) if f in SIX], **{f"group {g}": [i for i, f in enumerate(feats) if f in FL.GROUPS[g]] for g in "abcd"}}
    groups = {k: v for k, v in groups.items() if v}
    out = {"base_auroc": base, "features": {}, "groups": {}}
    for tag, table in (("features", units), ("groups", groups)):
        for name, idx in table.items():
            drops = []
            for _ in range(n_rep):
                perm = rng.permutation(len(X))
                Xp = X.copy()
                Xp[:, idx] = X[np.ix_(perm, idx)]
                drops.append(base - FL.auroc(y, model.predict_proba(Xp)[:, 1]))
            out[tag][name] = float(np.mean(drops))
    return out


def validity(ids: list[str], tdir: Path, refit_json: dict, ctrl: dict) -> dict:
    """V1: collection reproduces the refit's traces; V2: logged extras equal their post-hoc derivation; V3: the control reproduces the refit's numbers."""
    lp = OUT / "records/collect_S32c.jsonl"
    log = [json.loads(x) for x in lp.read_text().splitlines() if x.strip()] if lp.exists() else []
    v1 = [r.get("v1_vs_refit", {}) for r in log if r["example_id"] in set(ids)]
    bad, n_rows = {}, 0
    for qid in ids:
        z = np.load(tdir / f"{qid}.npz", allow_pickle=False)
        d = FL.derive_check({k: z[k] for k in z.files if z[k].ndim == 1})
        n_rows += d["rows"]
        for k, c in d["mismatches"].items():
            bad[k] = bad.get(k, 0) + c
    ref = refit_json["variants"]["all"]["refit"]
    return {"V1_questions_logged": len(v1), "V1_same_trajectory": int(sum(bool(r.get("same_trajectory")) for r in v1)), "V1_six_inputs_bit_equal": int(sum(bool(r.get("six_bit_equal")) for r in v1)),
            "V1_y_equal": int(sum(bool(r.get("y_equal")) for r in v1)), "V2_rows_checked": n_rows, "V2_mismatches": bad, "V2_derived_features_equal": not bad,
            "V3_refit_auroc": ref["auroc"], "V3_refit_share": ref["admitted_share"], "V3_control_auroc": ctrl["auroc"], "V3_control_share": ctrl["share"],
            "V3_control_reproduces_refit": bool(abs(ctrl["auroc"] - ref["auroc"]) <= 0.002 and abs(ctrl["share"] - ref["admitted_share"]) <= 0.01)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--traces", default="records/traces_S32c")
    ap.add_argument("--out", default="validation")
    args = ap.parse_args()
    t_start = time.time()
    meta0 = json.loads((WL / "rules/math512/rules.json").read_text())
    fit_ids, thr_ids = list(meta0["fit_questions"]), list(meta0["thr_questions"])
    tdir = OUT / args.traces
    missing = [q for q in fit_ids + thr_ids if not (tdir / f"{q}.npz").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} of 80 trace files missing (e.g. {missing[:3]}); nothing fitted")
    fit, thr = FL.select_rows(FL.load_traces(tdir, fit_ids), "all"), FL.select_rows(FL.load_traces(tdir, thr_ids), "all")
    template = ContextRule.load(WL / "rules/math512/Lctx_c0.8.pkl").payload
    y = thr["y"].to_numpy()
    conf = thr["confidence"].to_numpy()
    q = pd.factorize(thr["example_id"])[0]
    nq = int(q.max()) + 1
    ones = np.ones(nq)
    target = float(y[conf >= C_TARGET].mean())
    print(f"[ablation] rows fit {len(fit)} thr {len(thr)} questions {nq}; C-0.8 target precision {target:.5f}; share {(conf >= C_TARGET).mean():.4f}", flush=True)
    scores, models, res_sets = {"conf_raw": conf.astype(np.float64)}, {}, {}
    Xthr = {}
    for name, feats in SETS.items():
        t0 = time.time()
        models[name] = FL.fit_gbm(fit, template, SEED, feats)
        Xthr[name] = thr[feats].to_numpy("float32")
        scores[name] = models[name].predict_proba(Xthr[name])[:, 1]
        print(f"[ablation] fitted {name} ({len(feats)} features) in {time.time() - t0:.0f}s", flush=True)
    boots = {k: Boot(y, q, s) for k, s in scores.items()}
    # exactness check of the weighted statistics against the direct computation (point estimates, m = 1)
    point = {}
    for k, s in scores.items():
        lam = FL.pooled_lambda(s, y, target)
        auc_d = FL.auroc(y, s)
        auc_w, share_w, prec_w = boots[k].stats(ones, target)
        share_d = float((s >= lam).mean()) if lam is not None else 0.0
        assert abs(auc_d - auc_w) < 1e-9 and abs(share_d - share_w) < 1e-9, (k, auc_d, auc_w, share_d, share_w)
        point[k] = {"auroc": auc_d, "share": share_d, "lambda": lam, "precision": float(y[s >= lam].mean()) if lam is not None else None}
    # bootstrap: the same resamples for every set; target and lambda re-chosen in each
    rng = np.random.default_rng(SEED)
    B = args.boot
    M = np.stack([np.bincount(rng.integers(0, nq, size=nq), minlength=nq) for _ in range(B)])
    auc_b = {k: np.full(B, np.nan) for k in scores}
    share_b = {k: np.full(B, np.nan) for k in scores}
    c08_share_b = np.zeros(B)
    for b in range(B):
        t_b, c08_share_b[b] = c08(conf, y, q, M[b])
        for k in scores:
            a, s_, _ = boots[k].stats(M[b], t_b)
            auc_b[k][b], share_b[k][b] = a, s_
    print(f"[ablation] bootstrap {B} resamples done ({time.time() - t_start:.0f}s elapsed)", flush=True)
    res_sets = {}
    for k in SETS:
        d = {"features": SETS[k], "n_features": len(SETS[k]), **point[k], "auroc_ci95": ci(auc_b[k]), "share_ci95": ci(share_b[k]),
             "share_minus_c08": diff_block(point[k]["share"] - float((conf >= C_TARGET).mean()), share_b[k] - c08_share_b)}
        for ref in ("six", "conf"):
            d[f"auroc_diff_vs_{ref}"] = diff_block(point[k]["auroc"] - point[ref]["auroc"], auc_b[k] - auc_b[ref])
            d[f"share_diff_vs_{ref}"] = diff_block(point[k]["share"] - point[ref]["share"], share_b[k] - share_b[ref])
        res_sets[k] = d
    cands = ["six+a", "six+b", "six+c", "six+d", "six+all"]
    best_b = np.argmax(np.stack([share_b[k] - share_b["six"] for k in cands]), axis=0)
    best_pt = max(cands, key=lambda k: point[k]["share"] - point["six"]["share"])
    optimism = float(np.mean(np.max(np.stack([share_b[k] - share_b["six"] for k in cands]), axis=0)) - np.mean(share_b[best_pt] - share_b["six"]))
    refit_json = json.loads((REFIT / "validation/refit_S32c.json").read_text())
    valid = validity(fit_ids + thr_ids, tdir, refit_json, {"auroc": point["six"]["auroc"], "share": point["six"]["share"]})
    imp = {}
    for k in dict.fromkeys(["six+all", best_pt]):
        imp[k] = perm_importance(models[k], Xthr[k], y, SETS[k])
        print(f"[ablation] permutation importance {k} done", flush=True)
    result = {"geometry": "S32c", "rows": {"fit": int(len(fit)), "thr": int(len(thr))}, "n_boot": B, "seed": SEED, "target_precision_c08": target,
              "c08": {"share": float((conf >= C_TARGET).mean()), "share_ci95": ci(np.array([c08(conf, y, q, M[b])[1] for b in range(B)])), "auroc_raw_confidence": point["conf_raw"]["auroc"]},
              "sets": res_sets, "importance": imp, "validity": valid, "selection": {"best_set_by_share": best_pt, "mean_bootstrap_optimism_of_best_of_5_share_diff": optimism,
                                                                                     "share_of_resamples_where_each_candidate_is_best": {k: float((best_b == i).mean()) for i, k in enumerate(cands)}}}
    gate = evaluate_gate(result)
    result["gate"] = gate
    outdir = OUT / args.out
    outdir.mkdir(exist_ok=True)
    (outdir / "validity.json").write_text(json.dumps(valid, indent=1) + "\n")
    if gate["passes"]:
        best = gate["best_set"]
        rdir = OUT / "rules/X"
        rdir.mkdir(parents=True, exist_ok=True)
        name = "Lctx_X_c0.8"
        rule = ContextRule("gbm", point[best]["lambda"], features=list(SETS[best]), payload=models[best], name=name)
        path = rdir / f"{name}.pkl"
        rule.save(path)
        back = ContextRule.load(path)
        assert np.array_equal(back.scores({f: thr[f].to_numpy() for f in SETS[best]}), scores[best])
        tcm = json.loads((OUT / "assets/token_class.json").read_text())
        FL.dump(rdir / "rules.json", {"name": name, "dataset": "math512-collected-S32c-extra-features", "gen_cap": 512, "geometry": "S32c", "best_set": best, "fit_questions": sorted(fit_ids), "thr_questions": sorted(thr_ids),
                                      "token_class_sha256": tcm["sha256"], "rules": {name: {"lambda": point[best]["lambda"], "target_precision": target, "features": list(SETS[best]), "sha256": FL.sha256(path),
                                                                                         "threshold_set_admitted_share": point[best]["share"], "precision_on_threshold_set": point[best]["precision"]}}})
        result["rule"] = {"file": str(path.relative_to(OUT)), "sha256": FL.sha256(path), "lambda": point[best]["lambda"], "set": best}
    ddir = OUT / "rules/dry"                      # ungated six+all fit for the DRY smoke of the decode path (run_feature_cells.py H_<set> DRY); never a result
    ddir.mkdir(parents=True, exist_ok=True)
    dname = "Lctx_dry_c0.8"
    ContextRule("gbm", point["six+all"]["lambda"], features=list(SETS["six+all"]), payload=models["six+all"], name=dname).save(ddir / f"{dname}.pkl")
    FL.dump(ddir / "rules.json", {"name": dname, "NOTE": "DRY RUN ONLY: the six+all fit, written whether or not the gate passed; not a result", "rules": {dname: {"lambda": point["six+all"]["lambda"], "features": list(SETS["six+all"]),
                                                                                                                 "sha256": FL.sha256(ddir / f"{dname}.pkl")}}})
    FL.dump(outdir / "ablation_S32c.json", result)
    write_md(result, outdir / "ablation_S32c.md")
    print(f"[ablation] done in {time.time() - t_start:.0f}s; gate passes: {gate['passes']} (best set {gate['best_set']})", flush=True)


def write_md(r: dict, path: Path) -> None:
    s = r["sets"]
    L = ["# Offline feature ablation, R-S32c one-word-per-call traces (MATH level-5; 60 fit / 20 threshold questions)", "",
         f"Rows (non-EOS/EOT): fit {r['rows']['fit']}, threshold {r['rows']['thr']}. C-0.8 pooled precision on the threshold rows {r['target_precision_c08']:.5f}, admitted share {r['c08']['share']:.4f} "
         f"{[round(x, 4) for x in r['c08']['share_ci95']]}, AUROC of the raw confidence {r['c08']['auroc_raw_confidence']:.4f}. Cluster bootstrap over the 20 threshold questions, {r['n_boot']} resamples, target and lambda re-chosen in each. "
         "Differences are paired. Share = admitted share at the pooled precision of C-0.8.", "",
         "| set | features | AUROC [95%] | share [95%] | precision | AUROC diff vs six [95%] | share diff vs six, points [95%] | AUROC diff vs conf | share diff vs C-0.8, points |", "|---|---|---|---|---|---|---|---|---|"]
    f = lambda d, k=1.0: f"{d['point'] * k:+.4f} [{d['ci95'][0] * k:+.4f}, {d['ci95'][1] * k:+.4f}]" if k == 1.0 else f"{d['point'] * k:+.2f} [{d['ci95'][0] * k:+.2f}, {d['ci95'][1] * k:+.2f}]"
    for k, d in s.items():
        L.append(f"| {k} | {d['n_features']} | {d['auroc']:.4f} [{d['auroc_ci95'][0]:.4f}, {d['auroc_ci95'][1]:.4f}] | {d['share']:.4f} [{d['share_ci95'][0]:.4f}, {d['share_ci95'][1]:.4f}] | "
                 f"{d['precision']:.4f} | {f(d['auroc_diff_vs_six'])} | {f(d['share_diff_vs_six'], 100)} | {f(d['auroc_diff_vs_conf'])} | {f(d['share_minus_c08'], 100)} |")
    L += ["", "## Permutation importance (drop in AUROC on the threshold rows when shuffled; 3 repeats)", ""]
    for k, imp in r["importance"].items():
        L += [f"**{k}** (base AUROC {imp['base_auroc']:.4f})", "", "| unit | AUROC drop |", "|---|---|"]
        L += [f"| {n} | {v:.5f} |" for n, v in sorted(imp["groups"].items(), key=lambda kv: -kv[1])]
        L += [f"| feature {n} | {v:.5f} |" for n, v in sorted(imp["features"].items(), key=lambda kv: -kv[1])]
        L.append("")
    v = r["validity"]
    L += ["## Validity", "", f"- V1 (collection reproduces the refit's R-S32c traces): trajectory identical on {v['V1_same_trajectory']}/{v['V1_questions_logged']} questions, six inputs bit-equal on {v['V1_six_inputs_bit_equal']}/{v['V1_questions_logged']}, labels equal on {v['V1_y_equal']}.",
          f"- V2 (logged extras equal their post-hoc derivation, {v['V2_rows_checked']} rows): {v['V2_derived_features_equal']} {v['V2_mismatches']}.",
          f"- V3 (control reproduces the refit): refit AUROC {v['V3_refit_auroc']:.4f}, share {v['V3_refit_share']:.4f}; control {v['V3_control_auroc']:.4f}, {v['V3_control_share']:.4f}: {v['V3_control_reproduces_refit']}.", "",
          "## Gate", "", "```", json.dumps(r["gate"], indent=1), "```", "",
          f"Selection among five candidate sets on the same 20 questions: mean bootstrap optimism of the best-of-5 share difference {r['selection']['mean_bootstrap_optimism_of_best_of_5_share_diff'] * 100:+.2f} points "
          "(diagnostic, not corrected in the gate)."]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
