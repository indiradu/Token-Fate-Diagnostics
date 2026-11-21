"""CPU (login node). Fits the three learned word locks of the comparison grid on the G1 traces of lctx_features2_20261111 (60 fit questions, lambda on the 20 threshold questions),
each exactly as its offline ablation defined it, and checks the live features3 module against the offline one on all 80 traces.

  Lctx-six   six inputs                         rows: not EOS/EOT                         (features2 recipe; its offline numbers: ablation_G1 'six')
  Lctx-F2    six + groups a-g (30 inputs)        rows: not EOS/EOT                         (features2 'six+all')
  Lctx-F3    six + groups h, i, k, j (23 inputs) rows: not EOS/EOT, not written on its call (features3's pre-registered best set 'six+hikj')
Learner HistGradientBoosting (200 rounds, lr 0.1, seed 23). lambda = the lowest score whose pooled precision on the threshold rows reaches C-0.8's pooled precision on the same rows
(features2_lib.pooled_lambda; the original Lctx-0.8 was fitted to the same kind of target).
Checks (exit 1 if any fails): V-F3 = lock_features3 driven call by call equals features3_lib.compute on every row of all 80 traces; V-REP = each rule's AUROC and admitted share on the
threshold rows equal its offline ablation's numbers (to 1e-9).
Writes rules/<name>/<name>.pkl, rules/rules.json (sha256, lambda, target, features, fit rows, AUROC, share), validation/fit_rules.json.
Usage: <python> fit_rules.py"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
F2DIR, F3DIR = ROOT / "results/lctx_features2_20261111", ROOT / "results/lctx_features3_20261115"
sys.path.insert(0, str(ROOT / "src"))
from regret_remasking import lock_features3 as LF3  # noqa: E402  (imported first: the features2 helpers below put their frozen source/ on sys.path, which lacks this module)
from regret_remasking.context_rules import ContextRule  # noqa: E402
_spec = importlib.util.spec_from_file_location("fit_ablation3_grid", F3DIR / "orchestration/fit_ablation3.py")
A = importlib.util.module_from_spec(_spec)
sys.path.insert(0, str(F3DIR / "orchestration"))
_spec.loader.exec_module(A)
FL = A.FL

SEED = 23
RULES = {"Lctx-six": ("all", A.SIX, F2DIR / "validation/ablation_G1.json", "six"),
         "Lctx-F2": ("all", A.SIX + FL.EXTRA + FL.NEW, F2DIR / "validation/ablation_G1.json", "six+all"),
         "Lctx-F3": ("not_written", A.SIX + A.HIKJ, F3DIR / "validation/ablation3_G1.json", "six+hikj")}


def v_f3(df, info, prompts) -> dict:
    """lock_features3 driven call by call (the decoder's path) against the offline columns already in df (features3_lib.compute)."""
    bad = {}
    for qid, g in df.groupby("example_id", sort=False):
        p = prompts[qid]
        call, r = g["call"].to_numpy(np.int64), g["position"].to_numpy(np.int64) - len(p)
        st = LF3.F3State(p, info)
        got = {f: np.full(len(g), np.nan, np.float32) for f in LF3.FEATURES3}
        for c in np.unique(call):
            idx = np.nonzero(call == c)[0]
            idx = idx[np.argsort(r[idx], kind="stable")]
            f = st.features(r[idx], g["token"].to_numpy()[idx], g["confidence"].to_numpy()[idx])
            for k in LF3.FEATURES3:
                got[k][idx] = f[k]
            w = idx[g["written_now"].to_numpy(bool)[idx]]
            st.update(r[w], g["token"].to_numpy()[w])
        for k in LF3.FEATURES3:
            if not np.array_equal(got[k], g[k].to_numpy(np.float32), equal_nan=True):
                bad.setdefault(qid, []).append(k)
    return {"questions": int(df["example_id"].nunique()), "rows": int(len(df)), "mismatches": bad, "ok": not bad}


def main() -> int:
    info, prompts, _ = A.load_assets()
    split = json.loads((F2DIR / "manifests/split.json").read_text())
    tdir = F2DIR / "records/traces_G1"
    fit_all, _ = A.load_with_features(tdir, split["fit_questions"], info, prompts)
    thr_all, _ = A.load_with_features(tdir, split["thr_questions"], info, prompts)
    checks = {"V_F3": v_f3(__import__("pandas").concat([fit_all, thr_all]), info, prompts)}
    print("[V-F3]", {k: v for k, v in checks["V_F3"].items() if k != "mismatches"}, "mismatching questions:", len(checks["V_F3"]["mismatches"]), flush=True)
    tmpl = FL.default_template()
    meta, rep = {}, {}
    for name, (filt, feats, ref_path, ref_key) in RULES.items():
        fit, thr = FL.select_rows(fit_all, filt), FL.select_rows(thr_all, filt)
        y, conf = thr["y"].to_numpy(), thr["confidence"].to_numpy()
        target = float(y[conf >= 0.8].mean())
        model = FL.fit_gbm(fit, tmpl, SEED, feats)
        s = model.predict_proba(thr[feats].to_numpy("float32"))[:, 1]
        lam = FL.pooled_lambda(s, y, target)
        if lam is None:
            print(f"[{name}] no lambda reaches the target", flush=True)
            return 1
        auc, share, prec = FL.auroc(y, s), float((s >= lam).mean()), float(y[s >= lam].mean())
        ref = json.loads(Path(ref_path).read_text())["sets"][ref_key]
        rep[name] = {"auroc": auc, "share": share, "ref_auroc": ref["auroc"], "ref_share": ref["share"], "ok": abs(auc - ref["auroc"]) < 1e-9 and abs(share - ref["share"]) < 1e-9}
        d = OUT / "rules" / name
        d.mkdir(parents=True, exist_ok=True)
        rule = ContextRule("gbm", lam, features=list(feats), payload=model, name=name)
        path = d / f"{name}.pkl"
        rule.save(path)
        back = ContextRule.load(path)
        assert np.array_equal(back.scores({f: thr[f].to_numpy() for f in feats}), s.astype(np.float64)) or np.allclose(back.scores({f: thr[f].to_numpy() for f in feats}), s)
        meta[name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "lambda": lam, "target_precision": target, "row_filter": filt, "features": list(feats),
                      "fit_rows": int(len(fit)), "thr_rows": int(len(thr)), "auroc_thr": auc, "share_thr": share, "precision_thr": prec, "offline_reference": f"{ref_path.relative_to(ROOT)}:{ref_key}"}
        print(f"[{name}] lambda {lam:.6f} target {target:.5f} AUROC {auc:.4f} share {share:.4f} (offline {ref['auroc']:.4f} / {ref['share']:.4f}) features {len(feats)}", flush=True)
    checks["V_REP"] = rep
    (OUT / "rules/rules.json").write_text(json.dumps({"rules": meta, "seed": SEED, "traces": str(tdir.relative_to(ROOT)), "split": str((F2DIR / "manifests/split.json").relative_to(ROOT))}, indent=1) + "\n")
    (OUT / "validation").mkdir(exist_ok=True)
    (OUT / "validation/fit_rules.json").write_text(json.dumps(checks, indent=1) + "\n")
    ok = checks["V_F3"]["ok"] and all(v["ok"] for v in rep.values())
    print("[fit_rules]", "OK" if ok else "CHECK FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
