"""CPU. PREREGISTERED.md: offline ablation of groups h (provenance), i (structure), k (neighbour window), j (call-relative) and the lookahead reference O on the G1 one-word-per-call
traces of lctx_features2_20261111 (60 fit / 20 threshold MATH level-5 questions), primary rows = not EOS/EOT AND not written on their own call.
Learner, lambda rule, Boot statistics and the 2,000-resample cluster bootstrap (seed 23, target and lambda re-chosen per resample) are those of lctx_features2's fit_ablation.py (imported).
Usage: <python> fit_ablation3.py [--boot 2000] [--g2-traces <dir>] [--smoke]     (--smoke: 6 fit + 4 threshold questions, 50 resamples, V2 not checked, writes validation_smoke/)
Writes validation/ablation3_G1.{json,md}, validation/validity3.json, validation/gate3.json; never writes a rule (a decode stage needs its own document)."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("SLURM_CPUS_PER_TASK", "4"))
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
F2 = OUT.parent / "lctx_features2_20261111"
_spec = importlib.util.spec_from_file_location("f2_fit_ablation", F2 / "orchestration/fit_ablation.py")
FA = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(FA)                                     # puts F2/orchestration and the frozen src on sys.path
FL = FA.FL
sys.path.insert(0, str(HERE))
import features3_lib as F3  # noqa: E402
from check_gate3 import CANDS, CONTROLS, evaluate_gate  # noqa: E402

SEED, C_TARGET, V2_TOL = 23, 0.8, 1e-6
SIX, B = FL.FEATURES, FL.GROUPS["b"]
Hh, Ii, Kk, Jj, Oo = F3.H, F3.I, F3.K, F3.J, F3.O
HIKJ = Hh + Ii + Kk + Jj
SETS = {"conf": ["confidence"], "six": SIX, "six+b": SIX + B,
        "six+h": SIX + Hh, "six+i": SIX + Ii, "six+k": SIX + Kk, "six+j": SIX + Jj, "six+hikj": SIX + HIKJ,
        "six+b+k": SIX + B + Kk, "six+b+hikj": SIX + B + HIKJ,
        "six+O": SIX + Oo, "six+b+hikj+O": SIX + B + HIKJ + Oo}
NOISE_W = len(HIKJ)
assert NOISE_W == 17


def load_assets():
    adir = OUT / "assets"
    meta = json.loads((adir / "assets.json").read_text())
    for f, k in (("token_info.npz", "token_info_sha256"), ("prompt_ids.npz", "prompt_ids_sha256")):
        if hashlib.sha256((adir / f).read_bytes()).hexdigest() != meta[k]:
            raise SystemExit(f"{f} does not match assets.json")
    info = dict(np.load(adir / "token_info.npz"))
    prompts = dict(np.load(adir / "prompt_ids.npz"))
    return info, prompts, meta


def load_with_features(tdir: Path, ids: list[str], info, prompts) -> tuple[pd.DataFrame, dict]:
    frames, metas = [], {}
    for qid in ids:
        z = np.load(FL.trace_path(tdir, qid), allow_pickle=False)
        rows = {k: z[k] for k in z.files if z[k].ndim == 1}
        p = prompts[qid]
        feats, meta = F3.compute(rows, p, info)
        meta["prompt_len_equals_min_position"] = bool(int(rows["position"].min()) == len(p))
        metas[qid] = meta
        frames.append(pd.DataFrame({**rows, **feats}))
    return pd.concat(frames, ignore_index=True), metas


def design(df: pd.DataFrame, name: str, q: np.ndarray, seed: int) -> np.ndarray:
    if name == "six+noise17":
        return np.concatenate([df[SIX].to_numpy("float32"), np.random.default_rng(seed).random((len(df), NOISE_W), dtype=np.float32)], axis=1)
    if name == "six+hikj_shuf":
        return np.concatenate([df[SIX].to_numpy("float32"), FL.shuffle_within_question(df[HIKJ].to_numpy("float32"), q, seed)], axis=1)
    return df[SETS[name]].to_numpy("float32")


def perm_importance(model, X, y, feats, n_rep=3):
    rng = np.random.default_rng(SEED)
    base = FL.auroc(y, model.predict_proba(X)[:, 1])
    gmap = {"six": SIX, "b": B, "h": Hh, "i": Ii, "k": Kk, "j": Jj, "O": Oo}
    units = {**{f: [i] for i, f in enumerate(feats)}, **{f"group {g}": [i for i, f in enumerate(feats) if f in m] for g, m in gmap.items()}}
    out = {"base_auroc": base, "units": {}}
    for name, idx in units.items():
        if not idx:
            continue
        drops = []
        for _ in range(n_rep):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[:, idx] = X[np.ix_(perm, idx)]
            drops.append(base - FL.auroc(y, model.predict_proba(Xp)[:, 1]))
        out["units"][name] = float(np.mean(drops))
    return out


def v2_old_filter(fit_all: pd.DataFrame, thr_all: pd.DataFrame, template) -> dict:
    """The six control on the earlier ablations' row filter must reproduce lctx_features2's ablation_G1.json six numbers."""
    ref = json.loads((F2 / "validation/ablation_G1.json").read_text())["sets"]["six"]
    m = FL.fit_gbm(fit_all, template, SEED, SIX)
    s = m.predict_proba(thr_all[SIX].to_numpy("float32"))[:, 1]
    y, conf = thr_all["y"].to_numpy(), thr_all["confidence"].to_numpy()
    lam = FL.pooled_lambda(s, y, float(y[conf >= C_TARGET].mean()))
    auc, share = FL.auroc(y, s), float((s >= lam).mean())
    return {"auroc": auc, "share": share, "ref_auroc": ref["auroc"], "ref_share": ref["share"], "abs_diff": [abs(auc - ref["auroc"]), abs(share - ref["share"])],
            "ok": bool(abs(auc - ref["auroc"]) <= V2_TOL and abs(share - ref["share"]) <= V2_TOL)}


def validity(df: pd.DataFrame, metas: dict, v2: dict | None, gen_length=512) -> dict:
    v1_bad = [q for q, m in metas.items() if not (m["v1"]["positions_written"] == gen_length and m["v1"]["writes"] == gen_length and m["v1"]["written_equals_final"])]
    v4_prompt_bad = [q for q, m in metas.items() if not m["prompt_len_equals_min_position"]]
    viol = {}
    for side in ("l", "r"):
        near = df[f"_dist_{side}"] <= 2
        bad = near & (df["k_min_w2"] > df[f"nbr_conf_{side}"] + 1e-6)
        viol[f"k_min_w2_above_nbr_conf_{side}"] = int(bad.sum())
    viol["k_run_vs_conf"] = int(((df["k_run"] >= 1) != (df["confidence"] >= 0.5)).sum())
    rng = {"j_rank_frac": (df["j_rank_frac"] <= 0) | (df["j_rank_frac"] > 1), "j_block_written": (df["j_block_written"] < 0) | (df["j_block_written"] > 1),
           "i_brace_depth": df["i_brace_depth"] < 0}
    for f in ["h_in_prompt", "h_in_written", "h_bigram_prompt", "h_bigram_written", "h_num_novel", "i_digit", "i_after_eq", "i_after_op", "i_in_math", "o_same_next", "o_nbr_y_l", "o_nbr_y_r"]:
        rng[f] = ~(df[f].isna() | df[f].isin([0, 1]))
    v4 = {k: int(m.sum()) for k, m in rng.items() if m.sum()}
    return {"V1_questions": len(metas), "V1_failed": v1_bad, "V1_ok": not v1_bad,
            "V2": v2, "V2_ok": bool(v2 and v2["ok"]),
            "V3_violations": viol, "V3_ok": not any(viol.values()),
            "V4_range_violations": v4, "V4_prompt_length_mismatch": v4_prompt_bad, "V4_ok": not v4 and not v4_prompt_bad}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--traces", default=str(F2 / "records/traces_G1"))
    ap.add_argument("--g2-traces", default=None)
    ap.add_argument("--ids-json", default=str(F2 / "manifests/split.json"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    info, prompts, ameta = load_assets()
    split = json.loads(Path(args.ids_json).read_text())
    fit_ids, thr_ids = list(split["fit_questions"]), list(split["thr_questions"])
    B_ = args.boot
    if args.smoke:
        fit_ids, thr_ids, B_ = fit_ids[:6], thr_ids[:4], min(B_, 50)
    tdir = Path(args.traces)
    fit_df, mf = load_with_features(tdir, fit_ids, info, prompts)
    thr_df, mt = load_with_features(tdir, thr_ids, info, prompts)
    metas = {**mf, **mt}
    print(f"[ablation3] features computed for {len(metas)} questions in {time.time() - t_start:.0f}s", flush=True)
    template = FL.default_template()
    v2 = None if args.smoke else v2_old_filter(FL.select_rows(fit_df, "all"), FL.select_rows(thr_df, "all"), template)
    if v2 is not None:
        print(f"[ablation3] V2 six on the old filter: AUROC {v2['auroc']:.6f} (ref {v2['ref_auroc']:.6f}) share {v2['share']:.6f} (ref {v2['ref_share']:.6f}) ok={v2['ok']}", flush=True)
    valid = validity(pd.concat([fit_df, thr_df], ignore_index=True), metas, v2)
    fit, thr = FL.select_rows(fit_df, "not_written"), FL.select_rows(thr_df, "not_written")
    y, conf = thr["y"].to_numpy(), thr["confidence"].to_numpy()
    q, qf = pd.factorize(thr["example_id"])[0], pd.factorize(fit["example_id"])[0]
    nq = int(q.max()) + 1
    target = float(y[conf >= C_TARGET].mean())
    print(f"[ablation3] rows fit {len(fit)} thr {len(thr)}; C-0.8 target {target:.5f} share {(conf >= C_TARGET).mean():.4f}", flush=True)
    names = list(SETS) + CONTROLS
    scores, models, Xthr = {"conf_raw": conf.astype(np.float64)}, {}, {}
    for name in names:
        t0 = time.time()
        Xf = design(fit, name, qf, SEED) if name in CONTROLS else None
        models[name] = FL.fit_gbm(fit, template, SEED, SETS.get(name), X=Xf)
        Xthr[name] = design(thr, name, q, SEED + 1)
        scores[name] = models[name].predict_proba(Xthr[name])[:, 1]
        print(f"[ablation3] fitted {name} ({Xthr[name].shape[1]} columns) in {time.time() - t0:.0f}s", flush=True)
    boots = {k: FA.Boot(y, q, s) for k, s in scores.items()}
    point, ones = {}, np.ones(nq)
    for k, s in scores.items():
        lam = FL.pooled_lambda(s, y, target)
        auc_w, share_w, _ = boots[k].stats(ones, target)
        share_d = float((s >= lam).mean()) if lam is not None else 0.0
        assert abs(FL.auroc(y, s) - auc_w) < 1e-9 and abs(share_d - share_w) < 1e-9, k
        point[k] = {"auroc": auc_w, "share": share_d, "lambda": lam, "precision": float(y[s >= lam].mean()) if lam is not None else None}
    rng = np.random.default_rng(SEED)
    M = np.stack([np.bincount(rng.integers(0, nq, size=nq), minlength=nq) for _ in range(B_)])
    auc_b = {k: np.full(B_, np.nan) for k in scores}
    sh_b = {k: np.full(B_, np.nan) for k in scores}
    c08_b = np.zeros(B_)
    for b in range(B_):
        t_b, c08_b[b] = FA.c08(conf, y, q, M[b])
        for k in scores:
            auc_b[k][b], sh_b[k][b], _ = boots[k].stats(M[b], t_b)
    print(f"[ablation3] bootstrap {B_} done ({time.time() - t_start:.0f}s)", flush=True)
    res = {}
    for k in names:
        d = {"features": SETS.get(k, f"control ({Xthr[k].shape[1]} columns)"), "n_features": int(Xthr[k].shape[1]), **point[k],
             "auroc_ci95": FA.ci(auc_b[k]), "share_ci95": FA.ci(sh_b[k]), "share_minus_c08": FA.diff_block(point[k]["share"] - float((conf >= C_TARGET).mean()), sh_b[k] - c08_b)}
        for ref in ("six", "conf", "six+b"):
            d[f"auroc_diff_vs_{ref}"] = FA.diff_block(point[k]["auroc"] - point[ref]["auroc"], auc_b[k] - auc_b[ref])
            d[f"share_diff_vs_{ref}"] = FA.diff_block(point[k]["share"] - point[ref]["share"], sh_b[k] - sh_b[ref])
        res[k] = d
    imp = {}
    for k in ("six+hikj", "six+b+hikj", "six+O"):
        imp[k] = perm_importance(models[k], Xthr[k], y, SETS[k])
    print(f"[ablation3] importance done ({time.time() - t_start:.0f}s)", flush=True)
    ms = [m["ms_per_call"] for m in metas.values()]
    result = {"geometry": "G1", "rows_filter": "not EOS/EOT and not written on its own call", "smoke": bool(args.smoke), "assets": {k: ameta[k] for k in ("token_info_sha256", "prompt_ids_sha256")},
              "rows": {"fit": int(len(fit)), "thr": int(len(thr))}, "questions": {"fit": len(fit_ids), "thr": len(thr_ids)}, "n_boot": B_, "seed": SEED, "target_precision_c08": target,
              "c08": {"share": float((conf >= C_TARGET).mean()), "share_ci95": FA.ci(c08_b), "auroc_raw_confidence": point["conf_raw"]["auroc"]},
              "nan_share_thr": {f: float(thr[f].isna().mean()) for f in HIKJ + Oo},
              "sets": res, "importance": imp, "validity": valid, "cost": {"ms_per_call_hikj": float(np.mean(ms)), "ms_per_call_max_question": float(np.max(ms))}}
    if args.g2_traces:
        gdir = Path(args.g2_traces)
        have = [i for i in fit_ids + thr_ids if FL.trace_path(gdir, i).exists()]
        g2, _ = load_with_features(gdir, have, info, prompts)
        g2nw = FL.select_rows(g2, "not_written")
        aucs = {}
        for k in ("conf", "six", "six+b", "six+h", "six+i", "six+k", "six+j", "six+hikj", "six+b+hikj"):
            aucs[k] = FL.auroc(g2nw["y"].to_numpy(), models[k].predict_proba(g2nw[SETS[k]].to_numpy("float32"))[:, 1])
        lo = {f: float(fit[f].min()) for f in HIKJ}
        hi = {f: float(fit[f].max()) for f in HIKJ}
        result["shift_G2"] = {"questions": len(have), "rows_not_written": int(len(g2nw)), "y_mean_not_written": float(g2nw["y"].mean()),
                              "quantiles_G1_fit": FL.quantiles(fit, HIKJ), "quantiles_G2_not_written": FL.quantiles(g2nw, HIKJ),
                              "share_outside_G1_fit_range": {f: float(((g2nw[f] < lo[f]) | (g2nw[f] > hi[f])).mean()) for f in HIKJ}, "auroc_of_G1_models": aucs}
        print(f"[ablation3] G2 shift on {len(have)} questions done", flush=True)
    result["gate"] = evaluate_gate(result)
    odir = OUT / ("validation_smoke" if args.smoke else "validation")
    odir.mkdir(parents=True, exist_ok=True)
    FL.dump(odir / "validity3.json", valid)
    FL.dump(odir / "ablation3_G1.json", result)
    FL.dump(odir / "gate3.json", result["gate"])
    write_md(result, odir / "ablation3_G1.md")
    print(f"[ablation3] done in {time.time() - t_start:.0f}s; gate passes: {result['gate']['passes']} (best {result['gate']['best_set']}); "
          f"reference six+O share diff {100 * res['six+O']['share_diff_vs_six']['point']:+.2f} points", flush=True)


def write_md(r: dict, path: Path) -> None:
    f4 = lambda d: f"{d['point']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}]"
    f2 = lambda d: f"{100 * d['point']:+.2f} [{100 * d['ci95'][0]:+.2f}, {100 * d['ci95'][1]:+.2f}]"
    L = [f"# Feature ablation 3 (provenance h, structure i, neighbour window k, call-relative j; lookahead reference O){' SMOKE: not a result' if r['smoke'] else ''}", "",
         f"G1 traces, rows {r['rows_filter']}: fit {r['rows']['fit']} ({r['questions']['fit']} questions), threshold {r['rows']['thr']} ({r['questions']['thr']} questions). "
         f"C-0.8 pooled precision {r['target_precision_c08']:.5f}, share {r['c08']['share']:.4f}; raw-confidence AUROC {r['c08']['auroc_raw_confidence']:.4f}. "
         f"Cluster bootstrap over threshold questions, {r['n_boot']} resamples, target and lambda re-chosen per resample; differences paired. Share = admitted share at C-0.8's pooled precision.", "",
         "| set | cols | AUROC | share | precision | AUROC diff vs six | share diff vs six, points | share diff vs six+b, points | share diff vs C-0.8, points |", "|---|---|---|---|---|---|---|---|---|"]
    for k, d in r["sets"].items():
        L.append(f"| {k} | {d['n_features']} | {d['auroc']:.4f} | {d['share']:.4f} | {d['precision'] if d['precision'] is None else round(d['precision'], 4)} | {f4(d['auroc_diff_vs_six'])} | "
                 f"{f2(d['share_diff_vs_six'])} | {f2(d['share_diff_vs_six+b'])} | {f2(d['share_minus_c08'])} |")
    L += ["", f"Cost of groups h/i/k/j: {r['cost']['ms_per_call_hikj']:.3f} ms per call (mean over questions; max {r['cost']['ms_per_call_max_question']:.3f}).", "",
          "NaN share on threshold rows: " + ", ".join(f"{k} {v:.3f}" for k, v in r["nan_share_thr"].items() if v > 0), "", "## Permutation importance (AUROC drop, 3 repeats)", ""]
    for k, imp in r["importance"].items():
        L += [f"**{k}** (base {imp['base_auroc']:.4f}): " + ", ".join(f"{n} {v:.5f}" for n, v in sorted(imp["units"].items(), key=lambda kv: -kv[1])[:14]), ""]
    if "shift_G2" in r:
        g = r["shift_G2"]
        L += [f"## G2 shift (C-0.8 multi-write, {g['questions']} questions, {g['rows_not_written']} rows not written on their call, y mean {g['y_mean_not_written']:.3f})", "",
              "| feature | G1 fit p10 / p50 / p90 | G2 p10 / p50 / p90 | G2 share outside G1 fit range |", "|---|---|---|---|"]
        qs = lambda d: " / ".join(f"{d[p]:.3g}" for p in ("p10", "p50", "p90")) if "p50" in d else "-"
        for fe in HIKJ:
            L.append(f"| {fe} | {qs(g['quantiles_G1_fit'][fe])} | {qs(g['quantiles_G2_not_written'][fe])} | {g['share_outside_G1_fit_range'][fe]:.3f} |")
        L += ["", "AUROC of the G1-fitted models on G2 not-written rows: " + ", ".join(f"{k} {v:.4f}" for k, v in g["auroc_of_G1_models"].items()), ""]
    L += ["## Validity", "", "```", json.dumps(r["validity"], indent=1), "```", "", "## Gate", "", "```", json.dumps(r["gate"], indent=1), "```"]
    path.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
