"""Part A analysis (CPU): which signal predicts that a cached row's K/V have gone stale?

Usage: <abs venv python> partA_analyze.py [--dir DIR] [--out OUT] [--boot 100]  (reads DIR/g256 and DIR/g512 npz written by partA_collect.py)
Writes OUT/results_partA.md and OUT/summary_partA.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from partA_capture import AGES, CLASS_NAMES, HIST_BINS, NA, NB, NCLS, SIG_NAMES, THRESHOLDS  # noqa: E402

GROUPS = {"0-10": range(0, 11), "11-21": range(11, 22), "22-31": range(22, 32)}
SIGNALS = ["age", "E_cq", "E_fresh", "WM_acc", "WM_inst", "dist", "cos_heads", "cos_keys"]
LONG_NAMES = {"age": "age (calls since snapshot)", "E_cq": "exposure, cached query (accumulated)", "E_fresh": "exposure, fresh query (accumulated)",
              "WM_acc": "window mass on the row (accumulated)", "WM_inst": "window mass on the row (this call)", "dist": "distance to nearest new write (closer = staler)",
              "cos_heads": "Elastic trigger, 1 - cosine across heads (per call)", "cos_keys": "Elastic trigger, 1 - cosine across keys (per call)"}


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    y = y.astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(score)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def load_dir(d: Path):
    qs = {}
    for f in sorted(d.glob("*.npz")):
        z = np.load(f)
        qs[f.stem] = {k: z[k] for k in z.files}
    return qs


def long_table(qs: dict, group: range) -> dict[str, np.ndarray]:
    """One row per (question, sampled row, call, age) with a valid pair; signals averaged over the layers of `group`."""
    cols: dict[str, list] = {k: [] for k in ["qid", "y", "yk", "yv", "age", "cls", "E_fresh", "E_cq", "WM_acc", "WM_inst", "dist", "cos_heads", "cos_keys", "eos", "since", "call"]}
    layers = list(group)
    for qi, (name, z) in enumerate(qs.items()):
        sv = z["s_sv"][layers].mean(0)                         # (N, NA, 6)
        wm = z["s_wm_inst"][layers].mean(0)                    # (N,)
        call = z["samp_call"]
        with np.errstate(all="ignore"):
            ch = np.nanmean(z["cosh"][:, layers, :], axis=1)   # (calls, NA)
            ck = np.nanmean(z["cosk"][:, layers, :], axis=1)
        valid = z["s_valid"].astype(bool)                      # (N, NA)
        ni, ai = np.nonzero(valid)
        cols["qid"].append(np.full(len(ni), qi))
        cols["y"].append(sv[ni, ai, 0]); cols["yk"].append(sv[ni, ai, 1]); cols["yv"].append(sv[ni, ai, 2])
        cols["age"].append(np.array(AGES)[ai].astype(np.float32))
        cols["cls"].append(z["s_cls"][ni])
        cols["E_fresh"].append(sv[ni, ai, 3]); cols["E_cq"].append(sv[ni, ai, 4]); cols["WM_acc"].append(sv[ni, ai, 5])
        cols["WM_inst"].append(wm[ni])
        cols["dist"].append(z["s_dist"][ni, ai])
        cols["cos_heads"].append(1.0 - ch[call[ni], ai]); cols["cos_keys"].append(1.0 - ck[call[ni], ai])
        cols["eos"].append(z["s_is_eos"][ni]); cols["since"].append(z["s_since_written"][ni]); cols["call"].append(call[ni])
    return {k: np.concatenate(v) for k, v in cols.items()}


def signal_score(tab: dict, sig: str) -> np.ndarray:
    """Higher score = predicted staler."""
    if sig == "dist":
        return -tab["dist"]
    return tab[sig]


def cell_metrics(tab: dict, mask: np.ndarray, thr: float, within_age: bool = False) -> dict[str, float]:
    out = {}
    y = tab["y"][mask]
    lab = y > thr
    for sig in SIGNALS:
        s = signal_score(tab, sig)[mask]
        ok = np.isfinite(s)
        if sig == "age" and within_age:
            out[sig] = float("nan")
            continue
        if not within_age:
            out[sig] = auroc(s[ok], lab[ok]) if ok.sum() > 10 else float("nan")
        else:
            ages = tab["age"][mask]
            vals = []
            for a in AGES:
                m = ok & (ages == a)
                if m.sum() > 50:
                    vals.append(auroc(s[m], lab[m]))
            vals = [v for v in vals if np.isfinite(v)]
            out[sig] = float(np.mean(vals)) if vals else float("nan")
    return out


def spearman_cell(tab: dict, mask: np.ndarray) -> dict[str, float]:
    out = {}
    y = tab["y"][mask]
    for sig in SIGNALS:
        s = signal_score(tab, sig)[mask]
        ok = np.isfinite(s)
        out[sig] = spearman(s[ok], y[ok]) if ok.sum() > 10 else float("nan")
    return out


def stale_rates(qs: dict) -> dict:
    """Aggregate agg (n, n>0.05, n>0.10, sum drift) over questions, calls and the layers of each group -> [group][class][age]."""
    tot = np.zeros((len(GROUPS), NA, NCLS, 4))
    for z in qs.values():
        a = z["agg"]                                           # (calls, L, NA, NCLS, 4)
        for gi, (g, layers) in enumerate(GROUPS.items()):
            tot[gi] += a[:, list(layers)].sum((0, 1))
    return tot


def hist_quantiles(qs: dict) -> np.ndarray:
    h = sum(z["hist"] for z in qs.values())                    # (L, NA, NCLS, NB)
    centers = np.concatenate([[-5.5], 0.5 * (HIST_BINS[1:-2] + HIST_BINS[2:-1]), [1.5]])
    res = np.full((len(GROUPS), NA, NCLS, 3), np.nan)
    for gi, (g, layers) in enumerate(GROUPS.items()):
        hg = h[list(layers)].sum(0)
        for si in range(NA):
            for c in range(NCLS):
                v = hg[si, c].astype(float)
                if v.sum() < 20:
                    continue
                cum = np.cumsum(v) / v.sum()
                for qi_, q in enumerate((0.5, 0.9, 0.99)):
                    res[gi, si, c, qi_] = 10 ** centers[np.searchsorted(cum, q)]
    return res


def boot_multi(tab: dict, mask: np.ndarray, thr: float, sigs: list[str], nb: int, rng) -> dict[str, tuple[float, float]]:
    """Bootstrap over questions: 95% intervals of each signal's AUROC and of the differences E_cq-age, E_cq-dist, E_fresh-age, E_cq-WM_acc (same resamples)."""
    qids = np.unique(tab["qid"][mask])
    scores = {s: signal_score(tab, s) for s in sigs}
    lab = tab["y"] > thr
    idx_by_q = {q: np.nonzero(mask & (tab["qid"] == q))[0] for q in qids}
    vals: dict[str, list] = {s: [] for s in sigs}
    for k in ("E_cq-age", "E_cq-dist", "E_fresh-age", "E_cq-WM_acc"):
        vals[k] = []
    for _ in range(nb):
        pick = rng.choice(qids, len(qids), replace=True)
        ii = np.concatenate([idx_by_q[q] for q in pick])
        a = {}
        for s in sigs:
            ok = np.isfinite(scores[s][ii])
            a[s] = auroc(scores[s][ii][ok], lab[ii][ok])
            vals[s].append(a[s])
        vals["E_cq-age"].append(a["E_cq"] - a["age"])
        vals["E_cq-dist"].append(a["E_cq"] - a["dist"])
        vals["E_fresh-age"].append(a["E_fresh"] - a["age"])
        vals["E_cq-WM_acc"].append(a["E_cq"] - a["WM_acc"])
    out = {}
    for k, v in vals.items():
        v = [x for x in v if np.isfinite(x)]
        out[k] = (float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))) if v else (float("nan"), float("nan"))
    return out


def cross_layer(qs: dict, thr: float = 0.10, nmax: int = 300000) -> dict:
    """AUROC of the cached-query exposure measured at layer a for drift > thr at the deeper layer groups (what a score taken at one layer can say about the layers above)."""
    sv_all, ages = [], []
    for z in qs.values():
        valid = z["s_valid"].astype(bool)
        ni, ai = np.nonzero(valid)
        sv_all.append(z["s_sv"][:, ni, ai, :])
        ages.append(np.array(AGES)[ai])
    sv = np.concatenate(sv_all, axis=1)
    rng = np.random.default_rng(1)
    sel = rng.choice(sv.shape[1], min(sv.shape[1], nmax), replace=False)
    sv = sv[:, sel]
    L = sv.shape[0]
    res = {}
    for a in (1, 2, 4, 6, 8, 10, 12, 16, 20, 24):
        if a >= L - 1:
            continue
        row = {}
        for gname, layers in GROUPS.items():
            deeper = [l for l in layers if l > a]
            if not deeper:
                continue
            drift = sv[deeper, :, 0].mean(0)
            row[gname] = auroc(sv[a, :, 4], drift > thr)
        row["all deeper"] = auroc(sv[a, :, 4], sv[a + 1:, :, 0].mean(0) > thr)
        res[a] = row
    return res


def heldout_logistic(tab: dict, thr: float, folds: int = 5, seed: int = 0) -> dict[str, float]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    qids = np.unique(tab["qid"])
    folds = max(2, min(folds, len(qids)))
    rng = np.random.default_rng(seed)
    fold_of = {q: i % folds for i, q in enumerate(rng.permutation(qids))}
    f = np.array([fold_of[q] for q in tab["qid"]])
    y = tab["y"] > thr
    onehot = np.stack([(tab["cls"] == c).astype(float) for c in range(NCLS)], 1)
    feats = {
        "age": np.log(tab["age"])[:, None],
        "class": onehot,
        "dist": np.log1p(tab["dist"])[:, None],
        "E_cq": np.log(tab["E_cq"] + 1e-6)[:, None],
        "WM_acc": np.log(tab["WM_acc"] + 1e-6)[:, None],
        "cos_heads": np.nan_to_num(tab["cos_heads"], nan=0.0)[:, None],
    }
    sets = {"age": ["age"], "age+dist": ["age", "dist"], "age+E_cq": ["age", "E_cq"], "age+WM_acc": ["age", "WM_acc"], "class+age": ["class", "age"],
            "class+age+dist": ["class", "age", "dist"], "class+age+E_cq": ["class", "age", "E_cq"], "class+age+dist+E_cq+WM_acc": ["class", "age", "dist", "E_cq", "WM_acc"]}
    out = {}
    for name, keys in sets.items():
        X = np.concatenate([feats[k] for k in keys], 1)
        pred = np.zeros(len(y))
        for k in range(folds):
            tr, te = f != k, f == k
            if not te.any() or not tr.any() or len(np.unique(y[tr])) < 2:
                continue
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=300, C=1.0).fit(sc.transform(X[tr]), y[tr])
            pred[te] = m.decision_function(sc.transform(X[te]))
        out[name] = auroc(pred, y)
    return out


def per_layer_auroc(qs: dict, thr: float, nmax: int = 400000) -> dict[str, np.ndarray]:
    """AUROC per layer, pooled over classes and ages (valid pairs; cached classes only: everything except 'written' rows are included)."""
    L = next(iter(qs.values()))["s_sv"].shape[0]
    cols = {"y": [], "age": [], "dist": [], "E_cq": [], "E_fresh": [], "WM_acc": []}
    sv_all = []
    for z in qs.values():
        valid = z["s_valid"].astype(bool)
        ni, ai = np.nonzero(valid)
        sv_all.append(z["s_sv"][:, ni, ai, :])                  # (L, n, 6)
        cols["age"].append(np.array(AGES)[ai]); cols["dist"].append(z["s_dist"][ni, ai])
    sv = np.concatenate(sv_all, axis=1)
    age, dist = np.concatenate(cols["age"]), np.concatenate(cols["dist"])
    n = sv.shape[1]
    rng = np.random.default_rng(0)
    sel = rng.choice(n, min(n, nmax), replace=False)
    sv, age, dist = sv[:, sel], age[sel], dist[sel]
    out = {k: np.zeros(L) for k in ["age", "dist", "E_cq", "E_fresh", "WM_acc"]}
    for l in range(L):
        lab = sv[l, :, 0] > thr
        out["age"][l] = auroc(age.astype(float), lab)
        out["dist"][l] = auroc(-dist, lab)
        out["E_cq"][l] = auroc(sv[l, :, 4], lab)
        out["E_fresh"][l] = auroc(sv[l, :, 3], lab)
        out["WM_acc"][l] = auroc(sv[l, :, 5], lab)
    return out


def trigger_analysis(qs: dict, gamma: float = 0.9) -> dict:
    """Per call: does Elastic's per-call trigger signal track the fraction of drifted rows? Rows = every valid row of the call (all classes)."""
    res = {}
    for kind in ("cosh", "cosk"):
        per_age = {}
        for si, s in enumerate(AGES):
            sp_layers = np.full(32, np.nan)
            xs, ys, frac_all, trig, mincos = [], [], [], [], []
            for z in qs.values():
                a = z["agg"][:, :, si]                          # (calls, L, NCLS, 4)
                n = a[..., 0].sum(2)                            # (calls, L)
                with np.errstate(all="ignore"):
                    f10 = a[..., 2].sum(2) / n
                cos = z[kind][:, :, si]                        # (calls, L)
                ok = np.isfinite(cos).all(1) & (n[:, 0] > 20)
                xs.append(1 - cos[ok]); ys.append(f10[ok])
            if not xs:
                continue
            X, Y = np.concatenate(xs), np.concatenate(ys)      # (m, L)
            L = X.shape[1]
            for l in range(L - 1):
                deeper = np.nanmean(Y[:, l + 1:], axis=1)
                sp_layers[l] = spearman(X[:, l], deeper)
            glob_frac = np.nanmean(Y, axis=1)
            first_layer = np.array([np.argmax(r) if (r > 1 - gamma).any() else -1 for r in X])    # first layer with cos < gamma
            fire = first_layer >= 0
            per_age[s] = dict(
                spearman_by_layer=[None if not np.isfinite(v) else float(v) for v in sp_layers],
                spearman_mean_layers_0_30=float(np.nanmean(sp_layers)),
                spearman_mincos_vs_global_frac=spearman(X.max(1), glob_frac),
                spearman_meancos_vs_global_frac=spearman(X.mean(1), glob_frac),
                fire_rate=float(fire.mean()), fire_first_layer_median=float(np.median(first_layer[fire])) if fire.any() else None,
                global_stale_frac_mean=float(np.nanmean(glob_frac)), global_stale_frac_when_fire=float(np.nanmean(glob_frac[fire])) if fire.any() else None,
                global_stale_frac_when_quiet=float(np.nanmean(glob_frac[~fire])) if (~fire).any() else None,
                auroc_fire_vs_stale_gt5pct=auroc(fire.astype(float) + 1e-6 * X.max(1), glob_frac > 0.05),
                auroc_maxdev_vs_stale_gt5pct=auroc(X.max(1), glob_frac > 0.05),
                n_calls=int(len(X)),
                cos_median_by_layer=[float(v) for v in np.median(1 - X, axis=0)])
        res[kind] = per_age
    return res


def fmt(v, d=3):
    return "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{d}f}"


def analyse_length(gen: int, dirpath: Path, nboot: int, md: list[str], summary: dict):
    qs = load_dir(dirpath / f"g{gen}")
    if not qs:
        return
    summ = [json.loads(x) for x in (dirpath / f"g{gen}/summary.jsonl").read_text().splitlines()] if (dirpath / f"g{gen}/summary.jsonl").exists() else []
    md.append(f"\n## Generation length {gen}: {len(qs)} questions (dense Conf-0.9, block 32; {sum(len(z['call_t']) for z in qs.values())} calls)\n")
    if summ:
        md.append(f"Decoded correct {sum(r['correct'] for r in summ)}/{len(summ)}; mean calls {np.mean([r['nfe'] for r in summ]):.1f}; mean prompt length {np.mean([r['prompt_len'] for r in summ]):.0f}.\n")
    out = summary.setdefault(f"g{gen}", {})
    # ---- stale rates
    tot = stale_rates(qs)
    md.append("### Stale rates (share of valid pairs with relative L2 of K/V above the threshold), by layer group, row class and snapshot age\n")
    md.append("| layer group | class | age | pairs | share > 0.05 | share > 0.10 | mean drift |\n|---|---|---|---|---|---|---|")
    rates = {}
    for gi, g in enumerate(GROUPS):
        for c in range(NCLS):
            for si, s in enumerate(AGES):
                n, n5, n10, sd = tot[gi, si, c]
                if n < 50:
                    continue
                md.append(f"| {g} | {CLASS_NAMES[c]} | {s} | {int(n)} | {n5 / n:.3f} | {n10 / n:.3f} | {sd / n:.4f} |")
                rates[f"{g}|{CLASS_NAMES[c]}|{s}"] = dict(n=int(n), s05=n5 / n, s10=n10 / n, mean=sd / n)
    out["stale_rates"] = rates
    q = hist_quantiles(qs)
    md.append("\n### Drift quantiles (relative L2 of K/V; median / 90th / 99th percentile; histogram bins of 0.125 in log10)\n")
    md.append("| layer group | class | age | median | p90 | p99 |\n|---|---|---|---|---|---|")
    qd = {}
    for gi, g in enumerate(GROUPS):
        for c in range(NCLS):
            for si, s in enumerate(AGES):
                if np.isnan(q[gi, si, c, 0]):
                    continue
                md.append(f"| {g} | {CLASS_NAMES[c]} | {s} | {q[gi, si, c, 0]:.4f} | {q[gi, si, c, 1]:.4f} | {q[gi, si, c, 2]:.4f} |")
                qd[f"{g}|{CLASS_NAMES[c]}|{s}"] = q[gi, si, c].tolist()
    out["drift_quantiles"] = qd
    # ---- AUROC / Spearman
    rng = np.random.default_rng(0)
    tabs = {g: long_table(qs, layers) for g, layers in GROUPS.items()}
    out["auroc"], out["spearman"], out["auroc_within_age"] = {}, {}, {}
    for thr in THRESHOLDS:
        md.append(f"\n### AUROC of each signal for 'K/V drift > {thr}' (pooled over ages unless stated; 0.5 = no information; layer-group means)\n")
        md.append("| layer group | class (n pairs, stale share) | " + " | ".join(SIGNALS) + " |\n|---|---|" + "---|" * len(SIGNALS))
        for g, tab in tabs.items():
            for c in [-1] + list(range(NCLS)):
                mask = np.ones(len(tab["y"]), bool) if c < 0 else tab["cls"] == c
                if c == 1:
                    pass
                if mask.sum() < 200:
                    continue
                m = cell_metrics(tab, mask, thr)
                lab = tab["y"][mask] > thr
                name = "all classes" if c < 0 else CLASS_NAMES[c]
                md.append(f"| {g} | {name} ({int(mask.sum())}, {lab.mean():.3f}) | " + " | ".join(fmt(m[s]) for s in SIGNALS) + " |")
                out["auroc"][f"{thr}|{g}|{name}"] = m
        md.append(f"\nWithin-age AUROC (mean over the five ages; removes the age confound) for drift > {thr}:\n")
        md.append("| layer group | class | " + " | ".join(s for s in SIGNALS if s != "age") + " |\n|---|---|" + "---|" * (len(SIGNALS) - 1))
        for g, tab in tabs.items():
            for c in [-1] + list(range(NCLS)):
                mask = np.ones(len(tab["y"]), bool) if c < 0 else tab["cls"] == c
                if mask.sum() < 200:
                    continue
                m = cell_metrics(tab, mask, thr, within_age=True)
                name = "all classes" if c < 0 else CLASS_NAMES[c]
                md.append(f"| {g} | {name} | " + " | ".join(fmt(m[s]) for s in SIGNALS if s != "age") + " |")
                out["auroc_within_age"][f"{thr}|{g}|{name}"] = m
    md.append("\n### Spearman correlation of each signal with the continuous drift (K/V relative L2), pooled over ages\n")
    md.append("| layer group | class | " + " | ".join(SIGNALS) + " |\n|---|---|" + "---|" * len(SIGNALS))
    for g, tab in tabs.items():
        for c in [-1] + list(range(NCLS)):
            mask = np.ones(len(tab["y"]), bool) if c < 0 else tab["cls"] == c
            if mask.sum() < 200:
                continue
            m = spearman_cell(tab, mask)
            name = "all classes" if c < 0 else CLASS_NAMES[c]
            md.append(f"| {g} | {name} | " + " | ".join(fmt(m[s]) for s in SIGNALS) + " |")
            out["spearman"][f"{g}|{name}"] = m
    # ---- bootstrap CIs for the headline cells
    md.append(f"\n### Headline AUROCs with 95% intervals (bootstrap over questions, {nboot} resamples; all classes and ages pooled) and paired differences\n")
    md.append("| layer group | threshold | age | E_cq | E_fresh | WM_acc | dist | cos_heads | E_cq - age | E_cq - dist | E_fresh - age | E_cq - WM_acc |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    out["boot"] = {}
    sigs = ["age", "E_cq", "E_fresh", "WM_acc", "dist", "cos_heads"]
    for g, tab in tabs.items():
        mask = np.ones(len(tab["y"]), bool)
        for thr in THRESHOLDS:
            ci = boot_multi(tab, mask, thr, sigs, nboot, rng)
            cells = []
            for sig in sigs:
                sc = signal_score(tab, sig)
                ok = np.isfinite(sc)
                a = auroc(sc[ok], (tab["y"] > thr)[ok])
                cells.append(f"{a:.3f} [{ci[sig][0]:.3f}, {ci[sig][1]:.3f}]")
                out["boot"][f"{g}|{thr}|{sig}"] = [a, *ci[sig]]
            for k in ("E_cq-age", "E_cq-dist", "E_fresh-age", "E_cq-WM_acc"):
                cells.append(f"[{ci[k][0]:+.3f}, {ci[k][1]:+.3f}]")
                out["boot"][f"{g}|{thr}|{k}"] = list(ci[k])
            md.append(f"| {g} | {thr} | " + " | ".join(cells) + " |")
    # ---- held-out logistic
    md.append("\n### Held-out combination (5-fold by question; logistic regression on log features; AUROC for drift > 0.10)\n")
    md.append("| layer group | " + " | ".join(["age", "age+dist", "age+E_cq", "age+WM_acc", "class+age", "class+age+dist", "class+age+E_cq", "class+age+dist+E_cq+WM_acc"]) + " |\n|---|" + "---|" * 8)
    out["heldout"] = {}
    for g, tab in tabs.items():
        r = heldout_logistic(tab, 0.10)
        out["heldout"][g] = r
        md.append(f"| {g} | " + " | ".join(fmt(r[k]) for k in ["age", "age+dist", "age+E_cq", "age+WM_acc", "class+age", "class+age+dist", "class+age+E_cq", "class+age+dist+E_cq+WM_acc"]) + " |")
    # ---- per layer
    md.append("\n### AUROC per layer (all classes and ages pooled, drift > 0.10; random 400k pairs)\n")
    pl = per_layer_auroc(qs, 0.10)
    md.append("| layer | " + " | ".join(pl) + " |\n|---|" + "---|" * len(pl))
    for l in range(len(pl["age"])):
        md.append(f"| {l} | " + " | ".join(fmt(pl[k][l]) for k in pl) + " |")
    out["per_layer_auroc"] = {k: v.tolist() for k, v in pl.items()}
    # ---- cross-layer
    cx = cross_layer(qs)
    out["cross_layer"] = {str(k): v for k, v in cx.items()}
    md.append("\n### Exposure scored at ONE layer a vs drift (> 0.10) at the layers above it (E_cq at layer a; pooled classes and ages)\n")
    md.append("| score layer a | drift 0-10 (layers > a) | drift 11-21 | drift 22-31 | all layers > a |\n|---|---|---|---|---|")
    for a_, row in cx.items():
        md.append(f"| {a_} | {fmt(row.get('0-10'))} | {fmt(row.get('11-21'))} | {fmt(row.get('22-31'))} | {fmt(row.get('all deeper'))} |")
    # ---- trigger
    trig = trigger_analysis(qs)
    out["trigger"] = trig
    md.append("\n### Elastic's per-call trigger signal vs the fraction of drifted rows (dense decode: fresh attention now vs fresh attention s calls ago)\n")
    md.append("| cosine axis | age s | calls | fire rate (cos < 0.9 at some layer) | median first layer | stale share (drift>0.10) when fires / quiet | Spearman 1-cos vs deeper-layer stale share (mean over layers) | Spearman max(1-cos) vs global stale share | AUROC fire vs >5% rows stale |\n|---|---|---|---|---|---|---|---|---|")
    for kind, name in (("cosh", "heads (as coded)"), ("cosk", "keys")):
        for s, r in trig.get(kind, {}).items():
            md.append(f"| {name} | {s} | {r['n_calls']} | {fmt(r['fire_rate'])} | {r['fire_first_layer_median']} | {fmt(r['global_stale_frac_when_fire'])} / {fmt(r['global_stale_frac_when_quiet'])} "
                      f"| {fmt(r['spearman_mean_layers_0_30'])} | {fmt(r['spearman_mincos_vs_global_frac'])} | {fmt(r['auroc_fire_vs_stale_gt5pct'])} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(HERE.parent / "partA"))
    ap.add_argument("--out", default=str(HERE.parent))
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--gens", default="256,512")
    a = ap.parse_args()
    md = ["# Part A results: what predicts that a cached row's K/V have gone stale\n", "(generated by orchestration/partA_analyze.py; dense Conf-0.9 decodes of 40 MATH train questions)\n"]
    summary: dict = {}
    nf = Path(HERE.parent / "validation/noise_floor.json")
    if nf.exists():
        n = json.loads(nf.read_text())
        summary["noise_floor"] = n
        md.append(f"## Numerical noise floor\nIdentical repeated dense forward: max abs difference of K and V over all layers = {n['identical_repeat']['max_abs_difference_over_all_layers']}. "
                  f"K/V of a random 64-row subset recomputed from the same layer inputs (what a compact pass computes): mean relative L2 {n['row_subset_recompute']['mean_relL2']:.5f}, "
                  f"p99 {n['row_subset_recompute']['p99_relL2']:.5f}, max {n['row_subset_recompute']['max_relL2']:.5f}; bf16 spacing 2^-8 = {n['bf16_eps']:.5f}. The thresholds 0.05 and 0.10 are "
                  f"{0.05 / max(n['row_subset_recompute']['p99_relL2'], 1e-9):.0f}x and {0.10 / max(n['row_subset_recompute']['p99_relL2'], 1e-9):.0f}x the p99 numerical floor.\n")
    for gen in [int(x) for x in a.gens.split(",")]:
        analyse_length(gen, Path(a.dir), a.boot, md, summary)
    Path(a.out, "results_partA.md").write_text("\n".join(md) + "\n")
    Path(a.out, "summary_partA.json").write_text(json.dumps(summary, indent=1, default=float) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
