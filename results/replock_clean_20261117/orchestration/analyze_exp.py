"""Analyses of the follow-up experiments. Usage: python analyze_exp.py thr|abl|par|noise  -> prints markdown and writes results_<name>.md in results/replock_clean_20261117/.
thr: threshold sweep: per set and threshold, calls per block m, word-lock factor (dense vs the B0 point), cache factors of S32 and R0, the formula S_cache(m) = m*tau_d/(tau_d + (m-1)*tau_c) with measured tau_d and tau_c = the cached call time, and the interaction I = S_cache(thr) / S_cache(B0).
abl: R0 ablation: per set and arm, speed vs dense (sum of latencies), calls, correct and the paired change vs dense and vs R0.
par: extra-prompt parity of R0 against dense Conf-0.9, per set and pooled (also pooled with the Study 5 LLaDA-8B cells).
noise: MoE noise-matched control (premise P3)."""
import json
import sys
from pathlib import Path

import numpy as np

REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
rng = np.random.default_rng(0)


def load(prefix, tag, ds):
    p = REC / f"{prefix}_{tag}_{ds}.scored.jsonl"
    p = p if p.exists() else REC / f"{prefix}_{tag}_{ds}.jsonl"
    out = []
    if p.exists():
        out = [json.loads(l) for l in open(p)]
    return out


def ratio(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [a[i].sum() / b[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
    return a.sum() / b.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def delta(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    bs = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{int(d.sum()):+d} [{lo:+.0f}, {hi:+.0f}]" + ("" if lo > 0 or hi < 0 else " (incl. 0)")


def thr():
    out = ["# Threshold sweep: the cache factor as the word lock gets stronger (LLaDA-8B)", "m = calls per 32-block of the dense decode; tau_d = dense ms per call; tau_c = 33.6 ms (cached call, from the cost benchmark).",
           "Formula: S_cache(m) = m*tau_d / (tau_d + (m-1)*tau_c) (one dense call per block, the rest cached; no refresh cost, no call inflation). I = S_cache(threshold) / S_cache(B0 point): the interaction of the multiplication rule (1 = the cache factor does not depend on the word lock).", ""]
    for ds, gen in (("he164", 512), ("gsm_held", 256)):
        recs = load("thr", "llada8b", ds)
        if not recs:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["thr"], {}).setdefault(r["arm"], {})[r["example_id"]] = r
        ths = sorted(by)
        b0 = by[max(ths)] if max(ths) >= 1.0 else None
        out += [f"## {ds}", "| threshold | prompts | m (calls/block) | S32 calls/block | word-lock factor (dense over B0) | S32 cache factor | R0 cache factor | formula S_cache(m) | I (R0) | R0 total over B0 | correct dense / S32 / R0 |", "|---|---|---|---|---|---|---|---|---|---|---|"]
        for t in ths:
            a = by[t]
            ids = sorted(set(a["DENSE"]) & set(a["S32"]) & set(a["R0"]))
            if not ids:
                continue
            nb = gen / 32
            m = np.mean([a["DENSE"][i]["nfe"] for i in ids]) / nb
            m32 = np.mean([a["S32"][i]["nfe"] for i in ids]) / nb
            td = 1000 * sum(a["DENSE"][i]["latency_s"] for i in ids) / sum(a["DENSE"][i]["nfe"] for i in ids)
            tc = 33.6
            pred = m * td / (td + (m - 1) * tc)
            s32 = ratio([a["DENSE"][i]["latency_s"] for i in ids], [a["S32"][i]["latency_s"] for i in ids])[0]
            r0 = ratio([a["DENSE"][i]["latency_s"] for i in ids], [a["R0"][i]["latency_s"] for i in ids])
            wl = tot = inter = None
            if b0:
                jb = sorted(set(b0["DENSE"]) & set(b0["R0"]) & set(ids))
                if jb:
                    wl = ratio([b0["DENSE"][i]["latency_s"] for i in jb], [a["DENSE"][i]["latency_s"] for i in jb])[0]
                    tot = ratio([b0["DENSE"][i]["latency_s"] for i in jb], [a["R0"][i]["latency_s"] for i in jb])[0]
                    r0b = ratio([b0["DENSE"][i]["latency_s"] for i in jb], [b0["R0"][i]["latency_s"] for i in jb])[0]
                    inter = r0[0] / r0b
            cs = lambda arm: "pending" if any(a[arm][i].get("correct") is None for i in ids) else str(int(sum(a[arm][i]["correct"] for i in ids)))
            out.append(f"| {t} | {len(ids)} | {m:.1f} | {m32:.1f} | {'–' if wl is None else f'{wl:.1f}x'} | {s32:.2f}x | {r0[0]:.2f}x | {pred:.2f}x | {'–' if inter is None else f'{inter:.2f}'} | {'–' if tot is None else f'{tot:.1f}x'} | {cs('DENSE')} / {cs('S32')} / {cs('R0')} |")
        out.append("")
    return out


def abl():
    out = ["# R0 ablation (LLaDA-8B, Conf-0.9)", "Speed = dense time / arm time over the prompts; paired change in correct answers vs dense and vs R0 (95% bootstrap over prompts).", ""]
    for ds in ("he164", "kod_held", "gsm_held"):
        recs = load("abl", "llada8b", ds)
        if not recs:
            continue
        by = {}
        for r in recs:
            by.setdefault(r["arm"], {})[r["example_id"]] = r
        ids = sorted(set.intersection(*[set(v) for v in by.values()]))
        out += [f"## {ds} (n={len(ids)})", "| arm | schedule | speed vs dense | calls | correct | Δ vs dense | Δ vs R0 |", "|---|---|---|---|---|---|---|"]
        for a in ["DENSE", "S32", "R0", "R0_A25", "R0_A50", "R0_B05", "R0_B20", "FIX_K2", "FIX_K4", "FIX_K8"]:
            if a not in by:
                continue
            sched = ""
            if "rule" in by[a][ids[0]]:
                u = by[a][ids[0]]["rule"]
                sched = f"layer {u['partial_layer_start']}, {u['partial_join']}, k={u['partial_every_k']}"
            elif a.startswith("FIX"):
                sched = f"layer 10, blank, k={a[-1]}"
            sp = ratio([by["DENSE"][i]["latency_s"] for i in ids], [by[a][i]["latency_s"] for i in ids])[0]
            c = [by[a][i].get("correct") for i in ids]
            ok = all(x is not None for x in c)
            dd = delta(c, [by["DENSE"][i]["correct"] for i in ids]) if ok and a != "DENSE" else ""
            dr = delta(c, [by["R0"][i]["correct"] for i in ids]) if ok and a not in ("R0", "DENSE") else ""
            out.append(f"| {a} | {sched} | {sp:.2f}x | {np.mean([by[a][i]['nfe'] for i in ids]):.1f} | {int(sum(c)) if ok else 'pending'} | {dd} | {dr} |")
        out.append("")
    return out


def par():
    out = ["# Extra-prompt parity of R0 against dense Conf-0.9 (LLaDA-8B)", "Paired change in correct answers (R0 minus dense); percentage points of prompts; 95% bootstrap over prompts.", "", "| set | prompts | dense correct | R0 correct | Δ answers | Δ points | speed vs dense |", "|---|---|---|---|---|---|---|"]
    vecs = []
    for ds, prefix, tag in (("kod_ext", "par", "llada8b"), ("math_ext", "par", "llada8b"), ("gsm_ext", "par", "llada8b"), ("he164", "eng", "llada8b"), ("kod_held", "eng", "llada8b"), ("gsm_held", "eng", "llada8b"), ("math108", "eng", "llada8b")):
        recs = load(prefix, tag, ds)
        by = {}
        for r in recs:
            by.setdefault({"DENSE_C09": "DENSE"}.get(r["arm"], r["arm"]), {})[r["example_id"]] = r
        if "DENSE" not in by or "R0" not in by:
            continue
        ids = sorted(set(by["DENSE"]) & set(by["R0"]))
        c0 = [by["DENSE"][i].get("correct") for i in ids]
        c1 = [by["R0"][i].get("correct") for i in ids]
        if not ids or any(x is None for x in c0 + c1):
            out.append(f"| {ds} | {len(ids)} | pending | pending | | | |")
            continue
        d = np.array(c1, float) - np.array(c0, float)
        vecs.append((ds, d))
        sp = ratio([by["DENSE"][i]["latency_s"] for i in ids], [by["R0"][i]["latency_s"] for i in ids])[0]
        out.append(f"| {ds}{' (Study 5)' if prefix == 'eng' else ' (new)'} | {len(ids)} | {int(sum(c0))} | {int(sum(c1))} | {delta(c1, c0)} | {100 * d.mean():+.2f} | {sp:.2f}x |")

    def pool(sel):
        vs = [d for n, d in vecs if sel(n)]
        if not vs:
            return None
        n = sum(len(v) for v in vs)
        bs = np.array([sum(v[rng.integers(0, len(v), len(v))].sum() for v in vs) / n for _ in range(4000)])
        return sum(v.sum() for v in vs) / n, np.percentile(bs, 2.5), np.percentile(bs, 97.5), n
    for lab, sel in (("new prompts only", lambda n: n.endswith("_ext")), ("new + Study 5 (LLaDA-8B)", lambda n: True)):
        p = pool(sel)
        if p:
            out.append(f"\nPooled, {lab}: {100 * p[0]:+.2f} pts [{100 * p[1]:+.2f}, {100 * p[2]:+.2f}] over {p[3]} prompts" + ("" if p[1] > 0 or p[2] < 0 else " (interval includes 0)"))
    return out


def noise():
    out = ["# Noise-matched control of the stale K/V on LLaDA-MoE (HumanEval; premise P3)", "Arms: P-stale (= plain S32), P-fresh (oracle), P-rand1a/b (fresh + noise of the size of the staleness).", ""]
    recs = load("noise", "moe", "he164")
    by = {}
    for r in recs:
        by.setdefault(r["arm"], {})[r["example_id"]] = r
    ref = {}
    for r in map(json.loads, open(REC / "eng_moe_he164.scored.jsonl")):
        ref.setdefault(r["arm"], {})[r["example_id"]] = r
    if not by:
        return out + ["no records yet"]
    ids = sorted(set.intersection(*[set(v) for v in by.values()]))
    out += [f"n = {len(ids)} prompts", "| arm | correct | Δ vs P-fresh | Δ vs P-stale | identical to dense decode | identical to plain S32 |", "|---|---|---|---|---|---|"]
    ok = all(by[a][i].get("correct") is not None for a in by for i in ids)
    for a in ("P-fresh", "P-stale", "P-rand1a", "P-rand1b"):
        if a not in by:
            continue
        c = [by[a][i].get("correct") for i in ids]
        eqd = sum(by[a][i]["tokens"] == ref["DENSE_C09"][i]["tokens"] for i in ids)
        eqs = sum(by[a][i]["tokens"] == ref["S32"][i]["tokens"] for i in ids)
        out.append(f"| {a} | {int(sum(c)) if ok else 'pending'} | {delta(c, [by['P-fresh'][i]['correct'] for i in ids]) if ok and a != 'P-fresh' else ''} | {delta(c, [by['P-stale'][i]['correct'] for i in ids]) if ok and a != 'P-stale' else ''} | {eqd}/{len(ids)} | {eqs}/{len(ids)} |")
    if ok and all(a in by for a in ("P-rand1a", "P-rand1b", "P-stale")):
        n1 = [(by["P-rand1a"][i]["correct"] + by["P-rand1b"][i]["correct"]) / 2 for i in ids]
        out.append(f"\nPre-registered contrast (as on LLaDA-8B): mean of the two noise seeds minus P-stale: {delta(n1, [by['P-stale'][i]['correct'] for i in ids])}   (LLaDA-8B: +11.0 [+2, +20])")
    return out


if __name__ == "__main__":
    name = sys.argv[1]
    lines = {"thr": thr, "abl": abl, "par": par, "noise": noise}[name]()
    (REC.parent / f"results_{name}.md").write_text("\n".join(lines))
    print("\n".join(lines))
