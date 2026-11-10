"""Analysis of the Learn2PD control (CPU). Usage: <abs python> analyze_l2pd.py [--stages 1,2] [--records records/stage1.jsonl ...] [--out DIR]
Paired bootstrap over QUESTIONS (10,000 resamples, one resample matrix shared by every statistic of a group). Speed = ratio of summed decode seconds; a ratio between two cells uses only
the questions on which both cells were decoded in the same process (`proc`), the number used is reported. Answers: net = correct(cell) - correct(reference), lost / gained counts, exact
two-sided sign test on lost vs gained (scipy.stats.binomtest, p = 0.5), churn = lost + gained."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
NB = 10000
ORDER_D = ["C-0.9", "C-0.8", "C-0.7", "C-0.6", "Lctx-0.8", "L2PD-pub", "L2PD-match", "L2PD-off"]
ARMS = [f"D:{l}" for l in ORDER_D] + [f"T:{l}" for l in ORDER_D] + [f"S32:{l}" for l in ORDER_D]
SETS = {"he164": ("HumanEval 0-163, gen 512", 512), "math108": ("MATH 108 (54 L5 + 54 L1-4), gen 512", 512), "math54half": ("MATH 54 (27 L5 + 27 L1-4), gen 1024", 1024),
        "gsm8k200": ("GSM8K test 0-199, gen 256", 256), "pilot": ("PILOT (held-out training questions)", 512)}
REF_D, REF_T = "D:C-0.9", "T:C-0.9"


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def ci(v):
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def fmt_ci(p, lo, hi, nd=3):
    return f"{p:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]"


class SetData:
    def __init__(self, name, recs):
        self.name = name
        by = {}
        for r in recs:
            if r["set"] == name and r.get("status") == "ok":
                by.setdefault(r["arm"], {})[r["example_id"]] = r
        self.arms = [a for a in ARMS if a in by]
        ids = sorted(set.intersection(*[set(by[a]) for a in self.arms])) if self.arms else []
        self.ids, self.by = ids, by
        self.n = len(ids)
        g = lambda key, a, default=0.0: np.array([by[a][i].get(key, default) if by[a][i].get(key) is not None else default for i in ids], dtype=float)
        self.lat = {a: g("latency_s", a) for a in self.arms}
        self.nfe = {a: g("nfe", a) for a in self.arms}
        self.rt = {a: g("rule_time_s", a) for a in self.arms}
        self.rpc = {a: g("rows_per_call_mean", a) for a in self.arms}
        self.dcalls = {a: g("dense_calls", a) for a in self.arms}
        self.fb = {a: g("fallback_calls", a) for a in self.arms}
        self.words = {a: g("words_written", a) for a in self.arms}
        self.cor = {a: np.array([int(bool(by[a][i]["correct"])) if by[a][i]["correct"] is not None else 0 for i in ids]) for a in self.arms}
        self.txt = {a: [by[a][i]["text"] for i in ids] for a in self.arms}
        self.proc = {a: np.array([by[a][i].get("proc", "?") for i in ids]) for a in self.arms}
        gen = SETS[name][1]
        self.feos = {a: np.array([by[a][i]["first_eos"] if by[a][i]["first_eos"] is not None else gen for i in ids], dtype=float) for a in self.arms}
        self.stratum = np.array([by[self.arms[0]][i]["stratum"] for i in ids]) if self.arms else np.array([])
        self.l2 = {a: {k: g_stat(by[a], ids, k) for k in ("calls", "words", "filter_words", "empty", "top1_added")} for a in self.arms if "L2PD" in a}
        self.loadavg = np.array([by[a][i].get("loadavg1", 0.0) for a in self.arms for i in ids])
        self.qload = np.array([max(by[a][i].get("loadavg1", 0.0) for a in self.arms) for i in ids])      # worst node load average over the question's cells


def g_stat(d, ids, k):
    return np.array([d[i].get("l2pd_stats", {}).get(k, 0) for i in ids], dtype=float)


def analyse_group(D, gname, sel, rng, md, summary):
    m = int(sel.sum())
    if m == 0:
        return
    idx_all = np.nonzero(sel)[0]
    idx = idx_all[rng.integers(0, m, size=(NB, m))]

    def valid(a, b):
        return (D.proc[a] == D.proc[b]) & sel

    def speed(ref, a):
        v = valid(ref, a).astype(float)
        num, den = D.lat[ref] * v, D.lat[a] * v
        p = num.sum() / den.sum()
        lo, hi = ci(num[idx].sum(1) / den[idx].sum(1))
        return p, lo, hi, int(v.sum())

    def net(a, b):
        d = (D.cor[a] - D.cor[b]).astype(float)
        lost = int(((D.cor[b] == 1) & (D.cor[a] == 0))[sel].sum())
        gained = int(((D.cor[b] == 0) & (D.cor[a] == 1))[sel].sum())
        lo, hi = ci(d[idx].sum(1))
        pv = binomtest(lost, lost + gained, 0.5).pvalue if lost + gained else 1.0
        return int(d[sel].sum()), lo, hi, lost, gained, float(pv)

    title = SETS[D.name][0]
    md.append(f"\n### {title}: {gname} (n = {m}; dense C-0.9 correct {int(D.cor[REF_D][sel].sum())})\n")
    md.append("| cell | speed vs D:C-0.9 [CI] | speed vs T:C-0.9 [CI] | speed vs own dense D | speed vs own twin T | calls | ms/call | rows/call | dense calls | content tok/call | words written/call | fallback % | rule ms/call | correct | net vs D:C-0.9 [CI] | lost/gained | churn | sign p | changed text vs D:C-0.9 |\n"
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    S = summary.setdefault(D.name, {}).setdefault(gname, {"n": m, "cells": {}})
    for a in D.arms:
        kind, lock = a.split(":")
        row = {}
        cells = []
        for ref in (REF_D, REF_T):
            if ref not in D.arms or a == ref:
                cells.append("-")
                continue
            p, lo, hi, nu = speed(ref, a)
            cells.append(fmt_ci(p, lo, hi) + ("" if nu == m else f" (n={nu})"))
            row["speed_vs_" + ref.replace(":", "_")] = [p, lo, hi, nu]
        for own in (f"D:{lock}", f"T:{lock}"):
            if own not in D.arms or own == a or (own.startswith("T:") and kind == "T") or (own.startswith("D:") and kind == "D"):
                cells.append("-")
                continue
            p, lo, hi, nu = speed(own, a)
            cells.append(fmt_ci(p, lo, hi))
            row["speed_vs_own_" + own[0]] = [p, lo, hi, nu]
        if a == REF_D:
            netc, lg, ch, pv, chg = "-", "-", "-", "-", "-"
        else:
            nn, lo, hi, lost, gained, pval = net(a, REF_D)
            row["net_vs_dC09"] = [nn, lo, hi, lost, gained, pval]
            netc, lg, ch, pv = f"{nn:+d} [{lo:+.0f}, {hi:+.0f}]", f"{lost}/{gained}", str(lost + gained), f"{pval:.3f}"
            chg = f"{float(np.mean([t != u for t, u, s in zip(D.txt[a], D.txt[REF_D], sel) if s])):.2f}"
        nfe_s = D.nfe[a][sel].sum()
        content = np.minimum(D.feos[a], SETS[D.name][1])[sel].sum() / nfe_s
        wpc = D.words[a][sel].sum() / nfe_s
        fbp = 100 * D.fb[a][sel].sum() / nfe_s
        row.update(calls=float(D.nfe[a][sel].mean()), ms_per_call=float(1000 * D.lat[a][sel].sum() / nfe_s), rows_per_call=float(D.rpc[a][sel].mean()), dense_calls=float(D.dcalls[a][sel].mean()),
                   content_tokens_per_call=float(content), words_written_per_call=float(wpc), fallback_pct=float(fbp), rule_ms_per_call=float(1000 * D.rt[a][sel].sum() / nfe_s),
                   correct=int(D.cor[a][sel].sum()))
        S["cells"][a] = row
        md.append(f"| {a} | " + " | ".join(cells) + f" | {D.nfe[a][sel].mean():.1f} | {row['ms_per_call']:.1f} | {row['rows_per_call']:.0f} | {row['dense_calls']:.1f} | {content:.2f} | {wpc:.2f} | {fbp:.0f} | {row['rule_ms_per_call']:.2f} | "
                  f"{int(D.cor[a][sel].sum())}/{m} | {netc} | {lg} | {ch} | {pv} | {chg} |")
    if gname != "pooled":
        return
    # ---- load sensitivity: another job on the node (CPU load) slows the CPU-bound launch path of every cell, the Lctx / filter predicts most
    quiet = sel & (D.qload <= 3.0)
    if 0 < quiet.sum() < m:
        idq_all = np.nonzero(quiet)[0]
        idq = idq_all[rng.integers(0, len(idq_all), size=(NB, len(idq_all)))]
        md.append(f"\nLoad sensitivity: {int(quiet.sum())} of {m} questions were decoded with node load average <= 3 in every cell (the rest overlapped another job on the node). Speed vs D:C-0.9 / T:C-0.9 on those questions only:\n")
        md.append("| cell | speed vs D:C-0.9 [CI] | speed vs T:C-0.9 [CI] |\n|---|---|---|")
        for a in D.arms:
            out = []
            for ref in (REF_D, REF_T):
                if ref not in D.arms or a == ref:
                    out.append("-")
                    continue
                v = ((D.proc[ref] == D.proc[a]) & quiet).astype(float)
                num, den = D.lat[ref] * v, D.lat[a] * v
                lo, hi = ci(num[idq].sum(1) / den[idq].sum(1))
                out.append(fmt_ci(num.sum() / den.sum(), lo, hi))
                S["cells"][a].setdefault("quiet_speed_vs_" + ref.replace(":", "_"), [float(num.sum() / den.sum()), lo, hi, int(quiet.sum())])
            md.append(f"| {a} | {out[0]} | {out[1]} |")
    # ---- learned-rule diagnostics
    if D.l2:
        md.append("\nLearn2PD filter behaviour (per call; `empty` = the filter accepted no blank, so only the forced most-confident blank was written; `top1 added` = the most confident blank was written although the filter rejected it; `filter words` = blanks accepted by the filter).\n")
        md.append("| cell | calls/q | filter words/call | words written/call | empty % | top1 added % |\n|---|---|---|---|---|---|")
        for a, st in D.l2.items():
            c = st["calls"][sel].sum()
            md.append(f"| {a} | {st['calls'][sel].mean():.1f} | {st['filter_words'][sel].sum() / c:.2f} | {st['words'][sel].sum() / c:.2f} | {100 * st['empty'][sel].sum() / c:.0f} | {100 * st['top1_added'][sel].sum() / c:.0f} |")
    # ---- truncation vs cache split
    if REF_T in D.arms:
        md.append("\nHow much of the cache speed-up is end-of-text truncation (end-of-text truncation = a dense call on every call, but blanks after the first written EOS dropped) and how much is the cache itself:\n")
        md.append("| lock | truncation alone: T vs D | cache alone: S32 vs T | both: S32 vs D (own lock) | truncation share of log-speed-up |\n|---|---|---|---|---|")
        for lock in ORDER_D:
            d, t, s = f"D:{lock}", f"T:{lock}", f"S32:{lock}"
            if not (d in D.arms and t in D.arms and s in D.arms):
                continue
            p1, l1, h1, _ = speed(d, t)
            p2, l2_, h2, _ = speed(t, s)
            p3, l3, h3, _ = speed(d, s)
            share = np.log(p1) / np.log(p3) if p3 != 1 else float("nan")
            S["cells"].setdefault(f"split:{lock}", {}).update(trunc=[p1, l1, h1], cache=[p2, l2_, h2], both=[p3, l3, h3], trunc_share=float(share))
            md.append(f"| {lock} | {fmt_ci(p1, l1, h1)} | {fmt_ci(p2, l2_, h2)} | {fmt_ci(p3, l3, h3)} | {100 * share:.0f}% |")
    # ---- pairwise learned rule vs lowered thresholds
    md.append("\nPaired comparisons (A vs B: speed A/B = B's summed time over A's; net = correct(A) - correct(B); positive speed > 1 means A is faster):\n")
    md.append("| kind | A | B | speed A vs B [CI] | net answers A - B [CI] | lost/gained (A vs B) | sign p |\n|---|---|---|---|---|---|---|")
    pairs = []
    for kind in ("D", "S32"):
        for A in ("Lctx-0.8", "L2PD-pub", "L2PD-match", "L2PD-off"):
            for B in ("C-0.9", "C-0.8", "C-0.7", "C-0.6", "Lctx-0.8"):
                if A != B and f"{kind}:{A}" in D.arms and f"{kind}:{B}" in D.arms:
                    pairs.append((f"{kind}:{A}", f"{kind}:{B}"))
    for a, b in pairs:
        p, lo, hi, nu = speed(b, a)
        nn, nlo, nhi, lost, gained, pv = net(a, b)
        S["cells"].setdefault("pair:" + a + "|" + b, {}).update(speed=[p, lo, hi, nu], net=[nn, nlo, nhi, lost, gained, pv])
        md.append(f"| {a.split(':')[0]} | {a.split(':')[1]} | {b.split(':')[1]} | {fmt_ci(p, lo, hi)} | {nn:+d} [{nlo:+.0f}, {nhi:+.0f}] | {lost}/{gained} | {pv:.3f} |")
    # ---- equal speed: each learned cell against the C-c cell nearest in speed
    md.append("\nEqual-speed reading: the confidence-threshold cell (same kind) whose speed vs D:C-0.9 is closest to the learned cell's, and the learned cell's answer difference to it.\n")
    md.append("| kind | learned cell | its speed vs D:C-0.9 | nearest C-c cell | its speed | net learned - C-c [CI] | lost/gained | sign p |\n|---|---|---|---|---|---|---|---|")
    for kind in ("D", "S32"):
        cs = [(f"{kind}:{c}", speed(REF_D, f"{kind}:{c}")[0]) for c in ("C-0.9", "C-0.8", "C-0.7", "C-0.6") if f"{kind}:{c}" in D.arms]
        for A in ("Lctx-0.8", "L2PD-pub", "L2PD-match", "L2PD-off"):
            a = f"{kind}:{A}"
            if a not in D.arms or not cs:
                continue
            sa = speed(REF_D, a)[0]
            b, sb = min(cs, key=lambda t: abs(t[1] - sa))
            nn, nlo, nhi, lost, gained, pv = net(a, b)
            S["cells"].setdefault("equal:" + a, {}).update(nearest=b, speed_learned=sa, speed_nearest=sb, net=[nn, nlo, nhi, lost, gained, pv])
            md.append(f"| {kind} | {A} | {sa:.3f} | {b.split(':')[1]} | {sb:.3f} | {nn:+d} [{nlo:+.0f}, {nhi:+.0f}] | {lost}/{gained} | {pv:.3f} |")


def analyse(name, recs, md, summary):
    D = SetData(name, recs)
    if REF_D not in D.arms or D.n == 0:
        return
    rng = np.random.default_rng(23)
    groups = [("pooled", np.ones(D.n, bool))]
    if name.startswith("math"):
        groups += [("L5", D.stratum == "L5"), ("L1-4", D.stratum == "L1-4")]
    for gname, sel in groups:
        analyse_group(D, gname, sel, rng, md, summary)
    procs = sorted({p for a in D.arms for p in D.proc[a]})
    md.append(f"\nProcesses contributing to this set: {len(procs)} ({', '.join(procs)}); node load average at decode time: median {np.median(D.loadavg):.1f}, max {D.loadavg.max():.1f} (12 cores).\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1")
    ap.add_argument("--records", nargs="*")
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    recs = []
    for p in (a.records or [OUT / f"records/stage{s}.jsonl" for s in a.stages.split(",")]):
        recs += rows(p)
    md = ["# Results: Learn2PD as a word lock, against confidence thresholds and Lctx, dense / dense + truncation / selective cache\n",
          "Cells: D = dense word lock (paper_decode path, no truncation); T = dense + first-EOS truncation twin (selective engine, a dense call on every call); S32 = selective cache "
          "(query rows = the current 32-block's blanks, all 32 block slots for Learn2PD, + rows written last call; a dense call at every block start; first-EOS truncation). "
          "Speeds are ratios of summed decode seconds inside one process (paired bootstrap over questions, 10,000 resamples, 95% intervals).\n"]
    summary = {}
    for name in ("he164", "math108", "gsm8k200", "math54half", "pilot"):
        analyse(name, recs, md, summary)
    Path(a.out, "results.md").write_text("\n".join(md) + "\n")
    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(md)[:4000])


if __name__ == "__main__":
    main()
