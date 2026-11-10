"""Analysis of the far-blank cache x word lock stack (CPU). Usage: <abs python> analyze_stack.py [--stages 1,2] [--out DIR]
Paired bootstrap over questions (10,000 resamples) for every ratio; ratio of summed decode seconds inside one process."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
NB = 10000
ARMS = ["D:C-0.9", "D:C-0.8", "D:Lctx-0.8", "S16:C-0.9", "S16:C-0.8", "S16:Lctx-0.8", "S32:C-0.9", "S32:C-0.8", "S32:Lctx-0.8"]
SETS = {"math108": ("MATH 108 (54 L5 + 54 L1-4), gen 512", 512, 3), "he164": ("HumanEval 0-163, gen 512", 512, 5), "math54half": ("MATH 54 (27 L5 + 27 L1-4), gen 1024", 1024, 2),
        "gsm8k200": ("GSM8K test 0-199, gen 256", 256, 6)}      # (title, gen, quality-preserving floor: net >= -floor, about 2.8% of n)


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def own_dense(arm):
    return "D:" + arm.split(":")[1]


def ci(v):
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def analyse(name, recs, md, summary):
    title, gen, floor = SETS[name]
    by = {}
    for r in recs:
        if r["set"] == name and r.get("status") == "ok":
            by.setdefault(r["arm"], {})[r["example_id"]] = r
    arms = [a for a in ARMS if a in by]
    if "D:C-0.9" not in arms:
        return
    ids = sorted(set.intersection(*[set(by[a]) for a in arms]))
    n = len(ids)
    if n == 0:
        return
    strat = np.array([by["D:C-0.9"][i]["stratum"] for i in ids])
    g = lambda key, a: np.array([by[a][i][key] for i in ids], dtype=float)
    lat, nfe, rt = {a: g("latency_s", a) for a in arms}, {a: g("nfe", a) for a in arms}, {a: g("rule_time_s", a) for a in arms}
    cor = {a: np.array([int(by[a][i]["correct"]) for i in ids]) for a in arms}
    txt = {a: [by[a][i]["text"] for i in ids] for a in arms}
    rpc = {a: g("rows_per_call_mean", a) for a in arms}
    fills = {a: np.array([(by[a][i]["first_eos"] is None) or (by[a][i]["first_eos"] >= gen - 8) for i in ids]) for a in arms}
    feos = {a: np.array([by[a][i]["first_eos"] if by[a][i]["first_eos"] is not None else gen for i in ids]) for a in arms}
    dcalls = {a: g("dense_calls", a) for a in arms}
    rng = np.random.default_rng(23)
    summary[name] = {"n": n, "floor": floor, "cells": {}}
    groups = [("pooled", np.ones(n, bool))]
    if name.startswith("math"):
        groups += [("L5", strat == "L5"), ("L1-4", strat == "L1-4")]
    for gname, sel in groups:
        idx_all = np.nonzero(sel)[0]
        m = len(idx_all)
        idx = idx_all[rng.integers(0, m, size=(NB, m))]
        S = lambda d, a: d[a][idx].sum(1)
        spd = lambda base, a: S(lat, base) / S(lat, a)
        md.append(f"\n### {title}: {gname} (n = {m}; Conf-0.9 dense correct {int(cor['D:C-0.9'][sel].sum())})\n")
        md.append("| cell | speed vs D:C-0.9 [CI] | speed vs own dense lock [CI] | interaction R [CI] | calls | ms/call | rows/call | dense calls | correct | net vs D:C-0.9 [CI] (lost/gained) | net vs own dense lock [CI] (lost/gained) | changed text vs D:C-0.9 / own dense | fills budget | median first EOS | rule ms/call |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for a in arms:
            row = {}
            cells = []
            if a == "D:C-0.9":
                cells += ["-", "-", "-"]
            else:
                p = lat["D:C-0.9"][sel].sum() / lat[a][sel].sum()
                lo, hi = ci(spd("D:C-0.9", a))
                row["speed_vs_dC09"] = [float(p), lo, hi]
                cells.append(f"{p:.3f} [{lo:.3f}, {hi:.3f}]")
                od = own_dense(a)
                if od == a:
                    cells.append("-")
                else:
                    p2 = lat[od][sel].sum() / lat[a][sel].sum()
                    lo2, hi2 = ci(spd(od, a))
                    row["speed_vs_own_dense"] = [float(p2), lo2, hi2]
                    cells.append(f"{p2:.3f} [{lo2:.3f}, {hi2:.3f}]")
                if a.startswith("S") and not a.endswith("C-0.9"):
                    shape = a.split(":")[0]
                    ref = f"{shape}:C-0.9"
                    Rb = spd("D:C-0.9", a) / (spd("D:C-0.9", od) * spd("D:C-0.9", ref))
                    R = (lat["D:C-0.9"][sel].sum() / lat[a][sel].sum()) / ((lat["D:C-0.9"][sel].sum() / lat[od][sel].sum()) * (lat["D:C-0.9"][sel].sum() / lat[ref][sel].sum()))
                    lo3, hi3 = ci(Rb)
                    row["R"] = [float(R), lo3, hi3]
                    cells.append(f"{R:.3f} [{lo3:.3f}, {hi3:.3f}]")
                else:
                    cells.append("-")
            nets = []
            for base in ("D:C-0.9", own_dense(a)):
                if base == a:
                    nets.append("-")
                    continue
                d = (cor[a] - cor[base]).astype(float)
                lost = int(((cor[base] == 1) & (cor[a] == 0))[sel].sum())
                gained = int(((cor[base] == 0) & (cor[a] == 1))[sel].sum())
                lo, hi = ci(d[idx].sum(1))
                nets.append(f"{int(d[sel].sum()):+d} [{lo:+.0f}, {hi:+.0f}] ({lost}/{gained})")
                row["net_vs_" + ("dC09" if base == "D:C-0.9" else "own")] = [int(d[sel].sum()), lo, hi, lost, gained]
            ch0 = float(np.mean([t != u for t, u, s in zip(txt[a], txt["D:C-0.9"], sel) if s])) if a != "D:C-0.9" else 0.0
            od = own_dense(a)
            ch1 = float(np.mean([t != u for t, u, s in zip(txt[a], txt[od], sel) if s])) if od != a else 0.0
            msc = 1000 * lat[a][sel].sum() / nfe[a][sel].sum()
            rms = 1000 * rt[a][sel].sum() / nfe[a][sel].sum()
            row.update(calls=float(nfe[a][sel].mean()), ms_per_call=float(msc), rows_per_call=float(rpc[a][sel].mean()), dense_calls=float(dcalls[a][sel].mean()), correct=int(cor[a][sel].sum()),
                       changed_vs_dC09=ch0, changed_vs_own=ch1, fills=float(fills[a][sel].mean()), median_first_eos=float(np.median(feos[a][sel])), rule_ms_per_call=float(rms))
            summary[name]["cells"].setdefault(gname, {})[a] = row
            md.append(f"| {a} | " + " | ".join(cells) + f" | {nfe[a][sel].mean():.1f} | {msc:.1f} | {rpc[a][sel].mean():.0f} | {dcalls[a][sel].mean():.1f} | {int(cor[a][sel].sum())}/{m} | {nets[0]} | {nets[1]} | {ch0:.2f} / {ch1:.2f} | {fills[a][sel].mean():.2f} | {np.median(feos[a][sel]):.0f} | {rms:.2f} |")
        if gname == "pooled":
            md.append(f"\nQuality-preserving rule for this set: net vs dense C-0.9 >= -{floor} (point rule; a cell whose net interval contains 0 is not called a loss). Cells meeting it: "
                      + ", ".join(a for a in arms if a != "D:C-0.9" and summary[name]["cells"]["pooled"][a]["net_vs_dC09"][0] >= -floor) + ".")
    if name == "he164":
        sel0 = np.array([int(i.split("/")[1]) < 40 for i in ids])
        md.append("\n#### HumanEval tasks 0-39 only (the set on which FIX-both lost 4-5 answers; dense C-0.9 correct %d/40)\n" % int(cor["D:C-0.9"][sel0].sum()))
        md.append("| cell | correct (of 40) | net vs D:C-0.9 (lost/gained) |\n|---|---|---|")
        for a in arms:
            d = cor[a] - cor["D:C-0.9"]
            md.append(f"| {a} | {int(cor[a][sel0].sum())} | {int(d[sel0].sum()):+d} ({int(((cor['D:C-0.9'] == 1) & (cor[a] == 0))[sel0].sum())}/{int(((cor['D:C-0.9'] == 0) & (cor[a] == 1))[sel0].sum())}) |")
    la = np.array([by[a][i]["loadavg1"] for a in arms for i in ids])
    md.append(f"\nNode load average at decode time: median {np.median(la):.1f}, max {la.max():.1f} (12 cores). Peak GPU memory (GiB): " + ", ".join(f"{a} {max(by[a][i]['peak_mem_gib'] for i in ids):.1f}" for a in arms) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", default="1,2")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dir", default=str(OUT))
    a = ap.parse_args()
    md = ["# Results: far-blank cache stacked with three word locks\n",
          "Cells: D = dense word lock only (existing dense decoder); S16 = far-blank cache, query rows = 16-blank window + rows written last call; S32 = far-blank cache, query rows = the current 32-block's blanks + rows written last call; "
          "both with first-EOS truncation and a dense refresh at every block start. Speeds are ratios of summed decode seconds inside one process (paired bootstrap over questions, 10,000 resamples). "
          "Interaction R = (cache cell vs D:C-0.9) / [(own dense lock vs D:C-0.9) x (same-shape C-0.9 cache cell vs D:C-0.9)]: R = 1 means word-lock and cache savings multiply.\n"]
    summary = {}
    recs = []
    for st in a.stages.split(","):
        recs += rows(Path(a.dir) / f"records/stage{st}.jsonl")
    for name in ("he164", "math108", "gsm8k200", "math54half"):
        analyse(name, recs, md, summary)
    Path(a.out, "results.md").write_text("\n".join(md) + "\n")
    Path(a.out, "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(md)[:3000])


if __name__ == "__main__":
    main()
