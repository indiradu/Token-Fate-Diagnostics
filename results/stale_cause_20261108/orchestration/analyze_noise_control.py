"""Analysis 1 (CPU): noise-matched control. Reads records/noise_control.jsonl and the stage-1 dense / S32 records of farcache_stack_20261105 (HumanEval 0-163, 512 tokens, word lock C-0.9).
Usage: <python> analyze_noise_control.py [--records DIR] [--ref STAGE1_JSONL] [--out DIR]   -> noise_control_results.md, noise_control_summary.json
Bootstrap over questions (10,000 resamples, seed 23); net = answers gained - answers lost relative to a reference arm; no speed is measured."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
NB = 10000
CLASSES = ("prompt", "prior_written", "inblock_old", "next_blank", "far_blank")
ARMS = ["P-fresh", "P-stale", "P-rand0.5", "P-rand1a", "P-rand1b", "P-rand2", "B-stale", "B-rand1"]
rows = lambda p: [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()] if Path(p).exists() else []


def ci(v):
    return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(OUT / "records"))
    ap.add_argument("--ref", default=str(HERE.parents[1] / "farcache_stack_20261105/records/stage1.jsonl"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    ref = {}
    for r in rows(a.ref):
        if r.get("set") == "he164" and r.get("status") == "ok":
            ref.setdefault(r["arm"], {})[r["example_id"]] = r
    by = {}
    for r in rows(Path(a.records) / "noise_control.jsonl"):
        if r.get("status") == "ok":
            by.setdefault(r["arm"], {})[r["example_id"]] = r
    arms = [x for x in ARMS if x in by]
    ids = sorted(set.intersection(*[set(by[x]) for x in arms]) & set(ref["D:C-0.9"]))
    n = len(ids)
    rng = np.random.default_rng(23)
    idx = rng.integers(0, n, size=(NB, n))
    dense = np.array([int(ref["D:C-0.9"][i]["correct"]) for i in ids])
    s32 = np.array([int(ref["S32:C-0.9"][i]["correct"]) for i in ids])
    cor = {x: np.array([int(by[x][i]["correct"]) for i in ids]) for x in arms}
    cor["dense"], cor["S32 (stage 1)"] = dense, s32
    summ: dict = {"n": n, "dense_correct": int(dense.sum())}
    md = ["# Analysis 1: noise-matched control for the stale K/V of the S32 representation cache", "",
          f"HumanEval 0-163 (n = {n}), 512 tokens, word lock C-0.9, LLaDA-8B-Instruct bf16, block 32. Reference: dense C-0.9 ({int(dense.sum())} correct, from `farcache_stack_20261105/records/stage1.jsonl`; deterministic decode).",
          "The cached rows are the prompt, written-answer and blank rows outside the query set of a call. Arms differ only in where the attention of a non-dense call reads their K/V (cache, dense pass, dense pass + random perturbation of norm = scale x ||stale - fresh|| per row and layer). Bootstrap over questions, 10,000 resamples. No speed is measured.", ""]

    # ---- parity
    eq_s = sum(by["P-stale"][i]["tokens"] == ref["S32:C-0.9"][i]["tokens"] for i in ids) if "P-stale" in by else None
    par = {}
    if "P-stale" in by:
        par["P-stale == stage-1 S32:C-0.9 (tokens)"] = [eq_s, n]
    if "P-fresh" in by:
        par["P-fresh == stage-1 dense C-0.9 (all 512 tokens)"] = [sum(by["P-fresh"][i]["tokens"] == ref["D:C-0.9"][i]["tokens"] for i in ids), n]
        def to_eos(t, fe):
            return t[:(fe + 1) if fe is not None else None]
        par["P-fresh == stage-1 dense C-0.9 (tokens up to the first EOS/EOT)"] = [sum(to_eos(by["P-fresh"][i]["tokens"], by["P-fresh"][i]["first_eos"]) == to_eos(ref["D:C-0.9"][i]["tokens"], ref["D:C-0.9"][i]["first_eos"]) for i in ids), n]
        par["P-fresh and dense agree on pass/fail"] = [int((cor["P-fresh"] == dense).sum()), n]
    summ["parity"] = par
    md += ["## 0. Parity", "", "| check | tasks equal |", "|---|---|"] + [f"| {k} | {v[0]}/{v[1]} |" for k, v in par.items()] + [""]
    if "P-fresh" in by:
        bad = [i for i in ids if by["P-fresh"][i]["tokens"] != ref["D:C-0.9"][i]["tokens"]]
        md += [f"P-fresh tasks whose tokens differ from the dense decode: {', '.join(bad) if bad else 'none'}.", ""]

    # ---- main table
    md += ["## 1. Answers", "",
           "Net = gained - lost against the reference in the column header (lost = reference passes, arm fails); 95% interval by bootstrap over questions. `text differs` = the generated text differs from the reference text.", "",
           "| arm | correct | net vs dense [CI] (lost/gained) | net vs P-fresh [CI] | net vs P-stale [CI] | text differs from dense | text differs from P-stale | median first differing token vs dense |", "|---|---|---|---|---|---|---|---|"]
    txt = lambda x, i: by[x][i]["text"]
    def net(x, y):
        d = (cor[x] - cor[y]).astype(float)
        lo, hi = ci(d[idx].sum(1))
        lost = int(((cor[y] == 1) & (cor[x] == 0)).sum())
        gained = int(((cor[y] == 0) & (cor[x] == 1)).sum())
        return int(d.sum()), lo, hi, lost, gained
    for x in ["S32 (stage 1)"] + arms:
        nd, lo, hi, lost, gained = net(x, "dense")
        c = f"{nd:+d} [{lo:+.0f}, {hi:+.0f}] ({lost}/{gained})"
        nf = net(x, "P-fresh") if "P-fresh" in cor else None
        ns = net(x, "P-stale") if "P-stale" in cor else None
        cf = f"{nf[0]:+d} [{nf[1]:+.0f}, {nf[2]:+.0f}]" if nf else "-"
        cs = f"{ns[0]:+d} [{ns[1]:+.0f}, {ns[2]:+.0f}]" if ns else "-"
        if x.startswith("S32"):
            td = sum(ref["S32:C-0.9"][i]["text"] != ref["D:C-0.9"][i]["text"] for i in ids) / n
            ts, ftd = "-", "-"
        else:
            td = sum(txt(x, i) != ref["D:C-0.9"][i]["text"] for i in ids) / n
            ts = f"{sum(txt(x, i) != txt('P-stale', i) for i in ids) / n:.2f}" if "P-stale" in by else "-"
            v = [by[x][i]["first_token_diff_vs_dense"] for i in ids]
            ftd = f"{int(np.median([512 if t is None else t for t in v]))} (identical to dense: {sum(t is None for t in v)})"
        md.append(f"| {x} | {int(cor[x].sum())} | {c} | {cf} | {cs} | {td:.2f} | {ts} | {ftd} |")
        summ.setdefault("arms", {})[x] = {"correct": int(cor[x].sum()), "net_vs_dense": [nd, lo, hi, lost, gained], "net_vs_fresh": nf, "net_vs_stale": ns, "text_differs_dense": td}
    md.append("")

    # ---- decision quantities
    md += ["## 2. Pre-registered contrasts", ""]
    q = {}
    if all(k in cor for k in ("P-rand1a", "P-rand1b", "P-stale", "P-fresh")):
        n1 = (cor["P-rand1a"] + cor["P-rand1b"]) / 2.0
        for name, base in (("dense", dense), ("P-fresh", cor["P-fresh"]), ("P-stale", cor["P-stale"])):
            d = n1 - base
            lo, hi = ci(d[idx].sum(1))
            q[f"N1 (mean of two seeds) - {name}"] = [float(d.sum()), lo, hi]
        for x, y in (("P-rand1a", "P-rand1b"),):
            d = (cor[x] - cor[y]).astype(float)
            lo, hi = ci(d[idx].sum(1))
            q["seed a - seed b (P-rand1)"] = [float(d.sum()), lo, hi]
    if "B-rand1" in cor and "B-stale" in cor:
        d = (cor["B-rand1"] - cor["B-stale"]).astype(float)
        lo, hi = ci(d[idx].sum(1))
        q["B-rand1 - B-stale"] = [float(d.sum()), lo, hi]
    for k, v in q.items():
        md.append(f"- {k}: {v[0]:+.1f} answers [{v[1]:+.1f}, {v[2]:+.1f}]")
    summ["contrasts"] = q
    md.append("")

    # ---- overlap of lost sets
    if "P-stale" in cor:
        md += ["## 3. Which tasks are lost", "", "Dense solves {} tasks. `overlap` = tasks lost by both the arm and P-stale; `expected` = lost(arm) x lost(P-stale) / {} (independent losses).".format(int(dense.sum()), int(dense.sum())), "",
               "| arm | lost | overlap with P-stale lost | expected by chance |", "|---|---|---|---|"]
        ls = (dense == 1) & (cor["P-stale"] == 0)
        for x in arms:
            lx = (dense == 1) & (cor[x] == 0)
            md.append(f"| {x} | {int(lx.sum())} | {int((lx & ls).sum())} | {lx.sum() * ls.sum() / max(dense.sum(), 1):.1f} |")
        md.append("")

    # ---- distances
    md += ["## 4. How far is the stale cache from a dense pass (measured in the P-stale / P-fresh arms)", "",
           "Mean over rows, calls and questions of ||K_cache - K_dense|| / ||K_dense|| (per row, all kv heads and dims flattened) at layers 1-31 (layer 0 depends on the token only), and the same for V. Rows are the cached rows of non-dense calls.", "",
           "| class | rel K (layers 1-31) | rel V (layers 1-31) | rel K at layers 8 / 16 / 24 / 31 | rows per question (layer 1) |", "|---|---|---|---|---|"]
    src = "P-stale" if "P-stale" in by else None
    if src:
        for c in CLASSES:
            num_k = num_v = den = 0.0
            lay = {8: [0, 0], 16: [0, 0], 24: [0, 0], 31: [0, 0]}
            tot_rows = 0.0
            for i in ids:
                st = by[src][i]["ctrl_stats"][c]
                nrow = st["rows_per_layer"]
                for L in range(1, len(nrow)):
                    num_k += st["rel_k"][L] * nrow[L]
                    num_v += st["rel_v"][L] * nrow[L]
                    den += nrow[L]
                    if L in lay:
                        lay[L][0] += st["rel_k"][L] * nrow[L]
                        lay[L][1] += nrow[L]
                tot_rows += nrow[1]
            fm = lambda t: f"{t[0] / t[1]:.3f}" if t[1] else "-"
            md.append(f"| {c} | {num_k / max(den, 1):.3f} | {num_v / max(den, 1):.3f} | {' / '.join(fm(lay[L]) for L in (8, 16, 24, 31))} | {tot_rows / n:.0f} |")
            summ.setdefault("distance", {})[c] = {"rel_k": num_k / max(den, 1), "rel_v": num_v / max(den, 1)}
        md.append("")
    for x in ("P-rand0.5", "P-rand1a", "P-rand2", "B-rand1"):
        if x in by:
            vals = []
            for i in ids:
                for c in CLASSES:
                    st = by[x][i]["ctrl_stats"][c]
                    vals += [r for r, nr in zip(st["ratio_k"][1:] + st["ratio_v"][1:], st["rows_per_layer"][1:] * 2) if nr > 0 and r > 0]
            if vals:
                md.append(f"- Realized noise norm / target (stale-fresh) norm in {x}: mean {np.mean(vals):.4f}, min {np.min(vals):.4f}, max {np.max(vals):.4f} (perturbed classes only).")
    md.append("")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "noise_control_results.md").write_text("\n".join(md) + "\n")
    (Path(a.out) / "noise_control_summary.json").write_text(json.dumps(summ, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
