"""Part C analysis (CPU): speed ratios (ratio of summed decode seconds, paired bootstrap over questions, 10,000 resamples), answers (lost/gained/net), calls,
ms/call, changed text, fills-budget share, rows computed, refreshes. Usage: <abs python> analyze_partC.py [--records PATH] [--out DIR]"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
GEN = 512
EOS_FILL_FROM = GEN - 8
ARMS = ["Conf-0.9", "SPARSE-none", "W16-dense", "W16-cache-block", "W16-cache-block+EOS", "W16-cache+EOS+trigger", "W16-cache+EOS+trigger-keys"]
SEL = "W16-cache+EOS+selective-"
NB = 10000


def rows(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def boot(fn, n, seed=23):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(NB, n))
    vals = np.array([fn(i) for i in idx])
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", default=str(OUT / "partC/records.jsonl"))
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()
    recs = [r for r in rows(a.records) if r.get("status") == "ok"]
    by = {}
    for r in recs:
        by.setdefault(r["arm"], {})[r["example_id"]] = r
    arms = [x for x in ARMS if x in by] + sorted((x for x in by if x.startswith(SEL)), key=lambda x: x[len(SEL):])
    ids = sorted(set.intersection(*[set(by[x]) for x in arms]))
    n = len(ids)
    sets = np.array([by[arms[0]][i]["set"] for i in ids])
    lat = {x: np.array([by[x][i]["latency_s"] for i in ids]) for x in arms}
    nfe = {x: np.array([by[x][i]["nfe"] for i in ids]) for x in arms}
    cor = {x: np.array([int(by[x][i]["correct"]) for i in ids]) for x in arms}
    txt = {x: [by[x][i]["text"] for i in ids] for x in arms}
    fills = {x: np.array([(by[x][i]["first_eos"] is None) or (by[x][i]["first_eos"] >= EOS_FILL_FROM) for i in ids]) for x in arms}
    rowsq = {x: np.array([by[x][i]["rows_per_layer_per_call_mean"] * by[x][i]["nfe"] for i in ids]) for x in arms}

    def speed(base, arm, idx=None, sel=None):
        s = np.ones(n, bool) if sel is None else sel
        i = np.arange(n)[s] if idx is None else idx
        return lat[base][i].sum() / lat[arm][i].sum()

    summary = {"n_questions": n, "arms": {}}
    md = [f"# Part C results: which ingredient buys the speed (gen {GEN}, MATH, {n} questions: {int((sets == 'L5').sum())} L5 + {int((sets == 'L1-4').sum())} L1-4; one process, one timed repeat)\n",
          "Speeds are ratios of summed decode seconds inside this one process (paired bootstrap over questions, 10,000 resamples, 95% interval). Answers: lost = correct under the "
          "reference and wrong under the arm; gained the reverse; net = gained - lost. Resolution about +-4 answers at 54 questions.\n"]
    # --- main table
    md.append("| cell | word lock | speed vs Conf-0.9 [CI] | speed vs SPARSE-none [CI] | speed vs W16-dense [CI] | calls | ms/call | correct | net vs Conf-0.9 [CI] (lost/gained) | net vs W16-dense [CI] (lost/gained) | changed text vs Conf-0.9 | fills budget | rows computed vs Conf-0.9 | refreshes / dense calls per question |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    lock = {"Conf-0.9": "block-32 Conf-0.9", "SPARSE-none": "block-32 Conf-0.9"}
    for x in arms:
        row = {"word_lock": lock.get(x, "window-16 Conf-0.9" + (" + EOS truncation" if "EOS" in x or x == "W16-dense" else ""))}
        cells = []
        for base in ("Conf-0.9", "SPARSE-none", "W16-dense"):
            if base not in lat or x == base:
                cells.append("-")
                continue
            sp = speed(base, x)
            lo, hi = boot(lambda i: lat[base][i].sum() / lat[x][i].sum(), n)
            cells.append(f"{sp:.3f} [{lo:.3f}, {hi:.3f}]")
            row[f"speed_vs_{base}"] = [sp, lo, hi]
        net = {}
        for base in ("Conf-0.9", "W16-dense"):
            if base not in cor or x == base:
                net[base] = "-"
                continue
            d = cor[x] - cor[base]
            lost = int(((cor[base] == 1) & (cor[x] == 0)).sum())
            gained = int(((cor[base] == 0) & (cor[x] == 1)).sum())
            lo, hi = boot(lambda i: d[i].sum(), n)
            net[base] = f"{int(d.sum()):+d} [{lo:+.0f}, {hi:+.0f}] ({lost}/{gained})"
            row[f"net_vs_{base}"] = [int(d.sum()), lo, hi, lost, gained]
        changed = float(np.mean([t != u for t, u in zip(txt[x], txt["Conf-0.9"])])) if x != "Conf-0.9" else 0.0
        refresh = np.mean([by[x][i].get("refreshes", 0) for i in ids])
        dense_calls = np.mean([by[x][i].get("dense_calls", 0) for i in ids]) if x not in ("Conf-0.9", "SPARSE-none") else 0.0
        mpc = 1000 * lat[x].sum() / nfe[x].sum()
        rc = rowsq[x].sum() / rowsq["Conf-0.9"].sum()
        row.update(calls=float(nfe[x].mean()), ms_per_call=float(mpc), correct=int(cor[x].sum()), changed_text=changed, fills=float(fills[x].mean()), rows_vs_conf=float(rc),
                   refreshes=float(refresh), dense_calls=float(dense_calls))
        summary["arms"][x] = row
        md.append(f"| {x} | {row['word_lock']} | {cells[0]} | {cells[1]} | {cells[2]} | {nfe[x].mean():.1f} | {mpc:.1f} | {int(cor[x].sum())}/{n} | {net['Conf-0.9']} | {net['W16-dense']} | {changed:.2f} | {fills[x].mean():.2f} | {rc:.3f} | {refresh:.1f} / {dense_calls:.1f} |")
    # --- strata
    md.append("\n## Strata (indicative: 27 questions each)\n\n| cell | stratum | speed vs Conf-0.9 | speed vs W16-dense | correct | net vs Conf-0.9 |\n|---|---|---|---|---|---|")
    for x in arms:
        for st in ("L5", "L1-4"):
            sel = sets == st
            if x == "Conf-0.9":
                continue
            d = (cor[x] - cor["Conf-0.9"])[sel]
            sw = f"{speed('W16-dense', x, sel=sel):.3f}" if "W16-dense" in lat and x != "W16-dense" else "-"
            md.append(f"| {x} | {st} | {speed('Conf-0.9', x, sel=sel):.3f} | {sw} | {int(cor[x][sel].sum())}/{int(sel.sum())} | {int(d.sum()):+d} |")
    # --- compute breakdown: rows computed per layer per call
    md.append("\n## Rows computed per layer, averaged over calls and questions (dense = prompt + answer rows; layers 0, 4, 8, 12, 16, 20, 24, 28, 31)\n\n| cell | " + " | ".join(f"L{l}" for l in (0, 4, 8, 12, 16, 20, 24, 28, 31)) + " | mean | calls with a refresh | rows joined per refresh |\n|---|" + "---|" * 12)
    prof = {}
    for x in arms:
        if x in ("Conf-0.9", "SPARSE-none"):
            m = float(np.mean([by[x][i]["rows_per_layer_per_call_mean"] for i in ids]))
            md.append(f"| {x} | " + " | ".join(f"{m:.0f}" for _ in range(9)) + f" | {m:.0f} | - | - |")
            continue
        lrs = np.array([np.array(by[x][i]["layer_rows_sum"]) / by[x][i]["nfe"] for i in ids]).mean(0)
        prof[x] = lrs.tolist()
        joined = [by[x][i].get("joined_rows_total", 0) for i in ids]
        ref = [by[x][i].get("refreshes", 0) for i in ids]
        per = (sum(joined) / max(sum(ref), 1)) if sum(ref) else 0.0
        md.append(f"| {x} | " + " | ".join(f"{lrs[l]:.0f}" for l in (0, 4, 8, 12, 16, 20, 24, 28, 31)) + f" | {lrs.mean():.0f} | {np.mean(ref):.1f} | {per:.0f} |")
    summary["rows_per_layer"] = prof
    # --- ingredient steps
    steps = [("W16-dense vs Conf-0.9 (word lock change: window 16 + EOS truncation; no cache)", "Conf-0.9", "W16-dense"),
             ("W16-cache-block vs W16-dense (far-blank cache + block refresh)", "W16-dense", "W16-cache-block"),
             ("W16-cache-block+EOS vs W16-cache-block (EOS truncation)", "W16-cache-block", "W16-cache-block+EOS"),
             ("trigger vs block schedule (both with EOS truncation)", "W16-cache-block+EOS", "W16-cache+EOS+trigger"),
             ("trigger-keys vs trigger (cosine axis)", "W16-cache+EOS+trigger", "W16-cache+EOS+trigger-keys"),
             *[(f"selective-{x[len(SEL):]} vs trigger", "W16-cache+EOS+trigger", x) for x in arms if x.startswith(SEL)],
             *[(f"selective-{x[len(SEL):]} vs block schedule", "W16-cache-block+EOS", x) for x in arms if x.startswith(SEL)],
             ("SPARSE-none vs Conf-0.9 (runtime overhead of row-by-row execution)", "Conf-0.9", "SPARSE-none")]
    md.append("\n## Ladder: what each ingredient adds (speed ratio of summed seconds, calls ratio, net answers)\n\n| step | speed | calls ratio | net answers [CI] |\n|---|---|---|---|")
    summary["steps"] = {}
    for name, b, x in steps:
        if b not in lat or x not in lat:
            continue
        sp = lat[b].sum() / lat[x].sum()
        lo, hi = boot(lambda i: lat[b][i].sum() / lat[x][i].sum(), n)
        cr = nfe[x].sum() / nfe[b].sum()
        d = cor[x] - cor[b]
        nlo, nhi = boot(lambda i: d[i].sum(), n)
        md.append(f"| {name} | {sp:.3f} [{lo:.3f}, {hi:.3f}] | {cr:.3f} | {int(d.sum()):+d} [{nlo:+.0f}, {nhi:+.0f}] |")
        summary["steps"][name] = {"speed": [sp, lo, hi], "calls_ratio": cr, "net": [int(d.sum()), nlo, nhi]}
    peaks = {x: float(max(by[x][i]["peak_mem_gib"] for i in ids)) for x in arms}
    summary["peak_mem_gib"] = peaks
    md.append("\nPeak GPU memory (GiB, max over questions): " + ", ".join(f"{k} {v:.1f}" for k, v in peaks.items()) + "\n")
    Path(a.out, "results_partC.md").write_text("\n".join(md) + "\n")
    Path(a.out, "summary_partC.json").write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(md[:14]))


if __name__ == "__main__":
    main()
