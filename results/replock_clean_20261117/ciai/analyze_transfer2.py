"""Analyze the ciai comparisons with the ORIGINAL decoder: records transfer_<TAG>_<SET>(.scored).jsonl, TAG in a100full (five arms, 40 prompts: gsm_held, he164) and a100n100 (four arms, 100 prompts: gsm_held, kod_held, math108).
Per set: speed vs dense and (where the original decoder ran) vs the ORIGINAL, split into word lock x cache; answers paired. Prints markdown."""
import json, sys
from pathlib import Path
import numpy as np
REC = Path(sys.argv[1]); rng = np.random.default_rng(0)
LAB = {"ORIG": "Original decoder", "DENSE_C09": "Dense Conf-0.9", "S32": "Plain cached (S32)", "R0_OWN": "R0 derived on the A100", "R0_RTX": "R0 with the RTX schedule"}
def load(tag, ds):
    p = REC / f"transfer_{tag}_{ds}.scored.jsonl"; p = p if p.exists() else REC / f"transfer_{tag}_{ds}.jsonl"
    if not p.exists(): return None
    arms = {}
    for l in open(p):
        r = json.loads(l); arms.setdefault(r["arm"], {})[r["example_id"]] = r
    return arms
def ratio(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [a[i].sum() / b[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
    return a.sum() / b.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)
def delta(x, y):
    d = np.asarray(x, float) - np.asarray(y, float); bs = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5]); return f"{int(d.sum()):+d} [{lo:+.0f}, {hi:+.0f}]" + ("" if lo > 0 or hi < 0 else " (incl. 0)")
for tag, sets in (("a100full", ("gsm_held", "he164")), ("a100n100", ("gsm_held", "kod_held", "math108"))):
    for ds in sets:
        arms = load(tag, ds)
        if not arms: continue
        ids = sorted(set.intersection(*[set(v) for v in arms.values()]))
        have_o = "ORIG" in arms
        print(f"\n### {ds} ({tag}, n={len(ids)} prompts, A100)")
        print("| arm | s/prompt | calls | speed vs dense | speed vs original | correct | Δ vs dense | Δ vs original |\n|---|---|---|---|---|---|---|---|")
        L = lambda a: [arms[a][i]["latency_s"] for i in ids]
        c = lambda a: [arms[a][i].get("correct") for i in ids]
        for a in ("ORIG", "DENSE_C09", "S32", "R0_OWN", "R0_RTX"):
            if a not in arms: continue
            sp = ratio(L("DENSE_C09"), L(a)); so = ratio(L("ORIG"), L(a)) if have_o else None
            acc = "pending" if any(x is None for x in c(a)) else f"{int(sum(c(a)))}/{len(ids)}"
            dd = do = ""
            if acc != "pending" and a != "DENSE_C09": dd = delta(c(a), c("DENSE_C09"))
            if acc != "pending" and have_o and a != "ORIG": do = delta(c(a), c("ORIG"))
            print(f"| {LAB[a]} | {np.mean(L(a)):.2f} | {np.mean([arms[a][i]['nfe'] for i in ids]):.1f} | {sp[0]:.2f}x [{sp[1]:.2f}, {sp[2]:.2f}] | " + (f"{so[0]:.2f}x [{so[1]:.2f}, {so[2]:.2f}]" if so else "–") + f" | {acc} | {dd} | {do} |")
        if have_o:
            wl = ratio(L("ORIG"), L("DENSE_C09")); ca = ratio(L("DENSE_C09"), L("R0_OWN")); tot = ratio(L("ORIG"), L("R0_OWN"))
            print(f"\nR0 (A100-derived): word lock {wl[0]:.2f}x × cache {ca[0]:.2f}x = {wl[0] * ca[0]:.2f}x (measured total {tot[0]:.2f}x)")
