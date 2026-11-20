"""Analyze the ciai GPU-transfer comparison: per set, the four arms interleaved on the same A100 (dense Conf-0.9, plain cached S32, R0 derived from this GPU's benchmark, R0 derived from the RTX benchmark).
Usage: python analyze_transfer.py <records dir> [TAG=a100]   -> prints markdown. Speed = total-time ratio over the prompts with a 95% bootstrap over prompts; answers = paired difference with a 95% bootstrap (needs scored records)."""
import json
import sys
from pathlib import Path

import numpy as np

REC = Path(sys.argv[1])
TAG = sys.argv[2] if len(sys.argv) > 2 else "a100"
rng = np.random.default_rng(0)
LABEL = {"DENSE_C09": "Dense Conf-0.9", "S32": "Plain cached (S32)", "R0_OWN": "R0 derived on this GPU (A100)", "R0_RTX": "R0 with the RTX 5000 Ada schedule"}


def load(ds):
    p = REC / f"transfer_{TAG}_{ds}.scored.jsonl"
    p = p if p.exists() else REC / f"transfer_{TAG}_{ds}.jsonl"
    arms = {}
    for line in open(p):
        r = json.loads(line)
        arms.setdefault(r["arm"], {})[r["example_id"]] = r
    return arms


def ratio(a, b, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    bs = [a[i].sum() / b[i].sum() for i in (rng.integers(0, len(a), len(a)) for _ in range(B))]
    return a.sum() / b.sum(), np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def delta(x, y):
    d = np.asarray(x, float) - np.asarray(y, float)
    bs = [d[rng.integers(0, len(d), len(d))].sum() for _ in range(2000)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return f"{int(d.sum()):+d} [{lo:+.0f}, {hi:+.0f}]" + ("" if lo > 0 or hi < 0 else " (incl. 0)")


for ds, name in (("gsm_held", "GSM8K"), ("he164", "HumanEval")):
    try:
        arms = load(ds)
    except FileNotFoundError:
        print(f"\n### {name}: no records yet")
        continue
    ids = sorted(set.intersection(*[set(v) for v in arms.values()]))
    rules = json.loads((REC / f"transfer_{TAG}_{ds}.rules.json").read_text())
    print(f"\n### {name} (n={len(ids)} prompts, {rules['gpu']})")
    for a in ("R0_OWN", "R0_RTX"):
        r = rules["rules"][a]
        print(f"- {LABEL[a]}: F {r['F_ms']:.1f} ms, c {r['c_ms_per_row_layer']:.4f} ms/row/layer, rejoin {r['partial_join']}, k {r['partial_every_k']}, refreshes/block {r['refreshes_per_block']}")
    print("\n| arm | s/prompt | speed vs dense [95% CI] | calls | ms/call | correct | Δ answers vs dense | Δ answers vs R0 (RTX schedule) |\n|---|---|---|---|---|---|---|---|")
    base = [arms["DENSE_C09"][i]["latency_s"] for i in ids]
    for a in ("DENSE_C09", "S32", "R0_OWN", "R0_RTX"):
        d = arms[a]
        L = [d[i]["latency_s"] for i in ids]
        sp, lo, hi = ratio(base, L)
        calls = np.mean([d[i]["nfe"] for i in ids])
        mspc = 1000 * sum(L) / sum(d[i]["nfe"] for i in ids)
        c = [d[i].get("correct") for i in ids]
        if any(x is None for x in c):
            acc, dd, dr = "pending", "", ""
        else:
            acc = f"{int(sum(c))}/{len(ids)}"
            dd = "" if a == "DENSE_C09" else delta(c, [arms["DENSE_C09"][i]["correct"] for i in ids])
            dr = "" if a == "R0_RTX" else delta(c, [arms["R0_RTX"][i]["correct"] for i in ids])
        print(f"| {LABEL[a]} | {np.mean(L):.2f} | {sp:.2f}x [{lo:.2f}, {hi:.2f}] | {calls:.1f} | {mspc:.0f} | {acc} | {dd} | {dr} |")
    own, rtx = [arms["R0_OWN"][i]["latency_s"] for i in ids], [arms["R0_RTX"][i]["latency_s"] for i in ids]
    sp, lo, hi = ratio(rtx, own)
    print(f"\nA100-derived R0 vs RTX-schedule R0, same GPU: {sp:.3f}x faster [{lo:.3f}, {hi:.3f}] (above 1 = the A100-derived schedule is faster)")
