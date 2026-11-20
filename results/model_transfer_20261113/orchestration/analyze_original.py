"""Stage O/B table: per set, original decoding vs dense C-0.9 vs the published rules vs the derived rule R0 (from the stage R0 records), LLaDA-8B, batch 1.
Correct answers (paired bootstrap over questions against the original), mean latency, speed vs original, calls (nfe), peak memory, share of outputs different from the original."""
import json, sys
from pathlib import Path
import numpy as np
R = Path(__file__).resolve().parent.parent / "original/records"
RF = Path(__file__).resolve().parents[2] / "refresh_family_20261111/records"
def rows(p): return [json.loads(l) for l in open(p)] if Path(p).exists() else []
def trunc(t, e): return t[: e] if e is not None else t
def load(ds):
    out = {}
    o = rows(R / f"original_llada8b_{ds}.scored.jsonl")
    for r in o: out.setdefault(r["arm"], {})[r["example_id"]] = r
    for r in rows(R / f"rule_llada8b_{ds}.scored.jsonl"):
        if r["arm"] in ("S32", "R0", "CTL", "K4"): out.setdefault(r["arm"], {})[r["example_id"]] = r
    for f in sorted(R.glob(f"published_llada8b_{ds}_*.scored.jsonl")):
        for r in rows(f): out.setdefault(r["arm"], {})[r["example_id"]] = r
    return out
def table(ds, extra=None):
    a = load(ds)
    if extra:
        for arm, d in extra.items(): a[arm] = d
    base = a["ORIG"]; ids = sorted(base)
    rng = np.random.default_rng(0)
    print(f"\n### {ds} (n={len(ids)})")
    print("| arm | correct | Δ vs original (95% CI) | s/question | speed vs original | calls | peak GiB | outputs differing from original |")
    print("|---|---|---|---|---|---|---|---|")
    for arm in ["ORIG", "D:C-0.9"] + [k for k in a if k not in ("ORIG", "D:C-0.9")]:
        d = a[arm]; ii = [i for i in ids if i in d]
        if not ii: continue
        c = np.array([d[i]["correct"] for i in ii], float); b = np.array([base[i]["correct"] for i in ii], float)
        lat = np.mean([d[i]["latency_s"] for i in ii]); lb = np.mean([base[i]["latency_s"] for i in ii])
        diff = np.mean([trunc(d[i]["tokens"], d[i].get("first_eos")) != trunc(base[i]["tokens"], base[i].get("first_eos")) for i in ii])
        bs = [(c - b)[rng.integers(0, len(ii), len(ii))].sum() for _ in range(2000)]
        print(f"| {arm} | {int(c.sum())}/{len(ii)} | {int((c-b).sum()):+d} [{np.percentile(bs,2.5):+.0f}, {np.percentile(bs,97.5):+.0f}] | {lat:.1f} | {lb/lat:.1f}x | {np.mean([d[i]['nfe'] for i in ii]):.0f} | {np.mean([d[i].get('peak_mem_gib', float('nan')) for i in ii]):.1f} | {diff:.0%} |")
if __name__ == "__main__":
    for ds in ("he164", "kod_held"): table(ds)
