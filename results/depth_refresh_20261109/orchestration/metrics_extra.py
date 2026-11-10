"""Extra metrics from the saved records of a depth-refresh stage (CPU only): the quantities that the related papers report besides wall-clock speed.
Usage: metrics_extra.py STAGE  (STAGE = e.g. E_he164, D_math108; reads records/stage<STAGE>[.scored].jsonl, writes metrics_stage<STAGE>.md)
Definitions (stated so that they can be compared with the papers; none is a hardware counter):
 - calls (NFE): model forward passes per question.   - ms per call: mean latency / calls.   - answer tokens/s: tokens before the first EOS divided by latency (the same answer length for every arm up to its own decode).
 - backbone TFLOPs per question: algorithmic matrix multiplications, 2 FLOPs per multiply-add, as in scripts/locking_flop_accounting.py (SureLock's convention): per layer and per computed row
   attention 4*d*T, projections 4*d*d + 4*d*d_kv, gated MLP 6*d*d_ff, with the ACTUAL per-layer row counts of every call (layer_rows); T = prompt + generated length. The output head and elementwise
   operations are excluded (the head rows per call are not logged); dense arms use T rows in all 32 layers at every call.
 - cache memory (MB): K and V of every row at every layer in bf16 (the minimum a cache needs); the engine additionally keeps each row's layer input and query (about 2x more); dense arms: 0.
 - text differs from dense C-0.9: share of questions whose generated token sequence up to the first EOS differs from the dense decode's; mean share of positions that differ (over the shorter of the two answers).
LLaDA-8B-Instruct: d = 4096, 32 layers, 32 heads, 32 kv heads, d_ff = 12288."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
D, L, DKV, DFF, EOS = 4096, 32, 4096, 12288, (126081, 126348)
stage = sys.argv[1]
scored = OUT / f"records/stage{stage}.scored.jsonl"
rows = [json.loads(x) for x in (scored if scored.exists() else OUT / f"records/stage{stage}.jsonl").read_text().splitlines() if x.strip()]
arms = []
for r in rows:
    if r["arm"] not in arms:
        arms.append(r["arm"])
by = {a: {r["example_id"]: r for r in rows if r["arm"] == a and r.get("status") == "ok"} for a in arms}
ids = sorted(set.intersection(*[set(v) for v in by.values()]))


def tflops(r):
    if r["layer_rows"] is None:                      # dense decoder: every call computes all T rows in all layers
        T = r["rows_per_call_mean"]
        per = L * T * (4 * D * T + 4 * D * D + 4 * D * DKV + 6 * D * DFF)
        return r["nfe"] * per / 1e12, int(round(T))
    T = r["layer_rows"][0][0]
    tot = 0
    for lr in r["layer_rows"]:
        tot += sum(x * (4 * D * T + 4 * D * D + 4 * D * DKV + 6 * D * DFF) for x in lr)
    return tot / 1e12, int(T)


def answer_len(r):
    t = r["tokens"]
    return next((i for i, x in enumerate(t) if x in EOS), len(t))


ref = "D:C-0.9"
lines = [f"# Extra metrics, stage {stage} (n = {len(ids)} questions; definitions in the script header)\n",
         "| arm | calls | ms per call | answer tokens/s | backbone TFLOPs / question | reduction vs dense C-0.9 | cache memory MB | text differs from dense C-0.9 | positions differing |", "|---|---|---|---|---|---|---|---|---|"]
base = None
for a in arms:
    fl, T = zip(*[tflops(by[a][i]) for i in ids])
    fl = float(np.mean(fl)); T = int(np.max(T))
    if a == ref:
        base = fl
    nfe = np.mean([by[a][i]["nfe"] for i in ids]); lat = np.mean([by[a][i]["latency_s"] for i in ids])
    tps = np.mean([answer_len(by[a][i]) / by[a][i]["latency_s"] for i in ids])
    mem = 0 if a.startswith("D:") else 2 * L * T * DKV * 2 / 2**20
    diff, frac = [], []
    for i in ids:
        x, y = by[a][i]["tokens"], by[ref][i]["tokens"]
        lx, ly = answer_len(by[a][i]), answer_len(by[ref][i])
        diff.append(x[:lx] != y[:ly])
        m = min(lx, ly)
        frac.append(sum(1 for k in range(m) if x[k] != y[k]) / max(m, 1))
    lines.append(f"| {a} | {nfe:.1f} | {1e3 * lat / nfe:.1f} | {tps:.1f} | {fl:,.0f} | {{red}} | {mem:,.0f} | {np.mean(diff):.2f} | {np.mean(frac):.2f} |")
    lines[-1] = lines[-1].replace("{red}", f"{100 * (1 - fl / base):.1f}%" if base else "-")
(OUT / f"metrics_stage{stage}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
