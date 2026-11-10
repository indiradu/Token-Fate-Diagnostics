"""CPU analysis of records/stage<STAGE>.jsonl (+ records/bench_layer_cost.json). Usage: analyze_depth_cells.py STAGE -> results_stage<STAGE>.md.
Per arm: correct, net vs the in-process dense C-0.9 [95% bootstrap over questions, 10,000, seed 23] with lost/gained, paired gain over S32:main, speed vs D:C-0.9 and vs S32:main
(ratio of mean latency, bootstrap over questions), calls, dense/partial calls, mean rows per call, and the cost-model time predicted from the logged per-layer row counts."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
stage = sys.argv[1]
_scored = OUT / f"records/stage{stage}.scored.jsonl"
rows = [json.loads(x) for x in (_scored if _scored.exists() else OUT / f"records/stage{stage}.jsonl").read_text().splitlines() if x.strip()]
if any(r.get("correct") is None for r in rows if r.get("status") == "ok"):
    raise SystemExit("some records have no `correct` (scoring was deferred): run rescore_depth_records.py first, on a host with the bwrap sandbox")
bench = json.loads((OUT / "records/bench_layer_cost.json").read_text())["cells"]
arms = []
for r in rows:
    if r["arm"] not in arms:
        arms.append(r["arm"])
by = {a: {r["example_id"]: r for r in rows if r["arm"] == a and r.get("status") == "ok"} for a in arms}
ids = sorted(set.intersection(*[set(v) for v in by.values()]))
n = len(ids)
rng = np.random.default_rng(23)
B = 10000
idx = rng.integers(0, n, size=(B, n))


def arr(a, k):
    return np.array([by[a][i][k] for i in ids], float)


# cost model: call time = F + c * (mean over layers of rows computed), fitted on the benchmark's dense and query-only cells (wall median)
pts = []
for k, v in bench.items():
    if k == "dense":
        pts.append((690.0, v["wall_med_ms"]))
    elif k.startswith("query_only_Q"):
        pts.append((float(k.split("Q")[1]), v["wall_med_ms"]))
A = np.array([[1.0, p[0]] for p in pts])
F, c = np.linalg.lstsq(A, np.array([p[1] for p in pts]), rcond=None)[0]


def predicted_ms(rec):
    if rec["layer_rows"] is None:
        return None
    return float(sum(F + c * (sum(lr) / len(lr)) for lr in rec["layer_rows"]))


lines = [f"# Depth-refresh cells, stage {stage}\n",
         f"n = {n} questions common to all arms. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000 resamples, seed 23). Cost model: call = {F:.1f} ms + {c:.4f} ms x (mean rows over layers), fitted on the layer benchmark (dense + query-only cells).\n",
         "| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | paired vs S32:main [CI] | speed vs D:C-0.9 [CI] | speed vs S32:main | calls | dense | partial | mean rows (layer-avg) | model time / measured |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
ref = arr("D:C-0.9", "correct")
def base_of(a):                       # the plain-S32 arm of the same word lock (stage E) or S32:main
    if "S32:main" in by:
        return "S32:main"
    return "S32:" + a.split(":")[1] if ":" in a and ("S32:" + a.split(":")[1]) in by else None


main = True
lat = {a: arr(a, "latency_s") for a in arms}
for a in arms:
    cr = arr(a, "correct")
    d = cr - ref
    boots = d[idx].sum(1)
    lost, gained = int(((ref == 1) & (cr == 0)).sum()), int(((ref == 0) & (cr == 1)).sum())
    net = f"{int(d.sum()):+d} [{np.percentile(boots, 2.5):+.0f}, {np.percentile(boots, 97.5):+.0f}] ({lost}/{gained})"
    b = base_of(a)
    if b is not None and a != b and not a.startswith("D:"):
        dm = (cr - arr(b, "correct"))
        bm = dm[idx].sum(1)
        pm = f"{int(dm.sum()):+d} [{np.percentile(bm, 2.5):+.0f}, {np.percentile(bm, 97.5):+.0f}]"
    else:
        pm = "-"
    sp = lat["D:C-0.9"][idx].mean(1) / lat[a][idx].mean(1)
    spm = lat[b].mean() / lat[a].mean() if b is not None else float("nan")
    pred = [predicted_ms(by[a][i]) for i in ids]
    pr = f"{np.mean(pred) / 1e3 / lat[a].mean():.2f}" if pred[0] is not None else "-"
    lines.append(f"| {a} | {int(cr.sum())}/{n} | {net} | {pm} | {lat['D:C-0.9'].mean() / lat[a].mean():.2f}x [{np.percentile(sp, 2.5):.2f}, {np.percentile(sp, 97.5):.2f}] | {spm:.2f}x | "
                 f"{arr(a, 'nfe').mean():.1f} | {arr(a, 'dense_calls').mean():.1f} | {arr(a, 'partial_calls').mean():.1f} | {arr(a, 'rows_per_call_mean').mean():.0f} | {pr} |")
eqs = {a: [by[a][i].get("eq_s32_stage1") for i in ids] for a in arms if a == "S32:main" or a.startswith("S32:")}
lines.append("\nParity with the stage-1 S32 tokens: " + "; ".join(f"{a} {sum(1 for e in v if e)}/{len(v)}" for a, v in eqs.items()) + ". Speed intervals under about 5% are not resolved (one timed repeat).")
if "D:Naive" in by:                    # stage F: the naive-diffusion family is read against ITS dense reference
    lines += ["", "Naive-diffusion family, reference D:Naive (dense, one token per call, no truncation):", "", "| arm | correct | net vs D:Naive [CI] (lost/gained) | speed vs D:Naive [CI] | calls | dense | partial |", "|---|---|---|---|---|---|---|"]
    rn = arr("D:Naive", "correct")
    for a in arms:
        if not a.endswith(":Naive"):
            continue
        cr = arr(a, "correct"); d = cr - rn; bo = d[idx].sum(1)
        sp = lat["D:Naive"][idx].mean(1) / lat[a][idx].mean(1)
        lines.append(f"| {a} | {int(cr.sum())}/{n} | {int(d.sum()):+d} [{np.percentile(bo, 2.5):+.0f}, {np.percentile(bo, 97.5):+.0f}] ({int(((rn == 1) & (cr == 0)).sum())}/{int(((rn == 0) & (cr == 1)).sum())}) | "
                     f"{lat['D:Naive'].mean() / lat[a].mean():.2f}x [{np.percentile(sp, 2.5):.2f}, {np.percentile(sp, 97.5):.2f}] | {arr(a, 'nfe').mean():.1f} | {arr(a, 'dense_calls').mean():.1f} | {arr(a, 'partial_calls').mean():.1f} |")
(OUT / f"results_stage{stage}.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
