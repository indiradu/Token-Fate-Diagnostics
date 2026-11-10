"""CPU analysis of stage G. Usage: <python> analyze_stage_g.py SET   (SET in {he164, math108})  -> results_stageG_<SET>.md and results_stageG_<SET>.json.
Reads records/stageG_<SET>.scored.jsonl when it exists (HumanEval after rescore_records.py), else records/stageG_<SET>.jsonl; refuses records whose `correct` is missing.
Bootstrap over questions (10,000 resamples, seed 23). Evaluates the three pre-registered decision rules (PREREGISTERED.md) for the set."""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[1]
S = sys.argv[1]
src = OUT / f"records/stageG_{S}.scored.jsonl"
if not src.exists():
    src = OUT / f"records/stageG_{S}.jsonl"
rows = [json.loads(x) for x in src.read_text().splitlines() if x.strip()]
rows = [r for r in rows if r.get("status") == "ok"]
if any(r.get("correct") is None for r in rows):
    raise SystemExit("some records have no `correct` (scoring was deferred): run rescore_records.py G_he164 first, on a host with the bwrap sandbox on PATH")
ARMS = [a for a in ["D:C-0.9", "S16:C-0.8", "S16:Lctx-0.8", "S16:Lctx-refit", "P4:C-0.8", "P4:Lctx-0.8", "P4:Lctx-refit"] if any(r["arm"] == a for r in rows)]
by = {a: {r["example_id"]: r for r in rows if r["arm"] == a} for a in ARMS}
ids = sorted(set.intersection(*[set(v) for v in by.values()]))
n = len(ids)
idx = np.random.default_rng(23).integers(0, n, size=(10000, n))
arr = lambda a, k: np.array([by[a][i][k] for i in ids], float)
lat = {a: arr(a, "latency_s") for a in ARMS}
cor = {a: arr(a, "correct") for a in ARMS}
ci = lambda v: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))


def code(t):
    m = re.search(r"```(?:python)?\n(.*?)(```|$)", t, re.S)
    return m.group(1) if m else t


def parses(t):
    try:
        ast.parse(code(t))
        return True
    except Exception:
        return False


def paired(a, b):
    d = cor[a] - cor[b]
    lo, hi = ci(d[idx].sum(1))
    return {"diff": int(d.sum()), "ci": [lo, hi], "lost": int(((cor[b] == 1) & (cor[a] == 0)).sum()), "gained": int(((cor[b] == 0) & (cor[a] == 1)).sum())}


def speed(a, b):
    r = lat[b][idx].mean(1) / lat[a][idx].mean(1)
    lo, hi = ci(r)
    return {"ratio": float(lat[b].mean() / lat[a].mean()), "ci": [lo, hi]}


code_set = S == "he164"
par = {a: float(np.mean([parses(by[a][i]["text"]) for i in ids])) for a in ARMS} if code_set else {}
res = {"set": S, "n": n, "arms": {}}
lines = [f"# Stage G, {S}: refit Lctx vs the original rule and C-0.8, 16-window shape (S16) and 32-block shape + blank-row refresh (P4)\n",
         f"n = {n} questions common to all arms, 512 tokens. Reference D:C-0.9 = dense word lock in the same process. Bootstrap over questions (10,000, seed 23). Resolution about +-6 answers.\n",
         "| arm | correct | net vs D:C-0.9 [CI] (lost/gained) | speed vs D:C-0.9 [CI] | calls | dense | partial | fallback share | " + ("parses | " if code_set else "") + "parity |", "|---|---|---|---|---|---|---|---|" + ("---|" if code_set else "") + "---|"]
for a in ARMS:
    pr = paired(a, "D:C-0.9")
    sp = speed(a, "D:C-0.9")
    eq = [by[a][i].get("eq_reference") for i in ids if by[a][i].get("eq_reference") is not None]
    fb = float(np.mean([by[a][i]["fallback_calls"] / by[a][i]["nfe"] for i in ids]))
    res["arms"][a] = {"correct": int(cor[a].sum()), "net_vs_dense": pr, "speed_vs_dense": sp, "calls": float(arr(a, "nfe").mean()), "dense_calls": float(arr(a, "dense_calls").mean()), "partial_calls": float(arr(a, "partial_calls").mean()),
                      "fallback_share": fb, "parse_share": par.get(a), "parity_equal": f"{sum(eq)}/{len(eq)}"}
    lines.append(f"| {a} | {int(cor[a].sum())}/{n} | {pr['diff']:+d} [{pr['ci'][0]:+.0f}, {pr['ci'][1]:+.0f}] ({pr['lost']}/{pr['gained']}) | {sp['ratio']:.2f}x [{sp['ci'][0]:.2f}, {sp['ci'][1]:.2f}] | {arr(a, 'nfe').mean():.1f} | "
                 f"{arr(a, 'dense_calls').mean():.1f} | {arr(a, 'partial_calls').mean():.1f} | {fb:.2f} | " + (f"{par[a]:.2f} | " if code_set else "") + f"{sum(eq)}/{len(eq)} |")
lines.append("\nParity column: tokens equal to the earlier-stage record of the same arm (stage 1 for D:C-0.9 / S16 arms, stage E for P4 arms) where one exists; refit arms have no earlier record.\n")
dec = {}
for shape, new in (("S16", "S16:Lctx-refit"), ("P4", "P4:Lctx-refit")):
    if new not in by:
        continue
    c, o = f"{shape}:C-0.8", f"{shape}:Lctx-0.8"
    vs_c, sp_c, vs_o, sp_o = paired(new, c), speed(new, c), paired(new, o), speed(new, o)
    d = {"refit_vs_C-0.8": {"paired": vs_c, "speed": sp_c}, "refit_vs_original": {"paired": vs_o, "speed": sp_o}, "net_vs_dense": res["arms"][new]["net_vs_dense"]["diff"], "parse_share": par.get(new)}
    d["R1_keeps_speed_advantage"] = bool(sp_c["ratio"] > 1.05 and vs_c["diff"] >= -3)
    if shape == "S16":
        d["R2_removes_S16_failure"] = bool(d["net_vs_dense"] >= -6 and (not code_set or abs(par[new] - par["D:C-0.9"]) <= 0.05 or par[new] > par["D:C-0.9"]))
        d["S16_original_net_vs_dense"] = res["arms"][o]["net_vs_dense"]["diff"]
    else:
        d["R3_cache_conditioned_refit_beats_original"] = bool(vs_o["diff"] >= 4)
    dec[new] = d
    lines.append(f"**{new}** vs {c}: {vs_c['diff']:+d} [{vs_c['ci'][0]:+.0f}, {vs_c['ci'][1]:+.0f}] answers (lost/gained {vs_c['lost']}/{vs_c['gained']}), speed {sp_c['ratio']:.3f}x [{sp_c['ci'][0]:.3f}, {sp_c['ci'][1]:.3f}]; "
                 f"vs {o}: {vs_o['diff']:+d} [{vs_o['ci'][0]:+.0f}, {vs_o['ci'][1]:+.0f}] answers (lost/gained {vs_o['lost']}/{vs_o['gained']}), speed {sp_o['ratio']:.3f}x [{sp_o['ci'][0]:.3f}, {sp_o['ci'][1]:.3f}]. "
                 + "; ".join(f"{k} = {v}" for k, v in d.items() if k.startswith("R")) + ".")
res["decisions"] = dec
(OUT / f"results_stageG_{S}.md").write_text("\n".join(lines) + "\n")
(OUT / f"results_stageG_{S}.json").write_text(json.dumps(res, indent=1) + "\n")
print("\n".join(lines))
