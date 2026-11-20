"""Experiment 4 (premise P1, CPU only, descriptive): where do the published caches' lost answers start? For every arm and cell, take the tasks the same stack's dense Conf-0.9 decode answers correctly and the arm answers wrongly ('lost') and locate the FIRST token that differs from the dense decode
(up to the first EOS of either): its block index, its offset inside the 32-block, and whether it lies in the first generated block. Control: tasks whose decode diverges but is not lost. Output: results_attribution.md.
This does not identify the frozen row or layer (that needs a repair run); it shows when in a block and how early in the generation the damage starts."""
import json
from pathlib import Path

import numpy as np

REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
EOS = {"llada": (126081, 126348), "dream": (151643, 151645), "moe": (156892, 156900)}


def rows(name):
    p = REC / name.replace(".jsonl", ".scored.jsonl")
    p = p if p.exists() else REC / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def cut(t, e):
    for i, x in enumerate(t):
        if x in e:
            return t[:i]
    return t


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def cell(tag, ds):
    arms = {}
    for r in rows(f"pd_{tag}_{ds}.jsonl"):
        arms.setdefault({"DENSE_C09": "pdDENSE"}.get(r["arm"], r["arm"]), {})[r["example_id"]] = r
    for r in rows(f"eng_{tag}_{ds}.jsonl"):
        arms.setdefault("engDENSE" if r["arm"] == "DENSE_C09" else r["arm"], {})[r["example_id"]] = r
    for a in ("DENSE_SDPA", "FD_DUAL", "DLLM"):
        for r in rows(f"d2c_{tag}_{ds}_{a}.jsonl"):
            arms.setdefault(a, {})[r["example_id"]] = r
    return arms


ARMS = {"R0": "engDENSE", "CTL": "engDENSE", "ELASTIC": "engDENSE", "FD_DUAL": "DENSE_SDPA", "DLLM": "DENSE_SDPA", "SURELOCK": "pdDENSE"}
stat = {a: {"lost": [], "div_not_lost": [], "n_cells": 0} for a in ARMS}
for tag, fam in (("llada8b", "llada"), ("llada15", "llada"), ("dream", "dream"), ("moe", "moe")):
    for ds in ("he164", "kod_held", "gsm_held", "math108"):
        arms = cell(tag, ds)
        for a, ref in ARMS.items():
            if a not in arms or ref not in arms:
                continue
            for i, r in arms[a].items():
                d = arms[ref].get(i)
                if d is None or r.get("correct") is None or d.get("correct") is None:
                    continue
                fd = first_diff(cut(r["tokens"], EOS[fam]), cut(d["tokens"], EOS[fam]))
                if fd is None:
                    continue
                rec = (fd, fd // 32, fd % 32)
                if d["correct"] and not r["correct"]:
                    stat[a]["lost"].append(rec)
                elif d["correct"] == r["correct"]:
                    stat[a]["div_not_lost"].append(rec)
out = ["# Where the damage starts (descriptive)", "For tasks the same stack's dense Conf-0.9 answers correctly: first token that differs from the dense decode. 'lost' = the arm answers wrongly; 'diverged, not lost' = the decode differs but the answer status is unchanged.",
       "Offset in block 0-31: a divergence at a low offset right after a block start means the first calls of a block already differ; a high offset means staleness built up during the block.", "",
       "| arm | lost tasks | median first-diff token (lost) | median block (lost) | share in the first 2 blocks (lost) | median offset in block (lost) | share offset >= 16 (lost) | diverged-not-lost: median first-diff token | median block |", "|---|---|---|---|---|---|---|---|---|"]
for a, s in stat.items():
    L, N = np.array(s["lost"]), np.array(s["div_not_lost"])
    if len(L) == 0:
        continue
    out.append(f"| {a} | {len(L)} | {np.median(L[:, 0]):.0f} | {np.median(L[:, 1]):.0f} | {100 * np.mean(L[:, 1] < 2):.0f}% | {np.median(L[:, 2]):.0f} | {100 * np.mean(L[:, 2] >= 16):.0f}% | {np.median(N[:, 0]) if len(N) else float('nan'):.0f} | {np.median(N[:, 1]) if len(N) else float('nan'):.0f} |")
(REC.parent / "results_attribution.md").write_text("\n".join(out))
print("\n".join(out))
