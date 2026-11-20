"""Pooled paired answer analysis (Study 5): for each arm, the change in correct answers against (a) the model's original decoder and (b) the arm's own stack's dense Conf-0.9 decode, POOLED over all model-set cells (and per model), as a share of prompts.
Pooled estimate = sum over cells of (arm correct - reference correct) / sum of prompts. Interval: 95% bootstrap resampling prompts WITHIN each cell (cells stay fixed), 4000 draws. 'excluding MoE' drops LLaDA-MoE.
Output: results/replock_clean_20261117/results_pooled_answers.md. No tolerance applied; it says only whether the interval includes 0."""
import json
from pathlib import Path

import numpy as np

REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
rng = np.random.default_rng(0)
SETS = ["he164", "kod_held", "gsm_held", "math108"]


def rows(name):
    p = REC / name.replace(".jsonl", ".scored.jsonl")
    p = p if p.exists() else REC / name
    return [json.loads(l) for l in open(p)] if p.exists() else []


def cell(tag, ds):
    arms = {}
    for r in rows(f"pd_{tag}_{ds}.jsonl"):
        arms.setdefault({"DENSE_C09": "pdDENSE"}.get(r["arm"], r["arm"]), {})[r["example_id"]] = r
    for r in rows(f"eng_{tag}_{ds}.jsonl"):
        arms.setdefault("engDENSE" if r["arm"] == "DENSE_C09" else ("ORIG" if r["arm"] == "ORIG" else r["arm"]), {})[r["example_id"]] = r
    for r in rows(f"eng_{tag}b32_{ds}.jsonl"):
        arms.setdefault("ORIG", {})[r["example_id"]] = r
    for a in ("DENSE_SDPA", "FD_DUAL", "DLLM"):
        for r in rows(f"d2c_{tag}_{ds}_{a}.jsonl"):
            arms.setdefault(a, {})[r["example_id"]] = r
    return arms


MODELS = {"LLaDA-8B": "llada8b", "LLaDA-1.5": "llada15", "Dream": "dream", "LLaDA-MoE": "moe"}
DATA = {m: {ds: cell(t, ds) for ds in SETS if cell(t, ds)} for m, t in MODELS.items()}
ARMS = {"R0 (ours)": ("R0", "engDENSE"), "Controller (ours)": ("CTL", "engDENSE"), "Elastic-Cache": ("ELASTIC", "engDENSE"), "Fast-dLLM dual": ("FD_DUAL", "DENSE_SDPA"), "dLLM-Cache": ("DLLM", "DENSE_SDPA"), "SureLock": ("SURELOCK", "pdDENSE")}


def diffs(model, arm, ref):
    out = []
    for ds, arms in DATA[model].items():
        if arm not in arms or ref not in arms:
            continue
        ids = sorted(set(arms[arm]) & set(arms[ref]))
        if any(arms[arm][i].get("correct") is None or arms[ref][i].get("correct") is None for i in ids) or not ids:
            continue
        out.append(np.array([float(arms[arm][i]["correct"]) - float(arms[ref][i]["correct"]) for i in ids]))
    return out


def pooled(vecs, B=4000):
    n = sum(len(v) for v in vecs)
    if n == 0:
        return None
    est = sum(v.sum() for v in vecs) / n
    bs = np.array([sum(v[rng.integers(0, len(v), len(v))].sum() for v in vecs) / n for _ in range(B)])
    return est, np.percentile(bs, 2.5), np.percentile(bs, 97.5), n, len(vecs)


def fmt(r):
    if r is None:
        return "–"
    est, lo, hi, n, k = r
    return f"{100 * est:+.1f} pts [{100 * lo:+.1f}, {100 * hi:+.1f}]{'' if lo > 0 or hi < 0 else ' (incl. 0)'}  n={n}, {k} cells"


md = ["# Pooled paired answer change (Study 5)", "Change in correct answers as a share of prompts (percentage points), pooled over cells; 95% bootstrap over prompts within each cell. Reference (a): the model's original decoder; (b): the arm's own stack's dense Conf-0.9 decode.", ""]
for title, refmode in (("(a) against the original decoder", "orig"), ("(b) against the same stack's dense Conf-0.9", "dense")):
    md += [f"## {title}", "| arm | all cells | excluding LLaDA-MoE | LLaDA-8B | LLaDA-1.5 | Dream | LLaDA-MoE |", "|---|---|---|---|---|---|---|"]
    for name, (arm, dense) in ARMS.items():
        ref = "ORIG" if refmode == "orig" else dense
        per = {m: diffs(m, arm, ref) for m in MODELS}
        allv = [v for m in MODELS for v in per[m]]
        nomoe = [v for m in MODELS if m != "LLaDA-MoE" for v in per[m]]
        md.append(f"| {name} | {fmt(pooled(allv))} | {fmt(pooled(nomoe))} | " + " | ".join(fmt(pooled(per[m])) for m in MODELS) + " |")
    md.append("")
(REC.parent / "results_pooled_answers.md").write_text("\n".join(md))
print("\n".join(md))
