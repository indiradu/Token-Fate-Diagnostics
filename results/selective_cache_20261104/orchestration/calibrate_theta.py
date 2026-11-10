"""Choose the row-selective refresh's score layer and exposure thresholds from the Part A TRAIN questions only (CPU).

Rules (fixed in PREREGISTERED.md before any Part C decode):
  rows considered = rows the engine would NOT recompute anyway: prompt, written, blank-beyond-window, blank-in-later-block (class window excluded);
  label = K/V drift (mean over layers 22-31 of the combined K/V relative L2) above 0.10, pooled over the snapshot ages {1,2,4,8,16};
  score = E_cq (cached-query exposure, accumulated since the snapshot) at one layer a; score layer = the smallest a in {2,4,6,8,10,12,16} whose AUROC is within 0.02 of the best;
  theta_rXX = the threshold on E_cq at that layer at which XX% of the stale rows (label 1) are above it; XX in {60, 80, 90} (generation 512 data only: Part C runs at 512).
Writes partC/params.json and partC/calibration.json (AUROC by layer, precision, share of cached rows above the threshold).
Usage: <abs python> calibrate_theta.py [--gen 512]"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
sys.path.insert(0, str(HERE))
from partA_analyze import auroc, load_dir  # noqa: E402
from partA_capture import AGES  # noqa: E402

CANDIDATES = (2, 4, 6, 8, 10, 12, 16)
RECALLS = (60, 80, 90)
DEEP = list(range(22, 32))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="512")
    ap.add_argument("--dir", default=str(OUT / "partA"))
    ap.add_argument("--out", default=str(OUT / "partC"))
    a = ap.parse_args()
    qs = load_dir(Path(a.dir) / f"g{a.gen}")
    drift, E, cls, age = [], {l: [] for l in CANDIDATES}, [], []
    for z in qs.values():
        valid = z["s_valid"].astype(bool)
        keep = z["s_cls"][:, None] != 2          # not the window class
        ni, ai = np.nonzero(valid & keep)
        sv = z["s_sv"]
        drift.append(sv[DEEP][:, ni, ai, 0].mean(0))
        for l in CANDIDATES:
            E[l].append(sv[l, ni, ai, 4])
        cls.append(z["s_cls"][ni])
        age.append(np.array(AGES)[ai])
    drift = np.concatenate(drift)
    E = {l: np.concatenate(v) for l, v in E.items()}
    lab = drift > 0.10
    au = {l: auroc(E[l], lab) for l in CANDIDATES}
    best = max(au.values())
    layer = min(l for l in CANDIDATES if au[l] >= best - 0.02)
    out = {"gen": a.gen, "n_pairs": int(len(lab)), "stale_share": float(lab.mean()), "auroc_by_score_layer": au, "score_layer": layer, "thetas": {}, "table": {}}
    agev = np.concatenate(age)
    for r in RECALLS:
        theta = float(np.quantile(E[layer][lab], 1 - r / 100.0))
        above = E[layer] > theta
        out["thetas"][f"r{r}"] = theta
        out["table"][f"r{r}"] = {"theta": theta, "recall": float(above[lab].mean()), "precision": float(lab[above].mean()) if above.any() else None,
                                 "share_of_cached_pairs_above": float(above.mean()),
                                 "recall_by_age": {str(s): float(above[lab & (agev == s)].mean()) if (lab & (agev == s)).any() else None for s in AGES},
                                 "share_above_by_age": {str(s): float(above[agev == s].mean()) for s in AGES}}
    Path(a.out).mkdir(exist_ok=True)
    (Path(a.out) / "calibration.json").write_text(json.dumps(out, indent=1) + "\n")
    (Path(a.out) / "params.json").write_text(json.dumps({"score_layer": layer, "score_heads": 0, "thetas": out["thetas"], "fit": f"Part A train questions, gen {a.gen}, drift>0.10 at layers 22-31"}, indent=1) + "\n")
    print(json.dumps({k: out[k] for k in ("n_pairs", "stale_share", "auroc_by_score_layer", "score_layer", "thetas")}, indent=1))
    print(json.dumps(out["table"], indent=1))


if __name__ == "__main__":
    main()
