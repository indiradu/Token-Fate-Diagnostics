"""GPU. One-word-per-call trace collection in the geometry of use (PREREGISTERED.md, part 1).
Usage: <abs venv python> collect_traces.py GEOM [SMOKE]     GEOM in {S16, S32c}; records/traces_<GEOM>/<example_id>.npz (resumable), records/collect_<GEOM>.jsonl (one line per question).
For each call and each eligible blank: the six rule inputs (the code path of lock == 'rule', SelectiveConfig.collect_rows), the proposal, the call kind; after the decode the label
y = (proposal == final token at that position). 80 questions = the 60 fit + 20 threshold questions of the ORIGINAL Lctx rule. Questions are interleaved fit/thr so a prefix is usable."""
from __future__ import annotations

import os
import sys
import time

import common as C
import numpy as np
import refit_lib as RL


def main():
    geom = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    tag = ("smoke_" if smoke else "") + geom
    tdir = C.OUT / f"records/traces_{tag}"
    tdir.mkdir(parents=True, exist_ok=True)
    log = C.OUT / f"records/collect_{tag}.jsonl"
    es = C.rows(C.OUT / "manifests/fit_thr.jsonl")
    fit, thr = [e for e in es if e["split"] == "fit"], [e for e in es if e["split"] == "thr"]
    order = []
    for i in range(max(len(fit), len(thr))):                       # interleave: 3 fit : 1 thr
        order += fit[3 * i:3 * i + 3] + thr[i:i + 1]
    order = [e for e in dict((e["example_id"], e) for e in order).values()]
    if smoke:
        order = [fit[0], thr[0]]
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    cfg = C.collect_cfg(geom)
    done = {p.stem for p in tdir.glob("*.npz")}
    started = time.time()
    for i, e in enumerate(order):
        qid = e["example_id"]
        if qid in done:
            continue
        C.gpu_guard()
        ids = C.prompt_ids(ctx, e)
        res = C.decode_selective(model, ids, cfg)
        rows = RL.labelled_rows(res.collected, res.tokens, int(ids.shape[1]), res.refresh_kind, qid)
        np.savez_compressed(tdir / f"{qid}.npz", **{k: v for k, v in rows.items()})
        keep = ~np.isin(rows["token"], [RL.EOS_ID, RL.EOT_ID])
        C.append(log, {"example_id": qid, "split": e["split"], "geom": geom, "nfe": res.nfe, "latency_s": res.latency_s, "rows": int(len(rows["y"])), "rows_non_eos": int(keep.sum()),
                       "y_mean_non_eos": float(rows["y"][keep].mean()) if keep.any() else None, "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")),
                       "partial_calls": sum(1 for k in res.refresh_kind if k == "partial"), "first_eos": C.first_eos(res.tokens), "loadavg1": os.getloadavg()[0]})
        print(f"[collect {tag}] {i + 1}/{len(order)} {qid} {e['split']} calls={res.nfe} rows={len(rows['y'])} {(time.time() - started) / 60:.1f} min", flush=True)
    print(f"[collect {tag}] done", flush=True)


if __name__ == "__main__":
    main()
