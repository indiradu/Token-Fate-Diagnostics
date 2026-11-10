"""GPU. One-word-per-call trace collection in the R-S32c geometry with the extra features (PREREGISTERED.md).
Usage: <abs venv python> collect_traces.py [SMOKE]     records/traces_S32c/<example_id>.npz (resumable), records/collect_S32c.jsonl (one line per question).
Per call and eligible blank: the six rule inputs + the 15 extra features of lock_features (SelectiveConfig.collect_rows + collect_extra), the proposal, the call kind; after the decode the label
y = (proposal == final token). 80 questions = the 60 fit + 20 threshold questions of the ORIGINAL Lctx rule, interleaved fit/thr. Check V1: the decode must reproduce the refit's collection
(results/lctx_refit_20261110/records/traces_S32c): tokens and the six inputs are compared per question; on the first 3 questions decoded IN THIS PROCESS a trajectory mismatch aborts (as the parity checks of the other stages); a V2 mismatch (derivable extras vs the log) is logged and the collection continues."""
from __future__ import annotations

import os
import sys
import time

import common as C
import features_lib as FL
import numpy as np


def compare_with_refit(qid: str, rows: dict, tokens: list[int]) -> dict:
    p = C.REFIT_DIR / f"records/traces_S32c/{qid}.npz"
    if not p.exists():
        return {"refit_trace": False}
    z = np.load(p, allow_pickle=False)
    same_shape = len(z["position"]) == len(rows["position"])
    out = {"refit_trace": True, "same_rows": bool(same_shape)}
    if same_shape:
        out["same_trajectory"] = bool(np.array_equal(z["call"], rows["call"]) and np.array_equal(z["position"], rows["position"]) and np.array_equal(z["token"], rows["token"]))
        out["six_bit_equal"] = bool(all(np.array_equal(z[f], rows[f]) for f in FL.FEATURES))
        out["six_max_abs_diff"] = float(max(np.abs(z[f].astype(np.float64) - rows[f].astype(np.float64)).max() for f in FL.FEATURES))
        out["y_equal"] = bool(np.array_equal(z["y"], rows["y"]))
    else:
        out["same_trajectory"] = False
    return out


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "SMOKE"
    tag = ("smoke_" if smoke else "") + "S32c"
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
    cfg = C.collect_cfg()
    done = {p.stem for p in tdir.glob("*.npz")}
    started = time.time()
    decoded = 0
    for i, e in enumerate(order):
        qid = e["example_id"]
        if qid in done:
            continue
        C.gpu_guard()
        ids = C.prompt_ids(ctx, e)
        res = C.decode_selective(model, ids, cfg)
        rows = FL.labelled_rows(res.collected, res.tokens, int(ids.shape[1]), res.refresh_kind, qid)
        chk = compare_with_refit(qid, rows, res.tokens)
        dchk = FL.derive_check(rows)
        decoded += 1
        if decoded <= 3 and (chk.get("refit_trace") and not chk.get("same_trajectory")):
            raise SystemExit(f"V1 failed: {qid} does not reproduce the refit's R-S32c trajectory ({chk}); nothing further collected")
        if not dchk["ok"]:          # V2 logs and continues (Amendment 2): the data are kept, fit_ablation.py recomputes V2 on all traces and G0 fails the gate closed if it is real
            print(f"[collect {tag}] WARNING V2 mismatch on {qid}: {dchk['mismatches']}", flush=True)
        np.savez_compressed(tdir / f"{qid}.npz", **{k: v for k, v in rows.items()})
        keep = ~np.isin(rows["token"], [FL.EOS_ID, FL.EOT_ID])
        C.append(log, {"example_id": qid, "split": e["split"], "nfe": res.nfe, "latency_s": res.latency_s, "rows": int(len(rows["y"])), "rows_non_eos": int(keep.sum()),
                       "y_mean_non_eos": float(rows["y"][keep].mean()) if keep.any() else None, "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")),
                       "partial_calls": sum(1 for k in res.refresh_kind if k == "partial"), "first_eos": C.first_eos(res.tokens), "loadavg1": os.getloadavg()[0],
                       "v1_vs_refit": chk, "v2_derive_ok": dchk["ok"]})
        print(f"[collect {tag}] {i + 1}/{len(order)} {qid} {e['split']} calls={res.nfe} rows={len(rows['y'])} V1={chk.get('six_bit_equal')} V2={dchk['ok']} {(time.time() - started) / 60:.1f} min", flush=True)
    print(f"[collect {tag}] done", flush=True)


if __name__ == "__main__":
    main()
