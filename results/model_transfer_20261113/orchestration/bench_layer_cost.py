"""Layer-resolved cost of one forward of the selective-cache engine on this GPU (RTX 5000 Ada). Output: records/bench_layer_cost.json + a printed table.
No decoding quality involved: T = 690 rows (178 prompt + 512 answer, the HumanEval mean), random prompt ids, answer rows = MASK.
Cells: (a) dense (fresh) call; (b) query-only calls with Q rows (no join); (c) partial calls: Q = 132 query rows + the cached rows join at layer l0 (join all / blank / context).
Wall time with cuda synchronize around each call (what a decode pays) and cuda-event time; median and min of REPS after WARM."""
from __future__ import annotations

import json
import statistics as st
import time

import common as C
import torch
from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine

T, P = 690, 178
REPS, WARM = 15, 3


def timed(fn):
    for _ in range(WARM):
        fn()
    torch.cuda.synchronize()
    wall, ev = [], []
    for _ in range(REPS):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        wall.append((time.perf_counter() - t0) * 1e3)
        ev.append(s.elapsed_time(e))
    return {"wall_med_ms": st.median(wall), "wall_min_ms": min(wall), "event_med_ms": st.median(ev)}


def main():
    C.gpu_guard()
    model, tok = C.load_model()
    dev = torch.device("cuda")
    cfg = SelectiveConfig(gen_length=T - P, window=16, far_mask_cache=True, mask_id=C.MASK_ID, eos_id=C.EOS_ID, attn_impl="sdpa", log_calls=False, log_detail=False,
                          refresh="block", shape="block", block_length=32, eos_truncation=True)
    eng = SelectiveEngine(model, T, cfg)
    g = torch.Generator().manual_seed(0)
    x = torch.full((1, T), C.MASK_ID, dtype=torch.long)
    x[0, :P] = torch.randint(1000, 50000, (P,), generator=g)
    # a few written answer rows so that "context" has written rows beyond the prompt
    x[0, P:P + 40] = torch.randint(1000, 50000, (40,), generator=g)
    x = x.to(dev)
    allr = torch.arange(T, device=dev)
    blank = (x[0] == C.MASK_ID)
    out = {"T": T, "P": P, "gpu": torch.cuda.get_device_name(0), "reps": REPS, "cells": {}}
    dl = torch.arange(0, 16, device=dev)

    def dense():
        eng.forward(x, allr, 0, allr[P + 40:P + 56], allr[P + 40:P + 56], blank, fresh=True)

    out["cells"]["dense"] = timed(dense)
    dense()
    # query-only calls: Q rows = 40 written (new) + Q-40 blank rows starting at P+40 ; decode rows = first 16 blank rows
    def qrows(Q):
        r = torch.cat([allr[P + 24:P + 40], allr[P + 40:P + 40 + max(Q - 16, 1)]]) if Q > 16 else allr[P + 40:P + 40 + Q]
        return r, torch.arange(r.numel() - 16, r.numel(), device=dev) if r.numel() >= 16 else torch.arange(r.numel(), device=dev)

    for Q in (16, 64, 132, 250, 425):
        r, d = qrows(Q)
        out["cells"][f"query_only_Q{r.numel()}"] = timed(lambda r=r, d=d: eng.forward(x, r, 0, d, d, blank))
    r, d = qrows(132)
    for l0 in (0, 4, 8, 11, 16, 24):
        for jm_name, jm in (("all", None), ("blank", blank), ("context", ~blank)):
            if l0 not in (0, 11, 16) and jm_name != "all":
                continue
            out["cells"][f"partial_Q{r.numel()}_l{l0}_{jm_name}"] = timed(lambda l0=l0, jm=jm: eng.forward(x, r, 0, d, d, blank, refresh_from=l0, join_mask=jm))
    (C.OUT / "records").mkdir(exist_ok=True)
    (C.OUT / "records/bench_layer_cost.json").write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["cells"].items():
        print(f"{k:34s} wall med {v['wall_med_ms']:7.2f}  min {v['wall_min_ms']:7.2f}  event {v['event_med_ms']:7.2f} ms")


if __name__ == "__main__":
    main()
