"""Stage G: record, at every call of the plain S32 decode (word lock C-0.9), the cached proposals and the proposals of a dense pass over the same tokens (shadow_tap.ShadowTap). QUALITY/SIGNAL ONLY, no timing.
Usage: <abs venv python> collect_agreement.py DATASET [SMOKE]   DATASET in he164 (HumanEval 0-163, 512 tokens) | kod_tune (first 100 of kod_tune400, 512) | gsm_tune (first 100, 256). Resumable.
Records: records/agree_<DATASET>.jsonl, one per question: kinds, call_blocks, written, written_tokens, and per call {pos, cached_tok, cached_p, dense_tok, dense_p}."""
from __future__ import annotations

import sys
import time

import common as C
from regret_remasking.selective_cache import SelectiveEngine, decode_selective
from regret_remasking.shadow_tap import ShadowTap

SETS = {"he164": (None, 512), "kod_tune": ("kod_tune400", 512), "gsm_tune": ("gsm8k_tune200", 256)}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    manifest, gen = SETS[ds]
    es = C.he164() if manifest is None else [dict(e, _set=ds) for e in C.rows(C.OUT / f"manifests/{manifest}.jsonl")][:100]
    if smoke:
        es = es[:2]
    path = C.OUT / f"records/{'smoke_' if smoke else ''}agree_{ds}.jsonl"
    done = {r["example_id"] for r in C.rows(path) if r.get("status") == "ok"}
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    started = time.time()
    for i, e in enumerate(es):
        if e["example_id"] in done:
            continue
        C.gpu_guard()
        cfg = C.cached_cfg("C-0.9", ctx, log_detail=True, gen_length=gen)
        ids = C.prompt_ids(ctx, e)
        T = int(ids.shape[1]) + gen
        tap = ShadowTap(SelectiveEngine(model, T, cfg), SelectiveEngine(model, T, cfg), C.MASK_ID)
        res = decode_selective(model, ids, cfg, engine=tap)
        assert len(tap.calls) == res.nfe
        C.append(path, {"example_id": e["example_id"], "set": ds, "prompt_len": int(ids.shape[1]), "nfe": res.nfe, "kinds": res.refresh_kind, "call_blocks": res.call_blocks, "written": res.written,
                        "written_tokens": res.written_tokens, "calls": tap.calls, "tokens": list(res.tokens), "status": "ok"})
        print(f"[agree {ds}] {i + 1}/{len(es)} {e['example_id']} nfe={res.nfe} ({(time.time() - started) / 60:.1f} min)", flush=True)
    print(f"[agree {ds}] done", flush=True)


if __name__ == "__main__":
    main()
