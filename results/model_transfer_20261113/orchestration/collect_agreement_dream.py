"""Stage G3 on Dream: the shadow-pass records (cached call vs a dense pass over the same tokens, plain S32, C-0.9) as collect_agreement.py. Usage: collect_agreement_dream.py DATASET [SMOKE] (he164 | kod_tune | gsm_tune)."""
from __future__ import annotations

import sys
import time

import dream_common as D
from regret_remasking.selective_cache import SelectiveEngine, decode_selective
from regret_remasking.shadow_tap import ShadowTap

C = D.C
SETS = {"he164": (None, 512), "kod_tune": ("kod_tune400", 512), "gsm_tune": ("gsm8k_tune200", 256)}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    manifest, gen = SETS[ds]
    es = C.he164() if manifest is None else [dict(e, _set=ds) for e in C.rows(C.OUT / f"manifests/{manifest}.jsonl")][:100]
    es = es[:2] if smoke else es
    path = C.OUT / f"records/{'smoke_' if smoke else ''}agree_{ds}.jsonl"
    done = {r["example_id"] for r in C.rows(path) if r.get("status") == "ok"}
    C.gpu_guard()
    model, tok = D.load()
    t0 = time.time()
    for i, e in enumerate(es):
        if e["example_id"] in done:
            continue
        cfg = D.cfg_for(gen, log_detail=True)
        ids = D.prompt_ids(tok, e)
        T = int(ids.shape[1]) + gen
        tap = ShadowTap(SelectiveEngine(model, T, cfg), SelectiveEngine(model, T, cfg), D.MASK)
        res = decode_selective(model, ids, cfg, engine=tap)
        assert len(tap.calls) == res.nfe
        C.append(path, {"example_id": e["example_id"], "set": ds, "prompt_len": int(ids.shape[1]), "nfe": res.nfe, "kinds": res.refresh_kind, "call_blocks": res.call_blocks, "written": res.written,
                        "written_tokens": res.written_tokens, "calls": tap.calls, "tokens": list(res.tokens), "status": "ok"})
        print(f"[agree dream {ds}] {i + 1}/{len(es)} {e['example_id']} nfe={res.nfe} ({(time.time() - t0) / 60:.1f} min)", flush=True)
    print(f"[agree dream {ds}] done", flush=True)


if __name__ == "__main__":
    main()
