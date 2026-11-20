"""Stage O: the model's ORIGINAL decoding (no acceleration) against the dense C-0.9 word-lock decode, per question set, LLaDA family.
Original decoding = arm B0 of paper_decode: steps = generation length (one token per step), low-confidence remasking, semi-autoregressive blocks of 32, temperature 0, no early stop (all gen_length calls).
Usage: <abs venv python> run_original.py SET [SMOKE]   SET in he164 | kod_held | gsm_held | math108.   Env: MODEL_TAG (manifests dir), MODEL_CHECKPOINT (weights), ORIG_TAG (records name), ORIG_LIMIT (first N questions).
Arms rotated per question (warm-up on question 0): ORIG (original decoding) and D:C-0.9 (dense word-lock decode, so the speed ratio is measured in one process on one GPU state).
Records: records/original_<ORIG_TAG>_<SET>.jsonl, one per (question, arm): tokens, latency, nfe, text, correct (deferred for code: no bwrap on the nodes), peak memory."""
from __future__ import annotations

import os
import sys
import time

import common as C
import torch
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from run_rule import CODE, DEFER, SETS, questions

TAG = os.environ.get("ORIG_TAG", os.environ.get("MODEL_TAG", "llada15"))


def decode(ctx, arm, e, gen):
    cfg = (PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="B0") if arm == "ORIG"
           else PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="C-block", threshold=0.9))
    torch.manual_seed(23)
    r = decode_paper_policy(ctx["model"], ctx["tok"], C._resolve_prompt(e), e["example_id"], cfg)
    return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = SETS[ds]
    es = questions(ds)
    if os.environ.get("ORIG_LIMIT"):
        es = es[: int(os.environ["ORIG_LIMIT"])]
    es_run = es[:2] if smoke else es
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    arms = ["ORIG", "D:C-0.9"]
    path = C.OUT / f"records/{'smoke_' if smoke else ''}original_{TAG}_{ds}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(ctx, arm, es_run[0], gen)
    started = time.time()
    for i, e in enumerate(es_run):
        rot = i % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            torch.cuda.reset_peak_memory_stats()
            d = decode(ctx, arm, e, gen)
            d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            correct, status = (None, "deferred") if (DEFER and ds in CODE) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            C.append(path, dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status,
                                first_eos=C.first_eos(d["tokens"]), status="ok", loadavg1=os.getloadavg()[0]))
        print(f"[original {TAG} {ds}] {i + 1}/{len(es_run)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[original {TAG} {ds}] done", flush=True)


if __name__ == "__main__":
    main()
