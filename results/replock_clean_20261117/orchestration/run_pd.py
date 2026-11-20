"""Study 5 (clean rerun), paper_decode engine arms measured in ONE process, interleaved per question (same GPU state, same prompts, same engine):
  ORIG       the model's original decoding (B0: one token per step, block 32, all gen_length calls)
  DENSE_C09  dense Conf-0.9 word lock (C-block, threshold 0.9): word lock only, no representation cache
  SURELOCK   SureLock-style representation freeze (ICLR 2026; epsilon 0.005, freeze the 20% most confident stable generated rows, skip their Q/FFN) under the SAME Conf-0.9 word lock (arm CS-sparse)
Usage: <venv python> run_pd.py SET [SMOKE]; env MODEL_CHECKPOINT, ORIG_TAG, PUB_LIMIT, PD_ARMS (comma list; default all three).
Records: results/replock_clean_20261117/records/pd_<TAG>_<SET>.jsonl, one per (question, arm): tokens, latency_s, nfe (forward calls), text, correct (deferred for code), peak memory."""
from __future__ import annotations

import contextlib
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
import torch
from flopcount import N_FLOPS, count
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from run_rule import CODE, DEFER, SETS, questions

OUT = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
TAG = os.environ.get("ORIG_TAG", "llada8b")


def cfg_for(arm, gen):
    kw = dict(gen_length=gen, block_length=32, mask_id=C.MASK_ID)
    if arm == "ORIG":
        return PaperDecodeConfig(arm="B0", **kw)
    if arm == "DENSE_C09":
        return PaperDecodeConfig(arm="C-block", threshold=0.9, **kw)
    if arm == "SURELOCK":
        return PaperDecodeConfig(arm="CS-sparse", threshold=0.9, epsilon=0.005, confidence_percentile=20.0, freeze_scope="generated", **kw)
    raise SystemExit(arm)


def decode(ctx, arm, e, gen):
    torch.manual_seed(23)
    r = decode_paper_policy(ctx["model"], ctx["tok"], C._resolve_prompt(e), e["example_id"], cfg_for(arm, gen))
    return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = SETS[ds]
    es = questions(ds)
    if os.environ.get("PUB_LIMIT"):
        es = es[: int(os.environ["PUB_LIMIT"])]
    FL = os.environ.get("FLOPS") == "1"
    es_run = es[:2] if smoke else (es[:N_FLOPS] if FL else es)
    arms = os.environ.get("PD_ARMS", "ORIG,DENSE_C09,SURELOCK").split(",")
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{'smoke_' if smoke else ''}{'flops_' if FL else ''}pd_{TAG}_{ds}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(ctx, arm, es_run[0], gen)                        # warm-up
    started = time.time()
    for i, e in enumerate(es_run):
        rot = i % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            torch.cuda.reset_peak_memory_stats()
            d = {}
            with count(d) if FL else contextlib.nullcontext():
                d = dict(decode(ctx, arm, e, gen), **d)
            d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            correct, status = (None, "deferred") if (DEFER and ds in CODE) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            C.append(path, dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status,
                                first_eos=C.first_eos(d["tokens"]), status="ok", loadavg1=os.getloadavg()[0]))
        print(f"[pd {TAG} {ds}] {i + 1}/{len(es_run)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[pd {TAG} {ds}] done", flush=True)


if __name__ == "__main__":
    main()
