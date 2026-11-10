"""ADDED AFTER SEEING INTERIM HUMANEVAL (not part of PREREGISTERED.md): why does the far-blank cache lose answers on code?
One GPU process, HumanEval 0-163 at gen 512, C-0.9 word lock only, the reference D:C-0.9 in the same process. Cells change ONE thing at a time:
  D:C-0.9                dense word lock only (reference)
  S32:main               S32 as in the stack (far-blank cache, block-start refresh, EOS truncation)
  S32:every+eos          dense call every call + EOS truncation  -> EOS truncation alone (no cache)
  S32:block-noeos        block-start refresh, NO EOS truncation  -> cache alone
  S32:nofar              block-start refresh, EOS; blank rows of later blocks are computed every call (only prompt + written rows cached)
  S32:block+k2 / +k4     block-start refresh AND a dense call every 2nd / 4th call
  S16:every+eos          dense call every call, window of 16 + EOS truncation -> the window decode + truncation, no cache
  S16:block-noeos        S16 cache without truncation
Run: <abs python> diag_he.py run [SMOKE]   -> records/diag_he.jsonl (resumable). Uses the frozen source_diag/ (adds the default-off refresh_every_k option).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(ROOT / "results/repfreeze_tests_20261029/runtime_cache/model_modules")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / "source_diag/src"))
sys.path.insert(0, str(HERE))
import run_cache_stack as R  # noqa: E402  (frozen source_diag is first on sys.path; its own source/ insert comes later in sys.path order, so this import re-inserts source/)

sys.path.insert(0, str(OUT / "source_diag/src"))
for m in [k for k in sys.modules if k.startswith("regret_remasking")]:
    del sys.modules[m]
import torch  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402

GEN = 512
CELLS = {
    "S32:main": dict(shape="block", lock="conf", refresh="block", eos_truncation=True),
    "S32:every+eos": dict(shape="block", lock="conf", refresh="every_call", eos_truncation=True),
    "S32:block-noeos": dict(shape="block", lock="conf", refresh="block", eos_truncation=False),
    "S32:nofar": dict(shape="block", lock="conf", refresh="block", eos_truncation=True, far_mask_cache=False),
    "S32:block+k2": dict(shape="block", lock="conf", refresh="block", eos_truncation=True, refresh_every_k=2),
    "S32:block+k4": dict(shape="block", lock="conf", refresh="block", eos_truncation=True, refresh_every_k=4),
    "S16:every+eos": dict(shape="window", lock="elastic", refresh="every_call", eos_truncation=True),
    "S16:block-noeos": dict(shape="window", lock="elastic", refresh="block", eos_truncation=False),
}
ARMS = ["D:C-0.9"] + list(CELLS)


def decode(ctx, arm, e):
    model, tok = ctx["model"], ctx["tok"]
    prompt = _resolve_prompt(e)
    if arm == "D:C-0.9":
        cfg = PaperDecodeConfig(gen_length=GEN, block_length=32, mask_id=MASK_ID, arm="C-block", threshold=0.9)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, prompt, e["example_id"], cfg)
        return list(r.token_ids), r.latency_s, r.nfe, r.text, r.total_query_rows / r.nfe, r.nfe
    ids, _ = _prepare_prompt(tok, prompt, "cuda")
    c = dict(gen_length=GEN, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=32, threshold=0.9)
    c.update(CELLS[arm])
    res = decode_selective(model, ids, SelectiveConfig(**c))
    L = len(res.layer_rows[0])
    dense = sum(1 for k in res.refresh_kind if k in ("first", "dense"))
    return list(res.tokens), res.latency_s, res.nfe, _decode_response(tok, list(res.tokens)), sum(sum(x) for x in res.layer_rows) / L / res.nfe, dense


def main():
    assert sys.argv[1] == "run"
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    path = OUT / ("records/smoke_diag_he.jsonl" if smoke else "records/diag_he.jsonl")
    R.gpu_guard()
    model, tok = R.load_local_llada(R.CHECKPOINT, "cuda", "bf16")
    model.eval()
    ctx = {"model": model, "tok": tok}
    es = R.question_set("he164")
    if smoke:
        es = es[:2]
    done = {(r["example_id"], r["arm"]) for r in R.rows(path) if r.get("status") == "ok"}
    for arm in ARMS:
        decode(ctx, arm, es[0])
    started = time.time()
    for i, e in enumerate(es):
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            if (e["example_id"], arm) in done:
                continue
            R.gpu_guard()
            tokens, lat, nfe, text, rpc, dense = decode(ctx, arm, e)
            correct, status = score_paper_generation(text, e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            first_eos = next((j for j, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)
            R.append(path, {"example_id": e["example_id"], "arm": arm, "tokens": tokens, "latency_s": lat, "nfe": nfe, "text": text, "rows_per_call_mean": rpc, "dense_calls": dense,
                            "first_eos": first_eos, "correct": bool(correct), "scoring_status": status, "status": "ok", "set": "he164", "loadavg1": os.getloadavg()[0]})
        print(f"[diag] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print("[diag] done", flush=True)


if __name__ == "__main__":
    main()
