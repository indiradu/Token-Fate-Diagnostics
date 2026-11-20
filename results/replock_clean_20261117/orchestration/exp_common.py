"""Shared helpers for the follow-up experiments (threshold sweep, R0 ablation, noise control, extra-prompt parity). New file; no runner with pending jobs is touched."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
import numpy as np
import torch
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.refresh_rule import Setting, config_kwargs, derive

OUT = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
NT = dict(eos_truncation=False)
N_CAL = 8


def cost_params(path):
    b = json.loads(Path(path).read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    return float(F), float(c)


def rec_sel(res, tok):
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)), "prompt_len": res.prompt_length,
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "refresh_kind": list(res.refresh_kind), "rows_per_call": list(res.rows_per_call),
            "layer_rows": [list(x) for x in res.layer_rows]}


def dense(ctx, e, gen, thr):
    cfg = PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="C-block", threshold=float(thr))
    torch.manual_seed(23)
    r = decode_paper_policy(ctx["model"], ctx["tok"], C._resolve_prompt(e), e["example_id"], cfg)
    return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "prompt_len": None}


def selective(ctx, e, gen, thr, **kw):
    cfg = C.cached_cfg(f"C-{thr}", ctx, gen_length=gen, **kw, **NT)
    return rec_sel(C.decode_selective(ctx["model"], C.prompt_ids(ctx, e), cfg), ctx["tok"])


def calibrate(ctx, es, gen, thr, bench, alpha0=None, beta=None):
    """Dry run of the plain cached decode on the first N_CAL prompts -> Setting for R0 (call counts and prompt length only)."""
    cal = [selective(ctx, e, gen, thr) for e in es[:N_CAL]]
    F, c = cost_params(bench)
    st = Setting(n_layers=32, block_length=32, gen_length=gen, prompt_len=float(np.mean([r["prompt_len"] for r in cal])), calls_per_block=float(np.mean([r["nfe"] / max(1, r["dense_calls"]) for r in cal])), fixed_ms=F, row_ms=c)
    return st


def rule_of(st, alpha0=None, beta=None):
    kw = {}
    if alpha0 is not None:
        kw["alpha0"] = alpha0
    if beta is not None:
        kw["beta"] = beta
    return derive(st, **kw)


def score(ds, d, e, code_sets=("he164", "kod_held")):
    if os.environ.get("DEPTH_DEFER_SCORING") == "1" and ds in code_sets:
        return None, "deferred"
    c, s = C.score_paper_generation(d["text"], e)
    if s == "sandbox_error":
        raise SystemExit("sandbox_error while scoring: abort")
    return (None if c is None else bool(c)), s


def append(path, e, arm, ds, gen, d, **extra):
    c, s = extra.pop("_score", (None, "deferred"))
    C.append(path, dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=c, scoring_status=s, first_eos=C.first_eos(d["tokens"]), status="ok", **extra))
