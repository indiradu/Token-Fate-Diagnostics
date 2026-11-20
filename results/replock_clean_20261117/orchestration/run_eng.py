"""Study 5 (clean rerun), selective-cache engine arms, interleaved per question in ONE process (same GPU state, same prompts, same word lock Conf-0.9):
  DENSE_C09  dense Conf-0.9 (paper_decode C-block): the in-process anchor, word lock only
  ELASTIC    Elastic-Cache (ICLR 2026) as reproduced in our engine: window 16, Conf-0.9 rule, attention-aware trigger gamma 0.9
  R0         our derived rule (refresh_rule.derive from L, B0, G, P, calls per block, cost model; constants never tuned)
  CTL        our learned controller (frozen risk model; budget thresholds recalibrated from the first CAL prompts of a non-LLaDA-8B model)
No first-EOS truncation in any arm (all positions decoded, as the released code and the original decoder do).
Usage: <venv python> run_eng.py SET [SMOKE]; env MODEL_TAG (manifests dir under model_transfer_20261113), MODEL_CHECKPOINT, ORIG_TAG, BENCH_JSON (cost benchmark of the model), RECAL (1: recalibrate controller thresholds), ENG_ARMS, PUB_LIMIT.
Records: results/replock_clean_20261117/records/eng_<TAG>_<SET>.jsonl; per call rows (layer_rows) and refresh kinds are stored so FLOPs can be computed exactly."""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C
import numpy as np
import run_rule as R
import torch
from flopcount import N_FLOPS, count
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.refresh_controller import RefreshController
from regret_remasking.refresh_rule import Setting, config_kwargs, derive

OUT = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
TAG = os.environ.get("ORIG_TAG", "llada8b")
RISK = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/refresh_family_20261111/records/risk_model_F1.json")
BENCH = Path(os.environ["BENCH_JSON"])
CAL = 20                                                        # prompts used to recalibrate the controller's thresholds on a new model
NT = dict(eos_truncation=False)
SHARES = R.CTL_SPEC["top_shares"]


def cost_params():
    b = json.loads(BENCH.read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    return float(F), float(c)


def ctl_spec(thresholds=None):
    spec = json.loads(RISK.read_text())
    if thresholds is not None:
        spec["top_share_thresholds"] = {str(q): float(t) for q, t in zip(SHARES, thresholds)}
    return spec


def sel_record(res, tok, extra=None):
    d = {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)), "prompt_len": res.prompt_length,
         "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "refresh_kind": list(res.refresh_kind), "rows_per_call": list(res.rows_per_call),
         "layer_rows": [list(x) for x in res.layer_rows]}
    d.update(extra or {})
    return d


def decode(ctx, arm, e, gen, rule, thr=None):
    model, tok = ctx["model"], ctx["tok"]
    if arm == "DENSE_C09":
        cfg = PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="C-block", threshold=0.9)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe}
    if arm == "ELASTIC":
        cfg = C.SelectiveConfig(gen_length=gen, window=16, far_mask_cache=True, mask_id=C.MASK_ID, eos_id=C.EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=32,
                                threshold=0.9, shape="window", lock="elastic", refresh="trigger", **NT)
        return sel_record(C.decode_selective(model, C.prompt_ids(ctx, e), cfg), tok)
    if arm == "S32":                                            # calibration only (calls per block)
        return sel_record(C.decode_selective(model, C.prompt_ids(ctx, e), C.cached_cfg("C-0.9", ctx, gen_length=gen, **NT)), tok)
    if arm == "R0":
        return sel_record(C.decode_selective(model, C.prompt_ids(ctx, e), C.cached_cfg("C-0.9", ctx, gen_length=gen, **config_kwargs(rule), **NT)), tok)
    if arm in ("CTL", "CTL_CAL"):
        spec = ctl_spec(thr if arm == "CTL" else [float("inf")] * len(SHARES))
        cfg = C.cached_cfg("C-0.9", ctx, gen_length=gen, partial_alpha=11 / 32, refresh_ctrl=RefreshController(spec, block_length=32, **R.CTL_SPEC), **NT)
        res = C.decode_selective(model, C.prompt_ids(ctx, e), cfg)
        kinds = {k: sum(1 for x in res.ctl_log if x["kind"] == k) for k in ("blank", "all", "dense", "first", "none")}
        return sel_record(res, tok, {"ctl_time_s": res.ctl_time_s, "ctl_kinds": kinds, "risks": [x["risk"] for x in res.ctl_log]})
    raise SystemExit(arm)


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = R.SETS[ds]
    es = R.questions(ds)
    if os.environ.get("PUB_LIMIT"):
        es = es[: int(os.environ["PUB_LIMIT"])]
    FL = os.environ.get("FLOPS") == "1"
    es_run = es[:2] if smoke else (es[:N_FLOPS] if FL else es)
    arms = os.environ.get("ENG_ARMS", "DENSE_C09,ELASTIC,R0,CTL").split(",")
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    cal = [decode(ctx, "S32", e, gen, None) for e in es[: R.N_CAL]]            # R0 calibration: calls per block and prompt length (no answers used)
    F, c = cost_params()
    setting = Setting(n_layers=32, block_length=32, gen_length=gen, prompt_len=float(np.mean([r["prompt_len"] for r in cal])),
                      calls_per_block=float(np.mean([r["nfe"] / max(1, r["dense_calls"]) for r in cal])), fixed_ms=F, row_ms=c)
    rule = derive(setting)
    print(f"[eng {TAG} {ds}] setting {setting}\n[eng {TAG} {ds}] derived {({k: v for k, v in rule.items() if k != 'detail'})}", flush=True)
    thr = None
    if os.environ.get("RECAL") == "1" and "CTL" in arms:                       # recalibrate the frozen risk model's budget thresholds from the first CAL prompts (clock-only trajectories)
        risks = [x for e in es[:CAL] for x in decode(ctx, "CTL_CAL", e, gen, rule)["risks"] if x is not None]
        thr = [float(np.quantile(risks, 1 - q)) for q in SHARES]
        print(f"[eng {TAG} {ds}] controller thresholds recalibrated from {len(risks)} calls of {CAL} prompts: {dict(zip(SHARES, thr))}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{'smoke_' if smoke else ''}{'flops_' if FL else ''}eng_{TAG}_{ds}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(ctx, arm, es_run[0], gen, rule, thr)                              # warm-up
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
                d = dict(decode(ctx, arm, e, gen, rule, thr), **d)
            d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            correct, status = (None, "deferred") if (R.DEFER and ds in R.CODE) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=C.first_eos(d["tokens"]),
                       status="ok", loadavg1=os.getloadavg()[0])
            if arm == "R0":
                rec["rule"] = {k: v for k, v in rule.items() if k != "detail"}
                rec["setting"] = setting.__dict__
            C.append(path, rec)
        print(f"[eng {TAG} {ds}] {i + 1}/{len(es_run)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[eng {TAG} {ds}] done", flush=True)


if __name__ == "__main__":
    main()
