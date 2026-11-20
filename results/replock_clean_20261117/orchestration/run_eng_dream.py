"""Study 5, Dream-v0-Instruct-7B, selective-cache engine arms interleaved in one process (word lock Conf-0.9, block 32, no first-EOS truncation, all positions decoded):
  DENSE_C09 (engine, a dense call every call), ELASTIC (Elastic-Cache reproduction), R0 (derived rule, 28 layers: start layer floor(28/3) = 9), CTL (frozen risk model, thresholds recalibrated from the first 20 prompts).
Usage: <venv python> run_eng_dream.py SET [SMOKE]   SET in he164 | kod_held | gsm_held; env MODEL_TAG=dream, BENCH_JSON, ENG_ARMS, FLOPS, PUB_LIMIT.   Records: records/eng_dream_<SET>.jsonl"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
os.environ.setdefault("MODEL_TAG", "dream")
import dream_common as D
import numpy as np
import run_rule_dream as RD
import torch
from flopcount import N_FLOPS, count
from regret_remasking.refresh_controller import RefreshController
from regret_remasking.refresh_rule import Setting, config_kwargs, derive
from regret_remasking.selective_cache import decode_selective

C = D.C
OUT = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
RISK = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/refresh_family_20261111/records/risk_model_F1.json")
NT = dict(eos_truncation=False)
CTL_SPEC = dict(rungs=("blank", "all", "dense"), top_shares=(0.3, 0.1, 0.03), i_min=3, i_max=8)
SHARES = CTL_SPEC["top_shares"]
CAL = 20


def ctl_spec(thr):
    spec = json.loads(RISK.read_text())
    spec["top_share_thresholds"] = {str(q): float(t) for q, t in zip(SHARES, thr)}
    return spec


def rec_of(res, tok, extra=None):
    d = {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": D.response_text(tok, list(res.tokens)), "prompt_len": res.prompt_length,
         "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "refresh_kind": list(res.refresh_kind), "rows_per_call": list(res.rows_per_call), "layer_rows": [list(x) for x in res.layer_rows]}
    d.update(extra or {})
    return d


def decode(model, tok, arm, e, gen, rule, thr=None):
    ids = D.prompt_ids(tok, e)
    if arm == "DENSE_C09":
        return rec_of(decode_selective(model, ids, D.cfg_for(gen, refresh="every_call", **NT)), tok)
    if arm == "ELASTIC":
        return rec_of(decode_selective(model, ids, D.cfg_for(gen, shape="window", lock="elastic", refresh="trigger", **NT)), tok)
    if arm == "S32":
        return rec_of(decode_selective(model, ids, D.cfg_for(gen, **NT)), tok)
    if arm == "R0":
        return rec_of(decode_selective(model, ids, D.cfg_for(gen, **config_kwargs(rule), **NT)), tok)
    if arm in ("CTL", "CTL_CAL"):
        spec = ctl_spec(thr if arm == "CTL" else [float("inf")] * len(SHARES))
        res = decode_selective(model, ids, D.cfg_for(gen, partial_alpha=11 / 32, refresh_ctrl=RefreshController(spec, block_length=32, **CTL_SPEC), **NT))
        kinds = {k: sum(1 for x in res.ctl_log if x["kind"] == k) for k in ("blank", "all", "dense", "first", "none")}
        return rec_of(res, tok, {"ctl_kinds": kinds, "risks": [x["risk"] for x in res.ctl_log]})
    raise SystemExit(arm)


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = RD.SETS[ds]
    es = RD.questions(ds)
    if os.environ.get("PUB_LIMIT"):
        es = es[: int(os.environ["PUB_LIMIT"])]
    FL = os.environ.get("FLOPS") == "1"
    es_run = es[:2] if smoke else (es[:N_FLOPS] if FL else es)
    arms = os.environ.get("ENG_ARMS", "DENSE_C09,ELASTIC,R0,CTL").split(",")
    C.gpu_guard()
    model, tok = D.load()
    cal = [decode(model, tok, "S32", e, gen, None) for e in es[:8]]
    b = json.loads(Path(os.environ["BENCH_JSON"]).read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    setting = Setting(n_layers=28, block_length=32, gen_length=gen, prompt_len=float(np.mean([r["prompt_len"] for r in cal])), calls_per_block=float(np.mean([r["nfe"] / max(1, r["dense_calls"]) for r in cal])),
                      fixed_ms=float(F), row_ms=float(c))
    rule = derive(setting)
    print(f"[eng dream {ds}] setting {setting}\n[eng dream {ds}] derived { {k: v for k, v in rule.items() if k != 'detail'} }", flush=True)
    thr = None
    if "CTL" in arms:
        risks = [x for e in es[:CAL] for x in decode(model, tok, "CTL_CAL", e, gen, rule)["risks"] if x is not None]
        thr = [float(np.quantile(risks, 1 - q)) for q in SHARES]
        print(f"[eng dream {ds}] controller thresholds recalibrated from {len(risks)} calls: {dict(zip(SHARES, thr))}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{'smoke_' if smoke else ''}{'flops_' if FL else ''}eng_dream_{ds}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(model, tok, arm, es_run[0], gen, rule, thr)
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
                d = dict(decode(model, tok, arm, e, gen, rule, thr), **d)
            d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
            correct, status = (None, "deferred") if (RD.DEFER and ds != "gsm_held") else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=D.first_eos(d["tokens"]),
                       status="ok", loadavg1=os.getloadavg()[0])
            if arm == "R0":
                rec["rule"] = {k: v for k, v in rule.items() if k != "detail"}
                rec["setting"] = setting.__dict__
            C.append(path, rec)
        print(f"[eng dream {ds}] {i + 1}/{len(es_run)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[eng dream {ds}] done", flush=True)


if __name__ == "__main__":
    main()
