"""Stage M2 (PREREGISTERED.md): the UNCHANGED derived rule on Dream-v0-Instruct-7B against plain S32, K4, the fixed P4 and the dense decode (the engine with a dense call every call: token-identical to the plain dense loop, see parity.json).
Usage: <abs venv python> run_rule_dream.py SET [SMOKE]   SET in he164 | kod_held | gsm_held (word lock C-0.9, 28 layers: R0's start layer is floor(28/3) = 9; the fixed P4's is floor(11/32 x 28) = 9 as well).
Calibration: the plain cached decode of the first 8 prompts gives calls per block and the mean prompt length; the cost parameters come from dream/records/bench_layer_cost.json (dream_parity.py). Records: records/rule_<SET>_C-0.9.jsonl. Scoring deferred for code."""
from __future__ import annotations

import json
import os
import sys
import time

import dream_common as D
import numpy as np
from regret_remasking.refresh_rule import Setting, config_kwargs, derive
from regret_remasking.selective_cache import decode_selective

C = D.C
DEFER = os.environ.get("DEPTH_DEFER_SCORING") == "1"
SETS = {"he164": 512, "kod_held": 512, "gsm_held": 256}
LOCK = "C-0.9"
FIXED = {"S32": {}, "K4": dict(refresh_every_k=4), "P4": dict(partial_every_k=4, partial_alpha=11 / 32, partial_join="blank")}


def questions(ds):
    if ds == "he164":
        return C.he164()
    if ds == "kod_held":
        return [dict(e, _set=ds) for e in C.rows(C.OUT / "manifests/kod_held400.jsonl")][:150]
    return [dict(e, _set=ds) for e in C.rows(C.OUT / "manifests/gsm8k_tune200.jsonl")][100:200]


def cost_params():
    b = json.loads((C.OUT / "records/bench_layer_cost.json").read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    return float(F), float(c)


def decode(model, tok, arm, e, gen, rule):
    kw = dict(refresh="every_call") if arm.startswith("D:") else config_kwargs(rule) if arm == "R0" else FIXED[arm]
    res = decode_selective(model, D.prompt_ids(tok, e), D.cfg_for(gen, **kw))
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": D.response_text(tok, list(res.tokens)), "prompt_len": res.prompt_length,
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "partial_calls": sum(1 for k in res.refresh_kind if k == "partial")}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = SETS[ds]
    es = questions(ds)
    cal, run = es[:8], (es[:2] if smoke else es)
    C.gpu_guard()
    model, tok = D.load()
    cc = [decode(model, tok, "S32", e, gen, None) for e in cal]
    F, c = cost_params()
    setting = Setting(n_layers=28, block_length=32, gen_length=gen, prompt_len=float(np.mean([r["prompt_len"] for r in cc])),
                      calls_per_block=float(np.mean([r["nfe"] / max(1, r["dense_calls"]) for r in cc])), fixed_ms=F, row_ms=c)
    rule = derive(setting)
    print(f"[R0 dream {ds}] setting {setting}\n[R0 dream {ds}] derived { {k: v for k, v in rule.items() if k != 'detail'} }", flush=True)
    path = C.OUT / f"records/{'smoke_' if smoke else ''}rule_{ds}_{LOCK}.jsonl"
    arms = [f"D:{LOCK}", "S32", "K4", "P4", "R0"]
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(model, tok, arm, run[0], gen, rule)
    started = time.time()
    for i, e in enumerate(run):
        rot = i % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            d = decode(model, tok, arm, e, gen, rule)
            correct, status = (None, "deferred") if (DEFER and ds != "gsm_held") else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=e["example_id"], arm=arm, set=ds, lock=LOCK, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=D.first_eos(d["tokens"]),
                       status="ok", loadavg1=os.getloadavg()[0])
            if arm == "R0":
                rec["rule"] = {k: v for k, v in rule.items() if k != "detail"}
                rec["setting"] = setting.__dict__
            C.append(path, rec)
        print(f"[rule dream {ds}] {i + 1}/{len(run)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[rule dream {ds}] done", flush=True)


if __name__ == "__main__":
    main()
