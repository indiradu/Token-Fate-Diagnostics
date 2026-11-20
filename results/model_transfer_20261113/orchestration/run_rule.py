"""Stage R0: the derived rule (refresh_rule.derive) against plain S32, K4, the fixed P4 and the dense decode, per SETTING = (question set, word lock). QUALITY AND SPEED.
Usage: <abs venv python> run_rule.py SET LOCK [SMOKE]   SET in he164 | kod_held | gsm_held | math108; LOCK in C-0.9 | C-0.8.
Per setting: (1) calibration: the plain cached decode (S32) of the first 8 questions of the set gives calls per block and the mean prompt length (call counts only, no answers); (2) derive the rule's hyperparameters
from L, B0, G, P, calls per block and the cost parameters F, c of records/bench_layer_cost.json (as analyze_depth_cells.py fits them); (3) arms D:<lock>, S32, K4, P4, R0 and CTL (the learned controller of stage C, setting frozen by stage C1) on the set, rotated per question, warm-up on question 0.
Records: records/rule_<SET>_<LOCK>.jsonl (the R0 records carry the derived rule). Scoring: HumanEval/KodCode deferred (no bwrap on the nodes), GSM8K and MATH in process."""
from __future__ import annotations

import json
import os
import sys
import time

import common as C
import numpy as np
import torch
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.refresh_rule import Setting, config_kwargs, derive

DEFER = os.environ.get("DEPTH_DEFER_SCORING") == "1"
SETS = {"he164": 512, "kod_held": 512, "gsm_held": 256, "math108": 512}
CODE = {"he164", "kod_held"}
N_CAL = 8
FIXED = {"S32": {}, "K4": dict(refresh_every_k=4), "P4": dict(partial_every_k=4, partial_alpha=11 / 32, partial_join="blank")}
AF_POOL = {"he164": "kod_tune", "kod_held": "kod_tune", "gsm_held": "gsm_tune", "math108": "gsm_tune"}      # stage AF2: the autofit report whose pick is evaluated on each set (pool the pick was fitted on)
CTL_SPEC = dict(rungs=("blank", "all", "dense"), top_shares=(0.3, 0.1, 0.03), i_min=3, i_max=8)      # stage C: the controller setting chosen by the pre-registered rule on stage C1 (C_q30_L3), frozen; risk model fitted on KodCode-tune C-0.9 stage G records


def questions(ds):
    if ds == "he164":
        return C.he164()
    if ds == "kod_held":
        return [dict(e, _set=ds) for e in C.rows(C.OUT / "manifests/kod_held400.jsonl")][:150]
    if ds == "gsm_held":
        return [dict(e, _set=ds) for e in C.rows(C.OUT / "manifests/gsm8k_tune200.jsonl")][100:200]
    l5, l14 = C.rows(C.OUT / "manifests/math_l5.jsonl"), C.rows(C.OUT / "manifests/math_l14.jsonl")
    return [dict(e, _set=ds, _stratum="L5") for e in l5] + [dict(e, _set=ds, _stratum="L1-4") for e in l14]


def cost_params():
    b = json.loads((C.OUT / "records/bench_layer_cost.json").read_text())["cells"]
    pts = [(690.0, v["wall_med_ms"]) if k == "dense" else (float(k.split("Q")[1]), v["wall_med_ms"]) for k, v in b.items() if k == "dense" or k.startswith("query_only_Q")]
    F, c = np.linalg.lstsq(np.array([[1.0, p[0]] for p in pts]), np.array([p[1] for p in pts]), rcond=None)[0]
    return float(F), float(c)


def decode(ctx, arm, e, gen, lock, rule):
    model, tok = ctx["model"], ctx["tok"]
    if arm.startswith("D:"):
        cfg = PaperDecodeConfig(gen_length=gen, block_length=32, mask_id=C.MASK_ID, arm="C-block", threshold=float(lock.split("-")[1]))
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "partial_calls": 0}
    if arm == "AF":
        kw = dict(AF_POLICY)
    elif arm == "CTL":
        from regret_remasking.refresh_controller import RefreshController
        kw = dict(partial_alpha=11 / 32, refresh_ctrl=RefreshController(C.OUT / "records/risk_model_F1.json", block_length=32, **CTL_SPEC))
    else:
        kw = config_kwargs(rule) if arm == "R0" else FIXED[arm]
    cfg = C.cached_cfg(lock, ctx, gen_length=gen, **kw)
    res = C.decode_selective(model, C.prompt_ids(ctx, e), cfg)
    ctl = {"ctl_time_s": res.ctl_time_s, "ctl_kinds": {k: sum(1 for x in res.ctl_log if x["kind"] == k) for k in ("blank", "all", "dense", "first", "none")}} if arm == "CTL" else {}
    return {**ctl, "tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)), "prompt_len": res.prompt_length,
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "partial_calls": sum(1 for k in res.refresh_kind if k == "partial")}


AF_POLICY = {}


def main():
    ds, lock = sys.argv[1], sys.argv[2]
    smoke = len(sys.argv) > 3 and sys.argv[3] == "SMOKE"
    gen = SETS[ds]
    es = questions(ds)
    cal, es_run = es[:N_CAL], (es[:2] if smoke else es)
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    cal_calls = []
    for e in cal:
        r = decode(ctx, "S32", e, gen, lock, None)
        cal_calls.append((r["nfe"] / max(1, r["dense_calls"]), r["prompt_len"]))
    F, c = cost_params()
    setting = Setting(n_layers=32, block_length=32, gen_length=gen, prompt_len=float(np.mean([x[1] for x in cal_calls])), calls_per_block=float(np.mean([x[0] for x in cal_calls])), fixed_ms=F, row_ms=c)
    rule = derive(setting)
    print(f"[R0 {ds} {lock}] setting {setting}\n[R0 {ds} {lock}] derived {({k: v for k, v in rule.items() if k != 'detail'})}", flush=True)
    tag = os.environ.get("RULE_TAG", "")
    path = C.OUT / f"records/{'smoke_' if smoke else ''}rule_{tag + '_' if tag else ''}{ds}_{lock}.jsonl"
    arms = [f"D:{lock}", "S32", "K4", "P4", "R0", "CTL"]
    if os.environ.get("RULE_ARMS"):                                              # e.g. RULE_ARMS=K4,AF: dense plus these arms (AF = the autofit pick of the pool in AF_POOL, from its report)
        arms = [f"D:{lock}"] + os.environ["RULE_ARMS"].split(",")
        if "AF" in arms:
            rep_ = json.loads((C.OUT / f"records/autofit_{AF_POOL[ds]}_{lock}.report.json").read_text())
            AF_POLICY.update(rep_["pareto"][0]["policy"])
            print(f"[AF] pick for {ds}: {rep_['pareto'][0]['name']} {AF_POLICY}", flush=True)
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(ctx, arm, es_run[0], gen, lock, rule)
    started = time.time()
    for i, e in enumerate(es_run):
        rot = i % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            d = decode(ctx, arm, e, gen, lock, rule)
            correct, status = (None, "deferred") if (DEFER and ds in CODE) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=e["example_id"], arm=arm, set=ds, lock=lock, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=C.first_eos(d["tokens"]),
                       status="ok", loadavg1=os.getloadavg()[0])
            if arm == "R0":
                rec["rule"] = {k: v for k, v in rule.items() if k != "detail"}
                rec["setting"] = setting.__dict__
            C.append(path, rec)
        print(f"[rule {ds} {lock}] {i + 1}/{len(es_run)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[rule {ds} {lock}] done", flush=True)


if __name__ == "__main__":
    main()
