"""TraceLock (the published learned word lock) under our representation caches. Usage: <abs venv python> run_trace_cells.py STAGE [SMOKE]
STAGE: PARITY (engine dense trace lock vs paper_decode T-block on 3 real questions; aborts with exit code 3 on any mismatch), SEL (matched-confidence threshold selection on 40 questions disjoint from every
evaluation set), M512 (MATH 108 @512, scored in-process), P256 (the earlier 124-case panel: HumanEval 0-69 + MATH level-5 54 @256), H512 (HumanEval 164 @512, scoring DEFERRED: no bwrap on the host).
Arms are '<cache>:<word lock>': cache D (dense decoder, published moving 32-window for TraceLock), DB (dense decoder, 32-block geometry = paper_decode T-block), S32 / K4 / P4 (selective-cache engine),
word lock TraceLock | C-0.9 | C-matched (threshold chosen by SEL) | C-0.9567 (the earlier E2 threshold) | C-<x>. Everything is pre-registered in PREREGISTERED.md. Resumable; one process per stage; arms rotated per
question; one unrecorded warm-up per arm on the first question."""
from __future__ import annotations

import json
import os
import socket
import sys
import time

import common as C
import torch
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy
from regret_remasking.selective_cache import SelectiveConfig, decode_selective
from regret_remasking.trace_lock import TraceScorer, bracket, pick_threshold

TOY = os.environ.get("TRACELOCK_TOY") == "1"                      # CPU dry run on a random toy model (tests of the orchestration only; never a result)
BLOCK = int(os.environ.get("TRACELOCK_BLOCK", C.BLOCK))
MASK, EOS_IDS = (3, (2,)) if TOY else (C.MASK_ID, (C.EOS_ID, C.EOT_ID))
DEFER = os.environ.get("TRACELOCK_DEFER_SCORING") == "1"           # HumanEval needs the bwrap sandbox: save the text, score later with rescore_trace_records.py
CUTOFF = 0.95
E2_C = 0.956667                                                    # the earlier E2 confidence threshold (selected at 256 tokens in the moving window), reused at 256 only
STAGES = {"M512": ("math108", 512), "P256": ("panel124", 256), "H512": ("he164", 512)}
ARMS_512 = ["D:C-0.9", "D:TraceLock", "DB:TraceLock", "D:C-matched", "S32:TraceLock", "K4:TraceLock", "P4:TraceLock", "S32:C-matched", "P4:C-matched"]
ARMS_256 = ["D:C-0.9", "D:C-adaptive", "D:TraceLock", "DB:TraceLock", "S32:TraceLock", "K4:TraceLock", "P4:TraceLock", "S32:C-0.9567", "P4:C-0.9567"]
GRID_COARSE = [0.80, 0.85, 0.90, 0.93, 0.95, 0.97]
GRID_UP, GRID_DOWN = [0.98, 0.99, 0.995], [0.7, 0.6, 0.5]
NEAR_FLIP = 2e-3                                                  # parity tolerance, see stage_parity
LOW_CALLS_FLAG = 20.0                                              # mean calls of DB:TraceLock below this on the selection questions: the controller has collapsed; the match is reported as unusable


def first_eos(tokens):
    return next((i for i, t in enumerate(tokens) if t in EOS_IDS), None)


def get_ctx():
    if TOY:
        import toy_dry
        return toy_dry.ctx()
    model, tok = C.load_model()
    controller, projector = C.load_trace()
    return {"model": model, "tok": tok, "scorer": TraceScorer(controller, projector), "controller": controller, "projector": projector}


def lock_threshold(lock, ctx):
    if lock == "TraceLock":
        return CUTOFF
    if lock == "C-matched":
        return float(ctx["cstar"])
    if lock == "C-0.9567":
        return E2_C
    return float(lock.split("-")[1])


def paper_cfg(arm_kind, thr, gen):
    kw = dict(gen_length=gen, block_length=BLOCK, window_length=32, mask_id=MASK, threshold=thr)
    return PaperDecodeConfig(arm=arm_kind, log_calls="writes" if arm_kind in ("T-adaptive", "T-block") else "none", **kw)


def cached_cfg(cache, lock, ctx, gen, *, dense=False, truncate=True, detail=False):
    c = dict(gen_length=gen, window=16, far_mask_cache=True, mask_id=MASK, eos_id=EOS_IDS[0], attn_impl="sdpa", log_calls=True, log_detail=detail, block_length=BLOCK,
             refresh="every_call" if dense else "block", eos_truncation=truncate, shape="block")
    c.update({} if dense else C.CACHE[cache])
    if lock == "TraceLock":
        c.update(lock="trace", trace=ctx["scorer"], threshold=CUTOFF)
    else:
        c.update(lock="conf", threshold=lock_threshold(lock, ctx))
    return SelectiveConfig(**c)


def _cuda():
    return C.DEVICE == "cuda"


def decode(ctx, arm, e, gen):
    model, tok = ctx["model"], ctx["tok"]
    cache, lock = arm.split(":")
    if cache in ("D", "DB"):
        if lock == "TraceLock":
            kind = "T-adaptive" if cache == "D" else "T-block"
        elif lock == "C-adaptive":
            kind = "C-adaptive"
        else:
            kind = "C-block"
        thr = E2_C if lock == "C-adaptive" else lock_threshold(lock, ctx)
        cfg = paper_cfg(kind, thr, gen)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg, controller=ctx.get("controller"), projector=ctx.get("projector")) if kind in ("T-adaptive", "T-block") \
            else decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "partial_calls": 0, "rows_per_call_mean": r.total_query_rows / r.nfe,
                "layer_rows": None, "rule_time_s": r.scorer_time_s, "controller_calls": r.controller_calls, "prompt_length": r.prompt_length, "fallback_calls": r.fallback_count,
                "call_writes": [c["w"] for c in r.call_log] if r.call_log else None}
    cfg = cached_cfg(cache, lock, ctx, gen)
    res = decode_selective(model, C.prompt_ids(ctx, e), cfg)
    L = len(res.layer_rows[0])
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)),
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "partial_calls": sum(1 for k in res.refresh_kind if k == "partial"),
            "rows_per_call_mean": sum(sum(x) for x in res.layer_rows) / L / res.nfe, "layer_rows": res.layer_rows, "refresh_kind": res.refresh_kind, "rule_time_s": res.rule_time_s,
            "controller_calls": len(res.ctrl_prefix), "ctrl_prefix": res.ctrl_prefix, "ctrl_proj_rows": res.ctrl_proj_rows, "prompt_length": res.prompt_length, "fallback_calls": res.fallback_calls}


def load_e2():
    """Earlier E2 tokens (repeat 1) of the dense TraceLock and C-adaptive arms on the 124-case panel (informational parity at 256)."""
    out = {}
    for r in C.rows(C.E2_RECORDS):
        if r.get("repeat") == 1 and r["policy_id"] in ("T-adaptive-0.95", "C-adaptive-0.956667"):
            out.setdefault(r["policy_id"], {})[r["example_id"]] = r["token_ids"]
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------------ parity
def call0_diagnostic(ctx, e, gen):
    """DIAGNOSTIC ONLY (never aborts; Amendment 1 of PREREGISTERED.md): on the first call of a question, one HF forward (output_hidden_states) and one fresh engine forward from the identical input; the largest
    differences of the hidden states 30/31/32 and of the TraceLock scores of the first block's blanks. Separates numerical noise of the hand-run layers from a logic bug if the parity gate fails."""
    try:
        from regret_remasking.selective_cache import SelectiveEngine
        model, scorer = ctx["model"], ctx["scorer"]
        ids = C.prompt_ids(ctx, e)
        P = int(ids.shape[1])
        T = P + gen
        x = torch.full((1, T), MASK, dtype=torch.long, device=ids.device)
        x[:, :P] = ids
        right = P + BLOCK
        with torch.no_grad():
            ref = model(x, output_hidden_states=True)
            hs = ref.hidden_states
            feats = ctx["projector"].encode_hidden(tuple(h[:, :right] for h in (hs[30], hs[31], hs[32])))
            ref_scores = torch.sigmoid(ctx["controller"](feats, valid_mask=torch.ones(feats.shape[:2], dtype=torch.bool, device=feats.device), position_ids=torch.arange(right, device=feats.device).unsqueeze(0))["margins"])[0, P:right]
            cfg = cached_cfg("S32", "TraceLock", ctx, gen, dense=True, truncate=False)
            scorer.reset(T, ids.device)
            eng = SelectiveEngine(model, T, cfg)
            rows = torch.arange(T, device=ids.device)
            _, info = eng.forward(x, rows, 0, torch.arange(P, right, device=ids.device), torch.arange(P, right, device=ids.device), x[0] == MASK, fresh=True)
            got = scorer.score(info["hid"], right, torch.arange(P, right, device=ids.device))
            dh = [float((h.float() - hs[i].float()).abs().max()) for (r, h), i in zip(info["hid"], (30, 31, 32))]
            return {"max_abs_hidden_diff_30_31_32": dh, "max_abs_score_diff": float((got - ref_scores).abs().max()), "min_abs_score_minus_cutoff": float((ref_scores - CUTOFF).abs().min())}
    except Exception as exc:                                  # diagnostic only
        return {"error": repr(exc)}


def stage_parity(ctx):
    """Engine in dense mode (refresh every call, 32-block shape, no truncation) with lock='trace' must equal paper_decode's T-block (HF forward) on the first 3 HumanEval questions: tokens, calls, per-call writes."""
    path = C.REC / "parity.json"
    es = C.manifest("he164")[:3]
    out, flips = [], []
    gen_parity = 512 if not TOY else int(os.environ.get("TRACELOCK_GEN", 32))
    for e in es:
        torch.manual_seed(23)
        ref = decode(ctx, "DB:TraceLock", e, gen_parity)
        cfg = cached_cfg("S32", "TraceLock", ctx, gen_parity, dense=True, truncate=False, detail=True)
        res = decode_selective(ctx["model"], C.prompt_ids(ctx, e), cfg)
        P = res.prompt_length
        writes = [[p - P for p in w] for w in res.written]
        same_tokens, same_nfe, same_writes = list(res.tokens) == ref["tokens"], res.nfe == ref["nfe"], writes == ref["call_writes"]
        item = dict(example_id=e["example_id"], same_tokens=same_tokens, same_nfe=same_nfe, same_writes=same_writes, engine_nfe=res.nfe, paper_nfe=ref["nfe"])
        if not (same_tokens and same_nfe and same_writes):
            n = min(len(writes), len(ref["call_writes"]))
            div = next((i for i in range(n) if writes[i] != ref["call_writes"][i]), n)
            sc = res.ctrl_scores[div] if div < len(res.ctrl_scores) else []
            item.update(first_divergent_call=div, engine_writes=writes[div] if div < len(writes) else None, paper_writes=ref["call_writes"][div] if div < len(ref["call_writes"]) else None,
                        engine_scores_at_divergence=sc, min_abs_score_minus_cutoff=min((abs(s - CUTOFF) for s in sc), default=None))
            pre = [abs(s - CUTOFF) for call in res.ctrl_scores[:div] for s in call]
            item["min_abs_score_minus_cutoff_before_divergence"] = min(pre, default=None)
            item["near_threshold_flip"] = bool(sc) and min(abs(x - CUTOFF) for x in sc) < NEAR_FLIP     # a divergence at a call where some score sits within NEAR_FLIP of the cutoff can be float noise, not a bug
            flips.append(item["near_threshold_flip"])
        item["call0_diagnostic"] = call0_diagnostic(ctx, e, gen_parity)
        out.append(item)
        print("[parity]", json.dumps(item), flush=True)
    # informational: the dense confidence decoder reproduces the earlier stage-E tokens (not an abort condition)
    info = {}
    if not TOY:
        prev = {r["example_id"]: r["tokens"] for r in C.rows(C.ROOT / "results/depth_refresh_20261109/records/stageE_he164.jsonl") if r.get("arm") == "D:C-0.9" and r.get("status") == "ok"}
        eq = []
        for e in es:
            if e["example_id"] in prev:
                eq.append(decode(ctx, "D:C-0.9", e, 512)["tokens"] == prev[e["example_id"]])
        info["D:C-0.9_equals_depth_refresh_stageE"] = eq
    ok = not flips or (len(flips) <= 1 and all(flips))             # pre-registered tolerance: at most ONE of the 3 questions may diverge, and only at a near-threshold call; anything else is a bug
    Path_write(path, {"ok": ok, "diverged_questions": len(flips), "near_threshold_tolerance": NEAR_FLIP, "questions": out, "informational": info, "host": socket.gethostname(), "cuda": torch.cuda.get_device_name(0) if _cuda() else "cpu"})
    if not ok:
        print("[parity] FAILED: the engine's dense trace lock does not reproduce paper_decode T-block; aborting before any evaluation decode. See records/parity.json.", flush=True)
        raise SystemExit(3)
    print("[parity] OK" + (" (one near-threshold divergence tolerated; see records/parity.json)" if flips else ""), flush=True)


def Path_write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")


# ------------------------------------------------------------------------------------------------------------------------------------------------ selection
def stage_sel(ctx, smoke):
    """Match a confidence threshold to TraceLock's mean call count (the E2 procedure): 40 questions (20 GSM8K train + 20 KodCode, the first of the E2 selection split in file order; disjoint from MATH-500 and
    HumanEval by construction), 512 tokens, dense block geometry: DB:TraceLock gives the target; C-block thresholds on a coarse grid, extended while the target lies outside, then refined with 4 interior points between
    the bracketing thresholds; the evaluated threshold whose mean calls is nearest to the target is c*. The window-geometry D:TraceLock's mean calls on the same questions is recorded, not used."""
    gen = 512 if not TOY else int(os.environ.get("TRACELOCK_GEN", 32))
    es = C.manifest("selection40")[:2 if smoke else 40]
    path = C.REC / f"{'smoke_' if smoke else ''}selection.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path)}
    t0 = time.time()

    def run_arm(arm):
        for e in es:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            d = decode(ctx, arm, e, gen)
            C.append(path, dict(example_id=e["example_id"], arm=arm, dataset=e["dataset"], nfe=d["nfe"], latency_s=d["latency_s"], first_eos=first_eos(d["tokens"]), controller_time_s=d["rule_time_s"], status="ok"))
            done.add((e["example_id"], arm))

    def mean_calls(arm):
        v = {r["example_id"]: r["nfe"] for r in C.rows(path) if r["arm"] == arm}
        return sum(v[e["example_id"]] for e in es) / len(es)

    for e in es[:1]:                                                    # warm-up (unrecorded) on the first question
        decode(ctx, "DB:TraceLock", e, gen)
    run_arm("DB:TraceLock")
    run_arm("D:TraceLock")
    target = mean_calls("DB:TraceLock")
    by = {}

    def evaluate(t):
        t = round(float(t), 4)
        if t not in by:
            run_arm(f"D:C-{t}")
            by[t] = mean_calls(f"D:C-{t}")
            print(f"[sel] C-{t}: mean calls {by[t]:.2f} (target {target:.2f}, {time.time() - t0:.0f} s)", flush=True)

    for t in ([0.9, 0.95] if smoke else GRID_COARSE):
        evaluate(t)
    for t in GRID_UP:
        if target > max(by.values()):
            evaluate(t)
    for t in GRID_DOWN:
        if target < min(by.values()):
            evaluate(t)
    br = bracket(target, by)
    if br is not None and not smoke:
        lo, hi = br
        for i in range(1, 5):
            evaluate(lo + (hi - lo) * i / 5)
    cstar, info = pick_threshold(target, by)
    collapsed = target < LOW_CALLS_FLAG
    out = dict(info, target_calls_DB_TraceLock=target, mean_calls_D_TraceLock_window=mean_calls("D:TraceLock"), mean_calls_by_threshold={str(k): v for k, v in sorted(by.items())},
               c_star=cstar, n_questions=len(es), datasets={d: sum(1 for e in es if e["dataset"] == d) for d in {e["dataset"] for e in es}}, gen_length=gen,
               controller_collapsed=collapsed, usable=(info["status"] == "bracketed" and not collapsed), fallback=None, smoke=smoke, seconds=time.time() - t0, host=socket.gethostname())
    Path_write(C.REC / f"{'smoke_' if smoke else ''}selection.json", out)
    print("[sel] c* =", cstar, json.dumps(out), flush=True)


def stage_sel_fallback():
    Path_write(C.REC / "selection.json", dict(c_star=0.8, status="fallback_C-0.8", usable=False, fallback="selection stage failed or was not reproducible: C-0.8 used as the matched threshold"))


# ------------------------------------------------------------------------------------------------------------------------------------------------ evaluation
def examples(dataset):
    l5, l14 = C.manifest("math_l5"), C.manifest("math_l14")
    if dataset == "math108":
        return l5 + l14
    if dataset == "he164":
        return C.manifest("he164")
    if dataset == "panel124":
        return C.manifest("he164")[:70] + l5
    raise ValueError(dataset)


def stage_eval(ctx, stage, smoke):
    dataset, gen = STAGES[stage]
    if TOY:
        gen = int(os.environ.get("TRACELOCK_GEN", 32))
    arms = ARMS_256 if stage == "P256" else ARMS_512
    sel = json.loads((C.REC / "selection.json").read_text()) if (C.REC / "selection.json").exists() else None
    if sel is None and stage != "P256":
        raise SystemExit("records/selection.json is missing: run SEL (or SELFALLBACK) first")
    if smoke and sel is None:
        sel = dict(c_star=0.9)
    ctx["cstar"] = sel["c_star"] if sel else None
    print(f"[{stage}] matched threshold c* = {ctx['cstar']} ({sel and sel.get('status')}, usable={sel and sel.get('usable')})", flush=True)
    es = examples(dataset)
    if smoke:
        es = es[:2]
    needs_code = any(e["dataset"] == "humaneval" for e in es)
    if needs_code and not DEFER and not TOY:
        C.scorer_guard()
    e2 = load_e2() if stage == "P256" and not TOY else {}
    path = C.REC / f"{'smoke_' if smoke else ''}stage{stage}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(ctx, arm, es[0], gen)                                  # unrecorded warm-up
    started = time.time()
    for i, e in enumerate(es):
        rot = i % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            qid = e["example_id"]
            if (qid, arm) in done:
                continue
            C.gpu_guard()
            if _cuda():
                torch.cuda.reset_peak_memory_stats()
            d = decode(ctx, arm, e, gen)
            d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30 if _cuda() else 0.0      # peak allocated GPU memory of this decode (includes the 15 GiB of weights)
            code = e["dataset"] == "humaneval"
            correct, status = (None, "deferred") if (DEFER and code) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=qid, arm=arm, dataset=e["dataset"], stage=stage, gen_length=gen, correct=(None if correct is None else bool(correct)), scoring_status=status,
                       first_eos=first_eos(d["tokens"]), status="ok", loadavg1=os.getloadavg()[0], host=socket.gethostname(), c_star=ctx["cstar"])
            if e2 and arm in ("D:TraceLock", "D:C-adaptive") and qid in e2.get("T-adaptive-0.95" if arm == "D:TraceLock" else "C-adaptive-0.956667", {}):
                rec["eq_e2_tokens"] = d["tokens"] == e2["T-adaptive-0.95" if arm == "D:TraceLock" else "C-adaptive-0.956667"][qid]
            C.append(path, rec)
        print(f"[{stage}] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[{stage}] done", flush=True)


def main():
    stage = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    if stage == "SELFALLBACK":
        stage_sel_fallback()
        return
    C.gpu_guard()
    ctx = get_ctx()
    if stage == "PARITY":
        stage_parity(ctx)
    elif stage == "SEL":
        stage_sel(ctx, smoke)
    elif stage in STAGES:
        stage_eval(ctx, stage, smoke)
    else:
        raise SystemExit(f"unknown stage {stage}")


if __name__ == "__main__":
    main()
