"""HumanEval 0-163, 512 tokens, word lock C-0.9, S32 representation cache with a SCHEDULED PARTIAL REFRESH (cached rows rejoin at layer l0 from their stored layer inputs).
Usage: <abs venv python> run_depth_cells.py STAGE [SMOKE]   (STAGE in CELL_SETS; records/<stage>[smoke_].jsonl, resumable). One process, arms rotated per question, warm-up on question 0.
Every stage contains the in-process references D:C-0.9 (dense word lock) and S32:main (block-start dense calls only), so ratios are within one process."""
from __future__ import annotations

import os
import sys
import time

import common as C
import torch
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy

GEN = 512
SETS = {"he164": 512, "he70": 256, "math108": 512, "math54half": 1024}      # dataset -> generation length; a stage name X_<set> runs the cells of stage X on that set (X_slurm etc. = he164)
MATH_REF = {"math108": "records/stage1.jsonl", "math54half": "records/stage2.jsonl"}
DEFER = os.environ.get("DEPTH_DEFER_SCORING") == "1"     # no sandbox (bwrap) on this host: save the text, score later with rescore_depth_records.py on a host that has it
REFS = ["D:C-0.9", "S32:main"]
PART = lambda k, l0, join: dict(partial_every_k=k, partial_layer=l0, partial_join=join)
CELL_SETS = {
    # stage A: does a depth-selective refresh restore the answers at all? (every non-dense call is partial)
    "A": {"P1-l0-context": PART(1, 0, "context"), "P1-l11-context": PART(1, 11, "context"), "P1-l11-all": PART(1, 11, "all"), "P1-l24-all": PART(1, 24, "all"), "P1-l11-blank": PART(1, 11, "blank")},
}
def _spec(v):
    f = v.split(",")
    if f[0] == "dense":                        # dense,k: a DENSE refresh every k-th call
        return dict(refresh_every_k=int(f[1]))
    if f[0] == "wdense":                       # wdense,W: a dense call once W words were written since the last refresh
        return dict(refresh_every_words=int(f[1]), words_refresh="dense")
    if f[0] == "wpart":                        # wpart,W,l0,join: a partial refresh once W words were written since the last refresh
        return dict(refresh_every_words=int(f[1]), words_refresh="partial", partial_layer=int(f[2]), partial_join=f[3])
    return PART(int(f[0]), int(f[1]), f[2])    # k,l0,join: a partial refresh every k-th call


for _st in ("B", "C", "D"):                          # later stages are defined by the environment after the previous stage is analysed: DEPTH_CELLS_<STAGE>='name=spec;name=spec'
    if os.environ.get(f"DEPTH_CELLS_{_st}"):
        CELL_SETS[_st] = {n: _spec(v) for n, v in (x.split("=") for x in os.environ[f"DEPTH_CELLS_{_st}"].split(";"))}


F_ARMS = ["D:C-0.9", "D:Naive", "S32D:Naive", "S32:Naive", "K4:Naive", "P4:Naive", "S16D:Naive", "EL:Naive", "S16:Naive", "S16P4:Naive",
          "S16D:Elastic", "EL:Elastic", "S16:C-0.9", "S16P4:Elastic"]       # stage F: Elastic-Cache's own evaluation protocol (HumanEval 0-69, 256 tokens); naive-diffusion family and Elastic-word-lock family
E_CACHE = {"S32": {}, "K4": dict(refresh_every_k=4), "P4": dict(partial_every_k=4, partial_layer=11, partial_join="blank")}      # stage E: cache strategy by name
E_ARMS = ["D:C-0.9", "D:C-0.8", "D:Lctx-0.8", "S32:C-0.9", "S32:C-0.8", "S32:Lctx-0.8", "K4:C-0.9", "K4:C-0.8", "K4:Lctx-0.8", "P4:C-0.9", "P4:C-0.8", "P4:Lctx-0.8", "S16:C-0.9", "EL:Elastic", "S16D:Elastic", "S16P4:Elastic"]   # the full 3 word locks x 4 caches grid + Elastic's shape and recipe


def decode_e(ctx, arm, e):
    """Stage E arms are '<cache>:<word lock>': cache in {D (dense, no cache), S32, K4 (dense refresh every 4th call), P4 (blank rows from layer 11 every 4th call)}, lock in {C-0.9, C-0.8, Lctx-0.8}."""
    model, tok = ctx["model"], ctx["tok"]
    cache, lock = arm.split(":")
    if cache == "D":
        spec = (dict(arm="B0") if lock == "Naive" else dict(arm="C-block", threshold=float(lock.split("-")[1])) if lock.startswith("C") else dict(arm="X-block", block_rule=ctx["rule"]))
        cfg = PaperDecodeConfig(gen_length=GEN, block_length=32, mask_id=C.MASK_ID, **spec)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "partial_calls": 0, "rows_per_call_mean": r.total_query_rows / r.nfe,
                "layer_rows": None, "rule_time_s": r.scorer_time_s}
    if cache in ("S16", "EL", "S16D", "S16P4"):
        lock_cfg = dict(lock="topk", topk=1) if lock == "Naive" else dict(lock="elastic")
        # The ELASTIC WORD-LOCK FAMILY (word lock held fixed, cache varied): S16D dense every call (reference), EL Elastic's trigger, S16 our block-start dense refresh, S16P4 our blank-row refresh every 4th call.
        # Elastic-Cache's shape and word lock in our engine: window of the next 16 blanks, Conf-0.9 with the min(threshold, max) rule, first-EOS truncation. S16 refreshes with a dense call at each block start (ours);
        # EL uses Elastic's adaptive trigger (one tracked row, per-layer head-axis cosine < gamma 0.9 makes every cached row rejoin at the next layer). Fused SDPA kernel as in every other arm.
        cfg = C.SelectiveConfig(gen_length=GEN, window=16, far_mask_cache=True, mask_id=C.MASK_ID, eos_id=C.EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=32,
                                threshold=0.9, shape="window", eos_truncation=True, **lock_cfg, refresh={"S16": "block", "EL": "trigger", "S16D": "every_call", "S16P4": "block"}[cache],
                                **(E_CACHE["P4"] if cache == "S16P4" else {}))
    else:
        dense = cache == "S32D"
        cfg = C.cached_cfg("C-0.9" if lock == "Naive" else lock, ctx, dense=dense, **dict({} if dense else E_CACHE[cache], gen_length=GEN))
        if lock == "Naive":
            import dataclasses
            cfg = dataclasses.replace(cfg, lock="topk", topk=1)
    res = C.decode_selective(model, C.prompt_ids(ctx, e), cfg)
    L = len(res.layer_rows[0])
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)),
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "partial_calls": sum(1 for k in res.refresh_kind if k == "partial"),
            "rows_per_call_mean": sum(sum(x) for x in res.layer_rows) / L / res.nfe, "layer_rows": res.layer_rows, "refresh_kind": res.refresh_kind, "rule_time_s": res.rule_time_s}


def decode(ctx, arm, e, cells):
    if cells is None:
        return decode_e(ctx, arm, e)
    model, tok = ctx["model"], ctx["tok"]
    if arm == "D:C-0.9":
        cfg = PaperDecodeConfig(gen_length=GEN, block_length=32, mask_id=C.MASK_ID, arm="C-block", threshold=0.9)
        torch.manual_seed(23)
        r = decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "partial_calls": 0, "rows_per_call_mean": r.total_query_rows / r.nfe,
                "layer_rows": None}
    extra = dict({} if arm == "S32:main" else cells[arm], gen_length=GEN)
    cfg = C.cached_cfg("C-0.9", ctx, **extra)
    ids = C.prompt_ids(ctx, e)
    res = C.decode_selective(model, ids, cfg)
    L = len(res.layer_rows[0])
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)),
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "partial_calls": sum(1 for k in res.refresh_kind if k == "partial"),
            "rows_per_call_mean": sum(sum(x) for x in res.layer_rows) / L / res.nfe, "layer_rows": res.layer_rows, "refresh_kind": res.refresh_kind}


def main():
    stage = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    global GEN
    suffix = stage.split("_", 1)[1] if "_" in stage else ""
    dataset = suffix if suffix in SETS else "he164"
    GEN = SETS[dataset]
    is_e = stage.split("_")[0] in ("E", "F")
    cells = None if is_e else CELL_SETS[stage.split("_")[0]]          # a stage name like C_slurm = the cells of stage C written to its own records file (a clean rerun on another node)
    arms = (F_ARMS if stage.startswith("F") else E_ARMS) if is_e else REFS + list(cells)
    path = C.OUT / f"records/{'smoke_' if smoke else ''}stage{stage}.jsonl"
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    if is_e:
        ctx["rule"], _sha = C.load_rule()
        print("[Lctx rule sha256]", _sha, flush=True)
    code = dataset in ("he164", "he70")                       # only HumanEval needs the bwrap sandbox; MATH scoring is answer matching
    if code and not DEFER:
        C.scorer_guard()
    if code:
        s32_all = C.stage1_by_arm()
        es = C.he164()[:70] if dataset == "he70" else C.he164()
    else:
        s32_all = {}
        for r in C.rows(C.ROOT / "results/farcache_stack_20261105" / MATH_REF[dataset]):
            if r.get("set") == dataset and r.get("status") == "ok":
                s32_all.setdefault(r["arm"], {})[r["example_id"]] = r
        l5, l14 = C.rows(C.OUT / "manifests/math_l5.jsonl"), C.rows(C.OUT / "manifests/math_l14.jsonl")
        if dataset == "math54half":
            l5, l14 = l5[::2], l14[::2]
        es = [dict(e, _set=dataset, _stratum="L5") for e in l5] + [dict(e, _set=dataset, _stratum="L1-4") for e in l14]
    if smoke:
        es = es[:2]
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in arms:
        decode(ctx, arm, es[0], cells)
    started = time.time()
    for i, e in enumerate(es):
        rot = i % len(arms)
        for arm in arms[rot:] + arms[:rot]:
            qid = e["example_id"]
            if (qid, arm) in done:
                continue
            C.gpu_guard()
            torch.cuda.reset_peak_memory_stats()
            d = decode(ctx, arm, e, cells)
            d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30        # peak allocated GPU memory of this decode (includes the 15 GiB of weights)
            correct, status = (None, "deferred") if (DEFER and code) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=qid, arm=arm, set=dataset, correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=C.first_eos(d["tokens"]), status="ok", loadavg1=os.getloadavg()[0])
            ref_arm = None if GEN != 512 else "S32:C-0.9" if arm == "S32:main" else arm if (is_e and arm in ("S32:C-0.9", "S32:C-0.8", "S32:Lctx-0.8", "S16:C-0.9")) else None
            if ref_arm is not None:
                rec["eq_s32_stage1"] = d["tokens"] == s32_all[ref_arm][qid]["tokens"]
                if i < 3 and not rec["eq_s32_stage1"]:
                    raise SystemExit(f"parity failure: S32:main differs from the stage-1 S32:C-0.9 tokens on {qid}")
            C.append(path, rec)
        print(f"[depth {stage}] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[depth {stage}] done", flush=True)


if __name__ == "__main__":
    main()
