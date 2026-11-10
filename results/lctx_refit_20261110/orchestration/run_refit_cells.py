"""Stage G: does a rule refitted in the geometry of use fix the 16-window transfer failure / change Lctx under the 32-block cache?  (PREREGISTERED.md)
Usage: <abs venv python> run_refit_cells.py G_<set> [SMOKE]     set in {he164, math108}; records/stageG_<set>.jsonl (resumable). One process, arms rotated per question, warm-up on question 0.
Arms (512 tokens, first-EOS truncation, block-start dense call):
  D:C-0.9          dense word lock (the reference, in-process)
  S16:C-0.8        16-blank window, Elastic's min(threshold, max) rule at 0.8          S16:Lctx-0.8   original rule          S16:Lctx-refit   rule refitted on R-S16 traces
  P4:C-0.8         32-block shape, S32 cache + blank-row refresh every 4th call (layer 11)   P4:Lctx-0.8   original rule   P4:Lctx-refit   rule refitted on R-S32c traces
Scoring: MATH in-process; HumanEval deferred (DEPTH_DEFER_SCORING=1: no bwrap on the node) and done afterwards by rescore_records.py."""
from __future__ import annotations

import os
import sys
import time

import common as C
import torch

ARMS = ["D:C-0.9", "S16:C-0.8", "S16:Lctx-0.8", "S16:Lctx-refit", "P4:C-0.8", "P4:Lctx-0.8", "P4:Lctx-refit"]
SETS = {"he164": 512, "math108": 512}
DEFER = os.environ.get("DEPTH_DEFER_SCORING") == "1"
STACK1 = C.ROOT / "results/farcache_stack_20261105/records/stage1.jsonl"                   # D:C-0.9, S16:C-0.8, S16:Lctx-0.8 (HumanEval 164 and MATH 108)
STAGE_E = {"he164": C.ROOT / "results/depth_refresh_20261109/records/stageE_he164.jsonl", "math108": C.ROOT / "results/depth_refresh_20261109/records/stageE_math108.jsonl"}   # P4 arms
PARITY_ARMS = {"D:C-0.9", "S16:C-0.8", "S16:Lctx-0.8", "P4:C-0.8", "P4:Lctx-0.8"}


def reference_tokens(dataset):
    by = {}
    for path in (STACK1, STAGE_E[dataset]):
        for r in C.rows(path):
            if r.get("set") == dataset and r.get("status") == "ok" and r["arm"] in PARITY_ARMS:
                by.setdefault(r["arm"], {})[r["example_id"]] = r["tokens"]
    return by


def decode(ctx, arm, e):
    model, tok = ctx["model"], ctx["tok"]
    shape, lock = arm.split(":")
    if shape == "D":
        cfg = C.PaperDecodeConfig(gen_length=C.GEN, block_length=C.BLOCK, mask_id=C.MASK_ID, arm="C-block", threshold=0.9)
        torch.manual_seed(23)
        r = C.decode_paper_policy(model, tok, C._resolve_prompt(e), e["example_id"], cfg)
        return {"tokens": list(r.token_ids), "latency_s": r.latency_s, "nfe": r.nfe, "text": r.text, "dense_calls": r.nfe, "partial_calls": 0, "rows_per_call_mean": r.total_query_rows / r.nfe,
                "layer_rows": None, "rule_time_s": r.scorer_time_s, "fallback_calls": r.fallback_count}
    rule = None
    if lock == "Lctx-0.8":
        rule = ctx["rule0"]
    elif lock == "Lctx-refit":
        rule = ctx["refit"]["S16" if shape == "S16" else "S32c"]
    cfg = C.eval_cfg(shape, lock, rule)
    res = C.decode_selective(model, C.prompt_ids(ctx, e), cfg)
    L = len(res.layer_rows[0])
    return {"tokens": list(res.tokens), "latency_s": res.latency_s, "nfe": res.nfe, "text": C._decode_response(tok, list(res.tokens)),
            "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "partial_calls": sum(1 for k in res.refresh_kind if k == "partial"),
            "rows_per_call_mean": sum(sum(x) for x in res.layer_rows) / L / res.nfe, "layer_rows": res.layer_rows, "refresh_kind": res.refresh_kind, "rule_time_s": res.rule_time_s,
            "fallback_calls": res.fallback_calls}


def main():
    stage = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    dataset = stage.split("_", 1)[1]
    code = dataset == "he164"
    path = C.OUT / f"records/{'smoke_' if smoke else ''}stage{stage}.jsonl"
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    ctx["rule0"], sha0 = C.load_rule()
    ctx["refit"] = {}
    for g in ("S16", "S32c"):
        ctx["refit"][g], sha = C.load_refit(g)
        print(f"[rule sha256] refit {g} {sha}", flush=True)
    print(f"[rule sha256] original {sha0}", flush=True)
    if code and not DEFER:
        C.scorer_guard()
    if code:
        es = [dict(e, _set=dataset, _stratum="he164") for e in C.rows(C.OUT / "manifests/he164.jsonl")]
    else:
        l5, l14 = C.rows(C.OUT / "manifests/math_l5.jsonl"), C.rows(C.OUT / "manifests/math_l14.jsonl")
        es = [dict(e, _set=dataset, _stratum="L5") for e in l5] + [dict(e, _set=dataset, _stratum="L1-4") for e in l14]
    if smoke:
        es = es[:2]
    ref = reference_tokens(dataset)
    print("[parity refs]", {a: len(v) for a, v in ref.items()}, flush=True)
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for arm in ARMS:
        decode(ctx, arm, es[0])
    started = time.time()
    for i, e in enumerate(es):
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            qid = e["example_id"]
            if (qid, arm) in done:
                continue
            C.gpu_guard()
            d = decode(ctx, arm, e)
            correct, status = (None, "deferred") if (DEFER and code) else C.score_paper_generation(d["text"], e)
            if status == "sandbox_error":
                raise SystemExit("sandbox_error while scoring: abort")
            rec = dict(d, example_id=qid, arm=arm, set=dataset, stratum=e["_stratum"], correct=(None if correct is None else bool(correct)), scoring_status=status, first_eos=C.first_eos(d["tokens"]),
                       status="ok", loadavg1=os.getloadavg()[0])
            rt = ref.get(arm, {}).get(qid)
            if rt is not None:
                rec["eq_reference"] = d["tokens"] == rt
                if i < 3 and not rec["eq_reference"]:
                    raise SystemExit(f"parity failure: {arm} differs from its earlier-stage tokens on {qid}")
            C.append(path, rec)
        print(f"[stage {stage}] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[stage {stage}] done", flush=True)


if __name__ == "__main__":
    main()
