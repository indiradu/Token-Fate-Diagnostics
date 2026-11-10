"""Analysis 2: oracle repair at the first divergence (HumanEval 0-163, 512 tokens). QUALITY ONLY, speed is not measured.
Usage: <abs venv python> run_oracle_repair.py [SMOKE]
  records/oracle_census_<lock>.jsonl  per task: dense decode (S32 engine with a dense call every call + first-EOS truncation; token-identical to the dense word-lock decoder) and cached decode (S32), both
                                      with per-call write logs; first divergence (call, position, kind), the last block-start dense call c0 before it.
  records/oracle_repair_<lock>.jsonl  per lost task x class x mode x window: the cached decode resumed from the dense call c0 (state rebuilt from the logged writes) with ONE row class repaired.
Locks: C-0.9 (all 164 tasks in the census, repair for the 17-ish lost ones) and Lctx-0.8 (census and repair for the tasks its dense lock solves and S32 loses).
Repair classes (stale_cause.REPAIR_CLASSES): next_blank (a), far2_blank (b), inblock_old (c), prompt (d), prior_written (extra, completes the partition), all_blank (e, positive control),
all_cached (g, dense-equivalent control). Modes: oracle = the class's cached K/V replaced by those of a dense pass over the current tokens; compute = the class's rows are computed in the call.
Windows: seg = repair only in the calls that decode the block of the first divergence; end = from c0 to the end of the decode. Null = the resumed decode with nothing repaired (must equal the cached decode).
"""
from __future__ import annotations

import sys
import time

import common as C
from regret_remasking.selective_cache import decode_selective, replay_state
from regret_remasking.stale_cause import REPAIR_CLASSES, events, first_divergence, first_token_diff, last_dense_before, repair_ctrl

LOCKS = ["C-0.9", "Lctx-0.8"]
VARIANTS = list(REPAIR_CLASSES)
MODES = ["oracle", "compute"]
WINDOWS = ["seg", "end"]


def census_one(ctx, e, lock, ref):
    qid = e["example_id"]
    dense_cfg = C.cached_cfg(lock, ctx, dense=True, log_detail=True)
    cache_cfg = C.cached_cfg(lock, ctx, log_detail=True)
    ids, d = C.run_cached(ctx, e, dense_cfg)
    _, c = C.run_cached(ctx, e, cache_cfg)
    ev_d, ev_c = events(d.written, d.written_tokens), events(c.written, c.written_tokens)
    div = first_divergence(ev_c, ev_d)
    fe_d = C.first_eos(d.tokens)
    rec = {"example_id": qid, "lock": lock, "prompt_len": int(ids.shape[1]), "dense_tokens": d.tokens, "cached_tokens": c.tokens, "dense_nfe": d.nfe, "cached_nfe": c.nfe,
           "dense_written": d.written, "dense_written_tokens": d.written_tokens, "cached_written": c.written, "cached_written_tokens": c.written_tokens,
           "cached_refresh_kind": c.refresh_kind, "cached_call_blocks": c.call_blocks, "divergence": div, "dense_first_eos": fe_d,
           "eq_stage1_dense_to_eos": d.tokens[:(fe_d + 1) if fe_d is not None else None] == ref["dense"][qid]["tokens"][:(fe_d + 1) if fe_d is not None else None],
           "eq_stage1_dense_full": d.tokens == ref["dense"][qid]["tokens"], "eq_stage1_cached": c.tokens == ref["cached"][qid]["tokens"],
           "dense_correct_stage1": ref["dense"][qid]["correct"], "cached_correct_stage1": ref["cached"][qid]["correct"], "status": "ok"}
    if div is not None:
        c0 = last_dense_before(c.refresh_kind, div["call"])
        rec.update({"c0": c0, "block": c.call_blocks[c0], "div_block": c.call_blocks[div["call"]]})
        if div["pos"] is not None:
            rec.update({"div_rel_pos": div["pos"] - rec["prompt_len"], "div_pos_in_block": (div["pos"] - rec["prompt_len"]) % C.BLOCK,
                        "div_calls_after_block_start": div["call"] - c0})
    return rec


def repair_one(ctx, e, lock, cen, variant, mode, window, dense_events):
    qid = e["example_id"]
    ids = C.prompt_ids(ctx, e)
    ctrl = None if variant is None else repair_ctrl(variant, mode, cen["block"] if window == "seg" else -1, stats=(mode == "oracle"))
    cfg = C.cached_cfg(lock, ctx, ctrl=ctrl, log_detail=True)
    st = replay_state(ids[0].tolist(), C.GEN, cen["cached_written"], cen["cached_written_tokens"], cen["c0"], cfg)
    res = decode_selective(ctx["model"], ids, cfg, resume=st)
    div = cen["divergence"]
    d2 = first_divergence(events(res.written, res.written_tokens), dense_events, offset_a=cen["c0"])
    tokens = list(res.tokens)
    text, correct, status = C.score(ctx, e, tokens)
    dt = cen["dense_tokens"]
    fe = cen["dense_first_eos"]
    rec = {"example_id": qid, "lock": lock, "variant": variant or "null", "mode": mode if variant else "-", "window": window if variant else "-", "c0": cen["c0"],
           "first_div_call": div["call"], "first_div_kind": div["kind"], "div_pos": div["pos"], "tokens": tokens, "text": text, "correct": correct, "scoring_status": status, "nfe_resumed": res.nfe,
           "rows_per_call": res.rows_per_call, "redivergence": d2, "restored_through_d": d2 is None or d2["call"] > div["call"],
           "identical_to_dense": tokens == dt, "identical_to_dense_to_eos": tokens[:(fe + 1) if fe is not None else None] == dt[:(fe + 1) if fe is not None else None],
           "equals_cached": tokens == cen["cached_tokens"], "first_token_diff_vs_dense": first_token_diff(tokens, dt), "status": "ok"}
    if div["pos"] is not None and div["kind"] in ("token", "a_only"):          # a position the cached decode wrote
        i = div["pos"] - cen["prompt_len"]
        rec["pos_token_restored"] = tokens[i] == dt[i]
    if res.ctrl_stats and variant is not None and mode == "oracle":
        rec["class_rows"] = {c: {"calls_nonempty": v["calls_nonempty"], "rows_layer1": v["rows_per_layer"][1]} for c, v in res.ctrl_stats.items()}
    return rec


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "SMOKE"
    C.gpu_guard()
    model, tok = C.load_model()
    rule, sha = C.load_rule()
    ctx = {"model": model, "tok": tok, "rule": rule}
    C.scorer_guard()
    s1 = C.stage1_by_arm()
    es = {e["example_id"]: e for e in C.he164()}
    started = time.time()
    plan = []
    for lock in LOCKS:
        dl, cl = s1["D:" + lock], s1["S32:" + lock]
        lost = [q for q in es if dl[q]["correct"] and not cl[q]["correct"]]
        rest = [q for q in es if q not in lost] if lock == "C-0.9" else []
        plan.append((lock, lost, rest, {"dense": dl, "cached": cl}))
    print("[repair] lost tasks:", {p[0]: len(p[1]) for p in plan}, flush=True)
    for phase in ("lost", "rest"):
        for lock, lost, rest, ref in plan:
            tasks = lost if phase == "lost" else rest
            if smoke:
                tasks = tasks[:2]
            cpath = C.OUT / f"records/{'smoke_' if smoke else ''}oracle_census_{lock}.jsonl"
            rpath = C.OUT / f"records/{'smoke_' if smoke else ''}oracle_repair_{lock}.jsonl"
            census = {r["example_id"]: r for r in C.rows(cpath) if r.get("status") == "ok"}
            rdone = {(r["example_id"], r["variant"], r["mode"], r["window"]) for r in C.rows(rpath) if r.get("status") == "ok"}
            for k, qid in enumerate(tasks):
                e = es[qid]
                C.gpu_guard()
                if qid not in census:
                    census[qid] = census_one(ctx, e, lock, ref)
                    C.append(cpath, census[qid])
                    cen = census[qid]
                    print(f"[census {lock}] {qid} div={cen['divergence']} eq_dense_to_eos={cen['eq_stage1_dense_to_eos']} eq_cached={cen['eq_stage1_cached']}", flush=True)
                    if k < 2 and not cen["eq_stage1_cached"]:
                        raise SystemExit(f"PARITY FAILED: cached decode with logs differs from stage-1 S32 tokens on {qid}")
                cen = census[qid]
                if phase != "lost" or cen["divergence"] is None or "c0" not in cen:
                    continue
                dense_events = events(cen["dense_written"], cen["dense_written_tokens"])
                combos = [(None, "-", "-")] + [(v, m, w) for v in VARIANTS for m in MODES for w in WINDOWS]
                for v, m, w in combos:
                    key = (qid, v or "null", m if v else "-", w if v else "-")
                    if key in rdone:
                        continue
                    if smoke and v not in (None, "all_cached", "next_blank"):
                        continue
                    rec = repair_one(ctx, e, lock, cen, v, m, w, dense_events)
                    C.append(rpath, rec)
                    rdone.add(key)
                    if v is None and not rec["equals_cached"]:
                        raise SystemExit(f"REPLAY FAILED: the resumed decode without repair differs from the cached decode on {qid} (c0 {cen['c0']})")
                print(f"[repair {lock}] {qid} done ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print("[repair] done", flush=True)


if __name__ == "__main__":
    main()
