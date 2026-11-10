"""Analysis 1: noise-matched control for the stale K/V of the S32 representation cache (HumanEval 0-163, 512 tokens, word lock C-0.9). QUALITY ONLY, speed is not measured.
Usage: <abs venv python> run_noise_control.py [SMOKE]       -> records/noise_control.jsonl (resumable; SMOKE: 2 questions, records/smoke_noise_control.jsonl)

Every arm is the S32 decode (dense call at each block start, first-EOS truncation; query rows = rows written on the previous call + the blanks of the current block). At the CACHED rows
(prompt / prior_written / inblock_old / next_blank / far_blank, see stale_cause.py) the K/V the attention reads at a non-dense call is, per arm:
  P-fresh      all five classes: K/V of a dense pass over the current tokens (oracle)       -> must equal the dense C-0.9 decode
  P-stale      all five classes: the cache itself (a dense pass is still run, for the distance statistics only) -> must equal S32:C-0.9 exactly
  P-rand1a/1b  all five classes: fresh + random perturbation, per row and layer ||noise|| = 1.0 x ||stale - fresh||, K and V separately (two random seeds)
  P-rand0.5 / P-rand2   the same with 0.5x / 2x the norm (seed "a")
  B-stale      next_blank + far_blank stale (the cache), prompt / prior_written / inblock_old fresh
  B-rand1      next_blank + far_blank: fresh + 1.0x noise, the other three classes fresh
Perturbation point: the K (post-RoPE, as cached) and V tensors of every layer, built for the attention of the call only; the caches are never written. Seeds: stale_cause.seed_for(example_id, "a"|"b").
"""
from __future__ import annotations

import os
import sys
import time

import common as C
from regret_remasking.stale_cause import BLANKS, CLASSES, CacheControl, first_token_diff, seed_for

FRESH_OTHERS = tuple((c, "fresh") for c in CLASSES if c not in BLANKS)


def arm_ctrl(arm: str, qid: str) -> CacheControl:
    a, b = seed_for(qid, "a"), seed_for(qid, "b")
    allc = lambda m: tuple((c, m) for c in CLASSES)
    blanks = lambda m: FRESH_OTHERS + tuple((c, m) for c in BLANKS)
    return {
        "P-fresh": CacheControl(sources=allc("fresh"), stats=True),
        "P-stale": CacheControl(stats=True),
        "P-rand1a": CacheControl(sources=allc("noise"), noise_scale=1.0, noise_seed=a, stats=True),
        "P-rand1b": CacheControl(sources=allc("noise"), noise_scale=1.0, noise_seed=b, stats=True),
        "P-rand0.5": CacheControl(sources=allc("noise"), noise_scale=0.5, noise_seed=a, stats=True),
        "P-rand2": CacheControl(sources=allc("noise"), noise_scale=2.0, noise_seed=a, stats=True),
        "B-stale": CacheControl(sources=blanks("stale"), stats=True),
        "B-rand1": CacheControl(sources=blanks("noise"), noise_scale=1.0, noise_seed=a, stats=True),
    }[arm]


ARMS = ["P-fresh", "P-stale", "P-rand1a", "P-rand1b", "P-rand0.5", "P-rand2", "B-stale", "B-rand1"]
if os.environ.get("NOISE_ARMS"):          # a pre-registered cut (see PREREGISTERED.md): comma-separated subset, order kept
    ARMS = [a for a in ARMS if a in os.environ["NOISE_ARMS"].split(",")]


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "SMOKE"
    path = C.OUT / f"records/{'smoke_' if smoke else ''}noise_control.jsonl"
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    C.scorer_guard()
    ref = C.stage1_by_arm()
    dense_ref, s32_ref = ref["D:C-0.9"], ref["S32:C-0.9"]
    es = C.he164()
    if smoke:
        es = es[:2]
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    started = time.time()
    n_checked = 0
    for i, e in enumerate(es):
        qid = e["example_id"]
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            if (qid, arm) in done:
                continue
            C.gpu_guard()
            cfg = C.cached_cfg("C-0.9", ctx, ctrl=arm_ctrl(arm, qid))
            t0 = time.perf_counter()
            _, res = C.run_cached(ctx, e, cfg)
            lat = time.perf_counter() - t0
            tokens = list(res.tokens)
            text, correct, status = C.score(ctx, e, tokens)
            rec = {"example_id": qid, "arm": arm, "set": "he164", "tokens": tokens, "text": text, "first_eos": C.first_eos(tokens), "correct": correct, "scoring_status": status,
                   "nfe": res.nfe, "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "latency_s_not_a_speed_measurement": lat, "ctrl_stats": res.ctrl_stats,
                   "eq_s32_stage1": tokens == s32_ref[qid]["tokens"], "eq_dense_stage1": tokens == dense_ref[qid]["tokens"],
                   "first_token_diff_vs_dense": first_token_diff(tokens, dense_ref[qid]["tokens"]), "status": "ok"}
            C.append(path, rec)
            if arm == "P-stale" and i < 3 and not rec["eq_s32_stage1"]:
                raise SystemExit(f"PARITY FAILED: P-stale differs from the stage-1 S32:C-0.9 tokens on {qid}")
            if arm in ("P-stale", "P-fresh") and i < 3:
                n_checked += 1
                print(f"[parity] {qid} {arm}: eq_s32_stage1={rec['eq_s32_stage1']} eq_dense_stage1={rec['eq_dense_stage1']}", flush=True)
        print(f"[noise {i + 1}/{len(es)}] {qid} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print("[noise] done", flush=True)


if __name__ == "__main__":
    main()
