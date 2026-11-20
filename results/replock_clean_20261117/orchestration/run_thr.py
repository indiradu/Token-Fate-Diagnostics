"""Experiment 1: threshold sweep for the speedup decomposition (LLaDA-8B). The Conf threshold sets the calls per block m; for each threshold three arms run interleaved prompt by prompt: DENSE (paper_decode C-block), S32 (plain cache) and R0
(derived per threshold from a calibration of calls per block on the first 8 prompts). Threshold 1.0 = (almost) only the single best blank is written per call (the original decoder's one token per step), the B0 point.
Usage: run_thr.py SET [SMOKE]; env N (prompts, default 60; B0 point uses 20), THRS (comma list), BENCH_JSON. Records: records/thr_<tag>_<SET>.jsonl. Answers: code deferred; GSM8K/MATH in process."""
import os
import sys
import time

import exp_common as X
import run_rule as R

C = X.C
THRS = [float(t) for t in os.environ.get("THRS", "0.5,0.6,0.7,0.8,0.9,0.95,0.99,1.0").split(",")]


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = R.SETS[ds]
    n = int(os.environ.get("N", "60"))
    es = R.questions(ds)[:n]
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    tag = os.environ.get("ORIG_TAG", "llada8b")
    path = X.OUT / f"{'smoke_' if smoke else ''}thr_{tag}_{ds}.jsonl"
    done = {(r["example_id"], r["arm"], r["thr"]) for r in C.rows(path) if r.get("status") == "ok"}
    X.dense(ctx, es[0], gen, 0.9)
    for thr in THRS if not smoke else THRS[:1] + [1.01]:
        es_t = es[: (20 if thr >= 1.0 else n)]
        es_run = es_t[:2] if smoke else es_t
        st = X.calibrate(ctx, es_t, gen, thr, os.environ["BENCH_JSON"])
        rule = X.rule_of(st)
        print(f"[thr {ds}] thr {thr} calls/block {st.calls_per_block:.2f} prompt {st.prompt_len:.0f} rule {({k: v for k, v in rule.items() if k != 'detail'})}", flush=True)
        arms = ["DENSE", "S32", "R0"]
        for i, e in enumerate(es_run):
            rot = i % len(arms)
            for arm in arms[rot:] + arms[:rot]:
                if (e["example_id"], arm, thr) in done:
                    continue
                C.gpu_guard()
                d = X.dense(ctx, e, gen, thr) if arm == "DENSE" else X.selective(ctx, e, gen, thr, **({} if arm == "S32" else X.config_kwargs(rule)))
                extra = dict(thr=thr, calls_per_block=st.calls_per_block, _score=X.score(ds, d, e))
                if arm == "R0":
                    extra["rule"] = {k: v for k, v in rule.items() if k != "detail"}
                X.append(path, e, arm, ds, gen, d, **extra)
            print(f"[thr {ds}] thr {thr} {i + 1}/{len(es_run)}", flush=True)
    print(f"[thr {ds}] done", flush=True)


if __name__ == "__main__":
    main()
