"""Experiment 2: ablation of R0 (LLaDA-8B, word lock Conf-0.9). Arms interleaved prompt by prompt with the dense decode:
  DENSE, S32 (plain cache), R0 (derived, as in Study 5), R0_A25 / R0_A50 (start-layer fraction 1/4 and 1/2 instead of 1/3), R0_B05 / R0_B20 (refresh budget 5% / 20% instead of 10%),
  FIX_K2 / FIX_K4 / FIX_K8 (blank rows rejoin from layer 10 every 2 / 4 / 8 calls, not derived).
Usage: run_abl.py SET [SMOKE]; env N (prompts, default 100), BENCH_JSON. Records: records/abl_<tag>_<SET>.jsonl."""
import os
import sys

import exp_common as X
import run_rule as R

C = X.C
FIX = {f"FIX_K{k}": dict(partial_every_k=k, partial_alpha=1 / 3, partial_join="blank") for k in (2, 4, 8)}
VAR = {"R0_A25": dict(alpha0=0.25), "R0_A50": dict(alpha0=0.5), "R0_B05": dict(beta=0.05), "R0_B20": dict(beta=0.20)}
ARMS = ["DENSE", "S32", "R0"] + list(VAR) + list(FIX)


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = R.SETS[ds]
    es = R.questions(ds)[: int(os.environ.get("N", "100"))]
    es_run = es[:2] if smoke else es
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    tag = os.environ.get("ORIG_TAG", "llada8b")
    st = X.calibrate(ctx, es, gen, 0.9, os.environ["BENCH_JSON"])
    rules = {"R0": X.rule_of(st), **{a: X.rule_of(st, **kw) for a, kw in VAR.items()}}
    for a, r in rules.items():
        print(f"[abl {ds}] {a}: {({k: v for k, v in r.items() if k != 'detail'})}", flush=True)
    path = X.OUT / f"{'smoke_' if smoke else ''}abl_{tag}_{ds}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    X.dense(ctx, es[0], gen, 0.9)
    for i, e in enumerate(es_run):
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            if arm == "DENSE":
                d = X.dense(ctx, e, gen, 0.9)
            elif arm == "S32":
                d = X.selective(ctx, e, gen, 0.9)
            elif arm in rules:
                d = X.selective(ctx, e, gen, 0.9, **X.config_kwargs(rules[arm]))
            else:
                d = X.selective(ctx, e, gen, 0.9, **FIX[arm])
            extra = dict(_score=X.score(ds, d, e))
            if arm in rules:
                extra["rule"] = {k: v for k, v in rules[arm].items() if k != "detail"}
            X.append(path, e, arm, ds, gen, d, **extra)
        print(f"[abl {ds}] {i + 1}/{len(es_run)}", flush=True)
    print(f"[abl {ds}] done", flush=True)


if __name__ == "__main__":
    main()
