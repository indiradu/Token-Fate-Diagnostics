"""Experiment 6: tight parity of R0 against dense Conf-0.9 on EXTRA prompts (LLaDA-8B): prompts not used in Study 5.
Sets: kod_ext = kod_held400[150:400] + kod_tune400 (650 code prompts; kod_tune was used to fit the controller's risk model and the screen, not R0), math_ext = math_tune200 (200), gsm_ext = gsm8k_tune200[0:100].
Arms DENSE (C-block 0.9) and R0 (derived from a calibration on the first 8 prompts of the set), interleaved. Usage: run_par.py SET [SMOKE]; env BENCH_JSON. Records: records/par_<tag>_<SET>.jsonl."""
import os
import sys

import exp_common as X
import run_rule as R

C = X.C
MAN = C.OUT / "manifests"


def questions(ds):
    if ds == "kod_ext":
        a = [dict(e, _set="kod_held") for e in C.rows(MAN / "kod_held400.jsonl")][150:400]
        b = [dict(e, _set="kod_held") for e in C.rows(MAN / "kod_tune400.jsonl")]
        return a + b
    if ds == "math_ext":
        return [dict(e, _set="math108") for e in C.rows(MAN / "math_tune200.jsonl")]
    if ds == "gsm_ext":
        return [dict(e, _set="gsm_held") for e in C.rows(MAN / "gsm8k_tune200.jsonl")][:100]
    raise SystemExit(ds)


GEN = {"kod_ext": 512, "math_ext": 512, "gsm_ext": 256}
SCORE_AS = {"kod_ext": "kod_held", "math_ext": "math108", "gsm_ext": "gsm_held"}


def main():
    ds = sys.argv[1]
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    gen = GEN[ds]
    es = questions(ds)
    es_run = es[:2] if smoke else es
    C.gpu_guard()
    model, tok = C.load_model()
    ctx = {"model": model, "tok": tok}
    tag = os.environ.get("ORIG_TAG", "llada8b")
    st = X.calibrate(ctx, es, gen, 0.9, os.environ["BENCH_JSON"])
    rule = X.rule_of(st)
    print(f"[par {ds}] rule {({k: v for k, v in rule.items() if k != 'detail'})}", flush=True)
    path = X.OUT / f"{'smoke_' if smoke else ''}par_{tag}_{ds}.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    X.dense(ctx, es[0], gen, 0.9)
    arms = ["DENSE", "R0"]
    for i, e in enumerate(es_run):
        rot = i % 2
        for arm in arms[rot:] + arms[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            d = X.dense(ctx, e, gen, 0.9) if arm == "DENSE" else X.selective(ctx, e, gen, 0.9, **X.config_kwargs(rule))
            extra = dict(_score=X.score(SCORE_AS[ds], d, e))
            if arm == "R0":
                extra["rule"] = {k: v for k, v in rule.items() if k != "detail"}
            X.append(path, e, arm, ds, gen, d, **extra)
        if (i + 1) % 10 == 0:
            print(f"[par {ds}] {i + 1}/{len(es_run)}", flush=True)
    print(f"[par {ds}] done", flush=True)


if __name__ == "__main__":
    main()
