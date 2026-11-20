"""Experiment 3 (premise P3 on other architectures): the noise-matched control of the stale K/V (results/stale_cause_20261108, LLaDA-8B: noise of the same size as the staleness is harmless, +2 answers, while the real staleness costs 9) on Dream or LLaDA-MoE.
Every arm is the plain S32 decode (word lock Conf-0.9, block 32, no first-EOS truncation) on HumanEval; at the cached rows (prompt / prior_written / inblock_old / next_blank / far_blank) the K/V the attention reads at a non-dense call is, per arm:
  P-fresh   a dense pass over the current tokens (oracle; should reproduce the dense decode)       P-stale  the cache itself (= plain S32; the dense pass runs only for the statistics)
  P-rand1a / P-rand1b   fresh + a random perturbation of norm = 1.0 x ||stale - fresh|| per row and layer (two seeds)
Usage: KEY=dream|moe run_noise.py [SMOKE]; env N (default 164). Records: records/noise_<KEY>_he164.jsonl (answers deferred; ctrl_stats = stale-vs-fresh distances per class and layer)."""
import os
import sys

import exp_common as X

KEY = os.environ["KEY"]
D = __import__("dream_common" if KEY == "dream" else "moe_common")
from regret_remasking.selective_cache import decode_selective
from regret_remasking.stale_cause import CLASSES, CacheControl, seed_for

C = X.C
allc = lambda m: tuple((c, m) for c in CLASSES)


def arm_ctrl(arm, qid):
    a, b = seed_for(qid, "a"), seed_for(qid, "b")
    return {"P-fresh": CacheControl(sources=allc("fresh"), stats=True), "P-stale": CacheControl(stats=True),
            "P-rand1a": CacheControl(sources=allc("noise"), noise_scale=1.0, noise_seed=a, stats=True), "P-rand1b": CacheControl(sources=allc("noise"), noise_scale=1.0, noise_seed=b, stats=True)}[arm]


ARMS = ["P-stale", "P-rand1a", "P-rand1b", "P-fresh"]


def main():
    smoke = len(sys.argv) > 1 and sys.argv[1] == "SMOKE"
    es = C.he164()[: int(os.environ.get("N", "164"))]
    es_run = es[:2] if smoke else es
    C.gpu_guard()
    model, tok = D.load()
    path = X.OUT / f"{'smoke_' if smoke else ''}noise_{KEY}_he164.jsonl"
    done = {(r["example_id"], r["arm"]) for r in C.rows(path) if r.get("status") == "ok"}
    for i, e in enumerate(es_run):
        rot = i % len(ARMS)
        for arm in ARMS[rot:] + ARMS[:rot]:
            if (e["example_id"], arm) in done:
                continue
            C.gpu_guard()
            res = decode_selective(model, D.prompt_ids(tok, e), D.cfg_for(512, eos_truncation=False, ctrl=arm_ctrl(arm, e["example_id"])))
            toks = list(res.tokens)
            C.append(path, {"example_id": e["example_id"], "arm": arm, "set": "he164", "tokens": toks, "text": D.response_text(tok, toks), "first_eos": D.first_eos(toks), "correct": None, "scoring_status": "deferred",
                            "nfe": res.nfe, "ctrl_stats": res.ctrl_stats, "status": "ok"})
        print(f"[noise {KEY}] {i + 1}/{len(es_run)}", flush=True)
    print(f"[noise {KEY}] done", flush=True)


if __name__ == "__main__":
    main()
