"""Far-blank cache x word lock (stacking), one GPU process per stage (speed ratios are valid only inside a process).

Usage: <abs venv python> run_cache_stack.py STAGE [SMOKE]
  STAGE 1: MATH 108 (gen 512), HumanEval 0-163 (gen 512)        -> records/stage1.jsonl
  STAGE 2: MATH 54 even-indexed half (gen 1024), GSM8K test 0-199 (gen 256) -> records/stage2.jsonl
  SMOKE:   2 questions per set, records -> records/smoke_stageN.jsonl
Cells (9 per question, arm order rotated per question, one warm-up decode per cell and set, one timed repeat):
  D:C-0.9, D:C-0.8, D:Lctx-0.8        dense word lock only, existing dense decoder (paper_decode C-block / X-block), standard model path, nothing cached
  S16:<lock>                           far-blank cache, query rows = window of 16 blanks + rows written last call, EOS truncation, dense refresh at every block start
  S32:<lock>                           far-blank cache, query rows = blanks of the current 32-block + rows written last call, EOS truncation, dense refresh at every block start
  lock in {C-0.9, C-0.8, Lctx-0.8}; the Lctx rule is the word-lock agent's math512 rule (sha256-checked, never refit). In S16 it is a TRANSFER (fitted on block-32 decodes).
The process aborts before the main run if a parity check fails (dense cells vs the word-lock agent's records; S16:C-0.9 vs selective_cache Part C block+EOS tokens;
S32 with a dense refresh every call vs the dense C-0.9 cell) and if the HumanEval scorer guard sees sandbox_error. OMP/MKL/OpenBLAS threads are fixed to 1 (the Lctx
gradient-boosting predict is load sensitive).
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_MODULES_CACHE"] = str(ROOT / "results/repfreeze_tests_20261029/runtime_cache/model_modules")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(OUT / "source/src"))   # frozen copy first
sys.path.insert(0, str(HERE))
import torch  # noqa: E402

torch.set_num_threads(1)
from regret_remasking.context_rules import ContextRule  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
WL = ROOT / "results/longlen_wordlock_20261103"
RULE = WL / "rules/math512/Lctx_c0.8.pkl"
BLOCK, SEED = 32, 23
LOCKS = {"C-0.9": 0.9, "C-0.8": 0.8, "Lctx-0.8": None}
ARMS = ["D:C-0.9", "D:C-0.8", "D:Lctx-0.8", "S16:C-0.9", "S16:C-0.8", "S16:Lctx-0.8", "S32:C-0.9", "S32:C-0.8", "S32:Lctx-0.8"]
STAGES = {1: [("he164", 512), ("math108", 512)], 2: [("gsm8k200", 256), ("math54half", 1024)]}   # the cheap set first: a scoring/loader bug shows up early
OTHER_MARKERS = [ROOT / "results/repfreeze_tests_20261029/GPU_IN_USE", ROOT / "results/longlen_wordlock_20261103/GPU_IN_USE", ROOT / "results/selective_cache_20261104/GPU_IN_USE"]


def rows(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def append(path, rec):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a") as h:
        h.write(json.dumps(rec, sort_keys=True, allow_nan=False) + "\n")
        h.flush()
        os.fsync(h.fileno())


def question_set(name):
    M = OUT / "manifests"
    if name in ("math108", "math54half"):
        l5, l14 = rows(M / "math_l5.jsonl"), rows(M / "math_l14.jsonl")
        if name == "math54half":
            l5, l14 = l5[::2], l14[::2]
        return [dict(e, _set=name, _stratum="L5") for e in l5] + [dict(e, _set=name, _stratum="L1-4") for e in l14]
    return [dict(e, _set=name, _stratum=name) for e in rows(M / ("he164.jsonl" if name == "he164" else "gsm8k200.jsonl"))]


def gpu_guard():
    uuid = "GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-")
    out = subprocess.run(["nvidia-smi", "--id=" + uuid, "--query-compute-apps=pid,process_name,used_gpu_memory", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True)
    others = [ln for ln in out.stdout.splitlines() if ln.split(",")[0].strip().isdigit() and int(ln.split(",")[0].strip()) != os.getpid()]
    if others:
        raise RuntimeError("GPU contention: " + repr(others))
    for m in OTHER_MARKERS:
        if m.exists() and Path(f"/proc/{m.read_text().split()[0]}").exists():
            raise RuntimeError(f"another agent's marker is live: {m}")


def load_rule():
    rules = json.loads((WL / "rules/math512/rules.json").read_text())["rules"]["Lctx_c0.8"]
    sha = hashlib.sha256(RULE.read_bytes()).hexdigest()
    assert sha == rules["sha256"], f"Lctx rule file does not match rules.json: {sha}"
    rule = ContextRule.load(RULE)
    assert abs(rule.lam - rules["lambda"]) < 1e-12
    return rule, sha


def cache_config(arm, gen, ctx, **over):
    shape, lock = arm.split(":")
    c = dict(gen_length=gen, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=BLOCK,
             refresh="block", eos_truncation=True)
    if shape == "S16":
        c.update(shape="window", lock="rule" if lock.startswith("Lctx") else "elastic")
    else:
        c.update(shape="block", lock="rule" if lock.startswith("Lctx") else "conf")
    if lock.startswith("Lctx"):
        c.update(rule=ctx["rule"], threshold=0.9)
    else:
        c.update(threshold=LOCKS[lock])
    c.update(over)
    return SelectiveConfig(**c)


def decode(ctx, arm, e, gen, **over):
    model, tok = ctx["model"], ctx["tok"]
    prompt = _resolve_prompt(e)
    torch.cuda.reset_peak_memory_stats()
    if arm.startswith("D:"):
        lock = arm[2:]
        spec = dict(arm="C-block", threshold=LOCKS[lock]) if lock.startswith("C") else dict(arm="X-block", block_rule=ctx["rule"])
        cfg = PaperDecodeConfig(gen_length=gen, block_length=BLOCK, mask_id=MASK_ID, **spec)
        torch.manual_seed(SEED)
        r = decode_paper_policy(model, tok, prompt, e["example_id"], cfg)
        tokens = list(r.token_ids)
        extra = {"latency_s": r.latency_s, "nfe": r.nfe, "prompt_len": r.prompt_length, "text": r.text, "rows_per_call_mean": r.total_query_rows / r.nfe,
                 "rule_time_s": r.scorer_time_s, "fallback_calls": r.fallback_count, "dense_calls": r.nfe, "refresh_calls": 0}
    else:
        ids, _ = _prepare_prompt(tok, prompt, "cuda")
        res = decode_selective(model, ids, cache_config(arm, gen, ctx, **over))
        tokens = list(res.tokens)
        L = len(res.layer_rows[0])
        extra = {"latency_s": res.latency_s, "nfe": res.nfe, "prompt_len": res.prompt_length, "text": _decode_response(tok, tokens),
                 "rows_per_call_mean": sum(sum(c) for c in res.layer_rows) / L / res.nfe, "rule_time_s": res.rule_time_s, "fallback_calls": res.fallback_calls,
                 "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "refresh_calls": 0, "eos_position": res.eos_position}
    first_eos = next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)
    return {"tokens": tokens, "first_eos": first_eos, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, **extra}


def parity(ctx, sets):
    """Before the main run (3 questions): dense cells vs the word-lock agent's 512 records; S16:C-0.9 vs selective_cache Part C block+EOS; S32 dense refresh every call vs D:C-0.9."""
    ref = {(r["example_id"], r["policy_id"]): r for r in json.loads((WL / "validation/parity_ref512.json").read_text())}
    partc = {r["example_id"]: r for r in rows(ROOT / "results/selective_cache_20261104/partC/records.jsonl") if r["arm"] == "W16-cache-block+EOS"}
    es = {e["example_id"]: e for e in question_set("math108")}
    out = []
    for qid in ("math500-l5-1", "math500-l5-9", "math500-l5-11"):
        e = es[qid]
        d = {}
        for arm, pol in (("D:C-0.9", "C-0.9"), ("D:C-0.8", "C-0.8"), ("D:Lctx-0.8", "Lctx-0.8")):
            r = decode(ctx, arm, e, 512)
            d[arm] = r
            old = ref[(qid, pol)]
            out.append({"q": qid, "check": f"{arm} vs word-lock records", "tokens_equal": r["tokens"] == list(old["token_ids"]), "calls_equal": r["nfe"] == old["nfe"]})
        if qid in partc:
            r = decode(ctx, "S16:C-0.9", e, 512)
            out.append({"q": qid, "check": "S16:C-0.9 vs Part C block+EOS", "tokens_equal": r["tokens"] == partc[qid]["tokens"], "calls_equal": r["nfe"] == partc[qid]["nfe"]})
        for arm in ("S32:C-0.9", "S32:Lctx-0.8"):
            r = decode(ctx, arm, e, 512, refresh="every_call", eos_truncation=False)
            dd = d["D:C-0.9"] if "C-0.9" in arm else d["D:Lctx-0.8"]
            out.append({"q": qid, "check": f"{arm} with dense refresh every call vs {'D:C-0.9' if 'C-0.9' in arm else 'D:Lctx-0.8'}", "tokens_equal": r["tokens"] == dd["tokens"], "calls_equal": r["nfe"] == dd["nfe"]})
        for x in out[-8:]:
            if x["q"] == qid:
                print("[parity]", json.dumps(x), flush=True)
    ok = all(x["tokens_equal"] and x["calls_equal"] for x in out)
    (OUT / "validation/parity.json").write_text(json.dumps({"rows": out, "all_match": ok}, indent=1) + "\n")
    if not ok:
        raise SystemExit("parity mismatch: see validation/parity.json")


def scorer_guard():
    he = rows(OUT / "manifests/he164.jsonl")[0]
    correct, status = score_paper_generation("```python\n" + he["question"] + he["metadata"]["canonical_solution"] + "\n```", he)
    print("[scorer guard]", correct, status, sys.executable, flush=True)
    if status == "sandbox_error" or not correct or not sys.executable.startswith("/"):
        raise SystemExit("HumanEval scorer guard failed: start with an absolute interpreter path")


def main():
    stage = int(sys.argv[1])
    smoke = len(sys.argv) > 2 and sys.argv[2] == "SMOKE"
    path = OUT / f"records/{'smoke_' if smoke else ''}stage{stage}.jsonl"
    gpu_guard()
    rule, sha = load_rule()
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    ctx = {"model": model, "tok": tok, "rule": rule, "rule_sha": sha}
    print(f"[stack] stage {stage} loaded; rule sha {sha[:12]}; load average {os.getloadavg()}; threads {torch.get_num_threads()}", flush=True)
    if stage == 1 and not smoke:
        scorer_guard()
        parity(ctx, None)
    for name, gen in STAGES[stage]:
        es = question_set(name)
        if smoke:
            es = es[:1] + es[-1:]
        done = {(r["set"], r["example_id"], r["arm"]) for r in rows(path) if r.get("status") == "ok"}
        for arm in ARMS:
            decode(ctx, arm, es[0], gen)      # warm-up, not recorded
        started = time.time()
        for i, e in enumerate(es):
            rot = i % len(ARMS)
            for arm in ARMS[rot:] + ARMS[:rot]:
                if (name, e["example_id"], arm) in done:
                    continue
                gpu_guard()
                r = decode(ctx, arm, e, gen)
                correct, status = score_paper_generation(r["text"], e)
                if status == "sandbox_error":
                    raise SystemExit("sandbox_error while scoring: abort")
                r.update({"example_id": e["example_id"], "set": name, "stratum": e["_stratum"], "arm": arm, "gen": gen, "correct": bool(correct), "scoring_status": status,
                          "status": "ok", "loadavg1": os.getloadavg()[0]})
                append(path, r)
            print(f"[stack {name}] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h) load {os.getloadavg()[0]:.1f}", flush=True)
    print("[stack] done", flush=True)


if __name__ == "__main__":
    main()
