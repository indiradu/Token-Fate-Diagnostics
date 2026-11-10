"""Learn2PD x C-c x Lctx word-lock control, one GPU process per stage (speed ratios are valid only inside a process).

Usage: <abs venv python> run_l2pd_stack.py SET [SMOKE|NOPARITY]      SET = he164 | math108 | gsm8k200 | math54half | pilot | (calib with mode CALIB)
  one process per SET -> records/stage_<SET>.jsonl (HumanEval 164 gen 512; MATH 108 gen 512; stage 2: GSM8K 200 gen 256, MATH 54 half gen 1024)
  pilot: 2 held-out TRAINING-split questions, all cells, no scoring, no parity (records/pilot.jsonl): crash test + timing only.
  calib: the 14 held-out training MATH questions, dense cells at several tau (records/calib.jsonl): freezes tau_matched.
Cells  <kind>:<lock>
  kind D    dense word lock, existing dense decoder (paper_decode C-block / X-block), standard model path, no EOS truncation
  kind T    "dense + truncation" twin: the selective engine with a dense call EVERY call (shape=block, refresh=every_call) and first-EOS truncation
  kind S32  selective cache: query rows = current 32-block's blanks (all 32 slots of the block for L2PD) + rows written last call; dense call at every block start; first-EOS truncation
  lock  C-0.9 C-0.8 C-0.7 C-0.6 (confidence thresholds) | Lctx-0.8 (learned context rule, 0.8 = precision target) | L2PD-pub (retrained filter, tau 0.96) |
        L2PD-match (retrained filter, tau chosen so that realized calls equal C-0.8's on the held-out training questions) | L2PD-off (official released weights, tau 0.96)
The process aborts before the main run if a parity check fails and if the HumanEval scorer guard sees sandbox_error. OMP/MKL/OpenBLAS threads fixed to 1.
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
from regret_remasking.l2pd_rule import Learn2PDRule  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _decode_response, _prepare_prompt  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, decode_selective  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
WL = ROOT / "results/longlen_wordlock_20261103"
FC = ROOT / "results/farcache_stack_20261105"
RULE = WL / "rules/math512/Lctx_c0.8.pkl"
BLOCK, SEED = 32, 23
THRESH = {"C-0.9": 0.9, "C-0.8": 0.8, "C-0.7": 0.7, "C-0.6": 0.6}
# Cell lists per set (edited once, after the pilot projection; recorded in PREREGISTERED.md)
# Budget cut (see PREREGISTERED.md): dense C-0.7 / C-0.6 and the twins of Lctx / L2PD-pub are NOT run; lowered thresholds are compared in the cache cells.
LOCKS_D = ["C-0.9", "C-0.8", "Lctx-0.8", "L2PD-pub", "L2PD-match", "L2PD-off"]
LOCKS_T = ["C-0.9", "L2PD-match"]
LOCKS_S = ["C-0.9", "C-0.8", "C-0.7", "C-0.6", "Lctx-0.8", "L2PD-pub", "L2PD-match"]
CELLS = {s: [f"D:{l}" for l in LOCKS_D] + [f"T:{l}" for l in LOCKS_T] + [f"S32:{l}" for l in LOCKS_S] for s in ("he164", "math108", "gsm8k200", "math54half", "pilot")}
CALIB_TAUS = [0.90, 0.93, 0.96, 0.98, 0.99]       # tau calibration on held-out TRAINING questions only (dense decodes; retrained filter)
CELLS["calib"] = ["D:C-0.9", "D:C-0.8", "D:Lctx-0.8", "D:L2PD-off"] + [f"D:L2PD-t{t:.2f}" for t in CALIB_TAUS]
STAGES = {1: [("he164", 512), ("math108", 512)], 2: [("gsm8k200", 256), ("math54half", 1024)]}
OTHER_MARKERS = [ROOT / "results/repfreeze_tests_20261029/GPU_IN_USE", ROOT / "results/longlen_wordlock_20261103/GPU_IN_USE", ROOT / "results/selective_cache_20261104/GPU_IN_USE",
                 ROOT / "results/farcache_stack_20261105/GPU_IN_USE"]
PROC = f"{os.getpid()}-{int(time.time())}"


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
    if name in ("pilot", "calib"):      # held-out training-split questions (role 'thr'), never a test question; the collected prefix only (index < 56)
        thr = [e for i, e in enumerate(rows(M / "train_math.jsonl")[:56]) if e["metadata"]["role"] == "thr"]
        return [dict(e, _set=name, _stratum=name) for e in (thr[:2] if name == "pilot" else thr)]
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


def verify_frozen_source():
    sha = json.loads((OUT / "source/SHA256.json").read_text())
    for rel, h in sha["files"].items():
        if hashlib.sha256((OUT / "source" / rel).read_bytes()).hexdigest() != h:
            raise SystemExit(f"frozen source changed: {rel}")
    return sha["aggregate"]


def load_rules():
    rules = json.loads((WL / "rules/math512/rules.json").read_text())["rules"]["Lctx_c0.8"]
    sha = hashlib.sha256(RULE.read_bytes()).hexdigest()
    assert sha == rules["sha256"], f"Lctx rule file does not match rules.json: {sha}"
    lctx = ContextRule.load(RULE)
    assert abs(lctx.lam - rules["lambda"]) < 1e-12
    meta = json.loads((OUT / "filters/rules.json").read_text())
    F = OUT / "filters"
    out = {"Lctx-0.8": lctx}
    for lock, key, tau in (("L2PD-pub", "retrained", meta["retrained"]["tau_published"]), ("L2PD-match", "retrained", meta["retrained"]["tau_matched"]), ("L2PD-off", "official", meta["official"]["tau_published"])):
        path = F / meta[key]["file"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == meta[key]["file_sha256"], f"{path} does not match filters/rules.json"
        out[lock] = Learn2PDRule.from_file(path, tau, name=lock)
    for t in CALIB_TAUS:
        out[f"L2PD-t{t:.2f}"] = Learn2PDRule.from_file(F / meta["retrained"]["file"], t, name=f"L2PD-t{t:.2f}")
    return out, {"lctx_sha": sha, "filters": meta}


def is_l2pd(lock):
    return lock.startswith("L2PD")


def selective_cfg(kind, lock, gen, ctx, **over):
    c = dict(gen_length=gen, window=16, far_mask_cache=True, mask_id=MASK_ID, eos_id=EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=BLOCK,
             refresh="every_call" if kind == "T" else "block", eos_truncation=True, shape="block")
    if lock in THRESH:
        c.update(lock="conf", threshold=THRESH[lock])
    elif is_l2pd(lock):
        c.update(lock="rule", rule=ctx["rules"][lock], block_rows=True, threshold=0.9)
    else:
        c.update(lock="rule", rule=ctx["rules"][lock], threshold=0.9)
    c.update(over)
    return SelectiveConfig(**c)


def decode(ctx, arm, e, gen, **over):
    model, tok = ctx["model"], ctx["tok"]
    prompt = _resolve_prompt(e)
    kind, lock = arm.split(":")
    rule = ctx["rules"].get(lock)
    if is_l2pd(lock):
        rule.reset_stats()
    torch.cuda.reset_peak_memory_stats()
    if kind == "D":
        spec = dict(arm="C-block", threshold=THRESH[lock]) if lock in THRESH else dict(arm="X-block", block_rule=rule)
        cfg = PaperDecodeConfig(gen_length=gen, block_length=BLOCK, mask_id=MASK_ID, **spec)
        torch.manual_seed(SEED)
        r = decode_paper_policy(model, tok, prompt, e["example_id"], cfg)
        tokens = list(r.token_ids)
        extra = {"latency_s": r.latency_s, "nfe": r.nfe, "prompt_len": r.prompt_length, "text": r.text, "rows_per_call_mean": r.total_query_rows / r.nfe,
                 "rule_time_s": r.scorer_time_s, "fallback_calls": r.fallback_count, "dense_calls": r.nfe, "words_written": gen, "mask_rejected": getattr(r, "mask_proposals_rejected", None)}
    else:
        ids, _ = _prepare_prompt(tok, prompt, "cuda")
        res = decode_selective(model, ids, selective_cfg(kind, lock, gen, ctx, **over))
        tokens = list(res.tokens)
        L = len(res.layer_rows[0])
        extra = {"latency_s": res.latency_s, "nfe": res.nfe, "prompt_len": res.prompt_length, "text": _decode_response(tok, tokens),
                 "rows_per_call_mean": sum(sum(c) for c in res.layer_rows) / L / res.nfe, "rule_time_s": res.rule_time_s, "fallback_calls": res.fallback_calls,
                 "dense_calls": sum(1 for k in res.refresh_kind if k in ("first", "dense")), "eos_position": res.eos_position, "words_written": res.written_total}
    first_eos = next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), None)
    out = {"tokens": tokens, "first_eos": first_eos, "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, **extra}
    if is_l2pd(lock):
        out["l2pd_stats"] = dict(rule.stats)
    return out


def to_eos(tokens):
    k = next((i for i, t in enumerate(tokens) if t in (EOS_ID, EOT_ID)), len(tokens) - 1)
    return tokens[:k + 1]


def parity(ctx, stage_sets):
    """Before the main run: (1) dense C-0.9 / C-0.8 / Lctx-0.8 reproduce the far-cache campaign's dense tokens; (2) each twin T:<lock> equals D:<lock> up to the first EOS;
    (3) S32 with a dense call every call and truncation is the twin (same engine); (4) L2PD dense vs twin."""
    fc = {(r["set"], r["example_id"], r["arm"]): r for r in rows(FC / "records/stage1.jsonl") if r.get("status") == "ok"}
    picks = [("he164", "HumanEval/0", 512), ("he164", "HumanEval/1", 512), ("math108", "math500-l5-1", 512), ("math108", "math500-l5-9", 512)]
    out = []
    for sname, qid, gen in picks:
        e = next(x for x in question_set(sname) if x["example_id"] == qid)
        d = {}
        for lock in ("C-0.9", "C-0.8", "Lctx-0.8"):
            r = decode(ctx, f"D:{lock}", e, gen)
            d[lock] = r
            old = fc[(sname, qid, f"D:{lock}")]
            out.append({"q": qid, "check": f"D:{lock} vs far-cache campaign dense record", "tokens_equal": r["tokens"] == old["tokens"], "calls_equal": r["nfe"] == old["nfe"]})
        for lock in ("C-0.9", "Lctx-0.8"):
            t = decode(ctx, f"T:{lock}", e, gen)
            out.append({"q": qid, "check": f"T:{lock} vs D:{lock} up to the first EOS", "tokens_equal": to_eos(t["tokens"]) == to_eos(d[lock]["tokens"]), "calls_equal": t["nfe"] <= d[lock]["nfe"]})   # truncation removes padding blanks: never more calls
        for lock in ("L2PD-match", "L2PD-pub", "L2PD-off"):
            dd = decode(ctx, f"D:{lock}", e, gen)
            tt = decode(ctx, f"T:{lock}", e, gen) if lock != "L2PD-off" else None
            if tt is not None:
                out.append({"q": qid, "check": f"T:{lock} vs D:{lock} up to the first EOS", "tokens_equal": to_eos(tt["tokens"]) == to_eos(dd["tokens"]), "calls_equal": tt["nfe"] <= dd["nfe"]})
        for x in out[-9:]:
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
    stage = sys.argv[1]                       # a set name (he164, math108, gsm8k200, math54half) or 'pilot' / 'calib'
    mode = sys.argv[2] if len(sys.argv) > 2 else ""
    smoke, pilot, calib = mode == "SMOKE", stage == "pilot", mode == "CALIB"
    agg = verify_frozen_source()
    if calib:
        path, plan = OUT / "records/calib.jsonl", [("calib", 512)]
    elif pilot:
        path, plan = OUT / "records/pilot.jsonl", [("pilot", 512)]
    else:
        path = OUT / f"records/{'smoke_' if smoke else ''}stage_{stage}.jsonl"
        plan = [(stage, {"he164": 512, "math108": 512, "gsm8k200": 256, "math54half": 1024}[stage])]
    gpu_guard()
    rules, meta = load_rules()
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    ctx = {"model": model, "tok": tok, "rules": rules}
    print(f"[l2pd] stage {stage} {mode} loaded; source {agg[:12]}; proc {PROC}; load average {os.getloadavg()}; threads {torch.get_num_threads()}", flush=True)
    if stage == "he164" and not smoke and not pilot and not calib:
        scorer_guard()
    if mode != "NOPARITY" and not smoke and not calib and not pilot:
        parity(ctx, plan)
    for name, gen in plan:
        es = question_set(name)
        if smoke:
            es = es[:1] + es[-1:]
        arms = CELLS[name]
        done = {(r["set"], r["example_id"], r["arm"]) for r in rows(path) if r.get("status") == "ok"}
        for arm in arms:
            decode(ctx, arm, es[0], gen)      # warm-up, not recorded
        started = time.time()
        for i, e in enumerate(es):
            rot = i % len(arms)
            for arm in arms[rot:] + arms[:rot]:
                if (name, e["example_id"], arm) in done:
                    continue
                gpu_guard()
                r = decode(ctx, arm, e, gen)
                if pilot or calib:
                    correct, status = None, "not_scored"
                else:
                    correct, status = score_paper_generation(r["text"], e)
                if status == "sandbox_error":
                    raise SystemExit("sandbox_error while scoring: abort")
                r.update({"example_id": e["example_id"], "set": name, "stratum": e["_stratum"], "arm": arm, "gen": gen, "correct": None if correct is None else bool(correct),
                          "scoring_status": status, "status": "ok", "loadavg1": os.getloadavg()[0], "proc": PROC})
                append(path, r)
            print(f"[l2pd {name}] {i + 1}/{len(es)} {e['example_id']} ({(time.time() - started) / 3600:.2f} h) load {os.getloadavg()[0]:.1f}", flush=True)
    print("[l2pd] done", flush=True)


if __name__ == "__main__":
    main()
