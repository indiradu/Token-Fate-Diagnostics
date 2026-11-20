"""[FLOP-mode copy of run_d2c_dream.py with the flops-field fix; use with FLOPS=1] Study 5 (clean rerun): representation caches from the d2Cache framework (github.com/Kamichanw/d2Cache), run UNPATCHED with the Conf-0.9 word lock passed as a setting.
Usage: <venv python> run_d2c.py ARM SET [SMOKE]   with PYTHONPATH=/l/users/rui.gao/envs/d2c:/l/users/rui.gao/tools/d2Cache (+ FAMILY_SRC for the shared prompt/scoring helpers).
ARM (one attention implementation per process; the model is loaded once):
  DENSE_SDPA    no cache, Conf-0.9 (generation.threshold 0.9, num_transfer_tokens 1), sdpa attention   (same-stack dense anchor)
  DENSE_EAGER   no cache, Conf-0.9, eager attention                                                   (anchor for d2Cache, which needs eager)
  FD_DUAL       Fast-dLLM dual cache (cache=prefix, use_dual=true), sdpa
  DLLM          dLLM-Cache (cache=dllm: kp 25, kr 5, rou 0.25: the settings of the dLLM-Cache paper's HumanEval runs), sdpa
  D2CACHE       d2Cache (cache=d2cache: rollout_p 0.1, current_k 32, inflate_w 0), eager. generation.sigma stays null so the word lock is still Conf-0.9
Word lock in every arm: write every blank of the block with confidence >= 0.9, at least one (confidence_unmasking(threshold=0.9, min_transfer_tokens=1)); block 32; temperature 0; stop_until_eos false (decode all positions, as the released code does).
Prompts, tokenizer and scorer are the campaign's shared ones (common.py), NOT lm-eval's, so every arm sees identical inputs.
Records: results/replock_clean_20261117/records/d2c_<TAG>_<SET>_<ARM>.jsonl (tokens, latency_s, nfe = number of forward calls (len of the decode record), text, correct (deferred for code), peak memory)."""
from __future__ import annotations

import contextlib
import functools
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, "/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/model_transfer_20261113/orchestration")
import common as C                                              # shared prompts, scorer, jsonl helpers (read-only use)
import torch
from flopcount import N_FLOPS, count
from run_rule import CODE, DEFER, SETS, questions

OUT = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
TAG = os.environ.get("ORIG_TAG", "llada8b")
MASK, EOS = 126336, 126081
ARMS = {"DENSE_SDPA": ("sdpa", None), "DENSE_EAGER": ("eager", None), "FD_DUAL": ("sdpa", "prefix_dual"), "DLLM": ("sdpa", "dllm"), "D2CACHE": ("eager", "d2cache"), "ORIG_NATIVE": ("sdpa", None)}      # ORIG_NATIVE: Dream's own decoding (entropy order, one token per step over the whole length, no cache)


def cache_cls(kind):
    from src.cache import PrefixCache, d2Cache, dLLMCache
    if kind is None:
        return None
    if kind == "prefix_dual":
        return functools.partial(PrefixCache, use_dual=True)
    if kind == "dllm":
        return functools.partial(dLLMCache, kr=5, kp=25, rou=0.25)
    if kind == "d2cache":
        return functools.partial(d2Cache, rollout_p=0.1, current_k=32, sigma=10.0, inflate_w=0)
    raise SystemExit(kind)


def main():
    arm, ds = sys.argv[1], sys.argv[2]
    smoke = len(sys.argv) > 3 and sys.argv[3] == "SMOKE"
    attn, kind = ARMS[arm]
    gen = SETS[ds]
    es = questions(ds)
    if os.environ.get("PUB_LIMIT"):
        es = es[: int(os.environ["PUB_LIMIT"])]
    FL = os.environ.get("FLOPS") == "1"
    es_run = es[:2] if smoke else (es[:N_FLOPS] if FL else es)
    C.gpu_guard()
    from omegaconf import OmegaConf
    from src.generation import generate
    from src.utils.models import load_pretrained_model
    from transformers import AutoTokenizer
    ck = str(C.CHECKPOINT)
    family = "dream" if "dream" in ck.lower() else "llada"
    cfg = OmegaConf.create({"model": {"name": f"{family}-inst", "path": ck}})
    model = load_pretrained_model(cfg, trust_remote_code=True, torch_dtype=torch.bfloat16, attn_implementation=attn).eval().to("cuda")
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    ctx = {"tok": tok}
    ccls = cache_cls(kind)
    import dream_common as D
    mask_id, eos_id = D.MASK, D.EOS
    prompt_of = lambda e: D.prompt_ids(tok, e)
    text_of, eos_of = (lambda toks: D.response_text(tok, toks)), D.first_eos
    native = arm == "ORIG_NATIVE"

    def run(e):
        p = prompt_of(e).reshape(1, -1).to("cuda")
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            rec = generate(model, p, strategy="vanilla", attention_mask=torch.ones_like(p), max_new_tokens=gen, block_length=(gen if native else 32), num_transfer_tokens=1, threshold=(None if native else 0.9), alg=("entropy" if native else "maskgit_plus"),
                           temperature=0.0, sigma=(float(os.environ["D2C_SIGMA"]) if os.environ.get("D2C_SIGMA") else None), mask_token_id=mask_id, eos_token_id=eos_id, stop_until_eos=False, cache_cls=ccls, ignore_unknown_args="warn")
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        toks = [int(v) for v in rec[-1].generated_tokens[0].tolist()]
        return {"tokens": toks, "latency_s": dt, "nfe": int(len(rec)), "text": text_of(toks), "prompt_len": int(p.shape[1])}

    path = OUT / f"{'smoke_' if smoke else ''}{'flops_' if FL else ''}d2c_{TAG}_{ds}_{arm}.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    done = {r["example_id"] for r in C.rows(path) if r.get("status") == "ok"}
    run(es_run[0])                                              # warm-up
    started = time.time()
    for i, e in enumerate(es_run):
        if e["example_id"] in done:
            continue
        C.gpu_guard()
        torch.cuda.reset_peak_memory_stats()
        fl_ = {}
        with count(fl_) if FL else contextlib.nullcontext():
            d = run(e)
        d = dict(d, **fl_)
        d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        correct, status = (None, "deferred") if (DEFER and ds in CODE) else C.score_paper_generation(d["text"], e)
        if status == "sandbox_error":
            raise SystemExit("sandbox_error while scoring: abort")
        C.append(path, dict(d, example_id=e["example_id"], arm=arm, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status,
                            first_eos=eos_of(d["tokens"]), status="ok", loadavg1=os.getloadavg()[0]))
        print(f"[d2c {arm} {ds}] {i + 1}/{len(es_run)} {e['example_id']} {d['latency_s']:.1f}s calls {d['nfe']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[d2c {arm} {ds}] done", flush=True)


if __name__ == "__main__":
    main()
