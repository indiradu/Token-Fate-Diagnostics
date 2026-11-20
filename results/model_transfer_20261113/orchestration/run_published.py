"""Stage B: the PUBLISHED cache rules, run with their own released code and their own schedules, on our prompts, our scorer and our GPU (LLaDA family, batch 1, bf16).
Usage: <abs venv python> run_published.py METHOD SET [SMOKE]; env PYTHONPATH must contain /l/users/rui.gao/envs/tf446b (transformers 4.46.2, what the released code targets).
METHOD (one per process: the three repos reuse module names):
  FD_DUAL, FD_PREFIX   Fast-dLLM (NVlabs) dual / prefix cache + parallel threshold decoding 0.9, steps = gen, block 32 (their published setting)
  FD_DUAL_NOTHR        Fast-dLLM dual cache with NO threshold (one token per step): the cache alone
  DLC                  dLLM-Cache (maomaocun) K_p 25, K_r 5, rho 0.25, steps = gen, block 32 (their HumanEval setting)
  DKV_DEC              dKV-Cache-Decode (horseee), cache reload every 8 steps (their LLaDA setting), steps = gen, block 32
  DKV_GREEDY           dKV-Cache-Greedy, window 4, reload every 2 steps
  ELASTIC              Elastic-Cache (Nguyen-Tri et al.) as reproduced in our engine (stage F): window 16, Conf-0.9 rule, attention-aware trigger gamma 0.9 (no PYTHONPATH override)
Records: records/published_<TAG>_<SET>_<METHOD>.jsonl (tokens, latency, nfe, text, correct (deferred for code), peak memory). Env: MODEL_TAG (manifests dir), MODEL_CHECKPOINT, ORIG_TAG, PUB_LIMIT.
Speed is wall-clock of the generate call only. The released generators decode all gen_length positions (no first-EOS stop), as published."""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import common as C
import torch
from run_rule import CODE, DEFER, SETS, questions

TOOLS = Path("/l/users/rui.gao/tools")
TAG = os.environ.get("ORIG_TAG", "llada8b")
MASK = 126336


def load(method):
    from transformers import AutoModel, AutoTokenizer
    ck = str(C.CHECKPOINT)
    tok = AutoTokenizer.from_pretrained(ck, trust_remote_code=True)
    if method.startswith("FD_"):
        sys.path.insert(0, str(TOOLS / "Fast-dLLM/v1/llada"))
        from model.modeling_llada import LLaDAModelLM
        model = LLaDAModelLM.from_pretrained(ck, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        import generate as G
        fn = {"FD_DUAL": lambda m, p, g: G.generate_with_dual_cache(m, p, steps=g, gen_length=g, block_length=32, temperature=0., remasking="low_confidence", mask_id=MASK, threshold=0.9),
              "FD_PREFIX": lambda m, p, g: G.generate_with_prefix_cache(m, p, steps=g, gen_length=g, block_length=32, temperature=0., remasking="low_confidence", mask_id=MASK, threshold=0.9),
              "FD_DUAL_NOTHR": lambda m, p, g: G.generate_with_dual_cache(m, p, steps=g, gen_length=g, block_length=32, temperature=0., remasking="low_confidence", mask_id=MASK)}[method]
        return model, tok, fn
    if method == "DLC":
        sys.path.insert(0, str(TOOLS / "dLLM-cache"))
        from dataclasses import asdict

        from dllm_cache.cache import dLLMCache, dLLMCacheConfig
        from dllm_cache.hooks import register_cache_LLaDA
        from utils import generate as dgen
        model = AutoModel.from_pretrained(ck, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        dLLMCache.new_instance(**asdict(dLLMCacheConfig(prompt_interval_steps=25, gen_interval_steps=5, transfer_ratio=0.25)))
        register_cache_LLaDA(model, "model.transformer.blocks")

        def fn(m, p, g):
            x = dgen(p, torch.ones_like(p), m, steps=g, gen_length=g, block_length=32, temperature=0., cfg_scale=0., remasking="low_confidence", mask_id=MASK)
            return torch.cat([p, x], dim=1), g                  # their generate returns only x[:, prompt_length:]
        return model, tok, fn
    if method in ("DKV_DEC", "DKV_GREEDY"):
        sys.path.insert(0, str(TOOLS / "dKV-Cache"))
        if method == "DKV_DEC":
            from generation_utils.llada_dkv_cache_decode import generate as dgen
            from models.modeling_llada_dkv_cache_decode import LLaDAModelLM
            kw = dict(cache_reloading_step=8)
        else:
            from generation_utils.llada_dkv_cache_greedy import generate as dgen
            from models.modeling_llada_dkv_cache_greedy import LLaDAModelLM
            kw = dict(cache_reloading_step=2, window_size=4)
        model = LLaDAModelLM.from_pretrained(ck, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()

        def fn(m, p, g):
            x = dgen(m, tok, p, steps=g, gen_length=g, block_length=32, temperature=0., cfg_scale=0., remasking="low_confidence", mask_id=MASK, enable_cache=True, **kw)
            return x, g
        return model, tok, fn
    if method == "ELASTIC":                                     # Elastic-Cache as reproduced in our engine (window 16, Conf-0.9 min(threshold, max) rule, attention-aware trigger gamma 0.9); run WITHOUT the tf446b path
        model, tok = C.load_model()

        def fn(m, p, g):
            cfg = C.SelectiveConfig(gen_length=g, window=16, far_mask_cache=True, mask_id=C.MASK_ID, eos_id=C.EOS_ID, attn_impl="sdpa", log_calls=True, log_detail=False, block_length=32,
                                    threshold=0.9, shape="window", eos_truncation=True, lock="elastic", refresh="trigger")
            res = C.decode_selective(m, p, cfg)
            return torch.cat([p, torch.tensor([list(res.tokens)], device=p.device, dtype=p.dtype)], dim=1), res.nfe
        return model, tok, fn
    raise SystemExit(f"unknown method {method}")


def main():
    method, ds = sys.argv[1], sys.argv[2]
    smoke = len(sys.argv) > 3 and sys.argv[3] == "SMOKE"
    gen = SETS[ds]
    es = questions(ds)
    if os.environ.get("PUB_LIMIT"):
        es = es[: int(os.environ["PUB_LIMIT"])]
    es_run = es[:2] if smoke else es
    C.gpu_guard()
    model, tok, fn = load(method)
    ctx = {"tok": tok}

    def run(e):
        p = C.prompt_ids(ctx, e).reshape(1, -1)
        torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.no_grad():
            x, nfe = fn(model, p, gen)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t
        toks = [int(v) for v in x[0, p.shape[1]:].tolist()]
        return {"tokens": toks, "latency_s": dt, "nfe": int(nfe), "text": C._decode_response(tok, toks), "prompt_len": int(p.shape[1])}

    path = C.OUT / f"records/{'smoke_' if smoke else ''}published_{TAG}_{ds}_{method}.jsonl"
    done = {r["example_id"] for r in C.rows(path) if r.get("status") == "ok"}
    run(es_run[0])                                              # warm-up
    started = time.time()
    for i, e in enumerate(es_run):
        if e["example_id"] in done:
            continue
        C.gpu_guard()
        torch.cuda.reset_peak_memory_stats()
        d = run(e)
        d["peak_mem_gib"] = torch.cuda.max_memory_allocated() / 2**30
        correct, status = (None, "deferred") if (DEFER and ds in CODE) else C.score_paper_generation(d["text"], e)
        if status == "sandbox_error":
            raise SystemExit("sandbox_error while scoring: abort")
        C.append(path, dict(d, example_id=e["example_id"], arm=method, set=ds, gen=gen, correct=(None if correct is None else bool(correct)), scoring_status=status,
                            first_eos=C.first_eos(d["tokens"]), status="ok", loadavg1=os.getloadavg()[0]))
        print(f"[published {method} {ds}] {i + 1}/{len(es_run)} {e['example_id']} {d['latency_s']:.1f}s nfe {d['nfe']} ({(time.time() - started) / 3600:.2f} h)", flush=True)
    print(f"[published {method} {ds}] done", flush=True)


if __name__ == "__main__":
    main()
