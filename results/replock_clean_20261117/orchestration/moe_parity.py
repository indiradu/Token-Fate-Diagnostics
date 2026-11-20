"""LLaDA-MoE adapter checks on the real weights (Study 5 prerequisites), MODEL_TAG=dream manifests. Output: records/moe_parity.json and records/moe_bench_layer_cost.json (the cost cells R0's rule derivation needs).
A. adapter against the model's own forward (remote code): logits on a fixed HumanEval prompt + masks; max abs / mean abs / argmax agreement. HF model freed before the adapter weights are loaded.
B. engine (refresh='every_call') against a plain dense loop through the adapter (Conf-0.9 block word lock), 3 HumanEval prompts at 256 tokens: token-identical?
C. cached decodes (dense, S32, partial refresh from layer 5) run, with calls, dense calls and wall time.
D. layer-cost cells (dense call and query-only calls) on this model's shapes."""
from __future__ import annotations

import json
import statistics as st
import time
from pathlib import Path

import moe_common as D
import torch
from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, _confident_write, decode_selective

C = D.C
REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")
OUT = {}
MASK, EOS = D.MASK, D.EOS


def save():
    REC.mkdir(parents=True, exist_ok=True)
    (REC / "moe_parity.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps({k: v for k, v in OUT.items() if k != "C"}, indent=1)[-1800:], flush=True)


def dense_logits(model, x):
    ad = model.cache_adapter
    pos = torch.arange(x.shape[1], device=x.device)
    h = ad.embed(x)
    for l in range(ad.n_layers):
        q, k, v = ad.qkv(l, ad.attn_norm(l, h), pos, x.shape[1])
        att = ad.attend(l, q, k, v).transpose(1, 2).contiguous().view(1, x.shape[1], -1)
        h = ad.ffn(l, h + ad.attn_out(l, att))
    return ad.final(h)


def plain_decode(model, ids, G, block, thr):
    P = ids.shape[1]
    x = torch.cat([ids, torch.full((1, G), MASK, device="cuda")], 1)
    masked = torch.arange(P, P + G, device="cuda")
    calls = 0
    while masked.numel():
        blk = (int(masked[0]) - P) // block
        win = masked[((masked - P) // block) == blk]
        lg = dense_logits(model, x)[:, win]
        pos, tok, _ = _confident_write(lg, win, x, thr, EOS)
        masked = masked[~torch.isin(masked, pos)]
        calls += 1
    return x[0, P:].tolist(), calls


def main():
    C.gpu_guard()
    es = C.he164()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(D.PATH, trust_remote_code=True)
    ids = D.prompt_ids(tok, es[0])
    x = torch.cat([ids, torch.full((1, 96), MASK, device="cuda")], 1)
    ref = None
    try:
        hf = AutoModel.from_pretrained(D.PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            out = hf(x)
        ref = (out.logits if hasattr(out, "logits") else out[0]).float().cpu()
        OUT["A"] = {"loaded": True}
        del hf, out
    except Exception as err:
        OUT["A"] = {"loaded": False, "error": f"{type(err).__name__}: {str(err)[:400]}"}
    torch.cuda.empty_cache()
    model, _ = D.load()
    ad = model.cache_adapter
    OUT["config"] = {"layers": ad.n_layers, "heads": ad.n_heads, "kv_heads": ad.n_kv_heads, "head_dim": ad.head_dim, "experts": ad.n_experts, "top_k": ad.top_k, "device": str(ad.device)}
    save()
    with torch.no_grad():
        mine = dense_logits(model, x).float().cpu()
    if ref is not None:
        d = (mine - ref).abs()
        OUT["A"].update({"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_abs_mean": float(ref.abs().mean()), "argmax_agree": float((mine.argmax(-1) == ref.argmax(-1)).float().mean())})
    cfg = SelectiveConfig(gen_length=96, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="every_call", shape="block", block_length=32, lock="conf", threshold=0.9, attn_impl="sdpa")
    eng = SelectiveEngine(model, x.shape[1], cfg)
    P = x.shape[1] - 96
    win = torch.arange(P, x.shape[1], device="cuda")
    with torch.no_grad():
        lg, _ = eng.forward(x, torch.arange(x.shape[1], device="cuda"), 0, win, win, torch.zeros(x.shape[1], dtype=torch.bool, device="cuda"), fresh=True)
    want = mine[:, win.cpu()].to("cuda")
    OUT["B_forward"] = {"max_abs": float((lg.float() - want).abs().max()), "argmax_agree": float((lg.argmax(-1) == want.argmax(-1)).float().mean())}
    save()
    same = []
    for e in es[:3]:
        ids = D.prompt_ids(tok, e)
        with torch.no_grad():
            ref_tokens, calls = plain_decode(model, ids, 256, 32, 0.9)
            res = decode_selective(model, ids, D.cfg_for(256, refresh="every_call", eos_truncation=False))
        same.append({"id": e["example_id"], "identical": res.tokens == ref_tokens, "calls_plain": calls, "calls_engine": res.nfe,
                     "first_diff": next((i for i, (a, b) in enumerate(zip(res.tokens, ref_tokens)) if a != b), None), "text_head": tok.decode(res.tokens[:60])})
    OUT["B_decode"] = same
    save()
    runs = []
    for name, kw in (("dense", dict(refresh="every_call")), ("S32", dict(refresh="block")), ("P_layer5_k3", dict(refresh="block", partial_every_k=3, partial_alpha=5 / 16, partial_join="blank"))):
        for e in es[:3]:
            res = decode_selective(model, D.prompt_ids(tok, e), D.cfg_for(256, eos_truncation=True, **kw))
            runs.append({"arm": name, "id": e["example_id"], "nfe": res.nfe, "dense_calls": sum(k in ("first", "dense") for k in res.refresh_kind), "latency_s": res.latency_s})
    OUT["C"] = runs
    save()
    T, Pq = 690, 178
    ids_b = torch.randint(0, 150000, (1, T), device="cuda"); ids_b[:, Pq:] = MASK
    cfgb = SelectiveConfig(gen_length=T - Pq, mask_id=MASK, eos_id=EOS, refresh="block", shape="block", block_length=32, lock="conf", attn_impl="sdpa")
    eb = SelectiveEngine(model, T, cfgb)
    allr = torch.arange(T, device="cuda")
    blank = torch.zeros(T, dtype=torch.bool, device="cuda")

    def timed(fn, reps=15, warm=3):
        for _ in range(warm):
            fn()
        ts = []
        for _ in range(reps):
            torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e3)
        return st.median(ts)
    with torch.no_grad():
        cells = {"dense": {"wall_med_ms": timed(lambda: eb.forward(ids_b, allr, 0, allr[Pq:], allr[Pq:], blank, fresh=True))}}
        for Q in (16, 32, 64, 132, 264):
            rows = allr[Pq:Pq + Q]
            loc = torch.arange(max(1, Q // 2), device="cuda")
            cells[f"query_only_Q{Q}"] = {"wall_med_ms": timed(lambda: eb.forward(ids_b, rows, 0, loc, loc, blank))}
    OUT["bench"] = {"T": T, "P": Pq, "gpu": torch.cuda.get_device_name(0), "cells": cells}
    REC.mkdir(parents=True, exist_ok=True)
    (REC / "moe_parity.json").write_text(json.dumps(OUT, indent=1))
    (REC / "moe_bench_layer_cost.json").write_text(json.dumps(OUT["bench"], indent=1))
    print(json.dumps(OUT["bench"], indent=1))


if __name__ == "__main__":
    main()
