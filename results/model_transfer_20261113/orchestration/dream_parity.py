"""Dream-v0-Instruct-7B parity and smoke on the real weights (stage M2 prerequisites; PREREGISTERED.md prediction 7). MODEL_TAG=dream. Output: dream/records/parity.json + a printed report.
A. adapter against the model's own forward (Dream's remote code, if it loads under this transformers): logits on a fixed HumanEval prompt + masks, before the shift; max abs/relative difference, argmax agreement.
   Done FIRST and the HF model is freed before the adapter weights are loaded (both would not fit on the 32 GB card).
B. engine against a plain dense loop with the shift (C-0.9 block word lock) on 3 HumanEval prompts at 256 tokens: refresh='every_call' decode token-identical.
C. cached decodes (S32, P4-style partial refresh from layer 9) run, with calls, dense calls and wall time.
D. the layer-cost benchmark cells that analysis needs: dense call, query-only calls (as bench_layer_cost.py), on Dream's shapes."""
from __future__ import annotations

import json
import os
import statistics as st
import time

os.environ.setdefault("MODEL_TAG", "dream")
import common as C  # noqa: E402
import torch  # noqa: E402
from regret_remasking.model_adapters import load_dream  # noqa: E402
from regret_remasking.selective_cache import SelectiveConfig, SelectiveEngine, _confident_write, decode_selective  # noqa: E402

PATH = os.environ.get("MODEL_CHECKPOINT", "/l/users/rui.gao/models/Dream-v0-Instruct-7B")
MASK, EOS = 151666, 151643
OUT = {}


def save():
    (C.OUT / "records").mkdir(parents=True, exist_ok=True)
    (C.OUT / "records/parity.json").write_text(json.dumps(OUT, indent=1))
    print(json.dumps({k: v for k, v in OUT.items() if k != "C"}, indent=1)[-1500:], flush=True)


def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(PATH, trust_remote_code=True)


def prompt_ids(tok, e):
    text = tok.apply_chat_template([{"role": "user", "content": C._resolve_prompt(e)}], add_generation_prompt=True, tokenize=False)
    return tok(text, return_tensors="pt")["input_ids"].to("cuda")


def part_a(tok, es):
    ids = prompt_ids(tok, es[0])
    x = torch.cat([ids, torch.full((1, 96), MASK, device="cuda")], 1)
    try:
        from transformers import AutoModel
        hf = AutoModel.from_pretrained(PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
        with torch.no_grad():
            logits = hf(x, "full").logits if True else None
        OUT["A"] = {"loaded": True}
        ref = logits.float().cpu()
        del hf, logits
    except Exception as err:                      # the remote code targets transformers 4.46; 5.x may not run it
        OUT["A"] = {"loaded": False, "error": f"{type(err).__name__}: {str(err)[:300]}"}
        ref = None
    torch.cuda.empty_cache()
    return x, ref


def dense_logits(model, x):
    """Plain dense forward through the adapter functions (all rows, full-length attention, no engine); logits before the shift."""
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
        lg = dense_logits(model, x)
        lg = torch.cat([lg[:, :1], lg[:, :-1]], 1)[:, win]
        pos, tok, _ = _confident_write(lg, win, x, thr, EOS)
        masked = masked[~torch.isin(masked, pos)]
        calls += 1
    return x[0, P:].tolist(), calls


def main():
    C.gpu_guard()
    es = C.he164()
    tok = tokenizer()
    x, ref = part_a(tok, es)
    model = load_dream(PATH, "cuda", torch.bfloat16)
    ad = model.cache_adapter
    OUT["config"] = {"layers": ad.n_layers, "heads": ad.n_heads, "kv_heads": ad.n_kv_heads, "head_dim": ad.head_dim, "device": str(ad.device)}
    save()
    mine = dense_logits(model, x).float().cpu()
    if ref is not None:
        d = (mine - ref).abs()
        OUT["A"].update({"max_abs": float(d.max()), "mean_abs": float(d.mean()), "ref_abs_mean": float(ref.abs().mean()), "argmax_agree": float((mine.argmax(-1) == ref.argmax(-1)).float().mean())})
    # engine over all rows against the adapter's dense forward (shifted)
    cfg = SelectiveConfig(gen_length=96, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="every_call", shape="block", block_length=32, lock="conf", threshold=0.9, attn_impl="sdpa")
    eng = SelectiveEngine(model, x.shape[1], cfg)
    P = x.shape[1] - 96
    win = torch.arange(P, x.shape[1], device="cuda")
    lg, _ = eng.forward(x, torch.arange(x.shape[1], device="cuda"), 0, win - 1, win - 1, torch.zeros(x.shape[1], dtype=torch.bool, device="cuda"), fresh=True)
    want = torch.cat([mine[:, :1], mine[:, :-1]], 1)[:, win.cpu()].to("cuda")
    OUT["B_forward"] = {"max_abs": float((lg.float() - want).abs().max()), "argmax_agree": float((lg.argmax(-1) == want.argmax(-1)).float().mean())}
    save()
    # token-identical decodes
    same = []
    for e in es[:3]:
        ids = prompt_ids(tok, e)
        ref_tokens, calls = plain_decode(model, ids, 256, 32, 0.9)
        cfg = SelectiveConfig(gen_length=256, mask_id=MASK, eos_id=EOS, eos_truncation=False, refresh="every_call", shape="block", block_length=32, lock="conf", threshold=0.9, attn_impl="sdpa", log_calls=True, log_detail=False)
        res = decode_selective(model, ids, cfg)
        same.append({"id": e["example_id"], "identical": res.tokens == ref_tokens, "calls_plain": calls, "calls_engine": res.nfe,
                     "first_diff": next((i for i, (a, b) in enumerate(zip(res.tokens, ref_tokens)) if a != b), None), "text_head": tok.decode(res.tokens[:60])})
    OUT["B_decode"] = same
    save()
    # cached decodes
    runs = []
    for name, kw in (("dense", dict(refresh="every_call")), ("S32", dict(refresh="block")), ("P_layer9_k3", dict(refresh="block", partial_every_k=3, partial_alpha=9 / 28, partial_join="blank"))):
        for e in es[:3]:
            ids = prompt_ids(tok, e)
            cfg = SelectiveConfig(gen_length=256, mask_id=MASK, eos_id=EOS, eos_truncation=True, shape="block", block_length=32, lock="conf", threshold=0.9, attn_impl="sdpa", log_calls=True, log_detail=False, **kw)
            res = decode_selective(model, ids, cfg)
            runs.append({"arm": name, "id": e["example_id"], "nfe": res.nfe, "dense_calls": sum(k in ("first", "dense") for k in res.refresh_kind), "latency_s": res.latency_s})
    OUT["C"] = runs
    save()
    # layer-cost cells on Dream's shapes
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
    cells = {"dense": {"wall_med_ms": timed(lambda: eb.forward(ids_b, allr, 0, allr[Pq:] - 1, allr[Pq:] - 1, blank, fresh=True))}}      # all rows computed: local index = position
    for Q in (16, 32, 64, 132, 264):
        rows = allr[Pq:Pq + Q]
        loc = torch.arange(max(1, Q // 2), device="cuda")                    # dec_local indexes INTO rows (local indices), not positions
        cells[f"query_only_Q{Q}"] = {"wall_med_ms": timed(lambda: eb.forward(ids_b, rows, 0, loc, loc, blank))}
    OUT["bench"] = {"T": T, "P": Pq, "gpu": torch.cuda.get_device_name(0), "cells": cells}
    (C.OUT / "records").mkdir(parents=True, exist_ok=True)
    (C.OUT / "records/parity.json").write_text(json.dumps(OUT, indent=1))
    (C.OUT / "records/bench_layer_cost.json").write_text(json.dumps(OUT["bench"], indent=1))
    print(json.dumps(OUT, indent=1))


if __name__ == "__main__":
    main()
