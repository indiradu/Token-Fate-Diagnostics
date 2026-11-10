"""Part A driver (GPU): dense Conf-0.9 decodes of the 40 TRAIN questions with the drift/signal capture (partA_capture.py).

Usage: <abs venv python> partA_collect.py GEN [LIMIT]       writes partA/g{GEN}/{example_id}.npz + partA/g{GEN}/summary.jsonl (resumable)
       <abs venv python> partA_collect.py noise              noise floor (identical repeat forward; row-subset recompute vs the full-batch rows)
"""
from __future__ import annotations

import json
import os
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
sys.path.insert(0, str(OUT / "source/src"))   # frozen copy first (freeze_source.py)
sys.path.insert(0, str(HERE))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from partA_capture import Capture  # noqa: E402
from regret_remasking.llada_trace import EOS_ID, EOT_ID, MASK_ID  # noqa: E402
from regret_remasking.locking_experiment import load_local_llada  # noqa: E402
from regret_remasking.locking_policy import _prepare_prompt  # noqa: E402
from regret_remasking.locking_reference import _apply_rope_at_positions, transformer_blocks  # noqa: E402
from regret_remasking.paper_decode import PaperDecodeConfig, decode_paper_policy  # noqa: E402
from regret_remasking.paper_experiment import _resolve_prompt, score_paper_generation  # noqa: E402

CHECKPOINT = Path("/home/rui.gao/.cache/huggingface/hub/models--GSAI-ML--LLaDA-8B-Instruct/snapshots/08b83a6feb34df1a6011b80c3c00c7563e963b07")
SEED, BLOCK, THR = 23, 32, 0.9
MAX_T = 1090


def rows(p):
    p = Path(p)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def load():
    model, tok = load_local_llada(CHECKPOINT, "cuda", "bf16")
    model.eval()
    return model, tok


@torch.no_grad()
def noise_floor(model, tok, entries):
    """(1) identical repeat of a dense forward; (2) K/V of a random row subset recomputed from the same layer input, as a compact (cached-row) pass would."""
    blocks = transformer_blocks(model)
    L = len(blocks)
    e = entries[0]
    ids, _ = _prepare_prompt(tok, _resolve_prompt(e), "cuda")
    P = ids.shape[1]
    x = torch.full((1, P + 256), MASK_ID, dtype=torch.long, device="cuda")
    x[:, :P] = ids
    # a mid-decode state: write every third answer slot with the model's own proposals so that K/V are not all-blank
    with torch.no_grad():
        lg = model(x).logits
        prop = lg[0].argmax(-1)
        sl = torch.arange(P, P + 256, 3, device="cuda")
        x[0, sl] = prop[sl]
    store, xin = {}, {}
    handles = []
    for l, b in enumerate(blocks):
        orig = b._scaled_dot_product_attention

        def wrapped(q, k, v, *a, _l=l, _o=orig, **kw):
            store.setdefault(_l, []).append((k.detach().clone(), v.detach().clone()))
            return _o(q, k, v, *a, **kw)
        b._scaled_dot_product_attention = wrapped
        handles.append((b, orig))
        b.register_forward_pre_hook(lambda m, args, kwargs, _l=l: xin.__setitem__(_l, args[0].detach().clone()), with_kwargs=True)
    with torch.no_grad():
        model(x)
        model(x)
    res = {"identical_repeat": {}, "row_subset_recompute": {}}
    rel = lambda a, b: ((a.float() - b.float()).flatten(2).norm(dim=-1) / b.float().flatten(2).norm(dim=-1)) if a.dim() == 3 else None
    g = torch.Generator().manual_seed(1)
    sub = torch.randperm(P + 256, generator=g)[:64].sort().values.cuda()
    worst_rep, subset_all = 0.0, []
    for l in range(L):
        (k1, v1), (k2, v2) = store[l][0], store[l][1]
        d = max(float((k1 - k2).abs().max()), float((v1 - v2).abs().max()))
        worst_rep = max(worst_rep, d)
        b = blocks[l]
        xn = b.attn_norm(xin[l][:, sub])
        ks = b.k_proj(xn)
        vs = b.v_proj(xn)
        nh = b.config.n_heads
        hd = ks.shape[-1] // nh
        ks = ks.view(1, -1, nh, hd).transpose(1, 2)
        vs = vs.view(1, -1, nh, hd).transpose(1, 2)
        ks = _apply_rope_at_positions(b.rotary_emb, ks, sub, P + 256)
        kf, vf = k1[:, :, sub], v1[:, :, sub]
        num = ((ks.float() - kf.float()) ** 2).sum((1, 3)) + ((vs.float() - vf.float()) ** 2).sum((1, 3))
        den = (kf.float() ** 2).sum((1, 3)) + (vf.float() ** 2).sum((1, 3))
        subset_all.append((num / den).sqrt()[0].cpu().numpy())
    sa = np.stack(subset_all)    # (L, 64)
    res["identical_repeat"] = {"max_abs_difference_over_all_layers": worst_rep}
    res["row_subset_recompute"] = {"mean_relL2": float(sa.mean()), "p99_relL2": float(np.quantile(sa, 0.99)), "max_relL2": float(sa.max()),
                                   "per_layer_mean": [float(v) for v in sa.mean(1)]}
    res["bf16_eps"] = 2.0 ** -8
    res["example_id"], res["prompt_len"] = e["example_id"], P
    for b, orig in handles:
        b._scaled_dot_product_attention = orig
    return res


def main():
    mode = sys.argv[1]
    entries = rows(OUT / "manifests/math40_train.jsonl")
    assert len(entries) == 40
    model, tok = load()
    if mode == "noise":
        res = noise_floor(model, tok, entries)
        (OUT / "validation/noise_floor.json").write_text(json.dumps(res, indent=1) + "\n")
        print(json.dumps({k: v for k, v in res.items() if k != "row_subset_recompute"}), json.dumps({k: v for k, v in res["row_subset_recompute"].items() if k != "per_layer_mean"}))
        return
    gen = int(sys.argv[1])
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else len(entries)
    out_dir = OUT / f"partA/g{gen}"
    out_dir.mkdir(parents=True, exist_ok=True)
    done = {r["example_id"] for r in rows(out_dir / "summary.jsonl")}
    blocks = transformer_blocks(model)
    started = time.time()
    for i, e in enumerate(entries[:limit]):
        if e["example_id"] in done:
            continue
        ids, _ = _prepare_prompt(tok, _resolve_prompt(e), "cuda")
        P = int(ids.shape[1])
        if P + gen > MAX_T:      # the K/V/Q rings (13.4 MB per row) must fit beside the 15.3 GiB model: T = 1117 peaked at 29.5 GiB
            with (out_dir / "skipped.jsonl").open("a") as h:
                h.write(json.dumps({"example_id": e["example_id"], "reason": f"prompt {P} + gen {gen} > {MAX_T} rows (capture memory)"}) + "\n")
            print(f"[A g{gen}] skip {e['example_id']} (T={P + gen} > {MAX_T})", flush=True)
            continue
        cap = Capture(blocks, P, P + gen, MASK_ID, (EOS_ID, EOT_ID), block_length=BLOCK)
        cap.install(model)
        cfg = PaperDecodeConfig(arm="C-block", threshold=THR, gen_length=gen, block_length=BLOCK, mask_id=MASK_ID)
        torch.manual_seed(SEED)
        t0 = time.time()
        try:
            r = decode_paper_policy(model, tok, _resolve_prompt(e), e["example_id"], cfg)
        except torch.cuda.OutOfMemoryError:
            cap.remove()
            del cap
            torch.cuda.empty_cache()
            with (out_dir / "skipped.jsonl").open("a") as h:
                h.write(json.dumps({"example_id": e["example_id"], "reason": "CUDA OOM during capture"}) + "\n")
            print(f"[A g{gen}] OOM skip {e['example_id']}", flush=True)
            continue
        cap.remove()
        packed = cap.pack()
        np.savez_compressed(out_dir / f"{e['example_id']}.npz", **packed)
        correct, status = score_paper_generation(r.text, e)
        rec = {"example_id": e["example_id"], "stratum": e["metadata"]["stratum"], "gen": gen, "prompt_len": P, "nfe": r.nfe, "calls_captured": len(cap.calls),
               "correct": bool(correct), "status": status, "wall_s": time.time() - t0, "decode_latency_s": r.latency_s,
               "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30, "n_samples": int(len(packed["samp_call"]))}
        with (out_dir / "summary.jsonl").open("a") as h:
            h.write(json.dumps(rec) + "\n")
            h.flush()
            os.fsync(h.fileno())
        del cap, packed
        torch.cuda.empty_cache()
        print(f"[A g{gen}] {i + 1}/{limit} {e['example_id']} P={P} nfe={r.nfe} wall={rec['wall_s']:.0f}s peak={rec['peak_mem_gib']:.1f}GiB ({(time.time() - started) / 60:.1f} min)", flush=True)
    nf = OUT / "validation/noise_floor.json"
    if not nf.exists():            # the standalone noise step of the chain crashed once (autograd); run it here with the model already loaded
        res = noise_floor(model, tok, entries)
        nf.write_text(json.dumps(res, indent=1) + "\n")
        print("[noise]", json.dumps(res["identical_repeat"]), json.dumps({k: v for k, v in res["row_subset_recompute"].items() if k != "per_layer_mean"}), flush=True)


if __name__ == "__main__":
    main()
