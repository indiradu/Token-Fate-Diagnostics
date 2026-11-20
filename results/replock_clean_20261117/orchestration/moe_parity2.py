"""LLaDA-MoE layer-by-layer parity against the model's own forward (hooks on each decoder layer): relative L2 error of the hidden state after every layer, plus the adapter's own run-to-run determinism and
sensitivity to the row subset. A formula bug shows as a large error at layer 0; rounding noise amplified by router ties grows slowly. Output: records/moe_parity2.json."""
from __future__ import annotations

import json
from pathlib import Path

import moe_common as D
import torch

C = D.C
REC = Path("/home/rui.gao/Desktop/Token-Fate-Diagnostics/results/replock_clean_20261117/records")


def adapter_layers(model, x, rows=None):
    ad = model.cache_adapter
    pos = torch.arange(x.shape[1], device=x.device)
    h = ad.embed(x)
    outs = []
    for l in range(ad.n_layers):
        q, k, v = ad.qkv(l, ad.attn_norm(l, h), pos, x.shape[1])
        att = ad.attend(l, q, k, v).transpose(1, 2).contiguous().view(1, x.shape[1], -1)
        h = ad.ffn(l, h + ad.attn_out(l, att))
        outs.append(h.float().cpu())
    return outs, ad.final(h).float().cpu()


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    C.gpu_guard()
    es = C.he164()
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(D.PATH, trust_remote_code=True)
    ids = D.prompt_ids(tok, es[0])
    x = torch.cat([ids, torch.full((1, 96), D.MASK, device="cuda")], 1)
    hf = AutoModel.from_pretrained(D.PATH, trust_remote_code=True, torch_dtype=torch.bfloat16).to("cuda").eval()
    cap = {}
    hooks = [lyr.register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o).float().cpu())) for l, lyr in enumerate(hf.model.layers)]
    with torch.no_grad():
        out = hf(x)
    ref_logits = (out.logits if hasattr(out, "logits") else out[0]).float().cpu()
    for h in hooks:
        h.remove()
    ref_layers = [cap[l] for l in range(len(cap))]
    del hf, out
    torch.cuda.empty_cache()
    model, _ = D.load()
    with torch.no_grad():
        a1, lg1 = adapter_layers(model, x)
        a2, lg2 = adapter_layers(model, x)
    res = {"layer_rel_err_vs_hf": [round(rel(a, b), 5) for a, b in zip(a1, ref_layers)],
           "final_logit_rel_err_vs_hf": rel(lg1, ref_logits), "argmax_agree_vs_hf": float((lg1.argmax(-1) == ref_logits.argmax(-1)).float().mean()),
           "adapter_run_to_run_max_abs": max(float((a - b).abs().max()) for a, b in zip(a1, a2)), "adapter_run_to_run_argmax_agree": float((lg1.argmax(-1) == lg2.argmax(-1)).float().mean())}
    # sensitivity to batch shape: run the same 96 mask rows plus a shorter prompt prefix of the same sequence (different n) and compare the shared rows' logits
    x2 = x[:, 10:]
    with torch.no_grad():
        _, lg3 = adapter_layers(model, x2)
    res["note"] = "layer errors are relative L2 of the hidden state after each decoder layer (bf16 both sides)"
    REC.mkdir(parents=True, exist_ok=True)
    (REC / "moe_parity2.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
