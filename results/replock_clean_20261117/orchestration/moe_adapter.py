"""LLaDA-MoE-7B-A1B-Instruct (inclusionAI) for the selective-cache engine, from raw safetensors (same role as DreamAdapter in src/regret_remasking/model_adapters.py; a new file so no shared module changes).
Architecture (modeling_lladamoe.py): 16 layers, hidden 2048, 16 heads (no GQA), pre-norm RMSNorm, per-head RMSNorm on q and k (q_norm / k_norm over head_dim, before RoPE), llama-style RoPE (theta 50000, full rotary),
bidirectional attention, MoE feed-forward in every layer: softmax router over 64 experts, top-8, weights NOT renormalised, SwiGLU experts (intermediate 1024), no shared expert; logits are NOT shifted (LLaDA-style).
The expert computation groups the (row, expert) pairs by expert and runs three batched matmuls over padded per-expert buckets (instead of a Python loop over 64 experts)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    h = x.shape[-1] // 2
    return torch.cat((-x[..., h:], x[..., :h]), dim=-1)


class LLaDAMoEAdapter:
    logit_shift = 0

    def __init__(self, weights: dict[str, torch.Tensor], cfg: dict[str, Any], device: torch.device | str = "cpu") -> None:
        self.W, self.cfg = weights, cfg
        self.n_layers, self.n_heads, self.n_kv_heads = int(cfg["num_hidden_layers"]), int(cfg["num_attention_heads"]), int(cfg["num_key_value_heads"])
        self.d_model = int(cfg["hidden_size"])
        self.head_dim = self.d_model // self.n_heads
        self.eps, theta = float(cfg.get("rms_norm_eps", 1e-5)), float(cfg.get("rope_theta", 50000))
        self.n_experts, self.top_k = int(cfg["num_experts"]), int(cfg["num_experts_per_tok"])
        self.device = torch.device(device)
        self.inv_freq = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=self.device) / self.head_dim))
        self.blocks = None
        self.experts: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []
        for l in range(self.n_layers):                           # stack the 64 experts of a layer into [E, D, I], [E, D, I], [E, I, D] and drop the per-expert tensors
            g = torch.stack([weights.pop(f"model.layers.{l}.mlp.experts.{e}.gate_proj.weight") for e in range(self.n_experts)]).transpose(1, 2).contiguous()
            u = torch.stack([weights.pop(f"model.layers.{l}.mlp.experts.{e}.up_proj.weight") for e in range(self.n_experts)]).transpose(1, 2).contiguous()
            d = torch.stack([weights.pop(f"model.layers.{l}.mlp.experts.{e}.down_proj.weight") for e in range(self.n_experts)]).transpose(1, 2).contiguous()
            self.experts.append((g, u, d))

    def _w(self, name: str) -> torch.Tensor:
        return self.W[name]

    def _rms(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        dt = x.dtype
        xf = x.to(torch.float32)
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return w * xf.to(dt)

    def embed(self, ids):
        return F.embedding(ids, self._w("model.embed_tokens.weight"))

    def attn_norm(self, l, x):
        return self._rms(x, self._w(f"model.layers.{l}.input_layernorm.weight"))

    def qkv(self, l, xn, positions, total_len):
        p, n = f"model.layers.{l}.self_attn.", xn.shape[1]
        q = F.linear(xn, self._w(p + "q_proj.weight")).view(1, n, self.n_heads, self.head_dim)
        k = F.linear(xn, self._w(p + "k_proj.weight")).view(1, n, self.n_kv_heads, self.head_dim)
        v = F.linear(xn, self._w(p + "v_proj.weight")).view(1, n, self.n_kv_heads, self.head_dim).transpose(1, 2)
        q = self._rms(q, self._w(p + "q_norm.weight")).transpose(1, 2)
        k = self._rms(k, self._w(p + "k_norm.weight")).transpose(1, 2)
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        cos, sin = emb.cos().to(q.dtype)[None, None], emb.sin().to(q.dtype)[None, None]
        return q * cos + _rotate_half(q) * sin, k * cos + _rotate_half(k) * sin, v

    def attend(self, l, q, k_full, v_full):
        return F.scaled_dot_product_attention(q, k_full, v_full, attn_mask=None, is_causal=False)

    def attn_out(self, l, att):
        return F.linear(att, self._w(f"model.layers.{l}.self_attn.o_proj.weight"))

    def _moe(self, l, h):                                        # h: [n, D]
        n, D = h.shape
        E, K = self.n_experts, self.top_k
        logits = F.linear(h, self._w(f"model.layers.{l}.mlp.gate.weight"))
        w = F.softmax(logits, dim=1, dtype=torch.float32)
        topw, topi = torch.topk(w, K, dim=-1)
        topw = topw.to(h.dtype)
        flat_e = topi.reshape(-1)
        order = torch.argsort(flat_e, stable=True)
        se = flat_e[order]
        counts = torch.bincount(flat_e, minlength=E)
        starts = torch.cumsum(counts, 0) - counts
        pos = torch.arange(n * K, device=h.device) - starts[se]
        tok = order // K
        m = int(counts.max())
        buf = torch.zeros(E, m, D, dtype=h.dtype, device=h.device)
        buf[se, pos] = h[tok]
        g, u, d = self.experts[l]
        a = F.silu(torch.bmm(buf, g)) * torch.bmm(buf, u)
        o = torch.bmm(a, d)[se, pos] * topw.reshape(-1)[order][:, None]          # contribution of each (row, expert) pair, in expert-sorted order
        contrib = torch.empty(n * K, D, dtype=h.dtype, device=h.device)
        contrib[order] = o                                                       # back to [row, top-k slot] order
        contrib = contrib.view(n, K, D)
        sj = torch.sort(topi, dim=1).indices                                     # per row, add the experts in ascending expert id, sequentially in the model dtype (the reference's index_add_ order)
        contrib = contrib.gather(1, sj[..., None].expand(n, K, D))
        y = contrib[:, 0]
        for j in range(1, K):
            y = y + contrib[:, j]
        return y

    def ffn(self, l, x):
        h = self._rms(x, self._w(f"model.layers.{l}.post_attention_layernorm.weight"))
        if x.shape[1] == 0:
            return x
        return x + self._moe(l, h[0]).unsqueeze(0)

    def final(self, x):
        return F.linear(self._rms(x, self._w("model.norm.weight")), self._w("lm_head.weight"))


def load_llada_moe(path: str, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from safetensors.torch import load_file

    from regret_remasking.model_adapters import AdapterModel
    root = Path(path)
    cfg = json.loads((root / "config.json").read_text())
    index = json.loads((root / "model.safetensors.index.json").read_text())["weight_map"]
    weights: dict[str, torch.Tensor] = {}
    for shard in sorted(set(index.values())):
        for k, v in load_file(str(root / shard), device=str(device)).items():
            weights[k] = v.to(dtype)
    return AdapterModel(LLaDAMoEAdapter(weights, cfg, device))
