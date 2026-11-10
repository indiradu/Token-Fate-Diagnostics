"""Tiny random LLaDA-shaped model for CPU tests of cache code (no downloads).

Same module layout as the real HF LLaDA (model.model.transformer.{wte, blocks, ln_f, ff_out}; each block has attn_norm, q/k/v_proj, attn_out, ff_norm,
ff_proj, up_proj, act, ff_out, rotary_emb, config.n_heads/effective_n_kv_heads, _scaled_dot_product_attention) so that code written against the real
blocks (PaperReferenceRuntime, selective_cache) runs on it unchanged.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import torch
import torch.nn.functional as F


class ToyRMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * x).to(dtype)


class ToyRotary:
    """Half-rotation RoPE with the same method surface as the LLaDA RotaryEmbedding."""

    def __init__(self, head_dim: int, theta: float = 10000.0) -> None:
        self.head_dim = head_dim
        self.theta = theta
        self.config = SimpleNamespace(rope_full_precision=True)

    def get_rotary_embedding(self, seq_len: int, device) -> tuple[torch.Tensor, torch.Tensor]:
        inv = 1.0 / (self.theta ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float) / self.head_dim))
        freqs = torch.einsum("i,j->ij", torch.arange(seq_len, device=device, dtype=torch.float), inv)
        pos = torch.cat((freqs, freqs), dim=-1)
        return pos.sin()[None, None], pos.cos()[None, None]

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        b, nh, t, hs = x.shape
        x = x.view(b, nh, t, 2, hs // 2)
        x1, x2 = x.unbind(dim=-2)
        return torch.cat((-x2, x1), dim=-1)

    def apply_rotary_pos_emb(self, pos_sin: torch.Tensor, pos_cos: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return ((t * pos_cos) + (self._rotate_half(t) * pos_sin)).to(t.dtype)


class ToyBlock(torch.nn.Module):
    def __init__(self, d: int, n_heads: int, ffn: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(n_heads=n_heads, effective_n_kv_heads=n_heads)
        self.attn_norm, self.ff_norm = ToyRMSNorm(d), ToyRMSNorm(d)
        self.q_proj = torch.nn.Linear(d, d, bias=False)
        self.k_proj = torch.nn.Linear(d, d, bias=False)
        self.v_proj = torch.nn.Linear(d, d, bias=False)
        self.attn_out = torch.nn.Linear(d, d, bias=False)
        self.ff_proj = torch.nn.Linear(d, ffn, bias=False)
        self.up_proj = torch.nn.Linear(d, ffn, bias=False)
        self.ff_out = torch.nn.Linear(ffn, d, bias=False)
        self.act = torch.nn.SiLU()
        self.dropout = torch.nn.Identity()
        self.q_norm = self.k_norm = None
        self.rotary_emb = ToyRotary(d // n_heads)

    def _scaled_dot_product_attention(self, q, k, v, attn_mask=None, dropout_p: float = 0.0, is_causal: bool = False):
        return F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask, dropout_p=dropout_p, is_causal=False)

    def forward(self, x, attention_bias=None, layer_past=None, use_cache: bool = False):
        b, t, d = x.shape
        h = self.config.n_heads
        xn = self.attn_norm(x)
        q = self.q_proj(xn).view(b, t, h, d // h).transpose(1, 2)
        k = self.k_proj(xn).view(b, t, h, d // h).transpose(1, 2)
        v = self.v_proj(xn).view(b, t, h, d // h).transpose(1, 2)
        sin, cos = self.rotary_emb.get_rotary_embedding(t, x.device)
        q = self.rotary_emb.apply_rotary_pos_emb(sin.to(q.dtype), cos.to(q.dtype), q)
        k = self.rotary_emb.apply_rotary_pos_emb(sin.to(k.dtype), cos.to(k.dtype), k)
        att = self._scaled_dot_product_attention(q, k, v, attn_mask=attention_bias)
        x = x + self.dropout(self.attn_out(att.transpose(1, 2).contiguous().view(b, t, d)))
        ff = self.ff_norm(x)
        ff = self.act(self.ff_proj(ff)) * self.up_proj(ff)
        return x + self.dropout(self.ff_out(ff)), None


class ToyLLaDA(torch.nn.Module):
    """forward(input_ids, attention_mask=None) -> object with .logits (bidirectional, no mask)."""

    def __init__(self, vocab: int = 64, d: int = 32, n_heads: int = 4, layers: int = 4, ffn: int = 64, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.model = torch.nn.Module()
        tr = torch.nn.Module()
        tr.wte = torch.nn.Embedding(vocab, d)
        tr.blocks = torch.nn.ModuleList([ToyBlock(d, n_heads, ffn) for _ in range(layers)])
        tr.ln_f = ToyRMSNorm(d)
        tr.ff_out = torch.nn.Linear(d, vocab, bias=False)
        self.model.transformer = tr
        self.vocab = vocab
        self.eval()

    def forward(self, input_ids, attention_mask=None, **_unused):
        tr = self.model.transformer
        x = tr.wte(input_ids)
        for block in tr.blocks:
            x, _ = block(x)
        return SimpleNamespace(logits=tr.ff_out(tr.ln_f(x)), hidden_states=None)
