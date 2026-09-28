"""Multi-layer drift and sampled attention-influence signals for PhaseLock.

This module is deliberately limited to Phase 2.  It observes a dense replay of
an already-fixed semantic plan and never commits tokens or removes compute.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def multilayer_max_drift(
    current: dict[int, torch.Tensor],
    previous: dict[int, torch.Tensor],
) -> tuple[torch.Tensor, dict[int, torch.Tensor]]:
    """Return the maximum cosine drift across identically shaped layers."""
    if not current:
        raise ValueError("at least one current representation is required")
    if set(current) != set(previous):
        raise ValueError("current and previous layers must match")
    per_layer: dict[int, torch.Tensor] = {}
    for layer in sorted(current):
        if current[layer].shape != previous[layer].shape:
            raise ValueError(f"shape mismatch at layer {layer}")
        left = current[layer].float()
        right = previous[layer].float()
        similarity = torch.nn.functional.cosine_similarity(left, right, dim=-1)
        per_layer[layer] = (1.0 - similarity).clamp_min(0.0)
    maximum = torch.stack([per_layer[layer] for layer in sorted(per_layer)]).amax(0)
    return maximum, per_layer


def update_joint_patience(
    count: torch.Tensor,
    candidate: torch.Tensor,
    max_drift: torch.Tensor,
    influence: torch.Tensor,
    *,
    drift_threshold: float,
    influence_threshold: float,
) -> torch.Tensor:
    """Update consecutive observations satisfying both independent gates."""
    if not (
        count.shape == candidate.shape == max_drift.shape == influence.shape
    ):
        raise ValueError("joint-gate tensors must have identical shapes")
    if drift_threshold < 0.0 or influence_threshold < 0.0:
        raise ValueError("joint-gate thresholds must be non-negative")
    below = (
        candidate
        & (max_drift <= drift_threshold)
        & (influence <= influence_threshold)
    )
    updated = count.clone()
    updated[below] += 1
    updated[candidate & ~below] = 0
    return updated


def multilayer_influence_ready(
    candidate: torch.Tensor,
    max_drift: torch.Tensor,
    influence: torch.Tensor,
    age: torch.Tensor,
    joint_count: torch.Tensor,
    *,
    drift_threshold: float,
    influence_threshold: float,
    patience: int,
    min_age: int,
) -> torch.Tensor:
    """Return candidates admitted by drift, influence, age, and patience."""
    if patience < 1 or min_age < 0:
        raise ValueError("patience must be positive and min_age non-negative")
    if not (
        candidate.shape
        == max_drift.shape
        == influence.shape
        == age.shape
        == joint_count.shape
    ):
        raise ValueError("multi-layer gate tensors must have identical shapes")
    return (
        candidate
        & (max_drift <= drift_threshold)
        & (influence <= influence_threshold)
        & (joint_count >= patience)
        & (age >= min_age)
    )


class SampledAttentionInfluenceCapture:
    """Capture normalized attention received from unresolved query samples.

    The wrapped LLaDA attention receives already-projected Q/K/V tensors.  At
    most ``max_queries`` unresolved query rows are selected deterministically
    per example.  The captured score for key row ``i`` is

        key_length * mean_{sampled queries, heads} attention(query, i),

    so a score of one is uniform-attention mass.  The original attention
    implementation still produces the model output.
    """

    def __init__(self, block: torch.nn.Module, *, max_queries: int = 8) -> None:
        if max_queries < 1:
            raise ValueError("max_queries must be positive")
        self.block = block
        self.max_queries = max_queries
        self.original_attention: Any | None = None
        self.query_mask: torch.Tensor | None = None
        self.latest: torch.Tensor | None = None

    def set_query_mask(self, query_mask: torch.Tensor) -> None:
        if query_mask.ndim != 2 or query_mask.dtype != torch.bool:
            raise ValueError("query mask must be a boolean [batch, rows] tensor")
        self.query_mask = query_mask
        self.latest = None

    @staticmethod
    def _sample(indices: torch.Tensor, maximum: int) -> torch.Tensor:
        if indices.numel() <= maximum:
            return indices
        offsets = torch.linspace(
            0,
            indices.numel() - 1,
            steps=maximum,
            device=indices.device,
        ).round().to(dtype=torch.long)
        return indices.index_select(0, offsets)

    def install(self) -> None:
        if self.original_attention is not None:
            raise RuntimeError("attention influence capture is already installed")
        original = self.block.attention
        self.original_attention = original

        def wrapped_attention(
            q: torch.Tensor,
            k: torch.Tensor,
            v: torch.Tensor,
            attention_bias: torch.Tensor | None = None,
            layer_past: Any | None = None,
            use_cache: bool = False,
        ):
            self._capture(q, k, attention_bias, layer_past)
            return original(
                q,
                k,
                v,
                attention_bias=attention_bias,
                layer_past=layer_past,
                use_cache=use_cache,
            )

        self.block.attention = wrapped_attention

    def remove(self) -> None:
        if self.original_attention is not None:
            self.block.attention = self.original_attention
            self.original_attention = None
        self.query_mask = None
        self.latest = None

    @torch.no_grad()
    def _capture(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        attention_bias: torch.Tensor | None,
        layer_past: Any | None,
    ) -> None:
        if self.query_mask is None:
            raise RuntimeError("set_query_mask must be called before model forward")
        batch_size, query_len, channels = q.shape
        if self.query_mask.shape != (batch_size, query_len):
            raise ValueError("query mask shape does not match attention input")
        dtype = k.dtype
        q_work = q
        k_work = k
        if self.block.q_norm is not None and self.block.k_norm is not None:
            q_work = self.block.q_norm(q_work).to(dtype=dtype)
            k_work = self.block.k_norm(k_work).to(dtype=dtype)
        heads = int(self.block.config.n_heads)
        kv_heads = int(self.block.config.effective_n_kv_heads)
        head_dim = channels // heads
        q_heads = q_work.view(batch_size, query_len, heads, head_dim).transpose(1, 2)
        k_heads = k_work.view(batch_size, query_len, kv_heads, head_dim).transpose(1, 2)
        if layer_past is not None:
            past_key, _past_value = layer_past
            k_heads = torch.cat((past_key, k_heads), dim=-2)
        key_len = k_heads.shape[-2]
        if self.block.config.rope:
            q_heads, k_heads = self.block.rotary_emb(q_heads, k_heads)
        if heads != kv_heads:
            if heads % kv_heads:
                raise ValueError("query heads must be divisible by K/V heads")
            k_heads = k_heads.repeat_interleave(heads // kv_heads, dim=1)

        cast_bias = None
        if attention_bias is not None:
            cast_bias = self.block._cast_attn_bias(
                attention_bias[:, :, key_len - query_len : key_len, :key_len],
                dtype,
            ).float()

        received = []
        for batch_index in range(batch_size):
            indices = torch.nonzero(
                self.query_mask[batch_index], as_tuple=False
            ).flatten()
            indices = self._sample(indices, self.max_queries)
            if indices.numel() == 0:
                received.append(
                    torch.full(
                        (key_len,),
                        float("inf"),
                        dtype=torch.float32,
                        device=q.device,
                    )
                )
                continue
            sampled_q = q_heads[batch_index].index_select(1, indices).float()
            scores = torch.matmul(sampled_q, k_heads[batch_index].float().transpose(-2, -1))
            scores = scores / math.sqrt(float(head_dim))
            if cast_bias is not None:
                bias_batch = 0 if cast_bias.shape[0] == 1 else batch_index
                bias = cast_bias[bias_batch]
                if bias.shape[-2] == 1:
                    bias = bias.expand(bias.shape[0], query_len, key_len)
                bias = bias.index_select(-2, indices)
                scores = scores + bias
            weights = torch.softmax(scores, dim=-1)
            received.append(weights.mean(dim=(0, 1)) * float(key_len))
        self.latest = torch.stack(received, dim=0)
