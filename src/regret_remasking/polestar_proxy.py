from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PolestarCacheProxyConfig:
    """Small Polestar-Cache-inspired single-refresh proxy configuration."""

    block_length: int = 32
    prefix_blocks: int = 2
    refresh_threshold: int = 3
    refresh_fraction: float = 0.5
    attention_layer: int = 32
    max_saved_events: int = 32
    seed: int = 23

    def __post_init__(self) -> None:
        if self.block_length <= 0:
            raise ValueError("block_length must be positive")
        if self.prefix_blocks < 0:
            raise ValueError("prefix_blocks must be non-negative")
        if self.refresh_threshold < 0:
            raise ValueError("refresh_threshold must be non-negative")
        if not 0.0 < self.refresh_fraction <= 1.0:
            raise ValueError("refresh_fraction must be in (0, 1]")
        if self.attention_layer <= 0:
            raise ValueError("attention_layer must be positive")
        if self.max_saved_events <= 0:
            raise ValueError("max_saved_events must be positive")


def attention_distribution_kl(
    current: torch.Tensor,
    previous: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Token-level KL(current || previous) for attention distributions.

    Inputs are probability tensors shaped [batch, query, key] or
    [batch, head, query, key]. Per-head KL values are averaged over heads. The
    returned tensor is shaped [batch, query].
    """
    if current.shape != previous.shape:
        raise ValueError("current and previous attention tensors must have matching shapes")
    if current.ndim not in {3, 4}:
        raise ValueError("attention tensors must be rank-3 or rank-4 probabilities")
    if eps <= 0.0:
        raise ValueError("eps must be positive")

    current_f = current.float().clamp_min(0.0)
    previous_f = previous.float().clamp_min(0.0)
    kl = current_f * ((current_f + eps).log() - (previous_f + eps).log())
    reduced = kl.sum(dim=-1)
    if current.ndim == 4:
        reduced = reduced.mean(dim=1)
    return reduced.nan_to_num(nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)


def should_trigger_refresh(
    decoded_this_step: int,
    decoded_since_refresh: int,
    threshold: int,
) -> bool:
    """Return true when this step or cumulative decoded tokens exceed tau."""
    if threshold < 0:
        raise ValueError("threshold must be non-negative")
    return decoded_since_refresh + decoded_this_step > threshold


def select_high_drift_positions(
    scores: torch.Tensor,
    eligible_mask: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select the deterministic top ceil(fraction * n) eligible positions."""
    if scores.shape != eligible_mask.shape:
        raise ValueError("scores and eligible_mask must have matching shapes")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be between 0 and 1")

    selected = torch.zeros_like(eligible_mask, dtype=torch.bool)
    eligible = eligible_mask.bool()
    eligible_indices = torch.nonzero(eligible.reshape(-1), as_tuple=False).flatten()
    eligible_count = int(eligible_indices.numel())
    if eligible_count == 0:
        return selected

    take = int(math.ceil(float(fraction) * eligible_count))
    if take == 0:
        return selected

    flat_scores = scores.reshape(-1).float()
    eligible_scores = flat_scores[eligible_indices]
    order = torch.argsort(-eligible_scores, stable=True)
    chosen = eligible_indices[order[:take]]
    selected.reshape(-1)[chosen] = True
    return selected


def local_prefix_bounds(
    prompt_len: int,
    block_start: int,
    block_length: int,
    prefix_blocks: int,
) -> tuple[int, int]:
    """Return half-open local prefix bounds clipped to the prompt boundary."""
    if prompt_len < 0:
        raise ValueError("prompt_len must be non-negative")
    if block_start < 0:
        raise ValueError("block_start must be non-negative")
    if block_length <= 0:
        raise ValueError("block_length must be positive")
    if prefix_blocks < 0:
        raise ValueError("prefix_blocks must be non-negative")

    start = max(prompt_len, block_start - prefix_blocks * block_length)
    return start, block_start
