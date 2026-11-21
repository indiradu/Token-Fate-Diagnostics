from __future__ import annotations

import torch


def local_window_average(values: torch.Tensor, window: int) -> torch.Tensor:
    """Mean value in a centered local window for each sequence position."""
    if values.ndim != 2:
        raise ValueError(f"expected [batch, length], got {tuple(values.shape)}")
    if window <= 0:
        return values.float()

    batch, length = values.shape
    values = values.float()
    out = torch.empty((batch, length), dtype=torch.float32, device=values.device)
    for pos in range(length):
        start = max(0, pos - window)
        end = min(length, pos + window + 1)
        out[:, pos] = values[:, start:end].mean(dim=1)
    return out


def safe_kl(
    log_probs: torch.Tensor,
    prev_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    """KL(p_t || p_{t-1}) for every position, zero on the first step."""
    if prev_log_probs is None:
        return torch.zeros(log_probs.shape[:-1], dtype=torch.float32, device=log_probs.device)
    probs = log_probs.exp()
    return (probs * (log_probs - prev_log_probs.float())).sum(dim=-1).clamp_min(0.0)


def safe_jsd(
    log_probs: torch.Tensor,
    prev_log_probs: torch.Tensor | None,
) -> torch.Tensor:
    """Jensen-Shannon divergence between consecutive distributions."""
    if prev_log_probs is None:
        return torch.zeros(log_probs.shape[:-1], dtype=torch.float32, device=log_probs.device)
    probs = log_probs.exp()
    prev_probs = prev_log_probs.float().exp()
    mixture = 0.5 * (probs + prev_probs)
    mixture_log = mixture.clamp_min(1e-30).log()
    current_kl = (probs * (log_probs - mixture_log)).sum(dim=-1)
    previous_kl = (prev_probs * (prev_log_probs.float() - mixture_log)).sum(dim=-1)
    return (0.5 * (current_kl + previous_kl)).clamp_min(0.0)


def entropy_from_log_probs(log_probs: torch.Tensor) -> torch.Tensor:
    probs = log_probs.exp()
    return (-(probs * log_probs).sum(dim=-1)).clamp_min(0.0)
