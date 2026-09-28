"""Paper-spec controllers used as external PhaseLock baselines.

These implementations are intentionally separate from :mod:`phase_lock` so a
baseline cannot silently inherit a PhaseLock safety or profitability gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SureLockDecision:
    """One SureLock decision after the sampler has updated its mask state."""

    selected: torch.Tensor
    candidate: torch.Tensor
    uncertainty: torch.Tensor
    posterior_kl: torch.Tensor
    uncertainty_threshold: torch.Tensor


def parse_example_indices(text: str) -> list[int]:
    """Parse comma-separated indices and inclusive ``start:end`` spans."""
    indices: list[int] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        if ":" not in item:
            indices.append(int(item))
            continue
        start_text, end_text = item.split(":", 1)
        start, end = int(start_text), int(end_text)
        if end < start:
            raise ValueError("example index span must be increasing")
        indices.extend(range(start, end + 1))
    if not indices:
        raise ValueError("no example indices were provided")
    if len(indices) != len(set(indices)):
        raise ValueError("example indices must be unique")
    return indices


def posterior_kl(
    current_log_probs: torch.Tensor,
    previous_log_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute ``KL(p_t || p_{t-1})`` at every batch/sequence position."""
    if current_log_probs.shape != previous_log_probs.shape:
        raise ValueError("current and previous posterior tensors must match")
    if current_log_probs.ndim != 3:
        raise ValueError("posterior tensors must have shape [batch, sequence, vocab]")
    current = current_log_probs.float()
    previous = previous_log_probs.float()
    return (
        current.exp() * (current - previous)
    ).sum(dim=-1).clamp_min(0.0)


def surelock_decision(
    *,
    current_log_probs: torch.Tensor,
    previous_log_probs: torch.Tensor | None,
    candidate: torch.Tensor,
    epsilon: float,
    confidence_percentile: float = 20.0,
) -> SureLockDecision:
    """Apply Algorithm 1 from Oba et al. (SureLock).

    ``candidate`` must be the positions that are active and unmasked *after*
    the sampler's ordinary mask update.  The confidence gate keeps positions
    at or below the empirical ``m``-th percentile of uncertainty, where
    uncertainty is ``1 - max_v p_t(v)``.  The first step cannot lock because
    the local posterior KL is undefined there.
    """
    if current_log_probs.ndim != 3:
        raise ValueError("current_log_probs must have shape [batch, sequence, vocab]")
    if candidate.shape != current_log_probs.shape[:-1] or candidate.dtype != torch.bool:
        raise ValueError("candidate must be a boolean [batch, sequence] tensor")
    if epsilon < 0.0:
        raise ValueError("epsilon must be non-negative")
    if not 0.0 < confidence_percentile <= 100.0:
        raise ValueError("confidence_percentile must be in (0, 100]")

    uncertainty = 1.0 - current_log_probs.float().amax(dim=-1).exp()
    thresholds = torch.full(
        (candidate.shape[0],),
        torch.nan,
        dtype=torch.float32,
        device=current_log_probs.device,
    )
    if previous_log_probs is None:
        divergence = torch.full_like(uncertainty, torch.inf)
        return SureLockDecision(
            selected=torch.zeros_like(candidate),
            candidate=candidate.clone(),
            uncertainty=uncertainty,
            posterior_kl=divergence,
            uncertainty_threshold=thresholds,
        )

    divergence = posterior_kl(current_log_probs, previous_log_probs)
    selected = torch.zeros_like(candidate)
    quantile = confidence_percentile / 100.0
    for batch_index in range(candidate.shape[0]):
        values = uncertainty[batch_index][candidate[batch_index]]
        if values.numel() == 0:
            continue
        threshold = torch.quantile(values, quantile)
        thresholds[batch_index] = threshold
        selected[batch_index] = (
            candidate[batch_index]
            & (uncertainty[batch_index] <= threshold)
            & (divergence[batch_index] <= epsilon)
        )
    return SureLockDecision(
        selected=selected,
        candidate=candidate.clone(),
        uncertainty=uncertainty,
        posterior_kl=divergence,
        uncertainty_threshold=thresholds,
    )
