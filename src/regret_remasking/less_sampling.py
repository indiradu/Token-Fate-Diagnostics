from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LessConfig:
    confidence_threshold: float = 0.75
    jsd_threshold: float = 0.04
    top_k: int = 8
    history_length: int = 2


@dataclass(frozen=True)
class LessDecision:
    transfer_index: torch.Tensor
    accepted_index: torch.Tensor
    persistence: torch.Tensor
    topk_jsd: torch.Tensor
    used_fallback: bool


def coarse_topk_js(
    current_ids: torch.Tensor,
    current_probs: torch.Tensor,
    previous_ids: torch.Tensor,
    previous_probs: torch.Tensor,
    eps: float = 1e-12,
) -> torch.Tensor:
    """LESS top-K JSD with all untracked vocabulary mass in one residual bin."""
    if current_ids.shape != previous_ids.shape or current_probs.shape != previous_probs.shape:
        raise ValueError("current and previous top-K summaries must have matching shapes")
    if current_ids.shape != current_probs.shape or current_ids.ndim != 2:
        raise ValueError("top-K ids and probabilities must be rank-2 tensors with matching shapes")

    rows, top_k = current_ids.shape
    combined_ids = torch.cat([current_ids.clamp_min(0), previous_ids.clamp_min(0)], dim=-1)
    zeros = torch.zeros_like(current_probs)
    current_mass = torch.cat([current_probs, zeros], dim=-1).float()
    previous_mass = torch.cat([zeros, previous_probs], dim=-1).float()
    sorted_ids, order = torch.sort(combined_ids, dim=-1)
    current_sorted = torch.gather(current_mass, -1, order)
    previous_sorted = torch.gather(previous_mass, -1, order)

    changed_id = sorted_ids[:, 1:] != sorted_ids[:, :-1]
    group_index = torch.zeros((rows, 2 * top_k), device=current_ids.device, dtype=torch.long)
    group_index[:, 1:] = changed_id.long().cumsum(dim=-1)
    current_groups = torch.zeros_like(current_mass, dtype=torch.float32)
    previous_groups = torch.zeros_like(previous_mass, dtype=torch.float32)
    current_groups.scatter_add_(1, group_index, current_sorted)
    previous_groups.scatter_add_(1, group_index, previous_sorted)

    current_other = (1.0 - current_probs.float().sum(dim=-1)).clamp_min(0.0)
    previous_other = (1.0 - previous_probs.float().sum(dim=-1)).clamp_min(0.0)
    mixture_groups = 0.5 * (current_groups + previous_groups)
    mixture_other = 0.5 * (current_other + previous_other)
    current_kl = (
        current_groups
        * ((current_groups + eps).log() - (mixture_groups + eps).log())
    ).sum(dim=-1)
    previous_kl = (
        previous_groups
        * ((previous_groups + eps).log() - (mixture_groups + eps).log())
    ).sum(dim=-1)
    current_kl += current_other * ((current_other + eps).log() - (mixture_other + eps).log())
    previous_kl += previous_other * (
        (previous_other + eps).log() - (mixture_other + eps).log()
    )
    return (0.5 * (current_kl + previous_kl)).clamp_min(0.0)


class LessSelectionState:
    """Stateful released-code LESS admission for one LLaDA sequence."""

    def __init__(self, sequence_length: int, device: torch.device, config: LessConfig):
        if config.history_length < 1:
            raise ValueError("LESS history_length must be at least 1")
        if config.top_k < 1:
            raise ValueError("LESS top_k must be at least 1")
        self.config = config
        self.previous_winners = torch.full(
            (1, sequence_length, config.history_length),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.previous_topk_ids = torch.full(
            (1, sequence_length, config.top_k),
            -1,
            dtype=torch.long,
            device=device,
        )
        self.previous_topk_probs = torch.zeros(
            (1, sequence_length, config.top_k),
            dtype=torch.float32,
            device=device,
        )

    def reset_range(self, start: int, end: int) -> None:
        self.previous_winners[:, start:end] = -1
        self.previous_topk_ids[:, start:end] = -1
        self.previous_topk_probs[:, start:end] = 0.0

    def select(
        self,
        probs: torch.Tensor,
        mask_index: torch.Tensor,
        default_k: int,
    ) -> LessDecision:
        if probs.ndim != 3 or probs.shape[:2] != mask_index.shape or probs.shape[0] != 1:
            raise ValueError("LESS selection expects probs [1, sequence, vocab] and matching mask")
        if probs.shape[-1] < self.config.top_k:
            raise ValueError("LESS top_k exceeds vocabulary size")

        transfer_index = torch.zeros_like(mask_index, dtype=torch.bool)
        accepted_index = torch.zeros_like(mask_index, dtype=torch.bool)
        persistence = torch.zeros_like(mask_index, dtype=torch.bool)
        topk_jsd = torch.full(mask_index.shape, float("nan"), device=probs.device)
        positions = torch.nonzero(mask_index[0], as_tuple=False).flatten()
        if positions.numel() == 0:
            return LessDecision(
                transfer_index, accepted_index, persistence, topk_jsd, used_fallback=False
            )

        position_probs = probs[0, positions]
        current_topk_probs, current_topk_ids = torch.topk(
            position_probs, k=self.config.top_k, dim=-1
        )
        current_topk_probs = current_topk_probs.float()
        winners = current_topk_ids[:, 0]
        confidence = current_topk_probs[:, 0]

        winner_history = self.previous_winners[0, positions]
        valid_history = (winner_history >= 0).all(dim=-1)
        persistent = valid_history & winner_history.eq(winners.unsqueeze(-1)).all(dim=-1)
        previous_ids = self.previous_topk_ids[0, positions]
        previous_probs = self.previous_topk_probs[0, positions]
        has_previous = previous_probs.sum(dim=-1) > 0.0
        drift = coarse_topk_js(
            current_topk_ids,
            current_topk_probs,
            previous_ids,
            previous_probs,
        )
        drift = torch.where(has_previous, drift, torch.ones_like(drift))
        accepted = (
            persistent
            & confidence.ge(self.config.confidence_threshold)
            & drift.le(self.config.jsd_threshold)
        )

        if self.config.history_length > 1:
            self.previous_winners[0, positions] = torch.cat(
                [winner_history[:, 1:], winners.unsqueeze(-1)], dim=-1
            )
        else:
            self.previous_winners[0, positions, 0] = winners
        self.previous_topk_ids[0, positions] = current_topk_ids
        self.previous_topk_probs[0, positions] = current_topk_probs

        persistence[0, positions] = persistent
        topk_jsd[0, positions] = drift
        accepted_positions = positions[accepted]
        used_fallback = accepted_positions.numel() == 0
        if not used_fallback:
            accepted_index[0, accepted_positions] = True
            transfer_index[0, accepted_positions] = True
        else:
            take = max(1, min(int(default_k), int(positions.numel())))
            _, fallback_indices = torch.topk(confidence, k=take)
            transfer_index[0, positions[fallback_indices]] = True

        return LessDecision(
            transfer_index,
            accepted_index,
            persistence,
            topk_jsd,
            used_fallback=used_fallback,
        )
